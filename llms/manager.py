import time
from typing import Dict, List, Any, Optional
from .base import BaseLLM
from .providers.openai_provider import OpenAILLM
from .providers.zhipuai_provider import ZhipuAILLM
from .providers.modelscope_provider import ModelScopeLLM
from .providers.lanyun_provider import LanyunLLM
from utils_set.utils import config_loader

class LLMManager:
    """
    Manager class to handle multiple LLM providers and models.
    """
    def __init__(self):
        self.config = config_loader
        self.models: Dict[str, BaseLLM] = {}
        self._initialize_models()

    def _initialize_models(self):
        """
        Initialize all models defined in config.yaml
        """
        # Initialize Text Models
        text_models_config = self.config.get('models.text_models', [])
        for model_conf in text_models_config:
            self._register_model(model_conf)

        # Initialize Image Models
        image_models_config = self.config.get('models.image_models', [])
        for model_conf in image_models_config:
            self._register_model(model_conf)

    def _register_model(self, model_conf: Dict[str, Any]):
        """
        Instantiate and register a model based on its API_provider.
        """
        name = model_conf.get('name')
        api_provider = model_conf.get('API_provider', '').lower()

        if not name:
            print(f"Warning: Found model config without name, skipping: {model_conf}")
            return

        try:
            model_instance = None
            
            # Factory logic based on API_provider
            if api_provider == 'openai':
                model_instance = OpenAILLM(model_conf)
            elif api_provider == 'zhipuai':
                model_instance = ZhipuAILLM(model_conf)
            elif api_provider == 'modelscope':
                model_instance = ModelScopeLLM(model_conf)
            elif api_provider == 'lanyun':
                model_instance = LanyunLLM(model_conf)
            # Add other providers here e.g., elif api_provider == 'anthropic': ...
            else:
                print(f"Warning: Unsupported API_provider '{api_provider}' for model '{name}'")
                return

            if model_instance:
                self.models[name] = model_instance
                
        except Exception as e:
            print(f"Error initializing model '{name}': {e}")

    def get_model(self, model_name: str) -> Optional[BaseLLM]:
        """
        Retrieve a loaded model instance by name.
        """
        return self.models.get(model_name)

    def list_models(self) -> List[str]:
        """
        Return a list of available model names.
        """
        return list(self.models.keys())

    def call_model(self, model_name: str, prompt: str, mode: str = 'text', **kwargs) -> str:
        """
        Unified interface to call a model.
        mode: 'text' or 'image'
        """
        model = self.get_model(model_name)
        if not model:
            raise ValueError(f"Model '{model_name}' not found. Available models: {self.list_models()}")

        # 调用 API
        if mode == 'text':
            result = model.generate(prompt, **kwargs)
        elif mode == 'image':
            result = model.generate_image(prompt, **kwargs)
        else:
            raise ValueError("Mode must be 'text' or 'image'")
        
        # 添加延迟避免 API 速率限制
        api_delay = self.config.get('api_delay', 0)
        if api_delay > 0:
            time.sleep(api_delay)
        
        return result

    def call_structured(self, model_name: str, prompt: str, response_format: Any, **kwargs) -> Any:
        """
        Call model for structured output.
        """
        model = self.get_model(model_name)
        if not model:
            raise ValueError(f"Model '{model_name}' not found. Available models: {self.list_models()}")
        
        # 调用 API
        result = model.generate_structured(prompt, response_format, **kwargs)
        
        # 添加延迟避免 API 速率限制
        api_delay = self.config.get('api_delay', 0)
        if api_delay > 0:
            time.sleep(api_delay)
        
        return result

# Global instance
llm_manager = LLMManager()