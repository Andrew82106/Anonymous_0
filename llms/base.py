from abc import ABC, abstractmethod
from typing import Any, List, Dict, Optional

class BaseLLM(ABC):
    """
    Abstract base class for all LLM providers.
    """
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.name = config.get('name', 'unknown')
        self.provider = config.get('provider', 'unknown')
        self.api_provider = config.get('API_provider', 'unknown')
        self.model_name = config.get('model_name', '')

    @abstractmethod
    def generate(self, prompt: str, **kwargs) -> str:
        """
        Generate text based on the prompt.
        """
        pass

    @abstractmethod
    def generate_image(self, prompt: str, **kwargs) -> str:
        """
        Generate image based on the prompt. Returns the image URL or path.
        Optional to implement if the model is text-only.
        """
        pass

    def generate_structured(self, prompt: str, response_format: Any, **kwargs) -> Any:
        """
        Generate structured output (optional).
        """
        raise NotImplementedError("Structured output not implemented for this provider.")