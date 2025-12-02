import json
from typing import Any, Dict, List, Type, Optional
from pydantic import BaseModel
from ..base import BaseLLM

try:
    from zhipuai import ZhipuAI
except ImportError:
    ZhipuAI = None

class ZhipuAILLM(BaseLLM):
    """
    ZhipuAI Provider implementation for Text generation.
    """
    def __init__(self, config: Dict[str, Any]):
        super().__init__(config)
        if ZhipuAI is None:
            raise ImportError("zhipuai package is not installed. Please run 'pip install zhipuai'")
        
        self.api_key = config.get('api_key')
        
        # Initialize ZhipuAI client
        self.client = ZhipuAI(api_key=self.api_key)
        
        self.temperature = config.get('temperature', 0.7)
        # ZhipuAI doesn't use 'max_tokens' in the same way or sometimes defaults are fine,
        # but we can pass it if needed. 'max_tokens' support varies by model version.
        # We'll keep it optional.
        self.max_tokens = config.get('max_tokens') 

    def generate(self, prompt: str, **kwargs) -> str:
        """
        Generate text completion using ZhipuAI.
        """
        try:
            # Prepare messages
            messages = [{"role": "user", "content": prompt}]
            
            # Build arguments
            create_args = {
                "model": self.model_name,
                "messages": messages,
                "temperature": kwargs.get('temperature', self.temperature),
            }
            
            if kwargs.get('max_tokens', self.max_tokens):
                create_args['max_tokens'] = kwargs.get('max_tokens', self.max_tokens)

            response = self.client.chat.completions.create(**create_args)
            return response.choices[0].message.content
        except Exception as e:
            return f"Error generating text with ZhipuAI: {str(e)}"

    def generate_structured(self, prompt: str, response_format: Type[BaseModel], **kwargs) -> Optional[BaseModel]:
        """
        Generate structured output using Pydantic models for ZhipuAI.
        Simulates structured output by injecting schema into system prompt and enforcing JSON mode.
        """
        try:
            # 1. Get JSON Schema from Pydantic model
            schema = response_format.model_json_schema()
            schema_str = json.dumps(schema, indent=2, ensure_ascii=False)
            
            # 2. Construct System Prompt with Schema
            system_instruction = f"""
            You are a helpful assistant. Please respond using valid JSON that strictly conforms to the following JSON Schema:
            
            {schema_str}
            
            Ensure the output is a valid JSON object matching this schema exactly.
            """
            
            messages = [
                {"role": "system", "content": system_instruction},
                {"role": "user", "content": prompt}
            ]
            
            # 3. Call API with json_object response format
            # Note: Removing response_format={"type": "json_object"} as some older SDK versions don't support it.
            # The system prompt is usually sufficient for GLM-4 to output JSON.
            create_args = {
                "model": self.model_name,
                "messages": messages,
                "temperature": kwargs.get('temperature', self.temperature),
                # "response_format": {"type": "json_object"} # Commented out for compatibility
            }
            
            if kwargs.get('max_tokens', self.max_tokens):
                create_args['max_tokens'] = kwargs.get('max_tokens', self.max_tokens)

            response = self.client.chat.completions.create(**create_args)
            content = response.choices[0].message.content
            
            # 4. Parse and Validate
            # Zhipu might wrap in markdown code blocks
            clean_content = content.strip()
            if "```json" in clean_content:
                clean_content = clean_content.split("```json")[1].split("```")[0].strip()
            elif "```" in clean_content:
                clean_content = clean_content.split("```")[1].split("```")[0].strip()
                
            json_data = json.loads(clean_content)
            return response_format.model_validate(json_data)
            
        except Exception as e:
            print(f"Error generating structured output with ZhipuAI: {e}")
            return None

    def generate_image(self, prompt: str, **kwargs) -> str:
        """
        ZhipuAI image generation (CogView) support can be added here if needed.
        For now, it raises NotImplementedError or returns error message.
        """
        return "Image generation not implemented for ZhipuAI provider yet."
