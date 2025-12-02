import os
from typing import Any, Dict, Type, Optional
from pydantic import BaseModel
from ..base import BaseLLM

try:
    from openai import OpenAI
except ImportError:
    OpenAI = None

class OpenAILLM(BaseLLM):
    """
    OpenAI Provider implementation for both Text, Image generation and Structured Output.
    """
    def __init__(self, config: Dict[str, Any]):
        super().__init__(config)
        if OpenAI is None:
            raise ImportError("openai package is not installed. Please run 'pip install openai'")
        
        self.api_key = config.get('api_key')
        self.base_url = config.get('base_url')
        
        # Initialize OpenAI client
        self.client = OpenAI(
            api_key=self.api_key,
            base_url=self.base_url
        )
        
        self.temperature = config.get('temperature', 0.7)
        self.max_tokens = config.get('max_tokens', 2048)
        self.size = config.get('size', "1024x1024")

    def generate(self, prompt: str, **kwargs) -> str:
        """
        Generate text completion.
        """
        try:
            response = self.client.chat.completions.create(
                model=self.model_name,
                messages=[{"role": "user", "content": prompt}],
                temperature=kwargs.get('temperature', self.temperature),
                max_tokens=kwargs.get('max_tokens', self.max_tokens)
            )
            return response.choices[0].message.content
        except Exception as e:
            return f"Error generating text: {str(e)}"

    def generate_structured(self, prompt: str, response_format: Type[BaseModel], **kwargs) -> Optional[BaseModel]:
        """
        Generate structured output using Pydantic models (OpenAI Structured Outputs).
        """
        try:
            completion = self.client.beta.chat.completions.parse(
                model=self.model_name,
                messages=[
                    {"role": "user", "content": prompt},
                ],
                response_format=response_format,
                temperature=kwargs.get('temperature', self.temperature),
            )
            return completion.choices[0].message.parsed
        except Exception as e:
            print(f"Error generating structured output: {e}")
            return None

    def generate_image(self, prompt: str, **kwargs) -> str:
        """
        Generate image using DALL-E.
        """
        try:
            response = self.client.images.generate(
                model=self.model_name,
                prompt=prompt,
                size=kwargs.get('size', self.size),
                quality="standard",
                n=1,
            )
            return response.data[0].url
        except Exception as e:
            return f"Error generating image: {str(e)}"
