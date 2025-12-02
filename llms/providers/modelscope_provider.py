import json
import time
import requests
from typing import Dict, Any, Optional, Type
from pydantic import BaseModel
from ..base import BaseLLM

try:
    from openai import OpenAI
except ImportError:
    OpenAI = None

class ModelScopeLLM(BaseLLM):
    """
    ModelScope Provider implementation for Text (via OpenAI interface) and Image generation (via Async API).
    """
    def __init__(self, config: Dict[str, Any]):
        super().__init__(config)
        self.api_key = config.get('api_key')
        self.base_url = config.get('base_url')
        
        # Normalize base_url for OpenAI client (needs /v1/)
        self.openai_base_url = self.base_url
        if self.openai_base_url and not self.openai_base_url.endswith('v1/'):
            if self.openai_base_url.endswith('/'):
                self.openai_base_url += 'v1/'
            else:
                self.openai_base_url += '/v1/'

        # Initialize OpenAI client for text generation
        if OpenAI:
            self.client = OpenAI(
                api_key=self.api_key,
                base_url=self.openai_base_url
            )
        else:
            self.client = None
            
        self.temperature = config.get('temperature', 0.7)
        self.max_tokens = config.get('max_tokens', 2048)

    def generate(self, prompt: str, **kwargs) -> str:
        """
        Generate text completion using OpenAI compatible interface.
        """
        if not self.client:
            return "Error: openai package not installed. Please run 'pip install openai'"
        
        try:
            response = self.client.chat.completions.create(
                model=self.model_name,
                messages=[{"role": "user", "content": prompt}],
                temperature=kwargs.get('temperature', self.temperature),
                max_tokens=kwargs.get('max_tokens', self.max_tokens),
                stream=False 
            )
            return response.choices[0].message.content
        except Exception as e:
            return f"Error generating text: {str(e)}"

    def generate_structured(self, prompt: str, response_format: Type[BaseModel], **kwargs) -> Optional[BaseModel]:
        """
        Generate structured output using Pydantic models (OpenAI Structured Outputs).
        Includes fallback for models that wrap JSON in Markdown code blocks.
        """
        if not self.client:
            print("Error: openai package not installed.")
            return None

        try:
            # First try the official structured output way
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
            # Check if it's a validation error due to Markdown wrapping or malformed JSON
            error_msg = str(e)
            if "validation error" in error_msg or "Invalid JSON" in error_msg:
                # Fallback: Manual Generation & Cleaning
                try:
                    # 1. Generate raw text
                    content = self.generate(prompt, **kwargs)
                    
                    # 2. Clean Markdown
                    content = content.strip()
                    if "```json" in content:
                        content = content.split("```json")[1].split("```")[0].strip()
                    elif "```" in content:
                        content = content.split("```")[1].split("```")[0].strip()
                    
                    # 3. Parse with Pydantic
                    return response_format.model_validate_json(content)
                except Exception as fallback_e:
                    print(f"Structured fallback failed: {fallback_e}")
                    raise e # Raise original error if fallback also fails
            
            # Re-raise other errors (e.g. Auth error, 400 Bad Request)
            raise e

    def generate_image(self, prompt: str, **kwargs) -> str:
        """
        Generate image using ModelScope Async API.
        """
        try:
            common_headers = {
                "Authorization": f"Bearer {self.api_key}",
                "Content-Type": "application/json",
            }
            
            # Use the original base_url for manual requests, ensuring it doesn't double up /v1/ if not present in config but needed for specific endpoints
            # User config: https://api-inference.modelscope.cn/
            # Endpoint: v1/images/generations
            
            request_url = self.base_url
            if not request_url.endswith('/'):
                request_url += '/'
            
            # Initial request
            response = requests.post(
                f"{request_url}v1/images/generations",
                headers={**common_headers, "X-ModelScope-Async-Mode": "true"},
                data=json.dumps({
                    "model": self.model_name, # ModelScope Model-Id
                    "prompt": prompt
                }, ensure_ascii=False).encode('utf-8')
            )
            
            response.raise_for_status()
            task_id = response.json().get("task_id")
            
            if not task_id:
                return "Error: No task_id received from ModelScope"

            # Poll for result
            max_retries = 60 # 5 minutes max (5s * 60)
            for _ in range(max_retries):
                result = requests.get(
                    f"{request_url}v1/tasks/{task_id}",
                    headers={**common_headers, "X-ModelScope-Task-Type": "image_generation"},
                )
                result.raise_for_status()
                data = result.json()

                if data["task_status"] == "SUCCEED":
                    # Return the URL of the first generated image
                    if "output_images" in data and len(data["output_images"]) > 0:
                        return data["output_images"][0]
                    else:
                        return "Error: Task succeeded but no output images found."
                elif data["task_status"] == "FAILED":
                    return f"Error: Image generation failed. Status: {data.get('task_status')}"
                
                time.sleep(5)
            
            return "Error: Image generation timed out."

        except Exception as e:
            return f"Error generating image: {str(e)}"
