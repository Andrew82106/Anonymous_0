"""
蓝云 API Provider - 支持 DeepSeek-V3 等模型
使用 OpenAI 兼容接口
"""

from typing import Any, Dict, Type, Optional
from pydantic import BaseModel
from ..base import BaseLLM

try:
    from openai import OpenAI
except ImportError:
    OpenAI = None


class LanyunLLM(BaseLLM):
    """
    蓝云 API Provider - 使用 OpenAI 兼容接口
    支持 DeepSeek-V3, DeepSeek-R1 等模型
    """
    def __init__(self, config: Dict[str, Any]):
        super().__init__(config)
        if OpenAI is None:
            raise ImportError("openai package is not installed. Please run 'pip install openai'")
        
        self.api_key = config.get('api_key')
        self.base_url = config.get('base_url', 'https://maas-api.lanyun.net/v1')
        
        # Initialize OpenAI client with Lanyun endpoint
        self.client = OpenAI(
            api_key=self.api_key,
            base_url=self.base_url
        )
        
        self.temperature = config.get('temperature', 0.7)
        self.max_tokens = config.get('max_tokens', 4096)

    def generate(self, prompt: str, **kwargs) -> str:
        """
        Generate text completion.
        支持流式和非流式两种模式
        """
        try:
            stream = kwargs.get('stream', False)
            
            response = self.client.chat.completions.create(
                model=self.model_name,
                messages=[{"role": "user", "content": prompt}],
                temperature=kwargs.get('temperature', self.temperature),
                max_tokens=kwargs.get('max_tokens', self.max_tokens),
                stream=stream
            )
            
            if stream:
                # 流式处理
                full_content = ""
                for chunk in response:
                    if hasattr(chunk.choices[0].delta, 'content'):
                        content = chunk.choices[0].delta.content
                        if content:
                            full_content += content
                return full_content
            else:
                # 非流式
                return response.choices[0].message.content
                
        except Exception as e:
            return f"Error generating text: {str(e)}"

    def generate_structured(self, prompt: str, response_format: Type[BaseModel], **kwargs) -> Optional[BaseModel]:
        """
        Generate structured output - 蓝云 API 可能不支持结构化输出
        回退到普通生成 + JSON 解析
        """
        try:
            # 尝试使用结构化输出
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
            # 如果不支持结构化输出，回退到普通生成
            print(f"Structured output not supported, falling back to regular generation: {e}")
            try:
                response = self.generate(prompt, **kwargs)
                # 尝试解析 JSON
                import json
                data = json.loads(response)
                return response_format(**data)
            except:
                return None

    def generate_image(self, prompt: str, **kwargs) -> str:
        """
        蓝云 API 可能不支持图像生成
        """
        return "Error: Image generation not supported by Lanyun API"
