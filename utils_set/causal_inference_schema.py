"""
因果推理响应结构定义 (Causal Inference Response Schema)
使用 Pydantic 定义 LLM 输出的结构化格式
"""

from pydantic import BaseModel, Field
from typing import Literal, Optional

class CausalInferenceResponse(BaseModel):
    """
    LLM 对因果方向的推理结果
    """
    direction: Literal["A->B", "B->A", "A<-Z->B", "A_|_B", "Unclear"] = Field(
        description="推断的因果方向。可选值：'A->B' (A导致B), 'B->A' (B导致A), 'A<-Z->B' (共同原因), 'A_|_B' (独立), 'Unclear' (无法判断)"
    )
    
    confidence: Literal["high", "medium", "low"] = Field(
        description="置信度水平。high=强信号, medium=中等信号, low=弱信号或矛盾"
    )
    
    primary_evidence: str = Field(
        description="支持该判断的主要证据，简洁描述（如：'A->B的残差独立性显著高于B->A'）"
    )
    
    reasoning_chain: str = Field(
        description="推理过程的详细说明，包括：1) 关键观察 2) 矛盾点（如果有）3) 最终判断依据"
    )
    
    statistical_signals: dict = Field(
        default_factory=dict,
        description="识别到的统计信号摘要，例如 {'lingam_signal': 'strong', 'anm_signal': 'weak'}"
    )
    
    alternative_hypothesis: Optional[str] = Field(
        default=None,
        description="如果存在较强的替代假设，说明是什么（如：'也可能存在混杂因素'）"
    )

class CausalInferenceRequest(BaseModel):
    """
    请求格式（可选，用于结构化输入）
    """
    narrative: str = Field(
        description="来自 StatTranslator 的统计叙事文本"
    )
    
    dataset_info: Optional[dict] = Field(
        default=None,
        description="可选的数据集元信息（不包含变量名）"
    )
