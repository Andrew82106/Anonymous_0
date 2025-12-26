"""
Prompt 模板库 (Prompt Templates)
为因果推理任务设计的提示词
"""

# ========== NEW: Statistical Judge Prompt for CAD Framework ==========
STATISTICAL_JUDGE_PROMPT = """You are a Statistical Judge specializing in causal inference. The PC algorithm has confirmed a link between Variable A and Variable B, but the direction remains ambiguous (Markov Equivalence Class edge).

## Your Task
Based on the statistical evidence below, determine the most likely causal direction.

## Statistical Evidence

### Context
{context}

### Functional Trace (Residual Independence - ANM Principle)
- Direction A→B: Residual Independence P-value = {p_xy:.4f}
- Direction B→A: Residual Independence P-value = {p_yx:.4f}
- Signal: {func_signal} (Strength: {func_strength:.3f})

**Interpretation**: Higher p-value indicates residuals are more independent of the predictor, suggesting a correct causal model. In Additive Noise Models (ANM), the true causal direction typically yields higher residual independence.

### Informational Trace (Entropy - IGCI Principle)
- Entropy of A: H(A) = {h_x:.4f}
- Entropy of B: H(B) = {h_y:.4f}
- Signal: {info_signal} (Strength: {info_strength:.3f})

**Interpretation**: In Information-Geometric Causal Inference (IGCI), causes typically have lower entropy than effects. Causal processes tend to increase entropy due to noise accumulation.

### Signal Consensus
- Functional Signal: {func_signal}
- Informational Signal: {info_signal}
- Overall Consensus: {consensus}

## Guidance Principles

1. **If signals ALIGN**: Be confident in the direction. Both functional and informational evidence point the same way.

2. **If signals CONFLICT**: 
   - **Prioritize the Functional/Residual signal** as it is often more robust for discrete data.
   - Lower your confidence to "medium" or "low".
   - Note the conflict in your reasoning.

3. **Confidence Calibration**:
   - `high`: Both signals align AND strength > 0.3
   - `medium`: Signals align with moderate strength (0.1-0.3) OR one signal is ambiguous
   - `low`: Signals conflict OR both are weak/ambiguous

## Response Format

Return a JSON object with:
```json
{{
    "direction": "A->B" | "B->A" | "Unclear",
    "confidence": "high" | "medium" | "low",
    "primary_evidence": "Brief description of the key deciding factor",
    "reasoning_chain": "Step-by-step reasoning explaining your judgment"
}}
```
"""

SHERLOCK_HOLMES_PROMPT = """你是一位精通统计学和因果推理的侦探，专门从**脱敏的统计证据**中推断变量之间的因果关系。

## 🔍 你的任务
我将为你提供两个**匿名变量 (Variable A 和 Variable B)** 的统计分析结果。你需要像福尔摩斯一样，**仅基于数据的统计行为**，推断它们之间的因果关系。

**重要约束**：
- 你**不知道**这些变量的真实含义（它们可能是温度、销量、身高、体重...任何东西）
- 你**只能**依靠统计特征来推理（分布形态、残差独立性、拟合度等）
- 你需要综合多个维度的证据，权衡矛盾信号

## 📊 统计分析报告

{narrative}

## 🎯 推理要求

请根据上述统计证据，推断最可能的因果结构：

1. **因果方向判断**：
   - `A->B`: A 是 B 的原因
   - `B->A`: B 是 A 的原因
   - `A<-Z->B`: A 和 B 由共同的隐藏变量 Z 引起（混杂因素）
   - `A_|_B`: A 和 B 统计独立，没有因果关系
   - `Unclear`: 证据不足或矛盾，无法判断

2. **置信度评估**：
   - `high`: 证据强烈且一致
   - `medium`: 证据中等，或存在轻微矛盾
   - `low`: 证据微弱或高度矛盾

3. **推理链条**：
   - 列出关键观察点（如：残差独立性差异、分布非高斯性等）
   - 指出任何矛盾或不确定性
   - 说明你的最终判断依据

## 🧠 背景知识与推理准则

### 核心原则：权衡多维证据，不要轻易放弃

**⚠️ 关键提醒**：统计分析报告现在采用**客观描述**风格，不会直接告诉你答案。你需要：
1. **重视相对差异 (Relative Difference)**：即使绝对值接近，10-20% 的相对差异也可能是有效信号
2. **综合权衡**：不要只看单一指标。结合拟合度、独立性、复杂度、熵等多个维度
3. **不要过早下"Unclear"判断**：除非差异真的微乎其微（<5%）或证据完全矛盾，否则尝试做出判断

### 统计证据优先级指南

**Level 1 - 黄金标准 (Gold Standard)**：
- **残差独立性 (HSIC/MI)**：这是因果方向最直接的证据
  - 相对差异 >50%：强烈信号，几乎可以确定方向
  - 相对差异 10-50%：中等信号，需结合其他证据
  - 相对差异 <10%：微弱信号，需谨慎

**Level 2 - 强辅助证据**：
- **条件熵 (Conditional Entropy)**（离散变量）：熵降低意味着预测能力
  - 相对差异 >15%：显著信号
  - 相对差异 5-15%：中等信号
- **模型拟合度 (R²)**（连续变量）：更高的 R² 意味着更好地捕捉了真实关系

**Level 3 - 辅助线索**：
- **异方差性 (Heteroscedasticity)**：错误方向通常表现出高异方差
- **非高斯性 (Skewness/Kurtosis)**：LiNGAM 的可识别性来源
- **模型复杂度**：如果非线性模型在独立性上有显著提升，复杂度是合理的

### 决策逻辑框架

**ANM (Additive Noise Model) 场景**：
- 优先选择：**残差独立性更好** 且 **拟合度更高** 的方向
- 即使模型是非线性的，只要独立性好，就是正确的因果机制
- **不要因为模型复杂就否定它**，关键看残差质量

**LiNGAM (Linear Non-Gaussian) 场景**：
- 看偏度差异：因变量通常继承因的非高斯性，偏度更极端
- 残差独立性仍然是核心判据

**离散变量场景（特别注意！）**：
- **离散数据悖论 (Discrete Data Paradox)**：在离散数据中，**不要**单纯因为“B 能更好地预测 A”（即 B->A 条件熵更低）就认为 B 是因。
  - **反例**：如果“地湿”能完美推断“下雨”，并不代表“地湿”导致了“下雨”。反向预测往往更确定（熵更低）。
- **黄金法则 (IGCI Principle)**：请优先关注 **边缘熵 (Marginal Entropy)**。
  - 通常 **熵更低（更简单）** 的变量是原因，熵更高（更混乱）的变量是结果（因为因果过程通常会引入噪音，导致熵增加）。
  - 如果 Condition Entropy 支持 B->A，但 Marginal Entropy 强烈支持 A->B（A的熵远低于B），请选择 **A->B**。
- **条件熵与独立性**：只有在边缘熵相似时，才更多参考条件熵和 p-value。

**混杂/独立判断**：
- 两个方向残差都很好 + 极强相关 → 可能是混杂
- 两个方向残差都很差 + 弱相关 → 可能是独立或复杂关系

### 推理示例

**案例 1：微小但一致的信号**
- 残差独立性：A->B (0.25) vs B->A (0.30)，相对差异 ~17%
- 条件熵：A->B (0.45) vs B->A (0.52)，相对差异 ~13%
- **推断**：虽然差异不大，但两个指标都指向 A->B，可以判断为 `A->B` (confidence: medium)

**案例 2：矛盾信号的权衡**
- 残差独立性：A->B 更好（相对差异 50%）
- 拟合度：B->A 更好（相对差异 20%）
- **推断**：残差独立性是更核心的证据，选择 `A->B` (confidence: medium，备注：拟合度略逆向但独立性决定性更强)

请以结构化的 JSON 格式返回你的推理结果。
"""

SIMPLE_PROMPT = """基于以下统计分析，判断变量 A 和 B 之间的因果关系：

{narrative}

请返回 JSON 格式的推理结果，包括：direction（因果方向）、confidence（置信度）、reasoning_chain（推理过程）。
"""

ABLATION_PROMPT_RESIDUAL_ONLY = """你是一位统计学家。以下是两个匿名变量的残差分析结果：

{narrative}

**仅基于残差独立性**，判断因果方向。返回 JSON 格式结果。
"""

def get_prompt(template_name: str = "sherlock", narrative: str = "", **kwargs) -> str:
    """
    获取指定的 Prompt 模板
    
    Parameters:
    -----------
    template_name : str
        模板名称：'sherlock', 'simple', 'residual_only', 'statistical_judge'
    narrative : str
        来自 StatTranslator 的统计叙事
    **kwargs : dict
        额外参数（用于 statistical_judge 模板）
    
    Returns:
    --------
    str : 完整的 prompt
    """
    templates = {
        'sherlock': SHERLOCK_HOLMES_PROMPT,
        'simple': SIMPLE_PROMPT,
        'residual_only': ABLATION_PROMPT_RESIDUAL_ONLY,
        'statistical_judge': STATISTICAL_JUDGE_PROMPT
    }
    
    template = templates.get(template_name, SHERLOCK_HOLMES_PROMPT)
    
    # statistical_judge 模板需要特殊处理
    if template_name == 'statistical_judge':
        return template.format(
            context=kwargs.get('context', 'PC algorithm confirmed edge, direction ambiguous.'),
            p_xy=kwargs.get('p_xy', 0.5),
            p_yx=kwargs.get('p_yx', 0.5),
            func_signal=kwargs.get('func_signal', 'ambiguous'),
            func_strength=kwargs.get('func_strength', 0.0),
            h_x=kwargs.get('h_x', 0.0),
            h_y=kwargs.get('h_y', 0.0),
            info_signal=kwargs.get('info_signal', 'ambiguous'),
            info_strength=kwargs.get('info_strength', 0.0),
            consensus=kwargs.get('consensus', 'weak')
        )
    
    return template.format(narrative=narrative)
