"""
因果推理引擎 (Causal Reasoning Engine)
结合 StatTranslator 和 LLMManager，实现端到端的因果发现流水线
"""

import json
import sys
from typing import Dict, List, Optional, Any
from pathlib import Path

# 添加项目根目录到 sys.path
project_root = Path(__file__).parent.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

from utils_set.stat_translator import StatTranslator
from utils_set.data_generator import CausalDataGenerator
from llms.manager import llm_manager
from utils_set.causal_inference_schema import CausalInferenceResponse
from utils_set.prompts import get_prompt
from utils_set.utils import config_loader

class CausalReasoningEngine:
    """
    因果推理引擎：将统计特征翻译为叙事，并使用 LLM 进行推理
    """
    
    def __init__(self, model_name: str = None, prompt_template: str = "sherlock"):
        """
        Parameters:
        -----------
        model_name : str
            要使用的 LLM 模型名称（需在 config.yaml 中配置）
        prompt_template : str
            Prompt 模板类型：'sherlock', 'simple', 'residual_only'
        """
        self.translator = StatTranslator()
        
        # 如果未指定模型，从配置文件读取默认模型
        if model_name is None:
            self.model_name = config_loader.get('used_model', 'deepseek-ai/DeepSeek-V3.1')
            print(f"📝 Using default model from config: {self.model_name}")
        else:
            self.model_name = model_name
        
        self.prompt_template = prompt_template
        
        # 验证模型是否可用
        available_models = llm_manager.list_models()
        if self.model_name not in available_models:
            raise ValueError(f"Model '{self.model_name}' not found. Available: {available_models}")
        
        print(f"✅ Initialized CausalReasoningEngine with model: {self.model_name}")
    
    def analyze_pair(self, X, Y, narrative_mode: str = "full") -> Dict[str, Any]:
        """
        分析一对变量并生成统计叙事
        
        Parameters:
        -----------
        X, Y : array-like
            要分析的变量对
        narrative_mode : str
            叙事模式：'full' (完整), 'low_order' (仅低阶统计量), 'raw' (原始数值)
        
        Returns:
        --------
        dict : 包含统计数据和叙事文本
        """
        stats = self.translator.analyze(X, Y)
        narrative = self.translator.generate_narrative(stats, mode=narrative_mode)
        
        return {
            'stats': stats,
            'narrative': narrative
        }
    
    def infer_causality(self, narrative: str, use_structured_output: bool = False) -> Dict[str, Any]:
        """
        使用 LLM 推断因果关系
        
        Parameters:
        -----------
        narrative : str
            统计叙事文本
        use_structured_output : bool
            是否使用结构化输出（默认为 False，因为许多兼容 API 支持不完善）
        
        Returns:
        --------
        dict : LLM 的推理结果
        """
        prompt = get_prompt(self.prompt_template, narrative)
        
        # 强制在 Prompt 中要求 JSON
        prompt += "\n\nIMPORTANT: Please ensure your response is a valid JSON object."
        
        try:
            if use_structured_output:
                # 使用结构化输出 (仅当确信 API 支持良好时使用)
                try:
                    model = llm_manager.get_model(self.model_name)
                    response = model.generate_structured(
                        prompt=prompt,
                        response_format=CausalInferenceResponse
                    )
                    if response:
                        return response.model_dump()
                except Exception as e:
                    print(f"⚠️  Structured output failed ({str(e)}), falling back to text...")
            
            # 默认：使用普通文本输出并手动解析
            text_response = llm_manager.call_model(self.model_name, prompt, mode='text')
            return self._parse_text_response(text_response)
                
        except Exception as e:
            print(f"❌ Error in LLM inference: {e}")
            return {
                'direction': 'Unclear',
                'confidence': 'low',
                'primary_evidence': f'Error: {str(e)}',
                'reasoning_chain': 'LLM call failed',
                'error': str(e)
            }
    
    def _parse_text_response(self, text: str) -> Dict[str, Any]:
        """
        解析 LLM 的文本响应（如果不是结构化输出）
        尝试提取 JSON，或返回原始文本
        """
        try:
            # 尝试查找 JSON 块
            start = text.find('{')
            end = text.rfind('}') + 1
            if start != -1 and end > start:
                json_str = text[start:end]
                return json.loads(json_str)
            else:
                # 如果找不到 JSON，返回原始文本
                return {
                    'direction': 'Unclear',
                    'confidence': 'low',
                    'primary_evidence': 'Failed to parse',
                    'reasoning_chain': text,
                    'raw_response': text
                }
        except json.JSONDecodeError:
            return {
                'direction': 'Unclear',
                'confidence': 'low',
                'primary_evidence': 'JSON parse error',
                'reasoning_chain': text,
                'raw_response': text
            }
    
    def run_experiment(self, datasets: List[Dict], save_results: bool = True, 
                      output_file: str = 'llm_inference_results.json') -> List[Dict]:
        """
        对一批数据集运行完整的推理流程
        
        Parameters:
        -----------
        datasets : List[Dict]
            数据集列表（来自 CausalDataGenerator）
        save_results : bool
            是否保存结果到文件
        output_file : str
            输出文件路径
        
        Returns:
        --------
        List[Dict] : 每个数据集的完整推理结果
        """
        results = []
        
        print("\n" + "="*80)
        print(f"🚀 Starting Causal Reasoning Experiment with {len(datasets)} datasets")
        print(f"   Model: {self.model_name}")
        print(f"   Prompt Template: {self.prompt_template}")
        print("="*80)
        
        for i, ds in enumerate(datasets, 1):
            print(f"\n[{i}/{len(datasets)}] Processing: {ds['name']} | Ground Truth: {ds['ground_truth']}")
            
            # 步骤 1: 生成统计叙事
            analysis = self.analyze_pair(ds['X'], ds['Y'])
            
            # 步骤 2: LLM 推理
            inference = self.infer_causality(analysis['narrative'], use_structured_output=False)
            
            # 步骤 3: 评估
            ground_truth = ds['ground_truth']
            
            # 兼容不同的字段名 (LLM 有时会返回 'causal_direction' 而不是 'direction')
            predicted = (
                inference.get('direction') or 
                inference.get('causal_direction') or 
                inference.get('causal_direction_judgment') or
                inference.get('judgment') or
                inference.get('因果方向判断') or
                'Unclear'
            )
            
            is_correct = (predicted == ground_truth)
            
            result = {
                'dataset_name': ds['name'],
                'ground_truth': ground_truth,
                'description': ds['description'],
                'llm_prediction': predicted,
                'llm_confidence': inference.get('confidence', 'unknown'),
                'is_correct': is_correct,
                'primary_evidence': inference.get('primary_evidence', ''),
                'reasoning_chain': inference.get('reasoning_chain', ''),
                'statistical_signals': inference.get('statistical_signals', {}),
                'narrative': analysis['narrative'],
                'full_llm_response': inference
            }
            
            results.append(result)
            
            # 打印简要结果
            status = "✅ CORRECT" if is_correct else "❌ WRONG"
            if ground_truth not in ['A->B', 'B->A']:
                status = "⚠️  SPECIAL CASE"
            
            print(f"   Predicted: {predicted} ({inference.get('confidence', 'N/A')}) | {status}")
        
        # 计算统计
        causal_cases = [r for r in results if r['ground_truth'] in ['A->B', 'B->A']]
        if causal_cases:
            correct_count = sum(1 for r in causal_cases if r['is_correct'])
            accuracy = correct_count / len(causal_cases) * 100
            
            print("\n" + "="*80)
            print("📊 EXPERIMENT SUMMARY")
            print("="*80)
            print(f"Total Datasets: {len(datasets)}")
            print(f"Causal Cases (A->B or B->A): {len(causal_cases)}")
            print(f"Correct Predictions: {correct_count}/{len(causal_cases)}")
            print(f"Accuracy: {accuracy:.1f}%")
            print("="*80)
        
        # 保存结果
        if save_results:
            output_path = Path(project_root) / output_file
            with open(output_path, 'w', encoding='utf-8') as f:
                json.dump(results, f, indent=2, ensure_ascii=False)
            print(f"\n💾 Results saved to: {output_path}")
        
        return results
