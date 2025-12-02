"""
运行因果推理实验的主程序
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import argparse
from utils_set.data_generator import CausalDataGenerator
from utils_set.causal_reasoning_engine import CausalReasoningEngine
from utils_set.utils import path_config

def main():
    parser = argparse.ArgumentParser(description='ACR 框架 - 因果推理实验')
    parser.add_argument('--model', type=str, default=None, 
                       help='LLM 模型名称（不指定则使用 config.yaml 中的 used_model）')
    parser.add_argument('--prompt', type=str, default='sherlock',
                       choices=['sherlock', 'simple', 'residual_only'],
                       help='Prompt 模板类型')
    parser.add_argument('--samples', type=int, default=1000,
                       help='每个数据集的样本数量')
    parser.add_argument('--output', type=str, default=str(path_config.experiment_results_file),
                       help='结果输出文件名')
    
    args = parser.parse_args()
    
    print("="*80)
    print("ACR 框架 - 因果推理实验")
    print("="*80)
    
    # 生成测试数据
    print("\n📊 生成合成因果数据集...")
    generator = CausalDataGenerator(random_seed=42)
    datasets = generator.generate_batch(n_samples=args.samples)
    print(f"✅ 生成了 {len(datasets)} 个数据集，每个包含 {args.samples} 个样本")
    
    # 初始化推理引擎
    print(f"\n🤖 初始化因果推理引擎...")
    engine = CausalReasoningEngine(
        model_name=args.model,
        prompt_template=args.prompt
    )
    
    # 运行实验
    print(f"\n🚀 开始推理实验...")
    results = engine.run_experiment(
        datasets,
        save_results=True,
        output_file=args.output
    )
    
    print("\n" + "="*80)
    print("✅ 实验完成！")
    print("="*80)

if __name__ == "__main__":
    main()
