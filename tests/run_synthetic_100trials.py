"""
合成数据100次重复实验
每种数据类型运行100次测试，每次1000个样本，计算准确率

Requirements: 验证 ACR 框架在合成数据上的统计显著性
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import json
import numpy as np
from datetime import datetime
from tqdm import tqdm

from utils_set.data_generator import CausalDataGenerator
from utils_set.causal_reasoning_engine import CausalReasoningEngine
from utils_set.utils import path_config

RESULTS_DIR = str(path_config.results_dir)


def run_single_trial(engine, generator, dataset_type, n_samples=1000, trial_seed=None):
    """
    运行单次实验
    
    Returns:
        dict: 包含预测结果和正确性
    """
    # 设置随机种子
    if trial_seed is not None:
        np.random.seed(trial_seed)
    
    # 生成数据
    X, Y, ground_truth, description = generator.generate_dataset(dataset_type, n_samples)
    
    try:
        # 运行 ACR 分析
        analysis = engine.analyze_pair(X, Y)
        result = engine.infer_causality(analysis['narrative'])
        
        # 提取预测
        prediction = (
            result.get('direction') or 
            result.get('causal_direction') or 
            result.get('causal_direction_judgment') or
            'Unclear'
        )
        
        confidence = result.get('confidence', 'unknown')
        
        # 判断正确性
        is_correct = False
        if ground_truth == 'A->B' and prediction == 'A->B':
            is_correct = True
        elif ground_truth == 'B->A' and prediction == 'B->A':
            is_correct = True
        elif ground_truth == 'A_|_B' and prediction in ['A_|_B', 'Independent', 'Unclear']:
            is_correct = True
        elif ground_truth == 'A<-Z->B':
            # 混淆因子情况：预测 A->B 或 B->A 都算"错误"（因为真实是无直接因果）
            # 但这是方法论的固有局限，不是错误
            is_correct = False
        
        return {
            'prediction': prediction,
            'ground_truth': ground_truth,
            'confidence': confidence,
            'is_correct': is_correct,
            'error': None
        }
        
    except Exception as e:
        return {
            'prediction': 'Error',
            'ground_truth': ground_truth,
            'confidence': 'unknown',
            'is_correct': False,
            'error': str(e)
        }


def run_100_trials(engine, dataset_type, n_trials=100, n_samples=1000):
    """
    对单一数据类型运行100次实验
    """
    print(f"\n{'='*60}")
    print(f"Running {n_trials} trials for: {dataset_type.upper()}")
    print(f"{'='*60}")
    
    results = []
    correct_count = 0
    
    for trial in tqdm(range(n_trials), desc=f"{dataset_type}"):
        # 每次使用不同的随机种子
        generator = CausalDataGenerator(random_seed=trial * 1000 + 42)
        
        result = run_single_trial(
            engine, generator, dataset_type, 
            n_samples=n_samples, 
            trial_seed=trial * 1000 + 42
        )
        results.append(result)
        
        if result['is_correct']:
            correct_count += 1
    
    accuracy = correct_count / n_trials
    
    # 统计预测分布
    predictions = [r['prediction'] for r in results]
    prediction_counts = {}
    for p in predictions:
        prediction_counts[p] = prediction_counts.get(p, 0) + 1
    
    # 统计置信度分布
    confidences = [r['confidence'] for r in results]
    confidence_counts = {}
    for c in confidences:
        confidence_counts[c] = confidence_counts.get(c, 0) + 1
    
    print(f"\n📊 {dataset_type.upper()} Results:")
    print(f"   Accuracy: {accuracy:.1%} ({correct_count}/{n_trials})")
    print(f"   Predictions: {prediction_counts}")
    print(f"   Confidences: {confidence_counts}")
    
    return {
        'dataset_type': dataset_type,
        'n_trials': n_trials,
        'n_samples': n_samples,
        'accuracy': accuracy,
        'correct_count': correct_count,
        'prediction_distribution': prediction_counts,
        'confidence_distribution': confidence_counts,
        'details': results
    }


def main():
    import argparse
    parser = argparse.ArgumentParser(description='Synthetic Data 100 Trials Experiment')
    parser.add_argument('--n_trials', type=int, default=100,
                        help='Number of trials per dataset type')
    parser.add_argument('--n_samples', type=int, default=1000,
                        help='Number of samples per trial')
    parser.add_argument('--types', type=str, nargs='+', 
                        default=['lingam', 'anm', 'reverse', 'independent', 'confounder'],
                        help='Dataset types to test')
    parser.add_argument('--output', type=str, default=None,
                        help='Output file path')
    args = parser.parse_args()
    
    print("\n" + "="*70)
    print("🔬 SYNTHETIC DATA 100 TRIALS EXPERIMENT")
    print("="*70)
    print(f"Trials per type: {args.n_trials}")
    print(f"Samples per trial: {args.n_samples}")
    print(f"Dataset types: {args.types}")
    
    # 初始化引擎
    try:
        engine = CausalReasoningEngine()
        print("✅ Causal Engine Initialized")
    except Exception as e:
        print(f"❌ Failed to initialize engine: {e}")
        return
    
    # 运行实验
    all_results = {}
    
    for dtype in args.types:
        result = run_100_trials(
            engine, dtype, 
            n_trials=args.n_trials, 
            n_samples=args.n_samples
        )
        all_results[dtype] = result
    
    # 汇总
    print("\n" + "="*70)
    print("📊 FINAL SUMMARY")
    print("="*70)
    print(f"\n{'Dataset Type':<20} {'Accuracy':<15} {'Correct/Total':<15}")
    print("-"*50)
    
    direct_causal_correct = 0
    direct_causal_total = 0
    
    for dtype, result in all_results.items():
        acc = result['accuracy']
        correct = result['correct_count']
        total = result['n_trials']
        print(f"{dtype:<20} {acc:.1%}{'':<10} {correct}/{total}")
        
        # 统计直接因果案例（排除混淆因子）
        if dtype != 'confounder':
            direct_causal_correct += correct
            direct_causal_total += total
    
    if direct_causal_total > 0:
        direct_acc = direct_causal_correct / direct_causal_total
        print("-"*50)
        print(f"{'Direct Causal Cases':<20} {direct_acc:.1%}{'':<10} {direct_causal_correct}/{direct_causal_total}")
    
    print("="*70)
    
    # 保存结果
    output_file = args.output or os.path.join(RESULTS_DIR, 'synthetic_100trials_results.json')
    
    # 移除 details 以减小文件大小（可选）
    save_results = {
        'experiment': 'synthetic_100_trials',
        'timestamp': datetime.now().isoformat(),
        'config': {
            'n_trials': args.n_trials,
            'n_samples': args.n_samples,
            'dataset_types': args.types
        },
        'results': {
            dtype: {
                'accuracy': r['accuracy'],
                'correct_count': r['correct_count'],
                'n_trials': r['n_trials'],
                'prediction_distribution': r['prediction_distribution'],
                'confidence_distribution': r['confidence_distribution']
            }
            for dtype, r in all_results.items()
        },
        'summary': {
            'direct_causal_accuracy': direct_acc if direct_causal_total > 0 else None,
            'direct_causal_correct': direct_causal_correct,
            'direct_causal_total': direct_causal_total
        }
    }
    
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(save_results, f, indent=2, ensure_ascii=False)
    
    print(f"\n💾 Results saved to: {output_file}")


if __name__ == "__main__":
    main()
