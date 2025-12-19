"""
消融实验脚本 (Ablation Experiments)
验证 StatTranslator 各组件对 ACR 性能的贡献

实验设计：
1. 完整 ACR 叙事 (full): 包含所有统计特征和因果推理指导
2. 低阶叙事 (low_order): 仅包含相关系数和 R²
3. 原始数值 (raw): 仅原始统计数值，无解释

Requirements: 6.1, 6.2, 6.4
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import json
import numpy as np
import pandas as pd
from datetime import datetime
from pgmpy.utils import get_example_model
from pgmpy.sampling import BayesianModelSampling

from utils_set.causal_reasoning_engine import CausalReasoningEngine
from utils_set.stat_translator import StatTranslator
from utils_set.utils import path_config

RESULTS_DIR = str(path_config.results_dir)


class AblationExperimenter:
    """消融实验执行器"""
    
    def __init__(self):
        try:
            self.engine = CausalReasoningEngine()
            self.translator = StatTranslator()
            print(f"✅ Ablation Experimenter Initialized")
        except Exception as e:
            print(f"❌ Failed to initialize: {e}")
            self.engine = None
            self.translator = None
    
    def run_single_edge_ablation(self, df, source, target, true_edges_set, narrative_mode="full"):
        """
        对单条边运行消融实验
        
        Args:
            df: 数据框
            source, target: 边的两个节点
            true_edges_set: 真实边集合
            narrative_mode: 叙事模式 ("full", "low_order", "raw")
        
        Returns:
            dict: 包含预测结果和正确性
        """
        X = df[source].values
        Y = df[target].values
        
        try:
            # 使用指定的叙事模式生成分析
            analysis = self.engine.analyze_pair(X, Y, narrative_mode=narrative_mode)
            result = self.engine.infer_causality(analysis['narrative'])
            
            prediction = (
                result.get('direction') or 
                result.get('causal_direction') or 
                result.get('causal_direction_judgment') or
                'Unclear'
            )
            
            # 判断正确方向
            true_direction = None
            if (source, target) in true_edges_set:
                true_direction = "A->B"
            elif (target, source) in true_edges_set:
                true_direction = "B->A"
            
            # 判断预测是否正确
            if prediction == "A->B":
                pred_edge = (source, target)
                is_correct = (true_direction == "A->B")
            elif prediction == "B->A":
                pred_edge = (target, source)
                is_correct = (true_direction == "B->A")
            else:
                # Unclear: 保持原方向
                pred_edge = (source, target)
                is_correct = (true_direction == "A->B")
            
            return {
                'edge': f"{source}-{target}",
                'prediction': prediction,
                'true_direction': true_direction,
                'is_correct': is_correct,
                'pred_edge': pred_edge,
                'confidence': result.get('confidence', 'unknown')
            }
            
        except Exception as e:
            print(f"  ⚠️  Error on {source}-{target}: {e}")
            return {
                'edge': f"{source}-{target}",
                'prediction': 'Error',
                'true_direction': None,
                'is_correct': False,
                'pred_edge': (source, target),
                'error': str(e)
            }
    
    def run_ablation_on_network(self, network_name, sample_size=1000, max_edges=None):
        """
        在单个网络上运行三种叙事模式的消融实验
        
        Args:
            network_name: 网络名称 (asia, sachs, alarm, etc.)
            sample_size: 采样大小
            max_edges: 最大测试边数（用于大网络）
        
        Returns:
            dict: 包含三种模式的实验结果
        """
        print(f"\n{'='*60}")
        print(f"Ablation Experiment on {network_name}")
        print(f"{'='*60}")
        
        # 加载网络
        model = get_example_model(network_name)
        sampler = BayesianModelSampling(model)
        df = sampler.forward_sample(size=sample_size)
        
        true_edges = list(model.edges())
        true_edges_set = set(true_edges)
        n_edges = len(true_edges)
        
        # 限制测试边数
        test_edges = true_edges
        if max_edges and n_edges > max_edges:
            np.random.seed(42)
            indices = np.random.choice(n_edges, max_edges, replace=False)
            test_edges = [true_edges[i] for i in indices]
            print(f"Network: {network_name}, Total Edges: {n_edges}, Testing: {len(test_edges)}")
        else:
            print(f"Network: {network_name}, Edges: {n_edges}")
        
        results = {}
        modes = ["full", "low_order", "raw"]
        
        for mode in modes:
            print(f"\n--- Running {mode.upper()} mode ---")
            
            mode_results = []
            correct_count = 0
            total_count = 0
            pred_edges = []
            
            for source, target in test_edges:
                result = self.run_single_edge_ablation(
                    df, source, target, true_edges_set, narrative_mode=mode
                )
                mode_results.append(result)
                pred_edges.append(result['pred_edge'])
                
                if result['true_direction']:
                    total_count += 1
                    if result['is_correct']:
                        correct_count += 1
                
                status = "✅" if result['is_correct'] else "❌"
                print(f"  {status} {result['edge']}: pred={result['prediction']}, true={result['true_direction']}")
            
            accuracy = correct_count / total_count if total_count > 0 else 0
            
            # 计算 SHD
            shd = self._compute_shd(true_edges, pred_edges)
            
            results[mode] = {
                'accuracy': accuracy,
                'correct': correct_count,
                'total': total_count,
                'shd': shd,
                'details': mode_results
            }
            
            print(f"\n  📊 {mode.upper()} Results: Accuracy={accuracy:.1%}, SHD={shd}")
        
        # 计算组件贡献
        contribution = self._compute_contribution(results)
        
        return {
            'network': network_name,
            'sample_size': sample_size,
            'n_edges': n_edges,
            'n_tested': len(test_edges),
            'results': results,
            'contribution': contribution
        }
    
    def _compute_shd(self, true_edges, pred_edges):
        """计算 SHD"""
        true_set = set(true_edges)
        pred_set = set(pred_edges)
        
        missing = len(true_set - pred_set)
        extra = len(pred_set - true_set)
        
        return missing + extra
    
    def _compute_contribution(self, results):
        """
        计算各组件的贡献百分比
        
        高阶组件贡献 = (SHD_low - SHD_full) / SHD_low * 100%
        叙事贡献 = (SHD_raw - SHD_full) / SHD_raw * 100%
        """
        shd_full = results['full']['shd']
        shd_low = results['low_order']['shd']
        shd_raw = results['raw']['shd']
        
        acc_full = results['full']['accuracy']
        acc_low = results['low_order']['accuracy']
        acc_raw = results['raw']['accuracy']
        
        # 高阶统计量贡献（HSIC、ANM 残差独立性等）
        if shd_low > 0:
            high_order_contribution_shd = (shd_low - shd_full) / shd_low * 100
        else:
            high_order_contribution_shd = 0 if shd_full == 0 else -100
        
        high_order_contribution_acc = (acc_full - acc_low) * 100
        
        # 叙事翻译贡献（相对于原始数值）
        if shd_raw > 0:
            narrative_contribution_shd = (shd_raw - shd_full) / shd_raw * 100
        else:
            narrative_contribution_shd = 0 if shd_full == 0 else -100
        
        narrative_contribution_acc = (acc_full - acc_raw) * 100
        
        return {
            'high_order_contribution_shd_pct': high_order_contribution_shd,
            'high_order_contribution_acc_pct': high_order_contribution_acc,
            'narrative_contribution_shd_pct': narrative_contribution_shd,
            'narrative_contribution_acc_pct': narrative_contribution_acc,
            'shd_full': shd_full,
            'shd_low_order': shd_low,
            'shd_raw': shd_raw,
            'acc_full': acc_full,
            'acc_low_order': acc_low,
            'acc_raw': acc_raw
        }
    
    def run_full_ablation_study(self, networks=None, sample_size=1000, max_edges_per_network=20):
        """
        运行完整的消融实验
        
        Args:
            networks: 要测试的网络列表
            sample_size: 采样大小
            max_edges_per_network: 每个网络最大测试边数
        
        Returns:
            dict: 完整实验结果
        """
        if networks is None:
            networks = ["asia", "sachs", "alarm"]
        
        print("\n" + "="*70)
        print("🔬 ABLATION STUDY: StatTranslator Component Contribution")
        print("="*70)
        print(f"Networks: {networks}")
        print(f"Sample size: {sample_size}")
        print(f"Max edges per network: {max_edges_per_network}")
        
        all_results = {}
        
        for network in networks:
            try:
                result = self.run_ablation_on_network(
                    network, 
                    sample_size=sample_size,
                    max_edges=max_edges_per_network
                )
                all_results[network] = result
            except Exception as e:
                print(f"❌ Failed on {network}: {e}")
                all_results[network] = {'error': str(e)}
        
        # 汇总统计
        summary = self._generate_summary(all_results)
        
        return {
            'experiment': 'ablation_study',
            'timestamp': datetime.now().isoformat(),
            'config': {
                'networks': networks,
                'sample_size': sample_size,
                'max_edges_per_network': max_edges_per_network
            },
            'results': all_results,
            'summary': summary
        }
    
    def _generate_summary(self, all_results):
        """生成汇总统计"""
        valid_results = {k: v for k, v in all_results.items() if 'error' not in v}
        
        if not valid_results:
            return {'error': 'No valid results'}
        
        # 计算平均贡献
        avg_high_order_shd = np.mean([
            v['contribution']['high_order_contribution_shd_pct'] 
            for v in valid_results.values()
        ])
        avg_high_order_acc = np.mean([
            v['contribution']['high_order_contribution_acc_pct'] 
            for v in valid_results.values()
        ])
        avg_narrative_shd = np.mean([
            v['contribution']['narrative_contribution_shd_pct'] 
            for v in valid_results.values()
        ])
        avg_narrative_acc = np.mean([
            v['contribution']['narrative_contribution_acc_pct'] 
            for v in valid_results.values()
        ])
        
        # 平均准确率
        avg_acc_full = np.mean([v['results']['full']['accuracy'] for v in valid_results.values()])
        avg_acc_low = np.mean([v['results']['low_order']['accuracy'] for v in valid_results.values()])
        avg_acc_raw = np.mean([v['results']['raw']['accuracy'] for v in valid_results.values()])
        
        summary = {
            'n_networks': len(valid_results),
            'avg_accuracy': {
                'full': avg_acc_full,
                'low_order': avg_acc_low,
                'raw': avg_acc_raw
            },
            'avg_contribution': {
                'high_order_shd_pct': avg_high_order_shd,
                'high_order_acc_pct': avg_high_order_acc,
                'narrative_shd_pct': avg_narrative_shd,
                'narrative_acc_pct': avg_narrative_acc
            }
        }
        
        print("\n" + "="*70)
        print("📊 ABLATION STUDY SUMMARY")
        print("="*70)
        print(f"\nAverage Accuracy by Mode:")
        print(f"  Full ACR:    {avg_acc_full:.1%}")
        print(f"  Low-Order:   {avg_acc_low:.1%}")
        print(f"  Raw Numeric: {avg_acc_raw:.1%}")
        print(f"\nComponent Contribution:")
        print(f"  High-Order Statistics (HSIC, ANM): {avg_high_order_acc:.1f}% accuracy gain")
        print(f"  Narrative Translation: {avg_narrative_acc:.1f}% accuracy gain over raw")
        print("="*70)
        
        return summary


def main():
    import argparse
    parser = argparse.ArgumentParser(description='Ablation Experiments')
    parser.add_argument('--networks', type=str, nargs='+', default=['asia', 'sachs', 'alarm'],
                        help='Networks to test')
    parser.add_argument('--sample_size', type=int, default=1000,
                        help='Sample size for experiments')
    parser.add_argument('--max_edges', type=int, default=20,
                        help='Max edges to test per network')
    parser.add_argument('--output', type=str, default=None,
                        help='Output file path')
    args = parser.parse_args()
    
    experimenter = AblationExperimenter()
    if not experimenter.engine:
        print("Engine initialization failed. Exiting.")
        return
    
    results = experimenter.run_full_ablation_study(
        networks=args.networks,
        sample_size=args.sample_size,
        max_edges_per_network=args.max_edges
    )
    
    # 保存结果
    output_file = args.output or os.path.join(RESULTS_DIR, 'ablation_experiment_results.json')
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(results, f, indent=2, ensure_ascii=False, default=str)
    print(f"\n💾 Results saved to: {output_file}")


if __name__ == "__main__":
    main()
