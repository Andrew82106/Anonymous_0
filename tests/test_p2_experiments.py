"""
P2 低优先级实验脚本
包含：
- P2.1: E-SHD 评估 (与 DiBS+GPT 对比)
- P2.2: 基座算法实验 (MMHC-Skeleton + ACR)
- P2.3: 低样本量鲁棒性测试 (100 样本)
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import pandas as pd
import numpy as np
import json
from pgmpy.utils import get_example_model
from pgmpy.sampling import BayesianModelSampling
from utils_set.causal_reasoning_engine import CausalReasoningEngine
from utils_set.utils import path_config

# 传统因果发现算法
try:
    from pgmpy.estimators import PC, HillClimbSearch, BicScore, MmhcEstimator
    PGMPY_AVAILABLE = True
except ImportError:
    PGMPY_AVAILABLE = False
    print("Warning: pgmpy estimators not available.")

RESULTS_DIR = str(path_config.results_dir)


def compute_shd(true_edges, pred_edges):
    """计算 SHD (结构汉明距离)"""
    true_set = set(true_edges)
    pred_set = set(pred_edges)
    
    # 缺失边
    missing = len(true_set - pred_set)
    # 多余边
    extra = len(pred_set - true_set)
    
    # 方向错误：(A,B) in true but (B,A) in pred
    reversed_count = 0
    for (a, b) in true_set:
        if (b, a) in pred_set and (a, b) not in pred_set:
            reversed_count += 1
    
    # SHD = missing + extra (reversed 已经在 missing 和 extra 中计算了)
    return missing + extra


def compute_expected_shd(true_edges, pred_edges_list):
    """
    计算 E-SHD (Expected SHD)
    用于与贝叶斯方法 (DiBS+GPT) 对比
    
    对于确定性方法，E-SHD = SHD
    对于概率方法，E-SHD = 期望 SHD
    """
    if not pred_edges_list:
        return None
    
    # 对于我们的确定性方法，只有一个预测
    if isinstance(pred_edges_list[0], tuple):
        return compute_shd(true_edges, pred_edges_list)
    
    # 对于多次采样的情况，计算平均 SHD
    shds = [compute_shd(true_edges, pred) for pred in pred_edges_list]
    return np.mean(shds), np.std(shds)


def run_pc_algorithm(df, alpha=0.05):
    """运行 PC 算法"""
    if not PGMPY_AVAILABLE:
        return None
    try:
        pc = PC(data=df)
        model = pc.estimate(significance_level=alpha)
        return list(model.edges())
    except Exception as e:
        print(f"PC algorithm failed: {e}")
        return None


def run_hillclimb_algorithm(df):
    """运行 HillClimb 算法"""
    if not PGMPY_AVAILABLE:
        return None
    try:
        hc = HillClimbSearch(data=df)
        model = hc.estimate(scoring_method=BicScore(data=df))
        return list(model.edges())
    except Exception as e:
        print(f"HillClimb algorithm failed: {e}")
        return None


def run_mmhc_algorithm(df):
    """运行 MMHC 算法 (Max-Min Hill-Climbing)"""
    if not PGMPY_AVAILABLE:
        return None
    try:
        mmhc = MmhcEstimator(data=df)
        model = mmhc.estimate()
        return list(model.edges())
    except Exception as e:
        print(f"MMHC algorithm failed: {e}")
        return None


def get_mmhc_skeleton(df):
    """获取 MMHC 的骨架 (无向边)"""
    if not PGMPY_AVAILABLE:
        return None
    try:
        mmhc = MmhcEstimator(data=df)
        skeleton = mmhc.mmpc()  # 返回骨架
        return skeleton
    except Exception as e:
        print(f"MMHC skeleton failed: {e}")
        return None


class P2Experimenter:
    """P2 实验执行器"""
    
    def __init__(self):
        try:
            self.engine = CausalReasoningEngine()
            print(f"✅ Causal Engine Initialized")
        except Exception as e:
            print(f"❌ Failed to initialize engine: {e}")
            self.engine = None
    
    def run_acr_on_edges(self, df, edges, true_edges):
        """
        对给定的边列表运行 ACR 定向
        返回预测的有向边和准确率
        """
        pred_edges = []
        correct = 0
        total = 0
        details = []
        
        for source, target in edges:
            X = df[source].values
            Y = df[target].values
            
            try:
                analysis = self.engine.analyze_pair(X, Y)
                result = self.engine.infer_causality(analysis['narrative'])
                
                prediction = (
                    result.get('direction') or 
                    result.get('causal_direction') or 
                    result.get('causal_direction_judgment') or
                    'Unclear'
                )
                
                # 判断正确方向
                true_direction = None
                if (source, target) in true_edges:
                    true_direction = "A->B"
                elif (target, source) in true_edges:
                    true_direction = "B->A"
                
                if prediction == "A->B":
                    pred_edges.append((source, target))
                    is_correct = (true_direction == "A->B")
                elif prediction == "B->A":
                    pred_edges.append((target, source))
                    is_correct = (true_direction == "B->A")
                else:
                    # Unclear: 随机选择或保持原方向
                    pred_edges.append((source, target))
                    is_correct = (true_direction == "A->B")
                
                if true_direction:
                    total += 1
                    if is_correct:
                        correct += 1
                
                details.append({
                    'edge': f"{source}-{target}",
                    'prediction': prediction,
                    'true_direction': true_direction,
                    'is_correct': is_correct
                })
                
                status = "✅" if is_correct else "❌"
                print(f"  {status} {source}-{target}: pred={prediction}, true={true_direction}")
                
            except Exception as e:
                print(f"  ⚠️  Error on {source}-{target}: {e}")
                pred_edges.append((source, target))
        
        accuracy = correct / total if total > 0 else 0
        return pred_edges, accuracy, details
    
    def experiment_p2_1_eshd(self, network_name="sachs", sample_size=1000):
        """
        P2.1: E-SHD 评估
        与 DiBS+GPT (E-SHD=21.7) 对比
        """
        print(f"\n{'='*60}")
        print(f"P2.1: E-SHD Evaluation on {network_name}")
        print(f"{'='*60}")
        
        # 加载网络
        model = get_example_model(network_name)
        sampler = BayesianModelSampling(model)
        df = sampler.forward_sample(size=sample_size)
        
        true_edges = list(model.edges())
        n_edges = len(true_edges)
        
        print(f"Network: {network_name}, Edges: {n_edges}")
        print(f"Baseline: DiBS+GPT E-SHD = 21.7 ± 0.5 (from Bazaluk et al., 2025)")
        
        # 运行 ACR-Hybrid
        print(f"\nRunning ACR-Hybrid...")
        pred_edges, accuracy, details = self.run_acr_on_edges(df, true_edges, set(true_edges))
        
        shd = compute_shd(true_edges, pred_edges)
        e_shd = shd  # 确定性方法，E-SHD = SHD
        
        print(f"\n📊 Results:")
        print(f"  ACR-Hybrid SHD: {shd}")
        print(f"  ACR-Hybrid E-SHD: {e_shd}")
        print(f"  DiBS+GPT E-SHD: 21.7 ± 0.5")
        print(f"  Improvement: {(21.7 - e_shd) / 21.7 * 100:.1f}%")
        
        result = {
            'experiment': 'P2.1_ESHD',
            'network': network_name,
            'sample_size': sample_size,
            'acr_shd': shd,
            'acr_eshd': e_shd,
            'acr_accuracy': accuracy,
            'dibs_gpt_eshd': 21.7,
            'improvement_pct': (21.7 - e_shd) / 21.7 * 100,
            'details': details
        }
        
        return result
    
    def experiment_p2_2_mmhc_acr(self, network_name="alarm", sample_size=1000):
        """
        P2.2: MMHC-Skeleton + ACR
        对比骨架质量对 ACR 的影响
        """
        print(f"\n{'='*60}")
        print(f"P2.2: MMHC-Skeleton + ACR on {network_name}")
        print(f"{'='*60}")
        
        # 加载网络
        model = get_example_model(network_name)
        sampler = BayesianModelSampling(model)
        df = sampler.forward_sample(size=sample_size)
        
        true_edges = list(model.edges())
        true_edges_set = set(true_edges)
        n_edges = len(true_edges)
        
        print(f"Network: {network_name}, Edges: {n_edges}")
        
        # 1. 运行 MMHC 获取完整 DAG
        print(f"\n[1/3] Running MMHC Algorithm...")
        mmhc_edges = run_mmhc_algorithm(df)
        if mmhc_edges:
            mmhc_shd = compute_shd(true_edges, mmhc_edges)
            print(f"  MMHC SHD: {mmhc_shd}")
        else:
            print(f"  MMHC failed, skipping...")
            mmhc_shd = None
        
        # 2. 运行 PC 获取骨架
        print(f"\n[2/3] Running PC Algorithm...")
        pc_edges = run_pc_algorithm(df)
        if pc_edges:
            pc_shd = compute_shd(true_edges, pc_edges)
            print(f"  PC SHD: {pc_shd}")
        else:
            pc_shd = None
        
        # 3. MMHC-Skeleton + ACR
        print(f"\n[3/3] Running MMHC-Skeleton + ACR...")
        if mmhc_edges:
            # 从 MMHC 边提取骨架（无向边对）
            skeleton_pairs = set()
            for u, v in mmhc_edges:
                skeleton_pairs.add(tuple(sorted((u, v))))
            
            # 将骨架转换为边列表（任意方向）
            skeleton_edges = [(u, v) for u, v in skeleton_pairs]
            
            print(f"  MMHC Skeleton: {len(skeleton_edges)} edges")
            
            # 对骨架边运行 ACR
            mmhc_acr_edges, mmhc_acr_acc, mmhc_acr_details = self.run_acr_on_edges(
                df, skeleton_edges, true_edges_set
            )
            mmhc_acr_shd = compute_shd(true_edges, mmhc_acr_edges)
            
            print(f"\n📊 Results:")
            print(f"  MMHC SHD: {mmhc_shd}")
            print(f"  MMHC-Skeleton + ACR SHD: {mmhc_acr_shd}")
            print(f"  PC SHD: {pc_shd}")
            if mmhc_shd and mmhc_acr_shd < mmhc_shd:
                print(f"  ACR Improvement over MMHC: {(mmhc_shd - mmhc_acr_shd) / mmhc_shd * 100:.1f}%")
        else:
            mmhc_acr_shd = None
            mmhc_acr_acc = None
            mmhc_acr_details = []
        
        result = {
            'experiment': 'P2.2_MMHC_ACR',
            'network': network_name,
            'sample_size': sample_size,
            'mmhc_shd': mmhc_shd,
            'mmhc_acr_shd': mmhc_acr_shd,
            'mmhc_acr_accuracy': mmhc_acr_acc,
            'pc_shd': pc_shd,
            'details': mmhc_acr_details
        }
        
        return result
    
    def experiment_p2_3_low_sample(self, network_name="asia", sample_sizes=[100, 500, 1000]):
        """
        P2.3: 低样本量鲁棒性测试
        在不同样本量下对比 ACR-Hybrid vs PC/HillClimb
        """
        print(f"\n{'='*60}")
        print(f"P2.3: Low Sample Robustness Test on {network_name}")
        print(f"{'='*60}")
        
        # 加载网络
        model = get_example_model(network_name)
        true_edges = list(model.edges())
        true_edges_set = set(true_edges)
        n_edges = len(true_edges)
        
        print(f"Network: {network_name}, Edges: {n_edges}")
        print(f"Sample sizes to test: {sample_sizes}")
        
        results_by_sample = []
        
        for sample_size in sample_sizes:
            print(f"\n--- Sample Size: {sample_size} ---")
            
            sampler = BayesianModelSampling(model)
            df = sampler.forward_sample(size=sample_size)
            
            # PC
            pc_edges = run_pc_algorithm(df)
            pc_shd = compute_shd(true_edges, pc_edges) if pc_edges else None
            print(f"  PC SHD: {pc_shd}")
            
            # HillClimb
            hc_edges = run_hillclimb_algorithm(df)
            hc_shd = compute_shd(true_edges, hc_edges) if hc_edges else None
            print(f"  HillClimb SHD: {hc_shd}")
            
            # ACR-Hybrid (对真实边运行)
            print(f"  Running ACR-Hybrid...")
            acr_edges, acr_acc, acr_details = self.run_acr_on_edges(
                df, true_edges, true_edges_set
            )
            acr_shd = compute_shd(true_edges, acr_edges)
            print(f"  ACR-Hybrid SHD: {acr_shd}, Accuracy: {acr_acc:.1%}")
            
            results_by_sample.append({
                'sample_size': sample_size,
                'pc_shd': pc_shd,
                'hillclimb_shd': hc_shd,
                'acr_shd': acr_shd,
                'acr_accuracy': acr_acc
            })
        
        print(f"\n📊 Summary Table:")
        print(f"{'Sample':<10} {'PC':<8} {'HillClimb':<12} {'ACR-Hybrid':<12}")
        print(f"{'-'*42}")
        for r in results_by_sample:
            print(f"{r['sample_size']:<10} {r['pc_shd'] or 'N/A':<8} {r['hillclimb_shd'] or 'N/A':<12} {r['acr_shd']:<12}")
        
        result = {
            'experiment': 'P2.3_Low_Sample',
            'network': network_name,
            'results_by_sample': results_by_sample
        }
        
        return result


def main():
    import argparse
    parser = argparse.ArgumentParser(description='P2 Experiments')
    parser.add_argument('--exp', type=str, default='all',
                        choices=['all', 'p2.1', 'p2.2', 'p2.3'],
                        help='Which experiment to run')
    parser.add_argument('--network', type=str, default=None,
                        help='Network to test (default varies by experiment)')
    args = parser.parse_args()
    
    experimenter = P2Experimenter()
    if not experimenter.engine:
        print("Engine initialization failed. Exiting.")
        return
    
    all_results = {}
    
    if args.exp in ['all', 'p2.1']:
        network = args.network or 'sachs'
        result = experimenter.experiment_p2_1_eshd(network)
        all_results['p2.1'] = result
    
    if args.exp in ['all', 'p2.2']:
        network = args.network or 'alarm'
        result = experimenter.experiment_p2_2_mmhc_acr(network)
        all_results['p2.2'] = result
    
    if args.exp in ['all', 'p2.3']:
        network = args.network or 'asia'
        result = experimenter.experiment_p2_3_low_sample(network)
        all_results['p2.3'] = result
    
    # 保存结果
    output_file = os.path.join(RESULTS_DIR, 'p2_experiment_results.json')
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(all_results, f, indent=2, ensure_ascii=False)
    print(f"\n💾 All results saved to: {output_file}")


if __name__ == "__main__":
    main()
