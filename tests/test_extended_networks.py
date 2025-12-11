"""
扩展网络测试脚本
测试 Child, Insurance, Hailfinder, Hepar II 等网络
使用 pgmpy 加载网络数据
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
    from pgmpy.estimators import PC, HillClimbSearch, BicScore
    PGMPY_AVAILABLE = True
except ImportError:
    PGMPY_AVAILABLE = False
    print("Warning: pgmpy estimators not available.")

RESULTS_FILE = str(path_config.results_dir / "extended_network_results.json")

def compute_shd(true_edges, pred_edges, n_nodes):
    """计算 SHD"""
    true_set = set(true_edges)
    pred_set = set(pred_edges)
    
    # 缺失边 + 多余边 + 方向错误边
    missing = len(true_set - pred_set)
    extra = len(pred_set - true_set)
    
    # 方向错误：(A,B) in true but (B,A) in pred
    reversed_edges = 0
    for (a, b) in true_set:
        if (b, a) in pred_set:
            reversed_edges += 1
    
    return missing + extra - reversed_edges  # 避免重复计算

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

def test_network_pgmpy(network_name, engine, sample_size=1000):
    """使用 pgmpy 测试网络"""
    print(f"\n{'='*60}")
    print(f"Testing Network: {network_name}")
    print(f"{'='*60}")
    
    # 1. 加载网络
    try:
        model = get_example_model(network_name)
        sampler = BayesianModelSampling(model)
        df = sampler.forward_sample(size=sample_size)
    except Exception as e:
        print(f"Error loading network {network_name}: {e}")
        return None
    
    nodes = list(model.nodes())
    edges = list(model.edges())
    n_nodes = len(nodes)
    n_edges = len(edges)
    
    print(f"Data: {df.shape[0]} samples, {df.shape[1]} variables")
    print(f"Ground Truth: {n_edges} edges in DAG")
    
    # 2. LLM-based Blind Causal Discovery
    print(f"\n[1/4] Running LLM-based Blind Causal Discovery...")
    
    llm_edges = []
    pairwise_results = []
    correct_count = 0
    llm_queries = 0
    
    for source, target in edges:
        X = df[source].values
        Y = df[target].values
        
        try:
            analysis = engine.analyze_pair(X, Y)
            result = engine.infer_causality(analysis['narrative'])
            llm_queries += 1
            
            prediction = (
                result.get('direction') or 
                result.get('causal_direction') or 
                result.get('causal_direction_judgment') or
                'Unclear'
            )
            
            if prediction == "A->B":
                llm_edges.append((source, target))
                is_correct = True
                correct_count += 1
            elif prediction == "B->A":
                llm_edges.append((target, source))
                is_correct = False
            else:
                is_correct = False
            
            pairwise_results.append({
                'pair': f"{source}->{target}",
                'prediction': prediction,
                'is_correct': is_correct
            })
            
            status = "✅" if is_correct else "❌"
            print(f"  {status} {source}->{target}: {prediction}")
            
        except Exception as e:
            print(f"  ⚠️  Error on {source}->{target}: {e}")
    
    llm_accuracy = correct_count / len(edges) if edges else 0
    llm_shd = compute_shd(edges, llm_edges, n_nodes)
    
    # 3. Traditional Baselines
    print(f"\n[2/4] Running PC Algorithm...")
    pc_edges = run_pc_algorithm(df)
    pc_shd = compute_shd(edges, pc_edges, n_nodes) if pc_edges else None
    if pc_shd is not None:
        print(f"  PC Algorithm SHD: {pc_shd}")
    
    print(f"\n[3/4] Running HillClimb Algorithm...")
    hc_edges = run_hillclimb_algorithm(df)
    hc_shd = compute_shd(edges, hc_edges, n_nodes) if hc_edges else None
    if hc_shd is not None:
        print(f"  HillClimb SHD: {hc_shd}")
    
    print(f"\n[4/4] Random Baseline...")
    random_shd = n_edges * 2  # 估计值
    print(f"  Random SHD (estimated): {random_shd}")
    
    # 4. Report
    print(f"\n{'='*60}")
    print(f"📊 RESULTS - {network_name}")
    print(f"{'='*60}")
    print(f"LLM (Blind): SHD={llm_shd}, Accuracy={llm_accuracy:.1%}, Queries={llm_queries}")
    if pc_shd is not None:
        print(f"PC Algorithm: SHD={pc_shd}")
    if hc_shd is not None:
        print(f"HillClimb: SHD={hc_shd}")
    
    return {
        'network': network_name,
        'n_nodes': n_nodes,
        'n_edges': n_edges,
        'llm': {
            'shd': llm_shd,
            'accuracy': llm_accuracy,
            'queries': llm_queries,
            'pairwise_details': pairwise_results
        },
        'pc': {'shd': pc_shd} if pc_shd is not None else None,
        'hillclimb': {'shd': hc_shd} if hc_shd is not None else None,
        'random': {'shd': random_shd}
    }

def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--network', type=str, default='child',
                        help='Network to test: child, insurance, hailfinder, hepar2')
    parser.add_argument('--sample_size', type=int, default=1000)
    args = parser.parse_args()
    
    # 初始化推理引擎
    try:
        engine = CausalReasoningEngine()
    except Exception as e:
        print(f"Failed to initialize engine: {e}")
        return
    
    result = test_network_pgmpy(args.network, engine, args.sample_size)
    
    if result:
        # 保存结果
        results_file = str(path_config.results_dir / f"{args.network}_results.json")
        with open(results_file, 'w', encoding='utf-8') as f:
            json.dump(result, f, indent=2, ensure_ascii=False)
        print(f"\nResults saved to: {results_file}")

if __name__ == "__main__":
    main()
