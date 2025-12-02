
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import bnlearn as bn
import pandas as pd
import numpy as np
import json
from utils_set.causal_reasoning_engine import CausalReasoningEngine
from utils_set.utils import ConfigLoader

# 设置结果保存路径
RESULTS_FILE = '../results/real_network_results.json'

def test_network(network_name, engine, sample_size=1000, max_pairs=10):
    """
    测试单个贝叶斯网络
    """
    print(f"\n{'='*20} Testing Network: {network_name} {'='*20}")
    
    # 1. 加载网络和采样数据
    try:
        # bnlearn 下载有时不稳定，如果本地没有可能会报错
        dag = bn.import_DAG(network_name)
        df = bn.sampling(dag, n=sample_size, verbose=0)
    except Exception as e:
        print(f"Error loading network {network_name}: {e}")
        return None

    print(f"Data generated: {df.shape[0]} samples, {df.shape[1]} variables")
    print(f"Variables: {list(df.columns)}")
    
    # 2. 提取真实因果边 (Ground Truth)
    # bnlearn 的 adjmat 是 pandas DataFrame，Row -> Col 表示 Row causes Col
    adjmat = dag['adjmat']
    edges = []
    for source in adjmat.index:
        for target in adjmat.columns:
            if adjmat.loc[source, target] == 1:
                edges.append((source, target))
    
    print(f"Total edges in graph: {len(edges)}")
    
    # 3. 选择要测试的边
    # 为了节省时间和 Token，我们可以限制测试数量
    if len(edges) > max_pairs:
        import random
        test_edges = random.sample(edges, max_pairs)
    else:
        test_edges = edges
        
    print(f"Selected {len(test_edges)} edges for testing...")
    
    results = []
    correct_count = 0
    
    # 4. 逐个边进行测试
    for source, target in test_edges:
        print(f"\nEvaluating pair: {source} -> {target}")
        
        # 获取数据
        X = df[source].values
        Y = df[target].values
        
        # 运行推理
        # 先生成统计叙事，再进行 LLM 推理
        try:
            # 1. 分析数据生成叙事
            analysis = engine.analyze_pair(X, Y)
            narrative = analysis['narrative']
            
            # 2. 调用 LLM 推理
            result = engine.infer_causality(narrative)
            
            # 记录结果
            # 引擎返回的结果中，prediction 如果是 "A->B" 则正确
            # 兼容不同的字段名
            prediction = (
                result.get('direction') or 
                result.get('causal_direction') or 
                result.get('causal_direction_judgment') or
                result.get('judgment') or
                'Unclear'
            )
            
            # 简单的准确率统计
            is_correct = (prediction == "A->B")
            if is_correct:
                correct_count += 1
                print(f"✅ Correct! Prediction: {prediction}")
            else:
                print(f"❌ Incorrect. Prediction: {prediction}, Expected: A->B")
                
            # 保存详细信息以便分析
            results.append({
                'network': network_name,
                'pair': f"{source}->{target}",
                'ground_truth': "A->B",
                'prediction': prediction,
                'confidence': result.get('confidence', 'unknown'),
                'narrative': narrative,
                'reasoning': result.get('reasoning_chain', ''),
                'is_correct': is_correct,
                'full_response': result
            })
            
        except Exception as e:
            print(f"Error inferring pair {source}->{target}: {e}")
            
    accuracy = correct_count / len(test_edges) if test_edges else 0
    print(f"\nNetwork {network_name} Accuracy: {accuracy:.2%} ({correct_count}/{len(test_edges)})")
    
    return {
        'network': network_name,
        'accuracy': accuracy,
        'details': results
    }

def main():
    # 初始化推理引擎
    try:
        engine = CausalReasoningEngine()
    except Exception as e:
        print(f"Failed to initialize engine: {e}")
        return

    # 要测试的网络列表
    # sprinkler: 简单，离散
    # asia: 中等，离散，医疗诊断
    # alarm: 较大，通常用于基准测试
    networks_to_test = ['sprinkler', 'asia'] 
    
    all_results = []
    
    for net in networks_to_test:
        net_result = test_network(net, engine, max_pairs=5) # 限制每组5个以快速测试
        if net_result:
            all_results.append(net_result)
            
    # 保存最终结果
    with open(RESULTS_FILE, 'w', encoding='utf-8') as f:
        json.dump(all_results, f, indent=2, ensure_ascii=False)
        
    print(f"\nTesting complete. Results saved to {RESULTS_FILE}")

if __name__ == "__main__":
    main()
