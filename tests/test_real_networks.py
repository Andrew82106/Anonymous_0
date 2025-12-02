
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import bnlearn as bn
import pandas as pd
import numpy as np
import json
from utils_set.causal_reasoning_engine import CausalReasoningEngine
from utils_set.utils import ConfigLoader, path_config

# 传统因果发现算法
try:
    from pgmpy.estimators import PC, HillClimbSearch, BicScore
    from pgmpy.base import DAG as PgmpyDAG
    PGMPY_AVAILABLE = True
except ImportError:
    PGMPY_AVAILABLE = False
    print("Warning: pgmpy not available. Traditional baselines will be skipped.")

# 设置结果保存路径（使用路径配置）
RESULTS_FILE = str(path_config.real_network_results_file)

def compute_shd(true_adjmat, pred_adjmat):
    """
    计算结构汉明距离 (Structural Hamming Distance)
    
    Args:
        true_adjmat: 真实邻接矩阵 (pandas DataFrame or numpy array)
        pred_adjmat: 预测邻接矩阵 (pandas DataFrame or numpy array)
    
    Returns:
        int: SHD 值（差异边的数量）
    """
    if isinstance(true_adjmat, pd.DataFrame):
        true_adjmat = true_adjmat.values
    if isinstance(pred_adjmat, pd.DataFrame):
        pred_adjmat = pred_adjmat.values
    
    return int(np.sum(np.abs(true_adjmat - pred_adjmat)))

def run_pc_algorithm(df, alpha=0.05):
    """
    运行 PC 算法（传统 Baseline）
    
    Args:
        df: 数据集
        alpha: 显著性水平
    
    Returns:
        邻接矩阵 (numpy array)
    """
    if not PGMPY_AVAILABLE:
        return None
    
    try:
        pc = PC(data=df)
        model = pc.estimate(significance_level=alpha)
        
        # 转换为邻接矩阵
        nodes = sorted(df.columns)
        n = len(nodes)
        adjmat = np.zeros((n, n))
        
        for edge in model.edges():
            source_idx = nodes.index(edge[0])
            target_idx = nodes.index(edge[1])
            adjmat[source_idx, target_idx] = 1
        
        return adjmat
    except Exception as e:
        print(f"PC algorithm failed: {e}")
        return None

def run_hillclimb_algorithm(df):
    """
    运行 HillClimb 算法（传统 Baseline）
    
    Args:
        df: 数据集
    
    Returns:
        邻接矩阵 (numpy array)
    """
    if not PGMPY_AVAILABLE:
        return None
    
    try:
        hc = HillClimbSearch(data=df)
        model = hc.estimate(scoring_method=BicScore(data=df))
        
        # 转换为邻接矩阵
        nodes = sorted(df.columns)
        n = len(nodes)
        adjmat = np.zeros((n, n))
        
        for edge in model.edges():
            source_idx = nodes.index(edge[0])
            target_idx = nodes.index(edge[1])
            adjmat[source_idx, target_idx] = 1
        
        return adjmat
    except Exception as e:
        print(f"HillClimb algorithm failed: {e}")
        return None

def run_random_baseline(n_nodes, n_edges):
    """
    生成随机图（Random Baseline）
    
    Args:
        n_nodes: 节点数量
        n_edges: 边数量
    
    Returns:
        邻接矩阵 (numpy array)
    """
    adjmat = np.zeros((n_nodes, n_nodes))
    
    # 随机生成 n_edges 条边（避免自环）
    edges_added = 0
    attempts = 0
    max_attempts = n_edges * 10
    
    while edges_added < n_edges and attempts < max_attempts:
        i = np.random.randint(0, n_nodes)
        j = np.random.randint(0, n_nodes)
        
        if i != j and adjmat[i, j] == 0:
            adjmat[i, j] = 1
            edges_added += 1
        
        attempts += 1
    
    return adjmat

def test_network(network_name, engine, sample_size=1000, test_all_edges=True):
    """
    测试单个贝叶斯网络（完整对比版）
    
    Args:
        network_name: 网络名称
        engine: 因果推理引擎
        sample_size: 采样数量
        test_all_edges: 是否测试所有边（True=完整SHD，False=采样边计算准确率）
    """
    print(f"\n{'='*60}")
    print(f"Testing Network: {network_name}")
    print(f"{'='*60}")
    
    # 1. 加载网络和采样数据
    try:
        dag = bn.import_DAG(network_name)
        df = bn.sampling(dag, n=sample_size, verbose=0)
    except Exception as e:
        print(f"Error loading network {network_name}: {e}")
        return None

    print(f"Data: {df.shape[0]} samples, {df.shape[1]} variables")
    print(f"Variables: {list(df.columns)}")
    
    # 2. 提取真实因果图 (Ground Truth)
    true_adjmat = dag['adjmat']
    nodes = list(true_adjmat.index)
    n_nodes = len(nodes)
    
    edges = []
    for source in true_adjmat.index:
        for target in true_adjmat.columns:
            if true_adjmat.loc[source, target] == 1:
                edges.append((source, target))
    
    print(f"Ground Truth: {len(edges)} edges in DAG")
    
    # ============================================================
    # PART 1: LLM-based Blind Causal Discovery
    # ============================================================
    print(f"\n[1/4] Running LLM-based Blind Causal Discovery...")
    
    # 初始化预测邻接矩阵
    llm_adjmat = pd.DataFrame(
        np.zeros((n_nodes, n_nodes)),
        index=nodes,
        columns=nodes
    )
    
    pairwise_results = []
    correct_count = 0
    
    # 逐对测试（全图）
    for source, target in edges:
        X = df[source].values
        Y = df[target].values
        
        try:
            analysis = engine.analyze_pair(X, Y)
            result = engine.infer_causality(analysis['narrative'])
            
            prediction = (
                result.get('direction') or 
                result.get('causal_direction') or 
                result.get('causal_direction_judgment') or
                'Unclear'
            )
            
            # 根据预测更新邻接矩阵
            if prediction == "A->B":
                llm_adjmat.loc[source, target] = 1
                is_correct = True
                correct_count += 1
            elif prediction == "B->A":
                llm_adjmat.loc[target, source] = 1
                is_correct = False
            else:
                # Unclear: 随机猜测
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
    llm_shd = compute_shd(true_adjmat, llm_adjmat)
    
    # ============================================================
    # PART 2: Traditional Baselines
    # ============================================================
    print(f"\n[2/4] Running PC Algorithm (Traditional Baseline)...")
    pc_adjmat = run_pc_algorithm(df)
    if pc_adjmat is not None:
        pc_adjmat_df = pd.DataFrame(pc_adjmat, index=nodes, columns=nodes)
        pc_shd = compute_shd(true_adjmat, pc_adjmat_df)
        print(f"  PC Algorithm SHD: {pc_shd}")
    else:
        pc_shd = None
        print("  PC Algorithm: FAILED or UNAVAILABLE")
    
    print(f"\n[3/4] Running HillClimb Algorithm (Traditional Baseline)...")
    hc_adjmat = run_hillclimb_algorithm(df)
    if hc_adjmat is not None:
        hc_adjmat_df = pd.DataFrame(hc_adjmat, index=nodes, columns=nodes)
        hc_shd = compute_shd(true_adjmat, hc_adjmat_df)
        print(f"  HillClimb SHD: {hc_shd}")
    else:
        hc_shd = None
        print("  HillClimb: FAILED or UNAVAILABLE")
    
    print(f"\n[4/4] Running Random Baseline...")
    random_adjmat = run_random_baseline(n_nodes, len(edges))
    random_adjmat_df = pd.DataFrame(random_adjmat, index=nodes, columns=nodes)
    random_shd = compute_shd(true_adjmat, random_adjmat_df)
    print(f"  Random SHD: {random_shd}")
    
    # ============================================================
    # PART 3: Comparison Report
    # ============================================================
    print(f"\n{'='*60}")
    print(f"📊 COMPARISON REPORT - {network_name}")
    print(f"{'='*60}")
    print(f"Ground Truth Edges: {len(edges)}")
    print(f"\n{'Method':<25} {'SHD':<10} {'Accuracy':<15}")
    print(f"{'-'*60}")
    print(f"{'LLM (Blind)':<25} {llm_shd:<10} {llm_accuracy:.1%}")
    if pc_shd is not None:
        print(f"{'PC Algorithm':<25} {pc_shd:<10} {'N/A'}")
    if hc_shd is not None:
        print(f"{'HillClimb':<25} {hc_shd:<10} {'N/A'}")
    print(f"{'Random Guess':<25} {random_shd:<10} {'~50%'}")
    print(f"{'='*60}")
    
    if pc_shd is not None:
        improvement = pc_shd - llm_shd
        print(f"\n💡 LLM vs PC: {improvement:+d} (Lower is better)")
    
    return {
        'network': network_name,
        'n_nodes': n_nodes,
        'n_edges': len(edges),
        'llm': {
            'shd': llm_shd,
            'accuracy': llm_accuracy,
            'pairwise_details': pairwise_results
        },
        'pc': {'shd': pc_shd} if pc_shd is not None else None,
        'hillclimb': {'shd': hc_shd} if hc_shd is not None else None,
        'random': {'shd': random_shd}
    }

def main():
    # 初始化推理引擎
    try:
        engine = CausalReasoningEngine()
    except Exception as e:
        print(f"Failed to initialize engine: {e}")
        return

    # 要测试的网络列表（按复杂度排序）
    # asia: 8 nodes, 8 edges (中等复杂度，医疗诊断)
    # sprinkler: 4 nodes, 4 edges (简单，经典例子)
    # alarm: 37 nodes, 46 edges (大型网络，医疗监控)
    # child: 20 nodes, 25 edges (中型网络，儿科诊断)
    # sachs: 11 nodes, 17 edges (真实生物学数据)
    networks_to_test = ['asia', 'sprinkler', 'alarm', 'child', 'sachs']
    
    all_results = []
    
    for net in networks_to_test:
        net_result = test_network(net, engine, sample_size=1000, test_all_edges=True)
        if net_result:
            all_results.append(net_result)
    
    # ============================================================
    # 生成最终总结报告
    # ============================================================
    print(f"\n\n{'#'*80}")
    print(f"# FINAL SUMMARY - Blind Causal Discovery Benchmark")
    print(f"{'#'*80}")
    
    # 汇总表格
    print(f"\n{'Network':<15} {'Nodes':<8} {'Edges':<8} {'LLM-SHD':<10} {'PC-SHD':<10} {'HC-SHD':<10} {'Random':<10} {'Acc':<10}")
    print(f"{'-'*95}")
    
    for result in all_results:
        net_name = result['network']
        n_nodes = result['n_nodes']
        n_edges = result['n_edges']
        llm_shd = result['llm']['shd']
        llm_acc = result['llm']['accuracy']
        
        pc_shd = result['pc']['shd'] if result['pc'] else 'N/A'
        hc_shd = result['hillclimb']['shd'] if result['hillclimb'] else 'N/A'
        random_shd = result['random']['shd']
        
        print(f"{net_name:<15} {n_nodes:<8} {n_edges:<8} {llm_shd:<10} {pc_shd!s:<10} {hc_shd!s:<10} {random_shd:<10} {llm_acc:.1%}")
    
    print(f"{'-'*95}")
    
    # 详细结果
    print(f"\n{'='*80}")
    print(f"DETAILED RESULTS")
    print(f"{'='*80}")
    
    for result in all_results:
        net_name = result['network']
        llm_shd = result['llm']['shd']
        llm_acc = result['llm']['accuracy']
        
        pc_shd = result['pc']['shd'] if result['pc'] else 'N/A'
        hc_shd = result['hillclimb']['shd'] if result['hillclimb'] else 'N/A'
        random_shd = result['random']['shd']
        
        print(f"\n## {net_name.upper()} Network")
        print(f"   LLM (Blind):  SHD={llm_shd}, Acc={llm_acc:.1%}")
        print(f"   PC:           SHD={pc_shd}")
        print(f"   HillClimb:    SHD={hc_shd}")
        print(f"   Random:       SHD={random_shd}")
        
        if pc_shd != 'N/A':
            improvement = pc_shd - llm_shd
            print(f"   💡 LLM vs PC: {improvement:+d} ({'Better' if improvement > 0 else 'Worse'})")
    
    print(f"\n{'='*80}")
    print(f"📝 KEY INSIGHTS:")
    print(f"   - SHD (Structural Hamming Distance): Lower is better")
    print(f"   - LLM (Blind) = Our method WITHOUT variable name information")
    print(f"   - PC/HillClimb = Traditional statistical methods")
    print(f"   - Random = Random graph baseline")
    print(f"   - Advantage: Works in privacy-preserving scenarios (no semantics)")
    print(f"{'='*80}\n")
            
    # 保存最终结果
    with open(RESULTS_FILE, 'w', encoding='utf-8') as f:
        json.dump(all_results, f, indent=2, ensure_ascii=False)
        
    print(f"Results saved to: {RESULTS_FILE}")

if __name__ == "__main__":
    main()
