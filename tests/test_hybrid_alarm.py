import sys
import os
import time
import json
import numpy as np
import pandas as pd
import bnlearn as bn
import networkx as nx
from pgmpy.estimators import PC
from pgmpy.base import DAG

# 添加项目根目录到路径
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from utils_set.causal_reasoning_engine import CausalReasoningEngine
from utils_set.utils import ConfigLoader, path_config

# 结果保存路径
RESULTS_FILE = os.path.join(os.path.dirname(__file__), '../results/alarm_hybrid_results.json')

def compute_shd(true_adjmat, pred_adjmat):
    """计算 SHD"""
    if isinstance(true_adjmat, pd.DataFrame):
        true_adjmat = true_adjmat.values
    if isinstance(pred_adjmat, pd.DataFrame):
        pred_adjmat = pred_adjmat.values
    return int(np.sum(np.abs(true_adjmat - pred_adjmat)))

def compute_metrics(true_adjmat, pred_adjmat):
    """
    计算更详细的评估指标 (Precision, Recall, F1)
    区分骨架 (Skeleton) 和定向 (Orientation)
    """
    if isinstance(true_adjmat, pd.DataFrame):
        true_adjmat = true_adjmat.values
    if isinstance(pred_adjmat, pd.DataFrame):
        pred_adjmat = pred_adjmat.values
        
    # 1. 骨架评估 (忽略方向)
    true_skeleton = (true_adjmat + true_adjmat.T) > 0
    pred_skeleton = (pred_adjmat + pred_adjmat.T) > 0
    
    # 由于是对称矩阵，只看上三角
    n = true_adjmat.shape[0]
    tp_skeleton = 0
    fp_skeleton = 0
    fn_skeleton = 0
    
    for i in range(n):
        for j in range(i+1, n):
            t = true_skeleton[i, j]
            p = pred_skeleton[i, j]
            if t and p: tp_skeleton += 1
            if not t and p: fp_skeleton += 1
            if t and not p: fn_skeleton += 1
            
    sk_prec = tp_skeleton / (tp_skeleton + fp_skeleton) if (tp_skeleton + fp_skeleton) > 0 else 0
    sk_rec = tp_skeleton / (tp_skeleton + fn_skeleton) if (tp_skeleton + fn_skeleton) > 0 else 0
    sk_f1 = 2 * sk_prec * sk_rec / (sk_prec + sk_rec) if (sk_prec + sk_rec) > 0 else 0
    
    # 2. 定向评估 (考虑方向)
    # 只在骨架正确的前提下评估定向 (或者评估所有预测边)
    # 这里采用标准定义：预测的 A->B 在真图中也是 A->B 才算 TP
    
    tp_orient = 0
    fp_orient = 0
    fn_orient = 0
    
    # 遍历所有可能的边
    for i in range(n):
        for j in range(n):
            if i == j: continue
            t = (true_adjmat[i, j] == 1)
            p = (pred_adjmat[i, j] == 1)
            
            if t and p: tp_orient += 1
            if not t and p: fp_orient += 1
            if t and not p: fn_orient += 1
            
    or_prec = tp_orient / (tp_orient + fp_orient) if (tp_orient + fp_orient) > 0 else 0
    or_rec = tp_orient / (tp_orient + fn_orient) if (tp_orient + fn_orient) > 0 else 0
    or_f1 = 2 * or_prec * or_rec / (or_prec + or_rec) if (or_prec + or_rec) > 0 else 0
    
    return {
        'shd': compute_shd(true_adjmat, pred_adjmat),
        'skeleton': {
            'precision': sk_prec,
            'recall': sk_rec,
            'f1': sk_f1
        },
        'orientation': {
            'precision': or_prec,
            'recall': or_rec,
            'f1': or_f1
        }
    }

def adjmat_from_edges(edges, nodes):
    """从边列表构建邻接矩阵"""
    n = len(nodes)
    adjmat = np.zeros((n, n))
    for u, v in edges:
        if u in nodes and v in nodes:
            i, j = nodes.index(u), nodes.index(v)
            adjmat[i, j] = 1
    return pd.DataFrame(adjmat, index=nodes, columns=nodes)

def run_campaign_a_orientation_challenge(engine, df, true_dag, nodes):
    """
    战役 A: 定向挑战赛
    给定完美骨架，测试 ACR 的定向准确率
    """
    print(f"\n{'='*60}")
    print(f"⚔️  CAMPAIGN A: THE ORIENTATION CHALLENGE")
    print(f"{'='*60}")
    print("Goal: Given Ground Truth Skeleton, predict direction (A->B or B->A)")
    
    # 1. 获取真实骨架 (Ground Truth Skeleton)
    true_adjmat = true_dag['adjmat']
    true_edges = []
    undirected_skeleton = []
    
    # 提取真实边
    for u in nodes:
        for v in nodes:
            if true_adjmat.loc[u, v] == 1:
                true_edges.append((u, v))
                undirected_skeleton.append(sorted((u, v))) # 存储无向对，避免重复
    
    # 去重得到骨架边
    skeleton_pairs = []
    seen = set()
    for pair in undirected_skeleton:
        pair_tuple = tuple(pair)
        if pair_tuple not in seen:
            skeleton_pairs.append(pair_tuple)
            seen.add(pair_tuple)
            
    print(f"Ground Truth Edges: {len(true_edges)}")
    print(f"Skeleton Pairs to Test: {len(skeleton_pairs)}")
    
    correct_count = 0
    results = []
    
    # 构建 ACR 预测的邻接矩阵（基于完美骨架）
    pred_adjmat = pd.DataFrame(np.zeros((len(nodes), len(nodes))), index=nodes, columns=nodes)
    
    # 2. 逐边测试
    for idx, (node_a, node_b) in enumerate(skeleton_pairs):
        # 确定真实方向用于验证
        if true_adjmat.loc[node_a, node_b] == 1:
            true_direction = "A->B"
            source, target = node_a, node_b
        else:
            true_direction = "B->A" # 因为是骨架边，必有一向
            source, target = node_b, node_a
            
        print(f"\n[{idx+1}/{len(skeleton_pairs)}] Testing pair: {node_a} -- {node_b}")
        
        try:
            # ACR 推理
            X = df[node_a].values
            Y = df[node_b].values
            
            analysis = engine.analyze_pair(X, Y)
            llm_result = engine.infer_causality(analysis['narrative'])
            
            prediction = (
                llm_result.get('direction') or 
                llm_result.get('causal_direction') or 
                'Unclear'
            )
            
            # 判定结果
            is_correct = False
            final_pred_dir = None
            
            if prediction == "A->B":
                pred_adjmat.loc[node_a, node_b] = 1
                if true_direction == "A->B": is_correct = True
                final_pred_dir = f"{node_a}->{node_b}"
                
            elif prediction == "B->A":
                pred_adjmat.loc[node_b, node_a] = 1
                if true_direction == "B->A": is_correct = True
                final_pred_dir = f"{node_b}->{node_a}"
            
            else:
                # Unclear: 随机选一个方向，或者算半对？严格来说算错
                # 或者是作为无向边处理？SHD会惩罚
                print(f"  ⚠️ Unclear result. No edge added to directed graph.")
                final_pred_dir = "Unclear"
            
            if is_correct:
                correct_count += 1
                print(f"  ✅ Correct! ({final_pred_dir})")
            else:
                print(f"  ❌ Incorrect. Pred: {prediction}, True: {true_direction}")
                
            results.append({
                'pair': f"{node_a}-{node_b}",
                'true_dir': f"{source}->{target}",
                'pred_raw': prediction,
                'is_correct': is_correct
            })
            
        except Exception as e:
            print(f"  ⚠️ Error: {e}")
            
    accuracy = correct_count / len(skeleton_pairs)
    
    # 计算在给定完美骨架下的 SHD
    # 注意：这里只看定向错误的惩罚
    acr_shd = compute_shd(true_adjmat, pred_adjmat)
    
    print(f"\n🏆 Campaign A Result:")
    print(f"   Orientation Accuracy: {accuracy:.2%} ({correct_count}/{len(skeleton_pairs)})")
    print(f"   SHD (Fixed Skeleton): {acr_shd}")
    
    return {
        'accuracy': accuracy,
        'shd': acr_shd,
        'details': results
    }

def run_campaign_b_hybrid_pipeline(engine, df, true_dag, nodes, acr_results_map):
    """
    战役 B: 混合流水线 (Hybrid Pipeline)
    1. PC 发现骨架
    2. ACR 重定方向 (利用 Campaign A 的结果缓存)
    """
    print(f"\n{'='*60}")
    print(f"🛡️  CAMPAIGN B: THE HYBRID PIPELINE")
    print(f"{'='*60}")
    print("Goal: Use PC for Skeleton + ACR for Orientation")
    
    true_adjmat = true_dag['adjmat']
    
    # 1. 运行 PC 算法 (Baseline)
    print("\nRunning PC Algorithm (pgmpy)...")
    start_time = time.time()
    pc = PC(data=df)
    # 使用默认显著性水平 0.05
    pc_model = pc.estimate(significance_level=0.05, return_type='dag') 
    print(f"PC finished in {time.time() - start_time:.2f}s")
    
    # 构建 PC 的邻接矩阵
    pc_edges = list(pc_model.edges())
    pc_adjmat = adjmat_from_edges(pc_edges, nodes)
    pc_metrics = compute_metrics(true_adjmat, pc_adjmat)
    
    print(f"PC Algorithm Result:")
    print(f"   Edges Found: {len(pc_edges)}")
    print(f"   SHD: {pc_metrics['shd']}")
    print(f"   Skeleton F1: {pc_metrics['skeleton']['f1']:.2f} (P={pc_metrics['skeleton']['precision']:.2f}, R={pc_metrics['skeleton']['recall']:.2f})")
    print(f"   Orient F1:   {pc_metrics['orientation']['f1']:.2f} (P={pc_metrics['orientation']['precision']:.2f}, R={pc_metrics['orientation']['recall']:.2f})")
    
    # 2. 混合模式：保留 PC 骨架，用 ACR 定向
    # 策略：遍历 PC 发现的每一条边。
    # 如果这条边在 Campaign A 中测试过（即在真实骨架中），我们用 ACR 的结果。
    # 如果是 PC 发现的"错误边"（不在真实骨架中），我们需要现场测一下，或者直接保留 PC 的方向（因为是假阳性）。
    # 为了严谨，我们假设 Hybrid 模式是：PC 给定骨架 -> ACR 对骨架内所有边定向。
    
    print("\nRefining PC Skeleton with ACR...")
    hybrid_adjmat = pd.DataFrame(np.zeros((len(nodes), len(nodes))), index=nodes, columns=nodes)
    
    hybrid_edges_count = 0
    
    # PC 发现的骨架 (无向)
    pc_skeleton_pairs = set()
    for u, v in pc_edges:
        pc_skeleton_pairs.add(tuple(sorted((u, v))))
        
    print(f"PC Skeleton has {len(pc_skeleton_pairs)} unique pairs.")
    
    # 对 PC 骨架中的每一对边进行定向
    for u, v in pc_skeleton_pairs:
        # 检查我们是否在 Campaign A 中已经有结果（缓存命中）
        cache_key = tuple(sorted((u, v)))
        
        # 在 Campaign A 的结果中查找
        # acr_results_map 的 key 也是 sorted tuple
        cached_pred = None
        
        # 我们需要将 list 形式的 results 转为 dict 方便查找
        # 但这里简单起见，如果边不在 Campaign A (即 PC 发现了不存在的边)，我们需要实时跑
        # 考虑到脚本运行时间，我们可以只处理 Campaign A 覆盖的边，
        # 对于假阳性边（False Positives），ACR 可能会说是 Independent/Unclear，或者强行定一个方向
        
        # 简化逻辑：
        # 如果边在 Campaign A 中（True Positive Skeleton），用 ACR 结果
        # 如果边不在 Campaign A 中（False Positive Skeleton），保留 PC 原结果（或者设为 Unclear?）
        # 通常 Hybrid 主要是为了修正 True Positive 骨架的方向
        
        # 查找 Campaign A 结果
        found_in_campaign_a = False
        predicted_dir = None
        
        # 实际上我们需要更智能的查找
        # 暂且只处理我们在 Campaign A 跑过的边
        # 对于没跑过的（假阳性），我们保留 PC 的方向（因为 ACR 没测）
        
        # 这里为了演示 Hybrid 的潜力，我们假设只对 PC 找对的骨架进行修正
        # 对于 PC 找错的骨架（假阳性），Hybrid 救不回来（除非 ACR 说是独立的）
        
        # 让我们看看是否可以直接调用 engine 跑假阳性边（如果数量不多）
        # 为了避免太慢，我们先只用缓存
        
        # 在 acr_results_map 中查找
        pair_key = f"{u}-{v}"
        reverse_key = f"{v}-{u}"
        
        prediction = "Unclear"
        
        if pair_key in acr_results_map:
            prediction = acr_results_map[pair_key]
        elif reverse_key in acr_results_map:
             # 结果里存的是 raw prediction (A->B for the tested pair)
             # 需要转换
             pass 
             # 这里的逻辑比较绕，还是直接重跑或者简单处理比较好
        
        # 实际上，Campaign A 跑的是所有 Ground Truth 边。
        # 如果 PC 发现了一条 Ground Truth 边，我们就用 ACR 的结果。
        # 如果 PC 发现了一条 False Positive，我们暂时保留 PC 的方向。
        
        is_true_edge = (true_adjmat.loc[u, v] == 1) or (true_adjmat.loc[v, u] == 1)
        
        if is_true_edge:
            # 这是一个真实存在的边，ACR 在 Campaign A 里肯定跑过
            # 找到 ACR 的判断
            # 为了方便，我们在 main 函数里构建好查找表
            
            # 查找表 logic: key=sorted_tuple, value=prediction(relative to key order)
            pred = acr_results_map.get(cache_key)
            if pred:
                # pred 是相对于 cache_key (u_sorted, v_sorted) 的方向
                u_s, v_s = cache_key
                if pred == "A->B":
                    hybrid_adjmat.loc[u_s, v_s] = 1
                elif pred == "B->A":
                    hybrid_adjmat.loc[v_s, u_s] = 1
            else:
                # 异常情况，没找到缓存，保留 PC
                hybrid_adjmat.loc[u, v] = 1 # 保留 PC 方向 (u->v)
        else:
            # 这是一个假阳性边，Campaign A 没跑过
            # 保留 PC 的方向
            # 注意 pc_edges 里是定向的 u->v
            # 我们需要找到原始 pc_edges 里这对节点的方向
            if (u, v) in pc_edges:
                hybrid_adjmat.loc[u, v] = 1
            elif (v, u) in pc_edges:
                hybrid_adjmat.loc[v, u] = 1
                
    hybrid_metrics = compute_metrics(true_adjmat, hybrid_adjmat)
    
    print(f"\nHybrid Pipeline Result:")
    print(f"   Combined Edges: {int(hybrid_adjmat.sum().sum())}")
    print(f"   Hybrid SHD: {hybrid_metrics['shd']}")
    print(f"   Improvement vs PC: {pc_metrics['shd'] - hybrid_metrics['shd']:+d}")
    print(f"   Skeleton F1: {hybrid_metrics['skeleton']['f1']:.2f}")
    print(f"   Orient F1:   {hybrid_metrics['orientation']['f1']:.2f} (vs PC: {pc_metrics['orientation']['f1']:.2f})")

    return {
        'pc_metrics': pc_metrics,
        'hybrid_metrics': hybrid_metrics
    }

def main():
    network_name = "alarm"
    sample_size = 1000
    
    try:
        engine = CausalReasoningEngine()
    except Exception as e:
        print(f"Failed to initialize engine: {e}")
        return

    print(f"Loading {network_name} network...")
    dag = bn.import_DAG(network_name)
    df = bn.sampling(dag, n=sample_size, verbose=0)
    nodes = list(df.columns)
    
    # Campaign A
    camp_a_res = run_campaign_a_orientation_challenge(engine, df, dag, nodes)
    
    # 构建结果缓存供 Campaign B 使用
    # Map: (u, v) sorted tuple -> prediction (A->B or B->A relative to u,v)
    acr_results_map = {}
    for detail in camp_a_res['details']:
        # detail['pair'] format "u-v"
        u, v = detail['pair'].split('-')
        # raw prediction "A->B" means u->v
        
        key = tuple(sorted((u, v)))
        # 归一化预测方向
        pred = detail['pred_raw'] # A->B or B->A
        
        # 确保预测是相对于 key 的顺序
        if key == (u, v):
            val = pred
        else:
            # key 是 (v, u)，而 pred 是基于 (u, v)
            if pred == "A->B": val = "B->A"
            elif pred == "B->A": val = "A->B"
            else: val = "Unclear"
            
        acr_results_map[key] = val
        
    # Campaign B
    camp_b_res = run_campaign_b_hybrid_pipeline(engine, df, dag, nodes, acr_results_map)
    
    # 汇总
    print(f"\n\n{'#'*60}")
    print(f"FINAL REPORT - ALARM NETWORK")
    print(f"{'#'*60}")
    print(f"Orientation Accuracy (ACR): {camp_a_res['accuracy']:.1%}")
    print(f"{'Metric':<15} {'PC Original':<15} {'Hybrid (Ours)':<15} {'Delta':<10}")
    print(f"{'-'*55}")
    print(f"{'SHD':<15} {camp_b_res['pc_metrics']['shd']:<15} {camp_b_res['hybrid_metrics']['shd']:<15} {camp_b_res['pc_metrics']['shd'] - camp_b_res['hybrid_metrics']['shd']:+d}")
    print(f"{'Orient F1':<15} {camp_b_res['pc_metrics']['orientation']['f1']:.3f}          {camp_b_res['hybrid_metrics']['orientation']['f1']:.3f}          {camp_b_res['hybrid_metrics']['orientation']['f1'] - camp_b_res['pc_metrics']['orientation']['f1']:+.3f}")
    print(f"{'Skeleton F1':<15} {camp_b_res['pc_metrics']['skeleton']['f1']:.3f}          {camp_b_res['hybrid_metrics']['skeleton']['f1']:.3f}          {camp_b_res['hybrid_metrics']['skeleton']['f1'] - camp_b_res['pc_metrics']['skeleton']['f1']:+.3f}")
    print(f"{'#'*60}")
    
    # 保存
    results = {
        'campaign_a': camp_a_res,
        'campaign_b': camp_b_res
    }
    with open(RESULTS_FILE, 'w') as f:
        json.dump(results, f, indent=2)

if __name__ == "__main__":
    main()
