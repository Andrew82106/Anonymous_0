"""
ACR-Hybrid-i 方法 - 侦探团 + 全部 LLM 定向（激进版本）

Pipeline:
1. 侦探团投票：PC + HillClimb + LiNGAM 三方投票
2. 对所有骨架边都用 LLM 重新判断方向（不信任算法共识）
3. 策略：Aggressive (>=1票也保留)
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))

import numpy as np
import pandas as pd
from tqdm import tqdm

from utils_set.causal_reasoning_engine import CausalReasoningEngine
from .metrics import compute_shd
from .skeleton_ensemble import SkeletonEnsemble, EdgeType, Strategy


class ACRHybridI:
    """
    ACR-Hybrid-i (Intensive): 侦探团 + 全部 LLM 定向
    
    与 ACR-Hybrid 的区别：
    - ACR-Hybrid: High Consensus 用共识，其他用 LLM
    - ACR-Hybrid-i: 所有边都用 LLM 重新定向（更激进）
    """
    
    def __init__(self, model_name: str = "deepseek-v3-lanyun"):
        self.engine = CausalReasoningEngine(model_name=model_name)
        self.model_name = model_name
        # 激进策略：1 票也保留
        self.skeleton = SkeletonEnsemble(strategy=Strategy.AGGRESSIVE)
    
    def run(self, df, true_adjmat, nodes, edges, network_name="network", verbose=True):
        """运行 ACR-Hybrid-i"""
        n_nodes = len(nodes)
        
        if verbose:
            print(f"  Ground truth: {len(edges)} edges")
        
        # Step 1: 侦探团投票
        if verbose:
            print(f"  Step 1: 侦探团投票 (PC + HillClimb + LiNGAM)...")
        
        edge_votes, all_edges = self.skeleton.discover(df, verbose=verbose)
        
        if verbose:
            print(f"    总骨架边: {len(all_edges)}")
        
        # Step 2: 初始化预测邻接矩阵
        pred_adjmat = pd.DataFrame(
            np.zeros((n_nodes, n_nodes)),
            index=nodes,
            columns=nodes
        )
        
        # Step 3: 对所有边用 LLM 定向
        if verbose:
            print(f"  Step 2: LLM 定向所有 {len(all_edges)} 条边...")
        
        acr_results = []
        acr_correct = 0
        acr_total = 0
        unclear_count = 0
        
        iterator = tqdm(list(all_edges), desc=f"  ACR-i ({network_name})") if verbose else list(all_edges)
        
        for edge in iterator:
            u, v = edge
            if u not in df.columns or v not in df.columns:
                continue
            
            X = df[u].values
            Y = df[v].values
            
            edge_info = edge_votes.get(edge, {})
            edge_type = edge_info.get('edge_type', EdgeType.UNDIRECTED)
            
            try:
                analysis = self.engine.analyze_pair(X, Y)
                
                # 在 narrative 中加入侦探团信息
                narrative = analysis['narrative']
                methods = edge_info.get('methods', [])
                directions = edge_info.get('directions', {})
                
                # 构建侦探团参考信息
                detective_info = []
                for method in methods:
                    direction = directions.get(method)
                    if direction:
                        detective_info.append(f"{method}: {direction[0]}->{direction[1]}")
                    else:
                        detective_info.append(f"{method}: 无方向")
                
                if detective_info:
                    narrative += f"\n\n[侦探团参考] {', '.join(detective_info)}"
                
                result = self.engine.infer_causality(narrative)
                
                prediction = (
                    result.get('direction') or 
                    result.get('causal_direction') or 
                    result.get('causal_direction_judgment') or
                    'Unclear'
                )
                
                if prediction == "A->B":
                    pred_adjmat.loc[u, v] = 1
                elif prediction == "B->A":
                    pred_adjmat.loc[v, u] = 1
                else:
                    unclear_count += 1
                
                # 检查正确性
                true_uv = true_adjmat.loc[u, v] if u in true_adjmat.index and v in true_adjmat.columns else 0
                true_vu = true_adjmat.loc[v, u] if v in true_adjmat.index and u in true_adjmat.columns else 0
                
                if true_uv == 1:
                    is_correct = (prediction == "A->B")
                elif true_vu == 1:
                    is_correct = (prediction == "B->A")
                else:
                    is_correct = False
                
                if is_correct:
                    acr_correct += 1
                acr_total += 1
                
                acr_results.append({
                    'edge': f"{u}-{v}",
                    'edge_type': edge_type.value,
                    'votes': edge_info.get('votes', 0),
                    'methods': methods,
                    'prediction': prediction,
                    'is_correct': is_correct,
                    'true_direction': 'u->v' if true_uv == 1 else ('v->u' if true_vu == 1 else 'none')
                })
                
            except Exception as e:
                if verbose:
                    print(f"    Error on {u}-{v}: {e}")
        
        # Step 4: 计算指标
        shd = compute_shd(true_adjmat, pred_adjmat)
        acr_accuracy = acr_correct / acr_total if acr_total > 0 else 0
        
        if verbose:
            print(f"    LLM 准确率: {acr_accuracy:.1%} ({acr_correct}/{acr_total})")
            print(f"    Final SHD: {shd}")
        
        return {
            'adjmat': pred_adjmat,
            'shd': shd,
            'accuracy': acr_accuracy,
            'correct_count': acr_correct,
            'total_edges': len(edges),
            'skeleton_edges': len(all_edges),
            'acr_oriented': acr_total,
            'unclear_count': unclear_count,
            'unclear_ratio': unclear_count / acr_total if acr_total > 0 else 0,
            'pairwise_results': acr_results
        }
