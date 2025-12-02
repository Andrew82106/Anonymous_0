import sys
import os
import time
import json
import numpy as np
import pandas as pd
import bnlearn as bn
from pgmpy.estimators import PC
from pgmpy.base import DAG

# 添加项目根目录到路径
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from utils_set.causal_reasoning_engine import CausalReasoningEngine
from utils_set.utils import ConfigLoader, path_config

RESULTS_DIR = os.path.join(os.path.dirname(__file__), '../results')
os.makedirs(RESULTS_DIR, exist_ok=True)

class HybridEvaluator:
    def __init__(self):
        try:
            self.engine = CausalReasoningEngine()
            print(f"✅ Causal Engine Initialized")
        except Exception as e:
            print(f"❌ Failed to initialize engine: {e}")
            self.engine = None

    def compute_metrics(self, true_adjmat, pred_adjmat):
        """计算 SHD, Precision, Recall, F1"""
        if isinstance(true_adjmat, pd.DataFrame): true_adjmat = true_adjmat.values
        if isinstance(pred_adjmat, pd.DataFrame): pred_adjmat = pred_adjmat.values
            
        # Skeleton Metrics
        true_skeleton = (true_adjmat + true_adjmat.T) > 0
        pred_skeleton = (pred_adjmat + pred_adjmat.T) > 0
        n = true_adjmat.shape[0]
        
        tp_sk, fp_sk, fn_sk = 0, 0, 0
        for i in range(n):
            for j in range(i+1, n):
                t, p = true_skeleton[i, j], pred_skeleton[i, j]
                if t and p: tp_sk += 1
                if not t and p: fp_sk += 1
                if t and not p: fn_sk += 1
                
        sk_prec = tp_sk / (tp_sk + fp_sk) if (tp_sk + fp_sk) > 0 else 0
        sk_rec = tp_sk / (tp_sk + fn_sk) if (tp_sk + fn_sk) > 0 else 0
        sk_f1 = 2 * sk_prec * sk_rec / (sk_prec + sk_rec) if (sk_prec + sk_rec) > 0 else 0
        
        # Orientation Metrics
        tp_or, fp_or, fn_or = 0, 0, 0
        for i in range(n):
            for j in range(n):
                if i == j: continue
                t, p = (true_adjmat[i, j] == 1), (pred_adjmat[i, j] == 1)
                if t and p: tp_or += 1
                if not t and p: fp_or += 1
                if t and not p: fn_or += 1
                
        or_prec = tp_or / (tp_or + fp_or) if (tp_or + fp_or) > 0 else 0
        or_rec = tp_or / (tp_or + fn_or) if (tp_or + fn_or) > 0 else 0
        or_f1 = 2 * tp_or / (2 * tp_or + fp_or + fn_or) if (2 * tp_or + fp_or + fn_or) > 0 else 0
        
        shd = int(np.sum(np.abs(true_adjmat - pred_adjmat)))
        
        return {
            'shd': shd,
            'skeleton': {'precision': sk_prec, 'recall': sk_rec, 'f1': sk_f1},
            'orientation': {'precision': or_prec, 'recall': or_rec, 'f1': or_f1}
        }

    def adjmat_from_edges(self, edges, nodes):
        n = len(nodes)
        adjmat = np.zeros((n, n))
        for u, v in edges:
            if u in nodes and v in nodes:
                i, j = nodes.index(u), nodes.index(v)
                adjmat[i, j] = 1
        return pd.DataFrame(adjmat, index=nodes, columns=nodes)

    def run_hybrid_pipeline(self, network_name="alarm", sample_size=1000):
        print(f"\n{'='*60}")
        print(f"🚀 HYBRID PIPELINE TEST: {network_name.upper()}")
        print(f"{'='*60}")
        
        # 1. 加载数据
        print(f"Loading {network_name} network...")
        try:
            dag = bn.import_DAG(network_name)
            df = bn.sampling(dag, n=sample_size, verbose=0)
            nodes = list(df.columns)
            true_adjmat = dag['adjmat']
        except Exception as e:
            print(f"Error loading network: {e}")
            return

        # 2. 运行 PC 算法 (PDAG Mode)
        print("\n[Step 1] Running PC Algorithm (Baseline)...")
        pc = PC(data=df)
        try:
            # 尝试获取 PDAG 以识别无向边
            pc_model = pc.estimate(significance_level=0.05, return_type='pdag')
            pc_edges = list(pc_model.edges())
        except:
            print("⚠️ PDAG estimation failed, falling back to DAG.")
            pc_model = pc.estimate(significance_level=0.05, return_type='dag')
            pc_edges = list(pc_model.edges())
            
        # 识别 PC 的无向边
        # 在 pgmpy PDAG 中，无向边通常没有显式表示，但我们可以通过邻接矩阵的双向性来检查
        # 或者更简单：如果在 PDAG 中 (u, v) 和 (v, u) 都不存在，则是无连接
        # 如果 (u, v) 存在但 (v, u) 不存在，是有向
        # PDAG 的 edges() 方法通常只返回有向边和无向边的一份拷贝？
        # 让我们用一个临时矩阵来检查
        temp_adj = self.adjmat_from_edges(pc_edges, nodes)
        
        undirected_pairs = []
        directed_edges = []
        
        # 重新遍历 pc_edges 来分类
        # 注意：pgmpy 的 PDAG edges() 可能包含 (u, v) 和 (v, u) 如果是无向的
        # 我们需要去重
        processed_pairs = set()
        
        for u, v in pc_edges:
            pair_key = tuple(sorted((u, v)))
            if pair_key in processed_pairs: continue
            processed_pairs.add(pair_key)
            
            # 检查是否双向
            is_undirected = (temp_adj.loc[u, v] == 1) and (temp_adj.loc[v, u] == 1)
            # 或者在某些版本中，无向边只存一次？
            # 我们假设 PDAG 正确返回了双向边
            
            # 更鲁棒的方法：检查 PC 的 undirect_edges 属性 (如果存在)
            if hasattr(pc_model, 'undirected_edges'):
                if (u, v) in pc_model.undirected_edges or (v, u) in pc_model.undirected_edges:
                    is_undirected = True
            
            if is_undirected:
                undirected_pairs.append((u, v))
            else:
                directed_edges.append((u, v))
                
        # 如果 pgmpy 返回的是 DAG，undirected_pairs 可能为空
        # 在这种情况下，PC 已经强行定向了。
        # 我们可以选择：
        # A) 接受 PC 的定向
        # B) 对所有边运行 ACR (之前的 Full Hybrid)
        # C) 识别 PC "不确定" 的边 (如果在 DAG 模式下很难)
        
        # 假设 PDAG 返回了一些无向边
        # 即使 list 为空，我们也可以测试一下 PC 留下的双向边
        # 实际上，上面的 temp_adj 检查更通用
        
        print(f"PC found {len(pc_edges)} edges total.")
        # 重新统计
        undirected_count = 0
        final_pc_dag_edges = []
        
        hybrid_adjmat = pd.DataFrame(np.zeros((len(nodes), len(nodes))), index=nodes, columns=nodes)
        
        # 3. Conservative Hybrid 策略
        print(f"\n[Step 2] Conservative Hybrid Refinement...")
        print(f"Strategy: Trust PC's directed edges, ask ACR for undirected ones.")
        
        acr_updates = 0
        
        # 再次遍历所有边
        # 为了处理方便，我们使用 temp_adj 遍历
        processed_pairs = set()
        
        for u in nodes:
            for v in nodes:
                if u == v: continue
                if temp_adj.loc[u, v] == 1:
                    pair_key = tuple(sorted((u, v)))
                    if pair_key in processed_pairs: continue
                    processed_pairs.add(pair_key)
                    
                    is_undirected = (temp_adj.loc[v, u] == 1)
                    
                    if is_undirected:
                        # === 无向边：调用 ACR ===
                        undirected_count += 1
                        print(f"  ❓ Undirected: {u}-{v} -> Asking ACR...")
                        
                        X, Y = df[u].values, df[v].values
                        try:
                            analysis = self.engine.analyze_pair(X, Y)
                            res = self.engine.infer_causality(analysis['narrative'])
                            pred = res.get('direction') or res.get('causal_direction') or 'Unclear'
                            
                            if pred == "A->B":
                                hybrid_adjmat.loc[u, v] = 1
                                print(f"     ✅ ACR decided: {u}->{v}")
                                acr_updates += 1
                            elif pred == "B->A":
                                hybrid_adjmat.loc[v, u] = 1
                                print(f"     ✅ ACR decided: {v}->{u}")
                                acr_updates += 1
                            else:
                                print(f"     ⚠️ ACR Unclear. Random orientation.")
                                hybrid_adjmat.loc[u, v] = 1 # Fallback
                        except Exception as e:
                            print(f"     ❌ Error: {e}")
                            hybrid_adjmat.loc[u, v] = 1 # Fallback
                            
                    else:
                        # === 有向边：信任 PC ===
                        hybrid_adjmat.loc[u, v] = 1
                        final_pc_dag_edges.append((u, v))

        # 4. 计算结果
        # PC 的 DAG 形式 (用于对比)
        # 将无向边任意定向以形成 DAG (Baseline)
        pc_dag_adj = temp_adj.copy()
        # 这里的 temp_adj 包含双向边， SHD 会惩罚
        # 我们需要把双向边变成单向才能公平对比
        # 简单的 baseline 是把无向边按字母顺序定向
        for u, v in nodes:
            if pc_dag_adj.loc[u, v] == 1 and pc_dag_adj.loc[v, u] == 1:
                # 简单定向 u->v (如果 u < v)
                if u < v:
                    pc_dag_adj.loc[v, u] = 0
                else:
                    pc_dag_adj.loc[u, v] = 0
                    
        pc_metrics = self.compute_metrics(true_adjmat, pc_dag_adj)
        hybrid_metrics = self.compute_metrics(true_adjmat, hybrid_adjmat)
        
        print(f"\n{'='*60}")
        print(f"📊 RESULTS SUMMARY: {network_name.upper()}")
        print(f"{'='*60}")
        print(f"PC Edges: {len(pc_edges)} | Undirected: {undirected_count}")
        print(f"ACR Updates: {acr_updates}")
        
        print(f"\nMetrics Comparison:")
        print(f"{'Metric':<15} {'PC (Base)':<12} {'Hybrid':<12} {'Delta':<10}")
        print(f"{'-'*55}")
        print(f"{'SHD':<15} {pc_metrics['shd']:<12} {hybrid_metrics['shd']:<12} {pc_metrics['shd'] - hybrid_metrics['shd']:+d}")
        print(f"{'Orient F1':<15} {pc_metrics['orientation']['f1']:.3f}       {hybrid_metrics['orientation']['f1']:.3f}       {hybrid_metrics['orientation']['f1'] - pc_metrics['orientation']['f1']:+.3f}")
        print(f"{'Skeleton F1':<15} {pc_metrics['skeleton']['f1']:.3f}       {hybrid_metrics['skeleton']['f1']:.3f}       {hybrid_metrics['skeleton']['f1'] - pc_metrics['skeleton']['f1']:+.3f}")
        print(f"{'='*60}")
        
        # 保存
        results = {
            'network': network_name,
            'pc_metrics': pc_metrics,
            'hybrid_metrics': hybrid_metrics,
            'details': {
                'undirected_count': undirected_count,
                'acr_updates': acr_updates
            }
        }
        outfile = os.path.join(RESULTS_DIR, f"{network_name}_hybrid_final.json")
        with open(outfile, 'w') as f:
            json.dump(results, f, indent=2)
        print(f"Results saved to {outfile}")

if __name__ == "__main__":
    evaluator = HybridEvaluator()
    if evaluator.engine:
        # 默认测试 Alarm
        evaluator.run_hybrid_pipeline("alarm")
