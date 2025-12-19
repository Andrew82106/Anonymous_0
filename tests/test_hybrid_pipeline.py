import sys
import os
import time
import json
import argparse
import numpy as np
import pandas as pd
import bnlearn as bn
from pgmpy.estimators import PC, MmhcEstimator
from pgmpy.base import DAG

# 添加项目根目录到路径
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from utils_set.causal_reasoning_engine import CausalReasoningEngine
from utils_set.utils import ConfigLoader, path_config

# FCI 算法支持 (causal-learn)
try:
    from causallearn.search.ConstraintBased.FCI import fci
    from causallearn.utils.cit import fisherz, chisq
    CAUSALLEARN_AVAILABLE = True
except ImportError:
    CAUSALLEARN_AVAILABLE = False

RESULTS_DIR = os.path.join(os.path.dirname(__file__), '../results')
os.makedirs(RESULTS_DIR, exist_ok=True)

# 支持的基座算法
SUPPORTED_BASE_ALGORITHMS = ['pc', 'dual_pc', 'fci', 'mmhc']

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

    def run_base_algorithm(self, df, algorithm='pc', alpha=0.05):
        """
        运行指定的基座算法
        
        Args:
            df: 数据框
            algorithm: 算法名称 ('pc', 'dual_pc', 'fci', 'mmhc')
            alpha: 显著性水平
        
        Returns:
            edges: 边列表
            info: 算法输出信息（包含无向边/可定向边等）
        """
        if algorithm == 'pc':
            return self._run_pc(df, alpha)
        elif algorithm == 'dual_pc':
            return self._run_dual_pc(df, alpha)
        elif algorithm == 'fci':
            return self._run_fci(df, alpha)
        elif algorithm == 'mmhc':
            return self._run_mmhc(df)
        else:
            raise ValueError(f"Unsupported algorithm: {algorithm}. Supported: {SUPPORTED_BASE_ALGORITHMS}")
    
    def _run_pc(self, df, alpha=0.05):
        """运行 PC 算法"""
        pc = PC(data=df)
        try:
            model = pc.estimate(significance_level=alpha, return_type='pdag')
            edges = list(model.edges())
        except:
            model = pc.estimate(significance_level=alpha, return_type='dag')
            edges = list(model.edges())
        
        # 分类有向边和无向边
        nodes = list(df.columns)
        temp_adj = self.adjmat_from_edges(edges, nodes)
        
        directed_edges = []
        undirected_edges = []
        processed = set()
        
        for u, v in edges:
            pair = tuple(sorted((u, v)))
            if pair in processed:
                continue
            processed.add(pair)
            
            is_undirected = (temp_adj.loc[u, v] == 1) and (temp_adj.loc[v, u] == 1)
            if is_undirected:
                undirected_edges.append((u, v))
            else:
                directed_edges.append((u, v))
        
        return edges, {
            'algorithm': 'pc',
            'directed_edges': directed_edges,
            'undirected_edges': undirected_edges,
            'all_edges': edges
        }
    
    def _run_dual_pc(self, df, alpha=0.05):
        """
        运行 Dual PC 算法
        Dual PC 是 PC 的变体，适用于高斯连续数据
        使用更严格的显著性水平
        """
        # Dual PC 使用更严格的 alpha 值
        strict_alpha = alpha / 2
        pc = PC(data=df)
        try:
            model = pc.estimate(significance_level=strict_alpha, return_type='pdag')
            edges = list(model.edges())
        except:
            model = pc.estimate(significance_level=strict_alpha, return_type='dag')
            edges = list(model.edges())
        
        # 分类有向边和无向边
        nodes = list(df.columns)
        temp_adj = self.adjmat_from_edges(edges, nodes)
        
        directed_edges = []
        undirected_edges = []
        processed = set()
        
        for u, v in edges:
            pair = tuple(sorted((u, v)))
            if pair in processed:
                continue
            processed.add(pair)
            
            is_undirected = (temp_adj.loc[u, v] == 1) and (temp_adj.loc[v, u] == 1)
            if is_undirected:
                undirected_edges.append((u, v))
            else:
                directed_edges.append((u, v))
        
        return edges, {
            'algorithm': 'dual_pc',
            'directed_edges': directed_edges,
            'undirected_edges': undirected_edges,
            'all_edges': edges
        }
    
    def _run_fci(self, df, alpha=0.05):
        """
        运行 FCI 算法
        FCI 可以处理潜在混淆因子，输出 PAG
        """
        nodes = list(df.columns)
        
        if CAUSALLEARN_AVAILABLE:
            try:
                data = df.values
                # 选择条件独立性检验
                if df.dtypes.apply(lambda x: x.kind in 'iub').all():
                    cit = chisq
                else:
                    cit = fisherz
                
                g, _ = fci(data, cit, alpha)
                
                # 解析 PAG 结果
                directed_edges = []
                bidirected_edges = []
                orientable_edges = []
                
                n = len(nodes)
                adj_matrix = g.graph
                
                for i in range(n):
                    for j in range(i + 1, n):
                        if adj_matrix[i, j] != 0 or adj_matrix[j, i] != 0:
                            u, v = nodes[i], nodes[j]
                            mark_at_j = adj_matrix[i, j]
                            mark_at_i = adj_matrix[j, i]
                            
                            if mark_at_j == 1 and mark_at_i == 2:
                                directed_edges.append((u, v))
                            elif mark_at_j == 2 and mark_at_i == 1:
                                directed_edges.append((v, u))
                            elif mark_at_j == 1 and mark_at_i == 1:
                                bidirected_edges.append((u, v))
                            elif mark_at_j == -1 or mark_at_i == -1:
                                orientable_edges.append((u, v))
                            else:
                                orientable_edges.append((u, v))
                
                all_edges = directed_edges.copy()
                for u, v in orientable_edges:
                    all_edges.append((u, v) if u < v else (v, u))
                
                return all_edges, {
                    'algorithm': 'fci',
                    'directed_edges': directed_edges,
                    'undirected_edges': orientable_edges,  # 可定向边作为无向边处理
                    'bidirected_edges': bidirected_edges,
                    'orientable_edges': orientable_edges,
                    'all_edges': all_edges,
                    'is_fallback': False
                }
            except Exception as e:
                print(f"FCI (causal-learn) failed: {e}, using PC fallback")
        
        # Fallback: 使用 PC 模拟
        print("Warning: Using PC as FCI fallback")
        edges, info = self._run_pc(df, alpha)
        info['algorithm'] = 'fci'
        info['is_fallback'] = True
        info['orientable_edges'] = info['undirected_edges']
        return edges, info
    
    def _run_mmhc(self, df):
        """
        运行 MMHC 算法
        MMHC 是混合类算法，结合约束和评分搜索
        """
        try:
            mmhc = MmhcEstimator(data=df)
            model = mmhc.estimate()
            edges = list(model.edges())
            
            # MMHC 输出的是 DAG，但我们可以提取骨架
            nodes = list(df.columns)
            skeleton_pairs = set()
            for u, v in edges:
                skeleton_pairs.add(tuple(sorted((u, v))))
            
            # 将骨架视为无向边
            undirected_edges = [(u, v) for u, v in skeleton_pairs]
            
            return edges, {
                'algorithm': 'mmhc',
                'directed_edges': edges,
                'undirected_edges': undirected_edges,  # 骨架边
                'skeleton_edges': undirected_edges,
                'all_edges': edges
            }
        except Exception as e:
            print(f"MMHC failed: {e}")
            return None, None

    def run_hybrid_pipeline(self, network_name="alarm", sample_size=1000, base_algorithm='pc'):
        """
        运行混合流水线
        
        Args:
            network_name: 网络名称
            sample_size: 样本大小
            base_algorithm: 基座算法 ('pc', 'dual_pc', 'fci', 'mmhc')
        """
        print(f"\n{'='*60}")
        print(f"🚀 HYBRID PIPELINE TEST: {network_name.upper()}")
        print(f"Base Algorithm: {base_algorithm.upper()}")
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
            return None

        # 2. 运行基座算法
        print(f"\n[Step 1] Running {base_algorithm.upper()} Algorithm (Baseline)...")
        base_edges, base_info = self.run_base_algorithm(df, base_algorithm)
        
        if base_edges is None:
            print(f"  {base_algorithm.upper()} failed, aborting...")
            return None
        
        # 获取无向边（需要 ACR 定向的边）
        undirected_edges = base_info.get('undirected_edges', [])
        directed_edges = base_info.get('directed_edges', [])
        
        print(f"  {base_algorithm.upper()} found {len(base_edges)} edges total.")
        print(f"  Directed: {len(directed_edges)}, Undirected: {len(undirected_edges)}")
        
        # 构建基座算法的邻接矩阵
        temp_adj = self.adjmat_from_edges(base_edges, nodes)
        
        # 3. Conservative Hybrid 策略
        print(f"\n[Step 2] Conservative Hybrid Refinement...")
        print(f"Strategy: Trust {base_algorithm.upper()}'s directed edges, ask ACR for undirected ones.")
        
        hybrid_adjmat = pd.DataFrame(np.zeros((len(nodes), len(nodes))), index=nodes, columns=nodes)
        
        # 首先添加所有有向边
        for u, v in directed_edges:
            if u in nodes and v in nodes:
                hybrid_adjmat.loc[u, v] = 1
        
        acr_updates = 0
        acr_unclear = 0
        acr_details = []
        
        # 对无向边运行 ACR
        for u, v in undirected_edges:
            print(f"  ❓ Undirected: {u}-{v} -> Asking ACR...")
            
            X, Y = df[u].values, df[v].values
            try:
                analysis = self.engine.analyze_pair(X, Y)
                res = self.engine.infer_causality(analysis['narrative'])
                pred = res.get('direction') or res.get('causal_direction') or 'Unclear'
                
                detail = {
                    'edge': f"{u}-{v}",
                    'prediction': pred
                }
                
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
                    hybrid_adjmat.loc[u, v] = 1  # Fallback
                    acr_unclear += 1
                
                acr_details.append(detail)
                
            except Exception as e:
                print(f"     ❌ Error: {e}")
                hybrid_adjmat.loc[u, v] = 1  # Fallback
                acr_details.append({'edge': f"{u}-{v}", 'prediction': 'Error', 'error': str(e)})

        # 4. 计算结果
        # 基座算法的 DAG 形式 (用于对比)
        base_dag_adj = temp_adj.copy()
        # 将无向边按字母顺序定向
        for u in nodes:
            for v in nodes:
                if u != v and base_dag_adj.loc[u, v] == 1 and base_dag_adj.loc[v, u] == 1:
                    if u < v:
                        base_dag_adj.loc[v, u] = 0
                    else:
                        base_dag_adj.loc[u, v] = 0
                    
        base_metrics = self.compute_metrics(true_adjmat, base_dag_adj)
        hybrid_metrics = self.compute_metrics(true_adjmat, hybrid_adjmat)
        
        print(f"\n{'='*60}")
        print(f"📊 RESULTS SUMMARY: {network_name.upper()}")
        print(f"Base Algorithm: {base_algorithm.upper()}")
        print(f"{'='*60}")
        print(f"Total Edges: {len(base_edges)} | Undirected: {len(undirected_edges)}")
        print(f"ACR Updates: {acr_updates} | Unclear: {acr_unclear}")
        
        print(f"\nMetrics Comparison:")
        print(f"{'Metric':<15} {base_algorithm.upper()+' (Base)':<15} {'Hybrid':<12} {'Delta':<10}")
        print(f"{'-'*55}")
        print(f"{'SHD':<15} {base_metrics['shd']:<15} {hybrid_metrics['shd']:<12} {base_metrics['shd'] - hybrid_metrics['shd']:+d}")
        print(f"{'Orient F1':<15} {base_metrics['orientation']['f1']:.3f}          {hybrid_metrics['orientation']['f1']:.3f}       {hybrid_metrics['orientation']['f1'] - base_metrics['orientation']['f1']:+.3f}")
        print(f"{'Skeleton F1':<15} {base_metrics['skeleton']['f1']:.3f}          {hybrid_metrics['skeleton']['f1']:.3f}       {hybrid_metrics['skeleton']['f1'] - base_metrics['skeleton']['f1']:+.3f}")
        print(f"{'='*60}")
        
        # 保存
        results = {
            'network': network_name,
            'base_algorithm': base_algorithm,
            'sample_size': sample_size,
            'base_metrics': base_metrics,
            'hybrid_metrics': hybrid_metrics,
            'details': {
                'total_edges': len(base_edges),
                'undirected_count': len(undirected_edges),
                'acr_updates': acr_updates,
                'acr_unclear': acr_unclear,
                'unclear_ratio': acr_unclear / len(undirected_edges) if undirected_edges else 0,
                'acr_details': acr_details
            },
            'is_fallback': base_info.get('is_fallback', False)
        }
        
        outfile = os.path.join(RESULTS_DIR, f"{network_name}_{base_algorithm}_hybrid.json")
        with open(outfile, 'w') as f:
            json.dump(results, f, indent=2)
        print(f"Results saved to {outfile}")
        
        return results

def main():
    parser = argparse.ArgumentParser(description='Hybrid Pipeline Test')
    parser.add_argument('--network', type=str, default='alarm',
                        help='Network to test (default: alarm)')
    parser.add_argument('--sample_size', type=int, default=1000,
                        help='Sample size (default: 1000)')
    parser.add_argument('--base_algorithm', type=str, default='pc',
                        choices=SUPPORTED_BASE_ALGORITHMS,
                        help=f'Base algorithm to use (default: pc). Supported: {SUPPORTED_BASE_ALGORITHMS}')
    parser.add_argument('--all_algorithms', action='store_true',
                        help='Run with all supported base algorithms')
    args = parser.parse_args()
    
    evaluator = HybridEvaluator()
    if not evaluator.engine:
        print("Engine initialization failed. Exiting.")
        return
    
    if args.all_algorithms:
        # 运行所有基座算法
        all_results = {}
        for algo in SUPPORTED_BASE_ALGORITHMS:
            print(f"\n{'#'*60}")
            print(f"# Running with base algorithm: {algo.upper()}")
            print(f"{'#'*60}")
            result = evaluator.run_hybrid_pipeline(
                args.network, 
                args.sample_size, 
                algo
            )
            if result:
                all_results[algo] = result
        
        # 保存汇总结果
        summary_file = os.path.join(RESULTS_DIR, f"{args.network}_all_algorithms_summary.json")
        with open(summary_file, 'w') as f:
            json.dump(all_results, f, indent=2)
        print(f"\n💾 Summary saved to: {summary_file}")
        
        # 打印对比表格
        print(f"\n{'='*70}")
        print(f"📊 ALGORITHM COMPARISON SUMMARY: {args.network.upper()}")
        print(f"{'='*70}")
        print(f"{'Algorithm':<12} {'Base SHD':<12} {'Hybrid SHD':<12} {'Improvement':<12} {'F1':<10}")
        print(f"{'-'*70}")
        for algo, result in all_results.items():
            base_shd = result['base_metrics']['shd']
            hybrid_shd = result['hybrid_metrics']['shd']
            improvement = base_shd - hybrid_shd
            f1 = result['hybrid_metrics']['orientation']['f1']
            print(f"{algo.upper():<12} {base_shd:<12} {hybrid_shd:<12} {improvement:+d}          {f1:.3f}")
        print(f"{'='*70}")
    else:
        # 运行单个基座算法
        evaluator.run_hybrid_pipeline(
            args.network, 
            args.sample_size, 
            args.base_algorithm
        )


if __name__ == "__main__":
    main()
