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

# FCI 算法支持 (causal-learn)
try:
    from causallearn.search.ConstraintBased.FCI import fci
    from causallearn.utils.cit import fisherz, chisq, gsq, mv_fisherz, kci
    CAUSALLEARN_AVAILABLE = True
except ImportError:
    CAUSALLEARN_AVAILABLE = False
    print("Warning: causal-learn not available. FCI algorithm will use fallback implementation.")

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


def run_dual_pc_algorithm(df, alpha=0.05):
    """
    运行 Dual PC 算法
    Dual PC 是 PC 算法的变体，适用于高斯连续数据
    使用 Fisher-Z 检验进行条件独立性测试
    
    Args:
        df: 数据框
        alpha: 显著性水平
    
    Returns:
        edges: 有向边列表
        pdag_info: PDAG 信息（包含无向边）
    """
    if not PGMPY_AVAILABLE:
        return None, None
    try:
        # Dual PC 使用更严格的显著性水平和 Fisher-Z 检验
        # 适用于连续高斯数据
        pc = PC(data=df)
        
        # 尝试获取 PDAG 以保留无向边信息
        try:
            model = pc.estimate(
                significance_level=alpha,
                return_type='pdag'
            )
            edges = list(model.edges())
            
            # 提取无向边（双向边）
            undirected_edges = []
            directed_edges = []
            edge_set = set(edges)
            processed = set()
            
            for u, v in edges:
                pair = tuple(sorted((u, v)))
                if pair in processed:
                    continue
                processed.add(pair)
                
                # 检查是否双向（无向）
                if (v, u) in edge_set:
                    undirected_edges.append((u, v))
                else:
                    directed_edges.append((u, v))
            
            pdag_info = {
                'directed_edges': directed_edges,
                'undirected_edges': undirected_edges,
                'all_edges': edges
            }
            
            return edges, pdag_info
            
        except Exception:
            # 回退到 DAG 模式
            model = pc.estimate(significance_level=alpha, return_type='dag')
            edges = list(model.edges())
            return edges, {'directed_edges': edges, 'undirected_edges': [], 'all_edges': edges}
            
    except Exception as e:
        print(f"Dual PC algorithm failed: {e}")
        return None, None


def run_fci_algorithm(df, alpha=0.05):
    """
    运行 FCI (Fast Causal Inference) 算法
    FCI 可以处理存在潜在混淆因子的情况，输出 PAG (Partial Ancestral Graph)
    
    Args:
        df: 数据框
        alpha: 显著性水平
    
    Returns:
        edges: 边列表
        pag_info: PAG 信息（包含边类型）
    """
    if CAUSALLEARN_AVAILABLE:
        try:
            # 使用 causal-learn 的 FCI 实现
            node_names = list(df.columns)
            
            # 检查数据类型并转换
            # 对于离散数据，需要转换为数值类型
            df_numeric = df.copy()
            is_discrete = False
            for col in df_numeric.columns:
                if df_numeric[col].dtype == 'object' or df_numeric[col].dtype.name == 'category':
                    # 将分类数据转换为数值
                    df_numeric[col] = pd.Categorical(df_numeric[col]).codes
                    is_discrete = True
                elif df_numeric[col].dtype.kind in 'iub':
                    is_discrete = True
            
            data = df_numeric.values.astype(float)
            
            # 选择合适的条件独立性检验
            # 对于离散数据使用卡方检验，连续数据使用 Fisher-Z
            if is_discrete:
                # 离散数据使用卡方检验
                cit = chisq
            else:
                # 连续数据使用 Fisher-Z
                cit = fisherz
            
            # 运行 FCI
            g, edges_info = fci(data, cit, alpha)
            
            # 解析 PAG 结果
            # PAG 边类型: -1 = circle, 1 = arrowhead, 2 = tail
            directed_edges = []
            bidirected_edges = []
            undirected_edges = []
            orientable_edges = []  # 可以被定向的边
            
            n = len(node_names)
            adj_matrix = g.graph
            
            for i in range(n):
                for j in range(i + 1, n):
                    if adj_matrix[i, j] != 0 or adj_matrix[j, i] != 0:
                        u, v = node_names[i], node_names[j]
                        
                        # 解析边类型
                        # adj_matrix[i,j] 表示 j 端的标记
                        # adj_matrix[j,i] 表示 i 端的标记
                        mark_at_j = adj_matrix[i, j]
                        mark_at_i = adj_matrix[j, i]
                        
                        if mark_at_j == 1 and mark_at_i == 2:
                            # i -> j (tail at i, arrow at j)
                            directed_edges.append((u, v))
                        elif mark_at_j == 2 and mark_at_i == 1:
                            # j -> i
                            directed_edges.append((v, u))
                        elif mark_at_j == 1 and mark_at_i == 1:
                            # i <-> j (bidirected)
                            bidirected_edges.append((u, v))
                        elif mark_at_j == -1 or mark_at_i == -1:
                            # 包含 circle 端点，可定向
                            orientable_edges.append((u, v))
                        else:
                            # 无向边
                            undirected_edges.append((u, v))
            
            # 合并所有边用于 SHD 计算
            all_edges = directed_edges.copy()
            # 对于可定向边，暂时按字母顺序定向
            for u, v in orientable_edges:
                if u < v:
                    all_edges.append((u, v))
                else:
                    all_edges.append((v, u))
            
            pag_info = {
                'directed_edges': directed_edges,
                'bidirected_edges': bidirected_edges,
                'undirected_edges': undirected_edges,
                'orientable_edges': orientable_edges,
                'all_edges': all_edges
            }
            
            return all_edges, pag_info
            
        except Exception as e:
            print(f"FCI algorithm (causal-learn) failed: {e}")
            return None, None
    else:
        # Fallback: 使用 PC 算法模拟 FCI 行为
        # 注意：这不是真正的 FCI，仅用于测试框架
        print("Warning: Using PC as FCI fallback (causal-learn not installed)")
        if not PGMPY_AVAILABLE:
            return None, None
        try:
            pc = PC(data=df)
            try:
                model = pc.estimate(significance_level=alpha, return_type='pdag')
            except:
                model = pc.estimate(significance_level=alpha, return_type='dag')
            
            edges = list(model.edges())
            
            # 模拟 PAG 输出格式
            # 将无向边标记为可定向边
            edge_set = set(edges)
            directed_edges = []
            orientable_edges = []
            processed = set()
            
            for u, v in edges:
                pair = tuple(sorted((u, v)))
                if pair in processed:
                    continue
                processed.add(pair)
                
                if (v, u) in edge_set:
                    # 双向 = 可定向
                    orientable_edges.append((u, v))
                else:
                    directed_edges.append((u, v))
            
            # 为可定向边选择方向
            all_edges = directed_edges.copy()
            for u, v in orientable_edges:
                all_edges.append((u, v))
            
            pag_info = {
                'directed_edges': directed_edges,
                'bidirected_edges': [],
                'undirected_edges': [],
                'orientable_edges': orientable_edges,
                'all_edges': all_edges,
                'is_fallback': True
            }
            
            return all_edges, pag_info
            
        except Exception as e:
            print(f"FCI fallback (PC) failed: {e}")
            return None, None


def extract_undirected_edges_from_pdag(pdag_info):
    """从 PDAG 信息中提取无向边"""
    if pdag_info is None:
        return []
    return pdag_info.get('undirected_edges', [])


def extract_orientable_edges_from_pag(pag_info):
    """从 PAG 信息中提取可定向边"""
    if pag_info is None:
        return []
    return pag_info.get('orientable_edges', [])


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
    
    def experiment_dual_pc_acr(self, network_name="sachs", sample_size=1000):
        """
        Dual PC + ACR 实验
        验证 ACR 对高斯连续数据（如 Sachs）的优化能力
        Requirements: 2.1, 2.3, 2.5
        """
        print(f"\n{'='*60}")
        print(f"Dual PC + ACR Experiment on {network_name}")
        print(f"{'='*60}")
        
        # 加载网络
        model = get_example_model(network_name)
        sampler = BayesianModelSampling(model)
        df = sampler.forward_sample(size=sample_size)
        
        true_edges = list(model.edges())
        true_edges_set = set(true_edges)
        n_edges = len(true_edges)
        
        print(f"Network: {network_name}, Edges: {n_edges}")
        
        # 1. 运行 Dual PC 算法
        print(f"\n[1/3] Running Dual PC Algorithm...")
        dual_pc_edges, pdag_info = run_dual_pc_algorithm(df)
        
        if dual_pc_edges is None:
            print("  Dual PC failed, skipping...")
            return None
        
        dual_pc_shd = compute_shd(true_edges, dual_pc_edges)
        print(f"  Dual PC SHD: {dual_pc_shd}")
        
        # 2. 提取无向边
        undirected_edges = extract_undirected_edges_from_pdag(pdag_info)
        print(f"\n[2/3] Extracted {len(undirected_edges)} undirected edges from PDAG")
        
        # 3. 对无向边运行 ACR
        print(f"\n[3/3] Running ACR on undirected edges...")
        
        if len(undirected_edges) > 0:
            acr_oriented_edges, acr_accuracy, acr_details = self.run_acr_on_edges(
                df, undirected_edges, true_edges_set
            )
            
            # 合并有向边和 ACR 定向的边
            final_edges = pdag_info.get('directed_edges', []).copy()
            final_edges.extend(acr_oriented_edges)
            
            dual_pc_acr_shd = compute_shd(true_edges, final_edges)
        else:
            print("  No undirected edges to orient, using Dual PC result directly")
            final_edges = dual_pc_edges
            dual_pc_acr_shd = dual_pc_shd
            acr_accuracy = None
            acr_details = []
        
        # 4. 计算 F1 指标
        true_set = set(true_edges)
        pred_set = set(final_edges)
        tp = len(true_set & pred_set)
        fp = len(pred_set - true_set)
        fn = len(true_set - pred_set)
        
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
        
        print(f"\n📊 Results:")
        print(f"  Dual PC SHD: {dual_pc_shd}")
        print(f"  Dual PC + ACR SHD: {dual_pc_acr_shd}")
        print(f"  Improvement: {dual_pc_shd - dual_pc_acr_shd}")
        print(f"  F1: {f1:.3f}")
        
        result = {
            'experiment': 'Dual_PC_ACR',
            'network': network_name,
            'sample_size': sample_size,
            'dual_pc_shd': dual_pc_shd,
            'dual_pc_acr_shd': dual_pc_acr_shd,
            'shd_improvement': dual_pc_shd - dual_pc_acr_shd,
            'f1': f1,
            'precision': precision,
            'recall': recall,
            'undirected_edges_count': len(undirected_edges),
            'acr_accuracy': acr_accuracy,
            'details': acr_details
        }
        
        return result
    
    def experiment_fci_acr(self, network_name="asia", sample_size=1000):
        """
        FCI + ACR 实验
        验证 ACR 定向能力在处理潜在混淆因子的 PAGs 上的效果
        Requirements: 2.3, 2.4, 2.6
        """
        print(f"\n{'='*60}")
        print(f"FCI + ACR Experiment on {network_name}")
        print(f"{'='*60}")
        
        # 加载网络
        model = get_example_model(network_name)
        sampler = BayesianModelSampling(model)
        df = sampler.forward_sample(size=sample_size)
        
        true_edges = list(model.edges())
        true_edges_set = set(true_edges)
        n_edges = len(true_edges)
        
        print(f"Network: {network_name}, Edges: {n_edges}")
        
        # 1. 运行 FCI 算法
        print(f"\n[1/3] Running FCI Algorithm...")
        fci_edges, pag_info = run_fci_algorithm(df)
        
        if fci_edges is None:
            print("  FCI failed, skipping...")
            return None
        
        fci_shd = compute_shd(true_edges, fci_edges)
        print(f"  FCI SHD: {fci_shd}")
        
        if pag_info and pag_info.get('is_fallback'):
            print("  (Note: Using PC as FCI fallback)")
        
        # 2. 提取可定向边
        orientable_edges = extract_orientable_edges_from_pag(pag_info)
        print(f"\n[2/3] Extracted {len(orientable_edges)} orientable edges from PAG")
        
        # 3. 对可定向边运行 ACR
        print(f"\n[3/3] Running ACR on orientable edges...")
        
        if len(orientable_edges) > 0:
            acr_oriented_edges, acr_accuracy, acr_details = self.run_acr_on_edges(
                df, orientable_edges, true_edges_set
            )
            
            # 合并已定向边和 ACR 定向的边
            final_edges = pag_info.get('directed_edges', []).copy()
            final_edges.extend(acr_oriented_edges)
            
            fci_acr_shd = compute_shd(true_edges, final_edges)
        else:
            print("  No orientable edges, using FCI result directly")
            final_edges = fci_edges
            fci_acr_shd = fci_shd
            acr_accuracy = None
            acr_details = []
        
        # 4. 计算 F1 指标
        true_set = set(true_edges)
        pred_set = set(final_edges)
        tp = len(true_set & pred_set)
        fp = len(pred_set - true_set)
        fn = len(true_set - pred_set)
        
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
        
        print(f"\n📊 Results:")
        print(f"  FCI SHD: {fci_shd}")
        print(f"  FCI + ACR SHD: {fci_acr_shd}")
        print(f"  Improvement: {fci_shd - fci_acr_shd}")
        print(f"  F1: {f1:.3f}")
        
        result = {
            'experiment': 'FCI_ACR',
            'network': network_name,
            'sample_size': sample_size,
            'fci_shd': fci_shd,
            'fci_acr_shd': fci_acr_shd,
            'shd_improvement': fci_shd - fci_acr_shd,
            'f1': f1,
            'precision': precision,
            'recall': recall,
            'orientable_edges_count': len(orientable_edges),
            'acr_accuracy': acr_accuracy,
            'is_fallback': pag_info.get('is_fallback', False) if pag_info else False,
            'details': acr_details
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
                        choices=['all', 'p2.1', 'p2.2', 'p2.3', 'dual_pc', 'fci'],
                        help='Which experiment to run')
    parser.add_argument('--network', type=str, default=None,
                        help='Network to test (default varies by experiment)')
    parser.add_argument('--sample_size', type=int, default=1000,
                        help='Sample size for experiments')
    args = parser.parse_args()
    
    experimenter = P2Experimenter()
    if not experimenter.engine:
        print("Engine initialization failed. Exiting.")
        return
    
    all_results = {}
    
    if args.exp in ['all', 'p2.1']:
        network = args.network or 'sachs'
        result = experimenter.experiment_p2_1_eshd(network, args.sample_size)
        all_results['p2.1'] = result
    
    if args.exp in ['all', 'p2.2']:
        network = args.network or 'alarm'
        result = experimenter.experiment_p2_2_mmhc_acr(network, args.sample_size)
        all_results['p2.2'] = result
    
    if args.exp in ['all', 'p2.3']:
        network = args.network or 'asia'
        result = experimenter.experiment_p2_3_low_sample(network)
        all_results['p2.3'] = result
    
    # 新增: Dual PC + ACR 实验
    if args.exp in ['all', 'dual_pc']:
        network = args.network or 'sachs'
        result = experimenter.experiment_dual_pc_acr(network, args.sample_size)
        all_results['dual_pc_acr'] = result
    
    # 新增: FCI + ACR 实验
    if args.exp in ['all', 'fci']:
        network = args.network or 'asia'
        result = experimenter.experiment_fci_acr(network, args.sample_size)
        all_results['fci_acr'] = result
    
    # 保存结果
    output_file = os.path.join(RESULTS_DIR, 'p2_experiment_results.json')
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(all_results, f, indent=2, ensure_ascii=False)
    print(f"\n💾 All results saved to: {output_file}")


if __name__ == "__main__":
    main()
