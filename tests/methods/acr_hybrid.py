"""
ACR-Hybrid 方法 - Signal-Prioritized Orienting

核心理论洞察：
- MEC 边携带强非对称性信号，但信号强度是异质的
- 错误做法：随机顺序处理边，弱信号边先定向会阻塞强信号边
- 正确做法：按信号强度优先级排序，强信号边先定向，通过 Meek 规则传播约束

Pipeline:
1. PC 初始化: 获取 PDAG (骨架 + V-结构)
2. 全局信号分析: 计算所有 MEC 边的复合信号强度，构建优先队列
3. 优先级定向循环:
   - 按信号强度降序处理边
   - 检查边是否已被 Meek 规则定向
   - LLM 查询 + 环检测验证
   - 提交后运行 Meek 规则传播
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))

import numpy as np
import pandas as pd
from tqdm import tqdm
from typing import Set, Tuple, Dict, Any, List
from heapq import heappush, heappop
import networkx as nx

from pgmpy.estimators import PC
from utils_set.stat_translator import StatTranslator
from utils_set.prompts import get_prompt
from llms.manager import llm_manager
from .metrics import compute_shd

import json


class ACRHybrid:
    """
    ACR-Hybrid: Signal-Prioritized Orienting
    
    核心改进：按信号强度优先级处理 MEC 边
    - 强信号边先定向，通过 Meek 规则传播约束
    - 弱信号边可能被强信号边的传播自动解决
    """
    
    def __init__(self, 
                 model_name: str = "deepseek-v3-lanyun",
                 prompt_template: str = "statistical_judge"):
        """
        Args:
            model_name: LLM 模型名称
            prompt_template: Prompt 模板 ('statistical_judge' 或 'sherlock')
        """
        self.model_name = model_name
        self.prompt_template = prompt_template
        self.translator = StatTranslator()
        
        # 验证模型
        available_models = llm_manager.list_models()
        if self.model_name not in available_models:
            print(f"⚠️  Model '{self.model_name}' not found. Available: {available_models}")
    
    def _get_pc_pdag(self, df: pd.DataFrame) -> Tuple[nx.DiGraph, Set[Tuple], Set[frozenset]]:
        """
        运行 PC 算法，返回:
        - pdag: NetworkX DiGraph (包含有向和无向边)
        - oriented_edges: PC 已定向的边 (V-structures)
        - undirected_pairs: MEC 边 (未定向，作为 frozenset)
        """
        pc = PC(df)
        skeleton_graph, separating_sets = pc.build_skeleton()
        pdag = pc.skeleton_to_pdag(skeleton_graph, separating_sets)
        
        # 转换为 NetworkX DiGraph
        nx_pdag = nx.DiGraph()
        nx_pdag.add_nodes_from(df.columns)
        
        oriented_edges = set()
        undirected_pairs = set()
        
        for edge in pdag.edges():
            u, v = edge
            nx_pdag.add_edge(u, v)
            
            # 检查是否双向边 (未定向)
            if pdag.has_edge(v, u):
                undirected_pairs.add(frozenset({u, v}))
            else:
                oriented_edges.add((u, v))
        
        return nx_pdag, oriented_edges, undirected_pairs
    
    def _compute_signal_strength(self, df: pd.DataFrame, node_x: str, node_y: str) -> Dict[str, Any]:
        """
        计算边的复合信号强度
        
        Returns:
            dict: {
                'priority_score': float,  # 用于排序的优先级分数
                'res_strength': float,    # 残差非对称性强度
                'ent_strength': float,    # 熵非对称性强度
                'profile': dict           # 完整的复合配置文件
            }
        """
        profile = self.translator.compute_composite_profile(df, node_x, node_y)
        
        if 'error' in profile:
            return {
                'priority_score': 0.0,
                'res_strength': 0.0,
                'ent_strength': 0.0,
                'profile': profile
            }
        
        res_strength = profile['functional']['strength']
        ent_strength = profile['informational']['strength']
        
        # 复合优先级分数 (可以调整权重)
        priority_score = res_strength + ent_strength
        
        return {
            'priority_score': priority_score,
            'res_strength': res_strength,
            'ent_strength': ent_strength,
            'profile': profile
        }
    
    def _is_edge_undirected(self, pdag: nx.DiGraph, u: str, v: str) -> bool:
        """检查边是否仍然未定向 (双向边)"""
        return pdag.has_edge(u, v) and pdag.has_edge(v, u)
    
    def _is_edge_oriented(self, pdag: nx.DiGraph, u: str, v: str) -> Tuple[bool, str]:
        """
        检查边是否已定向
        Returns: (is_oriented, direction)
            direction: 'u->v', 'v->u', or 'undirected'
        """
        has_uv = pdag.has_edge(u, v)
        has_vu = pdag.has_edge(v, u)
        
        if has_uv and not has_vu:
            return True, 'u->v'
        elif has_vu and not has_uv:
            return True, 'v->u'
        elif has_uv and has_vu:
            return False, 'undirected'
        else:
            return True, 'removed'  # 边被移除
    
    def _would_create_cycle(self, pdag: nx.DiGraph, u: str, v: str) -> bool:
        """检查添加 u->v 是否会创建环"""
        # 临时移除 v->u (如果存在)
        had_vu = pdag.has_edge(v, u)
        if had_vu:
            pdag.remove_edge(v, u)
        
        # 检查是否存在 v 到 u 的路径 (如果存在，添加 u->v 会创建环)
        try:
            has_cycle = nx.has_path(pdag, v, u)
        except nx.NetworkXError:
            has_cycle = False
        
        # 恢复边
        if had_vu:
            pdag.add_edge(v, u)
        
        return has_cycle
    
    def _orient_edge(self, pdag: nx.DiGraph, u: str, v: str):
        """定向边为 u->v (移除 v->u)"""
        if pdag.has_edge(v, u):
            pdag.remove_edge(v, u)
        if not pdag.has_edge(u, v):
            pdag.add_edge(u, v)
    
    def _apply_meek_rules(self, pdag: nx.DiGraph, max_iterations: int = 10) -> int:
        """
        应用 Meek 规则传播定向约束
        
        Meek Rules:
        R1: If a->b - c, and a and c are not adjacent, then b->c
        R2: If a->b->c and a - c, then a->c
        R3: If a - b, a - c, a - d, b->d, c->d, and b,c not adjacent, then a->d
        R4: If a - b, a - c, c->d, b - d, and b,c not adjacent, then a->b
        
        Returns: 新定向的边数
        """
        total_oriented = 0
        
        for _ in range(max_iterations):
            oriented_this_round = 0
            
            # 获取所有未定向边
            undirected = []
            for u, v in pdag.edges():
                if pdag.has_edge(v, u):
                    if frozenset({u, v}) not in [frozenset(e) for e in undirected]:
                        undirected.append((u, v))
            
            for u, v in undirected:
                # 检查边是否仍未定向
                if not self._is_edge_undirected(pdag, u, v):
                    continue
                
                # R1: a->b - c, a not adj c => b->c
                # 检查 u - v，是否存在 a->u 且 a 不与 v 相邻
                for a in pdag.predecessors(u):
                    if not pdag.has_edge(u, a):  # a->u (单向)
                        if not pdag.has_edge(a, v) and not pdag.has_edge(v, a):  # a 不与 v 相邻
                            if not self._would_create_cycle(pdag, u, v):
                                self._orient_edge(pdag, u, v)
                                oriented_this_round += 1
                                break
                
                if not self._is_edge_undirected(pdag, u, v):
                    continue
                
                # 同样检查反方向
                for a in pdag.predecessors(v):
                    if not pdag.has_edge(v, a):  # a->v (单向)
                        if not pdag.has_edge(a, u) and not pdag.has_edge(u, a):  # a 不与 u 相邻
                            if not self._would_create_cycle(pdag, v, u):
                                self._orient_edge(pdag, v, u)
                                oriented_this_round += 1
                                break
                
                if not self._is_edge_undirected(pdag, u, v):
                    continue
                
                # R2: a->b->c and a - c => a->c
                # 检查是否存在 a 使得 a->u->v 且 a - v
                for a in pdag.predecessors(u):
                    if not pdag.has_edge(u, a):  # a->u (单向)
                        if pdag.has_edge(u, v) and not pdag.has_edge(v, u):  # u->v (单向)
                            if pdag.has_edge(a, v) and pdag.has_edge(v, a):  # a - v (双向)
                                if not self._would_create_cycle(pdag, a, v):
                                    self._orient_edge(pdag, a, v)
                                    oriented_this_round += 1
            
            total_oriented += oriented_this_round
            if oriented_this_round == 0:
                break
        
        return total_oriented
    
    def _query_llm_with_profile(self, df: pd.DataFrame, node_x: str, node_y: str,
                                 profile: Dict, context: str = "") -> Dict[str, Any]:
        """
        使用 Composite Profile 查询 LLM
        """
        if 'error' in profile:
            return {
                'direction': 'Unclear',
                'confidence': 'low',
                'primary_evidence': f"Profile error: {profile['error']}",
                'reasoning_chain': ''
            }
        
        # 构建 Prompt
        if self.prompt_template == 'statistical_judge':
            prompt = get_prompt(
                'statistical_judge',
                context=context or f"PC confirmed edge, direction ambiguous.",
                p_xy=profile['functional']['p_xy'],
                p_yx=profile['functional']['p_yx'],
                func_signal=profile['functional']['direction_signal'],
                func_strength=profile['functional']['strength'],
                h_x=profile['informational']['h_x'],
                h_y=profile['informational']['h_y'],
                info_signal=profile['informational']['direction_signal'],
                info_strength=profile['informational']['strength'],
                consensus=profile['consensus']
            )
        else:
            # Fallback to sherlock with narrative
            stats = self.translator.analyze(df[node_x].values, df[node_y].values)
            narrative = self.translator.generate_narrative(stats)
            prompt = get_prompt('sherlock', narrative=narrative)
        
        prompt += "\n\nIMPORTANT: Please ensure your response is a valid JSON object."
        
        try:
            response = llm_manager.call_model(self.model_name, prompt, mode='text')
            result = self._parse_llm_response(response)
            return result
        except Exception as e:
            return {
                'direction': 'Unclear',
                'confidence': 'low',
                'primary_evidence': f"LLM error: {e}",
                'reasoning_chain': ''
            }
    
    def _parse_llm_response(self, text: str) -> Dict[str, Any]:
        """解析 LLM 响应"""
        try:
            start = text.find('{')
            end = text.rfind('}') + 1
            if start != -1 and end > start:
                json_str = text[start:end]
                parsed = json.loads(json_str)
                
                direction = (
                    parsed.get('direction') or 
                    parsed.get('causal_direction') or 
                    parsed.get('causal_direction_judgment') or
                    'Unclear'
                )
                
                return {
                    'direction': direction,
                    'confidence': parsed.get('confidence', 'unknown'),
                    'primary_evidence': parsed.get('primary_evidence', ''),
                    'reasoning_chain': parsed.get('reasoning_chain', ''),
                    'full_parsed': parsed
                }
        except json.JSONDecodeError:
            pass
        
        return {
            'direction': 'Unclear',
            'confidence': 'low',
            'primary_evidence': 'Parse failed',
            'reasoning_chain': text
        }
    
    def run(self, df: pd.DataFrame, true_adjmat: pd.DataFrame, 
            nodes: list, edges: list, network_name: str = "network", 
            verbose: bool = True) -> Dict[str, Any]:
        """
        运行 ACR-Hybrid (Signal-Prioritized Orienting)
        
        核心流程：
        1. PC 获取 PDAG
        2. 计算所有 MEC 边的信号强度，构建优先队列
        3. 按优先级处理边，每次定向后运行 Meek 规则传播
        """
        n_nodes = len(nodes)
        
        if verbose:
            print(f"  Ground truth: {len(edges)} edges")
        
        # ========== Step 1: PC Algorithm ==========
        if verbose:
            print(f"  Step 1: Running PC algorithm...")
        
        pdag, pc_oriented, undirected_pairs = self._get_pc_pdag(df)
        
        if verbose:
            print(f"    PC Oriented (V-structures): {len(pc_oriented)}")
            print(f"    MEC Edges (Undirected): {len(undirected_pairs)}")
        
        # ========== Step 2: Global Signal Profiling ==========
        if verbose:
            print(f"  Step 2: Computing signal strengths for {len(undirected_pairs)} MEC edges...")
        
        # 构建优先队列 (使用负数因为 heapq 是最小堆)
        priority_queue = []  # [(neg_priority, edge_tuple, signal_info)]
        edge_signals = {}    # 存储每条边的信号信息
        
        for pair in undirected_pairs:
            u, v = sorted(list(pair))  # 标准化顺序
            signal = self._compute_signal_strength(df, u, v)
            edge_signals[(u, v)] = signal
            
            # 负数优先级 (heapq 是最小堆，我们要最大优先)
            heappush(priority_queue, (-signal['priority_score'], (u, v), signal))
        
        if verbose and priority_queue:
            top_score = -priority_queue[0][0]
            print(f"    Top signal strength: {top_score:.4f}")
        
        # ========== Step 3: Prioritized Orientation Loop ==========
        if verbose:
            print(f"  Step 3: Signal-prioritized orientation...")
        
        llm_results = []
        llm_calls = 0
        llm_correct = 0
        meek_oriented = 0
        cycle_rejected = 0
        unclear_count = 0
        
        # 进度条
        total_mec = len(undirected_pairs)
        pbar = tqdm(total=total_mec, desc=f"  Orienting ({network_name})", disable=not verbose)
        processed = 0
        
        while priority_queue:
            neg_priority, (u, v), signal = heappop(priority_queue)
            priority = -neg_priority
            
            # 检查边是否仍未定向 (可能被 Meek 规则定向了)
            is_oriented, current_dir = self._is_edge_oriented(pdag, u, v)
            
            if is_oriented:
                if current_dir != 'removed':
                    meek_oriented += 1
                processed += 1
                pbar.update(1)
                continue
            
            # 查询 LLM
            profile = signal['profile']
            result = self._query_llm_with_profile(
                df, u, v, profile,
                context=f"PC confirmed edge {u}--{v}, direction ambiguous. Signal strength: {priority:.3f}"
            )
            llm_calls += 1
            
            prediction = result.get('direction', 'Unclear')
            
            # 确定要尝试的方向
            # A->B 对应 u->v, B->A 对应 v->u
            if prediction == "A->B":
                try_directions = [(u, v), (v, u)]
            elif prediction == "B->A":
                try_directions = [(v, u), (u, v)]
            else:
                unclear_count += 1
                try_directions = [(u, v), (v, u)]  # 尝试两个方向
            
            # 尝试定向 (带环检测)
            oriented = False
            final_direction = None
            
            for src, tgt in try_directions:
                if not self._would_create_cycle(pdag, src, tgt):
                    self._orient_edge(pdag, src, tgt)
                    final_direction = f"{src}->{tgt}"
                    oriented = True
                    break
                else:
                    cycle_rejected += 1
            
            # 检查正确性
            true_uv = true_adjmat.loc[u, v] if u in true_adjmat.index and v in true_adjmat.columns else 0
            true_vu = true_adjmat.loc[v, u] if v in true_adjmat.index and u in true_adjmat.columns else 0
            
            if oriented and final_direction:
                if final_direction == f"{u}->{v}" and true_uv == 1:
                    is_correct = True
                elif final_direction == f"{v}->{u}" and true_vu == 1:
                    is_correct = True
                else:
                    is_correct = False
                
                if is_correct:
                    llm_correct += 1
            else:
                is_correct = False
            
            llm_results.append({
                'edge': f"{u}-{v}",
                'priority_score': priority,
                'prediction': prediction,
                'final_direction': final_direction,
                'confidence': result.get('confidence', 'unknown'),
                'is_correct': is_correct,
                'true_direction': 'u->v' if true_uv == 1 else ('v->u' if true_vu == 1 else 'none'),
                'signal': {
                    'res_strength': signal['res_strength'],
                    'ent_strength': signal['ent_strength']
                }
            })
            
            # 运行 Meek 规则传播
            if oriented:
                meek_new = self._apply_meek_rules(pdag)
                if meek_new > 0 and verbose:
                    tqdm.write(f"    Meek propagated {meek_new} edges after {final_direction}")
            
            processed += 1
            pbar.update(1)
        
        pbar.close()
        
        # ========== Step 4: Build Final Adjacency Matrix ==========
        pred_adjmat = pd.DataFrame(
            np.zeros((n_nodes, n_nodes)),
            index=nodes,
            columns=nodes
        )
        
        for u, v in pdag.edges():
            if u in nodes and v in nodes:
                # 只添加单向边
                if not pdag.has_edge(v, u):
                    pred_adjmat.loc[u, v] = 1
                else:
                    # 双向边 (未定向) - 随机选一个方向或跳过
                    pass
        
        # ========== Step 5: Compute Metrics ==========
        shd = compute_shd(true_adjmat, pred_adjmat)
        llm_accuracy = llm_correct / llm_calls if llm_calls > 0 else 0
        
        if verbose:
            print(f"    LLM Calls: {llm_calls}")
            print(f"    LLM Accuracy: {llm_accuracy:.1%} ({llm_correct}/{llm_calls})")
            print(f"    Meek Propagated: {meek_oriented}")
            print(f"    Cycle Rejected: {cycle_rejected}")
            print(f"    Unclear: {unclear_count}")
            print(f"    Final SHD: {shd}")
        
        return {
            'adjmat': pred_adjmat,
            'shd': shd,
            'llm_accuracy': llm_accuracy,
            'llm_correct': llm_correct,
            'llm_total': llm_calls,
            'total_edges': len(edges),
            'pc_oriented_count': len(pc_oriented),
            'mec_edge_count': len(undirected_pairs),
            'meek_propagated': meek_oriented,
            'cycle_rejected': cycle_rejected,
            'unclear_count': unclear_count,
            'unclear_ratio': unclear_count / llm_calls if llm_calls > 0 else 0,
            'pairwise_results': llm_results
        }
