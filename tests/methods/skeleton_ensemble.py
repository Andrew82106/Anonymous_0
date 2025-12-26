"""
集成骨架发现 - 侦探团投票系统

三个侦探：
1. PC 侦探 - 约束基方法，保守稳重
2. HillClimb 侦探 - 评分基方法，激进发现
3. LiNGAM 侦探 - 线性非高斯方法，擅长定向

投票策略：
- High Consensus: 2-3 个侦探同意方向 → 直接采用
- Conflict: 侦探对方向有分歧 → LLM 仲裁
- Undirected: 有边但无方向 → LLM 定向

策略选择：
- Conservative: 至少 2 票才保留边（高精度）
- Aggressive: 1 票也保留，LLM 复核（高召回）
"""

import numpy as np
import pandas as pd
from collections import defaultdict
from enum import Enum
from typing import Dict, Set, Tuple, List, Optional

from pgmpy.estimators import PC, HillClimbSearch
from pgmpy.estimators import K2Score


class EdgeType(Enum):
    """边的分类"""
    HIGH_CONSENSUS = "high_consensus"  # 2-3 侦探同意方向
    CONFLICT = "conflict"              # 方向冲突
    UNDIRECTED = "undirected"          # 有边但无方向


class Strategy(Enum):
    """策略选择"""
    CONSERVATIVE = "conservative"  # 保守：至少 2 票
    AGGRESSIVE = "aggressive"      # 激进：1 票也保留


class SkeletonEnsemble:
    """
    侦探团集成骨架发现
    
    三个侦探投票，根据投票结果分类边：
    - High Consensus: 直接采用共识方向
    - Conflict: 需要 LLM 仲裁
    - Undirected: 需要 LLM 定向
    """
    
    def __init__(self, 
                 methods: List[str] = None,
                 pc_alpha: float = 0.05,
                 strategy: Strategy = Strategy.CONSERVATIVE):
        """
        Args:
            methods: 使用的方法列表，默认 ['pc', 'hillclimb', 'lingam']
            pc_alpha: PC 算法显著性水平
            strategy: 策略选择 (CONSERVATIVE 或 AGGRESSIVE)
        """
        self.methods = methods or ['pc', 'hillclimb', 'lingam']
        self.pc_alpha = pc_alpha
        self.strategy = strategy
        
        # 存储每个方法发现的边和方向
        self.method_results = {}
    
    def discover(self, df: pd.DataFrame, verbose: bool = True) -> Tuple[Dict, Set]:
        """
        发现骨架并统计投票
        
        Returns:
            edge_votes: dict, {(u,v): {'votes': n, 'methods': [...], 'directions': {...}}}
            all_edges: set of all discovered edges
        """
        nodes = list(df.columns)
        edge_votes = defaultdict(lambda: {
            'votes': 0, 
            'methods': [], 
            'directions': {},  # {method: (source, target) or None}
            'edge_type': None
        })
        
        # 1. PC 侦探
        if 'pc' in self.methods:
            if verbose:
                print("    [1] PC 侦探 (约束基，保守)...")
            pc_edges, pc_directions = self._run_pc(df, verbose)
            self.method_results['pc'] = {'edges': pc_edges, 'directions': pc_directions}
            self._merge_results(edge_votes, pc_edges, pc_directions, 'pc')
        
        # 2. HillClimb 侦探
        if 'hillclimb' in self.methods:
            if verbose:
                print("    [2] HillClimb 侦探 (评分基，激进)...")
            hc_edges, hc_directions = self._run_hillclimb(df, verbose)
            self.method_results['hillclimb'] = {'edges': hc_edges, 'directions': hc_directions}
            self._merge_results(edge_votes, hc_edges, hc_directions, 'hillclimb')
        
        # 3. LiNGAM 侦探
        if 'lingam' in self.methods:
            if verbose:
                print("    [3] LiNGAM 侦探 (非高斯，定向专家)...")
            lingam_edges, lingam_directions = self._run_lingam(df, verbose)
            self.method_results['lingam'] = {'edges': lingam_edges, 'directions': lingam_directions}
            self._merge_results(edge_votes, lingam_edges, lingam_directions, 'lingam')
        
        # 分类边
        self._classify_edges(edge_votes)
        
        # 根据策略筛选边
        all_edges = self._filter_by_strategy(edge_votes)
        
        if verbose:
            self._print_summary(edge_votes, all_edges)
        
        return dict(edge_votes), all_edges
    
    def _run_pc(self, df: pd.DataFrame, verbose: bool) -> Tuple[Set, Dict]:
        """运行 PC 算法"""
        edges = set()
        directions = {}  # {edge: (source, target) or None}
        
        try:
            pc = PC(data=df)
            # 返回 CPDAG（部分有向图）
            model = pc.estimate(significance_level=self.pc_alpha, return_type="cpdag")
            
            directed_edges = set(model.edges())
            
            for u, v in directed_edges:
                edge = tuple(sorted([u, v]))
                edges.add(edge)
                
                # 检查是否是有向边（反向不存在）
                if (v, u) not in directed_edges:
                    directions[edge] = (u, v)
                else:
                    # 双向边 = 无向边
                    directions[edge] = None
            
            if verbose:
                directed_count = sum(1 for d in directions.values() if d is not None)
                print(f"      发现 {len(edges)} 条边，{directed_count} 条有向")
                
        except Exception as e:
            if verbose:
                print(f"      PC 失败: {e}")
        
        return edges, directions
    
    def _run_hillclimb(self, df: pd.DataFrame, verbose: bool) -> Tuple[Set, Dict]:
        """运行 HillClimb 算法"""
        edges = set()
        directions = {}
        
        try:
            hc = HillClimbSearch(df)
            # 限制迭代次数，大网络用更少的迭代
            n_vars = len(df.columns)
            max_iter = min(100, n_vars * 3)  # 最多 100 次迭代
            model = hc.estimate(scoring_method=K2Score(df), max_iter=max_iter)
            
            for u, v in model.edges():
                edge = tuple(sorted([u, v]))
                edges.add(edge)
                # HillClimb 总是返回有向边
                directions[edge] = (u, v)
            
            if verbose:
                print(f"      发现 {len(edges)} 条有向边")
                
        except Exception as e:
            if verbose:
                print(f"      HillClimb 失败: {e}")
        
        return edges, directions
    
    def _run_lingam(self, df: pd.DataFrame, verbose: bool) -> Tuple[Set, Dict]:
        """运行 LiNGAM 算法"""
        edges = set()
        directions = {}
        
        try:
            import lingam
            
            # 预处理：将分类变量转换为数值
            df_numeric = df.copy()
            for col in df_numeric.columns:
                if df_numeric[col].dtype == 'object' or df_numeric[col].dtype.name == 'category':
                    # Label encoding
                    df_numeric[col] = pd.Categorical(df_numeric[col]).codes
            
            # 检查是否有足够的方差
            if df_numeric.std().min() < 1e-10:
                if verbose:
                    print(f"      LiNGAM 跳过: 数据方差过小")
                return edges, directions
            
            # 使用 DirectLiNGAM（更稳定）
            model = lingam.DirectLiNGAM()
            model.fit(df_numeric.values)
            
            # 获取邻接矩阵
            adj_matrix = model.adjacency_matrix_
            nodes = list(df.columns)
            
            # 阈值过滤弱连接
            threshold = 0.1
            
            for i, source in enumerate(nodes):
                for j, target in enumerate(nodes):
                    if i != j and abs(adj_matrix[j, i]) > threshold:
                        # LiNGAM: adj_matrix[j,i] 表示 i -> j
                        edge = tuple(sorted([source, target]))
                        edges.add(edge)
                        # LiNGAM 总是给出方向
                        directions[edge] = (source, target)
            
            if verbose:
                print(f"      发现 {len(edges)} 条有向边")
                
        except Exception as e:
            if verbose:
                print(f"      LiNGAM 失败: {e}")
        
        return edges, directions
    
    def _merge_results(self, edge_votes: Dict, edges: Set, directions: Dict, method: str):
        """合并某个方法的结果到投票统计"""
        for edge in edges:
            edge_votes[edge]['votes'] += 1
            edge_votes[edge]['methods'].append(method)
            edge_votes[edge]['directions'][method] = directions.get(edge)
    
    def _classify_edges(self, edge_votes: Dict):
        """分类每条边"""
        for edge, info in edge_votes.items():
            directions = info['directions']
            
            # 收集所有非 None 的方向
            dir_votes = {}
            for method, direction in directions.items():
                if direction is not None:
                    if direction not in dir_votes:
                        dir_votes[direction] = []
                    dir_votes[direction].append(method)
            
            if not dir_votes:
                # 所有方法都没给方向
                info['edge_type'] = EdgeType.UNDIRECTED
                info['consensus_direction'] = None
            elif len(dir_votes) == 1:
                # 只有一个方向（可能多个方法同意）
                direction, methods = list(dir_votes.items())[0]
                if len(methods) >= 2:
                    info['edge_type'] = EdgeType.HIGH_CONSENSUS
                    info['consensus_direction'] = direction
                else:
                    # 只有 1 个方法给了方向
                    info['edge_type'] = EdgeType.UNDIRECTED
                    info['consensus_direction'] = None
            else:
                # 多个方向 = 冲突
                info['edge_type'] = EdgeType.CONFLICT
                info['consensus_direction'] = None
                info['conflicting_directions'] = dir_votes
    
    def _filter_by_strategy(self, edge_votes: Dict) -> Set:
        """根据策略筛选边"""
        if self.strategy == Strategy.CONSERVATIVE:
            # 保守：至少 2 票
            return {e for e, info in edge_votes.items() if info['votes'] >= 2}
        else:
            # 激进：1 票也保留
            return set(edge_votes.keys())
    
    def _print_summary(self, edge_votes: Dict, all_edges: Set):
        """打印汇总信息"""
        total = len(edge_votes)
        filtered = len(all_edges)
        
        # 统计各类型
        type_counts = defaultdict(int)
        for edge in all_edges:
            edge_type = edge_votes[edge]['edge_type']
            type_counts[edge_type.value] += 1
        
        # 投票分布
        vote_dist = defaultdict(int)
        for e, info in edge_votes.items():
            vote_dist[info['votes']] += 1
        
        print(f"    总计发现 {total} 条边，策略筛选后 {filtered} 条")
        print(f"    投票分布: {dict(vote_dist)}")
        print(f"    边分类: High Consensus={type_counts['high_consensus']}, "
              f"Conflict={type_counts['conflict']}, Undirected={type_counts['undirected']}")
    
    # ========== 公共接口 ==========
    
    def get_high_consensus_edges(self, edge_votes: Dict, min_votes: int = 2) -> Set:
        """获取高共识边（有明确方向且多数同意）"""
        return {
            e for e, info in edge_votes.items() 
            if info['edge_type'] == EdgeType.HIGH_CONSENSUS and info['votes'] >= min_votes
        }
    
    def get_conflict_edges(self, edge_votes: Dict) -> Set:
        """获取冲突边（方向有分歧）"""
        return {
            e for e, info in edge_votes.items() 
            if info['edge_type'] == EdgeType.CONFLICT
        }
    
    def get_undirected_edges(self, edge_votes: Dict) -> Set:
        """获取无向边（有边但无方向共识）"""
        return {
            e for e, info in edge_votes.items() 
            if info['edge_type'] == EdgeType.UNDIRECTED
        }
    
    def get_low_confidence_edges(self, edge_votes: Dict, threshold: int = 2) -> Set:
        """获取低置信度边（投票数 < threshold）- 兼容旧接口"""
        return {e for e, info in edge_votes.items() if info['votes'] < threshold}
    
    def get_consensus_direction(self, edge_votes: Dict, edge: Tuple) -> Optional[Tuple]:
        """获取某条边的共识方向"""
        info = edge_votes.get(edge, {})
        return info.get('consensus_direction')
    
    def get_conflict_info(self, edge_votes: Dict, edge: Tuple) -> Dict:
        """获取冲突边的详细信息"""
        info = edge_votes.get(edge, {})
        return {
            'directions': info.get('directions', {}),
            'conflicting_directions': info.get('conflicting_directions', {})
        }
