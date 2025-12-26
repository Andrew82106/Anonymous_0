"""
基线方法模块 - PC, HillClimb, Random
"""

import numpy as np
import pandas as pd
from pgmpy.estimators import PC, HillClimbSearch, BicScore

from .metrics import compute_shd


class PC_Algorithm:
    """PC 算法 - 约束类因果发现"""
    
    def __init__(self, alpha=0.05):
        self.alpha = alpha
    
    def run(self, df, true_adjmat=None, nodes=None):
        """
        运行 PC 算法
        
        Args:
            df: 数据 (DataFrame)
            true_adjmat: 真实邻接矩阵（用于计算 SHD）
            nodes: 节点列表（用于对齐邻接矩阵）
        
        Returns:
            dict: {'adjmat': 预测邻接矩阵, 'shd': SHD值}
        """
        try:
            pc = PC(data=df)
            model = pc.estimate(significance_level=self.alpha)
            
            sorted_nodes = sorted(df.columns)
            n = len(sorted_nodes)
            adjmat = np.zeros((n, n))
            
            for edge in model.edges():
                source_idx = sorted_nodes.index(edge[0])
                target_idx = sorted_nodes.index(edge[1])
                adjmat[source_idx, target_idx] = 1
            
            adjmat_df = pd.DataFrame(adjmat, index=sorted_nodes, columns=sorted_nodes)
            
            # 如果提供了 nodes，重新排序
            if nodes is not None:
                adjmat_df = adjmat_df.reindex(index=nodes, columns=nodes, fill_value=0)
            
            result = {'adjmat': adjmat_df}
            
            if true_adjmat is not None:
                result['shd'] = compute_shd(true_adjmat, adjmat_df)
            
            return result
            
        except Exception as e:
            print(f"PC failed: {e}")
            return None


class HillClimb_Algorithm:
    """HillClimb 算法 - 评分类因果发现"""
    
    def run(self, df, true_adjmat=None, nodes=None):
        """
        运行 HillClimb 算法
        
        Args:
            df: 数据 (DataFrame)
            true_adjmat: 真实邻接矩阵（用于计算 SHD）
            nodes: 节点列表（用于对齐邻接矩阵）
        
        Returns:
            dict: {'adjmat': 预测邻接矩阵, 'shd': SHD值}
        """
        try:
            hc = HillClimbSearch(data=df)
            model = hc.estimate(scoring_method=BicScore(data=df))
            
            sorted_nodes = sorted(df.columns)
            n = len(sorted_nodes)
            adjmat = np.zeros((n, n))
            
            for edge in model.edges():
                source_idx = sorted_nodes.index(edge[0])
                target_idx = sorted_nodes.index(edge[1])
                adjmat[source_idx, target_idx] = 1
            
            adjmat_df = pd.DataFrame(adjmat, index=sorted_nodes, columns=sorted_nodes)
            
            if nodes is not None:
                adjmat_df = adjmat_df.reindex(index=nodes, columns=nodes, fill_value=0)
            
            result = {'adjmat': adjmat_df}
            
            if true_adjmat is not None:
                result['shd'] = compute_shd(true_adjmat, adjmat_df)
            
            return result
            
        except Exception as e:
            print(f"HillClimb failed: {e}")
            return None


class Random_Baseline:
    """随机基线"""
    
    def run(self, n_nodes, n_edges, true_adjmat=None, nodes=None):
        """
        生成随机图
        
        Args:
            n_nodes: 节点数量
            n_edges: 边数量
            true_adjmat: 真实邻接矩阵（用于计算 SHD）
            nodes: 节点列表
        
        Returns:
            dict: {'adjmat': 预测邻接矩阵, 'shd': SHD值}
        """
        adjmat = np.zeros((n_nodes, n_nodes))
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
        
        if nodes is not None:
            adjmat_df = pd.DataFrame(adjmat, index=nodes, columns=nodes)
        else:
            adjmat_df = pd.DataFrame(adjmat)
        
        result = {'adjmat': adjmat_df}
        
        if true_adjmat is not None:
            result['shd'] = compute_shd(true_adjmat, adjmat_df)
        
        return result
