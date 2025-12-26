"""
Methods module for causal discovery experiments.

包含论文中使用的所有方法：
- ACR-Hybrid: 侦探团 + 选择性 LLM 定向
  - PC + HillClimb + LiNGAM 三方投票
  - High Consensus: 直接采用共识方向
  - Conflict/Undirected: LLM 处理
- ACR-Hybrid-i: 侦探团 + 全部 LLM 定向（激进版本）
- PC: 传统约束类算法
- HillClimb: 传统评分类算法
- Random: 随机基线

以及工具函数：
- 网络加载
- 指标计算
- 侦探团集成骨架发现
"""

from .acr_hybrid import ACRHybrid
from .acr_hybrid_i import ACRHybridI
from .baselines import PC_Algorithm, HillClimb_Algorithm, Random_Baseline
from .data_loader import load_network, NETWORKS
from .metrics import compute_shd
from .skeleton_ensemble import SkeletonEnsemble, EdgeType, Strategy

__all__ = [
    'ACRHybrid',
    'ACRHybridI',
    'PC_Algorithm', 
    'HillClimb_Algorithm',
    'Random_Baseline',
    'load_network',
    'NETWORKS',
    'compute_shd',
    'SkeletonEnsemble',
    'EdgeType',
    'Strategy',
]
