"""
评估指标模块
"""

import numpy as np
import pandas as pd


def compute_shd(true_adjmat, pred_adjmat):
    """
    计算结构汉明距离 (Structural Hamming Distance)
    
    SHD = 边的增加 + 边的删除 + 边的反向
    
    Args:
        true_adjmat: 真实邻接矩阵 (DataFrame 或 ndarray)
        pred_adjmat: 预测邻接矩阵 (DataFrame 或 ndarray)
    
    Returns:
        int: SHD 值
    """
    if isinstance(true_adjmat, pd.DataFrame):
        true_adjmat = true_adjmat.values
    if isinstance(pred_adjmat, pd.DataFrame):
        pred_adjmat = pred_adjmat.values
    
    return int(np.sum(np.abs(true_adjmat - pred_adjmat)))


def compute_accuracy(correct_count, total_count):
    """计算准确率"""
    return correct_count / total_count if total_count > 0 else 0.0
