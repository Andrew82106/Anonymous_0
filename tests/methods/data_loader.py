"""
数据加载模块 - 加载贝叶斯网络基准数据
"""

import numpy as np
import pandas as pd

# pgmpy 用于加载大部分网络
from pgmpy.utils import get_example_model
from pgmpy.sampling import BayesianModelSampling

# bnlearn 用于 sprinkler（pgmpy 不支持）
import bnlearn as bn


# 支持的网络配置
NETWORKS = [
    {'name': 'sprinkler', 'nodes': 4, 'source': 'bnlearn'},
    {'name': 'asia', 'nodes': 8, 'source': 'pgmpy'},
    {'name': 'sachs', 'nodes': 11, 'source': 'pgmpy'},
    {'name': 'child', 'nodes': 20, 'source': 'pgmpy'},
    {'name': 'insurance', 'nodes': 27, 'source': 'pgmpy'},
    {'name': 'alarm', 'nodes': 37, 'source': 'pgmpy'},
    {'name': 'hailfinder', 'nodes': 56, 'source': 'pgmpy'},
    {'name': 'hepar2', 'nodes': 70, 'source': 'pgmpy'},
]


def get_network_config(network_name):
    """获取网络配置（不区分大小写）"""
    network_name_lower = network_name.lower()
    for net in NETWORKS:
        if net['name'].lower() == network_name_lower:
            return net
    return None


def load_network(network_name, sample_size=1000):
    """
    加载贝叶斯网络并采样数据
    
    Args:
        network_name: 网络名称（不区分大小写）
        sample_size: 采样数量
    
    Returns:
        df: 采样数据 (DataFrame)
        true_adjmat: 真实邻接矩阵 (DataFrame)
        nodes: 节点列表
        edges: 边列表 [(source, target), ...]
    """
    # 转换为小写以匹配配置
    network_name_lower = network_name.lower()
    config = get_network_config(network_name_lower)
    if config is None:
        raise ValueError(f"Unknown network: {network_name}. Available: {[n['name'] for n in NETWORKS]}")
    
    # 使用配置中的标准名称
    canonical_name = config['name']
    
    if config['source'] == 'bnlearn':
        return _load_network_bnlearn(canonical_name, sample_size)
    else:
        return _load_network_pgmpy(canonical_name, sample_size)


def _load_network_bnlearn(network_name, sample_size=1000):
    """使用 bnlearn 加载网络"""
    dag = bn.import_DAG(network_name)
    df = bn.sampling(dag, n=sample_size, verbose=0)
    
    true_adjmat = dag['adjmat']
    nodes = list(true_adjmat.index)
    
    edges = []
    for source in true_adjmat.index:
        for target in true_adjmat.columns:
            if true_adjmat.loc[source, target] == 1:
                edges.append((source, target))
    
    return df, true_adjmat, nodes, edges


def _load_network_pgmpy(network_name, sample_size=1000):
    """使用 pgmpy 加载网络"""
    model = get_example_model(network_name)
    
    # 采样数据
    sampler = BayesianModelSampling(model)
    df = sampler.forward_sample(size=sample_size)
    
    # 构建邻接矩阵
    nodes = sorted(model.nodes())
    n_nodes = len(nodes)
    
    true_adjmat = pd.DataFrame(
        np.zeros((n_nodes, n_nodes)),
        index=nodes,
        columns=nodes
    )
    
    edges = list(model.edges())
    for source, target in edges:
        true_adjmat.loc[source, target] = 1
    
    return df, true_adjmat, nodes, edges
