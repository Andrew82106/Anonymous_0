"""
数据生成模块 (Data Generator)
用于生成各种因果结构的合成数据集，用于测试 ACR 框架
"""

import numpy as np
import pandas as pd
from typing import Dict, Tuple, List

class CausalDataGenerator:
    def __init__(self, random_seed=42):
        self.random_seed = random_seed
        np.random.seed(random_seed)
    
    def generate_dataset(self, dataset_type: str, n_samples: int = 500) -> Tuple[np.ndarray, np.ndarray, str, str]:
        """
        生成指定类型的数据集
        
        Parameters:
        -----------
        dataset_type : str
            数据集类型：'lingam', 'anm', 'confounder', 'independent', 'reverse'
        n_samples : int
            样本数量
            
        Returns:
        --------
        X, Y : np.ndarray
            两个变量的数据
        ground_truth : str
            真实因果方向 ('A->B', 'B->A', 'A<-Z->B', 'A_|_B')
        description : str
            数据集描述
        """
        
        if dataset_type == 'lingam':
            return self._generate_lingam(n_samples)
        elif dataset_type == 'anm':
            return self._generate_anm(n_samples)
        elif dataset_type == 'confounder':
            return self._generate_confounder(n_samples)
        elif dataset_type == 'independent':
            return self._generate_independent(n_samples)
        elif dataset_type == 'reverse':
            return self._generate_reverse(n_samples)
        else:
            raise ValueError(f"Unknown dataset type: {dataset_type}")
    
    def _generate_lingam(self, n: int) -> Tuple[np.ndarray, np.ndarray, str, str]:
        """
        LiNGAM Case: Linear Non-Gaussian Additive Noise Model
        A -> B: B = a*A + noise_uniform
        """
        A = np.random.uniform(-1, 1, n)
        noise = np.random.uniform(-0.5, 0.5, n)
        B = 0.8 * A + noise
        
        return A, B, 'A->B', 'Linear Non-Gaussian (LiNGAM): A causes B with uniform noise'
    
    def _generate_anm(self, n: int) -> Tuple[np.ndarray, np.ndarray, str, str]:
        """
        ANM Case: Additive Noise Model with Non-linear function
        A -> B: B = f(A) + noise_gaussian
        """
        A = np.random.randn(n)
        noise = np.random.normal(0, 0.1, n)
        # 复杂的非线性函数
        B = np.tanh(A) + 0.5 * np.cos(A) + noise
        
        return A, B, 'A->B', 'Non-linear ANM: B = tanh(A) + 0.5*cos(A) + noise'
    
    def _generate_confounder(self, n: int) -> Tuple[np.ndarray, np.ndarray, str, str]:
        """
        Confounder Case: Z -> A, Z -> B
        A 和 B 没有直接因果关系，但由于共同原因 Z 而相关
        """
        Z = np.random.randint(0, 2, n)  # 二元混杂变量
        A = np.random.normal(0, 0.5, n) + 3 * Z
        B = np.random.normal(0, 0.5, n) + 3 * Z
        
        return A, B, 'A<-Z->B', 'Confounder: Both A and B are caused by hidden variable Z'
    
    def _generate_independent(self, n: int) -> Tuple[np.ndarray, np.ndarray, str, str]:
        """
        Independent Case: A _|_ B
        两个变量完全独立
        """
        A = np.random.randn(n)
        B = np.random.randn(n)
        
        return A, B, 'A_|_B', 'Independent: A and B are statistically independent'
    
    def _generate_reverse(self, n: int) -> Tuple[np.ndarray, np.ndarray, str, str]:
        """
        Reverse Case: B -> A (instead of A -> B)
        用于测试模型是否会被变量名顺序误导
        """
        B = np.random.exponential(1, n)  # 非高斯分布
        noise = np.random.uniform(-0.3, 0.3, n)
        A = 1.2 * B + noise
        
        return A, B, 'B->A', 'Reverse Causation: B causes A (exponential + uniform noise)'
    
    def generate_batch(self, n_samples: int = 500) -> List[Dict]:
        """
        生成一批测试数据集
        
        Returns:
        --------
        datasets : List[Dict]
            每个字典包含：{
                'name': str,
                'X': np.ndarray,
                'Y': np.ndarray,
                'ground_truth': str,
                'description': str
            }
        """
        dataset_types = ['lingam', 'anm', 'confounder', 'independent', 'reverse']
        datasets = []
        
        for dtype in dataset_types:
            X, Y, truth, desc = self.generate_dataset(dtype, n_samples)
            datasets.append({
                'name': dtype,
                'X': X,
                'Y': Y,
                'ground_truth': truth,
                'description': desc
            })
        
        return datasets
    
    def to_dataframe(self, datasets: List[Dict]) -> pd.DataFrame:
        """
        将数据集批次转换为 DataFrame 格式（用于保存或查看）
        
        Returns:
        --------
        pd.DataFrame with columns: name, ground_truth, description, n_samples
        """
        summary = []
        for ds in datasets:
            summary.append({
                'name': ds['name'],
                'ground_truth': ds['ground_truth'],
                'description': ds['description'],
                'n_samples': len(ds['X'])
            })
        return pd.DataFrame(summary)


class BNLearnDataLoader:
    """
    使用 bnlearn 库加载真实的贝叶斯网络基准数据
    """
    def __init__(self):
        try:
            import bnlearn as bn
            self.bn = bn
            self.available = True
        except ImportError:
            print("Warning: bnlearn is not installed. Install with: pip install bnlearn")
            self.available = False
    
    def load_network(self, network_name: str = 'sprinkler', n_samples: int = 1000):
        """
        从 bnlearn 加载预定义的贝叶斯网络并采样数据
        
        Parameters:
        -----------
        network_name : str
            网络名称，如 'sprinkler', 'asia', 'alarm' 等
        n_samples : int
            采样数量
            
        Returns:
        --------
        df : pd.DataFrame
            采样的数据
        dag : dict
            DAG 结构
        """
        if not self.available:
            raise RuntimeError("bnlearn is not available")
        
        # 加载预定义的 DAG
        dag = self.bn.import_DAG(network_name)
        
        # 从 DAG 采样数据
        df = self.bn.sampling(dag, n=n_samples)
        
        return df, dag
    
    def extract_pairwise_data(self, df: pd.DataFrame, node_a: str, node_b: str) -> Tuple[np.ndarray, np.ndarray]:
        """
        从 DataFrame 中提取成对的变量数据
        
        Parameters:
        -----------
        df : pd.DataFrame
            包含所有变量的数据
        node_a, node_b : str
            要提取的两个节点名称
            
        Returns:
        --------
        X, Y : np.ndarray
            脱敏后的数据（作为 Var_A 和 Var_B）
        """
        X = df[node_a].values
        Y = df[node_b].values
        return X, Y
    
    def get_ground_truth(self, dag, node_a: str, node_b: str) -> str:
        """
        从 DAG 结构中获取两个节点之间的真实因果关系
        
        Returns:
        --------
        ground_truth : str
            'A->B', 'B->A', 'A_|_B', 或 'A<-Z->B' (如果有共同父节点)
        """
        # 获取邻接矩阵
        adjmat = dag['adjmat']
        
        # 检查是否存在边
        if node_a in adjmat.index and node_b in adjmat.columns:
            if adjmat.loc[node_a, node_b] == 1:
                return 'A->B'
            elif adjmat.loc[node_b, node_a] == 1:
                return 'B->A'
        
        # 检查是否有共同父节点（简化版）
        # 这里可以进一步扩展来检测更复杂的关系
        return 'A_|_B'
