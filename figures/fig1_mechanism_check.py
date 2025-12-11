"""
Figure 1: Mechanism Check (机制验证图)
展示 ANM 数据的残差独立性差异，证明 LLM 看到的"证据"是真实存在的物理特征
"""

import numpy as np
import matplotlib.pyplot as plt
from sklearn.neural_network import MLPRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
import sys
sys.path.append('..')

# 设置中文字体和样式
plt.rcParams['font.family'] = ['Arial', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False
plt.style.use('seaborn-v0_8-whitegrid')

def generate_anm_data(n=1000, seed=42):
    """生成 ANM 数据: B = tanh(A) + 0.5*cos(A) + noise"""
    np.random.seed(seed)
    A = np.random.randn(n)
    noise = np.random.normal(0, 0.1, n)
    B = np.tanh(A) + 0.5 * np.cos(A) + noise
    return A, B

def fit_and_get_residuals(X, Y):
    """使用 MLP 拟合并返回残差"""
    X_2d = X.reshape(-1, 1)
    model = make_pipeline(
        StandardScaler(),
        MLPRegressor(hidden_layer_sizes=(100, 50), activation='tanh',
                     solver='lbfgs', max_iter=1000, random_state=42)
    )
    model.fit(X_2d, Y)
    Y_pred = model.predict(X_2d)
    residuals = Y - Y_pred
    r2 = model.score(X_2d, Y)
    return residuals, r2

def compute_hsic(X, Y, kernel_width='auto'):
    """计算 HSIC (Hilbert-Schmidt Independence Criterion)"""
    from scipy.spatial.distance import pdist, squareform
    
    n = len(X)
    X = X.reshape(-1, 1)
    Y = Y.reshape(-1, 1)
    
    if kernel_width == 'auto':
        median_x = np.median(pdist(X)) if len(X) > 1 else 1.0
        median_y = np.median(pdist(Y)) if len(Y) > 1 else 1.0
        kernel_width_x = median_x if median_x > 0 else 1.0
        kernel_width_y = median_y if median_y > 0 else 1.0
    else:
        kernel_width_x = kernel_width_y = kernel_width
    
    def rbf_kernel(x, width):
        pairwise_sq_dists = squareform(pdist(x, 'sqeuclidean'))
        return np.exp(-pairwise_sq_dists / (2 * width ** 2))
    
    K = rbf_kernel(X, kernel_width_x)
    L = rbf_kernel(Y, kernel_width_y)
    
    H = np.eye(n) - np.ones((n, n)) / n
    K_c = H @ K @ H
    L_c = H @ L @ H
    
    hsic = np.trace(K_c @ L_c) / (n ** 2)
    return min(hsic * 10, 1.0)

def main():
    # 生成数据
    A, B = generate_anm_data(n=1000)
    
    # 正向拟合: A -> B
    resid_ab, r2_ab = fit_and_get_residuals(A, B)
    hsic_ab = compute_hsic(A, resid_ab)
    
    # 反向拟合: B -> A
    resid_ba, r2_ba = fit_and_get_residuals(B, A)
    hsic_ba = compute_hsic(B, resid_ba)
    
    print(f"A->B: R²={r2_ab:.4f}, HSIC={hsic_ab:.4f}")
    print(f"B->A: R²={r2_ba:.4f}, HSIC={hsic_ba:.4f}")
    
    # 创建图表
    fig, axes = plt.subplots(1, 3, figsize=(14, 4))
    
    # 左图: 散点图 (非线性关系)
    ax1 = axes[0]
    ax1.scatter(A, B, alpha=0.4, s=10, c='steelblue', edgecolors='none')
    # 添加拟合曲线
    A_sorted = np.sort(A)
    B_true = np.tanh(A_sorted) + 0.5 * np.cos(A_sorted)
    ax1.plot(A_sorted, B_true, 'r-', linewidth=2, label='True: $B = tanh(A) + 0.5cos(A)$')
    ax1.set_xlabel('Variable A', fontsize=12)
    ax1.set_ylabel('Variable B', fontsize=12)
    ax1.set_title('(a) Non-linear Causal Relationship\n$A \\rightarrow B$', fontsize=13)
    ax1.legend(loc='upper left', fontsize=9)
    
    # 中图: 正向残差 (独立)
    ax2 = axes[1]
    ax2.scatter(A, resid_ab, alpha=0.4, s=10, c='forestgreen', edgecolors='none')
    ax2.axhline(y=0, color='gray', linestyle='--', linewidth=1)
    ax2.set_xlabel('Variable A (Predictor)', fontsize=12)
    ax2.set_ylabel('Residuals ($B - \\hat{B}$)', fontsize=12)
    ax2.set_title(f'(b) Forward Fit Residuals ($A \\rightarrow B$)\nHSIC = {hsic_ab:.4f} (Independent)', 
                  fontsize=13, color='darkgreen')
    
    # 右图: 反向残差 (不独立)
    ax3 = axes[2]
    ax3.scatter(B, resid_ba, alpha=0.4, s=10, c='crimson', edgecolors='none')
    ax3.axhline(y=0, color='gray', linestyle='--', linewidth=1)
    ax3.set_xlabel('Variable B (Predictor)', fontsize=12)
    ax3.set_ylabel('Residuals ($A - \\hat{A}$)', fontsize=12)
    ax3.set_title(f'(c) Reverse Fit Residuals ($B \\rightarrow A$)\nHSIC = {hsic_ba:.4f} (Dependent)', 
                  fontsize=13, color='darkred')
    
    plt.tight_layout()
    
    # 保存图片
    plt.savefig('fig1_mechanism_check.png', dpi=300, bbox_inches='tight')
    plt.savefig('fig1_mechanism_check.pdf', bbox_inches='tight')
    print("\nSaved: fig1_mechanism_check.png, fig1_mechanism_check.pdf")
    
    plt.show()

if __name__ == '__main__':
    main()
