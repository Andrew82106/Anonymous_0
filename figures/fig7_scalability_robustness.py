"""
Figure 7: Scalability and Robustness - 简洁顶刊风格
"""

import numpy as np
import matplotlib.pyplot as plt

# 设置中文字体 (macOS 简体中文)
plt.rcParams['font.sans-serif'] = ['Hiragino Sans GB', 'PingFang HK', 'Heiti TC', 'Arial Unicode MS']
plt.rcParams['axes.unicode_minus'] = False

def main():
    plt.rcParams['axes.facecolor'] = 'white'
    plt.rcParams['figure.facecolor'] = 'white'
    plt.rcParams['axes.edgecolor'] = '#333333'
    plt.rcParams['axes.linewidth'] = 0.8
    
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))
    
    # ========== 左图: 可扩展性 ==========
    ax1 = axes[0]
    
    networks = ['Sprinkler\n(4)', 'Asia\n(8)', 'Sachs\n(11)', 'Child\n(20)', 
                'Insurance\n(27)', 'Alarm\n(37)', 'Hailfinder\n(56)', 'Hepar II\n(70)']
    accuracy = [50.0, 62.5, 82.4, 76.0, 82.7, 89.1, 39.4, 86.2]
    
    ax1.plot(range(len(networks)), accuracy, 'o-', color='#D62728', linewidth=2, 
             markersize=8, markerfacecolor='white', markeredgewidth=2)
    
    # 随机基线
    ax1.axhline(y=50, color='#999999', linestyle='--', linewidth=1, label='随机基线')
    
    ax1.set_xlabel('网络 (节点数)', fontsize=11)
    ax1.set_ylabel('边方向准确率 (%)', fontsize=11)
    ax1.set_title('(a) 可扩展性: 网络规模 vs 准确率', fontsize=12, fontweight='bold', pad=10)
    ax1.set_xticks(range(len(networks)))
    ax1.set_xticklabels(networks, fontsize=8, rotation=0)
    ax1.set_ylim(25, 100)
    ax1.legend(loc='lower left', fontsize=9, frameon=False)
    
    ax1.grid(False)
    ax1.spines['top'].set_visible(False)
    ax1.spines['right'].set_visible(False)
    
    # ========== 右图: 样本量鲁棒性 ==========
    ax2 = axes[1]
    
    sample_sizes = [100, 500, 1000]
    acr_hybrid = [0, 2, 2]
    pc = [11, 8, 8]
    hillclimb = [8, 14, 8]
    
    ax2.plot(sample_sizes, acr_hybrid, 'o-', color='#D62728', linewidth=2, 
             markersize=8, markerfacecolor='white', markeredgewidth=2, label='ACR-Hybrid')
    ax2.plot(sample_sizes, pc, 's--', color='#7F7F7F', linewidth=1.5, markersize=6, label='PC')
    ax2.plot(sample_sizes, hillclimb, '^--', color='#1F77B4', linewidth=1.5, markersize=6, label='HillClimb')
    
    ax2.set_xlabel('样本量', fontsize=11)
    ax2.set_ylabel('SHD', fontsize=11)
    ax2.set_title('(b) 鲁棒性: 样本量 vs SHD (Asia)', fontsize=12, fontweight='bold', pad=10)
    ax2.set_xticks(sample_sizes)
    ax2.set_xticklabels(['N=100', 'N=500', 'N=1000'], fontsize=10)
    ax2.set_ylim(-1, 16)
    ax2.legend(loc='upper right', fontsize=9, frameon=False)
    
    ax2.grid(False)
    ax2.spines['top'].set_visible(False)
    ax2.spines['right'].set_visible(False)
    
    plt.tight_layout()
    plt.savefig('fig7_scalability_robustness.png', dpi=300, bbox_inches='tight', facecolor='white')
    plt.savefig('fig7_scalability_robustness.pdf', bbox_inches='tight', facecolor='white')
    print("Saved: fig7_scalability_robustness.png, fig7_scalability_robustness.pdf")
    
    import shutil, os
    if os.path.exists('../paper/assets'):
        shutil.copy('fig7_scalability_robustness.pdf', '../paper/assets/fig7_scalability_robustness.pdf')
        print("Copied to ../paper/assets/")

if __name__ == '__main__':
    main()
