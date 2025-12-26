"""
Figure 6: Method Analysis - 简洁顶刊风格
"""

import numpy as np
import matplotlib.pyplot as plt

# 设置中文字体 (macOS 简体中文)
plt.rcParams['font.sans-serif'] = ['Hiragino Sans GB', 'PingFang HK', 'Heiti TC', 'Arial Unicode MS']
plt.rcParams['axes.unicode_minus'] = False

def main():
    # 使用简洁的白色背景
    plt.rcParams['axes.facecolor'] = 'white'
    plt.rcParams['figure.facecolor'] = 'white'
    plt.rcParams['axes.edgecolor'] = '#333333'
    plt.rcParams['axes.linewidth'] = 0.8
    
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))
    
    # ========== 左图: 保守策略有效性 ==========
    ax1 = axes[0]
    
    # 更新数据：使用 real_network_results.json 中的 Alarm 数据
    # PC baseline: SHD=75, ACR-Hybrid: SHD=8
    strategies = ['PC\n(基线)', 'ACR-Hybrid\n(保守策略)']
    shd_values = [75, 8]
    
    x = np.arange(len(strategies))
    width = 0.5
    
    # 简洁配色：灰色系 + 强调色
    colors_shd = ['#A0A0A0', '#D62728']  # 最优用红色
    
    bars1 = ax1.bar(x, shd_values, width, label='SHD ↓', 
                    color=colors_shd, edgecolor='none')
    
    # 数值标注 - 简洁
    for bar, val in zip(bars1, shd_values):
        ax1.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 1.5,
                f'{int(val)}', ha='center', va='bottom', fontsize=12,
                fontweight='bold' if val == 8 else 'normal',
                color='#D62728' if val == 8 else '#333333')
    
    ax1.set_xlabel('方法', fontsize=11)
    ax1.set_ylabel('SHD', fontsize=11)
    ax1.set_title('(a) Alarm网络性能对比', fontsize=12, fontweight='bold', pad=10)
    ax1.set_xticks(x)
    ax1.set_xticklabels(strategies, fontsize=10)
    ax1.set_ylim(0, 90)
    
    # 去掉网格线，只保留简洁的边框
    ax1.grid(False)
    ax1.spines['top'].set_visible(False)
    ax1.spines['right'].set_visible(False)
    
    # ========== 右图: 基础算法通用性 ==========
    ax2 = axes[1]
    
    methods = ['PC', 'MMHC', 'PC+ACR', 'MMHC+ACR']
    shd_base = [12, 12, 10, 10]
    
    # 简洁配色：原始方法灰色，+ACR方法红色
    colors = ['#A0A0A0', '#A0A0A0', '#D62728', '#D62728']
    
    bars = ax2.bar(methods, shd_base, color=colors, edgecolor='none', width=0.6)
    
    # 数值标注
    for bar, val in zip(bars, shd_base):
        ax2.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.3,
                f'{int(val)}', ha='center', va='bottom', fontsize=12,
                fontweight='bold' if val == 10 else 'normal',
                color='#D62728' if val == 10 else '#333333')
    
    ax2.set_xlabel('方法', fontsize=11)
    ax2.set_ylabel('SHD', fontsize=11)
    ax2.set_title('(b) 基座算法通用性 (Asia)', fontsize=12, fontweight='bold', pad=10)
    ax2.set_ylim(0, 15)
    
    # 去掉网格线
    ax2.grid(False)
    ax2.spines['top'].set_visible(False)
    ax2.spines['right'].set_visible(False)
    
    plt.tight_layout()
    plt.savefig('fig6_method_analysis.png', dpi=300, bbox_inches='tight', facecolor='white')
    plt.savefig('fig6_method_analysis.pdf', bbox_inches='tight', facecolor='white')
    print("Saved: fig6_method_analysis.png, fig6_method_analysis.pdf")
    
    import shutil, os
    if os.path.exists('../paper/assets'):
        shutil.copy('fig6_method_analysis.pdf', '../paper/assets/fig6_method_analysis.pdf')
        shutil.copy('fig6_method_analysis.png', '../paper/assets/fig6_method_analysis.png')
        print("Copied to ../paper/assets/")

if __name__ == '__main__':
    main()
