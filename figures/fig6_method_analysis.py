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
    
    strategies = ['PC\n(基线)', '全量混合\n(所有边)', '保守混合\n(仅无向边)']
    shd_values = [32, 35, 27]
    f1_values = [0.61, 0.58, 0.64]
    
    x = np.arange(len(strategies))
    width = 0.35
    
    # 简洁配色：灰色系 + 强调色
    colors_shd = ['#A0A0A0', '#A0A0A0', '#D62728']  # 最优用红色
    colors_f1 = ['#C0C0C0', '#C0C0C0', '#FF7F7F']
    
    bars1 = ax1.bar(x - width/2, shd_values, width, label='SHD ↓', 
                    color=colors_shd, edgecolor='none')
    
    ax1_twin = ax1.twinx()
    bars2 = ax1_twin.bar(x + width/2, f1_values, width, label='F1 ↑', 
                         color=colors_f1, edgecolor='none')
    
    # 数值标注 - 简洁
    for bar, val in zip(bars1, shd_values):
        ax1.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.8,
                f'{int(val)}', ha='center', va='bottom', fontsize=11,
                fontweight='bold' if val == 27 else 'normal',
                color='#D62728' if val == 27 else '#333333')
    
    for bar, val in zip(bars2, f1_values):
        ax1_twin.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.015,
                     f'{val:.2f}', ha='center', va='bottom', fontsize=11,
                     fontweight='bold' if val == 0.64 else 'normal',
                     color='#D62728' if val == 0.64 else '#333333')
    
    ax1.set_xlabel('策略', fontsize=11)
    ax1.set_ylabel('SHD', fontsize=11)
    ax1_twin.set_ylabel('方向 F1', fontsize=11)
    ax1.set_title('(a) 保守策略有效性 (Alarm)', fontsize=12, fontweight='bold', pad=10)
    ax1.set_xticks(x)
    ax1.set_xticklabels(strategies, fontsize=9)
    ax1.set_ylim(0, 42)
    ax1_twin.set_ylim(0, 0.78)
    
    # 去掉网格线，只保留简洁的边框
    ax1.grid(False)
    ax1_twin.grid(False)
    ax1.spines['top'].set_visible(False)
    ax1_twin.spines['top'].set_visible(False)
    
    # 图例放在图内
    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax1_twin.get_legend_handles_labels()
    ax1.legend(lines1 + lines2, labels1 + labels2, loc='upper right', 
               fontsize=9, frameon=False)
    
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
