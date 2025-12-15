"""
Figure 6: Method Analysis
"""

import numpy as np
import matplotlib.pyplot as plt

plt.rcParams['axes.unicode_minus'] = False
plt.style.use('seaborn-v0_8-whitegrid')

def main():
    fig, axes = plt.subplots(1, 2, figsize=(14, 5.5))
    
    ax1 = axes[0]
    
    strategies = ['PC\n(Baseline)', 'Full Hybrid\n(All Edges)', 'Conservative\nHybrid']
    shd_values = [32, 35, 27]
    f1_values = [0.61, 0.58, 0.64]
    
    x = np.arange(len(strategies))
    width = 0.35
    colors_shd = ['#7F8C8D', '#3498DB', '#E74C3C']
    colors_f1 = ['#95A5A6', '#5DADE2', '#F1948A']
    
    bars1 = ax1.bar(x - width/2, shd_values, width, label='SHD', 
                    color=colors_shd, edgecolor='white', linewidth=1)
    
    ax1_twin = ax1.twinx()
    bars2 = ax1_twin.bar(x + width/2, f1_values, width, label='Orient F1', 
                         color=colors_f1, edgecolor='white', linewidth=1, alpha=0.8)
    
    for bar, val in zip(bars1, shd_values):
        height = bar.get_height()
        color = '#E74C3C' if val == min(shd_values) else 'black'
        weight = 'bold' if val == min(shd_values) else 'normal'
        ax1.annotate(f'{int(val)}',
                    xy=(bar.get_x() + bar.get_width() / 2, height),
                    xytext=(0, 3), textcoords='offset points',
                    ha='center', va='bottom', fontsize=11, color=color, fontweight=weight)
    
    for bar, val in zip(bars2, f1_values):
        height = bar.get_height()
        color = '#E74C3C' if val == max(f1_values) else 'black'
        weight = 'bold' if val == max(f1_values) else 'normal'
        ax1_twin.annotate(f'{val:.2f}',
                         xy=(bar.get_x() + bar.get_width() / 2, height),
                         xytext=(0, 3), textcoords='offset points',
                         ha='center', va='bottom', fontsize=11, color=color, fontweight=weight)
    
    ax1.annotate('* Best', xy=(2 - width/2, 27), xytext=(-30, 15), 
                textcoords='offset points', ha='center', fontsize=10,
                color='#E74C3C', fontweight='bold',
                arrowprops=dict(arrowstyle='->', color='#E74C3C', lw=1.5))
    
    ax1.set_xlabel('Strategy', fontsize=12)
    ax1.set_ylabel('SHD (lower is better)', fontsize=12, color='#2C3E50')
    ax1_twin.set_ylabel('Orientation F1 (higher is better)', fontsize=12, color='#7B7D7D')
    ax1.set_title('(a) Conservative Strategy Effectiveness\n(Alarm Network)', fontsize=13, fontweight='bold')
    ax1.set_xticks(x)
    ax1.set_xticklabels(strategies, fontsize=10)
    ax1.set_ylim(0, 45)
    ax1_twin.set_ylim(0, 0.85)
    
    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax1_twin.get_legend_handles_labels()
    ax1.legend(lines1 + lines2, labels1 + labels2, loc='upper left', fontsize=9)
    
    ax2 = axes[1]
    
    methods = ['PC\n(Original)', 'MMHC\n(Original)', 'PC + ACR', 'MMHC + ACR']
    shd_base = [12, 12, 10, 10]
    colors = ['#7F8C8D', '#95A5A6', '#E74C3C', '#C0392B']
    
    bars = ax2.bar(methods, shd_base, color=colors, edgecolor='white', linewidth=1, width=0.6)
    
    for bar, val in zip(bars, shd_base):
        height = bar.get_height()
        is_best = val == min(shd_base)
        ax2.annotate(f'{int(val)}',
                    xy=(bar.get_x() + bar.get_width() / 2, height),
                    xytext=(0, 3), textcoords='offset points',
                    ha='center', va='bottom', fontsize=12,
                    color='#E74C3C' if is_best else 'black',
                    fontweight='bold' if is_best else 'normal')
    
    ax2.annotate('', xy=(2, 10), xytext=(0, 12),
                arrowprops=dict(arrowstyle='->', color='#27AE60', lw=2))
    ax2.text(1, 11.5, '-17%', ha='center', fontsize=11, color='#27AE60', fontweight='bold')
    
    ax2.annotate('', xy=(3, 10), xytext=(1, 12),
                arrowprops=dict(arrowstyle='->', color='#27AE60', lw=2))
    ax2.text(2, 11.5, '-17%', ha='center', fontsize=11, color='#27AE60', fontweight='bold')
    
    ax2.set_xlabel('Method', fontsize=12)
    ax2.set_ylabel('SHD (lower is better)', fontsize=12)
    ax2.set_title('(b) Base Algorithm Generality\n(Asia Network)', fontsize=13, fontweight='bold')
    ax2.set_ylim(0, 16)
    ax2.yaxis.grid(True, linestyle='--', alpha=0.7)
    ax2.set_axisbelow(True)
    
    plt.tight_layout()
    plt.savefig('fig6_method_analysis.png', dpi=300, bbox_inches='tight')
    plt.savefig('fig6_method_analysis.pdf', bbox_inches='tight')
    print("Saved: fig6_method_analysis.png, fig6_method_analysis.pdf")
    
    import shutil, os
    if os.path.exists('../paper/assets'):
        shutil.copy('fig6_method_analysis.pdf', '../paper/assets/fig6_method_analysis.pdf')

if __name__ == '__main__':
    main()
