"""
Figure 7: Scalability and Robustness
"""

import numpy as np
import matplotlib.pyplot as plt

plt.rcParams['axes.unicode_minus'] = False
plt.style.use('seaborn-v0_8-whitegrid')

def main():
    fig, axes = plt.subplots(1, 2, figsize=(14, 5.5))
    
    ax1 = axes[0]
    
    networks = ['Sprinkler\n(4)', 'Asia\n(8)', 'Sachs\n(11)', 'Child\n(20)', 
                'Insurance\n(27)', 'Alarm\n(37)', 'Hailfinder\n(56)', 'Hepar II\n(70)']
    accuracy = [50.0, 62.5, 82.4, 76.0, 82.7, 89.1, 39.4, 86.2]
    
    ax1.plot(range(len(networks)), accuracy, 'o-', color='#E74C3C', linewidth=2.5, 
             markersize=10, markerfacecolor='white', markeredgewidth=2.5, label='ACR-Hybrid')
    
    for i, acc in enumerate(accuracy):
        offset = 10 if acc < 70 else -15
        va = 'bottom' if acc < 70 else 'top'
        color = '#E74C3C' if acc > 80 else '#7F8C8D'
        ax1.annotate(f'{acc:.1f}%',
                    xy=(i, acc), xytext=(0, offset), textcoords='offset points',
                    ha='center', va=va, fontsize=10, color=color, fontweight='bold')
    
    ax1.annotate('Complex weather\nrelationships',
                xy=(6, 39.4), xytext=(30, 30), textcoords='offset points',
                ha='left', fontsize=9, color='#7F8C8D',
                arrowprops=dict(arrowstyle='->', color='#7F8C8D', lw=1),
                bbox=dict(boxstyle='round', facecolor='#FEF9E7', edgecolor='#F39C12', alpha=0.8))
    
    ax1.axhline(y=50, color='gray', linestyle='--', linewidth=1.5, alpha=0.7, label='Random Baseline')
    
    ax1.set_xlabel('Network (nodes)', fontsize=12)
    ax1.set_ylabel('Edge Orientation Accuracy (%)', fontsize=12)
    ax1.set_title('(a) Scalability: Performance vs Network Size', fontsize=13, fontweight='bold')
    ax1.set_xticks(range(len(networks)))
    ax1.set_xticklabels(networks, fontsize=9, rotation=0)
    ax1.set_ylim(25, 100)
    ax1.legend(loc='lower left', fontsize=10)
    ax1.yaxis.grid(True, linestyle='--', alpha=0.7)
    ax1.set_axisbelow(True)
    
    ax2 = axes[1]
    
    sample_sizes = [100, 500, 1000]
    acr_hybrid = [0, 2, 2]
    pc = [11, 8, 8]
    hillclimb = [8, 14, 8]
    
    ax2.plot(sample_sizes, acr_hybrid, 'o-', color='#E74C3C', linewidth=2.5, 
             markersize=10, markerfacecolor='white', markeredgewidth=2.5, label='ACR-Hybrid')
    ax2.plot(sample_sizes, pc, 's--', color='#7F8C8D', linewidth=2, markersize=8, label='PC')
    ax2.plot(sample_sizes, hillclimb, '^--', color='#3498DB', linewidth=2, markersize=8, label='HillClimb')
    
    for i, shd in enumerate(acr_hybrid):
        ax2.annotate(f'{int(shd)}',
                    xy=(sample_sizes[i], shd), xytext=(0, 10), textcoords='offset points',
                    ha='center', va='bottom', fontsize=11, color='#E74C3C', fontweight='bold')
    
    ax2.annotate('Perfect\nReconstruction!',
                xy=(100, 0), xytext=(50, 30), textcoords='offset points',
                ha='left', fontsize=10, color='#27AE60', fontweight='bold',
                arrowprops=dict(arrowstyle='->', color='#27AE60', lw=1.5),
                bbox=dict(boxstyle='round', facecolor='#E8F8F5', edgecolor='#27AE60', alpha=0.8))
    
    ax2.set_xlabel('Sample Size (N)', fontsize=12)
    ax2.set_ylabel('SHD (lower is better)', fontsize=12)
    ax2.set_title('(b) Robustness: Performance vs Sample Size\n(Asia Network)', fontsize=13, fontweight='bold')
    ax2.set_xticks(sample_sizes)
    ax2.set_xticklabels(['N=100', 'N=500', 'N=1000'], fontsize=11)
    ax2.set_ylim(-1, 18)
    ax2.legend(loc='upper right', fontsize=10)
    ax2.yaxis.grid(True, linestyle='--', alpha=0.7)
    ax2.set_axisbelow(True)
    
    plt.tight_layout()
    plt.savefig('fig7_scalability_robustness.png', dpi=300, bbox_inches='tight')
    plt.savefig('fig7_scalability_robustness.pdf', bbox_inches='tight')
    print("Saved: fig7_scalability_robustness.png, fig7_scalability_robustness.pdf")
    
    import shutil, os
    if os.path.exists('../paper/assets'):
        shutil.copy('fig7_scalability_robustness.pdf', '../paper/assets/fig7_scalability_robustness.pdf')
        print("Copied to ../paper/assets/")

if __name__ == '__main__':
    main()
