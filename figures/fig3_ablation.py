"""
Figure 3: Ablation Study
"""

import numpy as np
import matplotlib.pyplot as plt

plt.rcParams['axes.unicode_minus'] = False
plt.style.use('seaborn-v0_8-whitegrid')

def main():
    settings = ['Full ACR\n(with Narrative)', 'Raw Data\n(no Narrative)', 'Random\nGuess']
    accuracy = [89.1, 67.4, 50.0]
    colors = ['#E74C3C', '#3498DB', '#BDC3C7']
    
    fig, ax = plt.subplots(figsize=(8, 6))
    bars = ax.bar(settings, accuracy, color=colors, edgecolor='white', linewidth=2, width=0.6)
    
    for bar, acc in zip(bars, accuracy):
        height = bar.get_height()
        ax.annotate(f'{acc:.1f}%',
                   xy=(bar.get_x() + bar.get_width() / 2, height),
                   xytext=(0, 5), textcoords='offset points',
                   ha='center', va='bottom', fontsize=14, fontweight='bold',
                   color='#2C3E50')
    
    ax.axhline(y=50, color='gray', linestyle='--', linewidth=1.5, alpha=0.7, label='Random Baseline (50%)')
    
    ax.annotate('', xy=(0, 89.1), xytext=(1, 67.4),
               arrowprops=dict(arrowstyle='<->', color='#27AE60', lw=2))
    ax.text(0.5, 78, '+21.7%', ha='center', fontsize=12, color='#27AE60', fontweight='bold')
    
    ax.annotate('', xy=(1, 67.4), xytext=(2, 50),
               arrowprops=dict(arrowstyle='<->', color='#F39C12', lw=2))
    ax.text(1.5, 58, '+17.4%', ha='center', fontsize=12, color='#F39C12', fontweight='bold')
    
    ax.set_ylabel('Edge Orientation Accuracy (%)', fontsize=13)
    ax.set_title('Ablation Study: StatTranslator Contribution\n(Alarm Network, 46 edges)', 
                 fontsize=14, fontweight='bold')
    ax.set_ylim(0, 105)
    ax.legend(loc='lower right', fontsize=10)
    
    plt.tight_layout()
    plt.savefig('fig3_ablation.png', dpi=300, bbox_inches='tight')
    plt.savefig('fig3_ablation.pdf', bbox_inches='tight')
    print("Saved: fig3_ablation.png, fig3_ablation.pdf")
    
    import shutil, os
    if os.path.exists('../paper/assets'):
        shutil.copy('fig3_ablation.pdf', '../paper/assets/fig3_ablation.pdf')
        print("Copied to ../paper/assets/")

if __name__ == '__main__':
    main()
