"""
Figure 5: Statistical Reasoning vs Semantic Memory
"""

import numpy as np
import matplotlib.pyplot as plt

plt.rcParams['axes.unicode_minus'] = False
plt.style.use('seaborn-v0_8-whitegrid')

def main():
    networks = ['Alarm\n(37)', 'Sachs\n(11)', 'Child\n(20)', 'Insurance\n(27)']
    
    data = {
        'ACR-Hybrid\n(Statistical)': [8, 4, 6, 9],
        'PromptBN\n(Semantic)': [41.8, None, None, 35.6],
        'ReActBN\n(Semantic)':  [35.4, None, 18.0, 40.2],
        'DiBS+GPT\n(Semantic)': [None, 21.7, None, None],
    }
    
    x = np.arange(len(networks))
    width = 0.2
    
    fig, ax = plt.subplots(figsize=(14, 7))
    
    colors = {
        'ACR-Hybrid\n(Statistical)': '#E74C3C',
        'PromptBN\n(Semantic)': '#3498DB',
        'ReActBN\n(Semantic)':  '#2ECC71',
        'DiBS+GPT\n(Semantic)': '#9B59B6',
    }
    
    bars = {}
    for i, (method, values) in enumerate(data.items()):
        offset = (i - 1.5) * width
        plot_values = [v if v is not None else 0 for v in values]
        bars[method] = ax.bar(x + offset, plot_values, width, label=method, 
                              color=colors[method], edgecolor='white', linewidth=0.5, alpha=0.9)
        for j, v in enumerate(values):
            if v is None:
                bars[method][j].set_height(0)
                bars[method][j].set_alpha(0)
    
    for method, bar_group in bars.items():
        for j, bar in enumerate(bar_group):
            height = bar.get_height()
            if height > 0:
                is_ours = 'ACR-Hybrid' in method
                ax.annotate(f'{height:.1f}' if isinstance(data[method][j], float) else f'{int(height)}',
                           xy=(bar.get_x() + bar.get_width() / 2, height),
                           xytext=(0, 3), textcoords='offset points',
                           ha='center', va='bottom', fontsize=10,
                           color='#E74C3C' if is_ours else 'black',
                           fontweight='bold' if is_ours else 'normal')
    
    improvement_labels = ['81%', '82%', '67%', '75%']
    for i, label in enumerate(improvement_labels):
        ax.annotate(label, 
                   xy=(x[i] - 1.5*width, data['ACR-Hybrid\n(Statistical)'][i]),
                   xytext=(0, -20), textcoords='offset points',
                   ha='center', va='top', fontsize=11,
                   color='#E74C3C', fontweight='bold',
                   bbox=dict(boxstyle='round,pad=0.3', facecolor='#FADBD8', edgecolor='#E74C3C', alpha=0.8))
    
    ax.set_xlabel('Benchmark Networks (nodes)', fontsize=14)
    ax.set_ylabel('Structural Hamming Distance (SHD)', fontsize=14)
    ax.set_title('Statistical Reasoning vs Semantic Memory\nACR-Hybrid outperforms SOTA on complex networks', 
                 fontsize=15, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels(networks, fontsize=12)
    ax.legend(loc='upper right', fontsize=11, framealpha=0.9)
    ax.yaxis.grid(True, linestyle='--', alpha=0.7)
    ax.set_axisbelow(True)
    
    max_val = max(v for values in data.values() for v in values if v is not None)
    ax.set_ylim(0, max_val * 1.2)
    
    plt.tight_layout()
    plt.savefig('fig5_blind_vs_semantic.png', dpi=300, bbox_inches='tight')
    plt.savefig('fig5_blind_vs_semantic.pdf', bbox_inches='tight')
    print("Saved: fig5_blind_vs_semantic.png, fig5_blind_vs_semantic.pdf")
    
    import shutil, os
    if os.path.exists('../paper/assets'):
        shutil.copy('fig5_blind_vs_semantic.pdf', '../paper/assets/fig5_blind_vs_semantic.pdf')
        print("Copied to ../paper/assets/")

if __name__ == '__main__':
    main()
