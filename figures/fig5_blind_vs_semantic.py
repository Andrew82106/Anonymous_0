"""
Figure 5: Statistical Reasoning vs Semantic Memory - 简洁顶刊风格
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
    
    networks = ['Alarm\n(37)', 'Sachs\n(11)', 'Child\n(20)', 'Insurance\n(27)']
    
    data = {
        'ACR-Hybrid': [8, 4, 6, 9],
        'PromptBN':   [41.8, None, None, 35.6],
        'ReActBN':    [35.4, None, 18.0, 40.2],
        'DiBS+GPT':   [None, 21.7, None, None],
    }
    
    x = np.arange(len(networks))
    width = 0.18
    
    fig, ax = plt.subplots(figsize=(10, 5))
    
    # 简洁配色
    colors = {
        'ACR-Hybrid': '#D62728',
        'PromptBN':   '#1F77B4',
        'ReActBN':    '#2CA02C',
        'DiBS+GPT':   '#9467BD',
    }
    
    for i, (method, values) in enumerate(data.items()):
        offset = (i - 1.5) * width
        plot_values = []
        positions = []
        for j, v in enumerate(values):
            if v is not None:
                plot_values.append(v)
                positions.append(x[j] + offset)
        if plot_values:
            ax.bar(positions, plot_values, width, label=method, 
                   color=colors[method], edgecolor='none')
    
    ax.set_xlabel('基准网络 (节点数)', fontsize=12)
    ax.set_ylabel('SHD', fontsize=12)
    ax.set_title('统计推理 vs 语义记忆', fontsize=13, fontweight='bold', pad=10)
    ax.set_xticks(x)
    ax.set_xticklabels(networks, fontsize=10)
    ax.legend(loc='upper right', fontsize=10, frameon=False)
    
    # 简洁样式
    ax.grid(False)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    
    max_val = max(v for values in data.values() for v in values if v is not None)
    ax.set_ylim(0, max_val * 1.15)
    
    plt.tight_layout()
    plt.savefig('fig5_blind_vs_semantic.png', dpi=300, bbox_inches='tight', facecolor='white')
    plt.savefig('fig5_blind_vs_semantic.pdf', bbox_inches='tight', facecolor='white')
    print("Saved: fig5_blind_vs_semantic.png, fig5_blind_vs_semantic.pdf")
    
    import shutil, os
    if os.path.exists('../paper/assets'):
        shutil.copy('fig5_blind_vs_semantic.pdf', '../paper/assets/fig5_blind_vs_semantic.pdf')
        print("Copied to ../paper/assets/")

if __name__ == '__main__':
    main()
