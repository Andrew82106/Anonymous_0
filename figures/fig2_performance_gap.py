"""
Figure 2: Performance Gap - 简洁顶刊风格
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
    
    networks = ['Sprinkler\n(4)', 'Asia\n(8)', 'Sachs\n(11)', 'Child\n(20)', 
                'Insurance\n(27)', 'Alarm\n(37)', 'Hailfinder\n(56)', 'Hepar II\n(70)']
    
    data = {
        'ACR-Hybrid': [3, 5, 4, 6, 9, 8, 40, 17],
        'PC':         [0, 12, 29, 14, 39, 75, 132, 117],
        'HillClimb':  [2, 16, 24, 16, 31, 85, 132, 100],
        'Random':     [6, 16, 30, 50, 104, 84, 132, 246],
    }
    
    x = np.arange(len(networks))
    width = 0.2
    
    fig, ax = plt.subplots(figsize=(14, 5))
    
    # 简洁配色
    colors = {
        'ACR-Hybrid': '#D62728',  # 红色强调
        'PC':         '#7F7F7F',  # 灰色
        'HillClimb':  '#1F77B4',  # 蓝色
        'Random':     '#C7C7C7',  # 浅灰
    }
    
    for i, (method, values) in enumerate(data.items()):
        offset = (i - 1.5) * width
        ax.bar(x + offset, values, width, label=method, 
               color=colors[method], edgecolor='none')
    
    ax.set_xlabel('基准网络 (节点数)', fontsize=12)
    ax.set_ylabel('SHD', fontsize=12)
    ax.set_title('真实贝叶斯网络基准性能对比', fontsize=13, fontweight='bold', pad=10)
    ax.set_xticks(x)
    ax.set_xticklabels(networks, fontsize=9)
    ax.legend(loc='upper left', fontsize=10, frameon=False)
    
    # 简洁样式
    ax.grid(False)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.set_ylim(0, max(max(v) for v in data.values()) * 1.1)
    
    plt.tight_layout()
    plt.savefig('fig2_performance_gap.png', dpi=300, bbox_inches='tight', facecolor='white')
    plt.savefig('fig2_performance_gap.pdf', bbox_inches='tight', facecolor='white')
    print("Saved: fig2_performance_gap.png, fig2_performance_gap.pdf")
    
    import shutil, os
    if os.path.exists('../paper/assets'):
        shutil.copy('fig2_performance_gap.pdf', '../paper/assets/fig2_performance_gap.pdf')
        print("Copied to ../paper/assets/")

if __name__ == '__main__':
    main()
