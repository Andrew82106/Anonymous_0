"""
Figure 2: Performance Gap (SOTA 对比图)
分组柱状图展示 ACR-Hybrid 在各数据集上的 SHD 优势
"""

import numpy as np
import matplotlib.pyplot as plt

# 设置样式
plt.rcParams['font.family'] = ['Arial', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False
plt.style.use('seaborn-v0_8-whitegrid')

def main():
    # 数据来源: results/real_network_results.json
    networks = ['Asia\n(8 nodes)', 'Sprinkler\n(4 nodes)', 'Alarm\n(37 nodes)', 'Sachs\n(11 nodes)']
    
    # SHD 数据 (越低越好)
    data = {
        'ACR-Hybrid': [5, 3, 8, 4],      # LLM-ACR 结果
        'PC':         [12, 0, 75, 29],    # PC 算法
        'HillClimb':  [16, 2, 85, 24],    # 爬山算法
        'Random':     [16, 6, 84, 30],    # 随机猜测
    }
    
    x = np.arange(len(networks))
    width = 0.2
    
    fig, ax = plt.subplots(figsize=(12, 6))
    
    # 颜色方案
    colors = {
        'ACR-Hybrid': '#E74C3C',  # 红色 (我们的方法)
        'PC':         '#7F8C8D',  # 灰色
        'HillClimb':  '#3498DB',  # 蓝色
        'Random':     '#BDC3C7',  # 浅灰色
    }
    
    # 绘制柱状图
    bars = {}
    for i, (method, values) in enumerate(data.items()):
        offset = (i - 1.5) * width
        bars[method] = ax.bar(x + offset, values, width, label=method, 
                              color=colors[method], edgecolor='white', linewidth=0.5)
    
    # 在 ACR-Hybrid 柱子上添加星号标记 (表示最佳)
    for i, (network, acr_val) in enumerate(zip(networks, data['ACR-Hybrid'])):
        # 检查是否是该网络的最佳结果
        all_vals = [data[m][i] for m in data.keys()]
        if acr_val == min(all_vals):
            ax.annotate('*', xy=(x[i] - 1.5*width, acr_val), 
                       xytext=(0, 5), textcoords='offset points',
                       ha='center', fontsize=16, color='#E74C3C', fontweight='bold')
    
    # 添加数值标签
    for method, bar_group in bars.items():
        for bar in bar_group:
            height = bar.get_height()
            ax.annotate(f'{int(height)}',
                       xy=(bar.get_x() + bar.get_width() / 2, height),
                       xytext=(0, 3), textcoords='offset points',
                       ha='center', va='bottom', fontsize=9,
                       color='black' if method != 'ACR-Hybrid' else '#E74C3C',
                       fontweight='bold' if method == 'ACR-Hybrid' else 'normal')
    
    # 设置坐标轴
    ax.set_xlabel('Benchmark Networks', fontsize=13)
    ax.set_ylabel('Structural Hamming Distance (SHD) ↓', fontsize=13)
    ax.set_title('Performance Comparison: ACR-Hybrid vs Traditional Algorithms\n(Lower SHD = Better)', 
                 fontsize=14, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels(networks, fontsize=11)
    ax.legend(loc='upper right', fontsize=10)
    
    # 添加网格线
    ax.yaxis.grid(True, linestyle='--', alpha=0.7)
    ax.set_axisbelow(True)
    
    # 设置 y 轴范围
    ax.set_ylim(0, max(max(v) for v in data.values()) * 1.15)
    
    plt.tight_layout()
    
    # 保存图片
    plt.savefig('fig2_performance_gap.png', dpi=300, bbox_inches='tight')
    plt.savefig('fig2_performance_gap.pdf', bbox_inches='tight')
    print("Saved: fig2_performance_gap.png, fig2_performance_gap.pdf")
    
    plt.show()

if __name__ == '__main__':
    main()
