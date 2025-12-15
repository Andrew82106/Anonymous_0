"""
Figure 2: Performance Gap (性能对比图)
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm

# 查找系统中的中文字体
def get_chinese_font():
    fonts = ['PingFang SC', 'Heiti SC', 'STHeiti', 'Songti SC', 'Hiragino Sans GB']
    for font in fonts:
        try:
            fp = fm.FontProperties(family=font)
            if fm.findfont(fp) != fm.findfont(fm.FontProperties()):
                return font
        except:
            continue
    return None

chinese_font = get_chinese_font()
if chinese_font:
    plt.rcParams['font.family'] = chinese_font
plt.rcParams['axes.unicode_minus'] = False
plt.style.use('seaborn-v0_8-whitegrid')

def main():
    # 使用英文标签避免字体问题
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
    
    fig, ax = plt.subplots(figsize=(16, 7))
    
    colors = {
        'ACR-Hybrid': '#E74C3C',
        'PC':         '#7F8C8D',
        'HillClimb':  '#3498DB',
        'Random':     '#BDC3C7',
    }
    
    bars = {}
    for i, (method, values) in enumerate(data.items()):
        offset = (i - 1.5) * width
        bars[method] = ax.bar(x + offset, values, width, label=method, 
                              color=colors[method], edgecolor='white', linewidth=0.5)
    
    for i, acr_val in enumerate(data['ACR-Hybrid']):
        all_vals = [data[m][i] for m in data.keys()]
        if acr_val == min(all_vals):
            ax.annotate('*', xy=(x[i] - 1.5*width, acr_val), 
                       xytext=(0, 5), textcoords='offset points',
                       ha='center', fontsize=16, color='#E74C3C', fontweight='bold')
    
    for method, bar_group in bars.items():
        for bar in bar_group:
            height = bar.get_height()
            ax.annotate(f'{int(height)}',
                       xy=(bar.get_x() + bar.get_width() / 2, height),
                       xytext=(0, 3), textcoords='offset points',
                       ha='center', va='bottom', fontsize=9,
                       color='black' if method != 'ACR-Hybrid' else '#E74C3C',
                       fontweight='bold' if method == 'ACR-Hybrid' else 'normal')
    
    ax.set_xlabel('Benchmark Networks (nodes)', fontsize=13)
    ax.set_ylabel('Structural Hamming Distance (SHD)', fontsize=13)
    ax.set_title('ACR-Hybrid vs Traditional Algorithms (Lower is Better)', 
                 fontsize=14, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels(networks, fontsize=10)
    ax.legend(loc='upper right', fontsize=10)
    ax.yaxis.grid(True, linestyle='--', alpha=0.7)
    ax.set_axisbelow(True)
    ax.set_ylim(0, max(max(v) for v in data.values()) * 1.15)
    
    plt.tight_layout()
    plt.savefig('fig2_performance_gap.png', dpi=300, bbox_inches='tight')
    plt.savefig('fig2_performance_gap.pdf', bbox_inches='tight')
    print("Saved: fig2_performance_gap.png, fig2_performance_gap.pdf")
    
    import shutil, os
    if os.path.exists('../paper/assets'):
        shutil.copy('fig2_performance_gap.pdf', '../paper/assets/fig2_performance_gap.pdf')
        print("Copied to ../paper/assets/")

if __name__ == '__main__':
    main()
