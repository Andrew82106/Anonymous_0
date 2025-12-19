"""
Figure 4: Framework Overview (框架图)
ACR 流水线架构图

Note: This figure can also be created using Banana diagram tool with the following prompt:

=== BANANA PROMPT FOR FIG4 ===
Create a scientific diagram showing a two-stage causal discovery pipeline:

Stage 1 (top): "Skeleton Learning"
- Input: Observational Data (matrix icon)
- Process: PC Algorithm (box)
- Output: PDAG with undirected edges (graph with some directed, some undirected edges)

Stage 2 (bottom): "Conservative Refinement"
- For each undirected edge:
  - Statistical Feature Extraction (HSIC, residual independence)
  - StatTranslator: Convert to natural language narrative
  - LLM Inference: Determine direction or "Unclear"
  - Fallback mechanism: Keep undirected if uncertain

Output: Final DAG

Style: Clean, academic, blue/gray color scheme, arrows showing data flow
=== END BANANA PROMPT ===
"""

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch

# 设置中文字体 (macOS 简体中文)
plt.rcParams['font.sans-serif'] = ['Hiragino Sans GB', 'PingFang HK', 'Heiti TC', 'Arial Unicode MS']
plt.rcParams['axes.unicode_minus'] = False

def main():
    fig, ax = plt.subplots(figsize=(14, 8))
    ax.set_xlim(0, 14)
    ax.set_ylim(0, 8)
    ax.axis('off')
    
    # 颜色方案
    colors = {
        'input': '#3498DB',
        'stat': '#9B59B6',
        'narrative': '#E67E22',
        'llm': '#E74C3C',
        'output': '#27AE60',
        'pc': '#1ABC9C',
        'arrow': '#2C3E50'
    }
    
    # === Phase 1: PC Algorithm ===
    ax.text(7, 7.5, '保守混合框架', fontsize=16, fontweight='bold', 
            ha='center', color='#2C3E50')
    
    # 输入数据
    input_box = FancyBboxPatch((0.5, 5), 2, 1.2, boxstyle="round,pad=0.05",
                                facecolor=colors['input'], edgecolor='white', linewidth=2, alpha=0.9)
    ax.add_patch(input_box)
    ax.text(1.5, 5.6, '观测数据\nD', ha='center', va='center', fontsize=10, 
            color='white', fontweight='bold')
    
    # PC 算法
    pc_box = FancyBboxPatch((3.5, 5), 2.5, 1.2, boxstyle="round,pad=0.05",
                             facecolor=colors['pc'], edgecolor='white', linewidth=2, alpha=0.9)
    ax.add_patch(pc_box)
    ax.text(4.75, 5.6, 'PC 算法\n(骨架 + V-结构)', ha='center', va='center', 
            fontsize=9, color='white', fontweight='bold')
    
    # PDAG
    pdag_box = FancyBboxPatch((7, 5), 2, 1.2, boxstyle="round,pad=0.05",
                               facecolor='#95A5A6', edgecolor='white', linewidth=2, alpha=0.9)
    ax.add_patch(pdag_box)
    ax.text(8, 5.6, 'PDAG\n(含无向边)', ha='center', va='center', 
            fontsize=9, color='white', fontweight='bold')
    
    # 箭头
    ax.annotate('', xy=(3.4, 5.6), xytext=(2.6, 5.6),
               arrowprops=dict(arrowstyle='->', color=colors['arrow'], lw=2))
    ax.annotate('', xy=(6.9, 5.6), xytext=(6.1, 5.6),
               arrowprops=dict(arrowstyle='->', color=colors['arrow'], lw=2))
    
    # Phase 1 标签
    ax.text(4.75, 6.5, '阶段 1: 骨架学习', fontsize=11, fontweight='bold',
            ha='center', color=colors['pc'], 
            bbox=dict(boxstyle='round', facecolor='white', edgecolor=colors['pc'], alpha=0.8))
    
    # === Phase 2: ACR Engine ===
    
    # 提取无向边
    extract_box = FancyBboxPatch((10, 5), 2, 1.2, boxstyle="round,pad=0.05",
                                  facecolor='#7F8C8D', edgecolor='white', linewidth=2, alpha=0.9)
    ax.add_patch(extract_box)
    ax.text(11, 5.6, '提取\n无向边', ha='center', va='center', 
            fontsize=9, color='white', fontweight='bold')
    
    ax.annotate('', xy=(9.9, 5.6), xytext=(9.1, 5.6),
               arrowprops=dict(arrowstyle='->', color=colors['arrow'], lw=2))
    
    # 向下箭头
    ax.annotate('', xy=(11, 4.9), xytext=(11, 4.2),
               arrowprops=dict(arrowstyle='->', color=colors['arrow'], lw=2))
    
    # === ACR Pipeline (下方) ===
    
    # 统计特征提取
    stat_box = FancyBboxPatch((0.5, 2), 2.5, 1.5, boxstyle="round,pad=0.05",
                               facecolor=colors['stat'], edgecolor='white', linewidth=2, alpha=0.9)
    ax.add_patch(stat_box)
    ax.text(1.75, 2.75, '统计特征\n提取', ha='center', va='center', 
            fontsize=10, color='white', fontweight='bold')
    ax.text(1.75, 2.2, '(偏度, HSIC,\nR², 异方差性)', ha='center', va='center', 
            fontsize=8, color='white', alpha=0.9)
    
    # StatTranslator
    trans_box = FancyBboxPatch((4, 2), 2.5, 1.5, boxstyle="round,pad=0.05",
                                facecolor=colors['narrative'], edgecolor='white', linewidth=2, alpha=0.9)
    ax.add_patch(trans_box)
    ax.text(5.25, 2.75, 'StatTranslator', ha='center', va='center', 
            fontsize=10, color='white', fontweight='bold')
    ax.text(5.25, 2.2, '(数值 → 自然\n语言叙述)', ha='center', va='center', 
            fontsize=8, color='white', alpha=0.9)
    
    # LLM 推理
    llm_box = FancyBboxPatch((7.5, 2), 2.5, 1.5, boxstyle="round,pad=0.05",
                              facecolor=colors['llm'], edgecolor='white', linewidth=2, alpha=0.9)
    ax.add_patch(llm_box)
    ax.text(8.75, 2.75, 'LLM 推理', ha='center', va='center', 
            fontsize=10, color='white', fontweight='bold')
    ax.text(8.75, 2.2, '(GPT-4 / DeepSeek\n配合 ACR 提示词)', ha='center', va='center', 
            fontsize=8, color='white', alpha=0.9)
    
    # 输出
    output_box = FancyBboxPatch((11, 2), 2.5, 1.5, boxstyle="round,pad=0.05",
                                 facecolor=colors['output'], edgecolor='white', linewidth=2, alpha=0.9)
    ax.add_patch(output_box)
    ax.text(12.25, 2.75, '因果方向', ha='center', va='center', 
            fontsize=10, color='white', fontweight='bold')
    ax.text(12.25, 2.2, '(A→B / B→A /\n不确定 + 置信度)', ha='center', va='center', 
            fontsize=8, color='white', alpha=0.9)
    
    # 箭头
    ax.annotate('', xy=(3.9, 2.75), xytext=(3.1, 2.75),
               arrowprops=dict(arrowstyle='->', color=colors['arrow'], lw=2))
    ax.annotate('', xy=(7.4, 2.75), xytext=(6.6, 2.75),
               arrowprops=dict(arrowstyle='->', color=colors['arrow'], lw=2))
    ax.annotate('', xy=(10.9, 2.75), xytext=(10.1, 2.75),
               arrowprops=dict(arrowstyle='->', color=colors['arrow'], lw=2))
    
    # 连接上下
    ax.annotate('', xy=(1.75, 3.6), xytext=(11, 4.1),
               arrowprops=dict(arrowstyle='->', color=colors['arrow'], lw=2,
                              connectionstyle="arc3,rad=-0.3"))
    ax.text(6, 4.3, '对每条无向边 (Xi, Xj)', fontsize=9, 
            ha='center', style='italic', color='#7F8C8D')
    
    # Phase 2 标签
    ax.text(6.5, 1, '阶段 2: 保守精炼 (ACR 引擎)', fontsize=11, fontweight='bold',
            ha='center', color=colors['llm'],
            bbox=dict(boxstyle='round', facecolor='white', edgecolor=colors['llm'], alpha=0.8))
    
    # 回退机制说明
    ax.text(12.25, 1.3, '回退机制: 若不确定,\n保持边为无向', 
            ha='center', fontsize=8, style='italic', color='#7F8C8D')
    
    # 最终输出箭头
    ax.annotate('', xy=(12.25, 5), xytext=(12.25, 3.6),
               arrowprops=dict(arrowstyle='->', color=colors['output'], lw=2))
    ax.text(12.8, 4.3, '最终\nDAG', fontsize=9, fontweight='bold', color=colors['output'])
    
    plt.tight_layout()
    
    plt.savefig('fig4_framework.png', dpi=300, bbox_inches='tight')
    plt.savefig('fig4_framework.pdf', bbox_inches='tight')
    print("Saved: fig4_framework.png, fig4_framework.pdf")
    
    plt.show()

if __name__ == '__main__':
    main()
