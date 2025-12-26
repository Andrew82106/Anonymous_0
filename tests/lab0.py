"""
Lab 0: Constraint-Asymmetry Decoupling (CAD) - Final Evidence Generator

验证论文核心洞察：
1. Level 1 (Existence): MI 区分 True Edge vs No Edge → PC 骨架必要性
2. Level 2 (Orientation): MEC Edge 比 PC Solved 有更强的非对称性信号 → ACR 必要性

输出:
- figures/lab0_proof_1_filtering.png: MI 分离能力证明
- figures/lab0_proof_2_complementarity.png: MEC vs PC Solved 非对称性对比
- results/lab0_evidence.json: 完整统计数据
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from itertools import combinations
from tqdm import tqdm
from datetime import datetime
from scipy.stats import entropy

from pgmpy.estimators import PC
from methods import load_network
from utils_set.stat_translator import StatTranslator

# 配置
TARGET_NETWORKS = ['sprinkler', 'child', 'insurance', 'alarm']
RESULTS_PATH = os.path.join(os.path.dirname(__file__), '..', 'results', 'lab0_evidence.json')
FIGURES_DIR = os.path.join(os.path.dirname(__file__), '..', 'figures')


def get_pc_oriented_edges(df):
    """运行 PC 算法，返回定向边集合"""
    pc = PC(df)
    skeleton, separating_sets = pc.build_skeleton()
    pdag = pc.skeleton_to_pdag(skeleton, separating_sets)
    
    oriented = set()
    for edge in pdag.edges():
        if not pdag.has_edge(edge[1], edge[0]):
            oriented.add(edge)
    return oriented


def categorize_edge(pair, true_edges_undirected, pc_oriented):
    """分类边: no_edge, pc_solved, mec_edge"""
    if frozenset(pair) not in true_edges_undirected:
        return 'no_edge'
    if (pair[0], pair[1]) in pc_oriented or (pair[1], pair[0]) in pc_oriented:
        return 'pc_solved'
    return 'mec_edge'


def calc_entropy(arr):
    """计算离散变量熵"""
    _, counts = np.unique(arr, return_counts=True)
    return entropy(counts / len(arr), base=2)


def calculate_metrics(df, var_a, var_b, translator):
    """计算三个核心指标: MI, Res_Asym, Ent_Asym"""
    epsilon = 1e-9
    
    try:
        stats = translator.analyze(df[var_a].values, df[var_b].values)
        if not stats.get('is_discrete', False):
            return None
        
        # MI (Constraint Signal)
        mi = stats['dir_ab']['mutual_information']
        
        # Res_Asym (Functional Signal)
        p_ab = stats['dir_ab']['error_independence_p']
        p_ba = stats['dir_ba']['error_independence_p']
        res_asym = abs(p_ab - p_ba)
        
        # Ent_Asym (Informational Signal)
        h_a, h_b = stats['x_entropy'], stats['y_entropy']
        ent_asym = abs(h_a - h_b) / (max(h_a, h_b) + epsilon)
        
        return {'mi': mi, 'res_asym': res_asym, 'ent_asym': ent_asym}
    except:
        return None


def run_network_analysis(network_name, sample_size=1000):
    """分析单个网络"""
    print(f"\n[{network_name.upper()}] Loading...")
    df, _, nodes, edges = load_network(network_name, sample_size)
    true_edges = set(frozenset({a, b}) for a, b in edges)
    
    print(f"  Nodes: {len(nodes)}, Edges: {len(edges)}")
    
    print(f"  Running PC...")
    pc_oriented = get_pc_oriented_edges(df)
    
    print(f"  Calculating metrics...")
    translator = StatTranslator()
    results = []
    
    for pair in tqdm(list(combinations(nodes, 2)), desc="  ", leave=False):
        metrics = calculate_metrics(df, pair[0], pair[1], translator)
        if metrics:
            category = categorize_edge(pair, true_edges, pc_oriented)
            results.append({'network': network_name, 'category': category, **metrics})
    
    return pd.DataFrame(results)


def plot_proof_1_filtering(df):
    """Figure 1: MI 分离 True Edge vs No Edge (Publication Quality)"""
    print("\n[Figure 1] Generating filtering proof...")
    
    plot_df = df.copy()
    plot_df['Edge Type'] = plot_df['category'].apply(
        lambda x: 'True Edge' if x in ['pc_solved', 'mec_edge'] else 'No Edge'
    )
    
    # Publication style setup
    plt.rcParams['font.family'] = 'Arial'
    sns.set_theme(style="ticks", context="paper", font_scale=1.5)
    
    fig, ax = plt.subplots(figsize=(7, 6))
    
    # High-contrast professional palette
    colors = {'No Edge': '#e74c3c', 'True Edge': '#2ecc71'}
    
    # Box plot with transparency
    box = sns.boxplot(data=plot_df, x='Edge Type', y='mi', palette=colors, 
                      order=['No Edge', 'True Edge'], ax=ax, width=0.5,
                      boxprops=dict(alpha=0.8), 
                      flierprops=dict(marker='o', markersize=4, alpha=0.5))
    
    # Strip plot for individual points
    sns.stripplot(data=plot_df, x='Edge Type', y='mi', palette=colors,
                  order=['No Edge', 'True Edge'], ax=ax, alpha=0.25, size=3, jitter=0.15)
    
    # Calculate statistics
    no_edge_data = plot_df[plot_df['Edge Type'] == 'No Edge']['mi']
    true_edge_data = plot_df[plot_df['Edge Type'] == 'True Edge']['mi']
    no_edge_mean = no_edge_data.mean()
    true_edge_mean = true_edge_data.mean()
    noise_floor = no_edge_data.max()
    ratio = true_edge_mean / no_edge_mean if no_edge_mean > 0 else float('inf')
    
    # Add noise floor line
    ax.axhline(y=noise_floor, color='#e74c3c', linestyle='--', linewidth=1.5, alpha=0.7)
    ax.text(1.02, noise_floor, 'Noise Floor', transform=ax.get_yaxis_transform(),
            fontsize=10, color='#c0392b', va='center', fontweight='bold')
    
    # Mean annotations with arrows
    for i, (edge_type, color) in enumerate([('No Edge', '#c0392b'), ('True Edge', '#27ae60')]):
        mean_val = plot_df[plot_df['Edge Type'] == edge_type]['mi'].mean()
        ax.annotate(f'μ = {mean_val:.3f}', xy=(i, mean_val), 
                   xytext=(i + 0.35, mean_val * 1.1),
                   fontsize=12, fontweight='bold', color=color,
                   arrowprops=dict(arrowstyle='->', color=color, lw=1.2))
    
    # Labels and title
    ax.set_xlabel('Edge Category', fontsize=14, fontweight='bold')
    ax.set_ylabel('Mutual Information (MI)', fontsize=14, fontweight='bold')
    ax.set_title('MI Separates True Edges from Noise', 
                fontsize=16, fontweight='bold', pad=15)
    
    # Separation ratio annotation box
    ax.text(0.97, 0.97, f'Separation: {ratio:.1f}×', transform=ax.transAxes,
           fontsize=14, fontweight='bold', ha='right', va='top',
           bbox=dict(boxstyle='round,pad=0.4', facecolor='#f8f9fa', 
                    edgecolor='#2c3e50', linewidth=1.5, alpha=0.95))
    
    # Clean up spines (NeurIPS style)
    sns.despine(ax=ax, top=True, right=True)
    ax.yaxis.grid(True, linestyle='-', alpha=0.3, color='gray')
    ax.xaxis.grid(False)
    
    # Tick styling
    ax.tick_params(axis='both', which='major', labelsize=12)
    
    plt.tight_layout()
    path = os.path.join(FIGURES_DIR, 'lab0_proof_1_filtering.png')
    plt.savefig(path, dpi=300, bbox_inches='tight', facecolor='white')
    plt.close()
    print(f"  Saved: {path}")
    
    return ratio


def plot_proof_2_complementarity(df):
    """Figure 2: MEC Edge vs PC Solved 非对称性对比 (Publication Quality)"""
    print("\n[Figure 2] Generating complementarity proof...")
    
    # 只保留 True Edge (排除 No Edge)
    plot_df = df[df['category'].isin(['pc_solved', 'mec_edge'])].copy()
    plot_df['Category'] = plot_df['category'].map({
        'pc_solved': 'PC Solved', 'mec_edge': 'MEC Edge'
    })
    
    # Publication style setup
    plt.rcParams['font.family'] = 'Arial'
    sns.set_theme(style="ticks", context="paper", font_scale=1.5)
    
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    
    # High-contrast professional palette
    colors = {'PC Solved': '#3498db', 'MEC Edge': '#2ecc71'}
    order = ['PC Solved', 'MEC Edge']
    
    # 按网络聚合
    agg_df = plot_df.groupby(['network', 'Category']).agg({
        'res_asym': ['mean', 'std'],
        'ent_asym': ['mean', 'std']
    }).reset_index()
    agg_df.columns = ['network', 'Category', 'res_mean', 'res_std', 'ent_mean', 'ent_std']
    
    network_order = ['sprinkler', 'child', 'insurance', 'alarm']
    network_labels = ['Sprinkler\n(4)', 'Child\n(20)', 'Insurance\n(27)', 'Alarm\n(37)']
    
    x = np.arange(len(network_order))
    width = 0.35
    
    # ========== Left Panel: Residual Asymmetry ==========
    ax1 = axes[0]
    
    # Calculate gains for annotations
    res_gains = {}
    for net in network_order:
        net_data = agg_df[agg_df['network'] == net]
        pc_val = net_data[net_data['Category'] == 'PC Solved']['res_mean'].values
        mec_val = net_data[net_data['Category'] == 'MEC Edge']['res_mean'].values
        if len(pc_val) > 0 and len(mec_val) > 0 and pc_val[0] > 0:
            res_gains[net] = (mec_val[0] / pc_val[0] - 1) * 100
        else:
            res_gains[net] = 0
    
    for i, cat in enumerate(order):
        cat_data = agg_df[agg_df['Category'] == cat].set_index('network').reindex(network_order)
        bars = ax1.bar(x + (i - 0.5) * width, cat_data['res_mean'], width, 
                      label=cat, color=colors[cat], alpha=0.85,
                      yerr=cat_data['res_std'], capsize=5, 
                      error_kw=dict(lw=1.5, capthick=1.5))
        
        # Add gain annotations on MEC bars
        if cat == 'MEC Edge':
            for j, net in enumerate(network_order):
                if net in res_gains and res_gains[net] != 0:
                    bar_height = cat_data.loc[net, 'res_mean']
                    bar_std = cat_data.loc[net, 'res_std']
                    gain_text = f'+{res_gains[net]:.0f}%' if res_gains[net] > 0 else f'{res_gains[net]:.0f}%'
                    ax1.annotate(gain_text, 
                               xy=(x[j] + 0.5 * width, bar_height + bar_std + 0.01),
                               ha='center', va='bottom', fontsize=11, fontweight='bold',
                               color='#27ae60' if res_gains[net] > 0 else '#c0392b')
    
    ax1.set_xlabel('Network (Nodes)', fontsize=14, fontweight='bold')
    ax1.set_ylabel('Residual Asymmetry', fontsize=14, fontweight='bold')
    ax1.set_title('Functional Signal', fontsize=15, fontweight='bold', pad=10)
    ax1.set_xticks(x)
    ax1.set_xticklabels(network_labels, fontsize=11)
    ax1.tick_params(axis='y', labelsize=11)
    
    # Clean up spines
    sns.despine(ax=ax1, top=True, right=True)
    ax1.yaxis.grid(True, linestyle='-', alpha=0.3, color='gray')
    ax1.xaxis.grid(False)
    
    # ========== Right Panel: Entropy Asymmetry ==========
    ax2 = axes[1]
    
    # Calculate gains for annotations
    ent_gains = {}
    for net in network_order:
        net_data = agg_df[agg_df['network'] == net]
        pc_val = net_data[net_data['Category'] == 'PC Solved']['ent_mean'].values
        mec_val = net_data[net_data['Category'] == 'MEC Edge']['ent_mean'].values
        if len(pc_val) > 0 and len(mec_val) > 0 and pc_val[0] > 0:
            ent_gains[net] = (mec_val[0] / pc_val[0] - 1) * 100
        else:
            ent_gains[net] = 0
    
    for i, cat in enumerate(order):
        cat_data = agg_df[agg_df['Category'] == cat].set_index('network').reindex(network_order)
        bars = ax2.bar(x + (i - 0.5) * width, cat_data['ent_mean'], width,
                      label=cat, color=colors[cat], alpha=0.85,
                      yerr=cat_data['ent_std'], capsize=5,
                      error_kw=dict(lw=1.5, capthick=1.5))
        
        # Add gain annotations on MEC bars
        if cat == 'MEC Edge':
            for j, net in enumerate(network_order):
                if net in ent_gains and ent_gains[net] != 0:
                    bar_height = cat_data.loc[net, 'ent_mean']
                    bar_std = cat_data.loc[net, 'ent_std']
                    gain_text = f'+{ent_gains[net]:.0f}%' if ent_gains[net] > 0 else f'{ent_gains[net]:.0f}%'
                    ax2.annotate(gain_text,
                               xy=(x[j] + 0.5 * width, bar_height + bar_std + 0.01),
                               ha='center', va='bottom', fontsize=11, fontweight='bold',
                               color='#27ae60' if ent_gains[net] > 0 else '#c0392b')
    
    ax2.set_xlabel('Network (Nodes)', fontsize=14, fontweight='bold')
    ax2.set_ylabel('Entropy Asymmetry', fontsize=14, fontweight='bold')
    ax2.set_title('Informational Signal', fontsize=15, fontweight='bold', pad=10)
    ax2.set_xticks(x)
    ax2.set_xticklabels(network_labels, fontsize=11)
    ax2.tick_params(axis='y', labelsize=11)
    
    # Clean up spines
    sns.despine(ax=ax2, top=True, right=True)
    ax2.yaxis.grid(True, linestyle='-', alpha=0.3, color='gray')
    ax2.xaxis.grid(False)
    
    # Unified legend at top
    handles, labels = ax1.get_legend_handles_labels()
    fig.legend(handles, labels, loc='upper center', ncol=2, fontsize=13,
              frameon=True, fancybox=True, shadow=False, 
              bbox_to_anchor=(0.5, 1.02), edgecolor='#2c3e50')
    
    # Remove individual legends
    ax1.get_legend().remove() if ax1.get_legend() else None
    ax2.get_legend().remove() if ax2.get_legend() else None
    
    # Main title
    fig.suptitle('MEC Edges Exhibit Stronger Asymmetry Signals Than PC-Solved Edges', 
                fontsize=16, fontweight='bold', y=1.08)
    
    plt.tight_layout()
    path = os.path.join(FIGURES_DIR, 'lab0_proof_2_complementarity.png')
    plt.savefig(path, dpi=300, bbox_inches='tight', facecolor='white')
    plt.close()
    print(f"  Saved: {path}")


def print_summary(df):
    """打印汇总统计"""
    print("\n" + "="*80)
    print("CAD EVIDENCE SUMMARY")
    print("="*80)
    
    # Level 1: MI Separation
    no_edge_mi = df[df['category'] == 'no_edge']['mi'].mean()
    true_edge_mi = df[df['category'].isin(['pc_solved', 'mec_edge'])]['mi'].mean()
    mi_ratio = true_edge_mi / no_edge_mi if no_edge_mi > 0 else float('inf')
    
    print(f"\n📊 Level 1: Existence (MI Separation)")
    print(f"   No Edge MI:   {no_edge_mi:.4f}")
    print(f"   True Edge MI: {true_edge_mi:.4f}")
    print(f"   Separation:   {mi_ratio:.2f}x ✓")
    
    # Level 2: Complementarity
    pc_solved = df[df['category'] == 'pc_solved']
    mec_edge = df[df['category'] == 'mec_edge']
    
    pc_res = pc_solved['res_asym'].mean()
    mec_res = mec_edge['res_asym'].mean()
    res_gain = mec_res / pc_res if pc_res > 0 else float('inf')
    
    pc_ent = pc_solved['ent_asym'].mean()
    mec_ent = mec_edge['ent_asym'].mean()
    ent_gain = mec_ent / pc_ent if pc_ent > 0 else float('inf')
    
    print(f"\n📊 Level 2: Orientation (Signal Gain: MEC / PC Solved)")
    print(f"   Residual Asymmetry: PC={pc_res:.4f}, MEC={mec_res:.4f} → Gain={res_gain:.2f}x")
    print(f"   Entropy Asymmetry:  PC={pc_ent:.4f}, MEC={mec_ent:.4f} → Gain={ent_gain:.2f}x")
    
    if res_gain > 1.0 or ent_gain > 1.0:
        print(f"\n   ✓ COMPLEMENTARITY CONFIRMED: MEC edges carry stronger asymmetry signals!")
    else:
        print(f"\n   ✗ Complementarity not confirmed.")
    
    # Sample counts
    print(f"\n📊 Sample Counts:")
    print(f"   No Edge:   {len(df[df['category'] == 'no_edge'])}")
    print(f"   PC Solved: {len(pc_solved)}")
    print(f"   MEC Edge:  {len(mec_edge)}")
    
    print("="*80)
    
    return {
        'mi_separation': mi_ratio,
        'res_gain': res_gain,
        'ent_gain': ent_gain,
    }


def save_results(df, summary, sample_size):
    """保存结果到 JSON"""
    output = {
        'experiment': 'Lab 0: CAD Final Evidence',
        'timestamp': datetime.now().isoformat(),
        'sample_size': sample_size,
        'networks': TARGET_NETWORKS,
        'summary': summary,
        'counts': {
            'no_edge': int(len(df[df['category'] == 'no_edge'])),
            'pc_solved': int(len(df[df['category'] == 'pc_solved'])),
            'mec_edge': int(len(df[df['category'] == 'mec_edge'])),
        }
    }
    
    with open(RESULTS_PATH, 'w') as f:
        json.dump(output, f, indent=2)
    print(f"\n📁 Results saved to: {RESULTS_PATH}")


def main():
    import argparse
    parser = argparse.ArgumentParser(description='Lab 0: CAD Final Evidence')
    parser.add_argument('--sample_size', type=int, default=1000)
    args = parser.parse_args()
    
    print("\n" + "="*60)
    print("Lab 0: Constraint-Asymmetry Decoupling - Final Evidence")
    print("="*60)
    
    # 收集所有网络数据
    all_results = []
    for network in TARGET_NETWORKS:
        try:
            results = run_network_analysis(network, args.sample_size)
            all_results.append(results)
        except Exception as e:
            print(f"  ❌ Failed: {e}")
    
    df = pd.concat(all_results, ignore_index=True)
    
    # 生成图表
    plot_proof_1_filtering(df)
    plot_proof_2_complementarity(df)
    
    # 打印汇总
    summary = print_summary(df)
    
    # 保存结果
    save_results(df, summary, args.sample_size)
    
    print("\n" + "="*60)
    print("Evidence Generation Complete!")
    print("="*60)


if __name__ == "__main__":
    main()
