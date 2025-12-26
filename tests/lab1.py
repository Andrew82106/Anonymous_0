"""
Lab 1: 真实贝叶斯网络基准实验

网络：Sprinkler(4), Asia(8), Sachs(11), Child(20), Insurance(27), Alarm(37), Hailfinder(56), Hepar II(70)
方法：ACR-Hybrid, ACR-Hybrid-i, PC, HillClimb, Random
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import json
import argparse
from datetime import datetime

from utils_set.utils import path_config

# 使用 methods 模块
from methods import (
    ACRHybrid,
    ACRHybridI,
    PC_Algorithm,
    HillClimb_Algorithm,
    Random_Baseline,
    load_network,
    NETWORKS,
    compute_shd
)

# 结果保存路径
RESULTS_DIR = str(path_config.results_dir)


def get_results_file(model_name):
    """根据模型名称获取结果文件路径"""
    if model_name == 'acr-hybrid-i':
        return os.path.join(RESULTS_DIR, 'lab1_acr_hybrid_i.json')
    else:
        return os.path.join(RESULTS_DIR, 'lab1_real_networks.json')


def run_experiment_on_network(network_name, acr, pc, hc, random_baseline, sample_size=1000, model_name='acr-hybrid'):
    """在单个网络上运行所有方法"""
    
    # 不区分大小写匹配
    network_name_lower = network_name.lower()
    config = None
    for n in NETWORKS:
        if n['name'].lower() == network_name_lower:
            config = n
            break
    
    if config is None:
        print(f"Error: Network '{network_name}' not found. Available: {[n['name'] for n in NETWORKS]}")
        return None
    
    # 使用标准名称
    network_name = config['name']
    
    print(f"\n{'='*60}")
    print(f"Network: {network_name} ({config['nodes']} nodes)")
    print(f"{'='*60}")
    
    # 加载数据
    try:
        df, true_adjmat, nodes, edges = load_network(network_name, sample_size)
    except Exception as e:
        print(f"  Failed to load network: {e}")
        return None
    
    n_nodes = len(nodes)
    n_edges = len(edges)
    print(f"  Loaded: {n_nodes} nodes, {n_edges} edges, {sample_size} samples")
    
    result = {
        'network': network_name,
        'n_nodes': n_nodes,
        'n_edges': n_edges,
        'sample_size': sample_size,
        'model': model_name,
        'timestamp': datetime.now().isoformat()
    }
    
    # 1. ACR (Hybrid 或 Hybrid-i)
    method_label = 'ACR-Hybrid-i' if model_name == 'acr-hybrid-i' else 'ACR-Hybrid'
    print(f"\n  [1/4] Running {method_label}...")
    acr_result = acr.run(df, true_adjmat, nodes, edges, network_name)
    
    # 兼容新旧接口
    acr_accuracy = acr_result.get('llm_accuracy', acr_result.get('accuracy', 0))
    acr_correct = acr_result.get('llm_correct', acr_result.get('correct_count', 0))
    
    result['acr'] = {
        'method': method_label,
        'shd': acr_result['shd'],
        'accuracy': acr_accuracy,
        'correct_count': acr_correct,
        'total_edges': acr_result.get('total_edges', acr_result.get('llm_total', 0)),
        'llm_calls': acr_result.get('llm_total', 0),
        'unclear_count': acr_result.get('unclear_count', 0),
        'unclear_ratio': acr_result.get('unclear_ratio', 0),
        'pc_oriented_count': acr_result.get('pc_oriented_count', 0),
        'mec_edge_count': acr_result.get('mec_edge_count', 0),
        'meek_propagated': acr_result.get('meek_propagated', 0),
        'cycle_rejected': acr_result.get('cycle_rejected', 0),
        'pairwise_results': acr_result.get('pairwise_results', [])
    }
    print(f"    SHD: {acr_result['shd']}, LLM Accuracy: {acr_accuracy:.1%}, Meek: {acr_result.get('meek_propagated', 0)}")
    
    # 2. PC
    print(f"\n  [2/4] Running PC Algorithm...")
    pc_result = pc.run(df, true_adjmat, nodes)
    if pc_result:
        result['pc'] = {'shd': pc_result['shd']}
        print(f"    SHD: {pc_result['shd']}")
    else:
        result['pc'] = None
        print(f"    Failed")
    
    # 3. HillClimb
    print(f"\n  [3/4] Running HillClimb Algorithm...")
    hc_result = hc.run(df, true_adjmat, nodes)
    if hc_result:
        result['hillclimb'] = {'shd': hc_result['shd']}
        print(f"    SHD: {hc_result['shd']}")
    else:
        result['hillclimb'] = None
        print(f"    Failed")
    
    # 4. Random
    print(f"\n  [4/4] Running Random Baseline...")
    random_result = random_baseline.run(n_nodes, n_edges, true_adjmat, nodes)
    result['random'] = {'shd': random_result['shd']}
    print(f"    SHD: {random_result['shd']}")
    
    return result


def print_summary_table(results, model_name='acr-hybrid'):
    """打印汇总表格"""
    method_label = 'ACR-Hybrid-i' if model_name == 'acr-hybrid-i' else 'ACR-Hybrid'
    
    print(f"\n\n{'='*100}")
    print(f"SUMMARY TABLE ({method_label})")
    print(f"{'='*100}")
    
    header = f"{'Network':<12} {'Nodes':<8} {method_label:<14} {'PC':<12} {'HillClimb':<12} {'Random':<12} {'Accuracy':<10}"
    print(header)
    print("-" * 100)
    
    total_acr, total_pc, total_hc, total_random = 0, 0, 0, 0
    count_pc, count_hc = 0, 0
    
    for r in results:
        net = r['network']
        nodes = r['n_nodes']
        # 兼容旧格式 (acr_hybrid) 和新格式 (acr)
        acr_data = r.get('acr') or r.get('acr_hybrid', {})
        acr_shd = acr_data.get('shd', '-')
        acr_acc = acr_data.get('accuracy', 0)
        pc_shd = r['pc']['shd'] if r.get('pc') else '-'
        hc_shd = r['hillclimb']['shd'] if r.get('hillclimb') else '-'
        random_shd = r['random']['shd']
        
        print(f"{net:<12} {nodes:<8} {acr_shd:<14} {str(pc_shd):<12} {str(hc_shd):<12} {random_shd:<12} {acr_acc:.1%}")
        
        if isinstance(acr_shd, (int, float)):
            total_acr += acr_shd
        total_random += random_shd
        if r.get('pc'):
            total_pc += r['pc']['shd']
            count_pc += 1
        if r.get('hillclimb'):
            total_hc += r['hillclimb']['shd']
            count_hc += 1
    
    print("-" * 100)
    if results:
        avg_acr = total_acr / len(results)
        avg_random = total_random / len(results)
        avg_pc = f"{total_pc / count_pc:.1f}" if count_pc > 0 else '-'
        avg_hc = f"{total_hc / count_hc:.1f}" if count_hc > 0 else '-'
        print(f"{'Average':<12} {'':<8} {avg_acr:<14.1f} {avg_pc:<12} {avg_hc:<12} {avg_random:<12.1f}")
    print(f"{'='*100}")


def main():
    parser = argparse.ArgumentParser(description='Lab 1: Real Bayesian Network Benchmark')
    parser.add_argument('--network', type=str, default=None,
                        help='Network to test (e.g., child, insurance). If not specified, run all.')
    parser.add_argument('--model', type=str, default='acr-hybrid',
                        choices=['acr-hybrid', 'acr-hybrid-i'],
                        help='Model to use: acr-hybrid (default) or acr-hybrid-i (intensive)')
    parser.add_argument('--sample_size', type=int, default=1000,
                        help='Sample size (default: 1000)')
    parser.add_argument('--max_nodes', type=int, default=None,
                        help='Only test networks with <= max_nodes nodes')
    args = parser.parse_args()
    
    method_label = 'ACR-Hybrid-i' if args.model == 'acr-hybrid-i' else 'ACR-Hybrid'
    
    print("\n" + "="*80)
    print(f"Lab 1: Real Bayesian Network Benchmark ({method_label})")
    print("="*80)
    
    # 结果文件
    results_file = get_results_file(args.model)
    
    # 加载已有结果
    all_results = []
    if os.path.exists(results_file):
        try:
            with open(results_file, 'r', encoding='utf-8') as f:
                all_results = json.load(f)
            print(f"Loaded {len(all_results)} existing results from {results_file}")
        except:
            pass
    
    # 初始化方法
    if args.model == 'acr-hybrid-i':
        acr = ACRHybridI(model_name="deepseek-v3-lanyun")
    else:
        acr = ACRHybrid(model_name="deepseek-v3-lanyun")
    
    pc = PC_Algorithm()
    hc = HillClimb_Algorithm()
    random_baseline = Random_Baseline()
    
    # 筛选网络
    networks_to_test = NETWORKS
    if args.max_nodes:
        networks_to_test = [n for n in NETWORKS if n['nodes'] <= args.max_nodes]
        print(f"Filtering networks with <= {args.max_nodes} nodes: {[n['name'] for n in networks_to_test]}")
    
    if args.network:
        # 运行单个网络
        print(f"Testing: {args.network}")
        print("="*80)
        
        result = run_experiment_on_network(args.network, acr, pc, hc, random_baseline, args.sample_size, args.model)
        if result:
            # 更新结果
            all_results = [r for r in all_results if r['network'] != args.network]
            all_results.append(result)
            
            # 排序
            network_order = [n['name'] for n in NETWORKS]
            all_results.sort(key=lambda x: network_order.index(x['network']) if x['network'] in network_order else 999)
            
            with open(results_file, 'w', encoding='utf-8') as f:
                json.dump(all_results, f, indent=2, ensure_ascii=False, default=str)
            print(f"\nSaved to {results_file}")
    else:
        # 运行所有网络
        print(f"Testing networks: {[n['name'] for n in networks_to_test]}")
        print("="*80)
        
        for net in networks_to_test:
            # 跳过已有结果
            existing = [r for r in all_results if r['network'] == net['name']]
            if existing:
                print(f"\nSkipping {net['name']} (already exists)")
                continue
            
            result = run_experiment_on_network(net['name'], acr, pc, hc, random_baseline, args.sample_size, args.model)
            if result:
                all_results.append(result)
                
                # 排序
                network_order = [n['name'] for n in NETWORKS]
                all_results.sort(key=lambda x: network_order.index(x['network']) if x['network'] in network_order else 999)
                
                with open(results_file, 'w', encoding='utf-8') as f:
                    json.dump(all_results, f, indent=2, ensure_ascii=False, default=str)
                print(f"\nSaved to {results_file}")
    
    # 只显示当前测试范围内的结果
    if args.max_nodes:
        display_results = [r for r in all_results if r['n_nodes'] <= args.max_nodes]
    else:
        display_results = all_results
    
    print_summary_table(display_results, args.model)


if __name__ == "__main__":
    main()
