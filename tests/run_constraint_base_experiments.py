"""
约束类基座替代实验脚本
Task 2: 运行约束类基座替代实验

包含：
- 2.1: 在 Sachs 网络上运行 Dual PC + ACR
- 2.2: 在 Asia/Child 网络上运行 FCI + ACR

Requirements: 2.5, 2.6
"""

import sys
import os
import json
from datetime import datetime

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from tests.test_p2_experiments import P2Experimenter, compute_shd

RESULTS_DIR = os.path.join(os.path.dirname(__file__), '../results')
os.makedirs(RESULTS_DIR, exist_ok=True)


def run_task_2_1_dual_pc_sachs():
    """
    Task 2.1: 在 Sachs 网络上运行 Dual PC + ACR
    
    - 复用 test_p2_experiments.py 中的 run_acr_on_edges() 方法
    - 记录 SHD 和 F1 指标
    - Requirements: 2.5
    """
    print("\n" + "=" * 70)
    print("Task 2.1: Dual PC + ACR on Sachs Network")
    print("=" * 70)
    
    experimenter = P2Experimenter()
    if not experimenter.engine:
        print("❌ Engine initialization failed. Exiting.")
        return None
    
    # 运行 Dual PC + ACR 实验
    result = experimenter.experiment_dual_pc_acr(
        network_name="sachs",
        sample_size=1000
    )
    
    if result:
        # 保存结果
        output_file = os.path.join(RESULTS_DIR, 'task_2_1_dual_pc_sachs.json')
        result['task'] = '2.1'
        result['description'] = 'Dual PC + ACR on Sachs network'
        result['timestamp'] = datetime.now().isoformat()
        
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(result, f, indent=2, ensure_ascii=False)
        
        print(f"\n💾 Results saved to: {output_file}")
        
        # 打印关键指标
        print("\n" + "=" * 50)
        print("📊 Task 2.1 Summary (Dual PC + ACR on Sachs)")
        print("=" * 50)
        print(f"Dual PC SHD:       {result.get('dual_pc_shd', 'N/A')}")
        print(f"Dual PC + ACR SHD: {result.get('dual_pc_acr_shd', 'N/A')}")
        print(f"SHD Improvement:   {result.get('shd_improvement', 'N/A')}")
        print(f"F1 Score:          {result.get('f1', 'N/A'):.3f}" if result.get('f1') else "F1 Score: N/A")
        print(f"Precision:         {result.get('precision', 'N/A'):.3f}" if result.get('precision') else "Precision: N/A")
        print(f"Recall:            {result.get('recall', 'N/A'):.3f}" if result.get('recall') else "Recall: N/A")
        print(f"ACR Accuracy:      {result.get('acr_accuracy', 'N/A'):.1%}" if result.get('acr_accuracy') else "ACR Accuracy: N/A")
        print("=" * 50)
    
    return result


def run_task_2_2_fci_asia_child():
    """
    Task 2.2: 在 Asia/Child 网络上运行 FCI + ACR
    
    - 提取 FCI 的 PAG 输出中的可定向边
    - 对比 FCI + ACR 与原始 FCI 的 SHD
    - Requirements: 2.6
    """
    print("\n" + "=" * 70)
    print("Task 2.2: FCI + ACR on Asia/Child Networks")
    print("=" * 70)
    
    experimenter = P2Experimenter()
    if not experimenter.engine:
        print("❌ Engine initialization failed. Exiting.")
        return None
    
    results = {}
    
    # 在 Asia 网络上运行
    print("\n--- Running on Asia Network ---")
    asia_result = experimenter.experiment_fci_acr(
        network_name="asia",
        sample_size=1000
    )
    if asia_result:
        asia_result['task'] = '2.2'
        asia_result['timestamp'] = datetime.now().isoformat()
        results['asia'] = asia_result
    
    # 在 Child 网络上运行
    print("\n--- Running on Child Network ---")
    child_result = experimenter.experiment_fci_acr(
        network_name="child",
        sample_size=1000
    )
    if child_result:
        child_result['task'] = '2.2'
        child_result['timestamp'] = datetime.now().isoformat()
        results['child'] = child_result
    
    # 保存结果
    output_file = os.path.join(RESULTS_DIR, 'task_2_2_fci_asia_child.json')
    combined_result = {
        'task': '2.2',
        'description': 'FCI + ACR on Asia/Child networks',
        'timestamp': datetime.now().isoformat(),
        'networks': results
    }
    
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(combined_result, f, indent=2, ensure_ascii=False)
    
    print(f"\n💾 Results saved to: {output_file}")
    
    # 打印关键指标
    print("\n" + "=" * 60)
    print("📊 Task 2.2 Summary (FCI + ACR on Asia/Child)")
    print("=" * 60)
    print(f"{'Network':<10} {'FCI SHD':<12} {'FCI+ACR SHD':<14} {'Improvement':<12} {'F1':<8}")
    print("-" * 60)
    
    for network_name, result in results.items():
        fci_shd = result.get('fci_shd', 'N/A')
        fci_acr_shd = result.get('fci_acr_shd', 'N/A')
        improvement = result.get('shd_improvement', 'N/A')
        f1 = result.get('f1', 0)
        print(f"{network_name:<10} {fci_shd:<12} {fci_acr_shd:<14} {improvement:<12} {f1:.3f}")
    
    print("=" * 60)
    
    return combined_result


def main():
    """运行所有约束类基座替代实验"""
    import argparse
    
    parser = argparse.ArgumentParser(description='Run constraint-based base algorithm experiments')
    parser.add_argument('--task', type=str, default='all',
                        choices=['all', '2.1', '2.2'],
                        help='Which task to run (default: all)')
    args = parser.parse_args()
    
    all_results = {}
    
    if args.task in ['all', '2.1']:
        result = run_task_2_1_dual_pc_sachs()
        if result:
            all_results['task_2_1'] = result
    
    if args.task in ['all', '2.2']:
        result = run_task_2_2_fci_asia_child()
        if result:
            all_results['task_2_2'] = result
    
    # 保存汇总结果
    if all_results:
        summary_file = os.path.join(RESULTS_DIR, 'task_2_constraint_base_summary.json')
        summary = {
            'task': '2',
            'description': '约束类基座替代实验',
            'timestamp': datetime.now().isoformat(),
            'results': all_results
        }
        
        with open(summary_file, 'w', encoding='utf-8') as f:
            json.dump(summary, f, indent=2, ensure_ascii=False)
        
        print(f"\n💾 Summary saved to: {summary_file}")
    
    print("\n✅ All constraint-based experiments completed!")
    return all_results


if __name__ == "__main__":
    main()
