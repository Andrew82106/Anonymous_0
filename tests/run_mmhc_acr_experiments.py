"""
Task 3.1: MMHC + ACR 实验
在 Asia/Child/Alarm 网络上运行 MMHC + ACR 混合流水线
复用 test_p2_experiments.py 中的 experiment_p2_2_mmhc_acr() 方法

Requirements: 3.3 - 对比 MMHC-Skeleton + ACR 与原始 MMHC
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import json
import argparse
from datetime import datetime
from test_p2_experiments import P2Experimenter, compute_shd, run_mmhc_algorithm, run_pc_algorithm
from utils_set.utils import path_config

RESULTS_DIR = str(path_config.results_dir)


def run_mmhc_acr_experiments(networks=None, sample_size=1000):
    """
    运行 MMHC + ACR 实验
    在 Asia/Child/Alarm 网络上运行
    """
    print("=" * 70)
    print("Task 3.1: MMHC + ACR 实验")
    print("=" * 70)
    
    experimenter = P2Experimenter()
    if not experimenter.engine:
        print("❌ Engine initialization failed. Exiting.")
        return None
    
    if networks is None:
        networks = ['asia', 'child', 'alarm']
    
    all_results = {
        'experiment': 'Task_3.1_MMHC_ACR',
        'timestamp': datetime.now().isoformat(),
        'networks': {},
        'summary': {}
    }
    
    for network in networks:
        print(f"\n{'#' * 60}")
        print(f"# Running MMHC + ACR on {network.upper()}")
        print(f"{'#' * 60}")
        
        try:
            result = experimenter.experiment_p2_2_mmhc_acr(network, sample_size)
            if result:
                all_results['networks'][network] = result
                
                # 计算改进
                mmhc_shd = result.get('mmhc_shd')
                mmhc_acr_shd = result.get('mmhc_acr_shd')
                
                if mmhc_shd is not None and mmhc_acr_shd is not None:
                    improvement = mmhc_shd - mmhc_acr_shd
                    improvement_pct = (improvement / mmhc_shd * 100) if mmhc_shd > 0 else 0
                    
                    all_results['summary'][network] = {
                        'mmhc_shd': mmhc_shd,
                        'mmhc_acr_shd': mmhc_acr_shd,
                        'improvement': improvement,
                        'improvement_pct': improvement_pct
                    }
        except Exception as e:
            print(f"❌ Error on {network}: {e}")
            import traceback
            traceback.print_exc()
            all_results['networks'][network] = {'error': str(e)}
    
    # 打印汇总表格
    print(f"\n{'=' * 70}")
    print("📊 MMHC + ACR 实验汇总")
    print(f"{'=' * 70}")
    print(f"{'Network':<12} {'MMHC SHD':<12} {'MMHC+ACR SHD':<15} {'Improvement':<12} {'%':<10}")
    print(f"{'-' * 70}")
    
    for network, summary in all_results['summary'].items():
        print(f"{network.upper():<12} {summary['mmhc_shd']:<12} {summary['mmhc_acr_shd']:<15} "
              f"{summary['improvement']:+d}          {summary['improvement_pct']:.1f}%")
    
    print(f"{'=' * 70}")
    
    # 保存结果
    output_file = os.path.join(RESULTS_DIR, 'task_3_1_mmhc_acr_results.json')
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(all_results, f, indent=2, ensure_ascii=False)
    print(f"\n💾 Results saved to: {output_file}")
    
    return all_results


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Task 3.1: MMHC + ACR Experiments')
    parser.add_argument('--networks', type=str, nargs='+', default=['asia', 'child', 'alarm'],
                        help='Networks to test (default: asia child alarm)')
    parser.add_argument('--sample_size', type=int, default=1000,
                        help='Sample size (default: 1000)')
    args = parser.parse_args()
    
    run_mmhc_acr_experiments(args.networks, args.sample_size)
