"""
验证项目结构整理后的导入是否正常
"""

import sys
import os

print("="*80)
print("项目结构验证脚本")
print("="*80)

# 测试 1: 验证文件夹存在
print("\n📁 验证文件夹结构...")
folders = ['background', 'results', 'tests', 'utils_set', 'llms']
for folder in folders:
    path = os.path.join(os.path.dirname(__file__), folder)
    exists = os.path.exists(path)
    status = "✅" if exists else "❌"
    print(f"{status} {folder}/")

# 测试 2: 验证关键文件
print("\n📄 验证关键文件...")
files = [
    'background/task.md',
    'results/experiment_results.json',
    'results/real_network_results.json',
    'tests/run_experiment.py',
    'tests/test_real_networks.py',
    'utils_set/stat_translator.py',
    'utils_set/causal_reasoning_engine.py',
    'utils_set/data_generator.py',
    'PROJECT_STRUCTURE.md',
    'MODIFICATION_SUMMARY.md'
]
for file in files:
    path = os.path.join(os.path.dirname(__file__), file)
    exists = os.path.exists(path)
    status = "✅" if exists else "❌"
    print(f"{status} {file}")

# 测试 3: 验证导入
print("\n📦 验证模块导入...")
try:
    from utils_set.stat_translator import StatTranslator
    print("✅ StatTranslator 导入成功")
except Exception as e:
    print(f"❌ StatTranslator 导入失败: {e}")

try:
    from utils_set.data_generator import CausalDataGenerator
    print("✅ CausalDataGenerator 导入成功")
except Exception as e:
    print(f"❌ CausalDataGenerator 导入失败: {e}")

try:
    from utils_set.causal_reasoning_engine import CausalReasoningEngine
    print("✅ CausalReasoningEngine 导入成功")
except Exception as e:
    print(f"❌ CausalReasoningEngine 导入失败: {e}")

try:
    from llms.manager import llm_manager
    print("✅ LLMManager 导入成功")
except Exception as e:
    print(f"❌ LLMManager 导入失败: {e}")

# 测试 4: 快速功能测试
print("\n🧪 快速功能测试...")
try:
    import numpy as np
    translator = StatTranslator()
    X = np.random.randn(100)
    Y = 2 * X + np.random.randn(100) * 0.1
    stats = translator.analyze(X, Y)
    print("✅ StatTranslator.analyze() 运行成功")
except Exception as e:
    print(f"❌ StatTranslator.analyze() 失败: {e}")

try:
    generator = CausalDataGenerator(random_seed=42)
    datasets = generator.generate_batch(n_samples=10)
    print(f"✅ CausalDataGenerator.generate_batch() 成功生成 {len(datasets)} 个数据集")
except Exception as e:
    print(f"❌ CausalDataGenerator.generate_batch() 失败: {e}")

print("\n" + "="*80)
print("✅ 项目结构验证完成！")
print("="*80)
print("\n💡 下一步：")
print("   cd tests")
print("   python run_experiment.py --samples 100")
print("="*80)
