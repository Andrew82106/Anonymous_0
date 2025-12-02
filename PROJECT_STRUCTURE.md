# 项目结构说明 (Project Structure)

## 📁 文件夹组织

```
LLMBayesian/
├── 📂 background/              # 项目背景文档
│   └── task.md                 # 任务计划和进度追踪
│
├── 📂 results/                 # 实验结果存储
│   ├── experiment_results.json # 合成数据实验结果
│   └── real_network_results.json # 真实网络测试结果
│
├── 📂 tests/                   # 测试脚本
│   ├── run_experiment.py       # 运行合成数据实验
│   └── test_real_networks.py   # 测试真实贝叶斯网络
│
├── 📂 utils_set/               # 核心功能模块
│   ├── stat_translator.py      # 统计特征翻译器
│   ├── data_generator.py       # 合成数据生成器
│   ├── causal_reasoning_engine.py # 因果推理引擎
│   ├── prompts.py              # Prompt 模板库
│   ├── causal_inference_schema.py # 响应数据模型
│   └── utils.py                # 工具函数
│
├── 📂 llms/                    # LLM 管理系统
│   ├── manager.py              # LLM 管理器
│   ├── config.yaml             # 模型配置
│   ├── base.py                 # LLM 基类
│   └── providers/              # 各提供商实现
│       ├── openai_provider.py
│       ├── zhipuai_provider.py
│       └── modelscope_provider.py
│
├── README.md                   # 项目说明
├── MODIFICATION_SUMMARY.md     # 最新修改总结
└── debug.py                    # 调试脚本
```

## 🚀 快速开始

### 运行合成数据实验
```bash
cd tests
python run_experiment.py --model gpt-4-turbo --samples 1000
```

### 测试真实贝叶斯网络
```bash
cd tests
python test_real_networks.py
```

### 查看结果
```bash
# 合成数据结果
cat results/experiment_results.json

# 真实网络结果
cat results/real_network_results.json
```

## 📋 导入路径说明

### 在测试脚本中导入（从 tests/ 文件夹）
```python
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from utils_set.causal_reasoning_engine import CausalReasoningEngine
from utils_set.data_generator import CausalDataGenerator
from utils_set.stat_translator import StatTranslator
```

### 在根目录脚本中导入
```python
from utils_set.causal_reasoning_engine import CausalReasoningEngine
from utils_set.data_generator import CausalDataGenerator
from llms.manager import llm_manager
```

## 🔄 文件移动对照表

| 原位置 | 新位置 | 说明 |
|-------|--------|------|
| `task.md` | `background/task.md` | 任务文档 |
| `experiment_results.json` | `results/experiment_results.json` | 实验结果 |
| `real_network_results.json` | `results/real_network_results.json` | 网络测试结果 |
| `run_experiment.py` | `tests/run_experiment.py` | 实验脚本 |
| `test_real_networks.py` | `tests/test_real_networks.py` | 测试脚本 |
| `stat_translator.py` | `utils_set/stat_translator.py` | 核心模块 |
| `data_generator.py` | `utils_set/data_generator.py` | 核心模块 |
| `causal_reasoning_engine.py` | `utils_set/causal_reasoning_engine.py` | 核心模块 |
| `prompts.py` | `utils_set/prompts.py` | 核心模块 |
| `causal_inference_schema.py` | `utils_set/causal_inference_schema.py` | 核心模块 |
| `utils.py` | `utils_set/utils.py` | 核心模块 |

## 📝 注意事项

1. **结果文件路径**：所有实验结果现在默认保存到 `results/` 文件夹
2. **导入路径**：所有核心模块现在需要通过 `utils_set.` 前缀导入
3. **工作目录**：运行测试脚本时，建议在 `tests/` 文件夹内执行
4. **LLM 配置**：`llms/config.yaml` 位置未变，可直接使用
