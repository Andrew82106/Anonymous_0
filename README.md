# LLM Bayesian - Abstract Causal Reasoning (ACR) Framework

基于**抽象因果推理 (Abstract Causal Reasoning, ACR)** 的 LLM 因果发现框架

---

## 📖 项目简介

本项目实现了一个创新的因果发现方法，通过将**统计特征翻译为自然语言叙事**，让大型语言模型 (LLM) 在**完全脱敏**的条件下推断变量之间的因果关系。

### 核心思想
传统的 LLM 因果推断研究常被质疑只是"记住"了训练数据中的共现关系（如"吸烟→癌症"）。  
我们的方法通过：
1. **匿名化变量**（Var_A, Var_B）
2. **提取统计行为特征**（残差独立性、分布形态、拟合度）
3. **翻译为自然语言叙事**（如"A->B 的误差几乎是随机的，但 B->A 的误差中仍包含 B 的痕迹"）

强制 LLM 依靠**逻辑推理**而非语义知识来判断因果方向。

---

## 🎯 核心创新

### 1. 多维度统计特征提取器 (Multi-Dimensional Stat-to-Lang Translator)

本项目不仅仅是单一的 LiNGAM 或 ANM 实现，而是一个**自适应的统计特征提取系统**，能够根据数据类型自动选择最优策略：

| 数据类型 | 核心理论 | 关键指标 | 适用场景 |
|---------|---------|---------|---------|
| **连续变量** | LiNGAM | 偏度、峰度（非高斯性） | 线性因果关系 |
| | Non-linear ANM | 互信息 (MI)、多项式拟合 R² | 非线性因果关系 |
| | 稳定性分析 | 异方差性检测 | 捕捉反向拟合的不稳定性 |
| **离散变量** | 信息论 | 条件熵、边际熵 | 真实世界离散数据（bnlearn） |
| | 预测能力 | 逻辑回归准确率 | 分类变量的因果判断 |

**关键突破**：将数值统计特征（如 MI=0.35, H(Y|X)=0.72）转化为自然语言叙事，使 LLM 能够像人类专家一样进行因果推理。

### 2. LLM 作为"元统计学家" (LLM as Meta-Statistician)

LLM 不再是简单的分类器，而是一个能够：
- **综合多源证据**：权衡来自熵、拟合度、残差独立性等多个维度的证据。
- **处理矛盾信号**：在复杂度不对称与残差独立性冲突时做出合理判断。
- **输出可解释推理链**：提供完整的推理过程，而非黑盒决策。

---

## 🚀 快速开始

### 安装依赖
```bash
pip install numpy scipy scikit-learn pandas pydantic openai zhipuai pyyaml
```

### 配置模型
编辑 `llms/config.yaml`，设置默认使用的模型：
```yaml
used_model: "deepseek-ai/DeepSeek-V3.1"
```

### 运行实验
```bash
# 使用默认配置
python3 run_experiment.py

# 指定参数
python3 run_experiment.py --model "gpt-4-turbo" --samples 1000 --output results.json
```

### 输出
- **控制台**: 实时显示推理进度和准确率
- **JSON 文件**: `experiment_results.json` - 完整的推理结果

---

## 📁 项目结构

```
LLMBayesian/
├── 📂 background/                # 项目背景文档
│   └── task.md                   # 实验计划和进度追踪
├── 📂 results/                   # 实验结果存储
│   ├── experiment_results.json   # 合成数据实验结果
│   └── real_network_results.json # 真实网络测试结果
├── 📂 tests/                     # 测试脚本
│   ├── run_experiment.py         # 运行合成数据实验
│   └── test_real_networks.py     # 测试真实贝叶斯网络
├── 📂 utils_set/                 # 核心功能模块
│   ├── stat_translator.py        # 统计特征 -> 自然语言叙事（支持 HSIC）
│   ├── data_generator.py         # 合成因果数据生成器
│   ├── causal_reasoning_engine.py # 端到端推理引擎
│   ├── prompts.py                # Prompt 模板库（Sherlock Holmes 风格）
│   ├── causal_inference_schema.py # Pydantic 响应模型
│   └── utils.py                  # 配置加载工具
├── 📂 llms/                      # LLM 管理系统
│   ├── manager.py                # LLM 管理器
│   ├── config.yaml               # 模型配置（支持 OpenAI/ZhipuAI/ModelScope）
│   ├── base.py                   # LLM 基类
│   └── providers/                # 各提供商实现
├── README.md                     # 本文件
├── MODIFICATION_SUMMARY.md       # 最新修改总结
└── PROJECT_STRUCTURE.md          # 详细项目结构说明
```

💡 **详细的文件组织说明请参见** [`PROJECT_STRUCTURE.md`](./PROJECT_STRUCTURE.md)

---

## 🧪 实验结果

### 合成数据集 (Synthetic Data)
- **准确率**: **66.7% (2/3)** 因果案例
- ✅ **LiNGAM** (线性非高斯): 正确
- ✅ **Reverse** (反向因果): 正确（通过偏度 + LiNGAM 原理解决）
- ❌ **ANM** (非线性): 预测错误（特定分布下的挑战）
- ✅ **Confounder/Independent**: 正确识别为 Unclear

### 真实世界数据 (Real-World Benchmarks)
- **Asia 网络** (离散变量): **80% (4/5)** 🎉
  - 成功识别: `smoke->lung`, `tub->either`, `either->xray`, `bronc->dysp`
  - 失败: `either->dysp`
- **Sprinkler 网络** (离散变量): 0% (谨慎返回 Unclear)

### 关键发现
1. **互信息 (MI) 的威力**: 相比 Pearson 相关，MI 能更好地捕捉非线性残差依赖。
2. **边际熵** 对离散变量判断有显著贡献（Asia 从 60% 提升至 80%）。
3. **LLM 推理能力**: LLM 能够在复杂的多维证据中进行权衡，展现出类似人类专家的推理能力。

---

## 🔧 使用示例

### 1. 运行完整实验
```bash
cd tests
python run_experiment.py --model gpt-4-turbo --samples 1000
```

### 2. 测试真实网络
```bash
cd tests
python test_real_networks.py
```

### 3. 代码集成示例

**生成数据**：
```python
from utils_set.data_generator import CausalDataGenerator

generator = CausalDataGenerator(random_seed=42)
datasets = generator.generate_batch(n_samples=500)
```

**分析单个数据对**：
```python
from utils_set.stat_translator import StatTranslator

translator = StatTranslator()
stats = translator.analyze(X, Y)
narrative = translator.generate_narrative(stats)
print(narrative)
```

**使用 LLM 推理**：
```python
from utils_set.causal_reasoning_engine import CausalReasoningEngine

engine = CausalReasoningEngine(model_name="deepseek-chat")
results = engine.run_experiment(datasets, save_results=True)
```

### 4. 切换 LLM 模型
方法 1 - 修改配置文件 `llms/config.yaml`：
```yaml
used_model: "gpt-4-turbo"
```

方法 2 - 命令行指定：
```bash
cd tests
python run_experiment.py --model "gpt-4-turbo"
```

方法 3 - 代码中指定：
```python
engine = CausalReasoningEngine(model_name="gpt-4-turbo")
```

---

## 📊 数据集类型

### 合成数据集（已实现）
1. **LiNGAM**: 线性非高斯加性噪声 (`A -> B: B = 0.8A + uniform_noise`)
2. **ANM**: 非线性加性噪声 (`A -> B: B = tanh(A) + 0.5*cos(A) + noise`)
3. **Confounder**: 混杂因素 (`Z -> A, Z -> B`)
4. **Independent**: 统计独立 (`A ⊥ B`)
5. **Reverse**: 反向因果 (`B -> A`)

### 真实数据集（规划中）
- bnlearn 经典网络：Sprinkler, Asia, Alarm
- Tubingen Cause-Effect Pairs

---

## 🧠 Prompt 设计

### Sherlock Holmes Prompt
```
你是一位精通统计学和因果推理的侦探...

## 统计分析报告
{narrative}

## 推理要求
基于 LiNGAM 和 ANM 原理，判断因果方向...
```

支持多种模板：
- `sherlock`: 完整版（默认）
- `simple`: 简化版
- `residual_only`: 消融研究（仅残差信息）

---

## 🛠️ 配置

### LLM 模型配置 (`llms/config.yaml`)
```yaml
models:
  text_models:
    - name: "deepseek-chat"
      provider: "openai"
      api_key: "your-api-key"
      base_url: "https://api.example.com/v1"
      temperature: 0.7
```

支持的提供商：
- OpenAI (GPT-4, GPT-3.5)
- ZhipuAI (GLM-4)
- ModelScope (Qwen, DeepSeek-V3, MiniMax)

---

## 📈 下一步工作

### 短期
- [ ] 在 StatTranslator 中加入非线性拟合（GAM、多项式）
- [ ] 使用 bnlearn 真实网络进行测试
- [ ] 消融研究：不同 Prompt 和模型的影响

### 中期
- [ ] 实现基准算法（PC, GES, 标准 LiNGAM）进行对比
- [ ] 实现辛普森悖论检测（混杂因素识别）
- [ ] 完整的评估指标（Precision, Recall, F1, AUROC）

### 长期
- [ ] 实现"元认知仲裁"机制（创新点2）
- [ ] 在大规模真实数据集上验证
- [ ] 撰写论文并投稿

---

## 📝 论文进度

详见 [task.md](task.md)

**当前阶段**: 第三阶段已完成，进入第四阶段（评估与分析）

---

## 🤝 贡献

欢迎提交 Issue 和 Pull Request！

---

## 📄 许可证

MIT License

---

## 🙏 致谢

- bnlearn: https://github.com/erdogant/bnlearn
- LiNGAM 理论: Shimizu et al. (2006)
- ANM 理论: Hoyer et al. (2009)
