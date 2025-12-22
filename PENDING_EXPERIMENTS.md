# 待执行实验任务

> 这些实验需要在有足够计算资源的设备上运行，预计每个网络需要 30-60 分钟。

## 环境要求

```bash
# 确保安装了必要的依赖
pip install pgmpy causal-learn bnlearn pandas numpy

# 确保 LLM API 配置正确
# 检查 llms/config.yaml 中的 API 密钥
```

## 实验 1: MMHC + ACR 完整实验

### 目标
在 Asia/Child/Alarm 网络上运行 MMHC + ACR 混合流水线，验证 Requirements 3.3

### 已完成
- ✅ Asia 网络 (结果在 `results/asia_mmhc_hybrid.json`)

### 待完成

#### 1.1 Child 网络
```bash
python tests/test_hybrid_pipeline.py --network child --base_algorithm mmhc --sample_size 1000
```

**注意**: Child 网络在 bnlearn 中不可用，需要使用 pgmpy 版本：
```bash
python tests/run_mmhc_acr_experiments.py --networks child --sample_size 1000
```

#### 1.2 Alarm 网络
```bash
python tests/test_hybrid_pipeline.py --network alarm --base_algorithm mmhc --sample_size 1000
```

或使用 pgmpy 版本：
```bash
python tests/run_mmhc_acr_experiments.py --networks alarm --sample_size 1000
```

### 预期输出
- `results/child_mmhc_hybrid.json`
- `results/alarm_mmhc_hybrid.json`

### 结果格式
```json
{
  "network": "child/alarm",
  "base_algorithm": "mmhc",
  "sample_size": 1000,
  "base_metrics": {
    "shd": <int>,
    "skeleton": {"precision": <float>, "recall": <float>, "f1": <float>},
    "orientation": {"precision": <float>, "recall": <float>, "f1": <float>}
  },
  "hybrid_metrics": {
    "shd": <int>,
    "skeleton": {"precision": <float>, "recall": <float>, "f1": <float>},
    "orientation": {"precision": <float>, "recall": <float>, "f1": <float>}
  },
  "details": {
    "total_edges": <int>,
    "undirected_count": <int>,
    "acr_updates": <int>,
    "acr_unclear": <int>
  }
}
```

---

## 实验 2: 基座算法通用性验证

### 目标
对比 PC + ACR 与 MMHC-Skeleton + ACR，验证 Requirements 3.4

### 已完成
- ✅ Asia PC + ACR (结果在 `results/asia_pc_hybrid.json`)
- ✅ Asia MMHC + ACR (结果在 `results/asia_mmhc_hybrid.json`)

### 待完成

#### 2.1 Child 网络 PC + ACR
```bash
python tests/test_hybrid_pipeline.py --network child --base_algorithm pc --sample_size 1000
```

#### 2.2 汇总对比
实验 1 和 2.1 完成后，运行：
```bash
python tests/run_base_algorithm_comparison.py
```

### 预期输出
- `results/child_pc_hybrid.json`
- `results/task_3_2_base_algorithm_comparison.json` (更新)

---

## 实验 3: 全算法对比 (可选)

### 目标
在同一网络上运行所有基座算法，生成完整对比

```bash
# 在 Asia 上运行所有算法
python tests/test_hybrid_pipeline.py --network asia --all_algorithms

# 在 Alarm 上运行所有算法
python tests/test_hybrid_pipeline.py --network alarm --all_algorithms
```

### 预期输出
- `results/asia_all_algorithms_summary.json`
- `results/alarm_all_algorithms_summary.json`

---

## 完成后操作

1. 将所有结果文件同步回主仓库
2. 更新 `results/task_3_1_mmhc_acr_results.json` 添加 Child 和 Alarm 结果
3. 运行 `python tests/run_base_algorithm_comparison.py` 生成最终对比报告

## 注意事项

1. **MMHC 算法很慢**: 在 Alarm (37 节点) 上可能需要 30+ 分钟
2. **LLM API 调用**: 每条边需要一次 API 调用，注意 rate limit
3. **内存使用**: 大型网络可能需要 4GB+ 内存
4. **超时处理**: 如果命令超时，进程可能在后台继续运行，用 `ps aux | grep python` 检查

## 当前实验状态

| 网络 | PC+ACR | MMHC+ACR | Dual PC+ACR | FCI+ACR |
|------|--------|----------|-------------|---------|
| Asia | ✅ | ✅ | ✅ | ✅ |
| Child | ✅ | ⏳ (Sample 100) | ❌ | ✅ |
| Alarm | ✅ | ⏳ (Sample 100) | ❌ | ❌ |
| Sachs | ✅ | ❌ | ✅ | ❌ |
