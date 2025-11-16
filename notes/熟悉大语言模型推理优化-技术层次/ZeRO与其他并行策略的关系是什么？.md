---
created: '2025-10-19'
last_reviewed: null
next_review: '2025-10-19'
review_count: 0
difficulty: medium
mastery_level: 0.0
tags:
- 熟悉大语言模型推理优化-技术层次
- 熟悉大语言模型推理优化-技术层次/ZeRO与其他并行策略的关系是什么？.md
related_outlines: []
---
# ZeRO与其他并行策略的关系是什么？

## 面试标准答案

ZeRO是数据并行的增强版本，通过分片优化器状态、梯度和参数来减少冗余。可与其他并行策略组合：ZeRO+TP在TP组内做ZeRO数据并行；ZeRO+PP每个stage内用ZeRO；ZeRO-3类似参数分片可视为数据并行版的TP。关键区别是ZeRO动态gather参数(All-Gather)而TP静态切分，ZeRO训练时每GPU存完整梯度而TP只存局部。实践中ZeRO常在训练时与TP+PP组合，推理较少用ZeRO。

---

## 详细讲解

### ZeRO vs 数据并行

```python
# 标准DP
每个GPU: 完整模型副本
通信: 梯度All-Reduce

# ZeRO Stage 3
每个GPU: 模型参数的1/N
通信: 参数All-Gather + 梯度Reduce-Scatter

# ZeRO = 显存优化的DP
```

### ZeRO vs 张量并行

```python
# TP (静态切分)
前向: 每GPU计算部分维度，All-Reduce聚合
参数: 静态分片，不移动

# ZeRO-3 (动态gather)
前向: All-Gather完整参数，每GPU独立计算
参数: 动态gather，计算后释放

# 区别:
# TP: 低通信(2×All-Reduce)，需要代码改动
# ZeRO: 高通信(All-Gather每层)，无需改代码
```

### 组合使用

```python
# ZeRO + TP (推荐)
# 64 GPUs = TP(8) × ZeRO-DP(8)

# TP组内: 张量并行
tp_group = [0-7], [8-15], ..., [56-63]

# ZeRO组间: 数据并行 + ZeRO优化
zero_dp_group = [0,8,16,...,56], [1,9,17,...,57], ...

# 效果:
# - TP减少单卡显存(模型参数)
# - ZeRO减少冗余(优化器状态)
```

### ZeRO + PP

```python
# 每个PP stage内用ZeRO
# 64 GPUs = PP(8) × ZeRO-DP(8)

stage_0_gpus = [0-7]    # ZeRO在这8个GPU间
stage_1_gpus = [8-15]   # ZeRO在这8个GPU间
...

# PP: 跨stage流水线
# ZeRO: stage内显存优化
```

### 实际应用

```
训练大模型:
- ZeRO-1/2: 几乎总是启用
- ZeRO-3: 显存极限时使用

推理:
- 通常不用ZeRO (增加延迟)
- 除非模型超大无法用TP容纳
```

### DeepSpeed ZeRO配置

```json
{
  "zero_optimization": {
    "stage": 2,
    "offload_optimizer": {
      "device": "cpu"
    },
    "overlap_comm": true
  },
  "tensor_parallel": {
    "size": 8
  }
}
```

ZeRO是训练时的显存优化利器，推理时价值有限。


---

## 相关笔记
<!-- 自动生成 -->

- [ZeRO的三个阶段分别优化什么？](notes/熟悉大语言模型推理优化-技术层次/ZeRO的三个阶段分别优化什么？.md) - 相似度: 31% | 标签: 熟悉大语言模型推理优化-技术层次, 熟悉大语言模型推理优化-技术层次/ZeRO的三个阶段分别优化什么？.md

