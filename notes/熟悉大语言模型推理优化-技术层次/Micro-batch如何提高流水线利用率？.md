---
created: '2025-10-19'
last_reviewed: null
next_review: '2025-10-19'
review_count: 0
difficulty: medium
mastery_level: 0.0
tags:
- 熟悉大语言模型推理优化-技术层次
- 熟悉大语言模型推理优化-技术层次/Micro-batch如何提高流水线利用率？.md
related_outlines: []
---
# Micro-batch如何提高流水线利用率？

## 面试标准答案

Micro-batch通过将一个大batch分成多个小batch，使流水线能够同时处理多个mini-batch，从而让各stage保持忙碌状态。当有M个micro-batch时，气泡占比从接近100%降至(P-1)/(P-1+M)。例如4层流水线，1个batch时气泡75%，但有16个micro-batch时气泡仅16%。代价是需要更多的前向传播次数和通信次数，以及稍高的显存占用。最优micro-batch数量通常是流水线深度的4-8倍。

---

## 详细讲解

### 1. 无Micro-batch的问题

```
单batch直接流水线:
GPU0: [Batch] ████████      (空闲)
GPU1:         [Batch] ████████     (空闲)
GPU2:                 [Batch] ████████    (空闲)
GPU3:                         [Batch] ████████

总时间: 4T
有效计算: 4T
空闲时间: 3T (每个GPU平均空闲75%)
```

### 2. Micro-batch的作用

```
分成4个micro-batches:
GPU0: [m1][m2][m3][m4]
GPU1:     [m1][m2][m3][m4]
GPU2:         [m1][m2][m3][m4]
GPU3:             [m1][m2][m3][m4]

总时间: 7T
气泡: 3T
效率提升: 从25% → 57%
```

### 3. 数学关系

```python
# 气泡占比
bubble_ratio = (P - 1) / (P - 1 + M)

# 流水线效率
efficiency = M / (P - 1 + M)

# 有效并行度
parallel_efficiency = M / (P - 1 + M)

其中:
P = 流水线深度
M = micro-batch数量
```

### 4. 最优Micro-batch数量

**经验法则**:
```python
# 目标效率 >= 90%
# (P-1)/(P-1+M) <= 0.1
# M >= 9(P-1)

recommended_M = 4 * P  # 保守
optimal_M = 8 * P      # 理想

# 考虑显存限制
max_M = available_memory / (microbatch_activation_size)

final_M = min(optimal_M, max_M)
```

### 5. Micro-batch大小选择

```python
global_batch_size = 32
num_microbatches = 32
microbatch_size = global_batch_size // num_microbatches = 1

# 权衡:
# - 小micro-batch: 更多并行，但overhead大
# - 大micro-batch: 气泡多，但单次计算效率高

# 实践
if pp_depth <= 4:
    microbatch_size = 2-4
else:
    microbatch_size = 1-2
```

### 6. 显存与Micro-batch的权衡

**GPipe**: 需要存储所有micro-batch的激活
```
显存 = M × activation_per_microbatch
```

**1F1B**: 只需存储P个micro-batch
```
显存 = P × activation_per_microbatch
M可以更大!
```

### 7. 实际效果

```python
# GPT-3 175B, PP=8
配置1: M=8
- 气泡: 46.7%
- 吞吐: 低

配置2: M=32  
- 气泡: 17.9%
- 吞吐: 高 ✓

配置3: M=128
- 气泡: 5.2%
- 但显存不足 (GPipe)
- 需要1F1B调度
```

Micro-batch是流水线并行提高效率的核心技术，需要根据显存和气泡目标选择合适的数量。


---

## 相关笔记
<!-- 自动生成 -->

暂无相关笔记

