---
created: '2025-10-19'
last_reviewed: null
next_review: '2025-10-19'
review_count: 0
difficulty: medium
mastery_level: 0.0
tags:
- 熟悉大语言模型推理优化-技术层次
- 熟悉大语言模型推理优化-技术层次/序列并行对KV_Cache管理的影响是什么？.md
related_outlines: []
---
# 序列并行对KV Cache管理的影响是什么？

## 面试标准答案

序列并行下，KV Cache也按序列维度分片，每个GPU只存储和管理自己负责的序列片段的KV Cache，显存占用降至1/N。挑战在于：1)自回归生成时新token的KV需要添加到正确的分片；2)Attention计算需要访问所有分片的KV，需要通信或分布式计算；3)动态长度管理更复杂。实践中常用Ring Attention模式或将KV Cache全量复制到各GPU，牺牲显存换取计算简化。推理场景较少使用序列并行。

---

## 详细讲解

### KV Cache分片

```python
# 标准KV Cache
kv_cache: [B, S, num_layers, 2, H]
显存: B × S × layers × 2 × H

# 序列并行(8-way)
kv_cache_local: [B, S/8, num_layers, 2, H]
显存/GPU: B × S/8 × layers × 2 × H

节省87.5%
```

### 生成时的更新

```python
# 新token生成
new_kv = compute_kv(new_token)

# 问题: 新token属于哪个GPU？
gpu_id = determine_owner(current_pos)

# 只有对应GPU更新cache
if rank == gpu_id:
    kv_cache_local[:, local_pos, ...] = new_kv
```

### Attention计算

```python
# 方案1: Ring Attention
# KV在GPU间循环，逐块计算

# 方案2: All-Gather KV
kv_full = all_gather(kv_cache_local)
attn = compute_attention(Q, kv_full)

# 方案3: 分布式Attention
# 每个GPU计算部分，再reduce
```

### 实际选择

推理通常不用序列并行，KV Cache问题是原因之一。训练时序列并行有价值。


---

## 相关笔记
<!-- 自动生成 -->

- [Ring_Attention中的序列并行如何实现？](notes/熟悉大语言模型推理优化-技术层次/Ring_Attention中的序列并行如何实现？.md) - 相似度: 31% | 标签: 熟悉大语言模型推理优化-技术层次, 熟悉大语言模型推理优化-技术层次/Ring_Attention中的序列并行如何实现？.md

