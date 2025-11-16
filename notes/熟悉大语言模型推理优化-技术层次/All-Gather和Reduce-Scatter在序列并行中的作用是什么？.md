---
created: '2025-10-19'
last_reviewed: null
next_review: '2025-10-19'
review_count: 0
difficulty: medium
mastery_level: 0.0
tags:
- 熟悉大语言模型推理优化-技术层次
- 熟悉大语言模型推理优化-技术层次/All-Gather和Reduce-Scatter在序列并行中的作用是什么？.md
related_outlines: []
---
# All-Gather和Reduce-Scatter在序列并行中的作用是什么？

## 面试标准答案

在序列并行中，All-Gather和Reduce-Scatter主要用于反向传播的梯度处理。All-Gather收集各GPU的序列分片梯度形成完整梯度，Reduce-Scatter则将梯度规约后重新分片。前向传播主要使用All-to-All进行维度转换。具体：前向用All-to-All在特征/序列分片间切换，反向传播中对序列分片的梯度用All-Gather汇总或Reduce-Scatter分发。这确保了梯度正确传播并保持分片一致性。

---

## 详细讲解

### 前向vs反向的通信

**前向传播**:
- 主要使用：All-to-All
- 目的：维度转换（特征↔序列）

**反向传播**:
- All-Gather：收集分片梯度
- Reduce-Scatter：规约并重新分片
- 配合All-to-All的反向传播

### 梯度处理

```python
# 前向: 序列分片 → 特征分片
x_feat = all_to_all(x_seq)

# 反向: 梯度需要转回
grad_x_seq = all_to_all(grad_x_feat)

# 如需完整梯度
grad_full = all_gather(grad_x_seq)

# 或规约后分片
grad_reduced = reduce_scatter(grad_x_seq)
```

### 实际使用

训练时重要，推理时不涉及反向传播，主要关注All-to-All。


---

## 相关笔记
<!-- 自动生成 -->

- [序列并行需要哪些集合通信操作？](notes/熟悉大语言模型推理优化-技术层次/序列并行需要哪些集合通信操作？.md) - 相似度: 31% | 标签: 熟悉大语言模型推理优化-技术层次, 熟悉大语言模型推理优化-技术层次/序列并行需要哪些集合通信操作？.md

