---
created: '2025-10-19'
last_reviewed: null
next_review: '2025-10-19'
review_count: 0
difficulty: medium
mastery_level: 0.0
tags:
- 熟悉大语言模型推理优化-技术层次
- 熟悉大语言模型推理优化-技术层次/序列并行适用于哪些算子（LayerNorm、Dropout）？.md
related_outlines: []
---
# 序列并行适用于哪些算子（LayerNorm、Dropout）？

## 面试标准答案

序列并行适用于在特征维度进行操作的算子，主要包括：1)LayerNorm-在hidden维度归一化，可在序列分片上独立计算；2)Dropout-逐元素随机丢弃，完全独立；3)残差连接-逐元素加法；4)激活函数(GeLU、ReLU等)-逐元素操作；5)Bias加法。不适用于需要完整序列的算子如Attention的QK^T计算和GEMM的某些模式。这些算子在序列维度分片后可节省大量激活显存。

---

## 详细讲解

### 1. 适用算子

**LayerNorm**:
```python
# 在hidden维度归一化
def layer_norm_seq_parallel(x):
    # x: [B, S/N, H] - 序列分片
    mean = x.mean(dim=-1, keepdim=True)
    var = x.var(dim=-1, keepdim=True)
    normalized = (x - mean) / sqrt(var + eps)
    return gamma * normalized + beta

# 完全独立，无需通信
```

**Dropout**:
```python
# 逐元素随机丢弃
dropout_output = F.dropout(x, p=0.1)
# 在序列分片上独立计算
```

**残差连接**:
```python
# 逐元素加法
output = x + residual
# x和residual都是[B, S/N, H]，直接相加
```

**激活函数**:
```python
# GeLU、ReLU等
output = gelu(x)
# 逐元素操作，序列分片不影响
```

### 2. 不适用算子

**Attention**:
```python
# QK^T需要完整序列
attn_scores = Q @ K.transpose(-2, -1)  # [B, S, S]
# 如果Q和K都是序列分片，无法计算完整的attention matrix
```

**某些GEMM**:
```python
# 特征维度的矩阵乘法
output = x @ W  # x: [B, S, H], W: [H, H']
# 需要完整的H维度
```

### 3. 显存节省

```python
# 标准Transformer层激活
layernorm_1: [B, S, H]
dropout_1: [B, S, H]
layernorm_2: [B, S, H]
dropout_2: [B, S, H]
总计: 4 × B × S × H

# 序列并行(N=8)
layernorm_1: [B, S/8, H]
dropout_1: [B, S/8, H]
layernorm_2: [B, S/8, H]
dropout_2: [B, S/8, H]
总计: 4 × B × S/8 × H

节省: 87.5%
```

### 4. 实现要点

```python
# 在Megatron中的使用
def transformer_layer_with_sp(x):
    # x: [B, S/N, H] 序列分片输入
    
    # LayerNorm - 直接在分片上计算
    x_norm = layer_norm(x)
    
    # 转换为特征分片用于attention
    x_feat = all_to_all(x_norm)  # [B, S, H/N]
    
    # Attention计算...
    attn_out = attention(x_feat)
    
    # 转回序列分片
    attn_seq = all_to_all(attn_out)  # [B, S/N, H]
    
    # Dropout - 在序列分片上
    attn_seq = dropout(attn_seq)
    
    # 残差 - 在序列分片上
    x = x + attn_seq
    
    return x
```

### 5. 总结

**适用**: LayerNorm、Dropout、残差、激活函数、Bias
**不适用**: Attention计算、部分GEMM
**收益**: 显存节省87.5% (8-way并行)


---

## 相关笔记
<!-- 自动生成 -->

- [序列并行如何沿序列维度切分？](notes/熟悉大语言模型推理优化-技术层次/序列并行如何沿序列维度切分？.md) - 相似度: 33% | 标签: 熟悉大语言模型推理优化-技术层次, 熟悉大语言模型推理优化-技术层次/序列并行如何沿序列维度切分？.md
- [序列并行如何与张量并行结合？](notes/熟悉大语言模型推理优化-技术层次/序列并行如何与张量并行结合？.md) - 相似度: 31% | 标签: 熟悉大语言模型推理优化-技术层次, 熟悉大语言模型推理优化-技术层次/序列并行如何与张量并行结合？.md

