---
created: '2025-10-19'
last_reviewed: null
next_review: '2025-10-19'
review_count: 0
difficulty: medium
mastery_level: 0.0
tags:
- Transformer
- Transformer/输入序列到QKV的映射过程.md
related_outlines: []
---
# 输入序列到QKV的映射过程

## 面试标准答案

输入序列到QKV的映射过程：
1. **输入嵌入**：将token序列转换为d_model维的嵌入向量
2. **位置编码**：加入位置信息，得到最终输入表示
3. **线性投影**：通过三个独立的权重矩阵W_Q、W_K、W_V将输入映射为Query、Key、Value
4. **多头分割**：将QKV按头数切分，每个头处理不同的表示子空间
5. **注意力计算**：使用QKV进行缩放点积注意力计算

核心是通过可学习的线性变换将同一输入映射到不同的语义空间，实现自注意力机制。

## 详细技术解析

### 1. 完整映射流程

#### 1.1 输入预处理
```python
# 步骤1：Token嵌入
token_ids = [101, 2054, 2003, 102]  # [CLS, what, is, SEP]
embeddings = embedding_layer(token_ids)  # (seq_len, d_model)

# 步骤2：位置编码
pos_encoding = positional_encoding(seq_len, d_model)
input_repr = embeddings + pos_encoding  # (seq_len, d_model)
```

#### 1.2 QKV线性映射
```python
# 输入：X (batch_size, seq_len, d_model)
# 三个独立的线性变换
Q = X @ W_Q + b_Q  # (batch_size, seq_len, d_model)
K = X @ W_K + b_K  # (batch_size, seq_len, d_model)  
V = X @ W_V + b_V  # (batch_size, seq_len, d_model)

# 权重矩阵维度
# W_Q, W_K, W_V: (d_model, d_model)
# b_Q, b_K, b_V: (d_model,)
```

### 2. 多头注意力中的处理

#### 2.1 多头分割
```python
def split_heads(x, num_heads):
    """将输入分割为多个注意力头"""
    batch_size, seq_len, d_model = x.shape
    d_head = d_model // num_heads
    
    # 重塑并转置
    x = x.view(batch_size, seq_len, num_heads, d_head)
    return x.transpose(1, 2)  # (batch_size, num_heads, seq_len, d_head)

# 应用到QKV
Q_multi = split_heads(Q, num_heads)  # (batch, heads, seq_len, d_head)
K_multi = split_heads(K, num_heads)  # (batch, heads, seq_len, d_head)
V_multi = split_heads(V, num_heads)  # (batch, heads, seq_len, d_head)
```

#### 2.2 注意力计算
```python
# 缩放点积注意力
def scaled_dot_product_attention(Q, K, V, mask=None):
    d_k = Q.size(-1)
    
    # 计算注意力分数
    scores = torch.matmul(Q, K.transpose(-2, -1)) / math.sqrt(d_k)
    
    # 应用掩码（如果有）
    if mask is not None:
        scores = scores.masked_fill(mask == 0, -1e9)
    
    # Softmax归一化
    attention_weights = F.softmax(scores, dim=-1)
    
    # 加权求和
    output = torch.matmul(attention_weights, V)
    return output, attention_weights
```

### 3. 维度变化详解

#### 3.1 各阶段维度追踪
```
输入序列: (batch_size, seq_len)
↓ Token嵌入
嵌入向量: (batch_size, seq_len, d_model)
↓ 位置编码
输入表示: (batch_size, seq_len, d_model)
↓ 线性变换
Q/K/V: (batch_size, seq_len, d_model)
↓ 多头分割
Q/K/V多头: (batch_size, num_heads, seq_len, d_head)
↓ 注意力计算
注意力输出: (batch_size, num_heads, seq_len, d_head)
```

#### 3.2 参数量计算
```python
# 假设 d_model = 512, num_heads = 8
d_model = 512
num_heads = 8
d_head = d_model // num_heads  # 64

# QKV投影参数
params_QKV = 3 * (d_model * d_model + d_model)  # 3 * (512*512 + 512)
print(f"QKV投影参数量: {params_QKV:,}")  # 786,944

# 输出投影参数
params_output = d_model * d_model + d_model  # 512*512 + 512
print(f"输出投影参数量: {params_output:,}")  # 262,656
```

### 4. 关键设计原理

#### 4.1 为什么需要三个不同的投影矩阵？
- **查询矩阵Q**：表示"我在寻找什么信息"
- **键矩阵K**：表示"我能提供什么信息"  
- **值矩阵V**：表示"具体的信息内容"

这种设计实现了**非对称相似度计算**，比简单的内积更具表达力。

#### 4.2 多头的必要性
```python
# 单头注意力的局限
single_head_attention = softmax(QK^T/√d_k)V

# 多头注意力的优势
multi_head = concat(head_1, head_2, ..., head_h)W_O
# 每个head_i = attention(QW_Q^i, KW_K^i, VW_V^i)
```

**优势**：
- 不同头关注不同类型的依赖关系
- 增加模型的表示能力
- 允许并行计算

### 5. 实际示例

#### 5.1 具体数值示例
```python
import torch

# 模拟输入
batch_size, seq_len, d_model = 2, 4, 8
num_heads = 2

# 输入序列表示
X = torch.randn(batch_size, seq_len, d_model)
print(f"输入形状: {X.shape}")

# 权重矩阵（随机初始化）
W_Q = torch.randn(d_model, d_model) * 0.1
W_K = torch.randn(d_model, d_model) * 0.1  
W_V = torch.randn(d_model, d_model) * 0.1

# QKV变换
Q = X @ W_Q  # (2, 4, 8)
K = X @ W_K  # (2, 4, 8)
V = X @ W_V  # (2, 4, 8)

print(f"Q形状: {Q.shape}")
print(f"K形状: {K.shape}")  
print(f"V形状: {V.shape}")
```

#### 5.2 注意力可视化
```python
# 计算注意力权重
d_k = Q.size(-1)
attention_scores = torch.matmul(Q, K.transpose(-2, -1)) / math.sqrt(d_k)
attention_weights = F.softmax(attention_scores, dim=-1)

print("注意力权重矩阵:")
print(attention_weights[0])  # 第一个样本的注意力矩阵
```

### 6. 优化技巧

#### 6.1 计算优化
- **矩阵乘法融合**：将QKV计算合并为单个矩阵乘法
- **内存优化**：使用梯度检查点减少显存占用
- **并行化**：多头计算天然支持并行

#### 6.2 数值稳定性
```python
# 防止梯度消失/爆炸的技巧
def scaled_dot_product_attention_stable(Q, K, V, mask=None):
    d_k = Q.size(-1)
    scores = torch.matmul(Q, K.transpose(-2, -1)) / math.sqrt(d_k)
    
    if mask is not None:
        scores = scores.masked_fill(mask == 0, -1e9)
    
    # 数值稳定的softmax
    max_scores = torch.max(scores, dim=-1, keepdim=True)[0]
    scores = scores - max_scores
    attention_weights = F.softmax(scores, dim=-1)
    
    output = torch.matmul(attention_weights, V)
    return output, attention_weights
```

### 7. 常见问题与解答

**Q: 为什么不直接使用输入作为QKV？**
A: 直接使用输入限制了表达能力，无法学习任务特定的查询-键-值关系。

**Q: d_head为什么通常设为d_model/num_heads？**
A: 保持总参数量不变的同时实现多头并行，是效率和性能的平衡。

**Q: QKV的初始化策略？**
A: 通常使用Xavier/Glorot初始化，确保梯度在反向传播时不会过大或过小。

通过这个完整的映射过程，Transformer能够将原始序列转换为富含语义信息的注意力表示，为后续的特征提取和任务预测奠定基础。

---

## 相关笔记
<!-- 自动生成 -->

暂无相关笔记

