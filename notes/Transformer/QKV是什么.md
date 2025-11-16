---
created: '2025-10-19'
last_reviewed: null
next_review: '2025-10-19'
review_count: 0
difficulty: medium
mastery_level: 0.0
tags:
- Transformer
- Transformer/QKV是什么.md
related_outlines: []
---
# QKV是什么

## 面试标准答案

QKV是自注意力机制中的三个核心组件：
- **Query（查询）**：表示"当前位置想要什么信息"
- **Key（键）**：表示"每个位置能提供什么信息"  
- **Value（值）**：表示"每个位置的实际内容信息"

**计算过程**：通过Query和Key的点积计算注意力权重，然后用权重对Value进行加权求和，得到融合了全序列信息的新表示。

**数学公式**：$\text{Attention}(Q,K,V) = \text{softmax}(\frac{QK^T}{\sqrt{d_k}})V$

## 详细技术解析

### 1. QKV的生成过程

#### 线性变换
```python
# 输入序列 X: [seq_len, d_model]
Q = X @ W_Q  # [seq_len, d_k] - 查询矩阵
K = X @ W_K  # [seq_len, d_k] - 键矩阵  
V = X @ W_V  # [seq_len, d_v] - 值矩阵

# 变换矩阵
W_Q: [d_model, d_k]  # Query变换权重
W_K: [d_model, d_k]  # Key变换权重
W_V: [d_model, d_v]  # Value变换权重
```

#### 维度关系
```python
# 标准设置
d_k = d_v = d_model / num_heads
# 例如：d_model=512, num_heads=8
# 则 d_k = d_v = 64
```

### 2. 注意力计算步骤

#### 第一步：计算相似度分数
```python
# Query和Key做点积
scores = Q @ K.T  # [seq_len, seq_len]
# scores[i,j] 表示位置i对位置j的关注程度
```

#### 第二步：缩放处理
```python
# 避免梯度消失，稳定训练
scaled_scores = scores / sqrt(d_k)
```

#### 第三步：softmax归一化
```python
# 转换为概率分布
attention_weights = softmax(scaled_scores)  # [seq_len, seq_len]
# 每一行的权重和为1
```

#### 第四步：加权求和
```python
# 用注意力权重对Value加权
output = attention_weights @ V  # [seq_len, d_v]
```

### 3. 直观理解

#### 信息检索类比
```python
# 类比搜索引擎
Query:    "我想找关于猫的信息"
Key:      ["动物", "宠物", "狗", "猫", "鱼"]  
Value:    ["哺乳动物", "家养", "忠诚", "可爱", "游泳"]

# 计算相似度
相似度 = ["低", "中", "低", "高", "低"]
# Query "猫" 与 Key "猫" 最匹配

# 输出结果主要是Value "可爱"
```

#### 句子处理实例
```python
句子: "The cat sits on the mat"
位置:   0    1    2   3   4   5

当处理位置1 "cat"时：
Q[1]: 查询向量，表示"cat想要什么信息"
K[0-5]: 每个词的键向量，表示"能提供什么信息"  
V[0-5]: 每个词的值向量，表示"实际的语义内容"

注意力权重可能是：[0.1, 0.3, 0.4, 0.1, 0.05, 0.05]
                    ↑    ↑    ↑
                   The  cat  sits
# "cat"主要关注"sits"(动作)，部分关注自己和"The"
```

### 4. 关键特性

#### 位置无关性
```python
# Self-Attention是集合操作，天然没有位置信息
# 需要位置编码来补充
input_with_position = token_embedding + positional_encoding
```

#### 并行计算
```python
# 所有位置可以同时计算，不像RNN需要顺序处理
# 矩阵运算高度并行化
for i in range(seq_len):  # 可以并行
    output[i] = sum(attention_weights[i,j] * V[j] for j in range(seq_len))
```

#### 长距离依赖
```python
# 任意两个位置都可以直接建立连接
# 注意力权重直接建模远程依赖关系
attention_weights[0, 100] = 0.8  # 位置0直接关注位置100
```

### 5. 多头注意力中的QKV

#### 多个注意力头
```python
# 每个头有独立的QKV变换
for head in range(num_heads):
    Q_h = X @ W_Q[head]  # [seq_len, d_k]
    K_h = X @ W_K[head]  # [seq_len, d_k]
    V_h = X @ W_V[head]  # [seq_len, d_v]
    
    # 独立计算注意力
    head_output = attention(Q_h, K_h, V_h)

# 拼接所有头的输出
multi_head_output = concat([head_0, head_1, ..., head_7])
```

#### 不同头学习不同模式
```python
# Head 1: 学习语法依赖
# Q关注语法角色，K提供语法信息，V包含语法特征

# Head 2: 学习语义关系  
# Q关注语义相似，K提供语义标识，V包含语义内容

# Head 3: 学习位置关系
# Q关注相对位置，K提供位置信息，V包含位置特征
```


### 6. 实现细节

#### 完整的PyTorch实现
```python
import torch
import torch.nn as nn
import math

class SelfAttention(nn.Module):
    def __init__(self, d_model, num_heads):
        super().__init__()
        self.d_model = d_model
        self.num_heads = num_heads
        self.d_k = d_model // num_heads
        
        # QKV线性变换层
        self.W_Q = nn.Linear(d_model, d_model)
        self.W_K = nn.Linear(d_model, d_model) 
        self.W_V = nn.Linear(d_model, d_model)
        self.W_O = nn.Linear(d_model, d_model)
        
    def forward(self, x):
        batch_size, seq_len, d_model = x.shape
        
        # 生成QKV
        Q = self.W_Q(x)  # [batch, seq_len, d_model]
        K = self.W_K(x)
        V = self.W_V(x)
        
        # 重塑为多头
        Q = Q.view(batch_size, seq_len, self.num_heads, self.d_k).transpose(1,2)
        K = K.view(batch_size, seq_len, self.num_heads, self.d_k).transpose(1,2)  
        V = V.view(batch_size, seq_len, self.num_heads, self.d_k).transpose(1,2)
        # 形状: [batch, num_heads, seq_len, d_k]
        
        # 计算注意力
        scores = torch.matmul(Q, K.transpose(-2, -1)) / math.sqrt(self.d_k)
        attention_weights = torch.softmax(scores, dim=-1)
        output = torch.matmul(attention_weights, V)
        
        # 重新合并多头
        output = output.transpose(1,2).contiguous().view(
            batch_size, seq_len, d_model)
        
        # 最终线性变换
        return self.W_O(output)
```

### 7. 注意力模式分析

#### 常见的注意力模式
```python
# 1. 局部注意力：主要关注邻近位置
attention_weights[i] = [0.1, 0.8, 0.1, 0.0, 0.0]  # 关注附近

# 2. 全局注意力：均匀关注所有位置  
attention_weights[i] = [0.2, 0.2, 0.2, 0.2, 0.2]  # 平均分配

# 3. 稀疏注意力：只关注特定位置
attention_weights[i] = [0.0, 0.0, 1.0, 0.0, 0.0]  # 集中关注

# 4. 长距离注意力：关注远程依赖
attention_weights[i] = [0.7, 0.1, 0.1, 0.05, 0.05]  # 关注远处
```

### 8. QKV的作用总结

| 组件  | 作用     | 类比       | 学习内容               |
| ----- | -------- | ---------- | ---------------------- |
| Query | 查询信息 | 搜索关键词 | 当前位置需要什么信息   |
| Key   | 提供索引 | 数据库索引 | 每个位置能提供什么信息 |
| Value | 存储内容 | 数据记录   | 每个位置的实际语义内容 |

**核心洞察**：QKV机制将注意力建模为一个**可微分的信息检索过程**，通过学习查询、索引和内容的最优表示，实现了灵活而强大的序列建模能力。

这三个简单的线性变换，配合点积注意力的计算框架，构成了现代Transformer架构的核心，展现了"简单设计，强大功能"的深度学习设计哲学。

---

## 相关笔记
<!-- 自动生成 -->

- [Query、Key、Value的概念来源和物理意义](notes/Transformer/Query、Key、Value的概念来源和物理意义.md) - 相似度: 31% | 标签: Transformer, Transformer/Query、Key、Value的概念来源和物理意义.md

