---
created: '2025-10-19'
last_reviewed: null
next_review: '2025-10-19'
review_count: 0
difficulty: medium
mastery_level: 0.0
tags:
- 熟悉大语言模型推理优化-技术层次
- 熟悉大语言模型推理优化-技术层次/QKV矩阵和FFN层分别如何切分？.md
related_outlines: []
---
# QKV矩阵和FFN层分别如何切分？

## 面试标准答案

QKV矩阵使用列并行切分，将输出维度（注意力头维度）均匀分配到各GPU，每个GPU负责计算一部分注意力头，输出保持分片状态无需通信。FFN的第一层(W1)也使用列并行，按扩展后的中间维度切分；第二层(W2)使用行并行，输入是分片的，输出需要All-Reduce聚合。这种切分方式确保激活函数(GeLU)可以在各GPU独立计算，最小化通信开销。

---

## 详细讲解

### 1. QKV矩阵的切分策略

#### 1.1 标准QKV投影

在Multi-Head Attention中，QKV投影将输入映射到查询、键、值空间：

```python
# 输入
X: [Batch, SeqLen, Hidden]  # [B, S, H]

# 权重矩阵
W_q: [Hidden, Hidden]  # [H, H] 或 [H, num_heads × head_dim]
W_k: [Hidden, Hidden]  
W_v: [Hidden, Hidden]

# 投影
Q = X @ W_q  # [B, S, H]
K = X @ W_k
V = X @ W_v
```

#### 1.2 列并行切分

将QKV权重矩阵沿输出维度（列）切分：

```python
# N路并行，假设 H = num_heads × head_dim, num_heads % N == 0

# GPU i 的权重 (i ∈ [0, N-1])
W_q_i = W_q[:, i * (H/N) : (i+1) * (H/N)]  # [H, H/N]
W_k_i = W_k[:, i * (H/N) : (i+1) * (H/N)]
W_v_i = W_v[:, i * (H/N) : (i+1) * (H/N)]

# 前向计算（输入X在所有GPU复制）
Q_i = X @ W_q_i  # [B, S, H/N] - 无需通信
K_i = X @ W_k_i
V_i = X @ W_v_i
```

**关键特性**：
- 输入X在所有设备复制（或广播）
- 每个GPU计算一部分输出维度
- 完全独立，无需通信

#### 1.3 按注意力头切分

从多头注意力的角度理解：

```python
# 假设96个头，8路并行
num_heads = 96
head_dim = 128
H = num_heads × head_dim = 12288

# 每个GPU负责 96/8 = 12 个头
heads_per_gpu = num_heads // N = 12

# GPU 0 负责头 0-11
# GPU 1 负责头 12-23
# ...
# GPU 7 负责头 84-95

# GPU i 的QKV
Q_i: [B, S, heads_per_gpu, head_dim] = [B, S, 12, 128]
K_i: [B, S, 12, 128]
V_i: [B, S, 12, 128]

# 每个GPU独立计算自己负责的注意力头
for head in range(heads_per_gpu):
    attn_scores = Q_i[:,:,head,:] @ K_i[:,:,head,:].T / √head_dim
    attn_weights = Softmax(attn_scores)
    head_output = attn_weights @ V_i[:,:,head,:]
```

#### 1.4 具体示例

**LLaMA-65B** (8路张量并行):
```
总参数配置:
- Hidden size: 8192
- Num heads: 64
- Head dim: 128

每个GPU的QKV:
- W_q_i: [8192, 1024]  (8个头)
- W_k_i: [8192, 1024]
- W_v_i: [8192, 1024]
- 总参数: 3 × 8192 × 1024 = 25M 参数/GPU
```

### 2. Attention输出投影的切分

#### 2.1 行并行切分

Attention计算完成后，需要输出投影：

```python
# Attention输出（已经是分片的）
Attn_i: [B, S, H/N]  # 每个GPU持有

# 输出投影权重 - 按行切分
W_o_i = W_o[i * (H/N) : (i+1) * (H/N), :]  # [H/N, H]

# 局部计算
Output_i = Attn_i @ W_o_i  # [B, S, H]

# All-Reduce汇总
Output = All-Reduce(Output_i, op=SUM)  # [B, S, H]
```

**为什么用行并行**：
- 输入Attn_i已经是分片的（来自列并行的QKV）
- 行并行可以直接使用分片输入
- 输出需要完整，通过All-Reduce汇总

### 3. FFN第一层的切分

#### 3.1 列并行切分

FFN第一层通常将维度扩展4倍：

```python
# 标准FFN第一层
W_1: [H, 4H]
b_1: [4H]

# 列并行切分
W_1_i = W_1[:, i * (4H/N) : (i+1) * (4H/N)]  # [H, 4H/N]
b_1_i = b_1[i * (4H/N) : (i+1) * (4H/N)]     # [4H/N]

# 前向计算
X: [B, S, H]  # 输入，在所有GPU复制
H_i = X @ W_1_i + b_1_i  # [B, S, 4H/N] - 无需通信
```

#### 3.2 激活函数

激活函数在分片上独立应用：

```python
# 每个GPU独立应用GeLU
A_i = GeLU(H_i)  # [B, S, 4H/N] - 无需通信

# 或者 SwiGLU (LLaMA使用)
# FFN第一层分成两部分: gate 和 up
Gate_i = X @ W_gate_i  # [B, S, intermediate/N]
Up_i = X @ W_up_i      # [B, S, intermediate/N]
A_i = SiLU(Gate_i) * Up_i  # 逐元素操作，无通信
```

#### 3.3 具体示例

**GPT-3** (8路并行):
```
配置:
- Hidden: 12288
- FFN intermediate: 4 × 12288 = 49152

每个GPU:
- W_1_i: [12288, 49152/8] = [12288, 6144]
- 激活后: [B, S, 6144]
```

### 4. FFN第二层的切分

#### 4.1 行并行切分

FFN第二层将维度还原：

```python
# 标准FFN第二层
W_2: [4H, H]
b_2: [H]

# 行并行切分
W_2_i = W_2[i * (4H/N) : (i+1) * (4H/N), :]  # [4H/N, H]
# 偏置在所有GPU复制
b_2_copy = b_2  # [H]

# 前向计算
A_i: [B, S, 4H/N]  # 输入，来自FFN第一层

# 局部矩阵乘法
Y_i = A_i @ W_2_i  # [B, S, H]

# All-Reduce汇总
Y_partial = All-Reduce(Y_i, op=SUM)  # [B, S, H]

# 加偏置（只在一个GPU加，或都加然后除以N）
Y = Y_partial + b_2  # [B, S, H]
```

#### 4.2 偏置处理

有两种处理偏置的方式：

**方式1：只在rank 0加偏置**
```python
if rank == 0:
    Y_i = A_i @ W_2_i + b_2
else:
    Y_i = A_i @ W_2_i

Y = All-Reduce(Y_i, op=SUM)
```

**方式2：所有GPU加偏置，然后调整**
```python
# 每个GPU都加 b_2/N
Y_i = A_i @ W_2_i + b_2 / N
Y = All-Reduce(Y_i, op=SUM)  # 汇总后正好是 结果 + b_2
```

### 5. 完整的切分示意图

```
┌─────────────────────────────────────────────────────┐
│                  Input X [B,S,H]                    │
│              (复制到所有GPU)                         │
└─────────────────────────────────────────────────────┘
                          │
          ┌───────────────┼───────────────┬───────────┐
          │               │               │           │
        GPU 0           GPU 1           GPU 2       GPU 3
          │               │               │           │
    ┌─────▼─────┐   ┌─────▼─────┐   ┌─────▼─────┐   ┌──▼──┐
    │ QKV投影   │   │ QKV投影   │   │ QKV投影   │   │ ... │
    │ (列并行)  │   │ (列并行)  │   │ (列并行)  │   │     │
    └─────┬─────┘   └─────┬─────┘   └─────┬─────┘   └──┬──┘
          │               │               │           │
    Q₀K₀V₀ [H/4]    Q₁K₁V₁ [H/4]    Q₂K₂V₂ [H/4]    Q₃K₃V₃
          │               │               │           │
    ┌─────▼─────┐   ┌─────▼─────┐   ┌─────▼─────┐   ┌──▼──┐
    │ Attention │   │ Attention │   │ Attention │   │ ... │
    │ (独立)    │   │ (独立)    │   │ (独立)    │   │     │
    └─────┬─────┘   └─────┬─────┘   └─────┬─────┘   └──┬──┘
          │               │               │           │
    Attn₀ [H/4]     Attn₁ [H/4]     Attn₂ [H/4]     Attn₃
          │               │               │           │
    ┌─────▼─────┐   ┌─────▼─────┐   ┌─────▼─────┐   ┌──▼──┐
    │ 输出投影  │   │ 输出投影  │   │ 输出投影  │   │ ... │
    │ (行并行)  │   │ (行并行)  │   │ (行并行)  │   │     │
    └─────┬─────┘   └─────┬─────┘   └─────┬─────┘   └──┬──┘
          │               │               │           │
          └───────────────┴───────────────┴───────────┘
                          │
                   All-Reduce (SUM)
                          │
                          ▼
                   Output [B,S,H]
                          │
          ┌───────────────┼───────────────┬───────────┐
          │               │               │           │
    ┌─────▼─────┐   ┌─────▼─────┐   ┌─────▼─────┐   ┌──▼──┐
    │ FFN-W1    │   │ FFN-W1    │   │ FFN-W1    │   │ ... │
    │ (列并行)  │   │ (列并行)  │   │ (列并行)  │   │     │
    └─────┬─────┘   └─────┬─────┘   └─────┬─────┘   └──┬──┘
          │               │               │           │
    ┌─────▼─────┐   ┌─────▼─────┐   ┌─────▼─────┐   ┌──▼──┐
    │   GeLU    │   │   GeLU    │   │   GeLU    │   │ ... │
    │  (独立)   │   │  (独立)   │   │  (独立)   │   │     │
    └─────┬─────┘   └─────┬─────┘   └─────┬─────┘   └──┬──┘
          │               │               │           │
    H₀ [4H/4]       H₁ [4H/4]       H₂ [4H/4]       H₃
          │               │               │           │
    ┌─────▼─────┐   ┌─────▼─────┐   ┌─────▼─────┐   ┌──▼──┐
    │ FFN-W2    │   │ FFN-W2    │   │ FFN-W2    │   │ ... │
    │ (行并行)  │   │ (行并行)  │   │ (行并行)  │   │     │
    └─────┬─────┘   └─────┬─────┘   └─────┬─────┘   └──┬──┘
          │               │               │           │
          └───────────────┴───────────────┴───────────┘
                          │
                   All-Reduce (SUM)
                          │
                          ▼
                   Output [B,S,H]
```

### 6. 切分策略的数学表示

#### 6.1 QKV切分

$$
\begin{align}
W_q &= [W_{q,0} | W_{q,1} | ... | W_{q,N-1}] \\
Q &= X \cdot W_q = [X \cdot W_{q,0} | X \cdot W_{q,1} | ... | X \cdot W_{q,N-1}] \\
  &= [Q_0 | Q_1 | ... | Q_{N-1}]
\end{align}
$$

#### 6.2 输出投影切分

$$
\begin{align}
W_o &= \begin{bmatrix} W_{o,0} \\ W_{o,1} \\ ... \\ W_{o,N-1} \end{bmatrix} \\
\text{Attn} &= [\text{Attn}_0 | \text{Attn}_1 | ... | \text{Attn}_{N-1}] \\
O &= \text{Attn} \cdot W_o = \sum_{i=0}^{N-1} \text{Attn}_i \cdot W_{o,i}
\end{align}
$$

#### 6.3 FFN切分

$$
\begin{align}
W_1 &= [W_{1,0} | W_{1,1} | ... | W_{1,N-1}] \\
H &= \text{GeLU}(X \cdot W_1) = [\text{GeLU}(X \cdot W_{1,0}) | ... | \text{GeLU}(X \cdot W_{1,N-1})] \\
  &= [H_0 | H_1 | ... | H_{N-1}] \\
\\
W_2 &= \begin{bmatrix} W_{2,0} \\ W_{2,1} \\ ... \\ W_{2,N-1} \end{bmatrix} \\
Y &= H \cdot W_2 = \sum_{i=0}^{N-1} H_i \cdot W_{2,i}
\end{align}
$$

### 7. 内存和计算分析

#### 7.1 参数量分布

以Transformer层为例（H=12288, 4路并行）：

| 组件 | 原始大小      | 每GPU大小     | 节省比例 |
| ---- | ------------- | ------------- | -------- |
| W_q  | 12288²        | 12288 × 3072  | 75%      |
| W_k  | 12288²        | 12288 × 3072  | 75%      |
| W_v  | 12288²        | 12288 × 3072  | 75%      |
| W_o  | 12288²        | 3072 × 12288  | 75%      |
| W_1  | 12288 × 49152 | 12288 × 12288 | 75%      |
| W_2  | 49152 × 12288 | 12288 × 12288 | 75%      |

**总计**：参数显存减少 75%

#### 7.2 激活显存

```
输入 X: [B, S, H] - 复制到所有GPU
Q_i, K_i, V_i: 各 [B, S, H/N] - 减少到1/N
Attn_i: [B, S, H/N] - 减少到1/N
H_i: [B, S, 4H/N] - 减少到1/N

激活显存节省: 约 (N-1)/N
```

#### 7.3 计算量分布

每个GPU的计算量约为总计算量的 1/N，实现了线性加速。

### 8. 实现要点

#### 8.1 权重初始化

```python
# 确保所有GPU使用相同的初始化种子
def init_weights_parallel(tensor, rank, world_size, parallel_mode):
    torch.manual_seed(init_seed)  # 相同的种子
    
    # 初始化完整权重
    full_weight = torch.randn(full_shape) * init_std
    
    # 切分到对应GPU
    if parallel_mode == 'column':
        return full_weight[:, rank::world_size].contiguous()
    else:  # row
        return full_weight[rank::world_size, :].contiguous()
```

#### 8.2 梯度处理

```python
# 列并行层的权重梯度
# 已经是正确的局部梯度，无需额外通信

# 行并行层的权重梯度  
# 也是正确的局部梯度

# 输入的梯度需要通过f/g操作符处理通信
```

### 9. 变体和优化

#### 9.1 GQA (Grouped Query Attention)

```python
# 不是所有头都有独立的K,V
# 多个Q头共享一组K,V

# 切分策略类似，但K,V的切分粒度更粗
num_q_heads_per_gpu = num_q_heads // N
num_kv_heads_per_gpu = num_kv_heads // N
```

#### 9.2 SwiGLU FFN

```python
# LLaMA的FFN使用SwiGLU激活
# 需要两个投影: gate 和 up

W_gate_i: [H, intermediate/N]  # 列并行
W_up_i: [H, intermediate/N]    # 列并行
W_down_i: [intermediate/N, H]  # 行并行 (对应W_2)
```

理解QKV和FFN层的具体切分方式，是实现高效张量并行的核心。列并行和行并行的交替使用，确保了通信开销最小化。


---

## 相关笔记
<!-- 自动生成 -->

- [列并行和行并行的区别是什么？](notes/熟悉大语言模型推理优化-技术层次/列并行和行并行的区别是什么？.md) - 相似度: 39% | 标签: 熟悉大语言模型推理优化-技术层次, 熟悉大语言模型推理优化-技术层次/列并行和行并行的区别是什么？.md

