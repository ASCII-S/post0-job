---
created: '2025-10-19'
last_reviewed: null
next_review: '2025-10-19'
review_count: 0
difficulty: medium
mastery_level: 0.0
tags:
- 熟悉大语言模型推理优化-技术层次
- 熟悉大语言模型推理优化-技术层次/张量并行中的通信模式是什么（All-Reduce、All-Gather）？.md
related_outlines: []
---
# 张量并行中的通信模式是什么（All-Reduce、All-Gather）？

## 面试标准答案

张量并行主要使用两种集合通信原语：All-Reduce和All-Gather。All-Reduce用于行并行层，将各设备的部分结果求和后广播给所有设备；All-Gather用于需要收集分片数据的场景，将各设备的分片拼接成完整张量并分发。此外还有Reduce-Scatter用于反向传播。通过精心设计通信模式，可以使Transformer的一个完整Block（Attention + FFN）只需两次All-Reduce通信，大大降低通信开销。

---

## 详细讲解

### 1. 集合通信原语概述

#### 1.1 什么是集合通信

集合通信（Collective Communication）是多个进程/设备之间协同进行的通信操作，所有参与者都必须调用相同的通信原语。

**常见原语**：
- **All-Reduce**: 规约并广播
- **All-Gather**: 收集并广播
- **Reduce-Scatter**: 规约并分发
- **Broadcast**: 单点广播
- **Reduce**: 规约到单点

### 2. All-Reduce

#### 2.1 基本原理

All-Reduce = Reduce + Broadcast，将所有设备的数据按某种操作（求和、取最大值等）规约，然后将结果广播给所有设备。

**操作流程**：
```
输入 (每个设备):
  GPU 0: [a₀, b₀, c₀]
  GPU 1: [a₁, b₁, c₁]
  GPU 2: [a₂, b₂, c₂]
  GPU 3: [a₃, b₃, c₃]

执行 All-Reduce (SUM):

输出 (每个设备都得到):
  [a₀+a₁+a₂+a₃, b₀+b₁+b₂+b₃, c₀+c₁+c₂+c₃]
```

#### 2.2 在张量并行中的应用

**行并行层的输出聚合**：

```python
# 每个设备计算部分结果
# GPU 0: Y₀ = X₀ @ W₀
# GPU 1: Y₁ = X₁ @ W₁
# GPU 2: Y₂ = X₂ @ W₂
# GPU 3: Y₃ = X₃ @ W₃

# All-Reduce 求和
Y = All-Reduce(Y₀, Y₁, Y₂, Y₃, op=SUM)
# 现在每个GPU都有完整的 Y = Y₀ + Y₁ + Y₂ + Y₃
```

**具体示例**（FFN第二层）：
```
输入维度: d_model = 4096 (分片到4个GPU，每个1024)
输出维度: d_model = 4096 (完整)
Batch size: B=32, Seq len: S=2048

每个GPU:
- 输入: [32, 2048, 1024]
- 权重: [1024, 4096]  
- 局部输出: [32, 2048, 4096]

All-Reduce:
- 通信数据量: 32 × 2048 × 4096 × 2 bytes (FP16) = 512 MB
- 每个GPU得到完整的 [32, 2048, 4096]
```

#### 2.3 All-Reduce实现算法

**Ring All-Reduce**（最常用）：

```
步骤1 (Reduce-Scatter阶段):
  N-1轮，每轮发送 1/N 的数据
  
步骤2 (All-Gather阶段):  
  N-1轮，每轮发送 1/N 的数据

总通信量: 2 × (N-1)/N × Data ≈ 2 × Data
总时间: 2 × (N-1)/N × Data/BW
```

**Tree All-Reduce**（小规模）：
```
步骤1: 树形Reduce到根节点
步骤2: 树形Broadcast到所有节点

总通信量: 2 × log₂(N) × Data  
总时间: 2 × log₂(N) × (Data/BW + latency)
```

### 3. All-Gather

#### 3.1 基本原理

All-Gather 将各设备的分片数据收集起来，拼接成完整数据，然后分发给所有设备。

**操作流程**：
```
输入 (每个设备持有一个分片):
  GPU 0: [a₀, b₀]
  GPU 1: [a₁, b₁]
  GPU 2: [a₂, b₂]
  GPU 3: [a₃, b₃]

执行 All-Gather:

输出 (每个设备都得到完整数据):
  [a₀, b₀, a₁, b₁, a₂, b₂, a₃, b₃]
```

#### 3.2 在张量并行中的应用

**列并行层的输出合并**（如果需要）：

```python
# 每个设备计算自己的输出分片
# GPU 0: Y₀ = X @ W₀  (输出维度的第1/4)
# GPU 1: Y₁ = X @ W₁  (输出维度的第2/4)
# GPU 2: Y₂ = X @ W₂  (输出维度的第3/4)
# GPU 3: Y₃ = X @ W₃  (输出维度的第4/4)

# All-Gather 拼接
Y = All-Gather([Y₀, Y₁, Y₂, Y₃], dim=-1)
# 现在每个GPU都有完整的 Y = concat(Y₀, Y₁, Y₂, Y₃)
```

**注意**：在实际的张量并行中，列并行层的输出通常不需要立即All-Gather，而是保持分片状态传递给下一个行并行层。

#### 3.3 All-Gather实现

**Ring All-Gather**：
```
N-1轮迭代，每轮每个设备:
1. 发送当前块给下一个设备
2. 从上一个设备接收新块

总通信量: (N-1)/N × Data
总时间: (N-1)/N × Data/BW
```

### 4. Reduce-Scatter

#### 4.1 基本原理

Reduce-Scatter = All-Reduce的前半部分，将规约后的结果按设备数切分，每个设备得到一个分片。

**操作流程**：
```
输入 (每个设备):
  GPU 0: [a₀, b₀, c₀, d₀]
  GPU 1: [a₁, b₁, c₁, d₁]  
  GPU 2: [a₂, b₂, c₂, d₂]
  GPU 3: [a₃, b₃, c₃, d₃]

执行 Reduce-Scatter (SUM):

输出 (每个设备得到一个分片):
  GPU 0: [a₀+a₁+a₂+a₃]
  GPU 1: [b₀+b₁+b₂+b₃]
  GPU 2: [c₀+c₁+c₂+c₃]
  GPU 3: [d₀+d₁+d₂+d₃]
```

#### 4.2 在反向传播中的应用

```python
# 前向: 列并行 (输出分片)
Y_i = X @ W_i

# 反向: 梯度需要对输入X求和
# 每个设备有 dL/dY_i
dX_i = dY_i @ W_i^T

# Reduce-Scatter 汇总梯度并分片
dX = Reduce-Scatter([dX_0, dX_1, ..., dX_N])
```

### 5. 张量并行的完整通信模式

#### 5.1 Transformer Block的通信

```python
# 假设4路张量并行

# === Self-Attention ===
# 输入: X (完整，每个GPU都有)

# QKV投影 - 列并行 (无通信)
Q_i, K_i, V_i = X @ W_q_i, X @ W_k_i, X @ W_v_i

# Attention计算 (无通信)
Attn_i = Softmax(Q_i @ K_i^T / √d) @ V_i

# 输出投影 - 行并行 (All-Reduce)
O_i = Attn_i @ W_o_i
O = All-Reduce(O_i)  # ← 通信1

# === FFN ===
# 输入: O (完整，来自All-Reduce)

# 第一层 - 列并行 (无通信)
H_i = GeLU(O @ W_1_i)

# 第二层 - 行并行 (All-Reduce)  
Y_i = H_i @ W_2_i
Y = All-Reduce(Y_i)  # ← 通信2

# 总计: 2次All-Reduce / Transformer Block
```

#### 5.2 通信开销计算

**参数设置**：
- Model: GPT-3 175B
- Hidden dim: 12288
- Batch size: 32
- Sequence length: 2048
- 张量并行: 8-way
- 数据类型: FP16 (2 bytes)

**每个All-Reduce的通信量**：
$$
\text{Data} = B \times S \times H \times 2 = 32 \times 2048 \times 12288 \times 2 = 1.6 \text{ GB}
$$

**Ring All-Reduce时间**（假设NVLink 600 GB/s）：
$$
T = 2 \times \frac{7}{8} \times \frac{1.6 \text{ GB}}{600 \text{ GB/s}} \approx 4.7 \text{ ms}
$$

**每层总通信时间**：
$$
T_{comm} = 2 \times 4.7 = 9.4 \text{ ms}
$$

### 6. 通信优化技术

#### 6.1 通信与计算重叠

```python
# 启动All-Reduce (异步)
handle = async_all_reduce(Y_i)

# 同时进行其他计算
residual = X + dropout(Y)
norm_output = layer_norm(residual)

# 等待All-Reduce完成
Y = handle.wait()
```

#### 6.2 通信融合

将多个小的通信操作合并：
```python
# 不好: 多次小通信
Q = all_reduce(Q_i)
K = all_reduce(K_i)  
V = all_reduce(V_i)

# 更好: 合并为一次通信
QKV = all_reduce(concat(Q_i, K_i, V_i))
Q, K, V = split(QKV)
```

#### 6.3 分层通信

对于多节点部署：
```
节点内: NVLink (600 GB/s) - 使用张量并行
节点间: InfiniBand (200 Gb/s) - 使用流水线并行
```

### 7. 通信原语对比

| 原语               | 输入       | 输出       | 通信量  | 应用场景       |
| ------------------ | ---------- | ---------- | ------- | -------------- |
| **All-Reduce**     | 完整数据×N | 规约结果×N | ~2×Data | 行并行输出聚合 |
| **All-Gather**     | 分片×N     | 完整数据×N | ~1×Data | 列并行输出合并 |
| **Reduce-Scatter** | 完整数据×N | 规约分片×N | ~1×Data | 反向传播梯度   |
| **Broadcast**      | 完整数据×1 | 完整数据×N | ~1×Data | 分发输入       |

### 8. 实际性能考量

**通信开销占比**：
```
计算时间 (A100, FP16):
- Attention: ~20 ms
- FFN: ~30 ms  
- 总计: ~50 ms/layer

通信时间 (NVLink):
- 2×All-Reduce: ~10 ms/layer

通信占比: 10/(50+10) ≈ 17%
```

**扩展性分析**：
- **节点内** (8卡 NVLink): 通信几乎可忽略
- **节点间** (InfiniBand): 通信成为瓶颈，需要减少张量并行度

### 9. 实现库

**NCCL (NVIDIA Collective Communications Library)**：
```python
import torch.distributed as dist

# All-Reduce
dist.all_reduce(tensor, op=dist.ReduceOp.SUM)

# All-Gather
tensor_list = [torch.zeros_like(tensor) for _ in range(world_size)]
dist.all_gather(tensor_list, tensor)

# Reduce-Scatter
dist.reduce_scatter(output, input_list, op=dist.ReduceOp.SUM)
```

**性能优化**：
- 使用NCCL的in-place操作减少内存拷贝
- 启用NCCL的Graph模式减少启动开销
- 调整NCCL环境变量优化通信

理解张量并行中的通信模式，特别是All-Reduce和All-Gather的使用时机和开销，对于设计高效的分布式推理系统至关重要。


---

## 相关笔记
<!-- 自动生成 -->

- [序列并行如何与张量并行结合？](notes/熟悉大语言模型推理优化-技术层次/序列并行如何与张量并行结合？.md) - 相似度: 33% | 标签: 熟悉大语言模型推理优化-技术层次, 熟悉大语言模型推理优化-技术层次/序列并行如何与张量并行结合？.md
- [列并行和行并行的区别是什么？](notes/熟悉大语言模型推理优化-技术层次/列并行和行并行的区别是什么？.md) - 相似度: 31% | 标签: 熟悉大语言模型推理优化-技术层次, 熟悉大语言模型推理优化-技术层次/列并行和行并行的区别是什么？.md

