---
created: '2025-10-19'
last_reviewed: null
next_review: '2025-10-19'
review_count: 0
difficulty: medium
mastery_level: 0.0
tags:
- 熟悉大语言模型推理优化-技术层次
- 熟悉大语言模型推理优化-技术层次/GPipe、PipeDream等方法如何减少气泡？.md
related_outlines: []
---
# GPipe、PipeDream等方法如何减少气泡？

## 面试标准答案

GPipe通过将batch分成多个micro-batch来减少气泡，使用同步的填充-稳定-排空调度。PipeDream使用1F1B (One Forward One Backward)调度，交替执行前向和反向，减少了内存占用，允许更多micro-batch。PipeDream-Flush解决了权重版本问题，保证收敛性。Interleaved Pipeline让每个GPU负责多个不连续的stage，进一步减少气泡。推理场景主要使用GPipe的micro-batch策略，训练场景可以使用PipeDream的1F1B。

---

## 详细讲解

### 0. 什么是气泡（Bubble）？

**气泡定义**: 在流水线并行中，GPU处于空闲状态、没有执行有效计算的时间段。

**为什么会有气泡？**

假设有4个GPU，每个GPU负责模型的一层（stage）：

```
时间 →
GPU0: [F1] [F2] [F3] [F4]          [B1] [B2] [B3] [B4]
GPU1:      [F1] [F2] [F3] [F4]     [B1] [B2] [B3] [B4]
GPU2:           [F1] [F2] [F3] [F4][B1] [B2] [B3] [B4]
GPU3:                [F1] [F2] [F3][F4][B1] [B2] [B3] [B4]

空白部分 = 气泡（GPU空闲）
F = Forward（前向）
B = Backward（反向）
数字 = micro-batch编号
```

**问题严重性**:
- 4个GPU的例子中，GPU0前面有3个时间单位的气泡
- GPU利用率 = 有效计算时间 / 总时间
- 气泡越多，GPU利用率越低，浪费算力

**减少气泡的核心思路**: 
1. 增加micro-batch数量（M），让流水线更满
2. 优化调度策略，减少空闲时间
3. 让每个GPU负责多个stage，增加工作机会

---

### 1. GPipe策略

**核心思想**: Micro-batch分割 + 同步调度

#### 1.1 基本原理

**没有micro-batch的情况**:
```
只有1个大batch，极度浪费：
GPU0: [Forward]                                      [Backward]
GPU1:          [Forward]                             [Backward]
GPU2:                   [Forward]                    [Backward]
GPU3:                            [Forward]           [Backward]
      ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^气泡太多！
```

**GPipe的解决方案**: 把batch分成M个micro-batch

```
假设batch=32，分成M=8个micro-batch，每个micro-batch=4

GPU0: [F1][F2][F3][F4][F5][F6][F7][F8]          [B1][B2][B3][B4][B5][B6][B7][B8]
GPU1:     [F1][F2][F3][F4][F5][F6][F7][F8]     [B1][B2][B3][B4][B5][B6][B7][B8]
GPU2:         [F1][F2][F3][F4][F5][F6][F7][F8] [B1][B2][B3][B4][B5][B6][B7][B8]
GPU3:             [F1][F2][F3][F4][F5][F6][F7][F8][B1][B2][B3][B4][B5][B6][B7][B8]

气泡明显减少！
```

#### 1.2 三个阶段详解

**Fill（填充）**: 
- 流水线逐渐填满的过程
- GPU0先开始，GPU1等GPU0的输出，依此类推
- 持续时间: P-1 个micro-batch（P=pipeline深度）

**Steady（稳定）**: 
- 所有GPU都在工作
- 最高效的阶段，无气泡
- 持续时间: M-(P-1) 个micro-batch

**Drain（排空）**: 
- 流水线逐渐排空
- 最后的micro-batch依次完成
- 持续时间: P-1 个micro-batch

#### 1.3 气泡计算

```
总时间单位 = M个前向 + M个反向 = 2M
气泡时间 = 2(P-1) 个时间单位（填充+排空，前向和反向各一次）

气泡率 = 2(P-1) / (2M + 2(P-1)) 
       = (P-1) / (M + P-1)

示例: P=4, M=16
气泡率 = 3 / (16+3) = 3/19 ≈ 15.8%
```

**关键洞察**: M越大，气泡率越低！

#### 1.4 优点与缺点

**优点**:
- 简单易实现，调度逻辑清晰
- 完全同步，所有micro-batch用相同权重
- 收敛性好，训练稳定
- 适合推理（只有前向，气泡率更低）

**缺点**:
- 需要存储所有M个micro-batch的激活值（用于反向传播）
- 显存占用: O(M × 每层激活大小)
- M受显存限制，无法设置太大
- 实际气泡率可能还是偏高（10-30%）

### 2. PipeDream (1F1B)

**核心思想**: 交替前向反向，减少显存

#### 2.1 GPipe的显存问题

GPipe需要保存所有micro-batch的激活值：

```
GPU0处理8个micro-batch：
时刻1: 前向F1，保存激活A1
时刻2: 前向F2，保存激活A2
时刻3: 前向F3，保存激活A3
...
时刻8: 前向F8，保存激活A8
时刻9: 反向B1，用激活A1（终于可以释放A1）
时刻10: 反向B2，用激活A2
...

问题: 在时刻1-8，需要同时保存A1-A8，显存占用高！
```

#### 2.2 1F1B的改进

**策略**: 做完一个前向后，尽快做对应的反向，及时释放激活

**完整时间线示例**（4个GPU，8个micro-batch）:

```
时间 → 1  2  3  4  5  6  7  8  9  10 11 12 13 14 15 16
GPU0:  F1 F2 F3 F4 B1 F5 B2 F6 B3 F7 B4 F8 B5 B6 B7 B8
GPU1:     F1 F2 F3 B1 F4 B2 F5 B3 F6 B4 F7 B5 F8 B6 B7 B8
GPU2:        F1 F2 B1 F3 B2 F4 B3 F5 B4 F6 B5 F7 B6 F8 B7 B8
GPU3:           F1 B1 F2 B2 F3 B3 F4 B4 F5 B5 F6 B6 F7 B7 F8 B8

对比GPipe（所有前向→所有反向）:
GPU0:  F1 F2 F3 F4 F5 F6 F7 F8 B1 B2 B3 B4 B5 B6 B7 B8
       ^^^^^^^^^^^^^^^^^^^^^^^^ 这期间需要保存8个激活
       
1F1B: GPU0在时刻5就开始做B1，时刻6就可以释放A1
      最多同时保存4个激活（P=4）
```

#### 2.3 三个阶段详解

**Warmup（预热）**:
- GPU0先执行P-1次前向（填充流水线）
- 例如P=4时，GPU0执行F1, F2, F3
- 此时GPU3收到第一个micro-batch F1

**Steady（稳定）**:
- 执行"1个前向，1个反向"的交替模式
- GPU0: B1 F5 B2 F6 B3 F7 B4 F8 ...
- 每做完一个反向，立即释放对应激活
- 最多同时保存P个激活

**Cooldown（冷却）**:
- 所有前向完成后，处理剩余的反向
- GPU0: B5 B6 B7 B8

#### 2.4 显存优势

```
GPipe显存: 需要保存M个micro-batch的激活
例如: M=32, 每个激活4GB
总显存 = 32 × 4GB = 128GB

1F1B显存: 只需保存P个micro-batch的激活
例如: P=4, 每个激活4GB
总显存 = 4 × 4GB = 16GB

节省: 128GB - 16GB = 112GB！
```

**含义**: 
- 相同显存下，1F1B可以用更大的M（更多micro-batch）
- M更大 → 气泡率更低 → 效率更高

#### 2.5 气泡率对比

```
GPipe和1F1B的气泡率公式相同:
气泡率 = (P-1) / (M + P-1)

但是：
- GPipe: M受限于显存，可能只能设M=8
  气泡率 = 3/(8+3) = 27.3%
  
- 1F1B: 显存友好，可以设M=32
  气泡率 = 3/(32+3) = 8.6%
  
提升: 27.3% → 8.6%，效率提高约3倍！
```

#### 2.6 优点与缺点

**优点**:
- 显存占用从O(M)降到O(P)
- 允许使用更多micro-batch
- 实际气泡率更低
- 训练吞吐量更高

**缺点**:
- **权重版本不一致问题**（下节详细讨论）
- 实现复杂度较高
- 调度逻辑需要精心设计

### 3. PipeDream-Flush

#### 3.1 权重版本不一致问题

**问题描述**: 在1F1B调度中，不同micro-batch的前向和反向使用不同版本的权重

**具体例子**:

```
假设每个反向传播后更新一次权重：

时刻1: GPU0用权重W0做F1
时刻2: GPU0用权重W0做F2
时刻5: GPU0用权重W0做B1 → 权重更新为W1
时刻6: GPU0用权重W1做F5（注意！F5用的是W1）
时刻7: GPU0用权重W1做B2 → 权重更新为W2

问题: 
- F1的前向用W0，但可能在F5还没完成反向时，权重已经变成W2
- F2的前向用W0，但它的反向用W1
- 数学上不一致，可能影响收敛性
```

**为什么GPipe没有这个问题？**
```
GPipe所有前向完成后才做反向：
- F1-F8全部用W0
- 然后B1-B8全部用W0
- 最后统一更新权重为W1
- 下一轮F9-F16全部用W1
→ 前向和反向总是匹配的
```

#### 3.2 PipeDream-Flush解决方案

**核心思想**: 周期性同步，确保权重一致性

**调度策略**:
```
阶段1: 1F1B模式（N个micro-batch）
       F1 F2 ... B1 B2 ... (使用权重W0)

Flush: 排空流水线，等待所有反向完成
       等待所有GPU完成当前batch的梯度计算
       
梯度聚合: 收集所有梯度，更新权重W0→W1

阶段2: 1F1B模式（N个micro-batch）
       F(N+1) F(N+2) ... (使用权重W1)
       
→ 每个Flush周期内，权重保持一致
```

**完整时间线**:
```
=== Batch 1 (权重W0) ===
GPU0: F1 F2 F3 F4 B1 F5 B2 F6 B3 F7 B4 F8 B5 B6 B7 B8
GPU1:    F1 F2 F3 B1 F4 B2 F5 B3 F6 B4 F7 B5 F8 B6 B7 B8
...
[Flush点: 等待所有GPU完成]
[更新权重: W0 → W1]

=== Batch 2 (权重W1) ===
GPU0: F9 F10 F11 F12 B9 F13 B10 ...
...
```

#### 3.3 权衡

**优点**:
- 解决权重版本问题
- 保证收敛性和数学正确性
- 兼顾1F1B的显存优势

**缺点**:
- Flush期间会产生气泡（等待同步）
- 效率略低于原始1F1B
- 但仍比GPipe好很多

**适用场景**:
- 训练: 必须使用（保证正确性）
- 推理: 不需要（没有权重更新）

### 4. Interleaved Pipeline（交错流水线）

**核心思想**: 每个GPU负责多个不连续的stage，增加调度灵活性

#### 4.1 为什么需要Interleaved Pipeline？

**传统流水线的限制**:
```
4个GPU，模型分4个stage：
GPU0: [Stage 0]
GPU1: [Stage 1]
GPU2: [Stage 2]
GPU3: [Stage 3]

问题: 当GPU0在做B1时，必须等F4完成才能做F5
     灵活性差，气泡难以进一步减少
```

**Interleaved的改进**: 让每个GPU负责多个stage

#### 4.2 配置示例

**4个GPU，切分成8个虚拟stage**:
```
物理GPU和虚拟stage的映射：
GPU 0: Stage 0, Stage 4
GPU 1: Stage 1, Stage 5
GPU 2: Stage 2, Stage 6
GPU 3: Stage 3, Stage 7

每个GPU负责2个stage (V=2，V是virtual stages per GPU)
```

**数据流动**:
```
Micro-batch 1的计算路径:
m1: S0(GPU0) → S1(GPU1) → S2(GPU2) → S3(GPU3) 
    → S4(GPU0) → S5(GPU1) → S6(GPU2) → S7(GPU3)
    
注意: 数据在GPU0和GPU1之间来回传递多次
```

#### 4.3 时间线对比

**传统Pipeline** (P=4, M=8):
```
GPU0: F1 F2 F3 F4 B1 F5 B2 F6 B3 F7 B4 F8 B5 B6 B7 B8
GPU1:    F1 F2 F3 B1 F4 B2 F5 B3 F6 B4 F7 B5 F8 B6 B7 B8
GPU2:       F1 F2 B1 F3 B2 F4 B3 F5 B4 F6 B5 F7 B6 F8 B7 B8
GPU3:          F1 B1 F2 B2 F3 B3 F4 B4 F5 B5 F6 B6 F7 B7 F8 B8

气泡率 = 3/(8+3) = 27.3%
```

**Interleaved Pipeline** (P=4, V=2, M=8):
```
现在有8个虚拟stage，流水线深度增加：

GPU0: F1(S0) F2(S0) F1(S4) F3(S0) F2(S4) B1(S0) F4(S0) F1(S4) ...
GPU1: F1(S1) F2(S1) F1(S5) F3(S1) F2(S5) B1(S1) F4(S1) ...
...

每个GPU有更多工作机会，气泡明显减少
气泡率 = 3/(8×2+3) = 3/19 = 15.8%
```

#### 4.4 气泡率公式推导

**传统Pipeline**:
```
气泡率 = (P-1) / (M + P-1)
```

**Interleaved Pipeline**:
```
虚拟stage数: P_virtual = P × V
有效micro-batch数: M_effective = M × V (每个GPU处理V倍的stage)

气泡率 = (P-1) / (M×V + P-1)

示例: P=4, M=8, V=2
气泡率 = 3/(8×2+3) = 3/19 ≈ 15.8%

对比传统: 3/(8+3) = 27.3%
改进: (27.3-15.8)/27.3 = 42%气泡减少
```

#### 4.5 优点与缺点

**优点**:
- 气泡率大幅降低（通常减少30-50%）
- 提高GPU利用率
- 特别适合pipeline深度较小的场景

**缺点**:
- **通信次数增加**: 每个micro-batch需要在GPU间传递V倍次数
  ```
  传统: m1经过4个GPU，通信3次
  V=2: m1经过8个stage，通信7次（翻倍）
  ```
- 通信开销可能抵消气泡减少的收益
- 实现复杂度更高

**何时使用**:
- 通信带宽高（如NVLink）
- Pipeline深度小（P≤8）
- 气泡率>20%时考虑
- GPU间通信快于单个stage计算时间

### 5. 推理场景优化

#### 5.1 推理的特殊性

**与训练的区别**:
```
训练:
- 前向 + 反向
- 需要存储激活值
- 需要更新权重
- 可以用大batch（128-512）

推理:
- 只有前向
- 不需要存储激活（只需最终输出）
- 权重固定
- Batch通常较小（1-32）
```

**推理不需要考虑的问题**:
- 权重版本不一致 ✓（没有更新）
- 激活值显存占用 ✓（只需保存最后一层）
- PipeDream-Flush ✓（不需要同步）

#### 5.2 推理的气泡问题

**气泡率计算**（只有前向）:
```
传统: 气泡率 = (P-1) / (M + P-1)  (前向+反向)
推理: 气泡率 = (P-1) / M           (只有前向)

示例: P=4, M=8
训练气泡率 = 3/(8+3) = 27.3%
推理气泡率 = 3/8 = 37.5%

推理反而更高！因为缺少反向来填充气泡
```

**解决方案**: 需要更多的micro-batch

```
P=4, M=16:
推理气泡率 = 3/16 = 18.75%

P=4, M=32:
推理气泡率 = 3/32 = 9.4%
```

#### 5.3 推理最佳实践

**方案选择**:

```python
# 场景1: 在线推理（batch=1-4，低延迟）
建议: 不用Pipeline Parallelism
原因: 气泡率太高（75%+），延迟增加
替代: Tensor Parallelism (TP)
配置: tp_size=4-8

# 场景2: 批量推理（batch=16-64，吞吐优先）
建议: GPipe + Pipeline Parallelism
配置:
  pp_size = 2-4
  micro_batch_size = 2-4
  num_micro_batches = batch_size / micro_batch_size
  
气泡率计算:
  batch=32, pp=4, micro_batch=2
  M = 32/2 = 16
  气泡率 = 3/16 = 18.75%  ✓可接受

# 场景3: 超大模型推理（显存不够）
建议: PP + TP混合
配置:
  pp_size = 4-8（跨节点）
  tp_size = 2-4（节点内）
  micro_batch = 1-2
  
优先用TP（通信快），TP不够用PP
```

#### 5.4 实战配置示例

**GPT-3 175B推理** (8xA100 80GB):

```python
# 配置1: 低延迟（batch=1）
策略: 纯TP
tp_size = 8
pp_size = 1
延迟: ~50ms
吞吐: 20 tokens/s

# 配置2: 高吞吐（batch=32）
策略: PP + TP
tp_size = 4
pp_size = 2
micro_batch_size = 2
num_micro_batches = 16

气泡率 = (2-1)/16 = 6.25%  ✓很低
吞吐: 500 tokens/s
```

**LLaMA-70B推理** (4xA100 40GB):

```python
# batch=16的配置
pp_size = 4
micro_batch_size = 1
num_micro_batches = 16

气泡率 = 3/16 = 18.75%
实际GPU利用率: ~81%

# 优化: 使用Interleaved (如果通信快)
pp_size = 4
virtual_stages = 2
micro_batch_size = 1
num_micro_batches = 16

气泡率 = 3/(16×2) = 9.4%
实际GPU利用率: ~90%
```

#### 5.5 推理优化技巧

**1. Dynamic Batching**:
```python
# 动态调整micro-batch大小
if current_batch_size >= 32:
    micro_batch = 2
elif current_batch_size >= 16:
    micro_batch = 1
else:
    # 不用PP，直接TP
    disable_pipeline()
```

**2. Prefill vs Decode阶段**:
```python
# Prefill（首个token生成）: batch较大，用PP
prefill_stage:
    use_pipeline = True
    micro_batch = 4
    
# Decode（后续token）: batch逐渐变小，切换TP
decode_stage:
    if remaining_batch < 8:
        use_pipeline = False  # 切换到TP
```

**3. 流水线预热**:
```python
# 提前预热流水线，减少首次延迟
warmup_batches = pipeline_depth - 1
# 第一个batch延迟较高，后续batch摊平成本
```

### 6. 实际效果对比

#### 6.1 GPT-3 175B训练对比

**实验设置**: 
- 64个A100 GPU
- Pipeline深度 P=8
- Global batch size = 1024

**方案1: GPipe**
```
配置:
- Micro-batch size: 16
- Num micro-batches: 64
- 显存占用: 75GB/GPU

气泡率 = (8-1)/(64+8-1) = 7/71 = 9.9%

实际表现:
- 训练吞吐: 140 samples/s
- GPU利用率: 85%
- 显存成为瓶颈（无法增加M）
```

**方案2: PipeDream 1F1B**
```
配置:
- Micro-batch size: 8
- Num micro-batches: 128（显存友好！）
- 显存占用: 45GB/GPU

气泡率 = 7/(128+7) = 7/135 = 5.2%

实际表现:
- 训练吞吐: 210 samples/s  ✓提升50%
- GPU利用率: 92%
- 更多micro-batch，气泡更少
```

**方案3: Interleaved Pipeline (V=2)**
```
配置:
- Virtual stages per GPU: 2
- Micro-batch size: 8
- Num micro-batches: 128

气泡率 = 7/(128×2+7) = 7/263 = 2.7%

实际表现:
- 训练吞吐: 245 samples/s  ✓最高
- GPU利用率: 96%
- 但GPU间通信量翻倍（带宽需求高）
```

**总结**:
```
方法          吞吐(samples/s)  气泡率  显存(GB)  复杂度
GPipe              140         9.9%      75      低
1F1B               210         5.2%      45      中
Interleaved        245         2.7%      45      高
```

#### 6.2 LLaMA-70B推理对比

**实验设置**:
- 4个A100 40GB
- Batch size = 32
- Sequence length = 2048

**方案1: 纯TP (基准)**
```
配置:
- tp_size = 4
- pp_size = 1

性能:
- 延迟: 85ms/token
- 吞吐: 376 tokens/s
- 显存: 38GB/GPU（接近上限）
- 无气泡，但显存紧张
```

**方案2: GPipe PP**
```
配置:
- pp_size = 4
- micro_batch = 2
- num_micro_batches = 16

气泡率 = 3/16 = 18.75%

性能:
- 延迟: 92ms/token（略慢）
- 吞吐: 348 tokens/s
- 显存: 22GB/GPU（充裕）
- 气泡导致吞吐下降
```

**方案3: PP+TP混合**
```
配置:
- pp_size = 2
- tp_size = 2
- micro_batch = 2
- num_micro_batches = 16

气泡率 = 1/16 = 6.25%

性能:
- 延迟: 88ms/token
- 吞吐: 364 tokens/s  ✓接近纯TP
- 显存: 28GB/GPU（平衡）
- 最佳配置！
```

**总结**:
```
方法            吞吐(tok/s)  延迟(ms)  显存(GB)  推荐场景
纯TP                376        85        38       低延迟+显存够
GPipe PP            348        92        22       显存不足
PP+TP混合           364        88        28       平衡方案✓
```

#### 6.3 不同Pipeline深度的影响

**实验**: 固定M=64，改变P

```
P=2 (2个GPU):
气泡率 = 1/64 = 1.6%  ✓很低
吞吐: 98% of 理论峰值
推荐: 显存够的话优先选择

P=4:
气泡率 = 3/64 = 4.7%
吞吐: 93% of 理论峰值
推荐: 平衡选择

P=8:
气泡率 = 7/64 = 10.9%
吞吐: 85% of 理论峰值
推荐: 只在显存不够时使用

P=16:
气泡率 = 15/64 = 23.4%  ✗太高
吞吐: 72% of 理论峰值
推荐: 避免（除非M可以加大）
```

**关键洞察**: 
- P越小越好（气泡越少）
- 但受限于显存和模型大小
- 通常P=2-8是甜点区域

#### 6.4 Micro-batch数量的影响

**实验**: 固定P=4，改变M

```
M=4 (刚好填满流水线):
气泡率 = 3/4 = 75%  ✗极差
几乎不可用

M=8:
气泡率 = 3/8 = 37.5%  ✗较差
推理场景可能遇到

M=16:
气泡率 = 3/16 = 18.75%  ✓可接受
推理常见配置

M=32:
气泡率 = 3/32 = 9.4%  ✓良好
训练场景推荐

M=64:
气泡率 = 3/64 = 4.7%  ✓很好
大batch训练

M=128:
气泡率 = 3/128 = 2.3%  ✓极佳
大规模训练（1F1B解决显存问题）
```

**经验法则**: 
- M应该至少是P的4倍（气泡<20%）
- M越大越好，但受限于显存和延迟
- 训练用1F1B可以支持更大的M

### 7. 方法选择决策树

#### 7.1 训练场景

```
训练任务
│
├─ 小模型（显存充足）
│   └─ GPipe
│       - 简单易实现
│       - 收敛性好
│       - M可以设置较大
│
├─ 中大模型（显存紧张）
│   └─ PipeDream 1F1B
│       - 显存占用低
│       - 可以用更多micro-batch
│       - 气泡率更低
│
└─ 超大模型（通信带宽高）
    └─ Interleaved Pipeline
        - 气泡率最低
        - 需要快速互联（NVLink/InfiniBand）
        - 适合P<=8的场景
```

#### 7.2 推理场景

```
推理任务
│
├─ 在线服务（batch=1-4，低延迟）
│   └─ 纯Tensor Parallelism
│       - 无流水线气泡
│       - 延迟最低
│       - PP气泡率>75%，不可用
│
├─ 批量推理（batch=16-64）
│   ├─ 显存充足
│   │   └─ PP + TP混合
│   │       - pp_size = 2-4
│   │       - tp_size = 2-4
│   │       - 平衡延迟和吞吐
│   │
│   └─ 显存紧张
│       └─ 纯Pipeline Parallelism
│           - pp_size = 4-8
│           - 用GPipe（简单）
│           - M设置为batch/micro_batch
│
└─ 离线大batch（batch>64）
    └─ GPipe PP
        - micro_batch = 2-4
        - M >= 32
        - 气泡率<10%
```

#### 7.3 快速参考表

| 场景           | Pipeline深度 | Micro-batch数 | 推荐方法    | 气泡率目标 |
| -------------- | ------------ | ------------- | ----------- | ---------- |
| 在线推理       | 不用PP       | -             | 纯TP        | 0%         |
| 批量推理       | P=2-4        | M=16-32       | GPipe       | <15%       |
| 训练（显存够） | P=4-8        | M=32-64       | GPipe       | <10%       |
| 训练（显存紧） | P=4-8        | M=64-128      | 1F1B        | <5%        |
| 大规模训练     | P=4-8        | M=128+        | Interleaved | <3%        |

### 8. 总结与关键要点

#### 核心概念回顾

1. **气泡**: 流水线并行中GPU的空闲时间，由填充和排空阶段产生

2. **减少气泡的三个维度**:
   - 增加micro-batch数量M（分子不变，分母增大）
   - 减少pipeline深度P（如果显存允许）
   - 优化调度策略（1F1B、Interleaved）

3. **四种主要方法**:
   ```
   GPipe:         简单，同步，显存占用高
   1F1B:          显存友好，允许更大M，有权重版本问题
   PipeDream-Flush: 解决权重版本，保证收敛性
   Interleaved:   气泡最低，但通信多
   ```

4. **关键公式**:
   ```
   传统PP气泡率 = (P-1) / (M + P-1)
   Interleaved气泡率 = (P-1) / (M×V + P-1)
   推理气泡率 = (P-1) / M  （只有前向）
   ```

#### 实战建议

1. **优先考虑减少P**: P=2-4是最佳区域，除非显存不够

2. **M至少是P的4-8倍**: 保证气泡率<15%

3. **训练首选1F1B**: 显存优势明显，可用更多micro-batch

4. **推理batch>16才用PP**: 否则气泡太高，用TP

5. **通信快才用Interleaved**: NVLink/IB才能发挥优势

6. **混合并行是王道**: PP+TP结合，发挥各自优势

理解各种方法的权衡，根据实际场景选择最合适的策略！


---

## 相关笔记
<!-- 自动生成 -->

- [流水线并行中的气泡是什么？为什么会降低效率？](notes/熟悉大语言模型推理优化-技术层次/流水线并行中的气泡是什么？为什么会降低效率？.md) - 相似度: 31% | 标签: 熟悉大语言模型推理优化-技术层次, 熟悉大语言模型推理优化-技术层次/流水线并行中的气泡是什么？为什么会降低效率？.md
- [推理时的流水线并行与训练有何不同？](notes/熟悉大语言模型推理优化-技术层次/推理时的流水线并行与训练有何不同？.md) - 相似度: 31% | 标签: 熟悉大语言模型推理优化-技术层次, 熟悉大语言模型推理优化-技术层次/推理时的流水线并行与训练有何不同？.md

