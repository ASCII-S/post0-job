---
created: '2025-10-19'
last_reviewed: '2025-11-04'
next_review: '2025-11-14'
review_count: 3
difficulty: medium
mastery_level: 0.55
tags:
- 熟悉大语言模型推理优化-技术层次
- 熟悉大语言模型推理优化-技术层次/逐层量化、逐通道量化、分组量化（Group_Quantization）的优缺点是什么？.md
related_outlines: []
---
# 逐层量化、逐通道量化、分组量化（Group Quantization）的优缺点是什么？

## 面试标准答案

这三种是不同粒度的量化策略。逐层量化（Per-layer/Per-tensor）整个层共享一个scale，实现最简单但精度最低（精度损失5-10%）。逐通道量化（Per-channel）每个输出通道独立scale，精度显著提升（损失<1%）且存储开销可接受，是权重量化的标准做法。分组量化（Group Quantization）是折中方案，将通道分组（如每128个通道一组），在精度和开销间平衡，特别适合INT4等低比特量化，已被GPTQ、LLM.int8()等工具广泛采用。实际应用中，权重通常用逐通道或分组量化，激活值用逐层量化。

## 详细讲解

### 1. 三种量化粒度定义

#### 逐层量化（Per-layer / Per-tensor）

**定义**：
- 整个权重张量共享单一量化参数
- 一个scale（和可选的zero-point）
- 最粗粒度

**示例**：
```python
# 权重矩阵 W: [4096, 4096]
# 所有 16M 参数使用同一个scale

W_fp32 = model.layer.weight  # [4096, 4096]
scale = W_fp32.abs().max() / 127
W_int8 = (W_fp32 / scale).round().clamp(-128, 127)

# 存储: 1 个 scale (4 bytes)
```

#### 逐通道量化（Per-channel）

**定义**：
- 每个输出通道独立量化参数
- C个scales（C = 输出通道数）
- 细粒度

**示例**：
```python
# 权重矩阵 W: [4096, 4096]
# 4096 个输出通道，每个通道一个scale

out_channels, in_channels = W_fp32.shape  # [4096, 4096]
scales = []

for i in range(out_channels):
    scale_i = W_fp32[i, :].abs().max() / 127
    scales.append(scale_i)
    W_int8[i, :] = (W_fp32[i, :] / scale_i).round()

# 存储: 4096 个 scales (16 KB)
```

#### 分组量化（Group Quantization）

**定义**：
- 将通道划分为若干组
- 每组共享一个量化参数
- 中等粒度

**示例**：
```python
# 权重矩阵 W: [4096, 4096]
# 分组大小 = 128, 共 32 组

group_size = 128
n_groups = out_channels // group_size  # 32

for g in range(n_groups):
    start = g * group_size
    end = start + group_size
    
    # 该组的scale
    scale_g = W_fp32[start:end, :].abs().max() / 127
    W_int8[start:end, :] = (W_fp32[start:end, :] / scale_g).round()

# 存储: 32 个 scales (128 bytes)
```

### 2. 详细对比

| 维度         | 逐层量化          | 分组量化        | 逐通道量化        |
| ------------ | ----------------- | --------------- | ----------------- |
| **粒度**     | 粗（1个scale/层） | 中（N个scales） | 细（C个scales）   |
| **存储**     | 最小（4B）        | 小（N×4B）      | 中（C×4B，~16KB） |
| **精度**     | 低（5-10%损失）   | 中（1-3%损失）  | 高（<1%损失）     |
| **计算**     | 最快              | 中等            | 稍慢              |
| **实现**     | 最简单            | 中等            | 较复杂            |
| **硬件支持** | 最好              | 良好            | 良好              |

### 3. 优缺点深度分析

#### 逐层量化

**优点**：

1. **实现极简**
```python
# 仅需3行代码
scale = weight.abs().max() / 127
q_weight = (weight / scale).round()
dequant_weight = q_weight * scale
```

2. **存储开销最小**
```
每层仅 4 bytes (FP32 scale)
32层Transformer: 128 bytes
完全可忽略
```

3. **计算最快**
```python
# 反量化仅需标量乘法
output = int8_gemm(x_q, w_q) * (scale_x * scale_w)
# 无需逐元素或逐通道操作
```

4. **硬件最友好**
- 所有硬件都完美支持
- 无需特殊处理

**缺点**：

1. **精度损失严重**

**原因**：通道间分布差异大

```python
# 实际观察 (LLaMA-7B某层)
channel_max = [w[i].abs().max() for i in range(4096)]

min(channel_max) = 0.0082
max(channel_max) = 0.0634
ratio = 7.7x

# 逐层量化使用全局max
global_scale = 0.0634 / 127 = 0.000499

# 小通道量化
small_channel_value = 0.001
quantized = round(0.001 / 0.000499) = 2
dequant = 2 * 0.000499 = 0.000998

# 相对误差：(0.001 - 0.000998) / 0.001 = 0.2% (看似还行)

# 但更小的值
tiny_value = 0.0001  
quantized = round(0.0001 / 0.000499) = 0
dequant = 0

# 完全损失！
```

2. **离群值问题严重**

```python
# 如果有1个离群通道
channel_max = [0.02] * 4095 + [0.20]  # 1个10倍大的通道

global_scale = 0.20 / 127 = 0.00157

# 99.98%的通道只能使用：
normal_range = 0.02 / 0.00157 = ±13
# 仅用到 [-13, 13]，浪费了 [-127, 127] 的 90%！
```

3. **精度测试数据**

```
LLaMA-7B 逐层INT8量化:
- PPL: 6.24 (基线: 5.67)
- 增加: 10.1%
- 某些任务准确率下降 3-5%
- 不推荐用于生产
```

#### 分组量化

**优点**：

1. **平衡精度和开销**

```python
# Group size = 128
# 精度：接近per-channel (差距 < 5%)
# 开销：仅 per-channel 的 1/32

# 对于4096通道:
per_channel_scales = 4096 个 (16 KB)
group_scales (g=128) = 32 个 (128 B)
# 节省 99.2% 的参数存储
```

2. **适合低比特量化**

INT4 量化的必要性：
```python
# INT4 仅 16 个离散值
# 逐层量化精度灾难：
# - PPL 增加 > 20%

# Per-channel 略好但仍不足：
# - PPL 增加 8-12%

# 分组量化 (group_size=128)：
# - PPL 增加 2-4%
# - 可用！
```

3. **局部适应性**

```python
# 捕获局部模式
group1 = channels[0:128]     # 可能对应某些特征
group2 = channels[128:256]   # 可能对应其他特征

# 每组独立scale，适应局部分布
# 相比全局更精确
# 相比per-channel更高效
```

4. **工具广泛采用**

```python
# GPTQ 默认 group_size=128
model_gptq = GPTQModel.load(
    "llama-7b",
    bits=4,
    group_size=128  # 标准配置
)

# AutoGPTQ
quantize_config = BaseQuantizeConfig(
    bits=4,
    group_size=128,  # 推荐值
    desc_act=False
)
```

**缺点**：

1. **比逐通道精度略低**

```
LLaMA-7B INT8:
- Per-channel: PPL 5.71 (+0.7%)
- Group-128: PPL 5.79 (+2.1%)
- 差距: 0.08 PPL (1.4%)

实际影响小，通常可接受
```

2. **需要选择组大小**

```python
# Group size 的影响
group_size = 32:   精度更高，开销更大
group_size = 128:  平衡点 (推荐)
group_size = 256:  精度略降，开销更小

# 需要实验确定最优值
# 通常 128 或 256
```

3. **实现稍复杂**

```python
# 需要管理分组逻辑
def group_quantize(weight, group_size=128):
    out_channels = weight.shape[0]
    n_groups = out_channels // group_size
    
    scales = []
    q_weight = torch.zeros_like(weight, dtype=torch.int8)
    
    for g in range(n_groups):
        start = g * group_size
        end = start + group_size
        group = weight[start:end]
        
        scale_g = group.abs().max() / 127
        scales.append(scale_g)
        q_weight[start:end] = (group / scale_g).round()
    
    return q_weight, torch.tensor(scales)
```

#### 逐通道量化

**优点**：

1. **精度最高**

```
LLaMA-7B INT8 per-channel:
- PPL: 5.71 (基线: 5.67)
- 增加: 0.7%
- 几乎无损！

LLaMA-13B INT8 per-channel:
- MMLU: 54.5% (基线: 54.8%)
- 下降: 0.3%
- 生产可用
```

2. **每个通道最优化**

```python
# 每个通道独立优化
for i in range(out_channels):
    scale_i = weight[i].abs().max() / 127
    # 每个通道使用最小步长
    # 最大化精度
```

3. **消除离群值影响**

```python
# 即使有离群通道
channel_max = [0.02] * 4095 + [0.20]

# Per-channel: 每个通道独立
normal_channels: scale = 0.02 / 127 = 0.000157
outlier_channel: scale = 0.20 / 127 = 0.00157

# 正常通道不受影响！
# 充分利用 [-127, 127] 范围
```

4. **标准实践**

```python
# 所有主流工具默认 per-channel 权重量化
- GPTQ: per-channel
- AWQ: per-channel
- TensorRT-LLM: per-channel
- bitsandbytes: per-channel

# 已成为行业标准
```

**缺点**：

1. **存储开销（但可接受）**

```python
# LLaMA-7B 参数分布
- 7B 个权重参数
- 约 30 个 Linear 层
- 每层 4096 输出通道

总 scales = 30 × 4096 = 122,880 个
存储 = 122,880 × 4 bytes = 491 KB

# 相比模型大小 (7B × 1 byte = 7 GB):
491 KB / 7 GB = 0.007%
# 完全可忽略
```

2. **计算稍复杂（但优化好）**

```python
# 反量化需要逐通道缩放
y_int8 = gemm_int8(x_q, w_q)  # [B, N]
y_fp = y_int8 * scales[None, :]  # 广播，逐通道

# 现代硬件很好优化
# TensorRT、cuBLAS 高效支持
# 实际开销 < 5%
```

3. **激活值应用受限**

```python
# 激活值 per-channel 困难
activation = [batch, seq_len, hidden_dim]

# 如果对 hidden_dim 做 per-channel:
# - 需要 hidden_dim 个 scales
# - 每个 batch 都要计算（动态）
# - 或需要大量校准（静态）

# 硬件支持也不如权重好
# 通常激活仍用 per-tensor
```

### 4. 实际测试对比

#### INT8 量化精度对比（LLaMA-7B）

| 方法          | PPL  | PPL增加 | MMLU          | 存储开销 |
| ------------- | ---- | ------- | ------------- | -------- |
| FP16          | 5.67 | -       | 47.6%         | -        |
| 逐层 INT8     | 6.24 | +10.1%  | 44.1% (-3.5%) | 4B       |
| 分组-256 INT8 | 5.82 | +2.6%   | 46.8% (-0.8%) | 64B      |
| 分组-128 INT8 | 5.76 | +1.6%   | 47.1% (-0.5%) | 128B     |
| 分组-64 INT8  | 5.73 | +1.1%   | 47.3% (-0.3%) | 256B     |
| 逐通道 INT8   | 5.71 | +0.7%   | 47.4% (-0.2%) | 16KB     |

**观察**：
- 逐层：不可用（>10%损失）
- 分组-128：平衡点
- 逐通道：最佳精度

#### INT4 量化精度对比（LLaMA-7B）

| 方法          | PPL  | PPL增加 | MMLU  |
| ------------- | ---- | ------- | ----- |
| FP16          | 5.67 | -       | 47.6% |
| 逐层 INT4     | 12.8 | +126%   | 27.3% |
| 分组-256 INT4 | 6.45 | +13.8%  | 43.1% |
| 分组-128 INT4 | 6.02 | +6.2%   | 45.7% |
| 分组-64 INT4  | 5.89 | +3.9%   | 46.5% |
| 逐通道 INT4   | 5.95 | +4.9%   | 46.1% |

**观察**：
- 逐层：完全失败
- **分组-128：最佳**（INT4的标准）
- 逐通道：反而略差（过拟合？）

**结论**：INT4必须用分组量化！

### 5. 组大小（Group Size）选择

#### 常见配置

```python
# INT8 量化
group_size = -1  # -1 表示 per-channel（无分组）
# 或
group_size = 256  # 如果需要更少开销

# INT4 量化  
group_size = 128  # 标准配置，最常用
# 或
group_size = 64   # 追求更高精度
```

#### 影响因素

**1. 比特数**
```
INT8: 可以用 per-channel (group_size=-1)
INT4: 必须分组 (group_size=64-128)
INT2: 更小分组 (group_size=32-64)
```

**2. 模型大小**
```
小模型 (< 7B): 
- 推荐 group_size=64
- 更细粒度保证精度

大模型 (> 13B):
- group_size=128 足够
- 模型冗余度高，鲁棒
```

**3. 硬件约束**
```
# 某些硬件对 group_size 有要求
# 例如需要是 32 的倍数
group_size ∈ {32, 64, 128, 256}
```

### 6. 代码实现示例

#### 逐层量化

```python
def per_layer_quantize(weight, bits=8):
    """最简单的量化"""
    n_levels = 2 ** (bits - 1) - 1
    scale = weight.abs().max() / n_levels
    
    q_weight = (weight / scale).round()
    q_weight = q_weight.clamp(-n_levels-1, n_levels)
    
    return q_weight.to(torch.int8), scale
```

#### 逐通道量化

```python
def per_channel_quantize(weight, bits=8):
    """标准的权重量化方法"""
    out_channels = weight.shape[0]
    n_levels = 2 ** (bits - 1) - 1
    
    # 每个输出通道的scale
    scales = weight.abs().max(dim=1, keepdim=True)[0] / n_levels
    
    # 量化
    q_weight = (weight / scales).round()
    q_weight = q_weight.clamp(-n_levels-1, n_levels)
    
    return q_weight.to(torch.int8), scales.squeeze()
```

#### 分组量化

```python
def group_quantize(weight, bits=8, group_size=128):
    """分组量化，INT4标准做法"""
    out_channels, in_channels = weight.shape
    n_levels = 2 ** (bits - 1) - 1
    
    # 确保能整除
    assert out_channels % group_size == 0
    n_groups = out_channels // group_size
    
    # 重塑为 [n_groups, group_size, in_channels]
    weight_grouped = weight.reshape(n_groups, group_size, in_channels)
    
    # 每组计算scale
    scales = weight_grouped.abs().max(dim=(1, 2), keepdim=True)[0] / n_levels
    
    # 量化
    q_weight = (weight_grouped / scales).round()
    q_weight = q_weight.clamp(-n_levels-1, n_levels)
    
    # 恢复形状
    q_weight = q_weight.reshape(out_channels, in_channels)
    scales = scales.reshape(n_groups)
    
    return q_weight.to(torch.int8), scales
```

### 7. 使用工具的实际配置

#### GPTQ

```python
from auto_gptq import AutoGPTQForCausalLM, BaseQuantizeConfig

# INT4 分组量化（推荐）
quantize_config = BaseQuantizeConfig(
    bits=4,
    group_size=128,  # 标准配置
    desc_act=False,
)

# INT8 逐通道量化
quantize_config = BaseQuantizeConfig(
    bits=8,
    group_size=-1,  # -1 表示 per-channel
    desc_act=False,
)

model = AutoGPTQForCausalLM.from_pretrained(
    model_name,
    quantize_config=quantize_config
)
```

#### AWQ

```python
from awq import AutoAWQForCausalLM

# AWQ 默认配置
model = AutoAWQForCausalLM.from_pretrained(
    model_path,
    config={
        "quant_bit": 4,
        "group_size": 128,  # 标准
        "zero_point": True
    }
)
```

### 8. 实际建议

#### INT8 量化

**推荐：逐通道（Per-channel）**

```python
# 精度最佳，开销可接受
quantize_config = {
    "bits": 8,
    "group_size": -1  # per-channel
}
```

**理由**：
- 精度损失 < 1%
- 存储开销可忽略（< 0.01%）
- 工具支持成熟
- 硬件优化好

#### INT4 量化

**推荐：分组-128（Group-128）**

```python
# INT4 的标准配置
quantize_config = {
    "bits": 4,
    "group_size": 128
}
```

**理由**：
- 精度可用（3-5% PPL增加）
- 平衡精度和开销
- 行业标准配置
- 广泛验证

#### 激活值量化

**推荐：逐层（Per-tensor）**

```python
# 激活值通常逐层量化
def quantize_activation(act):
    scale = act.abs().max() / 127
    return (act / scale).round(), scale
```

**理由**：
- 硬件支持最好
- 动态计算开销小
- Per-channel 收益不明显

### 9. 特殊场景

#### 超大模型（> 70B）

```python
# 追求极致压缩
quantize_config = {
    "bits": 3,  # 甚至 3-bit
    "group_size": 64,  # 更细粒度
}
```

#### 边缘设备

```python
# 平衡精度和资源
quantize_config = {
    "bits": 8,
    "group_size": 256,  # 稍粗粒度，节省内存
}
```

#### 长上下文模型

```python
# 注意 KV Cache 量化
# 可以对 K、V 独立设置
kv_quantize_config = {
    "bits": 8,
    "group_size": -1,  # per-channel K/V
}
```

### 总结

| 场景             | 推荐方法        | Group Size | 理由                   |
| ---------------- | --------------- | ---------- | ---------------------- |
| **INT8 权重**    | 逐通道          | -1         | 精度最佳，开销小       |
| **INT4 权重**    | 分组            | 128        | 必须分组，128是标准    |
| **INT3/2 权重**  | 分组            | 64         | 更细粒度保证精度       |
| **激活值**       | 逐层            | N/A        | 硬件支持好             |
| **大模型(>30B)** | 分组            | 128-256    | 冗余度高，粗粒度可接受 |
| **小模型(<7B)**  | 逐通道或分组-64 | -1或64     | 冗余度低，需要细粒度   |

**最佳实践总结**：
1. **INT8优先per-channel**（除非资源极度受限）
2. **INT4必须group-128**（行业标准）
3. **永远不要用逐层量化权重**（精度差）
4. **激活值默认逐层**（硬件友好）


---

## 相关笔记
<!-- 自动生成 -->

- [Per-tensor量化和Per-channel量化有何不同？哪种精度更高？](notes/熟悉大语言模型推理优化-技术层次/Per-tensor量化和Per-channel量化有何不同？哪种精度更高？.md) - 相似度: 39% | 标签: 熟悉大语言模型推理优化-技术层次, 熟悉大语言模型推理优化-技术层次/Per-tensor量化和Per-channel量化有何不同？哪种精度更高？.md

