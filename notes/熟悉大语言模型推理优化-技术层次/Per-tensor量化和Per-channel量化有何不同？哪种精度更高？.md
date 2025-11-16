---
created: '2025-10-19'
last_reviewed: '2025-10-25'
next_review: '2025-10-30'
review_count: 2
difficulty: medium
mastery_level: 0.43
tags:
- 熟悉大语言模型推理优化-技术层次
- 熟悉大语言模型推理优化-技术层次/Per-tensor量化和Per-channel量化有何不同？哪种精度更高？.md
related_outlines: []
---
# Per-tensor量化和Per-channel量化有何不同？哪种精度更高？

## 面试标准答案

Per-tensor量化对整个张量使用单一的scale和zero-point参数，实现简单但可能无法适应不同通道间的分布差异。Per-channel量化为每个输出通道（或输入通道）维护独立的量化参数，能更精细地适应各通道的数值范围，精度显著更高，通常能将量化误差降低50-80%。在LLM推理中，权重通常采用Per-channel量化（几乎无精度损失），而激活值受限于硬件支持多用Per-tensor量化。Per-channel的代价是存储开销增加（需存储C个scale）和计算略复杂。

## 详细讲解

### 1. 基本概念

#### Per-tensor量化（Per-layer）

**定义**：
- 整个张量共享一组量化参数
- 一个scale，一个zero-point（如使用非对称）
- 最粗粒度的量化方式

**示例**：
```python
# 权重形状: [4096, 4096]
# 所有 4096×4096 个元素使用相同的scale

scale = tensor.abs().max() / 127
q_tensor = round(tensor / scale)
# 参数量: 1个scale
```

#### Per-channel量化

**定义**：
- 每个通道独立量化参数
- 对于权重矩阵 W[out_channels, in_channels]：
  - **Per-output-channel**：每个输出通道一个scale（最常见）
  - **Per-input-channel**：每个输入通道一个scale（少见）
- 更细粒度，适应通道间差异

**示例**：
```python
# 权重形状: [4096, 4096]
# 4096个输出通道，每个通道一个scale

scales = []
for i in range(out_channels):
    scale_i = tensor[i, :].abs().max() / 127
    scales.append(scale_i)
    q_tensor[i, :] = round(tensor[i, :] / scale_i)
# 参数量: 4096个scales
```

### 2. 图示对比

#### Per-tensor量化示例

```
权重矩阵 W [3, 4]:
Channel 0: [-0.5, -0.3,  0.2,  0.4]  → max=0.5
Channel 1: [-2.0, -1.5,  1.0,  1.8]  → max=2.0
Channel 2: [-0.1, -0.08, 0.05, 0.09] → max=0.1

Per-tensor:
全局max = 2.0
scale = 2.0 / 127 = 0.0157

量化后:
Channel 0: [-32, -19, 13, 25]      → 精度损失适中
Channel 1: [-127, -95, 64, 115]    → 精度好（接近满量程）
Channel 2: [-6, -5, 3, 6]          → 精度损失大（仅用5%范围）
```

#### Per-channel量化示例

```
相同权重矩阵 W [3, 4]:

Per-channel:
scale[0] = 0.5 / 127 = 0.0039
scale[1] = 2.0 / 127 = 0.0157
scale[2] = 0.1 / 127 = 0.0008

量化后:
Channel 0: [-127, -76, 51, 102]    → 精度好（充分利用）
Channel 1: [-127, -95, 64, 115]    → 精度好
Channel 2: [-125, -100, 63, 113]   → 精度好（充分利用）
```

**关键差异**：
- Per-channel每个通道都充分利用 [-127, 127] 范围
- Per-tensor被最大通道"拖累"，小通道精度差

### 3. 精度对比

#### 理论分析

**量化误差**：
```
量化误差 ∝ scale (步长越大，误差越大)
```

**Per-tensor**：
```
scale = max(|w_all|) / 127

如果通道分布差异大:
- 某些通道远小于max → 大步长 → 大误差
- 误差可能是最优的2-10倍
```

**Per-channel**：
```
scale[i] = max(|w_channel[i]|) / 127

每个通道独立优化:
- 每个通道使用最小可能的步长
- 误差接近理论最优
```

#### 实验数据

**LLaMA-7B权重量化精度**：

| 方法             | 困惑度(PPL) | PPL增加 | 参数开销       |
| ---------------- | ----------- | ------- | -------------- |
| FP16基线         | 5.67        | -       | -              |
| Per-tensor INT8  | 6.24        | +10.1%  | 1 scale/层     |
| Per-channel INT8 | 5.71        | +0.7%   | 4096 scales/层 |
| Per-channel INT4 | 5.89        | +3.9%   | 4096 scales/层 |

**关键发现**：
- Per-channel INT8几乎无损（<1%）
- Per-tensor INT8显著损失（~10%）
- Per-channel使INT4可用（<4%损失）

**ResNet-50 激活量化**：

| 方法        | Top-1准确率 | 下降  |
| ----------- | ----------- | ----- |
| FP32基线    | 76.1%       | -     |
| Per-tensor  | 74.2%       | -1.9% |
| Per-channel | 75.7%       | -0.4% |

### 4. 为什么Per-channel精度更高？

#### 原因1：通道间分布差异

**实际观察**（Transformer权重）：
```python
# 统计LLaMA某Linear层权重
W.shape = [4096, 4096]

# 各通道的最大绝对值
channel_maxs = [w[i].abs().max() for i in range(4096)]
print(f"Min: {min(channel_maxs):.4f}")  # 0.0123
print(f"Max: {max(channel_maxs):.4f}")  # 0.0856
print(f"Ratio: {max/min:.2f}x")         # 6.96x

# 结论：通道间差异近7倍！
```

**影响**：
- Per-tensor用0.0856统一量化
- 小通道（0.0123）被压缩到[-18, 18]范围
- 利用率仅14%，精度损失86%

#### 原因2：离群通道（Outlier Channels）

**现象**：
- 某些通道值域远大于平均
- 占比<5%但影响全局scale

**Per-tensor的困境**：
```python
channel_maxs = [0.05] * 95 + [0.50] * 5  # 5%的大通道

per_tensor_scale = 0.50 / 127 = 0.0039
# 95%的通道只能用到 ±0.05/0.0039 = ±13 的量化范围
# 浪费了 (-127, -13) 和 (13, 127) 的空间
```

**Per-channel的优势**：
```python
# 每个通道独立
for i in range(100):
    scale[i] = channel_max[i] / 127
# 小通道也能充分利用 [-127, 127]
```

#### 原因3：数值范围的自适应

Per-channel自动适应：
- 大通道 → 大scale → 保持表示范围
- 小通道 → 小scale → 提高精度
- 最优权衡

### 5. 在不同张量上的应用

#### 权重量化

**推荐：Per-channel（输出通道维度）**

```python
# W: [out_features, in_features]
# 对每个输出通道量化

def per_channel_quantize_weight(weight):
    out_channels = weight.shape[0]
    scales = []
    q_weight = torch.zeros_like(weight, dtype=torch.int8)
    
    for i in range(out_channels):
        scale = weight[i].abs().max() / 127
        scales.append(scale)
        q_weight[i] = torch.round(weight[i] / scale).clamp(-128, 127)
    
    return q_weight, torch.tensor(scales)
```

**原因**：
- 权重静态，可以离线量化
- 存储开销可接受（4096个float vs 4096×4096个int8）
- 计算时可高效处理（向量化）
- 精度提升显著

#### 激活值量化

**常用：Per-tensor（受限于硬件）**

```python
# Activation: [batch, seq_len, hidden_dim]
# 整个张量一个scale

def per_tensor_quantize_activation(activation):
    scale = activation.abs().max() / 127
    q_activation = torch.round(activation / scale).clamp(-128, 127)
    return q_activation, scale
```

**原因**：
- 激活动态变化，per-channel开销大
- 硬件（Tensor Core）优化per-tensor
- 实时量化需要高效

**高级方法：Per-token或Per-channel（研究中）**
- 某些框架支持（如SmoothQuant）
- 需要特殊硬件或kernel支持

### 6. 计算影响

#### Per-tensor的GEMM

```python
# Y = X @ W
# X_q: [B, K], scale_x (1个)
# W_q: [K, N], scale_w (1个)

Y_q = X_q @ W_q  # INT8 GEMM
Y = Y_q * (scale_x * scale_w)  # 标量乘法

# 反量化：一次全局乘法
```

#### Per-channel的GEMM

```python
# Y = X @ W
# X_q: [B, K], scale_x (1个)
# W_q: [K, N], scale_w (N个，每个输出通道一个)

Y_q = X_q @ W_q  # INT8 GEMM
Y = Y_q * (scale_x * scale_w[None, :])  # 广播乘法

# 反量化：逐通道乘法（向量化）
```

**性能影响**：
- 额外开销：逐元素乘法（memory-bound）
- GEMM仍是瓶颈，额外开销<5%
- 精度提升远超性能损失

### 7. 硬件支持

#### NVIDIA GPU

**Tensor Core（A100/H100）**：
- **Per-tensor**：原生高效支持
- **Per-channel（权重）**：支持且优化
  - WMMA API支持per-channel scale
  - TensorRT-LLM默认per-channel权重量化

**实现**：
```cuda
// Per-channel权重量化的高效实现
// 在shared memory中准备scales
__shared__ float scales[N_CHANNELS];

// GEMM后逐通道缩放
for (int i = tid; i < N; i += blockDim.x) {
    output[i] = int_output[i] * scales[i];
}
```

#### 移动端

**ARM CPU**：
- Per-tensor：支持
- Per-channel：支持但需手动优化

**专用AI加速器**：
- 大多支持per-channel权重量化
- 激活通常per-tensor

### 8. 存储开销

#### Per-tensor

**参数量**：
```
每个量化张量: 1个scale (+ 可选1个zero_point)
存储: 4 bytes (FP32) 或 2 bytes (FP16)

例如 LLaMA-7B (32层):
- 32层 × 3个Linear (QKV, O, Gate, Up, Down) = 96个张量
- 存储: 96 × 4 bytes = 384 bytes
- 可忽略不计
```

#### Per-channel

**参数量**：
```
每个量化张量: C个scales (C = 通道数)
存储: C × 4 bytes

例如 LLaMA-7B:
- 隐藏层维度: 4096
- 每层6个Linear × 4096 scales = 24576 scales
- 32层: 786432 scales
- 存储: 786432 × 4 = 3.1 MB

相比模型总大小 (7B × 1 byte = 7GB):
3.1MB / 7000MB = 0.04%
- 仍然可忽略
```

**结论**：存储开销不是问题

### 9. 实际应用建议

#### 权重量化

**强烈推荐：Per-channel**

```python
# 使用GPTQ、AWQ等工具默认per-channel
from transformers import AutoModelForCausalLM
from auto_gptq import AutoGPTQForCausalLM

model = AutoGPTQForCausalLM.from_pretrained(
    "llama-7b",
    quantize_config={
        "bits": 8,
        "group_size": -1,  # -1表示per-channel
        "desc_act": False
    }
)
# 自动使用per-output-channel量化
```

**几乎无精度损失的原因**：
- 权重分布稳定
- Per-channel完全适应
- 离线量化质量高

#### 激活值量化

**场景1：追求性能（主流）**
- 使用：Per-tensor
- 工具：TensorRT-LLM、vLLM
- 原因：硬件优化好，开销小

**场景2：追求精度（研究）**
- 使用：Per-token或Per-channel
- 工具：SmoothQuant、自定义kernel
- 原因：精度提升5-10%

#### 实际代码

```python
import torch

def quantize_model(model):
    """量化模型：权重per-channel，激活per-tensor"""
    for name, module in model.named_modules():
        if isinstance(module, torch.nn.Linear):
            # 权重：per-channel量化
            weight = module.weight.data
            scales = weight.abs().max(dim=1, keepdim=True)[0] / 127
            q_weight = (weight / scales).round().clamp(-128, 127)
            
            # 存储量化权重和scales
            module.register_buffer('q_weight', q_weight.to(torch.int8))
            module.register_buffer('weight_scales', scales)
            
            # 激活量化在forward中动态per-tensor
            def quantized_forward(self, x):
                # 激活per-tensor量化
                scale_x = x.abs().max() / 127
                x_q = (x / scale_x).round().clamp(-128, 127)
                
                # GEMM
                y_q = F.linear(x_q, self.q_weight.float())
                
                # Per-channel反量化
                y = y_q * (scale_x * self.weight_scales.T)
                return y
            
            # 替换forward
            module.forward = quantized_forward.__get__(module)
```

### 10. 更细粒度：Group Quantization

#### 概念

介于per-tensor和per-channel之间：
- 将通道分组
- 每组一个scale
- 例如：4096通道 → 32组 → 每组128通道

**Per-group量化**：
```python
def per_group_quantize(weight, group_size=128):
    out_channels = weight.shape[0]
    n_groups = out_channels // group_size
    scales = []
    
    for g in range(n_groups):
        start = g * group_size
        end = start + group_size
        scale_g = weight[start:end].abs().max() / 127
        scales.append(scale_g)
        # 量化该组
    
    return q_weight, scales
```

**优势**：
- 精度介于per-tensor和per-channel之间
- 存储开销更小
- GPTQ常用（group_size=128）

### 11. 精度排序

```
Per-channel > Per-group (128) > Per-group (512) > Per-tensor

精度差异（典型）:
- Per-channel vs Per-tensor: 精度提升50-80%
- Per-group (128) vs Per-tensor: 精度提升40-60%
- Per-channel vs Per-group (128): 精度提升5-15%
```

### 总结

| 维度           | Per-tensor       | Per-channel                    |
| -------------- | ---------------- | ------------------------------ |
| **精度**       | 低               | 高（提升50-80%）               |
| **存储**       | 最小（1个scale） | 小（C个scales，<0.1%模型大小） |
| **计算**       | 最快             | 稍慢（~5%开销）                |
| **硬件支持**   | 最好             | 良好                           |
| **权重量化**   | 不推荐           | **强烈推荐**                   |
| **激活量化**   | **主流选择**     | 研究中                         |
| **实现复杂度** | 简单             | 中等                           |

**最佳实践**：
- **权重**：始终使用Per-channel（几乎无理由不用）
- **激活**：默认Per-tensor（性能优先），研究场景可尝试Per-channel

现代LLM量化工具（GPTQ、AWQ、TensorRT-LLM）都默认对权重使用Per-channel量化，这已成为行业标准。


---

## 相关笔记
<!-- 自动生成 -->

- [逐层量化、逐通道量化、分组量化（Group_Quantization）的优缺点是什么？](notes/熟悉大语言模型推理优化-技术层次/逐层量化、逐通道量化、分组量化（Group_Quantization）的优缺点是什么？.md) - 相似度: 39% | 标签: 熟悉大语言模型推理优化-技术层次, 熟悉大语言模型推理优化-技术层次/逐层量化、逐通道量化、分组量化（Group_Quantization）的优缺点是什么？.md

