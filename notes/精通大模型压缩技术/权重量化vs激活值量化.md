---
created: '2025-10-19'
last_reviewed: null
next_review: '2025-10-19'
review_count: 0
difficulty: medium
mastery_level: 0.0
tags:
- 精通大模型压缩技术
- 精通大模型压缩技术/权重量化vs激活值量化.md
related_outlines: []
---
# 权重量化 vs 激活值量化

## 面试标准答案（精简版）

**权重量化**是对模型参数（权重和偏置）进行量化，特点是静态不变、可离线完成、分布相对稳定，常用对称量化。**激活值量化**是对神经网络层间的中间计算结果进行量化，特点是动态变化、依赖输入数据、分布范围大，常用非对称量化。仅权重量化（Weight-only Quantization）内存占用小但计算仍是浮点；权重+激活量化（W8A8）可实现完全整数推理，速度更快但精度损失更大。

---

## 详细讲解

### 1. 权重量化（Weight Quantization）

#### 定义
将神经网络的权重参数从高精度浮点数（FP32/FP16）转换为低精度表示（INT8/INT4）。

#### 特点
- **静态性**：权重在训练完成后固定不变
- **离线处理**：可以在模型转换阶段完成，无需运行时计算
- **分布可预测**：可以通过校准数据集统计权重分布
- **对称性**：权重分布通常接近高斯分布，均值接近0

#### 量化方法

**Per-tensor 量化**
- 整个张量共享一个 scale 和 zero point
- 简单但精度较低

**Per-channel 量化**（推荐）
- 每个输出通道（或卷积核）独立量化
- 更精细，精度更高
- 示例：对于权重 \(W \in \mathbb{R}^{C_{out} \times C_{in} \times K \times K}\)，有 \(C_{out}\) 个不同的 scale

```python
# Per-channel 权重量化示例
def per_channel_quantize_weight(weight):
    # weight shape: [out_channels, in_channels, h, w]
    scales = []
    quantized_weights = []
    
    for i in range(weight.shape[0]):  # 遍历输出通道
        channel_weight = weight[i]
        max_val = max(abs(channel_weight.max()), abs(channel_weight.min()))
        scale = max_val / 127  # INT8
        q_weight = np.round(channel_weight / scale).astype(np.int8)
        
        scales.append(scale)
        quantized_weights.append(q_weight)
    
    return np.array(quantized_weights), np.array(scales)
```

#### 优点
- **存储压缩**：模型大小减少 4倍（INT8）或 8倍（INT4）
- **易于实现**：离线处理，无需修改推理流程
- **精度保持好**：分布稳定，量化误差可控

#### 缺点
- **计算仍是浮点**：如果激活值不量化，矩阵乘法仍需反量化为FP32
- **加速有限**：内存带宽优化，但计算效率提升不大

### 2. 激活值量化（Activation Quantization）

#### 定义
将神经网络层间的中间激活值从浮点数量化为整数。

#### 特点
- **动态性**：激活值随输入数据变化
- **在线处理**：需要在推理时实时量化
- **分布多样**：不同层、不同输入的激活分布差异大
- **非对称性**：ReLU等激活函数导致激活值偏向正值

#### 量化方法

**静态量化**
- 使用校准数据集预先统计激活值的范围
- 推理时使用固定的 scale 和 zero point
- 速度快但可能对分布外数据精度下降

**动态量化**
- 推理时动态计算每个batch的激活值范围
- 精度高但有额外计算开销

```python
# 静态激活量化（校准阶段）
def calibrate_activation(model, calibration_data):
    activation_ranges = {}
    
    def hook_fn(name):
        def hook(module, input, output):
            if name not in activation_ranges:
                activation_ranges[name] = {'min': [], 'max': []}
            activation_ranges[name]['min'].append(output.min().item())
            activation_ranges[name]['max'].append(output.max().item())
        return hook
    
    # 注册钩子，收集激活范围
    for name, module in model.named_modules():
        module.register_forward_hook(hook_fn(name))
    
    # 运行校准数据
    for data in calibration_data:
        model(data)
    
    # 计算统计范围（如99.9%分位数）
    for name in activation_ranges:
        activation_ranges[name]['min'] = np.percentile(activation_ranges[name]['min'], 0.1)
        activation_ranges[name]['max'] = np.percentile(activation_ranges[name]['max'], 99.9)
    
    return activation_ranges
```

#### 优点
- **完全整数推理**：配合权重量化实现 INT8 推理
- **大幅加速**：利用硬件 INT8 算子，吞吐量提升 2-4倍
- **端到端优化**：整个推理链路都是低精度

#### 缺点
- **精度挑战**：激活值量化对精度影响更大
- **分布敏感**：异常值或分布偏移导致精度下降
- **实现复杂**：需要修改整个推理流程

### 3. 组合策略对比

| 策略            | 权重精度 | 激活精度 | 内存占用 | 计算速度 | 精度保持 | 应用场景          |
| --------------- | -------- | -------- | -------- | -------- | -------- | ----------------- |
| **FP16**        | FP16     | FP16     | 高       | 中       | 最好     | GPU训练/推理      |
| **Weight-only** | INT8/4   | FP16     | 低       | 中       | 好       | 大模型推理（LLM） |
| **W8A8**        | INT8     | INT8     | 低       | 快       | 较好     | 边缘设备          |
| **W4A8**        | INT4     | INT8     | 最低     | 快       | 中       | 极端压缩          |

### 4. Weight-only Quantization（仅权重量化）

近年来在**大语言模型（LLM）**中流行的方案。

#### 原理
- 仅量化权重到 INT4/INT8
- 激活值保持 FP16
- 推理时动态反量化权重

#### 优势
- **内存友好**：模型参数占主要内存（如 LLaMA-70B），压缩效果明显
- **精度损失小**：激活值保留高精度，对困难的 Outlier 激活不敏感
- **实现简单**：无需校准激活值范围

#### 劣势
- **计算效率有限**：矩阵乘法仍是 FP16，无法充分利用 INT8 加速

#### 代表框架
- **GPTQ**：基于Hessian的后训练量化
- **AWQ**：Activation-aware Weight Quantization
- **LLM.int8()**：混合精度量化

### 5. W8A8 全整数量化

#### 原理
- 权重和激活都量化到 INT8
- 整个推理链路使用整数运算

#### 量化感知训练（QAT）
在训练时模拟量化，让模型适应量化误差：

```python
# 量化感知训练伪代码
class QuantizedLinear(nn.Module):
    def __init__(self, in_features, out_features):
        super().__init__()
        self.weight = nn.Parameter(torch.randn(out_features, in_features))
        # 量化参数
        self.weight_scale = nn.Parameter(torch.ones(out_features))
        self.act_scale = nn.Parameter(torch.ones(1))
        
    def forward(self, x):
        # 模拟权重量化
        w_q = fake_quantize(self.weight, self.weight_scale)
        # 模拟激活量化
        x_q = fake_quantize(x, self.act_scale)
        # 浮点计算（训练时）
        output = F.linear(x_q, w_q)
        return output

def fake_quantize(x, scale):
    # 模拟量化+反量化过程
    x_q = torch.round(x / scale).clamp(-128, 127)
    x_dq = x_q * scale
    return x_dq
```

#### 优势
- **速度最快**：充分利用硬件 INT8 加速
- **能效最优**：INT8 计算功耗远低于 FP16

#### 挑战
- **激活 Outliers**：某些层存在极端激活值
- **逐层精度损失累积**：深层网络误差传播
- **需要 QAT 或精细校准**：后训练量化（PTQ）可能精度不足

### 6. 实践建议

#### 选择权重量化（Weight-only）
- 部署大语言模型（参数量 > 10B）
- 内存受限但算力充足
- 追求精度优先

#### 选择 W8A8 全量化
- 边缘设备部署（手机、IoT）
- 小模型（< 1B 参数）
- 追求推理速度

#### 混合精度策略
- **Sensitive layers**（如第一层、最后一层）保持高精度
- **Outlier channels** 单独处理或保持 FP16
- **Per-layer 量化**：不同层使用不同bit数

### 7. 激活值量化的难点

#### Outlier 问题
某些 Token 的激活值异常大（如大语言模型的特定位置），导致：
- 量化范围扩大，其他值精度损失
- 解决方案：Outlier 通道保持 FP16（如 LLM.int8()）

#### 动态范围
不同输入数据的激活值范围差异大：
- 解决方案：分位数剪裁、动态量化、Smooth Quant

#### 累积误差
多层网络中量化误差逐层传播：
- 解决方案：量化感知训练、重校准（Post-training calibration）

---

## 总结

- **权重量化**：静态、易实现、内存友好，是量化的第一步
- **激活值量化**：动态、高难度、速度提升关键，需要精细优化
- **LLM 时代**：Weight-only 成为主流（GPTQ, AWQ）
- **边缘部署**：W8A8 全量化仍是最优选择
- **未来趋势**：W4A16, W2A16 等极低比特权重 + 高精度激活


---

## 相关笔记
<!-- 自动生成 -->

暂无相关笔记

