---
created: '2025-10-19'
last_reviewed: null
next_review: '2025-10-19'
review_count: 0
difficulty: medium
mastery_level: 0.0
tags:
- 熟悉大语言模型推理优化-技术层次
- 熟悉大语言模型推理优化-技术层次/仅量化权重（Weight-Only Quantization）和同时量化激活有什么区别？.md
related_outlines: []
---
# 仅量化权重（Weight-Only Quantization）和同时量化激活有什么区别？

## 面试标准答案

仅量化权重（Weight-Only）只将模型参数从FP16压缩到INT8/INT4，激活值保持FP16，主要减少内存占用和带宽，但矩阵乘法仍需反量化为FP16计算。同时量化激活则权重和激活都用INT8，可以使用INT8 Tensor Core进行纯整数运算，速度更快（3-4倍）。权重量化简单无需校准数据、精度损失小（<1%），但加速有限；激活量化难度大、需要校准、精度损失较多（1-3%），但可充分利用硬件加速。LLM推理常是内存瓶颈，因此权重量化已能带来显著收益。

## 详细讲解

### 1. 核心区别概览

#### 1.1 量化对象

**Weight-Only量化**：
- 仅量化模型参数（权重矩阵）
- 激活值、梯度保持FP16/FP32
- 中间计算结果保持高精度

**Weight + Activation量化**：
- 权重和激活值都量化到INT8
- 整个前向传播路径都是低精度
- 可以实现完全的整数运算

#### 1.2 计算流程对比

**Weight-Only**：
```python
# 伪代码
W_int8 = quantize(W_fp16)  # 离线量化权重
X_fp16 = input  # 激活值保持FP16

# 推理时
W_fp16 = dequantize(W_int8)  # 运行时反量化
Y_fp16 = matmul(X_fp16, W_fp16)  # FP16计算
```

**Weight + Activation**：
```python
# 伪代码
W_int8 = quantize(W_fp16)  # 离线量化权重
X_fp16 = input

# 推理时
X_int8 = quantize(X_fp16, scale_x)  # 动态量化激活
Y_int32 = matmul_int8(X_int8, W_int8)  # INT8 Tensor Core
Y_fp16 = dequantize(Y_int32, scale_x * scale_w)
```

### 2. 内存占用对比

#### 2.1 模型大小

**Weight-Only**：
- **权重**：降低到1/4（INT8）或1/8（INT4）
- **激活值**：不变（仍为FP16）
- **总内存**：取决于batch size和序列长度

**示例（LLaMA-7B）**：
```
FP16: 权重13GB + 激活2GB(batch=1) = 15GB
Weight-Only INT8: 权重3.25GB + 激活2GB = 5.25GB (节省65%)
Weight-Only INT4: 权重1.6GB + 激活2GB = 3.6GB (节省76%)
```

**Weight + Activation INT8**：
```
权重3.25GB + 激活0.5GB = 3.75GB (节省75%)
```

#### 2.2 内存节省的实际意义

对于LLM推理：
- **小batch（1-4）**：权重占主导，Weight-Only效果显著
- **大batch（>32）**：激活值占主导，激活量化更重要
- **长序列**：KV Cache占主导（可单独量化）

### 3. 计算性能对比

#### 3.1 Weight-Only的性能

**优势**：
- 减少内存带宽：从DRAM加载权重的带宽降低4倍（INT8）
- **关键点**：LLM推理常是内存瓶颈（memory-bound）

**局限**：
- 需要运行时反量化：额外开销
- GEMM仍是FP16计算：不能用INT8 Tensor Core
- 加速比：**1.5-2.5倍**（主要来自带宽节省）

**性能分析**：
```
Decode阶段（batch=1, seq=1）：
- 访存时间占80%，计算时间占20%
- 权重量化减少80%中的4倍 = 约3倍加速
- 实际加速：2-2.5倍（考虑反量化开销）
```

#### 3.2 Weight + Activation的性能

**优势**：
- 使用INT8 Tensor Core：计算速度快16倍（vs FP16 CUDA Core）
- 内存带宽降低：权重和激活都减少
- **理论加速比**：3-4倍（vs FP16）

**局限**：
- 需要动态量化激活值：有开销
- 量化/反量化操作：增加额外计算
- 实际加速：**2.5-3.5倍**

#### 3.3 实测性能对比（A100 GPU）

| 模型     | 配置             | 延迟(ms) | 吞吐量(tokens/s) | 加速比 |
| -------- | ---------------- | -------- | ---------------- | ------ |
| LLaMA-7B | FP16             | 45       | 22               | 1.0x   |
|          | Weight-Only INT8 | 20       | 50               | 2.3x   |
|          | Weight-Only INT4 | 15       | 67               | 3.0x   |
|          | W8A8 (权重+激活) | 14       | 71               | 3.2x   |

**结论**：
- Weight-Only INT4 vs W8A8性能接近
- 但Weight-Only INT4精度损失更大

### 4. 实现复杂度对比

#### 4.1 Weight-Only实现

**优点**：
- **无需校准**：权重是静态的，离线量化
- **精度可控**：权重分布相对规则
- **实现简单**：标准量化流程

**实现步骤**：
```python
# 1. 离线量化权重
for layer in model.layers:
    W = layer.weight  # FP16
    scale = W.abs().max() / 127
    W_int8 = torch.round(W / scale).to(torch.int8)
    layer.weight_int8 = W_int8
    layer.weight_scale = scale

# 2. 推理时反量化
def forward(self, x):
    W_fp16 = self.weight_int8.to(torch.float16) * self.weight_scale
    return F.linear(x, W_fp16)
```

#### 4.2 Weight + Activation实现

**难点**：
- **需要校准**：激活值是动态的，需要统计分布
- **异常值处理**：激活值常有outliers
- **动态量化开销**：每次推理都需量化激活值

**实现步骤**：
```python
# 1. 校准阶段（收集激活值统计）
for batch in calibration_data:
    activations = model(batch, collect_stats=True)
    for layer_name, act in activations.items():
        stats[layer_name].update(act)

# 2. 确定量化参数
for layer_name in stats:
    scale = compute_scale(stats[layer_name])  # MinMax/KL散度等
    layer.activation_scale = scale

# 3. 推理时动态量化
def forward(self, x):
    # 量化激活
    x_int8 = torch.round(x / self.activation_scale).clamp(-128, 127).to(torch.int8)
    
    # INT8 GEMM
    y_int32 = torch.matmul(x_int8, self.weight_int8.T)
    
    # 反量化
    y_fp16 = y_int32.to(torch.float16) * (self.activation_scale * self.weight_scale)
    return y_fp16
```

### 5. 精度影响对比

#### 5.1 Weight-Only精度

**典型精度损失**：
- INT8: <0.5% PPL增加
- INT4: 1-2% PPL增加（需要GPTQ/AWQ等高级方法）
- INT3: 2-5% PPL增加（极限压缩）

**原因**：
- 权重分布相对规则（接近高斯）
- 可以使用Per-channel量化（每个输出通道独立scale）
- 静态量化，可以离线优化

**示例（LLaMA-7B）**：
```
FP16:           PPL = 5.68
Weight-Only INT8:  PPL = 5.71 (+0.5%)
Weight-Only INT4:  PPL = 5.82 (+2.5%)
```

#### 5.2 Weight + Activation精度

**典型精度损失**：
- W8A8: 1-3% PPL增加
- W4A8: 2-4% PPL增加
- W4A4: 5-10% PPL增加（通常不可接受）

**原因**：
- 激活值分布复杂（长尾、异常值）
- 动态量化：无法像权重那样精细优化
- 误差累积：多层量化误差会累积

**示例（LLaMA-7B）**：
```
FP16:     PPL = 5.68
W8A8:     PPL = 5.85 (+3.0%)
W8A8 + SmoothQuant: PPL = 5.75 (+1.2%)
```

#### 5.3 不同层的敏感度

| 层类型        | Weight-Only影响 | Activation量化影响 |
| ------------- | --------------- | ------------------ |
| Embedding     | 小              | 小                 |
| Q/K/V投影     | 小              | 中                 |
| Attention输出 | 小              | 大                 |
| FFN上投影     | 小              | 大                 |
| FFN下投影     | 中              | 大                 |
| LayerNorm     | N/A             | 大（通常保持FP16） |
| 输出层        | 中              | 大                 |

**结论**：激活量化对关键层影响更大，需要混合精度策略。

### 6. 适用场景对比

#### 6.1 Weight-Only适用场景

**最佳场景**：
- **小batch推理**（batch=1-4）：内存瓶颈主导
- **长序列生成**：每个token生成都需加载完整权重
- **内存受限**：单卡部署大模型
- **精度敏感**：金融、医疗等领域

**典型应用**：
- ChatGPT式的对话系统（batch=1）
- 个人设备部署（内存<16GB）
- API服务（低延迟优先）

**实际案例**：
```
场景：单卡A100(40GB)部署LLaMA-65B
- FP16: 不可能（需要130GB）
- Weight-Only INT4: 可行（需要32GB）
- W8A8: 也可行，但精度更低
选择：Weight-Only INT4
```

#### 6.2 Weight + Activation适用场景

**最佳场景**：
- **大batch推理**（batch>32）：激活占内存比例大
- **计算密集场景**：需要充分利用Tensor Core
- **吞吐量优先**：离线批处理任务
- **可接受精度损失**：搜索、推荐等

**典型应用**：
- 批量数据处理
- 离线生成任务
- 大规模推理集群

**实际案例**：
```
场景：A100集群，批量翻译任务（batch=128）
- FP16: 吞吐量100 samples/s
- Weight-Only INT8: 吞吐量230 samples/s（2.3x）
- W8A8: 吞吐量320 samples/s（3.2x）
选择：W8A8（吞吐量最重要）
```

### 7. 混合策略

#### 7.1 渐进式量化
```python
# 策略：先Weight-Only，再考虑激活
# 阶段1：Weight-Only INT8
model = quantize_weights(model, bits=8)
if accuracy_acceptable:
    deploy(model)
else:
    # 阶段2：尝试W8A8
    model = quantize_weights_and_activations(model)
```

#### 7.2 分层混合
```python
# 对不同层使用不同策略
quantization_config = {
    'embedding': 'weight_only_int8',
    'attention.qkv': 'weight_only_int8',
    'attention.out': 'w8a8',  # 激活分布较规则
    'ffn.up': 'weight_only_int8',  # 激活有异常值
    'ffn.down': 'w8a8',
    'lm_head': 'weight_only_int8',  # 最后一层保守
}
```

#### 7.3 动态切换
```python
# 根据batch size动态选择
def forward(self, x):
    if x.shape[0] <= 4:  # 小batch
        return self.forward_weight_only(x)
    else:  # 大batch
        return self.forward_w8a8(x)
```

### 8. 实现优化技巧

#### 8.1 Weight-Only优化

**技巧1：分组量化**
```python
# 权重按组量化，提高精度
def group_quantize(W, group_size=128):
    # W shape: [out_features, in_features]
    num_groups = in_features // group_size
    
    W_grouped = W.reshape(out_features, num_groups, group_size)
    scales = W_grouped.abs().max(dim=-1).values / 127
    W_int8 = (W_grouped / scales.unsqueeze(-1)).round()
    
    return W_int8, scales
```

**技巧2：快速反量化**
```python
# 使用向量化指令
@torch.jit.script
def fast_dequantize(W_int8, scales):
    return W_int8.to(torch.float16) * scales.unsqueeze(-1)
```

#### 8.2 Weight + Activation优化

**技巧1：激活值缓存**
```python
# 对于重复的输入模式，缓存量化参数
class ActivationQuantizer:
    def __init__(self):
        self.scale_cache = {}
    
    def quantize(self, x):
        key = (x.shape, x.dtype)
        if key in self.scale_cache:
            scale = self.scale_cache[key]
        else:
            scale = x.abs().max() / 127
            self.scale_cache[key] = scale
        
        return torch.round(x / scale).clamp(-128, 127).to(torch.int8), scale
```

**技巧2：融合量化算子**
```python
# 将量化、GEMM、反量化融合为一个kernel
# 减少数据搬运
@torch.jit.script
def fused_quantized_linear(x_fp16, W_int8, scale_w, scale_x):
    # 这个实际需要用CUDA实现
    x_int8 = quantize(x_fp16, scale_x)
    y_int32 = matmul_int8(x_int8, W_int8)
    y_fp16 = dequantize(y_int32, scale_x * scale_w)
    return y_fp16
```

### 9. 工具和库支持

#### 9.1 Weight-Only支持

**GPTQ**（最流行）：
```python
from auto_gptq import AutoGPTQForCausalLM

model = AutoGPTQForCausalLM.from_pretrained(
    "model_name",
    quantize_config={"bits": 4, "group_size": 128}
)
```

**AWQ**：
```python
from awq import AutoAWQForCausalLM

model = AutoAWQForCausalLM.from_pretrained("model_name")
model.quantize(calib_data, bits=4)
```

**llama.cpp**：
- 主要用于CPU推理
- 支持INT4/INT8 Weight-Only

#### 9.2 Weight + Activation支持

**TensorRT-LLM**：
```python
# 完整的W8A8支持
trt_model = tensorrt_llm.build(
    model,
    quantization='int8',  # Weight + Activation
    calibration_dataset=calib_data
)
```

**PyTorch**：
```python
from torch.quantization import quantize_dynamic, quantize_static

# Weight-Only（动态量化）
model = quantize_dynamic(model, {nn.Linear}, dtype=torch.qint8)

# Weight + Activation（静态量化）
model.qconfig = torch.quantization.get_default_qconfig('fbgemm')
model = quantize_static(model, calibrate_fn, eval_fn)
```

### 10. 决策指南

#### 10.1 选择流程图
```
你的主要目标是什么？
├─ 内存受限 → Weight-Only (INT4/INT8)
├─ 延迟敏感（小batch）→ Weight-Only (INT8)
├─ 吞吐量优先（大batch）→ Weight + Activation (W8A8)
└─ 精度最重要 → Weight-Only (INT8) 或 FP16
```

#### 10.2 快速决策表

| 因素                    | Weight-Only  | Weight + Activation      |
| ----------------------- | ------------ | ------------------------ |
| **内存节省**            | 高（4-8x）   | 高（4x）                 |
| **推理速度（小batch）** | 中（2-2.5x） | 中（2.5-3x）             |
| **推理速度（大batch）** | 中（2x）     | 高（3-4x）               |
| **精度损失**            | 小（<1%）    | 中（1-3%）               |
| **实现难度**            | 低           | 高                       |
| **需要校准**            | 否           | 是                       |
| **硬件要求**            | 低           | 中（需INT8 Tensor Core） |

#### 10.3 最终建议

**默认选择**：
1. **首选**：Weight-Only INT8（简单、精度高、效果好）
2. **进阶**：Weight-Only INT4 with GPTQ/AWQ（极致压缩）
3. **高级**：W8A8 with SmoothQuant（大batch场景）

**记住**：
- 对于大多数LLM推理场景，Weight-Only INT8是最佳起点
- 只有在吞吐量关键且可接受精度损失时，才考虑激活量化
- 使用成熟工具（GPTQ、AWQ、TensorRT-LLM）而非从头实现


---

## 相关笔记
<!-- 自动生成 -->

- [GPTQ、LLM.int8()等方法的原理和优缺点是什么？](notes/熟悉大语言模型推理优化-技术层次/GPTQ、LLM.int8()等方法的原理和优缺点是什么？.md) - 相似度: 33% | 标签: 熟悉大语言模型推理优化-技术层次, 熟悉大语言模型推理优化-技术层次/GPTQ、LLM.int8()等方法的原理和优缺点是什么？.md
- [INT8量化相比FP16能带来多大的性能提升？精度损失有多少？](notes/熟悉大语言模型推理优化-技术层次/INT8量化相比FP16能带来多大的性能提升？精度损失有多少？.md) - 相似度: 33% | 标签: 熟悉大语言模型推理优化-技术层次, 熟悉大语言模型推理优化-技术层次/INT8量化相比FP16能带来多大的性能提升？精度损失有多少？.md
- [AWQ（Activation-aware Weight Quantization）的核心思想是什么？](notes/熟悉大语言模型推理优化-技术层次/AWQ（Activation-aware Weight Quantization）的核心思想是什么？.md) - 相似度: 31% | 标签: 熟悉大语言模型推理优化-技术层次, 熟悉大语言模型推理优化-技术层次/AWQ（Activation-aware Weight Quantization）的核心思想是什么？.md
- [极低比特量化对模型性能的影响有多大？适用于哪些场景？](notes/熟悉大语言模型推理优化-技术层次/极低比特量化对模型性能的影响有多大？适用于哪些场景？.md) - 相似度: 31% | 标签: 熟悉大语言模型推理优化-技术层次, 熟悉大语言模型推理优化-技术层次/极低比特量化对模型性能的影响有多大？适用于哪些场景？.md
- [FP32、FP16、BF16、INT8、INT4之间有什么区别？各自的优缺点是什么？](notes/熟悉大语言模型推理优化-技术层次/FP32、FP16、BF16、INT8、INT4之间有什么区别？各自的优缺点是什么？.md) - 相似度: 31% | 标签: 熟悉大语言模型推理优化-技术层次, 熟悉大语言模型推理优化-技术层次/FP32、FP16、BF16、INT8、INT4之间有什么区别？各自的优缺点是什么？.md

