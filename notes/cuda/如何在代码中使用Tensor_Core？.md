---
created: '2025-10-19'
last_reviewed: null
next_review: '2025-10-19'
review_count: 0
difficulty: medium
mastery_level: 0.0
tags:
- cuda
- cuda/如何在代码中使用Tensor_Core？.md
related_outlines: []
---
# 如何在代码中使用Tensor Core？

## 面试标准答案

使用Tensor Core主要有三种方式：**1) 使用PyTorch/TensorFlow的自动混合精度（AMP）**，通过`torch.cuda.amp.autocast()`自动转换；**2) 手动使用FP16/BF16数据类型**并调用cuBLAS/cuDNN；**3) 直接使用WMMA（Warp Matrix Multiply-Accumulate）API**编写CUDA kernel。最常用的是第一种，简单高效。关键是确保矩阵维度对齐（8或16的倍数）且使用支持的数据类型。

---

## 详细讲解

### 1. 方法一：PyTorch自动混合精度（推荐）

#### 1.1 基础用法

```python
import torch
from torch.cuda.amp import autocast, GradScaler

model = MyModel().cuda()
optimizer = torch.optim.Adam(model.parameters())
scaler = GradScaler()  # 用于loss scaling

for data, target in dataloader:
    optimizer.zero_grad()
    
    # 自动混合精度上下文
    with autocast():
        output = model(data)
        loss = criterion(output, target)
    
    # 梯度缩放（避免FP16下溢）
    scaler.scale(loss).backward()
    scaler.step(optimizer)
    scaler.update()
```

**关键点：**
- `autocast()`自动选择FP16/FP32
- `GradScaler`处理梯度缩放
- 兼容所有PyTorch模型

#### 1.2 完整训练示例

```python
from torch.cuda.amp import autocast, GradScaler

def train_with_amp(model, train_loader, epochs=10):
    scaler = GradScaler()
    
    for epoch in range(epochs):
        for batch_idx, (data, target) in enumerate(train_loader):
            data, target = data.cuda(), target.cuda()
            
            optimizer.zero_grad()
            
            with autocast():
                output = model(data)
                loss = loss_fn(output, target)
            
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            
            if batch_idx % 100 == 0:
                print(f"Loss: {loss.item():.4f}")
```

**性能提升：** 通常2-3倍训练速度

#### 1.3 手动控制数据类型

```python
# 指定使用BF16（A100上推荐）
with autocast(dtype=torch.bfloat16):
    output = model(input)

# 或全局设置
torch.set_autocast_gpu_dtype(torch.bfloat16)
```

### 2. 方法二：TensorFlow/Keras混合精度

```python
from tensorflow.keras import mixed_precision

# 全局启用混合精度
mixed_precision.set_global_policy('mixed_float16')

# 或使用BF16
mixed_precision.set_global_policy('mixed_bfloat16')

# 构建模型（自动使用FP16）
model = tf.keras.Sequential([
    tf.keras.layers.Dense(128, activation='relu'),
    tf.keras.layers.Dense(10, activation='softmax', dtype='float32')  # 输出层用FP32
])

# 编译和训练
model.compile(
    optimizer=tf.keras.optimizers.Adam(),
    loss=tf.keras.losses.CategoricalCrossentropy(),
    metrics=['accuracy']
)

model.fit(train_data, epochs=10)  # 自动使用Tensor Core
```

### 3. 方法三：手动FP16转换

```python
# 将模型转为FP16
model = model.half()

# 数据也转为FP16
data = data.half()

# 前向传播
output = model(data)  # 自动使用Tensor Core

# 注意：需要手动处理loss scaling
loss = loss * scale
loss.backward()
for param in model.parameters():
    param.grad /= scale
optimizer.step()
```

**不推荐原因：** 需要手动管理缩放，容易出错

### 4. 方法四：cuBLAS直接调用

```c
#include <cublas_v2.h>

// 使用Tensor Core的GEMM
cublasHandle_t handle;
cublasCreate(&handle);

// 设置使用Tensor Core
cublasSetMathMode(handle, CUBLAS_TENSOR_OP_MATH);

// FP16矩阵乘法（自动使用Tensor Core）
cublasGemmEx(handle,
             CUBLAS_OP_N, CUBLAS_OP_N,
             m, n, k,
             &alpha,
             A, CUDA_R_16F, lda,  // FP16输入
             B, CUDA_R_16F, ldb,
             &beta,
             C, CUDA_R_16F, ldc,
             CUDA_R_32F,          // FP32计算
             CUBLAS_GEMM_DEFAULT_TENSOR_OP);
```

### 5. 方法五：WMMA API（高级）

```cuda
#include <mma.h>
using namespace nvcuda;

__global__ void wmma_kernel(half *a, half *b, float *c, int M, int N, int K) {
    // 声明WMMA片段
    wmma::fragment<wmma::matrix_a, 16, 16, 16, half, wmma::row_major> a_frag;
    wmma::fragment<wmma::matrix_b, 16, 16, 16, half, wmma::col_major> b_frag;
    wmma::fragment<wmma::accumulator, 16, 16, 16, float> c_frag;
    
    // 初始化累加器
    wmma::fill_fragment(c_frag, 0.0f);
    
    // 加载数据
    wmma::load_matrix_sync(a_frag, a, K);
    wmma::load_matrix_sync(b_frag, b, K);
    
    // 矩阵乘加（使用Tensor Core）
    wmma::mma_sync(c_frag, a_frag, b_frag, c_frag);
    
    // 存储结果
    wmma::store_matrix_sync(c, c_frag, N, wmma::mem_row_major);
}
```

**使用场景：** 需要极致性能优化的自定义kernel

### 6. 确保Tensor Core被使用

#### 6.1 检查方法

```python
# PyTorch中检查
import torch

# 方法1：查看Profiler
with torch.autograd.profiler.profile(use_cuda=True) as prof:
    with autocast():
        output = model(input)

print(prof.key_averages().table(sort_by="cuda_time_total"))
# 查看是否有 "volta_h884gemm" 或 "ampere_fp16" 相关kernel

# 方法2：检查cuBLAS设置
print(torch.backends.cuda.matmul.allow_tf32)  # TF32
print(torch.backends.cudnn.allow_tf32)        # cuDNN TF32
```

```bash
# 使用Nsight Compute分析
ncu --metrics sm__inst_executed_pipe_tensor.avg.pct_of_peak_sustained_active \
    python train.py

# 输出显示Tensor Core利用率
```

#### 6.2 常见问题排查

| 问题                  | 原因             | 解决方案               |
| --------------------- | ---------------- | ---------------------- |
| 性能没有提升          | 维度未对齐       | padding到8/16的倍数    |
| 精度损失大            | 数据范围超出FP16 | 使用BF16或loss scaling |
| 训练不收敛            | 梯度下溢         | 启用GradScaler         |
| 显示未使用Tensor Core | cuBLAS版本过旧   | 更新CUDA Toolkit       |

### 7. 维度对齐优化

```python
# 检查并padding维度
def make_divisible(v, divisor=8):
    return int(np.ceil(v / divisor) * divisor)

# 修改模型确保维度对齐
class AlignedLinear(nn.Module):
    def __init__(self, in_features, out_features):
        super().__init__()
        # 对齐到8的倍数
        aligned_in = make_divisible(in_features, 8)
        aligned_out = make_divisible(out_features, 8)
        self.linear = nn.Linear(aligned_in, aligned_out)
        
    def forward(self, x):
        # padding输入
        if x.shape[-1] % 8 != 0:
            pad_size = make_divisible(x.shape[-1], 8) - x.shape[-1]
            x = F.pad(x, (0, pad_size))
        return self.linear(x)
```

### 8. 不同场景的最佳实践

#### 8.1 训练场景

```python
# A100/H100: 使用BF16
with autocast(dtype=torch.bfloat16):
    output = model(input)

# V100: 使用FP16 + GradScaler
scaler = GradScaler()
with autocast(dtype=torch.float16):
    output = model(input)
scaler.scale(loss).backward()
```

#### 8.2 推理场景

```python
# 转换为FP16
model = model.half()
model.eval()

with torch.no_grad(), autocast():
    output = model(input.half())
```

#### 8.3 分布式训练

```python
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.cuda.amp import autocast, GradScaler

model = DDP(model.cuda(), device_ids=[local_rank])
scaler = GradScaler()

for data, target in train_loader:
    with autocast():
        output = model(data)
        loss = criterion(output, target)
    
    scaler.scale(loss).backward()
    scaler.step(optimizer)
    scaler.update()
```

### 9. 性能对比

**ResNet-50训练（V100，batch=256）：**

| 方法                  | 吞吐量     | 相对FP32 |
| --------------------- | ---------- | -------- |
| FP32（无Tensor Core） | 400 img/s  | 1×       |
| 手动FP16              | 950 img/s  | 2.4×     |
| PyTorch AMP           | 1200 img/s | 3×       |

**BERT-Large微调（A100）：**

| 配置         | 训练时间 | 加速比 |
| ------------ | -------- | ------ |
| FP32         | 8小时    | 1×     |
| TF32（自动） | 5.5小时  | 1.45×  |
| AMP (BF16)   | 3.2小时  | 2.5×   |

### 10. 快速启用清单

```
□ 安装要求
  □ PyTorch 1.6+ 或 TensorFlow 2.4+
  □ CUDA 11.0+
  □ 支持的GPU (Volta/Turing/Ampere/Hopper)

□ 代码修改（PyTorch）
  □ 导入 autocast, GradScaler
  □ 用 autocast() 包裹前向传播
  □ 用 scaler 处理梯度
  □ 检查维度对齐

□ 验证
  □ 运行profiler确认Tensor Core使用
  □ 测试精度损失在可接受范围
  □ 确认性能提升
```

### 11. 最佳实践总结

| 建议                     | 说明                           |
| ------------------------ | ------------------------------ |
| ✅ 优先使用框架AMP        | 简单、安全、高效               |
| ✅ A100用TF32作为baseline | 无需改代码，自动1.5×加速       |
| ✅ 训练用BF16，推理用FP16 | 平衡稳定性和性能               |
| ✅ 确保维度对齐           | padding到8/16倍数              |
| ✅ 使用GradScaler         | 防止FP16梯度下溢               |
| ❌ 避免手动FP16转换       | 容易出错，用AMP更好            |
| ❌ 不要在所有操作上用FP16 | 某些操作需要FP32（如loss计算） |

### 12. 记忆口诀

**"AMP自动最简单，autocast包一圈；GradScaler防下溢，训练稳定不翻船；维度对齐八倍数，Tensor Core才能用；BF16稳定TF32快，混合精度是首选"**


---

## 相关笔记
<!-- 自动生成 -->

暂无相关笔记

