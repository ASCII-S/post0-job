---
created: '2025-10-19'
last_reviewed: null
next_review: '2025-10-19'
review_count: 0
difficulty: medium
mastery_level: 0.0
tags:
- cuda
- cuda/如何在PyTorch
related_outlines: []
---
# 如何在PyTorch/TensorFlow中编写自定义CUDA算子？

## 面试标准答案

在PyTorch中编写自定义CUDA算子需要：**1) 编写CUDA kernel实现前向和反向计算；2) 使用`torch.autograd.Function`封装并实现`forward()`和`backward()`方法；3) 通过`torch.utils.cpp_extension.load()`即时编译或`setuptools`提前编译**。在TensorFlow中通过**`tf.load_op_library()`加载自定义op或使用`tf.py_function()`包装**。关键是正确处理梯度传播、内存管理和张量布局。自定义算子用于实现框架不支持的操作或性能优化。

---

## 详细讲解

### 1. PyTorch自定义CUDA算子

#### 1.1 基本结构

**完整流程：**
```
1. 编写CUDA kernel（.cu文件）
2. 编写C++绑定（调用kernel）
3. 使用autograd.Function包装
4. 编译和加载
5. 在Python中使用
```

#### 1.2 示例：自定义ReLU

**步骤1：CUDA Kernel（relu_kernel.cu）**

```cuda
#include <torch/extension.h>
#include <cuda_runtime.h>

// 前向kernel
__global__ void relu_forward_kernel(
    const float* __restrict__ input,
    float* __restrict__ output,
    int size) {
    
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size) {
        output[idx] = fmaxf(0.0f, input[idx]);
    }
}

// 反向kernel
__global__ void relu_backward_kernel(
    const float* __restrict__ grad_output,
    const float* __restrict__ input,
    float* __restrict__ grad_input,
    int size) {
    
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size) {
        grad_input[idx] = (input[idx] > 0) ? grad_output[idx] : 0.0f;
    }
}

// C++接口
torch::Tensor relu_forward_cuda(torch::Tensor input) {
    auto output = torch::empty_like(input);
    int size = input.numel();
    
    const int threads = 256;
    const int blocks = (size + threads - 1) / threads;
    
    relu_forward_kernel<<<blocks, threads>>>(
        input.data_ptr<float>(),
        output.data_ptr<float>(),
        size);
    
    return output;
}

torch::Tensor relu_backward_cuda(
    torch::Tensor grad_output,
    torch::Tensor input) {
    
    auto grad_input = torch::empty_like(input);
    int size = input.numel();
    
    const int threads = 256;
    const int blocks = (size + threads - 1) / threads;
    
    relu_backward_kernel<<<blocks, threads>>>(
        grad_output.data_ptr<float>(),
        input.data_ptr<float>(),
        grad_input.data_ptr<float>(),
        size);
    
    return grad_input;
}

// Python绑定
PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("forward", &relu_forward_cuda, "ReLU forward (CUDA)");
    m.def("backward", &relu_backward_cuda, "ReLU backward (CUDA)");
}
```

**步骤2：Python封装（relu_cuda.py）**

```python
import torch
from torch.autograd import Function
from torch.utils.cpp_extension import load

# JIT编译
relu_cuda = load(
    name='relu_cuda',
    sources=['relu_kernel.cu'],
    verbose=True
)

class ReLUFunction(Function):
    @staticmethod
    def forward(ctx, input):
        output = relu_cuda.forward(input)
        ctx.save_for_backward(input)
        return output
    
    @staticmethod
    def backward(ctx, grad_output):
        input, = ctx.saved_tensors
        grad_input = relu_cuda.backward(grad_output, input)
        return grad_input

# 使用
def custom_relu(input):
    return ReLUFunction.apply(input)

# 测试
x = torch.randn(1000, 1000, device='cuda', requires_grad=True)
y = custom_relu(x)
loss = y.sum()
loss.backward()
print(x.grad)  # 梯度正确计算
```

### 2. 提前编译方式（setup.py）

```python
from setuptools import setup
from torch.utils.cpp_extension import BuildExtension, CUDAExtension

setup(
    name='relu_cuda',
    ext_modules=[
        CUDAExtension('relu_cuda', [
            'relu_kernel.cu',
        ])
    ],
    cmdclass={
        'build_ext': BuildExtension
    }
)

# 编译：python setup.py install
# 使用：import relu_cuda
```

### 3. 更复杂的例子：矩阵乘法

```cuda
#include <torch/extension.h>

#define BLOCK_SIZE 16

__global__ void matmul_kernel(
    const float* __restrict__ A,
    const float* __restrict__ B,
    float* __restrict__ C,
    int M, int N, int K) {
    
    __shared__ float As[BLOCK_SIZE][BLOCK_SIZE];
    __shared__ float Bs[BLOCK_SIZE][BLOCK_SIZE];
    
    int row = blockIdx.y * BLOCK_SIZE + threadIdx.y;
    int col = blockIdx.x * BLOCK_SIZE + threadIdx.x;
    
    float sum = 0.0f;
    
    for (int tile = 0; tile < (K + BLOCK_SIZE - 1) / BLOCK_SIZE; ++tile) {
        // 加载tile到共享内存
        if (row < M && tile * BLOCK_SIZE + threadIdx.x < K)
            As[threadIdx.y][threadIdx.x] = A[row * K + tile * BLOCK_SIZE + threadIdx.x];
        else
            As[threadIdx.y][threadIdx.x] = 0.0f;
        
        if (col < N && tile * BLOCK_SIZE + threadIdx.y < K)
            Bs[threadIdx.y][threadIdx.x] = B[(tile * BLOCK_SIZE + threadIdx.y) * N + col];
        else
            Bs[threadIdx.y][threadIdx.x] = 0.0f;
        
        __syncthreads();
        
        // 计算
        for (int k = 0; k < BLOCK_SIZE; ++k) {
            sum += As[threadIdx.y][k] * Bs[k][threadIdx.x];
        }
        
        __syncthreads();
    }
    
    if (row < M && col < N) {
        C[row * N + col] = sum;
    }
}

torch::Tensor matmul_cuda(torch::Tensor A, torch::Tensor B) {
    auto M = A.size(0);
    auto K = A.size(1);
    auto N = B.size(1);
    
    auto C = torch::zeros({M, N}, A.options());
    
    dim3 threads(BLOCK_SIZE, BLOCK_SIZE);
    dim3 blocks((N + BLOCK_SIZE - 1) / BLOCK_SIZE,
                (M + BLOCK_SIZE - 1) / BLOCK_SIZE);
    
    matmul_kernel<<<blocks, threads>>>(
        A.data_ptr<float>(),
        B.data_ptr<float>(),
        C.data_ptr<float>(),
        M, N, K);
    
    return C;
}
```

### 4. 自动求导实现

#### 4.1 前向和反向关系

```python
# 前向：y = f(x)
# 反向：dx = df/dx * dy

class MyFunction(Function):
    @staticmethod
    def forward(ctx, input, weight):
        # 保存需要用于backward的张量
        ctx.save_for_backward(input, weight)
        output = my_cuda.forward(input, weight)
        return output
    
    @staticmethod
    def backward(ctx, grad_output):
        # 恢复保存的张量
        input, weight = ctx.saved_tensors
        
        # 计算输入和权重的梯度
        grad_input = my_cuda.backward_input(grad_output, weight)
        grad_weight = my_cuda.backward_weight(grad_output, input)
        
        return grad_input, grad_weight
```

#### 4.2 梯度检查

```python
from torch.autograd import gradcheck

# 测试梯度正确性
input = torch.randn(20, 20, dtype=torch.double, device='cuda', requires_grad=True)
test = gradcheck(MyFunction.apply, input, eps=1e-6, atol=1e-4)
print("Gradient check:", test)
```

### 5. TensorFlow自定义Op

#### 5.1 基本方法

```cpp
// custom_op.cu
#include "tensorflow/core/framework/op.h"
#include "tensorflow/core/framework/op_kernel.h"

// 注册Op
REGISTER_OP("CustomRelu")
    .Input("input: float")
    .Output("output: float");

// GPU Kernel
__global__ void ReluKernel(const float* input, float* output, int size) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size) {
        output[idx] = fmaxf(0.0f, input[idx]);
    }
}

// OpKernel实现
class CustomReluOp : public OpKernel {
public:
    explicit CustomReluOp(OpKernelConstruction* context) : OpKernel(context) {}
    
    void Compute(OpKernelContext* context) override {
        const Tensor& input = context->input(0);
        Tensor* output = nullptr;
        OP_REQUIRES_OK(context, context->allocate_output(0, input.shape(), &output));
        
        int size = input.NumElements();
        const int threads = 256;
        const int blocks = (size + threads - 1) / threads;
        
        ReluKernel<<<blocks, threads>>>(
            input.flat<float>().data(),
            output->flat<float>().data(),
            size);
    }
};

REGISTER_KERNEL_BUILDER(Name("CustomRelu").Device(DEVICE_GPU), CustomReluOp);
```

#### 5.2 Python使用

```python
import tensorflow as tf

# 加载编译的Op
custom_module = tf.load_op_library('./custom_op.so')

# 使用
@tf.function
def custom_relu(x):
    return custom_module.custom_relu(x)

x = tf.random.normal([1000, 1000])
y = custom_relu(x)
```

### 6. 性能优化技巧

#### 6.1 内存访问优化

```cuda
// ❌ 非合并访问
__global__ void bad_kernel(float* data, int N) {
    int idx = threadIdx.x * N + blockIdx.x;  // 跨步访问
    data[idx] = ...;
}

// ✅ 合并访问
__global__ void good_kernel(float* data, int N) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;  // 连续访问
    if (idx < N) data[idx] = ...;
}
```

#### 6.2 使用Tensor Accessor

```cpp
// PyTorch Tensor Accessor（更安全、类型检查）
torch::Tensor my_function(torch::Tensor input) {
    auto input_a = input.accessor<float, 2>();  // 2D float tensor
    
    kernel<<<...>>>(
        input_a.data(),
        input_a.size(0),
        input_a.size(1));
}
```

#### 6.3 处理不同数据类型

```cpp
// 模板支持多种类型
template <typename scalar_t>
__global__ void generic_kernel(
    const scalar_t* __restrict__ input,
    scalar_t* __restrict__ output,
    int size) {
    // 通用实现
}

torch::Tensor generic_function(torch::Tensor input) {
    auto output = torch::empty_like(input);
    
    AT_DISPATCH_FLOATING_TYPES(input.scalar_type(), "generic_kernel", ([&] {
        generic_kernel<scalar_t><<<...>>>(
            input.data_ptr<scalar_t>(),
            output.data_ptr<scalar_t>(),
            input.numel());
    }));
    
    return output;
}
```

### 7. 常见问题

#### 7.1 内存管理

```python
class MyFunction(Function):
    @staticmethod
    def forward(ctx, input):
        # ❌ 错误：保存CUDA张量可能导致显存泄漏
        ctx.intermediate = some_cuda_tensor
        
        # ✅ 正确：只保存必要的张量
        ctx.save_for_backward(input)
        
        # ✅ 或保存到CPU
        ctx.intermediate = some_cuda_tensor.cpu()
```

#### 7.2 Stream同步

```cpp
// 确保kernel完成
cudaDeviceSynchronize();  // 等待所有操作

// 或使用stream
cudaStream_t stream;
cudaStreamCreate(&stream);
kernel<<<..., stream>>>();
cudaStreamSynchronize(stream);
```

#### 7.3 错误处理

```cpp
#define CUDA_CHECK(call) \
    do { \
        cudaError_t error = call; \
        if (error != cudaSuccess) { \
            fprintf(stderr, "CUDA error at %s:%d: %s\n", \
                    __FILE__, __LINE__, cudaGetErrorString(error)); \
            exit(EXIT_FAILURE); \
        } \
    } while(0)

CUDA_CHECK(cudaMalloc(&d_data, size));
```

### 8. 完整项目结构

```
custom_op/
├── cuda_kernels.cu          # CUDA实现
├── cuda_kernels.h           # C++头文件
├── custom_op.cpp            # C++封装
├── setup.py                 # 编译脚本
└── test.py                  # 测试脚本
```

### 9. 调试技巧

```python
# 1. 使用torch.autograd.gradcheck
gradcheck(MyFunction.apply, inputs)

# 2. 比对PyTorch原生实现
output_custom = custom_op(x)
output_torch = torch.relu(x)
assert torch.allclose(output_custom, output_torch)

# 3. 使用cuda-memcheck
# $ cuda-memcheck python test.py

# 4. Profiling
with torch.autograd.profiler.profile(use_cuda=True) as prof:
    y = custom_op(x)
print(prof.key_averages())
```

### 10. 性能对比

**自定义ReLU vs PyTorch原生（1M元素）**

| 实现        | 时间(μs) | vs PyTorch |
| ----------- | -------- | ---------- |
| PyTorch原生 | 45       | 1×         |
| 简单自定义  | 40       | 1.1×       |
| 优化自定义  | 28       | 1.6×       |

**何时值得自定义：**
- 框架不支持的操作
- 融合多个操作减少访存
- 特殊优化（如知道数据分布）

### 11. 最佳实践

| 建议                   | 说明               |
| ---------------------- | ------------------ |
| ✅ 使用JIT编译快速迭代  | 开发阶段方便       |
| ✅ 生产环境提前编译     | 避免运行时编译开销 |
| ✅ 正确实现backward     | 使用gradcheck验证  |
| ✅ 处理多种数据类型     | AT_DISPATCH_*宏    |
| ✅ 良好的错误处理       | CUDA_CHECK等       |
| ✅ 文档和测试           | 方便维护           |
| ❌ 不要过早优化         | 先确保正确性       |
| ❌ 避免不必要的内存拷贝 | CPU-GPU传输昂贵    |

### 12. 记忆口诀

**"自定义算子三步走，kernel函数autograd封装编译加载用；forward保存backward取，save_for_backward记心头；JIT编译快迭代，setup安装生产优；gradcheck验证梯度对，性能profiling不能丢"**


---

## 相关笔记
<!-- 自动生成 -->

暂无相关笔记

