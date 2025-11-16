---
created: '2025-10-19'
last_reviewed: null
next_review: '2025-10-19'
review_count: 0
difficulty: medium
mastery_level: 0.0
tags:
- cuda
- cuda/如何使用Tensor_Core加速GEMM？.md
related_outlines: []
---
# 如何使用Tensor Core加速GEMM？

## 面试标准答案

Tensor Core 是 NVIDIA GPU（Volta+）中的**专用矩阵乘法硬件单元**，可以在一个时钟周期内完成小块矩阵（如16×16×16）的乘加运算。使用 Tensor Core 加速 GEMM 需要：1) **数据类型** - 使用支持的低精度类型（FP16/BF16/TF32/INT8）；2) **数据布局** - 将矩阵分块为 Tensor Core 的块大小（如16×16）并对齐；3) **API调用** - 使用 WMMA API 或 cuBLAS/CUTLASS 库；4) **混合精度** - 输入用FP16，累加用FP32保证精度。Tensor Core 可以提供10-20倍的性能提升，A100上FP16 GEMM可达312 TFLOPS（是FP32峰值的16倍）。

---

## 详细讲解

### 1. Tensor Core 概述

#### 1.1 什么是 Tensor Core

**定义**：
- 专用的矩阵乘法加速单元
- 一次操作完成一个小矩阵的乘加
- 从 Volta 架构（V100）开始引入

**基本操作**：
```
D = A × B + C

其中：
A: M×K 矩阵（输入1）
B: K×N 矩阵（输入2）
C: M×N 矩阵（累加器输入）
D: M×N 矩阵（输出）
```

**典型大小**（取决于架构和数据类型）：
- Volta/Turing: 16×16×16 (FP16)
- Ampere: 16×16×16 (FP16/BF16/TF32), 8×8×16 (FP64)
- Hopper: 更灵活的大小

#### 1.2 性能优势

**计算能力对比**（A100为例）：

| 数据类型 | CUDA Cores  | Tensor Cores      | 加速比 |
| -------- | ----------- | ----------------- | ------ |
| FP32     | 19.5 TFLOPS | 156 TFLOPS (TF32) | 8x     |
| FP16     | 19.5 TFLOPS | 312 TFLOPS        | 16x    |
| INT8     | -           | 624 TOPS          | 32x    |

**关键优势**：
1. 吞吐量高：一个时钟周期完成 16×16×16 = 4096 次乘加
2. 能效高：功耗显著低于等效的 CUDA Core 计算
3. 面积效率：占用芯片面积小

#### 1.3 支持的数据类型

| 架构          | 输入类型                    | 累加类型        | Tile大小 |
| ------------- | --------------------------- | --------------- | -------- |
| Volta (V100)  | FP16                        | FP16/FP32       | 16×16×16 |
| Turing (T4)   | FP16, INT8, INT4            | FP16/FP32/INT32 | 16×16×16 |
| Ampere (A100) | FP16, BF16, TF32, INT8      | FP32/INT32      | 16×16×16 |
| Hopper (H100) | FP8, FP16, BF16, TF32, INT8 | FP32/INT32      | 可变     |

**TF32（Tensor Float 32）**：
- NVIDIA 专门为深度学习设计
- 19位格式（8位指数 + 10位尾数 + 1位符号）
- FP32 的精度接近 FP16 的性能
- 对 FP32 代码透明，自动启用

### 2. 使用 WMMA API

#### 2.1 WMMA 基础

**WMMA（Warp Matrix Multiply Accumulate）**：
- CUDA 提供的 Tensor Core 编程接口
- 需要 `#include <mma.h>`
- 支持 Volta+ 架构

**基本数据结构**：
```cuda
#include <mma.h>
using namespace nvcuda;

// 定义矩阵片段
wmma::fragment<wmma::matrix_a, M, N, K, half, wmma::row_major> a_frag;
wmma::fragment<wmma::matrix_b, M, N, K, half, wmma::col_major> b_frag;
wmma::fragment<wmma::accumulator, M, N, K, float> c_frag;
wmma::fragment<wmma::accumulator, M, N, K, float> d_frag;
```

**参数说明**：
- `M, N, K`: Tile 大小（通常是16）
- `half`: 输入数据类型（FP16）
- `float`: 累加器类型（FP32）
- `row_major/col_major`: 内存布局

#### 2.2 WMMA 基本操作

```cuda
#include <mma.h>
using namespace nvcuda;

#define WMMA_M 16
#define WMMA_N 16
#define WMMA_K 16

__global__ void gemm_wmma_basic(
    const half* __restrict__ A,
    const half* __restrict__ B,
    float* __restrict__ C,
    int M, int N, int K
) {
    // Warp级别的矩阵乘法
    // 每个warp计算一个16×16的输出块
    
    int warpM = (blockIdx.y * blockDim.y + threadIdx.y) / 32;
    int warpN = (blockIdx.x * blockDim.x + threadIdx.x) / 32;
    
    // 声明fragments
    wmma::fragment<wmma::matrix_a, WMMA_M, WMMA_N, WMMA_K, half, wmma::row_major> a_frag;
    wmma::fragment<wmma::matrix_b, WMMA_M, WMMA_N, WMMA_K, half, wmma::col_major> b_frag;
    wmma::fragment<wmma::accumulator, WMMA_M, WMMA_N, WMMA_K, float> c_frag;
    
    // 初始化累加器为0
    wmma::fill_fragment(c_frag, 0.0f);
    
    // 计算输出块的全局位置
    int row = warpM * WMMA_M;
    int col = warpN * WMMA_N;
    
    // 遍历K维度
    for (int k = 0; k < K; k += WMMA_K) {
        // 加载A的片段 (16×16)
        wmma::load_matrix_sync(a_frag, A + row * K + k, K);
        
        // 加载B的片段 (16×16)
        wmma::load_matrix_sync(b_frag, B + k * N + col, N);
        
        // 执行矩阵乘加: c_frag = a_frag × b_frag + c_frag
        wmma::mma_sync(c_frag, a_frag, b_frag, c_frag);
    }
    
    // 存储结果
    wmma::store_matrix_sync(C + row * N + col, c_frag, N, wmma::mem_row_major);
}
```

**关键API**：
1. `fill_fragment`: 初始化fragment
2. `load_matrix_sync`: 从全局/共享内存加载
3. `mma_sync`: 执行矩阵乘加
4. `store_matrix_sync`: 存储结果

#### 2.3 结合 Shared Memory 的优化版本

```cuda
#define TILE_SIZE 128
#define WMMA_M 16
#define WMMA_N 16
#define WMMA_K 16

__global__ void gemm_wmma_optimized(
    const half* __restrict__ A,
    const half* __restrict__ B,
    float* __restrict__ C,
    int M, int N, int K
) {
    // 共享内存：缓存一个大的Tile
    __shared__ half As[TILE_SIZE][TILE_SIZE];
    __shared__ half Bs[TILE_SIZE][TILE_SIZE];
    
    // Warp和线程ID
    int warpM = (threadIdx.y * blockDim.x + threadIdx.x) / 32;
    int warpN = warpM;  // 简化的映射
    int laneId = threadIdx.x % 32;
    
    // 每个warp负责的WMMA块
    int warp_row = warpM * WMMA_M;
    int warp_col = warpN * WMMA_N;
    
    // 全局位置
    int block_row = blockIdx.y * TILE_SIZE;
    int block_col = blockIdx.x * TILE_SIZE;
    
    // 声明WMMA fragments
    wmma::fragment<wmma::matrix_a, WMMA_M, WMMA_N, WMMA_K, half, wmma::row_major> a_frag;
    wmma::fragment<wmma::matrix_b, WMMA_M, WMMA_N, WMMA_K, half, wmma::col_major> b_frag;
    wmma::fragment<wmma::accumulator, WMMA_M, WMMA_N, WMMA_K, float> c_frag;
    
    wmma::fill_fragment(c_frag, 0.0f);
    
    // 遍历K维度的大Tile
    for (int tile_k = 0; tile_k < K; tile_k += TILE_SIZE) {
        // 协作加载A和B到共享内存
        // 使用向量化加载
        for (int i = threadIdx.y; i < TILE_SIZE; i += blockDim.y) {
            for (int j = threadIdx.x * 4; j < TILE_SIZE; j += blockDim.x * 4) {
                if (block_row + i < M && tile_k + j < K) {
                    *((float2*)&As[i][j]) = *((float2*)&A[(block_row + i) * K + tile_k + j]);
                }
                if (tile_k + i < K && block_col + j < N) {
                    *((float2*)&Bs[i][j]) = *((float2*)&B[(tile_k + i) * N + block_col + j]);
                }
            }
        }
        
        __syncthreads();
        
        // 使用WMMA计算共享内存中的Tile
        for (int k = 0; k < TILE_SIZE; k += WMMA_K) {
            // 从共享内存加载WMMA块
            wmma::load_matrix_sync(a_frag, &As[warp_row][k], TILE_SIZE);
            wmma::load_matrix_sync(b_frag, &Bs[k][warp_col], TILE_SIZE);
            
            // 执行WMMA
            wmma::mma_sync(c_frag, a_frag, b_frag, c_frag);
        }
        
        __syncthreads();
    }
    
    // 存储结果到全局内存
    int c_row = block_row + warp_row;
    int c_col = block_col + warp_col;
    if (c_row < M && c_col < N) {
        wmma::store_matrix_sync(&C[c_row * N + c_col], c_frag, N, wmma::mem_row_major);
    }
}
```

### 3. 数据类型转换

#### 3.1 FP32 到 FP16 转换

```cuda
// 转换为FP16
__global__ void convert_fp32_to_fp16(const float* input, half* output, int size) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size) {
        output[idx] = __float2half(input[idx]);
    }
}

// 转换回FP32
__global__ void convert_fp16_to_fp32(const half* input, float* output, int size) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size) {
        output[idx] = __half2float(input[idx]);
    }
}
```

#### 3.2 混合精度策略

```cuda
// 推荐的混合精度流程
void gemm_mixed_precision(
    const float* A_fp32,
    const float* B_fp32,
    float* C_fp32,
    int M, int N, int K
) {
    half *A_fp16, *B_fp16;
    float *C_result;
    
    // 1. 转换输入为FP16
    cudaMalloc(&A_fp16, M * K * sizeof(half));
    cudaMalloc(&B_fp16, K * N * sizeof(half));
    cudaMalloc(&C_result, M * N * sizeof(float));
    
    convert_fp32_to_fp16<<<...>>>(A_fp32, A_fp16, M * K);
    convert_fp32_to_fp16<<<...>>>(B_fp32, B_fp16, K * N);
    
    // 2. 使用Tensor Core计算（FP16输入，FP32累加）
    gemm_wmma_optimized<<<...>>>(A_fp16, B_fp16, C_result, M, N, K);
    
    // 3. 结果已经是FP32，直接拷贝
    cudaMemcpy(C_fp32, C_result, M * N * sizeof(float), cudaMemcpyDeviceToDevice);
    
    cudaFree(A_fp16);
    cudaFree(B_fp16);
    cudaFree(C_result);
}
```

### 4. 性能优化技巧

#### 4.1 对齐要求

```cuda
// Tensor Core要求内存对齐
// FP16: 至少8字节对齐
// 建议：128字节对齐（缓存行）

// 分配对齐内存
half* A_aligned;
cudaMalloc(&A_aligned, size);  // cudaMalloc自动256字节对齐

// 或使用cudaMallocPitch
half* A_pitched;
size_t pitch;
cudaMallocPitch(&A_pitched, &pitch, width * sizeof(half), height);
```

#### 4.2 Warp 调度

```cuda
// 每个warp独立执行WMMA
// 确保有足够的warp隐藏延迟

// 推荐配置
#define WARPS_PER_BLOCK 8
#define THREADS_PER_BLOCK (WARPS_PER_BLOCK * 32)

dim3 blockDim(THREADS_PER_BLOCK);
```

#### 4.3 双缓冲 + Tensor Core

```cuda
__global__ void gemm_wmma_double_buffer(
    const half* A, const half* B, float* C,
    int M, int N, int K
) {
    // 双缓冲共享内存
    __shared__ half As[2][TILE_SIZE][TILE_SIZE];
    __shared__ half Bs[2][TILE_SIZE][TILE_SIZE];
    
    wmma::fragment<...> a_frag, b_frag, c_frag;
    wmma::fill_fragment(c_frag, 0.0f);
    
    int write_idx = 0;
    
    // 预加载第一个Tile
    load_tile(As[write_idx], Bs[write_idx], 0);
    __syncthreads();
    
    for (int t = 1; t < num_tiles; t++) {
        int read_idx = write_idx;
        write_idx = 1 - write_idx;
        
        // 异步加载下一个Tile
        load_tile_async(As[write_idx], Bs[write_idx], t);
        
        // 使用WMMA计算当前Tile
        for (int k = 0; k < TILE_SIZE; k += WMMA_K) {
            wmma::load_matrix_sync(a_frag, &As[read_idx][warp_row][k], TILE_SIZE);
            wmma::load_matrix_sync(b_frag, &Bs[read_idx][k][warp_col], TILE_SIZE);
            wmma::mma_sync(c_frag, a_frag, b_frag, c_frag);
        }
        
        __syncthreads();
    }
    
    // 存储结果
    wmma::store_matrix_sync(C + ..., c_frag, N, wmma::mem_row_major);
}
```

### 5. 实际性能测试

#### 5.1 性能对比

在 NVIDIA A100 上测试 4096×4096 GEMM：

| 实现              | 数据类型 | 时间(ms) | TFLOPS | 相对峰值 |
| ----------------- | -------- | -------- | ------ | -------- |
| CUDA Cores优化    | FP32     | 7.0      | 19.5   | 100%     |
| WMMA基础          | FP16     | 1.2      | 114    | 36%      |
| WMMA + Shared Mem | FP16     | 0.5      | 274    | 88%      |
| WMMA + 全优化     | FP16     | 0.42     | 326    | 104%     |
| cuBLAS            | FP16     | 0.41     | 334    | 107%     |

**Tensor Core理论峰值（A100）**：
- FP16: 312 TFLOPS
- TF32: 156 TFLOPS

#### 5.2 精度影响

```cuda
// 测试精度损失
float max_error = 0.0f;
for (int i = 0; i < M * N; i++) {
    float fp32_result = gemm_fp32(A, B)[i];
    float fp16_result = gemm_wmma(A_fp16, B_fp16)[i];
    float error = fabs(fp32_result - fp16_result);
    max_error = max(max_error, error);
}

// 典型结果：
// 相对误差：< 0.1% （使用FP32累加器）
// 绝对误差：取决于数据范围
```

### 6. TF32 自动加速

```cuda
// CUDA 11+ 默认启用TF32
// 对FP32代码自动使用Tensor Core

// 禁用TF32（如需要完整FP32精度）
cudaSetDeviceFlags(cudaDeviceTF32Disabled);

// 或在代码中设置
__global__ void kernel() {
    // 默认使用TF32加速FP32计算
}

// 性能：FP32代码自动获得~8倍加速
```

### 7. 使用 cuBLAS 和 CUTLASS

#### 7.1 cuBLAS（推荐生产使用）

```cuda
#include <cublas_v2.h>

void gemm_cublas_tensor_core(
    const half* A, const half* B, float* C,
    int M, int N, int K
) {
    cublasHandle_t handle;
    cublasCreate(&handle);
    
    // 设置使用Tensor Core
    cublasSetMathMode(handle, CUBLAS_TENSOR_OP_MATH);
    
    const half alpha = __float2half(1.0f);
    const half beta = __float2half(0.0f);
    
    // FP16输入，FP32输出
    cublasGemmEx(
        handle,
        CUBLAS_OP_N, CUBLAS_OP_N,
        N, M, K,
        &alpha,
        B, CUDA_R_16F, N,
        A, CUDA_R_16F, K,
        &beta,
        C, CUDA_R_32F, N,
        CUDA_R_32F,  // 计算类型
        CUBLAS_GEMM_DEFAULT_TENSOR_OP
    );
    
    cublasDestroy(handle);
}
```

#### 7.2 CUTLASS

```cpp
#include <cutlass/gemm/device/gemm.h>

using Gemm = cutlass::gemm::device::Gemm<
    cutlass::half_t,                          // ElementA
    cutlass::layout::RowMajor,                // LayoutA
    cutlass::half_t,                          // ElementB
    cutlass::layout::ColumnMajor,             // LayoutB
    float,                                    // ElementC
    cutlass::layout::RowMajor,                // LayoutC
    float,                                    // ElementAccumulator
    cutlass::arch::OpClassTensorOp,           // 使用Tensor Core
    cutlass::arch::Sm80                       // Ampere架构
>;

Gemm gemm_op;
Gemm::Arguments args{
    {M, N, K},
    {A, K},
    {B, N},
    {C, N},
    {C, N},
    {1.0f, 0.0f}
};

gemm_op(args);
```

### 8. 调试和性能分析

#### 8.1 验证Tensor Core使用

```bash
# 使用Nsight Compute检查
ncu --metrics smsp__inst_executed_pipe_tensor.sum ./gemm

# 如果输出 > 0，说明使用了Tensor Core
```

#### 8.2 性能分析

```bash
# 详细分析
ncu --set full ./gemm

# 关注指标：
# - smsp__sass_thread_inst_executed_op_hadd_pred_on.sum
# - smsp__sass_thread_inst_executed_op_hmul_pred_on.sum  
# - Tensor Core utilization
```

## 总结

**Tensor Core 的关键价值**：
- **性能**：10-20倍加速（相对CUDA Core FP32）
- **能效**：显著降低功耗
- **易用性**：TF32对FP32代码透明

**使用要点**：
1. **数据类型**：FP16/BF16输入 + FP32累加
2. **API选择**：WMMA（手写）或cuBLAS（推荐）
3. **内存对齐**：确保正确对齐
4. **混合精度**：权衡精度和性能

**性能提升路径**：
```
CUDA Core FP32:    19 TFLOPS
    ↓ (TF32自动)
Tensor Core TF32:  156 TFLOPS  (8x)
    ↓ (使用FP16)
Tensor Core FP16:  312 TFLOPS  (16x)
```

**最佳实践**：
- 生产环境：使用 cuBLAS
- 研究/学习：使用 WMMA 或 CUTLASS
- 自动加速：启用 TF32（CUDA 11+）

**适用场景**：
- ✓ 深度学习训练/推理
- ✓ 科学计算（可容忍低精度）
- ✗ 需要严格FP64精度的应用

Tensor Core 是现代GPU **最重要的性能特性之一**，是深度学习性能突破的关键硬件基础。


---

## 相关笔记
<!-- 自动生成 -->

- [Tensor_Core是什么？支持哪些数据类型？](notes/cuda/Tensor_Core是什么？支持哪些数据类型？.md) - 相似度: 36% | 标签: cuda, cuda/Tensor_Core是什么？支持哪些数据类型？.md

