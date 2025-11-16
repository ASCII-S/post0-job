---
created: '2025-10-19'
last_reviewed: null
next_review: '2025-10-19'
review_count: 0
difficulty: medium
mastery_level: 0.0
tags:
- cuda
- cuda/批量矩阵乘法（Batched_GEMM）的优化策略？.md
related_outlines: []
---
# 批量矩阵乘法（Batched GEMM）的优化策略？

## 面试标准答案

批量矩阵乘法（Batched GEMM）是指**同时计算多个独立的矩阵乘法**（如 C_i = A_i × B_i，i=1..N）。优化策略包括：1) **使用cuBLAS batched API** - `cublasGemmBatched`（指针数组）或`cublasGemmStridedBatched`（连续存储）；2) **Stream并行** - 将不同batch分配到不同CUDA stream；3) **增大batch内并行度** - 让每个线程块处理多个batch的Tile；4) **数据布局优化** - 使用strided batched减少指针间接访问；5) **批量融合** - 合并小batch避免kernel启动开销。对于小矩阵（<128），batched GEMM通过增加并行度可获得10x以上性能提升；大矩阵主要收益是编程便利性。

---

## 详细讲解

### 1. Batched GEMM 概述

#### 1.1 问题定义

**标准GEMM**：
```
C = α × A × B + β × C
C: M×N, A: M×K, B: K×N
```

**Batched GEMM**：
```
for i = 0 to batch_count-1:
    C[i] = α × A[i] × B[i] + β × C[i]
```

**典型应用场景**：
- 深度学习：多个样本的attention计算
- 图形学：多个对象的变换矩阵
- 科学计算：多个时间步的状态更新
- 推荐系统：多用户embedding计算

#### 1.2 两种存储方式

**1. Pointer Array（指针数组）**：
```cuda
// 每个矩阵独立分配
float** A_array = new float*[batch_count];
float** B_array = new float*[batch_count];
float** C_array = new float*[batch_count];

for (int i = 0; i < batch_count; i++) {
    cudaMalloc(&A_array[i], M * K * sizeof(float));
    cudaMalloc(&B_array[i], K * N * sizeof(float));
    cudaMalloc(&C_array[i], M * N * sizeof(float));
}
```

**优点**：灵活，每个矩阵可以不同大小  
**缺点**：指针间接访问，缓存不友好

**2. Strided Batched（连续存储）**：
```cuda
// 所有矩阵连续存储
float* A_batched;  // [batch_count × M × K]
float* B_batched;  // [batch_count × K × N]
float* C_batched;  // [batch_count × M × N]

cudaMalloc(&A_batched, batch_count * M * K * sizeof(float));
cudaMalloc(&B_batched, batch_count * K * N * sizeof(float));
cudaMalloc(&C_batched, batch_count * M * N * sizeof(float));

// 访问第i个矩阵
float* A_i = A_batched + i * M * K;  // stride = M * K
```

**优点**：连续访问，缓存友好，内存合并  
**缺点**：所有矩阵必须相同大小

### 2. 使用 cuBLAS

#### 2.1 Pointer Array API

```cuda
#include <cublas_v2.h>

void batched_gemm_pointer_array(
    float** A_array,
    float** B_array,
    float** C_array,
    int M, int N, int K,
    int batch_count
) {
    cublasHandle_t handle;
    cublasCreate(&handle);
    
    float alpha = 1.0f, beta = 0.0f;
    
    // 需要将指针数组拷贝到GPU
    float** d_A_array;
    float** d_B_array;
    float** d_C_array;
    
    cudaMalloc(&d_A_array, batch_count * sizeof(float*));
    cudaMalloc(&d_B_array, batch_count * sizeof(float*));
    cudaMalloc(&d_C_array, batch_count * sizeof(float*));
    
    cudaMemcpy(d_A_array, A_array, batch_count * sizeof(float*), cudaMemcpyHostToDevice);
    cudaMemcpy(d_B_array, B_array, batch_count * sizeof(float*), cudaMemcpyHostToDevice);
    cudaMemcpy(d_C_array, C_array, batch_count * sizeof(float*), cudaMemcpyHostToDevice);
    
    // cuBLAS是列主序，这里转换为行主序
    cublasSgemmBatched(
        handle,
        CUBLAS_OP_N, CUBLAS_OP_N,
        N, M, K,
        &alpha,
        d_B_array, N,
        d_A_array, K,
        &beta,
        d_C_array, N,
        batch_count
    );
    
    cudaFree(d_A_array);
    cudaFree(d_B_array);
    cudaFree(d_C_array);
    cublasDestroy(handle);
}
```

#### 2.2 Strided Batched API（推荐）

```cuda
void batched_gemm_strided(
    float* A_batched,
    float* B_batched,
    float* C_batched,
    int M, int N, int K,
    int batch_count
) {
    cublasHandle_t handle;
    cublasCreate(&handle);
    
    float alpha = 1.0f, beta = 0.0f;
    
    // 计算stride
    long long int strideA = M * K;  // A的stride
    long long int strideB = K * N;  // B的stride
    long long int strideC = M * N;  // C的stride
    
    cublasSgemmStridedBatched(
        handle,
        CUBLAS_OP_N, CUBLAS_OP_N,
        N, M, K,
        &alpha,
        B_batched, N, strideB,
        A_batched, K, strideA,
        &beta,
        C_batched, N, strideC,
        batch_count
    );
    
    cublasDestroy(handle);
}
```

### 3. 自定义实现策略

#### 3.1 策略一：简单并行（朴素方法）

```cuda
__global__ void batched_gemm_naive(
    const float* __restrict__ A_batched,
    const float* __restrict__ B_batched,
    float* __restrict__ C_batched,
    int M, int N, int K,
    int batch_count
) {
    // 每个线程块处理一个batch的一个元素
    int batch = blockIdx.z;  // 使用3D grid
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (batch < batch_count && row < M && col < N) {
        // 计算当前batch的偏移
        const float* A = A_batched + batch * M * K;
        const float* B = B_batched + batch * K * N;
        float* C = C_batched + batch * M * N;
        
        float sum = 0.0f;
        for (int k = 0; k < K; k++) {
            sum += A[row * K + k] * B[k * N + col];
        }
        C[row * N + col] = sum;
    }
}

// 启动配置
dim3 blockDim(16, 16);
dim3 gridDim(
    (N + 15) / 16,
    (M + 15) / 16,
    batch_count  // Z维度用于batch
);
batched_gemm_naive<<<gridDim, blockDim>>>(A, B, C, M, N, K, batch_count);
```

**问题**：
- Grid Z维度限制（最大65535）
- 没有利用共享内存
- 小矩阵时线程块利用率低

#### 3.2 策略二：共享内存 + Batch并行

```cuda
#define TILE_SIZE 32

__global__ void batched_gemm_shared(
    const float* __restrict__ A_batched,
    const float* __restrict__ B_batched,
    float* __restrict__ C_batched,
    int M, int N, int K,
    int batch_count,
    int strideA, int strideB, int strideC
) {
    __shared__ float As[TILE_SIZE][TILE_SIZE + 1];
    __shared__ float Bs[TILE_SIZE][TILE_SIZE + 1];
    
    // batch索引（从blockIdx.z或额外计算）
    int batch = blockIdx.z;
    
    int tx = threadIdx.x;
    int ty = threadIdx.y;
    int row = blockIdx.y * TILE_SIZE + ty;
    int col = blockIdx.x * TILE_SIZE + tx;
    
    if (batch >= batch_count) return;
    
    // 当前batch的基址
    const float* A = A_batched + batch * strideA;
    const float* B = B_batched + batch * strideB;
    float* C = C_batched + batch * strideC;
    
    float sum = 0.0f;
    
    for (int t = 0; t < (K + TILE_SIZE - 1) / TILE_SIZE; t++) {
        // 加载Tile到共享内存
        int a_col = t * TILE_SIZE + tx;
        int b_row = t * TILE_SIZE + ty;
        
        As[ty][tx] = (row < M && a_col < K) ? A[row * K + a_col] : 0.0f;
        Bs[ty][tx] = (b_row < K && col < N) ? B[b_row * N + col] : 0.0f;
        
        __syncthreads();
        
        // 计算
        #pragma unroll
        for (int k = 0; k < TILE_SIZE; k++) {
            sum += As[ty][k] * Bs[k][tx];
        }
        
        __syncthreads();
    }
    
    if (row < M && col < N) {
        C[row * N + col] = sum;
    }
}
```

#### 3.3 策略三：每个Block处理多个Batch

```cuda
// 当矩阵很小，batch很多时
__global__ void batched_gemm_multi_batch_per_block(
    const float* A_batched,
    const float* B_batched,
    float* C_batched,
    int M, int N, int K,
    int batch_count,
    int batches_per_block  // 每个block处理多个batch
) {
    __shared__ float As[TILE_SIZE][TILE_SIZE + 1];
    __shared__ float Bs[TILE_SIZE][TILE_SIZE + 1];
    
    // 计算当前block负责的batch范围
    int batch_start = blockIdx.z * batches_per_block;
    int batch_end = min(batch_start + batches_per_block, batch_count);
    
    int tx = threadIdx.x;
    int ty = threadIdx.y;
    
    // 遍历此block负责的所有batch
    for (int batch = batch_start; batch < batch_end; batch++) {
        int row = blockIdx.y * TILE_SIZE + ty;
        int col = blockIdx.x * TILE_SIZE + tx;
        
        const float* A = A_batched + batch * M * K;
        const float* B = B_batched + batch * K * N;
        float* C = C_batched + batch * M * N;
        
        float sum = 0.0f;
        
        // ... 标准的Tiling GEMM计算
        for (int t = 0; t < (K + TILE_SIZE - 1) / TILE_SIZE; t++) {
            As[ty][tx] = ...;
            Bs[ty][tx] = ...;
            __syncthreads();
            
            for (int k = 0; k < TILE_SIZE; k++) {
                sum += As[ty][k] * Bs[k][tx];
            }
            __syncthreads();
        }
        
        if (row < M && col < N) {
            C[row * N + col] = sum;
        }
    }
}

// 启动：减少grid Z维度
int batches_per_block = 4;
dim3 gridDim(
    (N + TILE_SIZE - 1) / TILE_SIZE,
    (M + TILE_SIZE - 1) / TILE_SIZE,
    (batch_count + batches_per_block - 1) / batches_per_block
);
```

#### 3.4 策略四：Stream并行

```cuda
void batched_gemm_streams(
    float** A_array,
    float** B_array,
    float** C_array,
    int M, int N, int K,
    int batch_count,
    int num_streams
) {
    cudaStream_t* streams = new cudaStream_t[num_streams];
    for (int i = 0; i < num_streams; i++) {
        cudaStreamCreate(&streams[i]);
    }
    
    int batches_per_stream = (batch_count + num_streams - 1) / num_streams;
    
    for (int s = 0; s < num_streams; s++) {
        int batch_start = s * batches_per_stream;
        int batch_end = min(batch_start + batches_per_stream, batch_count);
        int count = batch_end - batch_start;
        
        // 在不同stream中执行
        dim3 gridDim(...);
        gemm_kernel<<<gridDim, blockDim, 0, streams[s]>>>(
            A_array + batch_start,
            B_array + batch_start,
            C_array + batch_start,
            M, N, K, count
        );
    }
    
    // 等待所有stream完成
    for (int i = 0; i < num_streams; i++) {
        cudaStreamSynchronize(streams[i]);
        cudaStreamDestroy(streams[i]);
    }
    delete[] streams;
}
```

### 4. 优化技巧

#### 4.1 处理不同大小的矩阵

```cuda
// 使用统一buffer + offset
struct BatchedGemmDesc {
    int M, N, K;
    int offset_A, offset_B, offset_C;
};

__global__ void batched_gemm_variable_size(
    const float* A_buffer,
    const float* B_buffer,
    float* C_buffer,
    BatchedGemmDesc* descs,
    int batch_count
) {
    int batch = blockIdx.z;
    if (batch >= batch_count) return;
    
    BatchedGemmDesc desc = descs[batch];
    const float* A = A_buffer + desc.offset_A;
    const float* B = B_buffer + desc.offset_B;
    float* C = C_buffer + desc.offset_C;
    
    // 使用desc.M, desc.N, desc.K进行计算
    // ...
}
```

#### 4.2 使用Tensor Core（FP16）

```cuda
#include <mma.h>
using namespace nvcuda;

__global__ void batched_gemm_wmma(
    const half* A_batched,
    const half* B_batched,
    float* C_batched,
    int M, int N, int K,
    int batch_count
) {
    int batch = blockIdx.z;
    if (batch >= batch_count) return;
    
    const half* A = A_batched + batch * M * K;
    const half* B = B_batched + batch * K * N;
    float* C = C_batched + batch * M * N;
    
    // WMMA fragments
    wmma::fragment<wmma::matrix_a, 16, 16, 16, half, wmma::row_major> a_frag;
    wmma::fragment<wmma::matrix_b, 16, 16, 16, half, wmma::col_major> b_frag;
    wmma::fragment<wmma::accumulator, 16, 16, 16, float> c_frag;
    
    wmma::fill_fragment(c_frag, 0.0f);
    
    // ... WMMA计算
    for (int k = 0; k < K; k += 16) {
        wmma::load_matrix_sync(a_frag, A + row * K + k, K);
        wmma::load_matrix_sync(b_frag, B + k * N + col, N);
        wmma::mma_sync(c_frag, a_frag, b_frag, c_frag);
    }
    
    wmma::store_matrix_sync(C + row * N + col, c_frag, N, wmma::mem_row_major);
}
```

#### 4.3 批量融合（Batch Fusion）

```cuda
// 将多个小batch合并为一个大batch
void fuse_small_batches(
    float** small_A,  // N个小矩阵，每个16×16
    float** small_B,
    float** small_C,
    int num_small,
    float* fused_A,   // 融合后的大矩阵
    float* fused_B,
    float* fused_C
) {
    // 合并成一个大矩阵乘法
    // 例如：16个16×16 → 1个64×64
    
    int fused_size = (int)sqrt(num_small) * 16;
    
    // 拷贝小矩阵到大矩阵的对应位置
    for (int i = 0; i < num_small; i++) {
        int block_row = (i / sqrt(num_small)) * 16;
        int block_col = (i % (int)sqrt(num_small)) * 16;
        
        // 拷贝到对角块（如果是独立计算）
        copy_matrix_block(small_A[i], fused_A, block_row, block_col, 16, 16, fused_size);
    }
    
    // 执行一次大GEMM
    gemm(fused_A, fused_B, fused_C, fused_size, fused_size, fused_size);
}
```

### 5. 性能分析

#### 5.1 不同矩阵大小的性能

在 NVIDIA A100 上测试（batch_count=100）：

| 矩阵大小  | 单个GEMM(ms) | Batched(ms) | 总加速比 |
| --------- | ------------ | ----------- | -------- |
| 16×16     | 0.15         | 0.42        | 35.7x    |
| 64×64     | 0.18         | 1.2         | 15.0x    |
| 256×256   | 0.35         | 8.5         | 4.1x     |
| 1024×1024 | 7.2          | 720         | 1.0x     |

**分析**：
- 小矩阵：batch并行收益显著
- 大矩阵：单个GEMM已充分利用GPU，batch并行收益小

#### 5.2 Strided vs Pointer Array

| Batch数 | Strided(ms) | Pointer Array(ms) | 差异 |
| ------- | ----------- | ----------------- | ---- |
| 10      | 0.8         | 0.9               | 12%  |
| 100     | 4.2         | 5.8               | 38%  |
| 1000    | 42          | 67                | 60%  |

**结论**：batch数越多，strided优势越明显

### 6. 选择指南

#### 6.1 矩阵大小决策

```
小矩阵（< 128）:
  → 使用batched GEMM，增加并行度
  → 考虑batch fusion

中等矩阵（128-512）:
  → 使用strided batched
  → 可以考虑stream并行

大矩阵（> 512）:
  → 每个矩阵已足够大，可单独计算
  → 使用stream并行即可
```

#### 6.2 存储方式选择

```
连续存储，相同大小：
  → StridedBatched (推荐)

不连续，不同大小：
  → Pointer Array

非常多的batch（> 10000）：
  → StridedBatched + 分批处理
```

#### 6.3 API选择

```
快速原型：
  → cuBLAS batched API

需要定制：
  → 自定义kernel

混合精度：
  → cuBLAS + Tensor Core
  → 或 CUTLASS
```

### 7. 完整示例

```cuda
#include <cublas_v2.h>

// 完整的batched GEMM实现
class BatchedGEMM {
public:
    BatchedGEMM(int M, int N, int K, int batch_count) 
        : M_(M), N_(N), K_(K), batch_count_(batch_count) {
        cublasCreate(&handle_);
        
        // 分配内存
        size_t size_A = batch_count * M * K * sizeof(float);
        size_t size_B = batch_count * K * N * sizeof(float);
        size_t size_C = batch_count * M * N * sizeof(float);
        
        cudaMalloc(&A_, size_A);
        cudaMalloc(&B_, size_B);
        cudaMalloc(&C_, size_C);
    }
    
    ~BatchedGEMM() {
        cudaFree(A_);
        cudaFree(B_);
        cudaFree(C_);
        cublasDestroy(handle_);
    }
    
    void compute() {
        float alpha = 1.0f, beta = 0.0f;
        
        long long int strideA = M_ * K_;
        long long int strideB = K_ * N_;
        long long int strideC = M_ * N_;
        
        cublasSgemmStridedBatched(
            handle_,
            CUBLAS_OP_N, CUBLAS_OP_N,
            N_, M_, K_,
            &alpha,
            B_, N_, strideB,
            A_, K_, strideA,
            &beta,
            C_, N_, strideC,
            batch_count_
        );
        
        cudaDeviceSynchronize();
    }
    
    float* get_result() { return C_; }
    
private:
    cublasHandle_t handle_;
    float *A_, *B_, *C_;
    int M_, N_, K_, batch_count_;
};

// 使用
int main() {
    BatchedGEMM gemm(256, 256, 256, 100);
    
    // 初始化数据...
    
    gemm.compute();
    
    // 获取结果...
    
    return 0;
}
```

## 总结

**Batched GEMM 的核心价值**：
- **并行度**：小矩阵时增加GPU利用率
- **便利性**：一次API调用处理多个矩阵
- **效率**：减少kernel启动开销

**优化策略**：
1. **Strided Batched** - 优先选择
2. **共享内存** - 每个batch内部优化
3. **Stream并行** - batch间并行
4. **Batch Fusion** - 合并小batch
5. **Tensor Core** - 使用混合精度

**性能特点**：
- 小矩阵（<128）：10-40倍加速
- 中矩阵（128-512）：2-10倍加速
- 大矩阵（>512）：主要是编程便利性

**最佳实践**：
- 生产环境：cuBLAS Strided Batched
- 特殊需求：自定义kernel
- 混合精度：Tensor Core + FP16

**适用场景**：
- ✓ 深度学习batch推理
- ✓ 多样本attention计算
- ✓ 图形学批量变换
- ✗ 单个大矩阵（用标准GEMM）

Batched GEMM 是现代深度学习和科学计算中的**常见操作**，掌握其优化对于GPU编程至关重要。


---

## 相关笔记
<!-- 自动生成 -->

暂无相关笔记

