---
created: '2025-10-19'
last_reviewed: null
next_review: '2025-10-19'
review_count: 0
difficulty: medium
mastery_level: 0.0
tags:
- cuda
- cuda/WMMA_API的使用方法和注意事项？.md
related_outlines: []
---
# WMMA API的使用方法和注意事项？

## 面试标准答案

WMMA（Warp Matrix Multiply Accumulate）是CUDA提供的**Warp级别的Tensor Core编程接口**。使用流程包括：1) **声明fragment** - 定义矩阵片段类型（matrix_a/matrix_b/accumulator）和尺寸（通常16×16×16）；2) **加载数据** - 使用`load_matrix_sync`从内存加载到fragment；3) **执行计算** - 调用`mma_sync`进行矩阵乘加；4) **存储结果** - 用`store_matrix_sync`写回内存。注意事项包括：fragment是分布在warp的32个线程中的；需要整个warp同步执行；内存地址必须对齐；矩阵大小必须是WMMA_M/N/K的倍数；推荐使用FP16输入+FP32累加器保证精度。

---

## 详细讲解

### 1. WMMA API 概述

#### 1.1 什么是 WMMA

**定义**：
```cpp
namespace nvcuda::wmma {
    // Warp Matrix Multiply-Accumulate
    // 提供对 Tensor Core 的直接访问
}
```

**关键特点**：
- **Warp级别**：一个warp（32线程）协作完成一次矩阵乘法
- **同步操作**：所有操作都是warp同步的
- **高性能**：直接映射到Tensor Core硬件

**支持的架构**：
- Volta (SM 7.0): V100
- Turing (SM 7.5): T4, RTX 20系列
- Ampere (SM 8.0, 8.6): A100, RTX 30系列
- Hopper (SM 9.0): H100

#### 1.2 基本概念

**Fragment**：
- 矩阵的分布式表示
- 每个线程持有fragment的一部分
- 不能直接访问fragment的元素

**矩阵类型**：
```cpp
// matrix_a: 左侧矩阵 (M×K)
// matrix_b: 右侧矩阵 (K×N)
// accumulator: 累加器和结果 (M×N)
```

**操作**：
```
D (accumulator) = A (matrix_a) × B (matrix_b) + C (accumulator)
```

### 2. Fragment 声明和类型

#### 2.1 Fragment 声明

```cuda
#include <mma.h>
using namespace nvcuda;

// 基本语法
wmma::fragment<Use, m, n, k, T, Layout> frag;
```

**参数说明**：

| 参数   | 说明     | 可选值                                |
| ------ | -------- | ------------------------------------- |
| Use    | 用途     | `matrix_a`, `matrix_b`, `accumulator` |
| m      | 行数     | 通常 16 或 8                          |
| n      | 列数     | 通常 16 或 8                          |
| k      | K维度    | 通常 16 或 8                          |
| T      | 数据类型 | `half`, `float`, `int`, `unsigned`    |
| Layout | 内存布局 | `row_major`, `col_major`              |

#### 2.2 常用配置

```cuda
// FP16输入 + FP32累加（最常用）
wmma::fragment<wmma::matrix_a, 16, 16, 16, half, wmma::row_major> a_frag;
wmma::fragment<wmma::matrix_b, 16, 16, 16, half, wmma::col_major> b_frag;
wmma::fragment<wmma::accumulator, 16, 16, 16, float> acc_frag;

// TF32输入 + FP32累加（Ampere+）
wmma::fragment<wmma::matrix_a, 16, 16, 8, precision::tf32, wmma::row_major> a_frag_tf32;
wmma::fragment<wmma::matrix_b, 16, 16, 8, precision::tf32, wmma::col_major> b_frag_tf32;

// INT8输入 + INT32累加
wmma::fragment<wmma::matrix_a, 16, 16, 16, signed char, wmma::row_major> a_frag_int8;
wmma::fragment<wmma::matrix_b, 16, 16, 16, signed char, wmma::col_major> b_frag_int8;
wmma::fragment<wmma::accumulator, 16, 16, 16, int> acc_frag_int32;
```

#### 2.3 支持的尺寸

| 架构   | 数据类型            | 支持的 (M, N, K)                                    |
| ------ | ------------------- | --------------------------------------------------- |
| Volta  | FP16                | (16, 16, 16)                                        |
| Turing | FP16/INT8           | (16, 16, 16), (8, 32, 16), (32, 8, 16)              |
| Ampere | FP16/BF16/TF32/INT8 | (16, 16, 16), (8, 32, 16), (32, 8, 16), (16, 16, 8) |
| Hopper | FP8/FP16/BF16/TF32  | 更多组合                                            |

### 3. WMMA 核心API

#### 3.1 fill_fragment

**用途**：初始化accumulator fragment

```cuda
void fill_fragment(fragment<accumulator, ...>& frag, T value);

// 示例：初始化为0
wmma::fragment<wmma::accumulator, 16, 16, 16, float> c_frag;
wmma::fill_fragment(c_frag, 0.0f);

// 初始化为其他值
wmma::fill_fragment(c_frag, 1.0f);
```

#### 3.2 load_matrix_sync

**用途**：从内存加载矩阵到fragment

```cuda
void load_matrix_sync(
    fragment<Use, m, n, k, T, Layout>& frag,
    const T* ptr,
    unsigned ldm
);
```

**参数**：
- `frag`: 目标fragment
- `ptr`: 内存指针（全局或共享内存）
- `ldm`: leading dimension（行主序是列数，列主序是行数）

**示例**：
```cuda
__shared__ half As[16][16];

// 从共享内存加载（行主序）
wmma::fragment<wmma::matrix_a, 16, 16, 16, half, wmma::row_major> a_frag;
wmma::load_matrix_sync(a_frag, &As[0][0], 16);  // ldm = 16（列数）

// 从全局内存加载（列主序）
wmma::fragment<wmma::matrix_b, 16, 16, 16, half, wmma::col_major> b_frag;
wmma::load_matrix_sync(b_frag, B_global, N);  // ldm = N
```

**内存布局要求**：
```cuda
// 行主序 (row_major)：
// ptr[i][j] = ptr[i * ldm + j]

// 列主序 (col_major)：
// ptr[i][j] = ptr[j * ldm + i]
```

#### 3.3 mma_sync

**用途**：执行矩阵乘加运算

```cuda
void mma_sync(
    fragment<accumulator, ...>& d,
    const fragment<matrix_a, ...>& a,
    const fragment<matrix_b, ...>& b,
    const fragment<accumulator, ...>& c
);
```

**操作**：`d = a × b + c`

**示例**：
```cuda
wmma::fragment<wmma::matrix_a, 16, 16, 16, half, wmma::row_major> a_frag;
wmma::fragment<wmma::matrix_b, 16, 16, 16, half, wmma::col_major> b_frag;
wmma::fragment<wmma::accumulator, 16, 16, 16, float> c_frag, d_frag;

wmma::load_matrix_sync(a_frag, A, K);
wmma::load_matrix_sync(b_frag, B, N);
wmma::fill_fragment(c_frag, 0.0f);

// 执行矩阵乘加
wmma::mma_sync(d_frag, a_frag, b_frag, c_frag);
```

#### 3.4 store_matrix_sync

**用途**：将fragment存储回内存

```cuda
void store_matrix_sync(
    T* ptr,
    const fragment<accumulator, ...>& frag,
    unsigned ldm,
    layout_t layout
);
```

**参数**：
- `ptr`: 目标内存指针
- `frag`: 源fragment
- `ldm`: leading dimension
- `layout`: `wmma::mem_row_major` 或 `wmma::mem_col_major`

**示例**：
```cuda
__shared__ float Cs[16][16];

wmma::fragment<wmma::accumulator, 16, 16, 16, float> d_frag;

// 存储到共享内存（行主序）
wmma::store_matrix_sync(&Cs[0][0], d_frag, 16, wmma::mem_row_major);

// 存储到全局内存（行主序）
wmma::store_matrix_sync(&C[row * N + col], d_frag, N, wmma::mem_row_major);
```

### 4. 完整示例

#### 4.1 基础GEMM实现

```cuda
#include <mma.h>
using namespace nvcuda;

#define WMMA_M 16
#define WMMA_N 16
#define WMMA_K 16

__global__ void wmma_gemm_kernel(
    const half* __restrict__ A,  // M × K
    const half* __restrict__ B,  // K × N
    float* __restrict__ C,       // M × N
    int M, int N, int K
) {
    // 每个warp计算一个16×16的输出块
    int warpM = (blockIdx.x * blockDim.x + threadIdx.x) / 32;
    int warpN = (blockIdx.y * blockDim.y + threadIdx.y);
    
    // 全局行列索引
    int row = warpM * WMMA_M;
    int col = warpN * WMMA_N;
    
    // 边界检查
    if (row >= M || col >= N) return;
    
    // 声明fragments
    wmma::fragment<wmma::matrix_a, WMMA_M, WMMA_N, WMMA_K, half, wmma::row_major> a_frag;
    wmma::fragment<wmma::matrix_b, WMMA_M, WMMA_N, WMMA_K, half, wmma::col_major> b_frag;
    wmma::fragment<wmma::accumulator, WMMA_M, WMMA_N, WMMA_K, float> acc_frag;
    
    // 初始化累加器
    wmma::fill_fragment(acc_frag, 0.0f);
    
    // 遍历K维度
    for (int k = 0; k < K; k += WMMA_K) {
        // 加载A的16×16块
        wmma::load_matrix_sync(a_frag, A + row * K + k, K);
        
        // 加载B的16×16块
        wmma::load_matrix_sync(b_frag, B + k * N + col, N);
        
        // 累加：acc = a × b + acc
        wmma::mma_sync(acc_frag, a_frag, b_frag, acc_frag);
    }
    
    // 存储结果
    wmma::store_matrix_sync(C + row * N + col, acc_frag, N, wmma::mem_row_major);
}

// 启动配置
void launch_wmma_gemm(const half* A, const half* B, float* C, int M, int N, int K) {
    // 每个block包含多个warp
    dim3 blockDim(128);  // 4个warp
    dim3 gridDim(
        (M + WMMA_M - 1) / WMMA_M / 4,
        (N + WMMA_N - 1) / WMMA_N
    );
    
    wmma_gemm_kernel<<<gridDim, blockDim>>>(A, B, C, M, N, K);
}
```

#### 4.2 使用共享内存优化

```cuda
#define BLOCK_SIZE 128
#define WMMA_M 16
#define WMMA_N 16
#define WMMA_K 16

__global__ void wmma_gemm_shared(
    const half* __restrict__ A,
    const half* __restrict__ B,
    float* __restrict__ C,
    int M, int N, int K
) {
    // 共享内存Tile
    __shared__ half As[BLOCK_SIZE][BLOCK_SIZE];
    __shared__ half Bs[BLOCK_SIZE][BLOCK_SIZE];
    
    // Warp坐标
    int warpId = (threadIdx.x + threadIdx.y * blockDim.x) / 32;
    int laneId = threadIdx.x % 32;
    int warpM = (warpId / (BLOCK_SIZE / WMMA_M));
    int warpN = (warpId % (BLOCK_SIZE / WMMA_N));
    
    // 全局坐标
    int block_row = blockIdx.y * BLOCK_SIZE;
    int block_col = blockIdx.x * BLOCK_SIZE;
    
    // Fragments
    wmma::fragment<wmma::matrix_a, WMMA_M, WMMA_N, WMMA_K, half, wmma::row_major> a_frag;
    wmma::fragment<wmma::matrix_b, WMMA_M, WMMA_N, WMMA_K, half, wmma::col_major> b_frag;
    wmma::fragment<wmma::accumulator, WMMA_M, WMMA_N, WMMA_K, float> acc_frag;
    
    wmma::fill_fragment(acc_frag, 0.0f);
    
    // Tile循环
    for (int tile_k = 0; tile_k < K; tile_k += BLOCK_SIZE) {
        // 协作加载到共享内存
        for (int i = threadIdx.y; i < BLOCK_SIZE; i += blockDim.y) {
            for (int j = threadIdx.x; j < BLOCK_SIZE; j += blockDim.x) {
                int a_row = block_row + i;
                int a_col = tile_k + j;
                As[i][j] = (a_row < M && a_col < K) ? A[a_row * K + a_col] : __float2half(0.0f);
                
                int b_row = tile_k + i;
                int b_col = block_col + j;
                Bs[i][j] = (b_row < K && b_col < N) ? B[b_row * N + b_col] : __float2half(0.0f);
            }
        }
        
        __syncthreads();
        
        // WMMA计算
        for (int k = 0; k < BLOCK_SIZE; k += WMMA_K) {
            int a_row = warpM * WMMA_M;
            int b_col = warpN * WMMA_N;
            
            wmma::load_matrix_sync(a_frag, &As[a_row][k], BLOCK_SIZE);
            wmma::load_matrix_sync(b_frag, &Bs[k][b_col], BLOCK_SIZE);
            wmma::mma_sync(acc_frag, a_frag, b_frag, acc_frag);
        }
        
        __syncthreads();
    }
    
    // 写回结果
    int c_row = block_row + warpM * WMMA_M;
    int c_col = block_col + warpN * WMMA_N;
    if (c_row < M && c_col < N) {
        wmma::store_matrix_sync(C + c_row * N + c_col, acc_frag, N, wmma::mem_row_major);
    }
}
```

### 5. 重要注意事项

#### 5.1 Warp同步要求

```cuda
// ✓ 正确：整个warp执行WMMA
if (warpId < num_warps) {
    wmma::load_matrix_sync(a_frag, ptr, ldm);
    wmma::mma_sync(d, a, b, c);
}

// ✗ 错误：部分线程执行WMMA
if (threadIdx.x < 16) {  // 只有部分warp线程
    wmma::load_matrix_sync(...);  // 死锁或未定义行为
}
```

**规则**：
- WMMA操作必须被warp的所有32个线程执行
- 不能在分支中使用WMMA（除非整个warp同进同出）

#### 5.2 内存对齐

```cuda
// ✓ 正确：地址对齐
__shared__ __align__(16) half As[16][16];
wmma::load_matrix_sync(a_frag, &As[0][0], 16);

// ✓ 正确：cudaMalloc自动对齐
half* d_A;
cudaMalloc(&d_A, size);

// ✗ 可能有问题：未对齐的指针
half* unaligned = aligned_ptr + 1;
wmma::load_matrix_sync(a_frag, unaligned, ldm);  // 可能崩溃
```

**对齐要求**：
- FP16: 至少2字节对齐（推荐16字节）
- FP32: 至少4字节对齐（推荐16字节）

#### 5.3 矩阵维度要求

```cuda
// ✓ 正确：维度是16的倍数
int M = 1024;  // 1024 % 16 == 0
int N = 2048;  // 2048 % 16 == 0
int K = 512;   // 512 % 16 == 0

// ✗ 处理非对齐维度需要padding
int M = 1000;  // 1000 % 16 != 0
// 需要padding到1008
```

**处理方法**：
```cuda
// 方法1：padding
int M_padded = ((M + 15) / 16) * 16;

// 方法2：边界处理
if (row < M && col < N) {
    wmma::store_matrix_sync(...);
}
```

#### 5.4 Fragment元素访问限制

```cuda
wmma::fragment<wmma::accumulator, 16, 16, 16, float> frag;

// ✗ 错误：不能直接访问fragment元素
// float val = frag[0];  // 编译错误

// ✓ 正确：通过存储到内存访问
float temp[16 * 16];
wmma::store_matrix_sync(temp, frag, 16, wmma::mem_row_major);
float val = temp[i];
```

#### 5.5 数据类型匹配

```cuda
// ✓ 正确：类型匹配
wmma::fragment<wmma::matrix_a, 16, 16, 16, half, wmma::row_major> a_frag;
half* A_half;
wmma::load_matrix_sync(a_frag, A_half, K);

// ✗ 错误：类型不匹配
float* A_float;
wmma::load_matrix_sync(a_frag, A_float, K);  // 编译错误
```

### 6. 性能优化技巧

#### 6.1 多个WMMA操作流水线

```cuda
// 展开K循环，并行多个WMMA
#pragma unroll
for (int k = 0; k < BLOCK_SIZE; k += WMMA_K) {
    wmma::load_matrix_sync(a_frag, &As[warp_row][k], BLOCK_SIZE);
    wmma::load_matrix_sync(b_frag, &Bs[k][warp_col], BLOCK_SIZE);
    wmma::mma_sync(acc_frag, a_frag, b_frag, acc_frag);
}
```

#### 6.2 每个Warp计算多个输出Tile

```cuda
// 每个warp计算2×2个16×16输出块
wmma::fragment<wmma::accumulator, 16, 16, 16, float> acc[2][2];

for (int i = 0; i < 2; i++) {
    for (int j = 0; j < 2; j++) {
        wmma::fill_fragment(acc[i][j], 0.0f);
    }
}

for (int k = 0; k < TILE_K; k += WMMA_K) {
    wmma::load_matrix_sync(a_frag[0], &As[warp_row][k], BLOCK_SIZE);
    wmma::load_matrix_sync(a_frag[1], &As[warp_row + 16][k], BLOCK_SIZE);
    wmma::load_matrix_sync(b_frag[0], &Bs[k][warp_col], BLOCK_SIZE);
    wmma::load_matrix_sync(b_frag[1], &Bs[k][warp_col + 16], BLOCK_SIZE);
    
    for (int i = 0; i < 2; i++) {
        for (int j = 0; j < 2; j++) {
            wmma::mma_sync(acc[i][j], a_frag[i], b_frag[j], acc[i][j]);
        }
    }
}
```

#### 6.3 使用 `__restrict__`

```cuda
__global__ void wmma_kernel(
    const half* __restrict__ A,  // 告诉编译器A不会与B、C别名
    const half* __restrict__ B,
    float* __restrict__ C,
    int M, int N, int K
) {
    // ...
}
```

### 7. 调试技巧

#### 7.1 验证WMMA使用

```cuda
// 检查是否在warp内
__device__ void check_warp() {
    int warpId = (threadIdx.x + threadIdx.y * blockDim.x) / 32;
    int laneId = threadIdx.x % 32;
    
    if (laneId == 0) {
        printf("Warp %d executing WMMA\n", warpId);
    }
}
```

#### 7.2 验证结果正确性

```cuda
// CPU验证
void verify_wmma(half* A, half* B, float* C_gpu, int M, int N, int K) {
    float* C_cpu = new float[M * N];
    
    // CPU计算
    for (int i = 0; i < M; i++) {
        for (int j = 0; j < N; j++) {
            float sum = 0.0f;
            for (int k = 0; k < K; k++) {
                sum += __half2float(A[i * K + k]) * __half2float(B[k * N + j]);
            }
            C_cpu[i * N + j] = sum;
        }
    }
    
    // 比较
    float max_error = 0.0f;
    for (int i = 0; i < M * N; i++) {
        float error = fabs(C_gpu[i] - C_cpu[i]);
        max_error = max(max_error, error);
    }
    
    printf("Max error: %f\n", max_error);
    delete[] C_cpu;
}
```

### 8. 常见错误和解决方案

| 错误     | 原因                 | 解决方案                |
| -------- | -------------------- | ----------------------- |
| 程序挂起 | warp部分线程执行WMMA | 确保整个warp执行        |
| 结果错误 | 矩阵布局不匹配       | 检查row_major/col_major |
| 性能低   | 未使用共享内存       | 添加Tiling优化          |
| 编译错误 | 架构不支持           | 添加 `-arch=sm_70+`     |
| 对齐错误 | 地址未对齐           | 使用 `__align__`        |

### 9. 编译选项

```bash
# 基本编译
nvcc -arch=sm_70 wmma_kernel.cu -o wmma

# 优化编译
nvcc -arch=sm_80 -O3 \
     -use_fast_math \
     -gencode arch=compute_80,code=sm_80 \
     wmma_kernel.cu -o wmma

# 多架构支持
nvcc -gencode arch=compute_70,code=sm_70 \
     -gencode arch=compute_80,code=sm_80 \
     wmma_kernel.cu -o wmma
```

## 总结

**WMMA API 核心要点**：
1. **Fragment** - 分布式矩阵表示
2. **Warp同步** - 所有操作必须整个warp执行
3. **四个API** - fill, load, mma, store
4. **类型匹配** - 严格的类型和布局要求

**使用流程**：
```
声明fragment → 初始化 → 加载 → 计算 → 存储
```

**注意事项**：
- ✓ 整个warp同步执行
- ✓ 内存地址对齐
- ✓ 矩阵维度对齐
- ✓ 数据类型匹配
- ✓ 正确的内存布局

**性能优化**：
- 使用共享内存Tiling
- 展开循环
- 每个warp计算多个Tile
- 结合双缓冲

**最佳实践**：
- 生产代码优先使用cuBLAS
- 学习和研究使用WMMA
- 高级优化考虑CUTLASS

WMMA API提供了对Tensor Core的**精细控制**，是理解GPU矩阵加速硬件的重要工具。


---

## 相关笔记
<!-- 自动生成 -->

暂无相关笔记

