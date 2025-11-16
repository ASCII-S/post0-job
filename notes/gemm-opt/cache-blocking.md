---
created: '2025-10-19'
last_reviewed: null
next_review: '2025-10-19'
review_count: 0
difficulty: medium
mastery_level: 0.0
tags:
- gemm-opt
- gemm-opt/cache-blocking.md
related_outlines: []
---
# 缓存块技术 (Cache Blocking/Tiling)

## 基本概念

缓存块技术（Cache Blocking）又称为分块（Tiling），是一种重要的内存优化技术，特别在矩阵运算（如 GEMM）中广泛应用。其核心思想是将大型数据结构分割成较小的块，使这些块能够完全放入 CPU 缓存中，从而最大化缓存的利用率。

## 为什么需要分块？

### 1. 内存层次结构的挑战
```
CPU 寄存器    < 1KB     ~1 cycle
L1 Cache     32-64KB    ~1-3 cycles
L2 Cache     256KB-1MB  ~10-20 cycles
L3 Cache     8-32MB     ~40-75 cycles
主内存       GB级       ~200-300 cycles
```

### 2. 朴素矩阵乘法的问题
对于 C = A × B 的矩阵乘法：
```c
// 朴素实现 - 缓存性能差
for (int i = 0; i < N; i++) {
    for (int j = 0; j < N; j++) {
        for (int k = 0; k < N; k++) {
            C[i][j] += A[i][k] * B[k][j];
        }
    }
}
```

**问题分析：**
- 当矩阵很大时，B[k][j] 的访问模式导致大量 cache miss
- 数据在内存中不连续访问，破坏了空间局部性
- 对于 N×N 矩阵，每个 B 的列需要重复从内存加载 N 次

## 分块如何提升缓存命中率

### 1. 基本分块策略
```c
// 分块矩阵乘法
#define BLOCK_SIZE 64

for (int ii = 0; ii < N; ii += BLOCK_SIZE) {
    for (int jj = 0; jj < N; jj += BLOCK_SIZE) {
        for (int kk = 0; kk < N; kk += BLOCK_SIZE) {
            // 对每个子块进行矩阵乘法
            for (int i = ii; i < min(ii + BLOCK_SIZE, N); i++) {
                for (int j = jj; j < min(jj + BLOCK_SIZE, N); j++) {
                    for (int k = kk; k < min(kk + BLOCK_SIZE, N); k++) {
                        C[i][j] += A[i][k] * B[k][j];
                    }
                }
            }
        }
    }
}
```

### 2. 提升效果
- **空间局部性：** 块内数据访问连续，充分利用缓存行
- **时间局部性：** 同一块数据被重复使用多次，减少内存访问
- **缓存容量利用：** 确保工作集大小适合缓存大小

## 如何确定最优的块大小

### 1. 理论计算方法

对于 L1 缓存，假设缓存大小为 S，需要同时存储 A、B、C 三个子矩阵块：
```
3 × (BLOCK_SIZE)² × sizeof(data_type) ≤ S
```

对于双精度浮点数（8 bytes）和 32KB L1 缓存：
```
3 × B² × 8 ≤ 32768
B² ≤ 1365
B ≤ 37
```

实际中通常选择 2 的幂次，如 32 或 64。

### 2. 主要考虑因素

#### A. 缓存层次结构
- **L1 缓存：** 最快但最小，决定最内层块大小
- **L2/L3 缓存：** 用于多层分块策略
- **TLB 大小：** 影响虚拟内存页的访问效率

#### B. 数据类型和精度
```c
// 不同数据类型的内存需求
float:    4 bytes  → 块大小可以更大
double:   8 bytes  → 块大小需要相应减小
int8:     1 byte   → 可以使用更大的块
```

#### C. 矩阵维度和形状
- 正方形矩阵：块大小通常相等
- 长方形矩阵：可能需要不同的行列块大小
- 很小的矩阵：分块可能反而降低性能

#### D. 算法特性
```c
// GEMM 中的内存访问模式
A[i][k]: 按行访问，空间局部性好
B[k][j]: 按列访问，需要优化
C[i][j]: 累加操作，需要保持在缓存中
```

### 3. 实际优化策略

#### A. 多层分块
```c
// 三层分块示例
#define L1_BLOCK 32   // 适配 L1 缓存
#define L2_BLOCK 128  // 适配 L2 缓存  
#define L3_BLOCK 512  // 适配 L3 缓存

// L3 层
for (int iii = 0; iii < N; iii += L3_BLOCK) {
    // L2 层
    for (int ii = iii; ii < min(iii + L3_BLOCK, N); ii += L2_BLOCK) {
        // L1 层
        for (int i = ii; i < min(ii + L2_BLOCK, N); i += L1_BLOCK) {
            // 具体计算...
        }
    }
}
```

#### B. 自适应块大小
```c
// 根据矩阵大小动态调整
int adaptive_block_size(int N) {
    if (N < 100) return N;        // 小矩阵不分块
    if (N < 1000) return 64;      // 中等矩阵
    return 128;                   // 大矩阵
}
```

### 4. 性能测试和调优

#### A. 基准测试框架
```c
double benchmark_gemm(int N, int block_size) {
    // 分配矩阵
    double *A = aligned_alloc(32, N * N * sizeof(double));
    double *B = aligned_alloc(32, N * N * sizeof(double));
    double *C = aligned_alloc(32, N * N * sizeof(double));
    
    // 初始化数据
    init_matrices(A, B, C, N);
    
    // 测量时间
    double start = get_time();
    blocked_gemm(A, B, C, N, block_size);
    double end = get_time();
    
    return (end - start);
}
```

#### B. 性能指标
- **GFLOPS：** 每秒十亿次浮点运算
- **Cache Miss Rate：** 缓存失效率
- **内存带宽利用率**

### 5. 实际应用中的优化技巧

#### A. 数据预取
```c
// 软件预取指令
__builtin_prefetch(&B[k+1][j], 0, 3);
```

#### B. 循环展开
```c
// 展开内层循环减少控制开销
for (int k = kk; k < kk + BLOCK_SIZE; k += 4) {
    C[i][j] += A[i][k] * B[k][j] + 
               A[i][k+1] * B[k+1][j] +
               A[i][k+2] * B[k+2][j] + 
               A[i][k+3] * B[k+3][j];
}
```

#### C. 向量化
```c
// 使用 SIMD 指令
#include <immintrin.h>
__m256d a_vec = _mm256_load_pd(&A[i][k]);
__m256d b_vec = _mm256_load_pd(&B[k][j]);
__m256d c_vec = _mm256_fmadd_pd(a_vec, b_vec, c_vec);
```

## 总结

缓存块技术是 GEMM 优化的核心技术之一，通过合理的分块策略可以显著提升性能：

1. **理解原理：** 充分利用缓存层次结构和数据局部性
2. **选择块大小：** 基于缓存大小、数据类型和矩阵特性
3. **多层优化：** 结合多级缓存进行分层分块
4. **实际测试：** 通过基准测试验证和调优参数
5. **综合优化：** 与其他技术（预取、向量化等）结合使用

在实际应用中，最优的块大小往往需要通过实验来确定，因为它依赖于具体的硬件平台、编译器优化和应用场景。

---

## 相关笔记
<!-- 自动生成 -->

- [packing_microkernel](notes/gemm-opt/packing_microkernel.md) - 相似度: 31% | 标签: gemm-opt, gemm-opt/packing_microkernel.md

