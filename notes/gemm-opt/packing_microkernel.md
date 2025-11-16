---
created: '2025-10-19'
last_reviewed: null
next_review: '2025-10-19'
review_count: 0
difficulty: medium
mastery_level: 0.0
tags:
- gemm-opt
- gemm-opt/packing_microkernel.md
related_outlines: []
---
# GEMM 打包和微内核优化实现

## 概述

为GEMM算法添加了三种新的优化技术：
1. **打包优化** (`gemm_packed`)
2. **微内核优化** (`gemm_microkernel`) 
3. **打包+微内核组合优化** (`gemm_packed_microkernel`)

## 核心优化技术

### 1. 数据打包 (Data Packing)

**目的**：将矩阵数据重新排列到连续的内存区域，减少缓存未命中。

**实现**：
- `pack_matrix_A()`: 按行主序打包矩阵A的子块
- `pack_matrix_B()`: 按列主序打包矩阵B的子块

**优势**：
- 提高空间局部性
- 减少内存访问跨度
- 为向量化操作准备对齐的数据

```cpp
// 打包A矩阵：行主序存储
void pack_matrix_A(const double* A, double* A_packed, 
                   int start_row, int start_col, 
                   int block_rows, int block_cols, int N)

// 打包B矩阵：列主序存储  
void pack_matrix_B(const double* B, double* B_packed,
                   int start_row, int start_col,
                   int block_rows, int block_cols, int N)
```

### 2. 微内核 (Micro-kernel)

**目的**：使用高度优化的小型内核处理固定大小的矩阵块。

**实现**：
- **4x4 AVX2微内核**：使用SIMD指令处理4x4块
- **标量微内核**：处理边界和非对齐情况

**关键特性**：
- 使用FMA指令 (`_mm256_fmadd_pd`)
- 向量广播 (`_mm256_broadcast_sd`)
- 寄存器阻塞技术

```cpp
// 4x4微内核使用AVX2向量化
void microkernel_4x4_avx2(const double* A_packed, const double* B_packed, 
                          double* C, int N, int start_i, int start_j, int k_size)

// 标量微内核处理边界情况
void microkernel_scalar(const double* A_packed, const double* B_packed, 
                        double* C, int N, int start_i, int start_j, 
                        int block_rows, int block_cols, int k_size)
```

### 3. 分块参数配置

| 参数 | 含义             | 推荐值 | 说明                     |
| ---- | ---------------- | ------ | ------------------------ |
| MC   | macro-kernel行数 | 128    | L2缓存友好的A矩阵行分块  |
| NC   | macro-kernel列数 | 64     | L2缓存友好的B矩阵列分块  |
| KC   | 公共维度分块     | 256    | L1缓存友好的内层循环分块 |
| MR   | micro-kernel行数 | 4      | AVX2向量长度匹配         |
| NR   | micro-kernel列数 | 4      | 寄存器阻塞大小           |

## 算法分析

### gemm_packed
- **优化重点**：数据局部性
- **内存模式**：分层打包，减少缓存未命中
- **复杂度**：O(N³) + O(N²) 打包开销

### gemm_microkernel  
- **优化重点**：计算效率
- **SIMD利用**：4x4块完全向量化
- **寄存器使用**：最大化寄存器重用

### gemm_packed_microkernel
- **优化重点**：内存+计算双重优化
- **并行化**：OpenMP外层循环并行
- **缓存分层**：L1/L2/L3缓存层次优化

## 性能预期

基于典型的高性能GEMM库设计模式，预期性能提升：

| 算法                      | 预期加速比 | 主要优势              |
| ------------------------- | ---------- | --------------------- |
| `gemm_packed`             | 2-3x       | 数据局部性提升        |
| `gemm_microkernel`        | 4-6x       | SIMD向量化+寄存器优化 |
| `gemm_packed_microkernel` | 8-15x      | 综合优化+并行化       |

## 编译和测试

### 编译要求
```bash
# 需要AVX2支持
g++ -mavx2 -mfma -O3 -fopenmp

# 或使用Makefile
make clean && make
```

### 性能测试
```bash
# 运行基准测试
./gemm_benchmark

# 测试不同矩阵规模
./gemm_benchmark 512 1024 2048 4096
```

## 理论背景

### 1. Goto算法思想
这些优化基于Goto和van de Geijn的高性能GEMM算法设计：
- **分层内存优化**：匹配缓存层次结构
- **计算内核优化**：最大化浮点吞吐量
- **数据移动最小化**：减少内存带宽瓶颈

### 2. 内存层次优化
```
L3缓存: NC x KC 的B块
L2缓存: MC x KC 的A块  
L1缓存: MR x NR 的微内核块
寄存器: 4x4的数据复用
```

### 3. 计算强度分析
- **朴素算法**：2N³ flops, 3N² 内存访问 → 计算强度 = 2N/3
- **优化算法**：2N³ flops, ~N² 内存访问 → 计算强度 = 2N

## 进一步优化方向

1. **更大的微内核**：6x8, 8x6 等非对称微内核
2. **预取优化**：软件预取指令精确控制
3. **多精度支持**：float, int8 等数据类型
4. **NUMA优化**：多插槽系统的内存亲和性
5. **GPU移植**：CUDA/OpenCL版本实现

---

## 相关笔记
<!-- 自动生成 -->

- [cache-blocking](notes/gemm-opt/cache-blocking.md) - 相似度: 31% | 标签: gemm-opt, gemm-opt/cache-blocking.md

