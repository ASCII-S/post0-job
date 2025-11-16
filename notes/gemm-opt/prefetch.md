---
created: '2025-10-19'
last_reviewed: null
next_review: '2025-10-19'
review_count: 0
difficulty: medium
mastery_level: 0.0
tags:
- gemm-opt
- gemm-opt/prefetch.md
related_outlines: []
---
# GEMM 中预取技术的使用

## 基本概念

预取（Prefetching）是一种重要的内存优化技术，通过提前将数据从慢速存储（主内存）加载到快速存储（缓存）中，来隐藏内存访问延迟。在 GEMM（通用矩阵乘法）这种内存密集型操作中，合理使用预取技术可以显著提升性能。

## 预取技术的分类

### 1. 硬件预取（Hardware Prefetching）

#### A. 自动检测模式
现代 CPU 具有多种硬件预取器：

```
stride prefetcher:     检测固定步长的内存访问模式
stream prefetcher:     检测连续的内存访问
next-line prefetcher:  预取下一个缓存行
adjacent prefetcher:   预取相邻的缓存行
```

#### B. 硬件预取的优缺点
**优点：**
- 自动工作，无需程序员干预
- 对规律的访问模式效果很好
- 低开销

**缺点：**
- 只能检测简单的访问模式
- 预取距离有限
- 可能产生无用的预取，浪费带宽

### 2. 软件预取（Software Prefetching）

#### A. 显式预取指令
程序员通过插入预取指令来控制数据加载：

```c
// x86-64 预取指令
__builtin_prefetch(addr, rw, locality);
// addr: 要预取的地址
// rw: 0=读取, 1=写入
// locality: 0-3, 局部性提示（3=高局部性）
```

#### B. 编译器预取
编译器自动插入预取指令：
```c
// GCC/Clang 编译选项
-fprefetch-loop-arrays
```

## 在 GEMM 中应用预取技术

### 1. 内存访问模式分析

标准的 GEMM 操作 `C = A × B` 中：
```c
for (int i = 0; i < M; i++) {
    for (int j = 0; j < N; j++) {
        for (int k = 0; k < K; k++) {
            C[i][j] += A[i][k] * B[k][j];
        }
    }
}
```

**访问模式分析：**
- `A[i][k]`：按行访问，空间局部性好
- `B[k][j]`：按列访问，容易产生 cache miss
- `C[i][j]`：累加操作，需要保持在缓存中

### 2. 基本预取策略

#### A. 简单的软件预取
```c
#define PREFETCH_DISTANCE 8

for (int i = 0; i < M; i++) {
    for (int j = 0; j < N; j++) {
        // 预取 A 矩阵的下几行
        if (k + PREFETCH_DISTANCE < K) {
            __builtin_prefetch(&A[i][k + PREFETCH_DISTANCE], 0, 3);
        }
        
        // 预取 B 矩阵的下几行
        if (k + PREFETCH_DISTANCE < K) {
            __builtin_prefetch(&B[k + PREFETCH_DISTANCE][j], 0, 3);
        }
        
        for (int k = 0; k < K; k++) {
            C[i][j] += A[i][k] * B[k][j];
        }
    }
}
```

#### B. 分块 + 预取的组合策略
```c
#define BLOCK_SIZE 64
#define PREFETCH_DISTANCE 2

void blocked_gemm_with_prefetch(double *A, double *B, double *C, 
                                int M, int N, int K) {
    for (int ii = 0; ii < M; ii += BLOCK_SIZE) {
        for (int jj = 0; jj < N; jj += BLOCK_SIZE) {
            for (int kk = 0; kk < K; kk += BLOCK_SIZE) {
                
                // 预取下一个块
                if (kk + BLOCK_SIZE < K) {
                    for (int p = 0; p < BLOCK_SIZE; p += 8) {
                        __builtin_prefetch(&A[(ii + p) * K + kk + BLOCK_SIZE], 0, 2);
                        __builtin_prefetch(&B[(kk + BLOCK_SIZE) * N + jj + p], 0, 2);
                    }
                }
                
                // 计算当前块
                for (int i = ii; i < min(ii + BLOCK_SIZE, M); i++) {
                    for (int j = jj; j < min(jj + BLOCK_SIZE, N); j++) {
                        for (int k = kk; k < min(kk + BLOCK_SIZE, K); k++) {
                            C[i * N + j] += A[i * K + k] * B[k * N + j];
                        }
                    }
                }
            }
        }
    }
}
```

### 3. 高级预取策略

#### A. 多级预取
```c
void multilevel_prefetch_gemm(double *A, double *B, double *C, 
                              int M, int N, int K) {
    const int L1_DISTANCE = 4;   // L1 缓存预取距离
    const int L2_DISTANCE = 16;  // L2 缓存预取距离
    const int L3_DISTANCE = 64;  // L3 缓存预取距离
    
    for (int i = 0; i < M; i++) {
        for (int j = 0; j < N; j++) {
            for (int k = 0; k < K; k++) {
                
                // L1 预取（高局部性）
                if (k + L1_DISTANCE < K) {
                    __builtin_prefetch(&A[i * K + k + L1_DISTANCE], 0, 3);
                    __builtin_prefetch(&B[(k + L1_DISTANCE) * N + j], 0, 3);
                }
                
                // L2 预取（中等局部性）
                if (k + L2_DISTANCE < K) {
                    __builtin_prefetch(&A[i * K + k + L2_DISTANCE], 0, 2);
                    __builtin_prefetch(&B[(k + L2_DISTANCE) * N + j], 0, 2);
                }
                
                // L3 预取（低局部性）
                if (k + L3_DISTANCE < K) {
                    __builtin_prefetch(&A[i * K + k + L3_DISTANCE], 0, 1);
                    __builtin_prefetch(&B[(k + L3_DISTANCE) * N + j], 0, 1);
                }
                
                C[i * N + j] += A[i * K + k] * B[k * N + j];
            }
        }
    }
}
```

#### B. 自适应预取距离
```c
int calculate_prefetch_distance(int matrix_size, int cache_size) {
    // 基于矩阵大小和缓存大小动态计算预取距离
    int base_distance = cache_size / (matrix_size * sizeof(double));
    
    // 限制在合理范围内
    if (base_distance < 2) return 2;
    if (base_distance > 32) return 32;
    
    return base_distance;
}

void adaptive_prefetch_gemm(double *A, double *B, double *C, 
                           int M, int N, int K) {
    int l1_distance = calculate_prefetch_distance(M, 32 * 1024);    // 32KB L1
    int l2_distance = calculate_prefetch_distance(M, 256 * 1024);   // 256KB L2
    
    for (int i = 0; i < M; i++) {
        for (int j = 0; j < N; j++) {
            for (int k = 0; k < K; k++) {
                
                if (k + l1_distance < K) {
                    __builtin_prefetch(&A[i * K + k + l1_distance], 0, 3);
                    __builtin_prefetch(&B[(k + l1_distance) * N + j], 0, 3);
                }
                
                if (k + l2_distance < K) {
                    __builtin_prefetch(&A[i * K + k + l2_distance], 0, 2);
                    __builtin_prefetch(&B[(k + l2_distance) * N + j], 0, 2);
                }
                
                C[i * N + j] += A[i * K + k] * B[k * N + j];
            }
        }
    }
}
```

## 预取优化技巧

### 1. 预取距离的选择

#### A. 理论计算
```c
// 预取距离 = 内存延迟 / 计算时间
// 内存延迟：~200-300 cycles
// 每次 FMA 操作：~1 cycle
// 所以预取距离应该在 200-300 之间

int optimal_prefetch_distance() {
    int memory_latency = 250;  // cycles
    int computation_cycles = 1; // per operation
    return memory_latency / computation_cycles;
}
```

#### B. 实验确定
```c
double benchmark_prefetch_distance(int distance) {
    // 测试不同预取距离的性能
    double start = get_time();
    gemm_with_prefetch(A, B, C, M, N, K, distance);
    double end = get_time();
    return end - start;
}

int find_optimal_distance() {
    double best_time = DBL_MAX;
    int best_distance = 1;
    
    for (int d = 1; d <= 64; d *= 2) {
        double time = benchmark_prefetch_distance(d);
        if (time < best_time) {
            best_time = time;
            best_distance = d;
        }
    }
    return best_distance;
}
```

### 2. 预取策略的选择

#### A. 读取 vs 写入预取
```c
// 读取预取（更常用）
__builtin_prefetch(&A[i][k], 0, 3);  // rw=0

// 写入预取（用于输出矩阵）
__builtin_prefetch(&C[i][j], 1, 3);  // rw=1
```

#### B. 局部性级别的选择
```c
// 高局部性（数据将被多次使用）
__builtin_prefetch(addr, 0, 3);  // locality=3

// 中等局部性（数据可能被重复使用）
__builtin_prefetch(addr, 0, 2);  // locality=2

// 低局部性（数据只使用一次）
__builtin_prefetch(addr, 0, 1);  // locality=1

// 非时间性（绕过缓存）
__builtin_prefetch(addr, 0, 0);  // locality=0
```

### 3. 与其他优化技术的结合

#### A. 预取 + 循环展开
```c
void unrolled_prefetch_gemm(double *A, double *B, double *C, 
                           int M, int N, int K) {
    const int UNROLL = 4;
    const int PREFETCH_DIST = 8;
    
    for (int i = 0; i < M; i++) {
        for (int j = 0; j < N; j++) {
            for (int k = 0; k < K; k += UNROLL) {
                
                // 预取未来的数据
                if (k + PREFETCH_DIST < K) {
                    __builtin_prefetch(&A[i * K + k + PREFETCH_DIST], 0, 3);
                    __builtin_prefetch(&B[(k + PREFETCH_DIST) * N + j], 0, 3);
                }
                
                // 展开的计算循环
                C[i * N + j] += A[i * K + k] * B[k * N + j];
                if (k + 1 < K) C[i * N + j] += A[i * K + k + 1] * B[(k + 1) * N + j];
                if (k + 2 < K) C[i * N + j] += A[i * K + k + 2] * B[(k + 2) * N + j];
                if (k + 3 < K) C[i * N + j] += A[i * K + k + 3] * B[(k + 3) * N + j];
            }
        }
    }
}
```

#### B. 预取 + SIMD
```c
#include <immintrin.h>

void simd_prefetch_gemm(double *A, double *B, double *C, int M, int N, int K) {
    const int PREFETCH_DIST = 4;
    
    for (int i = 0; i < M; i++) {
        for (int j = 0; j < N; j += 4) {  // 处理 4 个元素
            __m256d c_vec = _mm256_setzero_pd();
            
            for (int k = 0; k < K; k++) {
                
                // 预取下几次迭代的数据
                if (k + PREFETCH_DIST < K) {
                    __builtin_prefetch(&A[i * K + k + PREFETCH_DIST], 0, 3);
                    __builtin_prefetch(&B[(k + PREFETCH_DIST) * N + j], 0, 3);
                }
                
                // SIMD 计算
                __m256d a_vec = _mm256_broadcast_sd(&A[i * K + k]);
                __m256d b_vec = _mm256_load_pd(&B[k * N + j]);
                c_vec = _mm256_fmadd_pd(a_vec, b_vec, c_vec);
            }
            
            _mm256_store_pd(&C[i * N + j], c_vec);
        }
    }
}
```

## 性能分析和调优

### 1. 性能测量指标

#### A. 缓存相关指标
```bash
# 使用 perf 工具测量缓存性能
perf stat -e cache-references,cache-misses,L1-dcache-loads,L1-dcache-load-misses \
    ./gemm_with_prefetch

# 关键指标：
# - L1 数据缓存失效率
# - L2/L3 缓存失效率
# - 内存带宽利用率
```

#### B. 预取效果评估
```c
void measure_prefetch_effectiveness() {
    // 测量有无预取的性能差异
    double time_without_prefetch = benchmark_basic_gemm();
    double time_with_prefetch = benchmark_prefetch_gemm();
    
    double speedup = time_without_prefetch / time_with_prefetch;
    printf("预取技术性能提升: %.2fx\n", speedup);
}
```

### 2. 常见的预取问题和解决方案

#### A. 过度预取
```c
// 问题：预取太多数据，污染缓存
// 解决：限制预取距离和频率

void controlled_prefetch_gemm(double *A, double *B, double *C, 
                             int M, int N, int K) {
    const int MAX_PREFETCH_DISTANCE = 16;
    const int PREFETCH_STRIDE = 4;  // 每 4 次迭代预取一次
    
    for (int i = 0; i < M; i++) {
        for (int j = 0; j < N; j++) {
            for (int k = 0; k < K; k++) {
                
                // 控制预取频率
                if (k % PREFETCH_STRIDE == 0 && 
                    k + MAX_PREFETCH_DISTANCE < K) {
                    __builtin_prefetch(&A[i * K + k + MAX_PREFETCH_DISTANCE], 0, 2);
                    __builtin_prefetch(&B[(k + MAX_PREFETCH_DISTANCE) * N + j], 0, 2);
                }
                
                C[i * N + j] += A[i * K + k] * B[k * N + j];
            }
        }
    }
}
```

#### B. 预取时机不当
```c
// 问题：预取太早或太晚
// 解决：动态调整预取距离

void dynamic_prefetch_gemm(double *A, double *B, double *C, 
                          int M, int N, int K) {
    int prefetch_distance = 8;  // 初始值
    double prev_performance = 0;
    
    for (int iteration = 0; iteration < 10; iteration++) {
        double start = get_time();
        
        // 使用当前预取距离运行 GEMM
        for (int i = 0; i < M; i++) {
            for (int j = 0; j < N; j++) {
                for (int k = 0; k < K; k++) {
                    if (k + prefetch_distance < K) {
                        __builtin_prefetch(&A[i * K + k + prefetch_distance], 0, 3);
                        __builtin_prefetch(&B[(k + prefetch_distance) * N + j], 0, 3);
                    }
                    C[i * N + j] += A[i * K + k] * B[k * N + j];
                }
            }
        }
        
        double current_performance = (get_time() - start);
        
        // 根据性能调整预取距离
        if (iteration > 0) {
            if (current_performance > prev_performance) {
                prefetch_distance = max(prefetch_distance / 2, 2);
            } else {
                prefetch_distance = min(prefetch_distance * 2, 32);
            }
        }
        
        prev_performance = current_performance;
    }
}
```

## 实际应用建议

### 1. 何时使用预取
- **大型矩阵**：内存访问成为瓶颈时
- **不规律访问模式**：硬件预取器效果不好时
- **内存带宽受限**：CPU 等待内存数据时

### 2. 预取参数的选择指南
```c
// 推荐的预取参数配置
struct PrefetchConfig {
    int l1_distance;    // 2-8 (L1 缓存)
    int l2_distance;    // 8-32 (L2 缓存)
    int l3_distance;    // 32-128 (L3 缓存)
    int locality;       // 1-3 (局部性级别)
    int stride;         // 1-4 (预取频率)
};

PrefetchConfig get_optimal_config(int matrix_size) {
    PrefetchConfig config;
    
    if (matrix_size < 1000) {
        // 小矩阵
        config = {4, 16, 64, 3, 1};
    } else if (matrix_size < 5000) {
        // 中等矩阵
        config = {6, 24, 96, 2, 2};
    } else {
        // 大矩阵
        config = {8, 32, 128, 1, 4};
    }
    
    return config;
}
```

## 总结

预取技术是 GEMM 优化的重要工具，可以有效隐藏内存访问延迟：

1. **理解原理**：掌握硬件和软件预取的区别
2. **分析访问模式**：识别 GEMM 中的内存访问瓶颈
3. **选择合适策略**：根据矩阵大小和硬件特性选择预取参数
4. **与其他技术结合**：配合分块、循环展开、SIMD 等技术
5. **性能测试验证**：通过实际测试确定最优配置

合理使用预取技术可以带来 10%-30% 的性能提升，但需要根据具体的硬件平台和应用场景进行调优。


---

## 相关笔记
<!-- 自动生成 -->

暂无相关笔记

