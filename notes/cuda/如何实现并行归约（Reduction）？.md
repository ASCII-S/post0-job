---
created: '2025-10-19'
last_reviewed: null
next_review: '2025-10-19'
review_count: 0
difficulty: medium
mastery_level: 0.0
tags:
- cuda
- cuda/如何实现并行归约（Reduction）？.md
related_outlines: []
---
# 如何实现并行归约（Reduction）？

## 面试标准答案

并行归约是将数组元素归约为单个值的并行算法。实现方法从简单到优化包括：**1) 交错寻址（存在分支分化）；2) 连续寻址（避免分化）；3) 顺序寻址（合并访存）；4) 展开循环（减少指令）；5) 完全展开+多元素处理（极致优化）**。关键优化点：避免线程分化、保证合并访存、利用Warp Shuffle、减少同步开销。Thrust库提供了高度优化的实现，实际开发中优先使用。

---

## 详细讲解

### 1. 归约问题定义

**任务：** 将N个元素归约为1个结果
```
输入：[a₀, a₁, a₂, ..., aₙ₋₁]
输出：a₀ ⊕ a₁ ⊕ a₂ ⊕ ... ⊕ aₙ₋₁

⊕ 可以是：求和、最大值、最小值、乘积等
```

**应用场景：**
- 求和、求平均
- 求最大/最小值
- 点积运算
- 范数计算

### 2. 实现演进

#### 2.1 版本1：交错寻址（Naive）

```cuda
__global__ void reduce_v1(int *g_idata, int *g_odata) {
    extern __shared__ int sdata[];
    
    unsigned int tid = threadIdx.x;
    unsigned int i = blockIdx.x * blockDim.x + threadIdx.x;
    sdata[tid] = g_idata[i];
    __syncthreads();
    
    // ❌ 问题：交错寻址导致严重的线程分化
    for (unsigned int s = 1; s < blockDim.x; s *= 2) {
        if (tid % (2*s) == 0) {  // 分支分化！
            sdata[tid] += sdata[tid + s];
        }
        __syncthreads();
    }
    
    if (tid == 0) g_odata[blockIdx.x] = sdata[0];
}
```

**问题：**
- `tid % (2*s) == 0` 导致严重warp分化
- 第一轮只有50%线程活跃，第二轮25%...
- 性能很差

#### 2.2 版本2：连续寻址

```cuda
__global__ void reduce_v2(int *g_idata, int *g_odata) {
    extern __shared__ int sdata[];
    
    unsigned int tid = threadIdx.x;
    unsigned int i = blockIdx.x * blockDim.x + threadIdx.x;
    sdata[tid] = g_idata[i];
    __syncthreads();
    
    // ✅ 改进：连续线程处理，减少分化
    for (unsigned int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (tid < s) {  // 更好的分支条件
            sdata[tid] += sdata[tid + s];
        }
        __syncthreads();
    }
    
    if (tid == 0) g_odata[blockIdx.x] = sdata[0];
}
```

**改进：**
- 前半部分线程活跃（连续的warp）
- 减少分支分化
- 性能提升~2倍

#### 2.3 版本3：顺序寻址

```cuda
__global__ void reduce_v3(int *g_idata, int *g_odata, unsigned int n) {
    extern __shared__ int sdata[];
    
    unsigned int tid = threadIdx.x;
    unsigned int i = blockIdx.x * (blockDim.x * 2) + threadIdx.x;
    
    // ✅ 每个线程加载2个元素并归约
    sdata[tid] = g_idata[i] + g_idata[i + blockDim.x];
    __syncthreads();
    
    // ✅ 顺序访问，合并访存
    for (unsigned int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (tid < s) {
            sdata[tid] += sdata[tid + s];
        }
        __syncthreads();
    }
    
    if (tid == 0) g_odata[blockIdx.x] = sdata[0];
}
```

**改进：**
- 减少一半block数量
- 全局内存加载合并
- 性能再提升~1.5倍

#### 2.4 版本4：展开最后的Warp

```cuda
// Warp内无需__syncthreads()
__device__ void warpReduce(volatile int* sdata, int tid) {
    sdata[tid] += sdata[tid + 32];
    sdata[tid] += sdata[tid + 16];
    sdata[tid] += sdata[tid + 8];
    sdata[tid] += sdata[tid + 4];
    sdata[tid] += sdata[tid + 2];
    sdata[tid] += sdata[tid + 1];
}

__global__ void reduce_v4(int *g_idata, int *g_odata, unsigned int n) {
    extern __shared__ int sdata[];
    
    unsigned int tid = threadIdx.x;
    unsigned int i = blockIdx.x * (blockDim.x * 2) + threadIdx.x;
    
    sdata[tid] = g_idata[i] + g_idata[i + blockDim.x];
    __syncthreads();
    
    // 归约到32个元素
    for (unsigned int s = blockDim.x / 2; s > 32; s >>= 1) {
        if (tid < s) {
            sdata[tid] += sdata[tid + s];
        }
        __syncthreads();
    }
    
    // ✅ 最后一个warp展开，无需同步
    if (tid < 32) warpReduce(sdata, tid);
    
    if (tid == 0) g_odata[blockIdx.x] = sdata[0];
}
```

**改进：**
- 减少6次`__syncthreads()`
- 性能提升~15%

#### 2.5 版本5：完全展开+模板

```cuda
template <unsigned int blockSize>
__device__ void warpReduce(volatile int* sdata, int tid) {
    if (blockSize >= 64) sdata[tid] += sdata[tid + 32];
    if (blockSize >= 32) sdata[tid] += sdata[tid + 16];
    if (blockSize >= 16) sdata[tid] += sdata[tid + 8];
    if (blockSize >= 8)  sdata[tid] += sdata[tid + 4];
    if (blockSize >= 4)  sdata[tid] += sdata[tid + 2];
    if (blockSize >= 2)  sdata[tid] += sdata[tid + 1];
}

template <unsigned int blockSize>
__global__ void reduce_v5(int *g_idata, int *g_odata, unsigned int n) {
    extern __shared__ int sdata[];
    
    unsigned int tid = threadIdx.x;
    unsigned int i = blockIdx.x * (blockSize * 2) + threadIdx.x;
    
    sdata[tid] = g_idata[i] + g_idata[i + blockSize];
    __syncthreads();
    
    // 完全展开循环（编译时确定）
    if (blockSize >= 512) {
        if (tid < 256) { sdata[tid] += sdata[tid + 256]; } __syncthreads();
    }
    if (blockSize >= 256) {
        if (tid < 128) { sdata[tid] += sdata[tid + 128]; } __syncthreads();
    }
    if (blockSize >= 128) {
        if (tid < 64) { sdata[tid] += sdata[tid + 64]; } __syncthreads();
    }
    
    if (tid < 32) warpReduce<blockSize>(sdata, tid);
    
    if (tid == 0) g_odata[blockIdx.x] = sdata[0];
}

// 调用
reduce_v5<256><<<grid, 256, 256*sizeof(int)>>>(d_in, d_out, n);
```

**改进：**
- 编译时展开，减少指令
- 消除循环开销
- 性能提升~10%

### 3. 使用Warp Shuffle（现代方法）

```cuda
__inline__ __device__
int warpReduceSum(int val) {
    for (int offset = 16; offset > 0; offset /= 2) {
        val += __shfl_down_sync(0xffffffff, val, offset);
    }
    return val;
}

__inline__ __device__
int blockReduceSum(int val) {
    static __shared__ int shared[32]; // 每个warp一个
    int lane = threadIdx.x % 32;
    int wid = threadIdx.x / 32;
    
    val = warpReduceSum(val);     // Warp内归约
    
    if (lane == 0) shared[wid] = val; // 每个warp的结果
    __syncthreads();
    
    // 最后一个warp归约所有warp的结果
    val = (threadIdx.x < blockDim.x / 32) ? shared[lane] : 0;
    if (wid == 0) val = warpReduceSum(val);
    
    return val;
}

__global__ void reduce_shuffle(int *in, int *out, int N) {
    int sum = 0;
    for (int i = blockIdx.x * blockDim.x + threadIdx.x; 
         i < N; 
         i += blockDim.x * gridDim.x) {
        sum += in[i];
    }
    
    sum = blockReduceSum(sum);
    
    if (threadIdx.x == 0)
        atomicAdd(out, sum);
}
```

**优势：**
- 无需共享内存（减少bank conflict）
- 更低延迟（寄存器级）
- 代码简洁

### 4. 多级归约策略

```cuda
// 第一级：Block内归约
__global__ void reduceLevel1(int *in, int *out, int N) {
    // ... Block内归约
    if (threadIdx.x == 0)
        out[blockIdx.x] = result;
}

// 第二级：归约Block的结果
__global__ void reduceLevel2(int *in, int *out, int N) {
    // 如果Block数量少，可以在一个Block内完成
}

// Host端调用
int numBlocks = (N + 255) / 256;
reduceLevel1<<<numBlocks, 256>>>(d_in, d_temp, N);
reduceLevel2<<<1, 256>>>(d_temp, d_out, numBlocks);
```

### 5. 性能对比

**测试：1M个元素求和（V100）**

| 版本           | 时间(μs) | 带宽(GB/s) | 相对v1 |
| -------------- | -------- | ---------- | ------ |
| v1 交错寻址    | 120      | 33         | 1×     |
| v2 连续寻址    | 65       | 62         | 1.8×   |
| v3 顺序访问    | 42       | 95         | 2.9×   |
| v4 Warp展开    | 36       | 111        | 3.3×   |
| v5 完全展开    | 32       | 125        | 3.8×   |
| Shuffle版本    | 28       | 143        | 4.3×   |
| Thrust::reduce | 25       | 160        | 4.8×   |

### 6. 使用Thrust（推荐）

```cpp
#include <thrust/device_vector.h>
#include <thrust/reduce.h>

// 求和
thrust::device_vector<int> d_vec(N);
int sum = thrust::reduce(d_vec.begin(), d_vec.end(), 0, thrust::plus<int>());

// 求最大值
int max_val = thrust::reduce(d_vec.begin(), d_vec.end(), 
                              INT_MIN, thrust::maximum<int>());

// 自定义归约操作
struct custom_op {
    __device__ int operator()(int a, int b) {
        return a * b + 1;
    }
};
int result = thrust::reduce(d_vec.begin(), d_vec.end(), 1, custom_op());
```

### 7. 实际应用示例

#### 7.1 向量点积

```cuda
__global__ void dot_product(float *a, float *b, float *out, int N) {
    __shared__ float partial[256];
    
    int tid = threadIdx.x;
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    
    // 计算部分积并加载到共享内存
    partial[tid] = (i < N) ? a[i] * b[i] : 0;
    __syncthreads();
    
    // Block内归约
    for (int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (tid < s) {
            partial[tid] += partial[tid + s];
        }
        __syncthreads();
    }
    
    if (tid == 0)
        atomicAdd(out, partial[0]);
}
```

#### 7.2 求数组最大值

```cuda
__global__ void find_max(float *in, float *out, int N) {
    __shared__ float smax[256];
    
    int tid = threadIdx.x;
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    
    smax[tid] = (i < N) ? in[i] : -FLT_MAX;
    __syncthreads();
    
    // 归约找最大
    for (int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (tid < s) {
            smax[tid] = fmaxf(smax[tid], smax[tid + s]);
        }
        __syncthreads();
    }
    
    if (tid == 0)
        atomicMax((int*)out, __float_as_int(smax[0]));
}
```

### 8. 优化要点总结

| 优化技术         | 效果     | 难度 |
| ---------------- | -------- | ---- |
| 避免线程分化     | 1.8×     | 低   |
| 合并全局内存访问 | 1.6×     | 中   |
| 减少同步次数     | 1.2×     | 中   |
| 展开循环         | 1.1×     | 低   |
| 使用Warp Shuffle | 1.3×     | 中   |
| 多元素/线程      | 1.5×     | 低   |
| **组合优化**     | **4-5×** | -    |

### 9. 常见陷阱

**1. 忘记处理非2的幂次大小**
```cuda
// ❌ 错误
for (int s = blockDim.x / 2; s > 0; s >>= 1)

// ✅ 正确
for (int s = 1; s < blockDim.x; s *= 2)
    if (tid % (2*s) == 0 && tid + s < n)
```

**2. 共享内存大小计算错误**
```cuda
// ✅ 正确分配
int smem_size = blockDim.x * sizeof(int);
kernel<<<grid, block, smem_size>>>(...)
```

**3. 多次归约时未重置结果**
```cuda
// ✅ 每次归约前清零
cudaMemset(d_result, 0, sizeof(int));
```

### 10. 最佳实践

| 建议                        | 说明               |
| --------------------------- | ------------------ |
| ✅ 优先使用Thrust            | 简单、高效、维护好 |
| ✅ 使用Warp Shuffle          | 现代GPU性能最好    |
| ✅ 多元素/线程               | 提高occupancy      |
| ✅ 完全展开最后几轮          | 减少同步和分支     |
| ✅ 使用模板避免运行时分支    | 编译时优化         |
| ❌ 避免atomicAdd作为主要方法 | 仅用于最终结果     |
| ❌ 不要忽视边界情况          | 处理非对齐大小     |

### 11. 记忆口诀

**"归约算法分五版，性能一版比一版；避免分化合并访存，展开循环减同步；Warp Shuffle最现代，Thrust库最简单；实战优先用库函数，手写优化为学习"**


---

## 相关笔记
<!-- 自动生成 -->

暂无相关笔记

