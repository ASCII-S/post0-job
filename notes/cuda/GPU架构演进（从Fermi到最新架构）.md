---
created: '2025-10-19'
last_reviewed: null
next_review: '2025-10-19'
review_count: 0
difficulty: medium
mastery_level: 0.0
tags:
- cuda
- cuda/GPU架构演进（从Fermi到最新架构）.md
related_outlines: []
---
# GPU架构演进（从Fermi到最新架构）

## 面试标准答案

GPU架构的主要演进路径：

**1. Fermi (2010) - GF100/GF110**
- 首次实现完整的CUDA架构，支持C++
- 引入统一L1/L2缓存层次
- 支持ECC内存保护
- 计算能力2.0/2.1

**2. Kepler (2012) - GK104/GK110** 
- 大幅提升能效比和并行性
- 引入动态并行性（Dynamic Parallelism）
- Hyper-Q技术提升CPU-GPU并发
- 计算能力3.0/3.5

**3. Maxwell (2014) - GM10x/GM20x**
- 重新设计的SM架构，提升能效
- 统一虚拟内存（Unified Virtual Memory）
- 动态并行性改进
- 计算能力5.0/5.2

**4. Pascal (2016) - GP100/GP104**
- 首次支持16nm工艺
- 引入HBM2高带宽内存
- NVLink高速互连技术
- 计算能力6.0/6.1

**5. Volta (2017) - GV100**
- 革命性引入Tensor Core
- 独立线程调度（Independent Thread Scheduling）
- 混合精度计算支持
- 计算能力7.0

**6. Turing (2018) - TU10x**
- RT Core硬件光线追踪
- 第二代Tensor Core
- 可变率着色（VRS）
- 计算能力7.5

**7. Ampere (2020) - GA100/GA102**
- 第三代Tensor Core，支持更多数据类型
- 稀疏性支持（Sparsity）
- Multi-Instance GPU (MIG)
- 计算能力8.0/8.6

**8. Ada Lovelace (2022) - AD10x**
- 第三代RT Core
- 第四代Tensor Core
- AV1编码支持
- 计算能力8.9

**9. Hopper (2022) - GH100**
- Transformer Engine
- 线程块集群（Thread Block Clusters）
- DPX指令集
- 计算能力9.0

---

## 深度技术解析

### Fermi架构：CUDA的成熟之作

#### 技术突破点
Fermi标志着GPU从图形专用处理器向通用并行计算平台的真正转变：

**统一架构设计**
- 统一的shader架构，消除了vertex/pixel/geometry shader的区别
- 512个CUDA Core，采用32个SM设计
- 每个SM包含32个CUDA Core，支持32个并发线程（warp）

**内存子系统革新**
```
L2 Cache: 768KB, 所有SM共享
L1 Cache: 16KB或48KB可配置（与shared memory共享64KB）
Shared Memory: 48KB或16KB可配置
Register File: 32,768 × 32-bit per SM
```

**计算精度提升**
- 首次支持IEEE 754-2008双精度浮点
- FMA（融合乘加）指令提升数值精度
- 支持64位整数运算

### Kepler架构：能效与并行性的突破

#### 核心创新技术

**SMX架构重新设计**
- 每个SMX包含192个CUDA Core（比Fermi的32个大幅增加）
- 32个特殊函数单元（SFU）
- 32个Load/Store单元

**动态并行性（Dynamic Parallelism）**
```cpp
// Kepler引入的GPU端kernel启动能力
__global__ void parent_kernel() {
    // GPU可以直接启动子kernel
    child_kernel<<<grid, block>>>();
    cudaDeviceSynchronize(); // GPU端同步
}
```

**Hyper-Q技术**
- 支持32个硬件工作队列（vs Fermi的1个）
- CPU多个线程可以同时向GPU提交工作
- 提升CPU-GPU并发效率

### Maxwell架构：能效革命

#### 架构优化重点

**SMM（Maxwell SM）设计**
- 每个SMM包含128个CUDA Core
- 重新划分为4个32-core处理块
- 每个处理块有独立的指令缓冲和调度器

**内存系统改进**
```
L2 Cache: 2MB（GM204）
L1 Cache: 24KB专用数据缓存
Texture Cache: 12KB专用纹理缓存
Shared Memory: 64KB可配置
```

**统一虚拟内存（UVM）**
- CPU和GPU共享统一的虚拟地址空间
- 自动页面迁移和数据一致性
- 简化异构编程模型

### Pascal架构：高性能计算的里程碑

#### 重大技术进步

**GP100的HPC设计**
- 3584个CUDA Core
- 56个SM，每个64个CUDA Core
- 双精度性能大幅提升（4.7 TFLOPS）

**HBM2内存技术**
```
内存带宽: 720 GB/s（vs GDDR5的 ~300 GB/s）
内存容量: 16GB
内存接口: 4096-bit
功耗效率: 比GDDR5提升2.5倍
```

**NVLink互连技术**
- 40GB/s双向带宽（vs PCIe 3.0的16GB/s）
- 支持GPU-GPU和GPU-CPU高速互连
- 为多GPU系统奠定基础

### Volta架构：AI计算的开端

#### 革命性特性

**Tensor Core的引入**
- 专用的4×4矩阵乘法单元
- 混合精度计算：FP16输入，FP32累加
- 单SM可达125 TFLOPS（Tensor操作）

**独立线程调度**
```
传统SIMT: 32个线程lockstep执行
Volta ITS: 每个线程独立PC和调用栈
优势: 支持更复杂的线程分歧和同步
```

**新的内存层次**
```
L1 Cache: 128KB per SM（可配置）
Shared Memory: 96KB per SM（vs Pascal的64KB）
L2 Cache: 6MB（vs Pascal的4MB）
```

### Turing架构：实时渲染的突破

#### 专用硬件单元

**RT Core光线追踪**
- 硬件加速BVH遍历和三角形相交测试
- 10 Giga Rays/sec性能
- 实时光线追踪游戏成为可能

**第二代Tensor Core**
- 支持INT8和INT4精度
- 稀疏计算优化
- 2.5倍AI推理性能提升

**可变率着色（VRS）**
- 根据场景复杂度动态调整着色率
- 最多节省50%的着色计算
- 保持视觉质量的同时提升性能

### Ampere架构：AI训练的黄金标准

#### 第三代Tensor Core进化

**多精度支持矩阵**
```
FP64: 科学计算
TF32: AI训练（19.5 TFLOPS）
BFLOAT16: AI训练优化
FP16: AI推理
INT8/INT4: 量化推理
```

**稀疏性支持**
- 2:4结构化稀疏（50%稀疏度）
- 硬件加速稀疏矩阵乘法
- 理论上2倍AI推理加速

**MIG技术**
- 单GPU划分为多个独立实例
- 硬件级别的资源隔离
- 提升多租户环境的利用率

### Hopper架构：Transformer时代的专用设计

#### Transformer Engine

**FP8数据类型**
- E4M3和E5M2两种FP8格式
- 相比FP16减少内存和带宽需求
- 专门为Transformer模型优化

**线程块集群（Thread Block Clusters）**
```cpp
// Hopper新特性：跨SM的线程块协作
__cluster_dims__(2, 2, 1) // 2×2 SM集群
__global__ void cluster_kernel() {
    // 跨SM的shared memory和同步
    cluster_sync();
}
```

**DPX指令集**
- 动态编程加速
- Smith-Waterman算法硬件加速
- 生物信息学计算优化

### 架构演进的核心趋势

#### 计算能力发展轨迹

**通用计算性能**
```
Fermi GF110:    1.5 TFLOPS (FP32)
Kepler GK110:   3.5 TFLOPS (FP32)  
Maxwell GM200:  6.1 TFLOPS (FP32)
Pascal GP100:   10.6 TFLOPS (FP32)
Volta GV100:    15.7 TFLOPS (FP32)
Ampere GA100:   19.5 TFLOPS (FP32)
Hopper GH100:   67 TFLOPS (FP32)
```

**AI专用性能**
```
Volta GV100:    125 TFLOPS (Tensor)
Turing TU102:   130 TFLOPS (Tensor)  
Ampere GA100:   312 TFLOPS (BFLOAT16)
Hopper GH100:   1000 TFLOPS (FP8)
```

#### 内存技术进步

**内存带宽演进**
- Fermi: ~180 GB/s (GDDR5)
- Kepler: ~336 GB/s (GDDR5)
- Maxwell: ~336 GB/s (GDDR5)
- Pascal: ~720 GB/s (HBM2)
- Volta: ~900 GB/s (HBM2)
- Ampere: ~1.6 TB/s (HBM2e)
- Hopper: ~3.4 TB/s (HBM3)

#### 互连技术发展

**NVLink演进**
```
Pascal: NVLink 1.0 - 40 GB/s
Volta: NVLink 2.0 - 50 GB/s  
Ampere: NVLink 3.0 - 100 GB/s
Hopper: NVLink 4.0 - 200 GB/s
```

### 面向未来的技术方向

#### 计算模式多样化
- **通用计算**：传统CUDA Core持续优化
- **AI计算**：Tensor Core专业化深化
- **图形计算**：RT Core实时光线追踪
- **科学计算**：双精度和特殊函数优化

#### 软件栈协同演进
- **编译器优化**：NVCC支持新指令集
- **库函数进化**：cuBLAS、cuDNN利用新硬件特性
- **框架集成**：PyTorch、TensorFlow自动利用Tensor Core

这种架构演进展现了NVIDIA在不同时代对计算需求的敏锐洞察，从通用并行计算到AI专用加速，每一代都针对当时的主流应用进行了深度优化。

---

## 相关笔记
<!-- 自动生成 -->

- [CUDA_Core,Tensor_Core的区别](notes/cuda/CUDA_Core,Tensor_Core的区别.md) - 相似度: 31% | 标签: cuda, cuda/CUDA_Core,Tensor_Core的区别.md

