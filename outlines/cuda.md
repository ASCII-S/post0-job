# CUDA面试大纲

## 1. CUDA核心概念 (必考基础，15-20分钟)

### 1.1 CUDA基础认知

#### **什么是CUDA？**
  - [CUDA与传统CPU计算的区别](../notes/cuda/CUDA与传统CPU计算的区别.md)
  - GPU并行计算的优势和局限性
  - CUDA的全称是什么？CUDA主要用于解决什么类型的问题？
  - GPU并行计算相比CPU计算有哪些优势？
  - GPU计算不适合哪些场景？为什么？
  - 什么样的算法适合用GPU加速？

#### **GPU硬件架构**
  - [GPU架构演进（从Fermi到最新架构）](../notes/cuda/GPU架构演进（从Fermi到最新架构）.md)
  - [SM（Streaming Multiprocessor）的概念和作用](../notes/cuda/SM（Streaming_Multiprocessor）的概念和作用.md)
  - [CUDA Core、Tensor Core的区别](../notes/cuda/CUDA_Core,Tensor_Core的区别.md)
  - SM（Streaming Multiprocessor）在GPU中扮演什么角色？
  - 一个SM包含哪些主要组件？
  - CUDA Core和Tensor Core有什么区别？各自适用于什么场景？
  - GPU的计算能力（Compute Capability）是什么？如何影响程序编写？

#### **内存层次结构**
  - [内存层次结构（Global、Shared、Constant、Texture、Register）](../notes/cuda/内存层次结构（Global、Shared、Constant、Texture、Register）.md)
  - [寄存器使用](../notes/cuda/寄存器使用.md)
  - CUDA中有哪些类型的内存？它们的访问速度如何排序？
  - Global Memory、Shared Memory、Register的生命周期和作用域分别是什么？
  - 不同类型内存的容量大小通常是多少？
  - Constant Memory和Texture Memory有什么特点？适合存储什么数据？
  - 寄存器溢出（Register Spilling）是什么？会带来什么影响？

### 1.2 CUDA编程模型基础

#### **线程层次结构**
  - [线程层次结构](../notes/cuda/线程层次结构.md)
  - [三级线程组织结构](../notes/cuda/线程层次结构.md)
  - [线程索引计算](../notes/cuda/线程索引计算.md)
  - [一维、二维、三维网格的使用场景](../notes/cuda/一维、二维、三维网格的使用场景.md)
  - Grid、Block、Thread三者是什么关系？
  - 如何计算线程的全局索引？给出一维、二维的计算公式
  - Block的大小有什么限制？如何选择合适的Block大小？
  - 什么时候使用二维或三维的Grid配置？举例说明
  - threadIdx、blockIdx、blockDim、gridDim分别表示什么？

#### **Warp执行模型**
  - Warp是什么？一个Warp包含多少个线程？
  - [Warp的大小和执行机制](../notes/cuda/Warp的大小和执行机制.md)
  - SIMT执行模型是如何工作的？
  - 什么是Warp分歧（Divergence）？它如何影响性能？
  - 如何判断代码是否会产生Warp分歧？
  - 为什么Block大小建议设置为32的倍数？

### 1.3 CUDA软件栈与工具链

#### **CUDA软件层次**
  - [CUDA_Driver_vs_CUDA_Runtime](../notes/cuda/CUDA_Driver_vs_CUDA_Runtime.md)
  - [CUDAToolkit组成部分](../notes/cuda/CUDAToolkit组成部分.md)
  - [NVCC编译器的作用](../notes/cuda/NVCC编译器的作用.md)
  - [PTX（Parallel Thread Execution）中间代码](../notes/cuda/PTX（Parallel_Thread_Execution）中间代码.md)
  - 如何为不同的GPU架构编译CUDA程序？

#### **计算能力与兼容性**
  - [Compute Capability](../notes/cuda/Compute_Capability.md)
  - 什么是Compute Capability？当前主流GPU的计算能力是多少？
  - 不同计算能力的GPU有哪些特性差异？
  - 如何查询GPU的计算能力？
  - 向前兼容和向后兼容分别是什么意思？

## 2. CUDA编程实践 (编程能力考察，25-30分钟)

### 2.1 Kernel函数编写

#### **Kernel函数基础语法**
  - [限定符](../notes/cuda/限定符.md)
  - [Kernel启动配置](../notes/cuda/Kernel启动配置.md)
  - `__global__`、`__device__`、`__host__`限定符分别有什么作用？
  - 如何定义和调用一个Kernel函数？
  - Kernel启动配置`<<<gridDim, blockDim, sharedMem, stream>>>`各参数的含义是什么？
  - Kernel函数有哪些限制？（如返回值、递归等）
  - 如何在Kernel中使用共享内存？静态和动态分配有什么区别？

#### **动态并行性**
  - [动态并行性（Dynamic Parallelism）](../notes/cuda/动态并行性.md)
  - 什么是动态并行性？它解决了什么问题？
  - 如何在Kernel内部启动新的Kernel？
  - 动态并行性有哪些限制和注意事项？
  - 什么场景下适合使用动态并行性？

### 2.2 内存管理与数据传输

#### **基础内存管理API**
  - [`cudaMalloc`和`malloc`有什么区别？](../notes/cuda/`cudaMalloc`和`malloc`有什么区别？.md)
  - `cudaMemcpy`的不同拷贝方向（H2D、D2H、D2D）如何指定？
  - 如何正确地释放GPU内存？内存泄漏如何检测？
  - `cudaMallocHost`（固定内存）有什么优势？什么时候使用？
  - 如何分配二维或三维的GPU内存？`cudaMallocPitch`和`cudaMalloc3D`的作用是什么？

#### **统一内存**
  - 什么是统一内存（Unified Memory）？
  - 使用统一内存有什么优缺点？
  - `cudaMallocManaged`分配的内存如何在CPU和GPU之间迁移？
  - 如何使用`cudaMemPrefetchAsync`优化统一内存性能？
  - 统一内存适合什么场景？什么时候不应该使用？

### 2.3 内存访问模式

#### **全局内存访问**
  - [什么是Global Memory](../notes/cuda/Global_Memory.md)
  - 什么是内存合并访问（Coalesced Access）？
  - 如何判断内存访问是否合并？
  - 未对齐的内存访问会带来多大的性能损失？
  - [如何优化跨步访问（Strided Access）？](../notes/cuda/如何优化跨步访问（Strided_Access）？.md)
  - [内存事务（Memory Transaction）是如何工作的？](../notes/cuda/内存事务（Memory_Transaction）是如何工作的？.md)

#### **共享内存使用**
  - [什么是Shared Memory](../notes/cuda/Shared_Memory.md)
  - 共享内存相比全局内存有什么优势？
  - [什么是Bank Conflict？它如何影响性能？](../notes/cuda/什么是Bank_Conflict？它如何影响性能？.md)
  - [如何避免Bank Conflict？举例说明](../notes/cuda/如何避免Bank_Conflict？举例说明.md)
  - 共享内存的大小限制是多少？如何在共享内存和寄存器之间权衡？
  - 共享内存的生命周期和可见范围是什么？

#### **常量内存和纹理内存**
  - [常量内存和纹理内存](../notes/cuda/常量内存和纹理内存.md)
  - 常量内存有什么特点？适合存储什么数据？
  - 常量内存的缓存机制是怎样的？
  - 纹理内存相比全局内存有哪些优势？
  - 纹理内存的插值和过滤功能如何使用？
  - 什么场景下使用纹理内存能带来性能提升？

### 2.4 同步与通信

#### **线程同步**
  - [`__syncthreads()`的作用是什么？什么时候必须使用？](../notes/cuda/`__syncthreads()`的作用是什么？什么时候必须使用？.md)
  - [`__syncthreads()`有什么使用限制？（如条件语句中）](../notes/cuda/`__syncthreads()`有什么使用限制？（如条件语句中）.md)
  - [如何同步整个Grid？Grid级别的同步有什么方法？](../notes/cuda/如何同步整个Grid？Grid级别的同步有什么方法？.md)
  - [Warp内的线程是否需要显式同步？](../notes/cuda/Warp内的线程是否需要显式同步？.md)
  - `__syncwarp()`的作用是什么？

#### **原子操作**
  - CUDA提供了哪些原子操作函数？
  - 原子操作如何保证线程安全？
  - 原子操作对性能有什么影响？什么时候使用原子操作？
  - 如何减少原子操作的性能开销？
  - `atomicAdd`、`atomicCAS`等函数的使用场景是什么？

### 2.5 基础算法实现

#### **向量加法**
  - 如何实现一个基本的向量加法Kernel？
  - 如何处理向量长度不是Block大小整数倍的情况？
  - 如何优化向量加法的内存访问模式？
  - 向量加法的性能瓶颈通常在哪里？

#### **矩阵乘法**
  - 朴素矩阵乘法的实现思路是什么？
  - 如何使用共享内存优化矩阵乘法？
  - 分块（Tiling）矩阵乘法的原理是什么？
  - 如何选择合适的Tile大小？
  - 矩阵乘法能达到的理论性能上限如何计算？

## 3. 性能优化技术 (核心重点，30-35分钟)

### 3.1 内存访问优化

#### **内存合并访问优化**
  - [内存带宽优化](../notes/cuda/内存带宽优化.md)
  - 如何测量内存带宽利用率？
  - 什么样的访问模式能实现最佳的内存合并？
  - 如何重组数据结构以优化内存访问？（AoS vs SoA）
  - 内存对齐对性能有多大影响？
  - 如何使用Profiler检测未合并的内存访问？

#### **共享内存Bank Conflict优化**
  - Bank Conflict是如何产生的？
  - 如何通过Padding避免Bank Conflict？
  - 多路Bank Conflict相比单路有多大性能差异？
  - 如何使用Profiler检测Bank Conflict？
  - 不同GPU架构的Bank配置有何差异？

#### **缓存利用优化**
  - L1/L2缓存在CUDA中如何工作？
  - 如何提高缓存命中率？
  - `__ldg()`函数的作用是什么？
  - 如何选择合适的内存访问模式以利用缓存？
  - 缓存一致性在多SM场景下如何保证？

### 3.2 占用率优化

#### **占用率基础**
  - [占用率（Occupancy）优化](../notes/cuda/占用率（Occupancy）优化.md)
  - [理论占用率 vs 实际占用率](../notes/cuda/理论占用率_vs_实际占用率.md)
  - [影响占用率的因素](../notes/cuda/影响占用率的因素.md)
  - [CUDA Occupancy Calculator使用](../notes/cuda/CUDA_Occupancy_Calculator使用.md)
  - 什么是占用率（Occupancy）？如何计算？
  - 理论占用率和实际占用率有什么区别？
  - 哪些因素会限制占用率？（寄存器、共享内存、Block大小）
  - 占用率越高性能就越好吗？什么时候低占用率也能有高性能？
  - 如何使用CUDA Occupancy Calculator？

#### **资源使用优化**
  - [寄存器使用](../notes/cuda/寄存器使用.md)
  - 如何查看Kernel使用了多少寄存器？
  - 如何通过编译选项限制寄存器使用？
  - 寄存器压力如何影响占用率？
  - 寄存器溢出到Local Memory会带来多大性能损失？
  - 如何在寄存器使用和占用率之间权衡？

#### **Block大小选择**
  - 如何选择最优的Block大小？
  - Block大小对占用率有什么影响？
  - 为什么Block大小通常是128或256？
  - 动态调整Block大小的策略是什么？
  - 不同算法的最优Block大小是否相同？

### 3.3 Warp级别优化

#### **避免Warp分歧**
  - [如何避免分支分歧](../notes/cuda/如何避免分支分歧.md)
  - 哪些代码会导致Warp分歧？
  - 如何重构代码以减少分支分歧？
  - 使用位操作替代条件语句的技巧
  - 如何使用Profiler检测Warp分歧？
  - 分歧对性能的影响有多大？如何量化？

#### **Warp级别原语**
  - Shuffle指令是什么？有哪些类型？
  - 如何使用Shuffle优化归约操作？
  - Warp级别的投票函数（`__ballot`、`__any`、`__all`）的作用是什么？
  - 如何使用Warp原语避免共享内存？
  - Warp原语的性能优势在哪里？

### 3.4 计算与访存优化

#### **计算强度优化**
- [计算强度优化](../notes/cuda/计算强度优化.md)
  - 什么是算术强度（Arithmetic Intensity）？
  - 如何判断程序是计算密集型还是访存密集型？
  - 如何提高算术强度？
  - Roofline模型是什么？如何使用？
  - 如何通过融合操作提高计算强度？

#### **指令级并行优化**
  - 什么是指令级并行（ILP）？
  - 如何通过循环展开提高ILP？
  - 如何避免数据依赖？
  - 指令吞吐量如何影响性能？
  - 不同类型指令的吞吐量差异是什么？

### 3.5 性能分析与调试

#### **性能分析工具**
  - [NVIDIA_Profiler工具](../notes/cuda/NVIDIA_Profiler工具.md)
  - Nsight Systems和Nsight Compute有什么区别？
  - 如何使用Nsight Systems分析时间线？
  - 如何使用Nsight Compute进行Kernel级别分析？
  - 哪些性能指标最重要？如何解读？
  - 如何识别性能瓶颈？

#### **性能瓶颈识别**
  - [GPU利用率分析](../notes/cuda/GPU利用率分析.md)
  - [内存瓶颈识别](../notes/cuda/内存瓶颈识别.md)
  - [计算瓶颈识别](../notes/cuda/计算瓶颈识别.md)
  - 如何判断程序是受内存带宽限制还是计算能力限制？
  - GPU利用率低的常见原因有哪些？
  - 如何分析Kernel的执行时间分布？
  - 如何识别CPU-GPU传输瓶颈？
  - 性能计数器（Performance Counter）如何使用？

#### **调试技术**
  - cuda-gdb的基本使用方法
  - CUDA Memcheck如何检测内存错误？
  - 常见的CUDA运行时错误有哪些？如何处理？
  - 如何进行CUDA错误检查？错误检查宏如何编写？
  - 如何调试异步错误？

## 4. 高级特性与工具 (进阶内容，20-25分钟)

### 4.1 CUDA流与异步执行

#### **CUDA流基础**
  - [流的定义和执行顺序](../notes/cuda/流的定义和执行顺序.md)
  - [默认流 vs 非默认流](../notes/cuda/默认流_vs_非默认流.md)
  - [流的创建和销毁](../notes/cuda/流的创建和销毁.md)
  - 什么是CUDA Stream？为什么需要Stream？
  - 默认流（NULL Stream）和非默认流有什么区别？
  - 如何创建和销毁Stream？
  - Stream的执行顺序是怎样的？
  - 不同Stream之间的操作能并发执行吗？

#### **异步执行模型**
  - [异步执行模型](../notes/cuda/异步执行模型.md)
  - 哪些CUDA操作是异步的？哪些是同步的？
  - `cudaMemcpyAsync`和`cudaMemcpy`有什么区别？
  - 如何实现CPU-GPU并发执行？
  - Kernel启动是异步的吗？如何验证？
  - 异步执行的错误处理有什么特殊之处？

#### **流同步机制**
  - `cudaStreamSynchronize()`和`cudaDeviceSynchronize()`有什么区别？
  - Event的作用是什么？如何使用Event进行同步？
  - 如何使用Event测量Kernel执行时间？
  - 如何实现Stream之间的依赖关系？
  - `cudaStreamWaitEvent()`如何使用？

#### **重叠计算与传输**
  - [重叠计算与数据传输](../notes/cuda/重叠计算与数据传输.md)
  - 如何实现数据传输和计算的重叠？
  - 双缓冲（Double Buffering）技术是什么？
  - 如何设计多Stream的Pipeline？
  - 重叠能带来多大的性能提升？
  - 固定内存在重叠中的作用是什么？

### 4.2 CUDA库与生态系统

#### **核心数学库**
  - [cuBLAS提供了哪些功能？如何使用？](../notes/cuda/cuBLAS提供了哪些功能？如何使用？.md)
  - cuBLAS的性能优势在哪里？
  - cuFFT如何使用？批处理FFT如何实现？
  - cuSPARSE支持哪些稀疏矩阵格式？
  - [如何选择合适的CUDA库函数？](../notes/cuda/如何选择合适的CUDA库函数？.md)

#### **深度学习库**
  - [cuDNN提供了哪些深度学习算子？](../notes/cuda/cuDNN提供了哪些深度学习算子？.md)
  - cuDNN的卷积算法有哪些？如何选择？
  - Tensor Core如何通过cuDNN使用？
  - cuDNN的性能调优技巧有哪些？
  - 如何集成cuDNN到自定义应用？

#### **并行算法库**
  - Thrust库的特点是什么？
  - 如何使用Thrust实现并行归约？
  - Thrust的性能如何？与手写Kernel相比？
  - CUB库和Thrust有什么区别？
  - [什么时候使用库函数，什么时候自己写Kernel？](../notes/cuda/什么时候使用库函数，什么时候自己写Kernel？.md)

### 4.3 多GPU编程

#### **多GPU基础**
  - 如何枚举和选择GPU设备？
  - `cudaSetDevice()`如何使用？
  - 如何在多个GPU之间分配任务？
  - 多GPU的内存管理有什么特殊之处？
  - Peer-to-Peer内存访问是什么？如何启用？

#### **多GPU通信**
  - [NCCL库的作用是什么？](../notes/cuda/NCCL库的作用是什么？.md)
  - [NCCL支持哪些集合通信操作？](../notes/cuda/NCCL支持哪些集合通信操作？.md)
  - [如何使用NCCL进行All-Reduce操作？](../notes/cuda/如何使用NCCL进行All-Reduce操作？.md)
  - [多GPU通信的性能瓶颈在哪里？](../notes/cuda/多GPU通信的性能瓶颈在哪里？.md)
  - [NVLink和PCIe在多GPU通信中的区别？](../notes/cuda/NVLink和PCIe在多GPU通信中的区别？.md)

#### **多GPU编程策略**
  - 数据并行和模型并行有什么区别？
  - 如何实现多GPU的负载均衡？
  - 多GPU扩展性如何评估？
  - 多GPU编程的常见陷阱有哪些？
  - 如何在多GPU之间同步？

### 4.4 Tensor Core与混合精度

#### **Tensor Core基础**
  - [Tensor Core是什么？支持哪些数据类型？](../notes/cuda/Tensor_Core是什么？支持哪些数据类型？.md)
  - [如何在代码中使用Tensor Core？](../notes/cuda/如何在代码中使用Tensor_Core？.md)
  - Tensor Core相比CUDA Core有多大性能优势？
  - 哪些GPU架构支持Tensor Core？
  - wmma（Warp Matrix Multiply Accumulate）API如何使用？

#### **混合精度计算**
  - [什么是混合精度计算？为什么使用？](../notes/cuda/什么是混合精度计算？为什么使用？.md)
  - FP16、BF16、TF32的区别是什么？
  - 混合精度训练的原理是什么？
  - 如何避免混合精度带来的精度损失？
  - 自动混合精度（AMP）如何工作？

## 5. 实战问题解决 (项目经验，15-20分钟)

### 5.1 典型算法优化案例

#### **归约操作优化**
  - [如何实现并行归约（Reduction）？](../notes/cuda/如何实现并行归约（Reduction）？.md)
  - 归约操作的常见优化技巧有哪些？
  - 如何使用Shuffle指令优化归约？
  - 如何避免归约中的Warp分歧？
  - 归约操作的性能上限是多少？

#### **前缀和/扫描算法**
  - 什么是前缀和（Prefix Sum）？有哪些应用？
  - [如何实现并行前缀和？](../notes/cuda/如何实现并行前缀和？.md)
  - Blelloch扫描算法的原理是什么？
  - 如何处理大数组的多级扫描？
  - 包含式扫描和排他式扫描有什么区别？

#### **卷积操作优化**
  - 直接卷积和FFT卷积如何选择？
  - im2col算法的原理是什么？
  - Winograd卷积算法的优势在哪里？
  - 如何优化卷积的内存访问？
  - 可分离卷积如何实现和优化？

#### **矩阵矩阵乘优化**
  - [朴素GEMM实现的主要性能瓶颈在哪里？](../notes/cuda/朴素GEMM实现的主要性能瓶颈在哪里？.md)
  - [如何使用分块（Tiling）优化矩阵乘法？](../notes/cuda/如何使用分块（Tiling）优化矩阵乘法？.md)
  - [共享内存在GEMM优化中起什么作用？](../notes/cuda/共享内存在GEMM优化中起什么作用？.md)
  - [如何避免shared memory的bank conflict？](../notes/cuda/如何避免shared_memory的bank_conflict？.md)
  - [向量化访存（vectorized load/store）如何提升GEMM性能？](../notes/cuda/向量化访存（vectorized_load_store）如何提升GEMM性能？.md)
  - [寄存器分块（Register Tiling）的原理是什么？](../notes/cuda/寄存器分块（Register_Tiling）的原理是什么？.md)
  - [双缓冲（Double Buffering）技术如何隐藏访存延迟？](../notes/cuda/双缓冲（Double_Buffering）技术如何隐藏访存延迟？.md)
  - [如何使用Tensor Core加速GEMM？](../notes/cuda/如何使用Tensor_Core加速GEMM？.md)
  - [WMMA API的使用方法和注意事项？](../notes/cuda/WMMA_API的使用方法和注意事项？.md)
  - [CUTLASS库的核心优化思想是什么？](../notes/cuda/CUTLASS库的核心优化思想是什么？.md)
  - [转置矩阵乘法如何优化？](../notes/cuda/转置矩阵乘法如何优化？.md)
  - [批量矩阵乘法（Batched GEMM）的优化策略？](../notes/cuda/批量矩阵乘法（Batched_GEMM）的优化策略？.md)
  - [如何选择合适的线程块大小和分块大小？](../notes/cuda/如何选择合适的线程块大小和分块大小？.md)
  - [GEMM性能如何接近理论峰值（cuBLAS水平）？](../notes/cuda/GEMM性能如何接近理论峰值（cuBLAS水平）？.md)

### 5.2 实际性能问题解决

#### **内存受限问题**
  - 如何处理显存不足的情况？
  - 流式计算（Streaming）如何实现？
  - 内存池（Memory Pool）管理的好处是什么？
  - 如何减少CPU-GPU数据传输？
  - Out-of-Core计算如何实现？

#### **计算精度问题**
  - 半精度计算会带来哪些精度问题？
  - 如何保证数值稳定性？
  - Kahan求和算法在GPU上如何实现？
  - 什么时候需要使用双精度计算？
  - 如何验证GPU计算结果的正确性？

#### **性能调优经验**
  - 遇到性能不达预期时的调试流程是什么？
  - 如何定位性能瓶颈？
  - 内存带宽瓶颈如何解决？
  - 计算瓶颈如何优化？
  - 如何权衡代码复杂度和性能提升？

### 5.3 项目集成与部署

#### **CUDA与深度学习框架集成**
  - [如何在PyTorch/TensorFlow中编写自定义CUDA算子？](../notes/cuda/如何在PyTorch/TensorFlow中编写自定义CUDA算子？.md)
  - CUDA算子如何实现自动求导？
  - 如何进行CUDA算子的性能基准测试？
  - CUDA算子与框架原生算子的性能对比
  - 什么时候需要自定义CUDA算子？

#### **CUDA程序部署**
  - 如何打包和部署CUDA应用？
  - Docker容器中如何使用CUDA？
  - 不同GPU架构的兼容性如何处理？
  - CUDA版本兼容性问题如何解决？
  - 生产环境中的错误处理和日志记录

#### **系统设计考虑**
  - CPU和GPU代码如何有效协作？
  - 异步执行和Pipeline设计的最佳实践
  - 多GPU系统的架构设计
  - 容错和错误恢复机制
  - 性能监控和自动调优策略

### 5.4 开放性项目讨论

#### **项目经验探讨**
  - 描述你做过的CUDA项目和遇到的挑战
  - 最复杂的性能优化问题是什么？如何解决的？
  - 项目中使用了哪些CUDA库？为什么？
  - 如何评估优化效果？性能提升了多少？
  - 有哪些失败的优化尝试？从中学到了什么？

#### **技术选型与权衡**
  - 什么时候选择CUDA而不是OpenCL或其他方案？
  - CPU和GPU计算如何分工？
  - 何时使用库函数，何时自己实现？
  - 开发效率和运行效率如何权衡？
  - 可移植性和性能如何平衡？

#### **新技术趋势**
  - 对最新GPU架构（如Hopper、Blackwell）的了解
  - CUDA新版本的特性（如异步拷贝、协作组等）
  - AI编译器（如TVM、XLA）对CUDA开发的影响
  - CUDA在大模型训练/推理中的应用
  - 未来CUDA技术的发展方向

---

## 面试评估维度

### 技术能力评估

**基础知识掌握程度**（25%）
- CUDA核心概念理解的准确性和深度
- 硬件架构认知的完整性
- 编程模型理解的清晰度
- 能否准确回答基础概念问题

**编程实现能力**（30%）
- 代码编写的正确性和规范性
- 算法实现的完整性
- 调试能力和问题定位能力
- 能否独立完成编程任务

**性能优化能力**（25%）
- 性能瓶颈识别的准确性
- 优化策略的合理性和有效性
- 优化工具的熟练使用
- 优化效果的量化评估能力

**项目经验与系统设计**（20%）
- 实际问题解决能力
- 系统架构设计思维
- 技术选型的合理性
- 新技术学习和适应能力

### 不同级别候选人考察重点

**初级开发者（0-2年经验）**
- 重点：基础概念理解、简单编程实现
- 考察内容：第1、2章为主
- 期望：能正确实现基础算法，理解基本优化概念
- 典型问题：向量加法、矩阵乘法、内存合并访问

**中级开发者（2-4年经验）**
- 重点：性能优化能力、库使用经验
- 考察内容：第2、3、4章为主
- 期望：能独立进行性能调优，熟练使用常用库
- 典型问题：共享内存优化、占用率调优、Stream使用

**高级开发者（4-6年经验）**
- 重点：复杂问题解决、系统设计能力
- 考察内容：第3、4、5章为主
- 期望：能解决复杂性能问题，设计高效系统架构
- 典型问题：多GPU系统设计、深度优化案例、自定义算子实现

**专家级/架构师（6年以上经验）**
- 重点：技术领导力、创新能力、前瞻性思维
- 考察内容：全部章节，重点在第5章
- 期望：能指导团队优化方向，设计创新解决方案
- 典型问题：架构设计、技术选型、前沿技术应用

### 面试实施建议

**面试流程设计**
1. **热身阶段（5分钟）**：简单概念问题，了解候选人水平
2. **基础考察（15-20分钟）**：核心概念和基础编程能力
3. **深入探讨（20-30分钟）**：性能优化或项目经验
4. **编程实践（20-30分钟，可选）**：现场编程或代码review
5. **开放讨论（10-15分钟）**：技术趋势、项目经验分享

**问题难度调整**
- 根据候选人回答质量动态调整问题深度
- 基础问题答不好则降低难度，答得好则深入追问
- 允许候选人展示自己最擅长的领域
- 注意区分"知道"和"深入理解"

**评分标准**
- **优秀**：概念清晰，能深入讲解原理，有丰富实践经验
- **良好**：概念准确，能正确实现，有一定优化经验
- **合格**：基础概念掌握，能完成简单任务
- **不合格**：基础概念模糊，编程能力不足

**注意事项**
- 理论与实践结合，不要只考理论概念
- 给予适当提示，观察学习和适应能力
- 鼓励候选人提问和讨论，评估主动性
- 关注候选人的思维过程而非单一答案
- 允许候选人使用文档和资料（模拟真实工作场景）

**红旗信号**
- 基础概念错误（如Warp大小、内存层次）
- 从未实际编写过CUDA代码
- 不了解性能分析工具的使用
- 无法解释项目中的技术决策
- 对错误回答缺乏自我纠正能力

**加分项**
- 有开源CUDA项目贡献
- 深入了解GPU硬件架构
- 能举出具体的优化案例和数据
- 关注最新技术发展
- 有跨领域应用经验（如HPC、AI、图形学）

---

## 附录：快速参考

### 常见性能优化检查清单

**内存优化**
- [ ] 是否使用了内存合并访问？
- [ ] 是否避免了Bank Conflict？
- [ ] 是否合理使用了共享内存？
- [ ] 是否使用了固定内存加速传输？
- [ ] 是否最小化了CPU-GPU数据传输？

**计算优化**
- [ ] Block大小是否合理（通常128-512）？
- [ ] 是否避免了Warp分歧？
- [ ] 占用率是否足够高？
- [ ] 是否合理使用了寄存器？
- [ ] 是否利用了指令级并行？

**执行优化**
- [ ] 是否使用Stream实现并发？
- [ ] 是否重叠了计算和传输？
- [ ] 是否避免了不必要的同步？
- [ ] 是否使用了合适的CUDA库？
- [ ] 是否针对目标架构进行了编译优化？

### 关键性能指标

- **内存带宽利用率**：实际带宽 / 理论峰值带宽
- **计算吞吐量利用率**：实际FLOPS / 理论峰值FLOPS
- **SM占用率**：活跃Warp数 / 最大Warp数
- **Warp执行效率**：无分歧的Warp百分比
- **内存效率**：有效事务数 / 实际事务数

### 常用CUDA函数速查

```cuda
// 设备管理
cudaGetDeviceCount()
cudaSetDevice()
cudaGetDeviceProperties()

// 内存管理
cudaMalloc() / cudaFree()
cudaMemcpy() / cudaMemcpyAsync()
cudaMallocHost() / cudaFreeHost()
cudaMallocManaged()

// 流和事件
cudaStreamCreate() / cudaStreamDestroy()
cudaEventCreate() / cudaEventDestroy()
cudaEventRecord() / cudaEventElapsedTime()
cudaStreamSynchronize() / cudaDeviceSynchronize()

// 错误处理
cudaGetLastError()
cudaGetErrorString()
```
