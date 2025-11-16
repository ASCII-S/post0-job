# AI高性能算子开发面试大纲

## 1. 基础理论与背景知识

### 1.1 深度学习基础
#### 1.1.1 深度学习框架概述
**深度学习框架架构**
- 主流深度学习框架（PyTorch、TensorFlow、PaddlePaddle等）的整体架构是什么？
- 前端API、中间表示（IR）、后端执行引擎各自的作用是什么？
- 静态图和动态图的区别及各自优缺点？

**计算图**
- 什么是计算图（Computation Graph）？
- 计算图的前向传播和反向传播是如何实现的？
- DAG（有向无环图）在深度学习中的应用？

#### 1.1.2 深度学习算子基础
**算子概念**
- 什么是深度学习算子（Operator/Kernel）？
- 算子与层（Layer）的区别是什么？
- 算子的输入、输出、参数分别指什么？

**常见算子分类**
- 深度学习中有哪些主要的算子类型（卷积、池化、激活、归一化等）？
- Element-wise算子、Reduction算子、GEMM算子的特点分别是什么？
- 哪些算子是计算密集型，哪些是访存密集型？

#### 1.1.3 张量与数据表示
**张量基础**
- 什么是张量（Tensor）？与矩阵、向量的关系？
- 张量的维度（rank）、形状（shape）、步长（stride）分别是什么？
- 连续张量（contiguous tensor）和非连续张量的区别？

**数据布局**
- NCHW、NHWC、NCHW4、NCHW32等数据布局的含义？
- 不同数据布局对性能的影响？
- 如何进行数据布局转换？

**数据类型**
- FP32、FP16、BF16、INT8等数据类型的区别？
- 混合精度训练的原理和优势？
- 量化（Quantization）的基本概念？

### 1.2 计算机体系结构基础
#### 1.2.1 CPU架构
**CPU基础**
- CPU的基本组成部分（ALU、控制单元、寄存器、缓存等）？
- 指令流水线的原理？
- SIMD（单指令多数据）指令集的作用？

**缓存层次**
- CPU的多级缓存（L1/L2/L3）架构？
- 缓存行（Cache Line）的概念和大小？
- 什么是缓存命中率？如何提高缓存命中率？
- 伪共享（False Sharing）问题及解决方案？

**内存访问**
- 内存访问的时间开销相比计算有多大？
- 什么是数据局部性（时间局部性和空间局部性）？
- Roofline模型的基本原理？

#### 1.2.2 GPU架构
**GPU vs CPU**
- GPU和CPU的架构差异？
- 为什么GPU适合深度学习计算？
- GPU的吞吐量优先vs CPU的延迟优先？

**CUDA架构基础**
- CUDA的硬件层次：GPU、SM（流多处理器）、CUDA Core的关系？
- Tensor Core是什么？有什么优势？
- GPU的内存层次：全局内存、共享内存、寄存器、常量内存、纹理内存？

**GPU计算能力**
- 什么是GPU的峰值算力（FLOPS）？
- 什么是内存带宽？如何计算？
- 如何评估一个算子是计算受限还是访存受限？

#### 1.2.3 其他加速器
**专用AI芯片**
- TPU、NPU、IPU等AI加速器的特点？
- 这些加速器与GPU的主要区别？

**异构计算**
- 什么是异构计算？
- CPU与GPU协同计算的基本流程？

## 2. CUDA编程基础

### 2.1 CUDA编程模型
#### 2.1.1 CUDA基本概念
**Host与Device**
- Host和Device分别指什么？
- Host与Device之间如何通信？
- 什么是统一内存（Unified Memory）？

**CUDA程序结构**
- 一个典型的CUDA程序包含哪些部分？
- `__global__`、`__device__`、`__host__`关键字的作用？
- CUDA核函数（Kernel）的特点？

#### 2.1.2 线程层次模型
**Grid、Block、Thread**
- Grid、Block、Thread三级层次的关系？
- 为什么需要这种层次结构？
- 如何确定Grid和Block的维度？

**线程索引计算**
- `threadIdx`、`blockIdx`、`blockDim`、`gridDim`的含义？
- 如何计算全局线程ID（1D、2D、3D情况）？
- 给定线程索引，如何映射到数据索引？

**Warp**
- 什么是Warp？Warp大小是多少？
- Warp调度的基本原理？
- Warp内的线程如何执行？

#### 2.1.3 内存层次
**全局内存**
- 全局内存的特点（大小、延迟、带宽）？
- 如何分配和释放全局内存？
- 合并访问（Coalesced Access）是什么？为什么重要？

**共享内存**
- 共享内存的特点和作用？
- 如何声明和使用共享内存？
- 动态共享内存 vs 静态共享内存？
- Bank冲突（Bank Conflict）是什么？如何避免？

**寄存器**
- 寄存器的特点（速度、容量）？
- 寄存器溢出（Register Spilling）是什么？如何避免？
- 寄存器数量对占用率的影响？

**常量内存和纹理内存**
- 常量内存的使用场景？
- 纹理内存的特点和优势？

### 2.2 CUDA编程实践
#### 2.2.1 内存管理
**内存分配与拷贝**
- `cudaMalloc`、`cudaFree`的使用？
- `cudaMemcpy`的不同类型（H2D、D2H、D2D）？
- 异步内存拷贝如何实现？

**内存访问优化**
- 如何实现合并访问？
- 内存对齐的重要性？
- Padding技术的应用？

#### 2.2.2 核函数设计
**核函数启动**
- 核函数启动语法 `<<<grid, block>>>`？
- 如何选择合适的Grid和Block大小？
- 核函数启动的开销？

**线程同步**
- `__syncthreads()`的作用和使用场景？
- 什么情况下会导致死锁？
- Warp内同步 vs Block内同步？

**原子操作**
- 常见的原子操作（atomicAdd、atomicMax等）？
- 原子操作的性能代价？
- 如何减少原子操作的使用？

#### 2.2.3 性能分析工具
**CUDA工具链**
- nvcc编译器的基本使用？
- `nvidia-smi`工具的作用？
- CUDA版本兼容性问题？

**性能分析**
- Nsight Systems和Nsight Compute的区别和使用场景？
- 如何使用nvprof进行性能分析？
- 如何读懂性能分析报告？

**常见性能指标**
- 占用率（Occupancy）是什么？如何优化？
- SM效率、内存吞吐量等指标的含义？
- 如何找到性能瓶颈？

### 2.3 CUDA高级特性
#### 2.3.1 流与并发
**CUDA Stream**
- CUDA Stream的概念和作用？
- 如何创建和使用Stream？
- 默认Stream和非默认Stream的区别？

**任务并发**
- Kernel并发执行的条件？
- 数据传输与计算的overlap如何实现？
- Stream的优先级？

#### 2.3.2 事件与同步
**CUDA Event**
- Event的作用？
- 如何使用Event测量时间？
- Event同步 vs Stream同步？

**同步机制**
- `cudaDeviceSynchronize`的作用？
- `cudaStreamSynchronize`和`cudaStreamWaitEvent`的区别？
- 显式同步 vs 隐式同步？

#### 2.3.3 高级内存特性
**Unified Memory**
- 统一内存的工作原理？
- 统一内存的优缺点？
- 页面迁移和预取机制？

**零拷贝内存**
- 零拷贝内存（Zero-Copy Memory）的概念？
- 适用场景？

**内存池**
- CUDA内存池（Memory Pool）的作用？
- 如何减少内存分配开销？

## 3. 算子优化技术

### 3.1 基础优化技术
#### 3.1.1 计算优化
**并行化策略**
- 数据并行和模型并行的区别？
- 如何设计算子的并行策略？
- 负载均衡问题如何解决？

**循环优化**
- 循环展开（Loop Unrolling）的作用？
- 循环合并（Loop Fusion）和循环分裂（Loop Fission）？
- 循环交换（Loop Interchange）对缓存的影响？

**分支优化**
- 分支分歧（Branch Divergence）是什么？
- 如何减少Warp内的分支分歧？
- 条件执行 vs 分支跳转？

#### 3.1.2 访存优化
**访存模式优化**
- 顺序访问 vs 随机访问的性能差异？
- Stride访问的影响？
- 如何重组数据访问模式？

**数据重用**
- 时间局部性和空间局部性在算子中的应用？
- 如何通过算法设计提高数据重用？
- Tiling技术的原理和应用？

**预取技术**
- 软件预取的概念？
- 如何在CUDA中实现预取？

#### 3.1.3 指令优化
**指令选择**
- 快速数学函数（`__fmul_rn`、`__fadd_rn`等）的使用？
- 内建函数（Intrinsic Functions）的优势？
- 精度 vs 性能的权衡？

**向量化**
- CUDA中的向量化访问（float4、int2等）？
- 向量化对内存带宽的影响？

### 3.2 高级优化技术
#### 3.2.1 共享内存优化
**Tiling技术**
- 什么是Tiling（分块）？
- 如何选择Tile的大小？
- Tiling对不同算子的效果？

**双缓冲技术**
- 双缓冲（Double Buffering）的原理？
- 如何实现计算和数据加载的overlap？

**Bank Conflict消除**
- 如何检测Bank Conflict？
- Padding、数组转置等解决方法？
- 广播访问（Broadcast）的利用？

#### 3.2.2 Warp级优化
**Warp级原语**
- Warp级shuffle操作（`__shfl_*`）的作用？
- 如何用shuffle实现Warp内reduction？
- Warp级投票函数（`__ballot`、`__all`等）？

**协作组（Cooperative Groups）**
- 协作组的概念和优势？
- 如何使用协作组进行灵活的同步？

#### 3.2.3 Tensor Core编程
**Tensor Core基础**
- Tensor Core的硬件特性？
- 支持的数据类型和矩阵大小？
- Tensor Core相比CUDA Core的加速比？

**WMMA API**
- WMMA（Warp Matrix Multiply-Accumulate）API的使用？
- fragment的概念？
- load、mma、store操作？

**Tensor Core优化**
- 数据对齐要求？
- 如何最大化Tensor Core利用率？

### 3.3 特定算子优化模式
#### 3.3.1 卷积优化
**卷积算法**
- Direct卷积、Im2col+GEMM、Winograd、FFT卷积的原理和适用场景？
- 各种卷积算法的时间复杂度和空间复杂度？

**优化技术**
- 如何优化不同kernel size的卷积？
- Depthwise卷积和Pointwise卷积的特点？
- 分组卷积（Group Convolution）的优化？

#### 3.3.2 矩阵乘法优化
**GEMM基础**
- GEMM的定义和参数（M、N、K、transA、transB等）？
- 朴素GEMM实现的性能瓶颈？

**分块GEMM**
- 多级分块的思路（Global、Shared、Register）？
- 如何确定各级分块大小？
- 数据加载和计算的overlap？

**cuBLAS库**
- cuBLAS的GEMM接口？
- cuBLAS的性能特点？

#### 3.3.3 Reduction优化
**Reduction模式**
- 什么是Reduction操作（sum、max、min等）？
- 树形归约的原理？

**优化策略**
- 如何避免Warp Divergence？
- 共享内存的使用？
- 多阶段Reduction的设计？
- 原子操作 vs 非原子实现？

#### 3.3.4 Transpose优化
**矩阵转置**
- 朴素转置的性能问题？
- 如何使用共享内存避免非合并访问？
- Bank Conflict的解决？

**高维张量转置**
- 任意维度排列（Permute）的实现？
- 性能优化策略？

## 4. 深度学习算子实现

### 4.1 基础算子
#### 4.1.1 Element-wise算子
**激活函数**
- ReLU、Sigmoid、Tanh、GELU等激活函数的实现？
- 激活函数的反向传播如何实现？
- 融合激活函数的优势？

**二元操作**
- Add、Mul、Sub、Div等操作的实现？
- 广播（Broadcasting）机制如何实现？
- 不同shape的广播优化？

#### 4.1.2 Reduction算子
**归约操作**
- Sum、Mean、Max、Min的实现？
- Softmax的前向和反向实现？
- LogSumExp技巧的应用？

**稳定性问题**
- Softmax的数值稳定性问题？
- 如何避免上溢和下溢？

#### 4.1.3 Normalization算子
**BatchNorm**
- Batch Normalization的前向和反向实现？
- 训练模式和推理模式的区别？
- 滑动平均（Exponential Moving Average）的计算？

**LayerNorm**
- Layer Normalization的实现？
- 与BatchNorm的区别？
- Welford算法用于在线计算均值和方差？

**其他归一化**
- GroupNorm、InstanceNorm的特点？
- RMSNorm的优势？

### 4.2 卷积与池化
#### 4.2.1 卷积算子
**标准卷积**
- 2D卷积的实现（不同padding、stride、dilation）？
- 反向传播中对输入和权重的梯度计算？

**特殊卷积**
- Depthwise Separable Convolution的实现？
- Transposed Convolution（反卷积）的原理和实现？
- Dilated Convolution的应用？

**优化库**
- cuDNN的卷积接口和使用？
- cuDNN的卷积算法选择机制？

#### 4.2.2 池化算子
**池化类型**
- MaxPooling、AvgPooling的实现？
- 反向传播的实现（特别是MaxPooling的索引保存）？
- Global Pooling的优化？

**自适应池化**
- Adaptive Pooling的原理？
- ROI Pooling和ROI Align的区别和实现？

### 4.3 高级算子
#### 4.3.1 注意力机制算子
**Self-Attention**
- Scaled Dot-Product Attention的实现？
- Q、K、V矩阵乘法的优化？
- Softmax在Attention中的实现？

**FlashAttention**
- FlashAttention的核心思想？
- Tiling和重计算的权衡？
- FlashAttention相比标准Attention的优势？

**Multi-Head Attention**
- 多头注意力的并行实现？
- 如何融合多个head的计算？

#### 4.3.2 位置编码算子
**Position Embedding**
- 绝对位置编码的实现？
- 正弦位置编码的计算？

**Rotary Position Embedding (RoPE)**
- RoPE的原理和实现？
- 如何高效计算旋转矩阵？

#### 4.3.3 采样与插值算子
**插值算法**
- Bilinear、Bicubic插值的实现？
- Grid Sample的原理和应用？

**采样算子**
- ROI Align的实现细节？
- Deformable Convolution的原理？

### 4.4 优化器算子
#### 4.4.1 常见优化器
**SGD**
- SGD和SGD with Momentum的实现？
- Weight Decay如何实现？

**Adam**
- Adam算法的实现（一阶和二阶动量）？
- AdamW的改进？
- 如何融合多个参数的更新到一个kernel？

**其他优化器**
- RMSprop、AdaGrad的实现？
- LAMB、LARS等分布式优化器的特点？

#### 4.4.2 优化器融合
**Fused Optimizer**
- 为什么要融合优化器kernel？
- 多个张量的并行更新？
- Mixed Precision训练中的优化器实现？

## 5. 算子融合与自动调优

### 5.1 算子融合
#### 5.1.1 融合基础
**融合概念**
- 什么是算子融合（Operator Fusion）？
- 算子融合的优势（减少访存、减少kernel启动开销）？
- 融合的限制和挑战？

**融合模式**
- Element-wise融合？
- Vertical融合和Horizontal融合？
- 哪些算子适合融合，哪些不适合？

#### 5.1.2 融合实现
**手动融合**
- 如何手动融合多个算子？
- Conv+BN+ReLU融合的实现？
- BERT中的融合模式（LayerNorm+Add等）？

**编译器融合**
- XLA、TVM、TensorRT等编译器的融合策略？
- 如何自动识别融合机会？

### 5.2 自动调优
#### 5.2.1 参数调优
**超参数空间**
- 算子的哪些参数需要调优（Block大小、Tile大小等）？
- 参数空间的规模和复杂度？

**搜索策略**
- 网格搜索、随机搜索的优缺点？
- 模拟退火、遗传算法等启发式搜索？
- 基于机器学习的搜索（如AutoTVM）？

#### 5.2.2 自动代码生成
**模板生成**
- 算子模板的设计？
- Jinja等模板引擎的应用？

**DSL方法**
- TVM的Tensor Expression（TE）？
- Halide语言的特点？
- Triton语言的设计思想？

### 5.3 编译优化
#### 5.3.1 图优化
**常量折叠**
- 什么是常量折叠（Constant Folding）？
- 编译期计算的优势？

**算子替换**
- 等价算子替换的例子？
- 如何利用硬件特性选择算子实现？

**公共子表达式消除**
- CSE（Common Subexpression Elimination）的原理？
- 在计算图中的应用？

#### 5.3.2 内存优化
**内存规划**
- 静态内存分配 vs 动态内存分配？
- 内存池的设计？

**In-place操作**
- 什么是In-place操作？
- 如何安全地进行In-place优化？

**内存重用**
- 张量生命周期分析？
- 内存复用的决策？

## 6. 算子开发工具与框架

### 6.1 深度学习框架算子扩展
#### 6.1.1 PyTorch算子扩展
**自定义算子**
- 如何使用C++扩展PyTorch？
- `torch.autograd.Function`的使用？
- 前向和反向函数的实现？

**CUDA扩展**
- PyTorch的CUDA扩展机制？
- `setuptools`和`pybind11`的使用？
- 如何集成自定义CUDA kernel到PyTorch？

**JIT编译**
- TorchScript的作用？
- `torch.jit.trace`和`torch.jit.script`的区别？

#### 6.1.2 TensorFlow算子扩展
**Custom Op**
- TensorFlow的Custom Op机制？
- 如何注册新的算子？

**XLA集成**
- XLA（Accelerated Linear Algebra）的原理？
- 如何编写XLA兼容的算子？

#### 6.1.3 其他框架
**PaddlePaddle**
- 飞桨的自定义算子API？

**MindSpore**
- 昇思的算子开发方式？

### 6.2 算子库
#### 6.2.1 NVIDIA库
**cuBLAS**
- cuBLAS的主要功能？
- 如何调用cuBLAS进行矩阵运算？
- cuBLASLt的高级特性？

**cuDNN**
- cuDNN支持的算子类型？
- cuDNN的卷积算法选择和benchmark？
- cuDNN的融合操作？

**cuSPARSE**
- 稀疏矩阵运算的优化？
- 稀疏算子的应用场景？

**Cutlass**
- Cutlass库的设计思想？
- Cutlass的模板化GEMM实现？
- 如何使用Cutlass开发高性能算子？

#### 6.2.2 开源算子库
**Eigen**
- Eigen库的特点和应用？
- Eigen的表达式模板技术？

**oneDNN (MKL-DNN)**
- Intel oneDNN的作用？
- CPU上的算子优化？

**OpenBLAS**
- OpenBLAS的使用场景？

### 6.3 算子开发工具
#### 6.3.1 TVM
**TVM架构**
- TVM的整体架构（前端、优化、后端）？
- Relay IR和TIR的作用？

**Tensor Expression**
- 如何使用TE描述算子？
- Schedule的概念和优化primitives？

**AutoTVM**
- 自动调优的流程？
- 模板定义和搜索空间？

#### 6.3.2 Triton
**Triton语言**
- Triton的设计目标？
- Triton相比CUDA的优势？

**编程模型**
- Triton的block-level编程？
- 如何用Triton实现GEMM、Softmax等算子？

#### 6.3.3 其他工具
**Halide**
- Halide的调度和算法分离？
- Halide的应用场景（图像处理等）？

**TensorRT**
- TensorRT的优化策略？
- 如何添加自定义plugin？

## 7. 性能分析与优化

### 7.1 性能指标
#### 7.1.1 时间指标
**执行时间**
- Wall time、GPU time、Kernel time的区别？
- 如何准确测量kernel执行时间？
- 预热（Warm-up）的必要性？

**吞吐量**
- 吞吐量（Throughput）的定义和计算？
- TFLOPS的计算方法？
- 批处理大小对吞吐量的影响？

#### 7.1.2 资源利用率
**计算利用率**
- 如何计算GPU计算单元的利用率？
- Roofline模型的应用？
- 计算效率 vs 峰值性能？

**内存带宽利用率**
- 如何计算实际内存带宽？
- 带宽利用率的瓶颈分析？

**占用率（Occupancy）**
- 占用率的定义和计算？
- 占用率 vs 性能的关系？
- 如何提高占用率？

#### 7.1.3 能效指标
**功耗**
- GPU功耗的测量？
- 性能 vs 功耗的权衡？

**能效比**
- FLOPS/W的意义？
- 如何优化能效？

### 7.2 性能分析方法
#### 7.2.1 Profile工具
**Nsight Systems**
- Timeline分析的方法？
- CPU-GPU交互的可视化？
- Stream并发分析？

**Nsight Compute**
- Kernel级性能分析？
- Roofline图表的解读？
- 指标收集和分析？

**PyTorch Profiler**
- PyTorch的profiling工具？
- 如何分析训练过程的性能？

#### 7.2.2 瓶颈识别
**计算瓶颈**
- 如何判断是计算受限？
- 提高计算密度的方法？

**访存瓶颈**
- 如何判断是访存受限？
- 减少访存的策略？

**同步瓶颈**
- 过度同步的影响？
- 如何减少不必要的同步？

#### 7.2.3 对比分析
**Baseline对比**
- 如何选择对比baseline？
- cuBLAS、cuDNN等库的性能对标？

**理论峰值对比**
- 如何计算理论峰值性能？
- 实际性能与理论峰值的差距分析？

### 7.3 优化流程
#### 7.3.1 优化方法论
**优化步骤**
- 性能优化的一般流程？
- 80/20原则在优化中的应用？

**Trade-off**
- 可读性 vs 性能？
- 通用性 vs 性能？
- 开发时间 vs 优化收益？

#### 7.3.2 数值精度
**精度保证**
- 如何验证优化后的正确性？
- 数值误差的容忍度？

**混合精度**
- FP16和FP32混合使用的策略？
- 自动混合精度（AMP）的原理？

## 8. 分布式与多卡优化

### 8.1 多卡通信
#### 8.1.1 通信原语
**集合通信**
- AllReduce、Broadcast、Reduce、Gather、Scatter等操作？
- Ring-AllReduce算法的原理？

**点对点通信**
- Send和Recv操作？
- P2P通信的应用场景？

#### 8.1.2 NCCL
**NCCL基础**
- NCCL（NVIDIA Collective Communications Library）的作用？
- NCCL的通信拓扑优化？

**NCCL编程**
- 如何使用NCCL API？
- NCCL与深度学习框架的集成？

### 8.2 分布式训练中的算子
#### 8.2.1 数据并行
**数据并行算子**
- 梯度同步的实现？
- AllReduce在数据并行中的应用？

**梯度累积**
- 梯度累积的实现？
- 大batch训练的优化？

#### 8.2.2 模型并行
**张量并行**
- 张量并行中的通信算子？
- Megatron-LM的并行策略？

**流水线并行**
- Pipeline并行的通信模式？
- Bubble的减少策略？

### 8.3 通信优化
#### 8.3.1 通信计算重叠
**Overlap技术**
- 如何实现通信和计算的overlap？
- 梯度分桶（Gradient Bucketing）？

**异步通信**
- 异步通信的实现？
- 同步 vs 异步的trade-off？

#### 8.3.2 通信压缩
**梯度压缩**
- 梯度量化和稀疏化？
- 压缩算法对精度的影响？

**ZeRO优化**
- ZeRO的分级优化策略？
- 显存优化 vs 通信开销？

## 9. 特定领域算子

### 9.1 Transformer算子
#### 9.1.1 Attention优化
**标准Attention**
- 标准Attention的性能瓶颈？
- 如何优化Softmax和矩阵乘法？

**高效Attention**
- FlashAttention-1和FlashAttention-2的区别？
- PagedAttention的原理和应用（vLLM）？
- Multi-Query Attention和Grouped-Query Attention？

#### 9.1.2 FFN算子
**Feed-Forward Network**
- FFN的并行优化？
- Gated FFN（如SwiGLU）的实现？

**专家混合（MoE）**
- MoE中的路由算子？
- 专家负载均衡问题？

#### 9.1.3 Transformer特定优化
**Sequence Length优化**
- 变长序列的处理？
- Padding和Masking的优化？

**KV Cache**
- KV Cache的实现和管理？
- Continuous Batching的支持？

### 9.2 推理优化算子
#### 9.2.1 量化算子
**量化方式**
- PTQ（训练后量化）和QAT（量化感知训练）？
- INT8、INT4量化的实现？

**量化kernel**
- 量化GEMM的实现？
- Dequantize-Compute-Quantize模式？

**低比特优化**
- 4-bit量化（GPTQ、AWQ等）的原理？
- Weight-Only量化的优势？

#### 9.2.2 稀疏化算子
**结构化稀疏**
- 2:4稀疏的硬件支持？
- Block稀疏的实现？

**非结构化稀疏**
- 稀疏矩阵格式（CSR、CSC、COO等）？
- 稀疏GEMM的优化？

#### 9.2.3 模型压缩算子
**知识蒸馏**
- 蒸馏loss的计算？

**剪枝**
- 剪枝算子的实现？
- 动态剪枝 vs 静态剪枝？

### 9.3 计算机视觉算子
#### 9.3.1 检测算子
**NMS**
- Non-Maximum Suppression的原理？
- 如何在GPU上高效实现NMS？
- Soft-NMS的改进？

**ROI相关**
- ROI Pooling、ROI Align的实现？
- RPN（Region Proposal Network）的算子？

#### 9.3.2 分割算子
**上采样**
- Bilinear Upsampling的优化？
- Pixel Shuffle的实现？

**特定分割算子**
- Mask RCNN的特定算子？
- Deformable Convolution？

### 9.4 图神经网络算子
#### 9.4.1 图计算基础
**图表示**
- 邻接矩阵、边列表等表示方式？
- 图数据在GPU上的存储？

**图遍历**
- BFS、DFS在GPU上的实现？

#### 9.4.2 GNN算子
**消息传递**
- Gather、Scatter、Reduce操作？
- 消息聚合的优化？

**图卷积**
- GCN、GAT等算子的实现？
- 稀疏矩阵乘法的应用？

## 10. 工程实践与项目经验

### 10.1 算子开发流程
#### 10.1.1 需求分析
**算子需求**
- 如何理解算子的功能需求？
- 输入输出shape的分析？
- 性能目标的设定？

**可行性评估**
- 硬件资源评估？
- 实现难度评估？
- 现有库的调研？

#### 10.1.2 设计与实现
**算法选择**
- 多种算法的比较？
- 参数空间的设计？

**分阶段实现**
- 正确性优先的baseline实现？
- 逐步优化的策略？

**单元测试**
- 如何设计测试用例？
- 边界条件的测试？
- 随机测试和fuzzing？

#### 10.1.3 优化与验证
**性能优化**
- 性能profile和瓶颈定位？
- 优化方案的实施？
- 优化前后的对比？

**正确性验证**
- 与标准实现的对比？
- 数值误差的检查？
- 梯度检查（Gradient Check）？

### 10.2 代码质量
#### 10.2.1 可读性与可维护性
**代码规范**
- 命名规范（变量、函数、kernel名称）？
- 注释的编写（特别是复杂的优化技巧）？
- 代码结构的组织？

**文档**
- 算子文档的内容（功能、接口、性能特性）？
- 示例代码的提供？

#### 10.2.2 健壮性
**错误处理**
- 输入检查和参数验证？
- CUDA错误检查（CUDA_CHECK宏）？
- 异常情况的处理？

**边界条件**
- 空输入、零大小张量的处理？
- 超大输入的处理？

#### 10.2.3 通用性与扩展性
**参数化设计**
- 如何设计灵活的接口？
- 支持不同数据类型（模板或宏）？

**可配置性**
- 编译时配置 vs 运行时配置？
- 环境变量的使用？

### 10.3 协作与版本管理
#### 10.3.1 团队协作
**代码审查**
- Code Review的要点？
- 性能关键代码的审查重点？

**沟通**
- 技术方案的讨论和文档化？
- 与上下游的接口定义？

#### 10.3.2 版本控制
**Git使用**
- 分支管理策略？
- Commit message的规范？

**CI/CD**
- 自动化测试的重要性？
- 性能回归测试？

### 10.4 常见问题与调试
#### 10.4.1 常见Bug
**内存问题**
- 越界访问的调试（cuda-memcheck、compute-sanitizer）？
- 内存泄漏的检测？
- 未初始化内存的使用？

**同步问题**
- Race Condition的识别？
- 死锁的调试？

**数值问题**
- NaN和Inf的产生和调试？
- 精度损失问题？

#### 10.4.2 调试技巧
**调试工具**
- cuda-gdb的使用？
- printf调试在CUDA中的应用？
- 断言（assert）的使用？

**性能调试**
- 性能下降的定位？
- 性能不稳定的原因分析？

## 11. 前沿技术与趋势

### 11.1 新硬件架构
#### 11.1.1 GPU架构演进
**NVIDIA架构**
- 从Volta到Hopper架构的演进？
- 各代架构的关键特性（Tensor Core、MIG、TMA等）？

**新特性应用**
- Hopper的Transformer Engine？
- FP8数据类型的使用？

#### 11.1.2 其他硬件
**AI专用芯片**
- TPU、Graphcore IPU、Cerebras WSE等的特点？
- 针对不同硬件的算子适配？

**异构架构**
- CPU+GPU+NPU的协同？

### 11.2 编译技术
#### 11.2.1 AI编译器
**MLIR**
- MLIR的多层IR设计？
- Dialect的概念和应用？

**Polyhedral模型**
- 多面体编译的基础？
- 在循环优化中的应用？

#### 11.2.2 自动化
**AutoML for Operators**
- 神经架构搜索在算子优化中的应用？
- 强化学习指导的算子优化？

**代码生成**
- 基于模板的代码生成 vs 基于搜索的代码生成？
- 大模型辅助的算子开发？

### 11.3 新范式
#### 11.3.1 大模型时代的算子
**超长序列**
- 百万级token长度的Attention优化？
- Ring Attention、Blockwise Attention？

**超大模型**
- 万亿参数模型的算子挑战？
- 3D并行中的算子需求？

#### 11.3.2 边缘部署
**移动端优化**
- ARM NEON、Mali GPU的优化？
- 移动端算子库（如NCNN、MNN）？

**嵌入式设备**
- 低功耗算子设计？
- 实时性要求的算子优化？

### 11.4 研究方向
#### 11.4.1 算法创新
**新型算子**
- 状态空间模型（Mamba）的算子？
- MoE路由算法的创新？

**近似计算**
- 近似算子的精度 vs 性能权衡？

#### 11.4.2 系统优化
**端到端优化**
- 跨层优化的机会？
- 算子融合、内存规划、调度的联合优化？

**能效优化**
- Green AI的算子设计？
- 动态电压频率调节（DVFS）？

## 12. 综合问题

### 12.1 系统设计问题
**场景问题**
- 如何为一个新的深度学习模型设计和优化关键算子？
- 给定一个性能瓶颈，如何系统性地分析和优化？
- 如何在有限时间内权衡算子开发的正确性、性能和可维护性？

**Trade-off问题**
- 在内存受限的情况下如何优化算子？
- 批大小增大后算子性能不升反降，可能的原因和解决方案？
- 如何在多个性能指标（延迟、吞吐量、内存占用）之间做权衡？

### 12.2 开放性问题
**优化思路**
- 描述你优化过的一个算子，具体做了哪些优化，性能提升如何？
- 遇到过哪些困难的算子优化问题，如何解决的？
- 如何从零开始学习和掌握CUDA编程和算子优化？

**技术视野**
- 你认为未来算子开发的主要挑战是什么？
- AI编译器和手写算子的关系和未来？
- 如何看待大模型推理优化中的算子开发机会？

### 12.3 项目经验
**项目描述**
- 描述一个你参与过的算子开发项目？
- 项目中遇到的主要技术难点是什么？
- 你的主要贡献是什么？

**工程能力**
- 如何保证算子的正确性？
- 如何进行性能测试和benchmark？
- 如何与团队协作完成复杂的算子库开发？

---

## 附录：学习资源建议

### A. 书籍
- 《CUDA C编程权威指南》（Professional CUDA C Programming）
- 《深度学习系统：算法、框架与实现》
- 《Programming Massively Parallel Processors》
- 《计算机体系结构：量化研究方法》

### B. 在线课程
- NVIDIA CUDA培训系列
- Stanford CS149: Parallel Computing
- CMU 15-418: Parallel Computer Architecture and Programming

### C. 实践项目
- 实现基础算子（GEMM、Conv、Softmax等）
- 参与开源项目（TVM、PyTorch等）
- 复现论文中的优化算法（FlashAttention等）
- Kaggle或类似平台的优化竞赛

### D. 社区与论坛
- NVIDIA Developer Forums
- PyTorch Discuss
- Reddit r/CUDA
- 知乎、博客等中文技术社区

