# CPU架构面试大纲

## 主题：CPU架构与系统优化

本大纲系统覆盖CPU架构的核心知识，从基础架构到高级优化技术，特别强化NUMA架构的深入理解。

---

## 第一章：CPU基础架构

### 1.1 指令集架构（ISA）

#### 1.1.1 指令集基础

**指令集分类**
- 什么是指令集架构（ISA）？它在计算机系统中的作用是什么？
- CISC和RISC的区别是什么？各自的优缺点是什么？
- x86、ARM、RISC-V等主流指令集的特点是什么？
- 为什么x86是CISC但现代x86 CPU内部使用RISC风格的微操作？
- 指令集的向后兼容性如何影响CPU设计？

**指令格式与编码**
- 定长指令和变长指令的区别是什么？
- x86的变长指令编码有什么优缺点？
- 指令的操作数来源有哪些（寄存器、立即数、内存）？
- 寻址模式有哪些？各自适用于什么场景？
- 微码（Microcode）是什么？它的作用是什么？

**特权级别**
- CPU的特权级别（Ring 0-3）是如何设计的？
- 用户态和内核态的切换过程是怎样的？
- 系统调用时CPU硬件做了哪些工作？
- 特权指令有哪些？为什么需要特权级别保护？

#### 1.1.2 寄存器架构

**通用寄存器**
- CPU的寄存器有哪些类型？各自的作用是什么？
- 通用寄存器的数量如何影响性能？
- x86-64相比x86-32增加了哪些寄存器？为什么？
- 寄存器重命名是什么？它解决了什么问题？
- 寄存器压力（Register Pressure）对性能有什么影响？

**特殊用途寄存器**
- 程序计数器（PC）、栈指针（SP）的作用是什么？
- 标志寄存器（EFLAGS/RFLAGS）包含哪些标志位？
- 控制寄存器（CR0-CR4）用于配置什么功能？
- 段寄存器在现代CPU中还有用吗？
- MSR（Model Specific Register）是什么？

**向量寄存器**
- SSE、AVX、AVX-512寄存器的演进历史是什么？
- 向量寄存器的位宽如何影响SIMD性能？
- 向量寄存器的上下文切换开销有多大？
- 如何查询CPU支持的向量扩展指令集？

### 1.2 流水线架构

#### 1.2.1 指令流水线基础

**流水线概念**
- 什么是指令流水线？它如何提高CPU吞吐量？
- 经典五级流水线（IF/ID/EX/MEM/WB）的各阶段做什么？
- 现代CPU的流水线有多少级？更深的流水线有什么优缺点？
- 流水线的CPI（Cycles Per Instruction）如何计算？
- 理论IPC（Instructions Per Cycle）的上限是多少？

**流水线冒险**
- 结构冒险、数据冒险、控制冒险分别是什么？
- 数据冒险的三种类型（RAW、WAR、WAW）有什么区别？
- 前递（Forwarding）技术如何解决数据冒险？
- 流水线停顿（Stall）对性能的影响有多大？
- 如何通过编译器优化减少流水线冒险？

#### 1.2.2 超标量与乱序执行

**超标量架构**
- 什么是超标量处理器？它如何实现指令级并行？
- 多发射（Multi-issue）如何工作？发射宽度如何选择？
- 超标量处理器的理论加速比是多少？实际能达到多少？
- 指令发射的限制因素有哪些？
- 如何评估超标量处理器的效率？

**乱序执行**
- 乱序执行的基本原理是什么？它解决了什么问题？
- 保留站（Reservation Station）的作用是什么？
- Tomasulo算法如何实现动态调度？
- 重排序缓冲区（ROB）如何保证程序语义正确？
- 乱序执行对程序员是否可见？内存模型如何处理？

**寄存器重命名**
- 为什么需要寄存器重命名？
- 物理寄存器和架构寄存器的区别是什么？
- 寄存器重命名如何消除WAR和WAW冒险？
- 物理寄存器文件的大小如何影响乱序窗口？
- 寄存器回收机制是如何工作的？

#### 1.2.3 分支预测

**分支预测基础**
- 为什么分支预测对性能至关重要？
- 分支预测错误的代价是什么？
- 静态分支预测和动态分支预测的区别是什么？
- 分支预测器的准确率通常能达到多少？

**分支预测器类型**
- 单比特预测器、两比特饱和计数器的工作原理是什么？
- 局部历史预测器和全局历史预测器有什么区别？
- gshare、gselect等预测器的设计思想是什么？
- 两级自适应预测器如何工作？
- 现代CPU使用的混合预测器是如何设计的？

**高级分支优化**
- BTB（Branch Target Buffer）的作用是什么？
- 返回地址栈（RAS）如何优化函数返回？
- 间接跳转预测如何实现？
- 如何通过代码优化帮助分支预测器？
- 分支预测失败时的恢复机制是什么？

### 1.3 执行单元

#### 1.3.1 功能单元组织

**算术逻辑单元（ALU）**
- ALU执行哪些基本操作？
- 现代CPU有多少个ALU？如何并行工作？
- 整数运算和浮点运算的性能差异有多大？
- 除法指令为什么特别慢？如何优化？
- 快速路径（Fast Path）和慢速路径的区别是什么？

**浮点单元（FPU）**
- FPU的基本架构是什么？
- IEEE 754浮点标准的关键要点是什么？
- FP32、FP64的性能差异有多大？
- 浮点运算的延迟和吞吐量如何？
- FMA（Fused Multiply-Add）指令的优势是什么？

**向量执行单元**
- SIMD执行单元如何实现数据级并行？
- SSE、AVX、AVX-512的性能对比如何？
- 向量指令的对齐要求是什么？
- 向量执行单元的功耗特性如何？
- AVX-512降频（Downclocking）是什么问题？

#### 1.3.2 执行流水线

**执行延迟与吞吐量**
- 指令的延迟（Latency）和吞吐量（Throughput）有什么区别？
- 为什么某些指令延迟很高但吞吐量很好？
- 流水化执行单元如何提高吞吐量？
- 如何查询不同指令的性能特性？
- 关键路径（Critical Path）如何影响性能？

**端口与调度**
- 执行端口（Execution Port）是什么？
- 现代CPU有多少个执行端口？各端口的功能是什么？
- 端口竞争如何影响性能？
- 微融合（Micro-fusion）和宏融合（Macro-fusion）是什么？
- 如何通过指令选择避免端口瓶颈？

---

## 第二章：存储层次结构

### 2.1 Cache架构

#### 2.1.1 Cache基础

**Cache原理**
- 为什么需要Cache？它解决了什么问题？
- Cache的局部性原理包括哪些？
- Cache的命中率如何影响性能？
- Cache访问的平均延迟如何计算？
- Cache与主存的速度差异有多大？

**Cache组织结构**
- 直接映射、全相联、组相联的区别是什么？
- Cache行（Cache Line）的大小通常是多少？为什么？
- 标记（Tag）、索引（Index）、偏移（Offset）如何从地址中提取？
- 如何计算Cache的容量、路数、组数的关系？
- Cache的替换策略有哪些（LRU、PLRU、Random）？

**写策略**
- 写直达（Write-Through）和写回（Write-Back）的区别是什么？
- 写分配（Write-Allocate）和非写分配有什么影响？
- 写缓冲（Write Buffer）的作用是什么？
- 脏位（Dirty Bit）如何跟踪Cache行状态？
- 写合并（Write Combining）如何优化连续写操作？

#### 2.1.2 多级Cache

**Cache层次**
- L1/L2/L3 Cache的典型大小、延迟、带宽是多少？
- 为什么需要多级Cache？
- L1分为L1i和L1d的原因是什么？
- 包含性（Inclusive）和排他性（Exclusive）Cache的区别是什么？
- Victim Cache是什么？它的作用是什么？

**Cache一致性协议**
- 什么是Cache一致性问题？
- MESI协议的四个状态分别是什么？状态转换如何进行？
- MESIF和MOESI协议相比MESI有什么改进？
- 窥探（Snooping）和目录（Directory）协议的区别是什么？
- 伪共享（False Sharing）是什么？如何避免？

**Cache性能优化**
- Cache友好的数据结构设计原则是什么？
- 如何通过数据对齐优化Cache性能？
- Loop Blocking（分块）如何提高Cache命中率？
- Cache预取如何工作？
- 如何使用性能计数器分析Cache性能？

#### 2.1.3 TLB与虚拟内存

**地址转换**
- 虚拟地址到物理地址的转换过程是怎样的？
- 页表（Page Table）的结构是什么？
- 多级页表如何节省内存？
- 大页（Huge Page）的优势是什么？何时使用？
- TLB（Translation Lookaside Buffer）的作用是什么？

**TLB架构**
- TLB的容量和相联度通常是多少？
- TLB未命中（TLB Miss）的代价有多大？
- 页表遍历（Page Walk）的开销如何？
- PCID（Process Context Identifier）如何减少TLB刷新？
- TLB shootdown在多核系统中如何工作？

### 2.2 内存系统

#### 2.2.1 内存访问模式

**内存带宽与延迟**
- DRAM的访问延迟通常是多少纳秒？
- 内存带宽如何计算？DDR4/DDR5的带宽是多少？
- 顺序访问和随机访问的性能差异有多大？
- 内存访问的颗粒度是什么？
- Row Buffer Hit/Miss对性能的影响是什么？

**内存控制器**
- 内存控制器的作用是什么？
- 内存通道（Channel）和Rank的概念是什么？
- 交错（Interleaving）如何提高内存带宽利用率？
- 内存请求调度器的优化目标是什么？
- 内存控制器的队列深度如何影响延迟？

#### 2.2.2 预取机制

**硬件预取**
- 硬件预取器的基本原理是什么？
- 步长预取（Stride Prefetcher）如何工作？
- 流预取器（Stream Prefetcher）适用于什么场景？
- 预取的提前量（Distance）如何选择？
- 预取错误（Prefetch Pollution）如何影响性能？

**软件预取**
- 软件预取指令（如`prefetch`）如何使用？
- 软件预取相比硬件预取有什么优缺点？
- 预取的时机如何把握？
- Non-temporal预取（如`prefetchnta`）的作用是什么？
- 如何评估预取的效果？

---

## 第三章：并行与多核架构

### 3.1 多核处理器

#### 3.1.1 多核基础

**多核架构**
- 单核和多核处理器的区别是什么？
- 为什么要从提高频率转向多核？
- 对称多处理（SMP）架构是什么？
- 多核处理器的可扩展性面临哪些挑战？
- Amdahl定律如何限制并行加速比？

**核心间通信**
- 核心之间如何通信？
- 共享Cache在多核通信中的作用是什么？
- 核心间中断（IPI）如何工作？
- Coherency带宽如何影响可扩展性？
- Ring Bus、Mesh等互连拓扑的优缺点是什么？

#### 3.1.2 同步与原子操作

**硬件同步原语**
- 原子操作的硬件实现原理是什么？
- Compare-And-Swap（CAS）如何工作？
- Load-Link/Store-Conditional的语义是什么？
- 原子操作的开销有多大？
- 内存屏障（Memory Barrier）的作用是什么？

**锁与无锁编程**
- 自旋锁（Spinlock）和互斥锁的区别是什么？
- 锁的缓存行竞争（Cache Line Bouncing）是什么？
- 如何实现高效的自旋锁？
- 无锁数据结构的设计原则是什么？
- 读写锁（RW Lock）如何优化读多写少场景？

#### 3.1.3 超线程（SMT）

**SMT基础**
- 什么是超线程（Simultaneous Multithreading）？
- SMT和时间片多任务的区别是什么？
- 超线程如何提高硬件利用率？
- 两个硬件线程共享哪些资源？
- 超线程的理论加速比和实际加速比是多少？

**SMT性能特性**
- 超线程在什么场景下收益最大？
- 什么时候超线程反而降低性能？
- 超线程如何影响Cache和TLB性能？
- 安全漏洞（如超线程泄露）如何产生？
- 何时应该禁用超线程？

### 3.2 SIMD与向量化

#### 3.2.1 SIMD架构

**SIMD基础**
- 什么是SIMD（Single Instruction Multiple Data）？
- SIMD如何实现数据级并行？
- 向量宽度对性能的影响是什么？
- SIMD相比标量指令有多大加速比？
- SIMD的局限性有哪些？

**x86向量扩展**
- MMX、SSE、AVX的演进历史是什么？
- SSE到AVX-512的位宽和性能提升如何？
- AVX的VEX编码有什么优势？
- AVX-512的掩码（Mask）寄存器如何使用？
- 为什么有些场景下AVX-512并不比AVX2快？

#### 3.2.2 向量化编程

**自动向量化**
- 编译器如何实现自动向量化？
- 哪些代码模式可以被自动向量化？
- 循环依赖如何阻止向量化？
- 如何使用编译器指令辅助向量化（如`#pragma`）？
- 如何查看编译器的向量化报告？

**内联汇编与Intrinsics**
- SIMD Intrinsics是什么？如何使用？
- Intrinsics相比汇编有什么优势？
- 如何编写跨平台的SIMD代码？
- 如何处理向量宽度不是数据长度整数倍的情况？
- 混合标量和向量代码的注意事项是什么？

**向量化优化技巧**
- 数据对齐对SIMD性能的影响有多大？
- 如何重组数据布局以适应SIMD（SoA vs AoS）？
- Gather/Scatter指令的性能特性如何？
- 如何优化条件分支密集的向量代码？
- 向量归约（Reduction）如何高效实现？

---

## 第四章：NUMA架构

### 4.1 NUMA基础

#### 4.1.1 NUMA架构原理

**NUMA概念**
- [什么是NUMA（Non-Uniform Memory Access）？](../notes/熟悉CPU架构/什么是NUMA（Non-Uniform_Memory_Access）？.md)
- NUMA相比UMA（SMP）有什么优势？
- 为什么大规模多核系统需要NUMA？
- NUMA如何解决内存带宽瓶颈？
- NUMA架构的trade-off是什么？

**NUMA节点组织**
- 什么是NUMA节点（Node）？
- 每个NUMA节点包含哪些资源？
- 本地内存和远程内存的访问延迟差异有多大？
- NUMA节点间如何互连（QPI、UPI、Infinity Fabric）？
- NUMA距离（NUMA Distance）如何衡量？

**NUMA拓扑**
- 如何查看系统的NUMA拓扑结构？
- `numactl`、`lscpu`等工具如何使用？
- NUMA节点和物理CPU socket的关系是什么？
- 非对称NUMA拓扑是什么？
- Sub-NUMA Clustering（SNC）是什么？

#### 4.1.2 NUMA内存管理

**内存分配策略**
- NUMA系统的内存分配策略有哪些？
- Local Allocation（本地分配）的优势是什么？
- Interleaved Allocation（交错分配）适用于什么场景？
- Preferred Node和Bind策略的区别是什么？
- 默认的First-Touch策略是如何工作的？

**页面迁移**
- 什么是NUMA页面迁移（Page Migration）？
- 自动NUMA平衡（Automatic NUMA Balancing）如何工作？
- 页面迁移的开销有多大？何时值得迁移？
- 如何手动触发页面迁移？
- 页面迁移如何与透明大页（THP）交互？

**内存回收**
- NUMA系统的内存回收策略有什么特殊之处？
- Zone Reclaim是什么？何时启用？
- 本地内存不足时是分配远程内存还是回收？
- `zone_reclaim_mode`参数如何配置？
- NUMA内存碎片化如何影响性能？

### 4.2 NUMA与Cache一致性

#### 4.2.1 ccNUMA架构

**Cache一致性NUMA**
- 什么是ccNUMA（Cache-Coherent NUMA）？
- ccNUMA如何保证跨节点的Cache一致性？
- Directory-based协议在NUMA中如何工作？
- Home Node和Local Node的概念是什么？
- NUMA系统的一致性代价有多大？

**跨节点一致性**
- 跨NUMA节点的Cache一致性流量如何产生？
- Remote Cache Line访问的延迟是多少？
- 一致性流量如何影响互连带宽？
- 如何通过profiling工具观察NUMA流量？
- 伪共享在NUMA系统中的影响更大吗？

#### 4.2.2 NUMA感知的Cache优化

**减少跨节点访问**
- 如何通过数据放置减少远程内存访问？
- NUMA-local Cache如何优化性能？
- 如何避免跨节点的Cache Line竞争？
- 数据复制（Replication）vs 数据迁移的权衡是什么？
- 只读数据和读写数据的NUMA优化策略有何不同？

**Cache亲和性**
- 什么是NUMA Cache亲和性？
- 如何保证数据和访问它的线程在同一节点？
- Last Level Cache（LLC）在NUMA中如何分布？
- 跨节点LLC访问的性能影响是什么？
- 如何使用`perf`分析NUMA Cache性能？

### 4.3 NUMA编程

#### 4.3.1 NUMA API与工具

**Linux NUMA API**
- `numa.h`库提供了哪些API？
- `numa_alloc_onnode()`如何使用？
- `numa_run_on_node()`如何绑定线程到节点？
- `get_mempolicy()`和`set_mempolicy()`的作用是什么？
- `move_pages()`如何迁移页面？

**numactl工具**
- `numactl`的常用选项有哪些？
- 如何使用`numactl`启动NUMA感知的程序？
- `--membind`、`--cpunodebind`、`--interleave`的区别是什么？
- 如何查看进程的NUMA内存分布（`numastat`）？
- `numademo`如何演示NUMA效果？

**NUMA监控**
- `/proc/[pid]/numa_maps`文件包含哪些信息？
- 如何监控NUMA命中率和远程访问比例？
- `perf`如何分析NUMA相关的性能事件？
- 系统级NUMA统计信息在哪里查看（`/sys/devices/system/node/`）？
- 如何识别NUMA瓶颈？

#### 4.3.2 NUMA感知编程

**线程与内存亲和性**
- 如何实现线程绑定（CPU Affinity）？
- `sched_setaffinity()`如何使用？
- 线程绑定的粒度如何选择（Node、Core、Thread）？
- 如何确保线程访问的是本地内存？
- First-Touch策略如何在多线程初始化中正确使用？

**数据布局优化**
- 如何设计NUMA友好的数据结构？
- 数据分区（Partitioning）的策略有哪些？
- 共享数据和私有数据的NUMA优化有何不同？
- 如何处理跨节点的数据依赖？
- 数据复制如何在多个NUMA节点上实现？

**并行模式**
- 数据并行模式在NUMA上如何实现？
- 任务并行如何考虑NUMA局部性？
- MapReduce等模式的NUMA优化策略是什么？
- 如何在NUMA上实现高效的生产者-消费者模式？
- Work Stealing在NUMA系统上的挑战是什么？

#### 4.3.3 NUMA性能调优

**性能分析**
- 如何量化NUMA的性能影响？
- NUMA命中率多少是可接受的？
- 远程访问比例如何影响性能？
- 如何使用`perf c2c`分析NUMA竞争？
- Intel VTune等工具如何分析NUMA性能？

**调优策略**
- 何时使用本地分配，何时使用交错分配？
- 如何平衡负载均衡和NUMA局部性？
- 内存绑定（Memory Binding）和CPU绑定如何协同？
- 如何调优NUMA系统的大页配置？
- 容器和虚拟机中的NUMA优化有何特殊之处？

**常见问题与解决方案**
- NUMA不均衡（NUMA Imbalance）如何诊断和解决？
- 跨节点内存分配导致的性能下降如何处理？
- 为什么有时禁用NUMA反而性能更好？
- 如何处理NUMA和超线程的交互影响？
- NUMA配置错误的常见症状有哪些？

### 4.4 NUMA在不同场景的应用

#### 4.4.1 数据库系统

**数据库NUMA优化**
- 数据库如何利用NUMA架构？
- Buffer Pool的NUMA分区策略是什么？
- 如何减少数据库中的跨节点锁竞争？
- 查询执行器的NUMA感知调度如何实现？
- OLTP和OLAP在NUMA上的优化有何不同？

#### 4.4.2 高性能计算（HPC）

**HPC NUMA优化**
- MPI程序如何绑定到NUMA节点？
- 共享内存并行（OpenMP）的NUMA优化策略是什么？
- 如何在NUMA上优化矩阵运算？
- 科学计算中的数据分布模式有哪些？
- NUMA如何影响HPC的可扩展性？

#### 4.4.3 虚拟化与云环境

**虚拟NUMA**
- 什么是虚拟NUMA（vNUMA）？
- Hypervisor如何向VM暴露NUMA拓扑？
- VM的NUMA配置最佳实践是什么？
- 跨NUMA节点的VM放置有什么影响？
- 云环境中如何保证NUMA性能？

#### 4.4.4 大数据与AI

**大数据框架NUMA优化**
- Hadoop、Spark等框架的NUMA感知如何配置？
- 内存计算框架如何利用NUMA？
- 分布式缓存的NUMA优化策略是什么？

**AI推理与训练**
- 深度学习框架如何处理NUMA？
- 模型并行和数据并行的NUMA考虑是什么？
- 推理服务的NUMA优化要点是什么？
- GPU服务器中CPU-GPU的NUMA关系如何影响性能？

---

## 第五章：内存模型与一致性

### 5.1 内存一致性模型

#### 5.1.1 顺序一致性

**内存模型基础**
- 什么是内存一致性模型（Memory Consistency Model）？
- 顺序一致性（Sequential Consistency）的定义是什么？
- 为什么现代处理器不实现严格的顺序一致性？
- Program Order和Memory Order的区别是什么？
- 内存一致性模型如何影响并发编程？

**内存重排序**
- 哪些类型的内存重排序可能发生？
- Load-Load、Load-Store、Store-Store、Store-Load重排序的含义是什么？
- 编译器重排序和硬件重排序有什么区别？
- 如何通过Volatile或Atomic防止重排序？
- 内存依赖（Memory Dependency）如何限制重排序？

#### 5.1.2 x86内存模型

**x86-TSO模型**
- x86的Total Store Order（TSO）模型是什么？
- x86允许哪些重排序，禁止哪些重排序？
- Store Buffer如何影响内存可见性？
- x86的隐式内存屏障在哪些指令中存在？
- x86相比ARM等弱内存模型有什么优势和劣势？

**内存屏障指令**
- `mfence`、`lfence`、`sfence`的作用分别是什么？
- 何时需要显式使用内存屏障？
- 内存屏障的性能开销有多大？
- Lock前缀指令如何提供内存屏障语义？
- 现代CPU如何优化内存屏障的实现？

### 5.2 Cache一致性协议

#### 5.2.1 Snooping协议

**MESI协议详解**
- MESI的四个状态（Modified、Exclusive、Shared、Invalid）的含义是什么？
- 状态转换是如何触发的？
- 读/写操作分别导致哪些一致性消息？
- MESI如何处理多核同时读写的情况？
- Store Buffer和Invalidate Queue如何优化MESI？

**MESIF和MOESI**
- MESIF的Forward状态解决了什么问题？
- MOESI的Owned状态有什么优势？
- Intel和AMD的一致性协议有什么区别？
- 一致性协议如何影响多核可扩展性？

#### 5.2.2 Directory协议

**目录一致性**
- Directory-based协议的基本原理是什么？
- 目录如何记录Cache行的共享状态？
- Directory相比Snooping有什么优缺点？
- 何时使用Directory协议？
- Hybrid协议如何结合两者优势？

---

## 第六章：性能优化技术

### 6.1 性能分析方法

#### 6.1.1 性能计数器

**硬件性能监控**
- 什么是性能监控单元（PMU）？
- 性能计数器可以监控哪些事件？
- 如何使用`perf`工具进行性能分析？
- Top-down分析方法是什么？
- 如何识别性能瓶颈（前端、后端、内存、分支）？

**微架构分析**
- IPC（Instructions Per Cycle）如何解读？
- Cache Miss Rate如何影响性能？
- Branch Misprediction Rate的可接受范围是多少？
- 如何分析流水线停顿的原因？
- 资源竞争如何通过计数器识别？

#### 6.1.2 Profiling工具

**Linux perf**
- `perf stat`、`perf record`、`perf report`如何使用？
- 如何使用`perf top`进行热点分析？
- `perf mem`如何分析内存访问？
- `perf c2c`如何检测Cache竞争？
- 火焰图（Flame Graph）如何生成和解读？

**Intel VTune**
- VTune的主要分析类型有哪些？
- Hotspot分析、Memory Access分析、Threading分析分别关注什么？
- 如何使用VTune进行微架构探索？
- VTune的Bottom-up分析如何进行？

**AMD uProf**
- uProf的特色功能有哪些？
- 如何分析AMD CPU的性能特性？

### 6.2 代码优化技术

#### 6.2.1 编译器优化

**优化级别**
- `-O0`到`-O3`、`-Ofast`的区别是什么？
- LTO（Link Time Optimization）如何工作？
- PGO（Profile Guided Optimization）的原理是什么？
- 如何查看编译器生成的汇编代码？
- 编译器的自动向量化和自动并行化能力如何？

**代码生成优化**
- 循环展开（Loop Unrolling）的效果如何？
- 函数内联（Inlining）的权衡是什么？
- 如何使用`__builtin_expect`提示分支概率？
- 如何使用`__restrict`关键字优化指针别名？
- 属性（Attributes）如`__attribute__((aligned))`如何使用？

#### 6.2.2 算法与数据结构优化

**Cache友好设计**
- 如何设计Cache友好的数据结构？
- 数组 vs 链表的Cache性能差异有多大？
- SoA（Structure of Arrays）和AoS（Array of Structures）如何选择？
- 数据对齐和Padding如何避免伪共享？
- 分块算法（Blocking/Tiling）如何提高局部性？

**减少分支**
- 分支预测失败的代价如何？
- 如何通过查表法减少分支？
- 位运算如何替代条件语句？
- Branchless编程的适用场景是什么？
- CMOV等条件移动指令何时有用？

**减少依赖**
- 数据依赖如何限制ILP？
- 如何重组计算以打破依赖链？
- 循环展开如何增加并行性？
- 软件流水线（Software Pipelining）是什么？

### 6.3 多线程优化

#### 6.3.1 线程模型

**线程创建与管理**
- 线程池相比动态创建线程的优势是什么？
- 线程数量如何选择？
- Work Stealing和Work Sharing的区别是什么？
- 线程亲和性（Thread Affinity）如何设置？
- 上下文切换的开销有多大？

**并行模式**
- Fork-Join模式如何工作？
- Pipeline并行的设计要点是什么？
- 数据并行和任务并行如何选择？
- 分治算法（Divide-and-Conquer）的并行化策略是什么？

#### 6.3.2 同步优化

**减少同步开销**
- 粗粒度锁和细粒度锁的权衡是什么？
- 如何减少锁的持有时间？
- 读写锁何时比互斥锁更好？
- 无锁编程的适用场景是什么？
- Hazard Pointer和RCU的原理是什么？

**避免竞争**
- 如何通过数据分区减少竞争？
- Thread-local Storage的使用场景是什么？
- Per-CPU数据结构如何设计？
- Lock-free Queue如何实现？
- 原子操作的内存顺序（memory_order）如何选择？

---

## 第七章：高级主题

### 7.1 处理器微架构

#### 7.1.1 现代微架构

**Intel架构演进**
- Sandy Bridge到最新架构的主要改进是什么？
- Golden Cove、Gracemont等微架构的特点是什么？
- 大小核（P-core/E-core）混合架构的设计思想是什么？
- Intel的10nm/7nm工艺带来了哪些架构改进？

**AMD架构演进**
- Zen到Zen 4架构的演进路径是什么？
- Chiplet设计的优势和挑战是什么？
- AMD的Infinity Fabric如何工作？
- AMD的NUMA拓扑有什么特点？

**ARM架构**
- ARM的架构特点是什么？
- Neoverse等服务器CPU的性能如何？
- ARM的内存模型与x86有何不同？
- ARM的SVE（Scalable Vector Extension）是什么？

#### 7.1.2 能效优化

**功耗管理**
- CPU的功耗来源有哪些？
- 动态功耗和静态功耗的区别是什么？
- P-state和C-state分别控制什么？
- Turbo Boost如何工作？
- 热设计功耗（TDP）的含义是什么？

**能效优化技术**
- DVFS（动态电压频率调节）如何工作？
- Race-to-Idle策略是什么？
- 如何在性能和功耗之间权衡？
- 能效比（Performance per Watt）如何衡量？

### 7.2 虚拟化与安全

#### 7.2.1 硬件虚拟化

**虚拟化扩展**
- Intel VT-x和AMD-V的主要特性是什么？
- EPT/NPT（扩展页表）如何加速地址转换？
- VM-exit的开销有多大？如何减少？
- 直通（Passthrough）和虚拟化I/O的区别是什么？

**容器与虚拟化**
- 容器相比虚拟机的性能优势在哪里？
- Cgroup和Namespace如何隔离资源？
- 容器中的NUMA感知如何配置？

#### 7.2.2 硬件安全

**安全特性**
- NX位（No-Execute）如何防止代码注入？
- SMEP/SMAP如何防止特权升级？
- Intel SGX和AMD SEV的作用是什么？

**侧信道攻击**
- Spectre和Meltdown的原理是什么？
- 这些漏洞如何利用CPU的推测执行？
- 缓解措施（如Retpoline）如何工作？
- 侧信道攻击对性能的影响有多大？

### 7.3 异构计算

#### 7.3.1 CPU-GPU协同

**异构架构**
- CPU和GPU的架构差异是什么？
- APU（加速处理单元）的设计思想是什么？
- 统一内存架构（UMA）在异构计算中的作用是什么？
- HSA（异构系统架构）标准是什么？

**CPU-GPU数据传输**
- PCIe传输的瓶颈在哪里？
- NVLink等高速互连的优势是什么？
- CPU和GPU的NUMA关系如何影响性能？
- Zero-copy技术如何减少数据传输？

#### 7.3.2 专用加速器

**AI加速器**
- TPU、NPU等AI加速器的架构特点是什么？
- 脉动阵列（Systolic Array）的原理是什么？
- AI加速器如何与CPU协同工作？

**FPGA与ASIC**
- FPGA在数据中心的应用场景是什么？
- FPGA相比CPU和GPU的优势在哪里？
- 软硬件协同设计的考虑因素有哪些？

---

## 第八章：性能调优实战

### 8.1 性能调优流程

#### 8.1.1 性能目标设定

**基准测试**
- 如何设计有代表性的基准测试？
- 性能指标如何选择（延迟、吞吐量、能效）？
- 如何建立性能基线？
- 性能退化如何监控？

**瓶颈识别**
- 性能瓶颈的常见类型有哪些？
- 如何使用Top-down方法定位瓶颈？
- 如何区分CPU、内存、I/O瓶颈？
- 如何量化优化的潜在收益？

#### 8.1.2 优化策略

**优化优先级**
- 如何确定优化的优先顺序？
- 热点代码（Hot Path）如何识别？
- 哪些优化收益最大？
- 过早优化的风险是什么？

**优化验证**
- 如何验证优化的有效性？
- A/B测试如何设计？
- 性能回归如何预防？
- 优化的可维护性如何评估？

### 8.2 典型场景优化

#### 8.2.1 计算密集型优化

**向量化优化**
- 如何识别可向量化的代码？
- 手动向量化的技巧有哪些？
- 向量化瓶颈如何解决？
- 如何衡量向量化效率？

**并行化**
- 如何识别并行机会？
- 并行粒度如何选择？
- 负载均衡如何实现？
- 并行开销如何控制？

#### 8.2.2 内存密集型优化

**减少内存访问**
- 如何通过算法改进减少内存访问？
- 寄存器阻塞（Register Blocking）如何实现？
- 数据复用如何最大化？

**优化内存访问模式**
- 如何实现顺序访问？
- Strided访问如何优化？
- Gather/Scatter操作如何避免？
- 预取如何正确使用？

#### 8.2.3 多线程优化实战

**并发性能调优**
- 线程数如何调优？
- 负载不均衡如何解决？
- 锁竞争如何诊断和优化？
- NUMA感知如何实现？

**可扩展性优化**
- 如何评估可扩展性？
- Amdahl定律和Gustafson定律的应用是什么？
- 串行瓶颈如何消除？
- 通信开销如何降低？

---

## 附录：综合性问题

### A. 系统设计

**如何为高性能计算任务设计系统架构？**
**CPU选型的考虑因素有哪些？**
**如何在NUMA系统上部署大规模应用？**

### B. 故障诊断

**如何诊断CPU性能异常？**
**NUMA不均衡如何排查？**
**如何识别和解决伪共享问题？**

### C. 前沿技术

**CXL（Compute Express Link）如何改变内存架构？**
**处理内存（Processing-in-Memory）的前景如何？**
**光计算和量子计算对CPU架构的影响是什么？**

---

**说明**：本大纲系统覆盖了CPU架构的核心知识体系，特别详细地阐述了NUMA架构（第四章）的方方面面，包括基础原理、Cache一致性、编程接口、性能调优和实际应用。面试官可以根据候选人的职位要求和背景，选择不同深度的问题进行考核。对于系统工程师、性能优化工程师或高性能计算领域的候选人，NUMA相关知识是必考内容。

