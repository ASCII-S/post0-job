---
created: '2025-10-25'
last_reviewed: '2025-11-04'
next_review: '2025-11-14'
review_count: 3
difficulty: medium
mastery_level: 0.55
tags:
- 熟悉CPU架构
- 熟悉CPU架构/什么是NUMA（Non-Uniform_Memory_Access）？.md
related_outlines: []
---
# 什么是NUMA（Non-Uniform Memory Access）？

## 面试标准答案

NUMA（非统一内存访问）是一种多处理器计算机系统的内存架构。在NUMA架构中，每个CPU都有自己的本地内存，访问本地内存的速度快，访问远程内存（其他CPU的本地内存）的速度慢，因此内存访问时间是非统一的。这与传统的UMA（统一内存访问）架构不同，UMA中所有CPU访问共享内存的速度相同。NUMA架构主要用于提高多处理器系统的可扩展性，减少内存总线的竞争，但需要操作系统和应用程序进行NUMA感知优化以获得最佳性能。

## 详细讲解

### 1. NUMA架构概述

NUMA是现代多核服务器系统中广泛采用的内存架构。随着处理器核心数量的增加，传统的UMA（Uniform Memory Access，统一内存访问）架构面临内存带宽瓶颈——所有CPU通过共享总线访问同一块内存，导致竞争和延迟。

NUMA通过将系统划分为多个**NUMA节点（Node）**来解决这个问题。每个NUMA节点包含：
- 一个或多个CPU核心
- 本地内存（Local Memory）
- 本地I/O资源

### 2. NUMA的核心特点
    
#### 2.1 内存访问的非统一性

NUMA最显著的特点是内存访问延迟的差异：

- **本地访问（Local Access）**：CPU访问自己节点的本地内存，延迟低、速度快
- **远程访问（Remote Access）**：CPU访问其他节点的内存，需要通过互连总线（如Intel的QPI/UPI、AMD的Infinity Fabric），延迟高、速度慢

通常远程访问的延迟是本地访问的1.5到2倍。

#### 2.2 可扩展性优势

NUMA架构通过以下方式提高系统可扩展性：
- 减少内存总线竞争
- 增加总体内存带宽（每个节点有独立的内存控制器）
- 支持更多CPU核心和更大内存容量

### 3. NUMA架构示意

```
+----------------+              +----------------+
|   NUMA Node 0  |              |   NUMA Node 1  |
|                |              |                |
|  CPU 0-7       |   QPI/UPI    |  CPU 8-15      |
|  Local Memory  |<------------>|  Local Memory  |
|  (64GB)        |   Interconn  |  (64GB)        |
|                |              |                |
+----------------+              +----------------+
     快速访问                          快速访问
     慢速访问 -----------------------> 慢速访问
```

### 4. NUMA的关键概念

#### 4.1 NUMA Distance

表示不同节点间的访问"距离"，通过`numactl --hardware`可以查看：
- 本地节点：距离为10（基准值）
- 远程节点：距离通常为20、21等（表示访问延迟比例）

#### 4.2 CPU亲和性（CPU Affinity）

将进程或线程绑定到特定的CPU核心，确保在同一NUMA节点上运行。

#### 4.3 内存策略（Memory Policy）

- **本地分配（Local Allocation）**：优先在当前CPU的本地节点分配内存
- **交错分配（Interleave）**：在多个节点间均匀分配内存
- **优先节点（Preferred）**：优先在指定节点分配，满了再去其他节点
- **绑定节点（Bind）**：强制在指定节点分配

### 5. NUMA对性能的影响

#### 5.1 正面影响
- 本地内存访问带宽高、延迟低
- 多个节点可并行访问各自内存，总带宽增加
- 更好的缓存局部性

#### 5.2 负面影响
- 远程内存访问性能下降
- 进程在节点间迁移可能导致性能抖动
- 不当的内存分配策略会降低性能

### 6. NUMA优化实践

#### 6.1 查看NUMA配置

```bash
# 查看NUMA节点信息
numactl --hardware

# 查看进程的NUMA状态
numastat -p <pid>

# 查看系统NUMA统计
cat /proc/buddyinfo
cat /sys/devices/system/node/node*/meminfo
```

#### 6.2 NUMA绑定

```bash
# 在Node 0上运行程序
numactl --cpunodebind=0 --membind=0 ./myapp

# 交错内存分配
numactl --interleave=all ./myapp
```

#### 6.3 程序优化建议

1. **数据局部性**：尽量让线程访问本地内存
2. **线程绑定**：使用CPU亲和性绑定线程到固定核心
3. **内存预分配**：在程序启动时完成内存分配，避免后续的远程分配
4. **First-touch策略**：Linux默认在首次访问内存时分配到当前CPU的节点
5. **避免跨节点通信**：设计算法时考虑数据分区

### 7. NUMA在大语言模型推理中的应用

在大模型推理场景中，NUMA优化尤为重要：

- **模型参数分布**：将模型参数分配到对应GPU所在的NUMA节点
- **数据预处理**：在相同NUMA节点上进行数据准备和推理
- **多实例部署**：每个推理实例绑定到特定NUMA节点，避免跨节点访问

### 8. NUMA vs UMA

| 特性         | UMA            | NUMA                         |
| ------------ | -------------- | ---------------------------- |
| 内存访问延迟 | 统一           | 非统一（本地快，远程慢）     |
| 可扩展性     | 受限于总线带宽 | 更好，每个节点独立内存控制器 |
| 编程复杂度   | 简单           | 需要NUMA感知优化             |
| 适用场景     | 小规模多核系统 | 大规模多路服务器             |

### 9. 常见问题

**Q: 如何判断系统是否是NUMA架构？**
```bash
# 方法1
lscpu | grep NUMA

# 方法2
numactl --hardware
```

**Q: NUMA导致性能下降怎么办？**
- 检查内存分配是否跨节点
- 使用numastat查看NUMA统计
- 考虑禁用NUMA（在BIOS中）或使用`numa=off`内核参数（不推荐）
- 优化程序的内存访问模式

**Q: Docker容器如何利用NUMA？**
```bash
# 绑定容器到特定NUMA节点
docker run --cpuset-cpus="0-7" --cpuset-mems="0" myimage
```

## 参考文献

1. [NUMA (Non-Uniform Memory Access): An Overview](https://queue.acm.org/detail.cfm?id=2513149) - ACM Queue
2. [What is NUMA?](https://www.kernel.org/doc/html/latest/vm/numa.html) - Linux Kernel Documentation
3. [NUMA Deep Dive Series](https://frankdenneman.nl/2016/07/06/introduction-2016-numa-deep-dive-series/) - Frank Denneman
4. [Understanding NUMA Architecture](https://software.intel.com/content/www/us/en/develop/articles/optimizing-applications-for-numa.html) - Intel Developer Zone
5. [Red Hat Enterprise Linux NUMA Support](https://access.redhat.com/documentation/en-us/red_hat_enterprise_linux/7/html/virtualization_tuning_and_optimization_guide/sect-virtualization_tuning_optimization_guide-numa-numa_and_libvirt) - Red Hat Documentation

