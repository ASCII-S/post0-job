---
created: '2025-10-25'
last_reviewed: '2025-10-27'
next_review: '2025-10-29'
review_count: 1
difficulty: medium
mastery_level: 0.23
tags:
- 精通vllm源码
- 精通vllm源码/请画出vLLM主要模块之间的依赖关系图.md
related_outlines: []
---
# 请画出vLLM主要模块之间的依赖关系图

## 面试标准答案（可背诵）

vLLM的核心架构分为四个层次：**API层**（LLMEngine）、**调度层**（Scheduler）、**执行层**（Worker/Executor）和**内存管理层**（BlockManager + PagedAttention）。依赖关系自上而下：LLMEngine接收请求并调用Scheduler进行调度，Scheduler根据BlockManager的内存状态分配KV cache块，然后将任务分发给Worker执行，Worker通过ModelRunner加载模型并使用PagedAttention进行高效推理。关键创新是PagedAttention和BlockManager的解耦设计，使得KV cache可以像操作系统虚拟内存一样分页管理，大幅提升了吞吐量。

---

## vLLM架构详解

### 一、整体架构概览

vLLM采用分层设计，将大语言模型推理服务拆分成清晰的模块，各司其职又相互协作。整体可分为以下几层：

```
┌─────────────────────────────────────────────────────────┐
│                      用户请求入口                          │
│                    (API Server/Client)                   │
└────────────────────────┬────────────────────────────────┘
                         │
                         ▼
┌─────────────────────────────────────────────────────────┐
│                   API层 - LLMEngine                      │
│  • 请求管理                                               │
│  • 生成控制                                               │
│  • 流式输出                                               │
└────────────────────────┬────────────────────────────────┘
                         │
                         ▼
┌─────────────────────────────────────────────────────────┐
│              调度层 - Scheduler + BlockManager            │
│  • Scheduler: 请求调度、批处理决策                         │
│  • BlockManager: KV Cache内存分配与管理                   │
└────────────────────────┬────────────────────────────────┘
                         │
                         ▼
┌─────────────────────────────────────────────────────────┐
│           执行层 - Worker + ModelRunner                  │
│  • Worker: 分布式执行协调                                 │
│  • ModelRunner: 模型加载与推理执行                        │
└────────────────────────┬────────────────────────────────┘
                         │
                         ▼
┌─────────────────────────────────────────────────────────┐
│            计算核心 - PagedAttention Kernel              │
│  • 分页式KV Cache访问                                     │
│  • 高效GPU计算                                            │
└─────────────────────────────────────────────────────────┘
```

### 二、核心模块详解

#### 1. **LLMEngine（引擎层）**

**职责**：
- 对外暴露统一的推理接口
- 管理请求的生命周期（接收、处理、完成）
- 协调Scheduler和Worker完成推理任务
- 支持同步和异步生成模式

**关键方法**：
- `add_request()`: 添加新请求到队列
- `step()`: 执行一次推理迭代
- `abort_request()`: 取消请求

**依赖关系**：
- 依赖 `Scheduler` 进行请求调度
- 依赖 `Worker` 执行实际计算
- 管理 `OutputProcessor` 处理输出

#### 2. **Scheduler（调度层）**

**职责**：
- 决定哪些请求可以被执行（调度策略）
- 实现Continuous Batching（连续批处理）
- 处理抢占和重计算逻辑
- 优化批处理大小以最大化吞吐量

**核心逻辑**：
```
对于每个调度周期：
1. 从waiting队列选择可运行的请求
2. 检查running队列中的请求状态
3. 根据内存情况决定是否抢占低优先级请求
4. 构造SchedulerOutputs返回给LLMEngine
```

**依赖关系**：
- 依赖 `BlockManager` 查询和分配内存
- 被 `LLMEngine` 调用
- 输出 `SchedulerOutputs` 给执行层

#### 3. **BlockManager（内存管理层）**

**职责**：
- 管理KV Cache的物理和逻辑块映射
- 实现Copy-on-Write机制（用于Beam Search等场景）
- 处理内存分配、释放和抢占
- 维护物理块的空闲列表

**核心概念**：
- **逻辑块（Logical Block）**：每个序列的KV Cache被分成固定大小的逻辑块
- **物理块（Physical Block）**：GPU显存中的实际存储块
- **块表（Block Table）**：逻辑块到物理块的映射表

**关键方法**：
- `can_allocate()`: 检查是否有足够内存
- `allocate()`: 分配物理块
- `free()`: 释放物理块
- `fork()`: 实现序列分叉（CoW）

**依赖关系**：
- 被 `Scheduler` 频繁查询和调用
- 管理的块信息被 `PagedAttention` 使用

#### 4. **Worker（执行层）**

**职责**：
- 在单个GPU进程中执行推理
- 管理模型加载和初始化
- 协调分布式推理（张量并行、流水线并行）
- 缓存KV Cache并执行实际的forward计算

**关键组件**：
- `CacheEngine`: 在GPU上分配和管理KV Cache物理内存
- `ModelRunner`: 封装模型执行逻辑

**依赖关系**：
- 被 `LLMEngine` 调用执行推理步骤
- 依赖 `ModelRunner` 执行模型前向传播
- 依赖 `CacheEngine` 管理GPU上的KV Cache

#### 5. **ModelRunner（模型执行层）**

**职责**：
- 准备模型输入（token ids, positions, block tables等）
- 调用模型的forward方法
- 采样生成下一个token
- 处理不同的采样参数（temperature, top_p等）

**执行流程**：
```
1. 准备输入数据（prepare_input）
2. 调用model.forward()
3. 从logits中采样下一个token
4. 返回采样结果
```

**依赖关系**：
- 被 `Worker` 调用
- 依赖加载的 `Model` 实例
- 调用 `PagedAttention` kernel进行attention计算

#### 6. **PagedAttention（计算核心）**

**职责**：
- 实现分页式的Attention计算
- 高效利用GPU内存和计算资源
- 支持不连续的KV Cache存储

**核心创新**：
传统Attention需要连续的KV Cache内存，PagedAttention通过块表可以访问分散存储的KV Cache块，就像操作系统的虚拟内存一样。

**计算过程**：
```
对于每个query token:
1. 根据block_table找到该序列的所有KV块
2. 逐块读取K和V
3. 计算attention scores和outputs
4. 累积最终结果
```

**依赖关系**：
- 被模型的Attention层调用
- 接收来自 `BlockManager` 的块表信息
- 直接访问 `CacheEngine` 管理的GPU显存

### 三、数据流示意

以一次推理迭代为例，展示数据在各模块间的流动：

```
1. 用户请求 → LLMEngine.add_request()
   └─> 请求加入waiting队列

2. LLMEngine.step() 触发调度
   └─> Scheduler.schedule()
       ├─> BlockManager.can_allocate() [检查内存]
       ├─> 选择requests加入running
       └─> 返回 SchedulerOutputs

3. LLMEngine 分发任务
   └─> Worker.execute_model(SchedulerOutputs)
       ├─> ModelRunner.prepare_input()
       │   └─> 构造 input_ids, positions, block_tables
       ├─> Model.forward()
       │   └─> Attention层调用 PagedAttention
       │       └─> 根据block_tables访问KV Cache
       └─> Sample下一个token

4. 返回结果
   └─> LLMEngine处理输出
       ├─> 更新序列状态
       ├─> 释放完成序列的blocks
       └─> 返回生成的tokens给用户
```

### 四、关键设计模式

#### 1. **解耦的内存管理**
BlockManager独立于执行层，使得内存分配决策可以在调度时高效完成，无需等待GPU操作。

#### 2. **Continuous Batching**
Scheduler可以在任意时刻添加新请求到batch中，无需等待整个batch完成，极大提升了系统吞吐量。

#### 3. **Copy-on-Write**
在Beam Search或Parallel Sampling时，多个序列可以共享相同的KV Cache块，只在需要修改时才复制。

#### 4. **分层抽象**
从API层到Kernel层的清晰分层，使得各模块可以独立优化和测试。

### 五、模块依赖关系图

```
                 ┌──────────────┐
                 │  LLMEngine   │
                 └───────┬──────┘
                         │
            ┌────────────┼────────────┐
            ▼            ▼            ▼
      ┌──────────┐ ┌──────────┐ ┌─────────┐
      │Scheduler │ │  Worker  │ │ Output  │
      │          │ │ (Manager)│ │Processor│
      └────┬─────┘ └────┬─────┘ └─────────┘
           │            │
           ▼            ▼
    ┌─────────────┐ ┌──────────┐
    │BlockManager │ │  Worker  │ (多个Worker实例)
    └─────────────┘ └────┬─────┘
                         │
            ┌────────────┼────────────┐
            ▼            ▼            ▼
      ┌──────────┐ ┌───────────┐ ┌─────────────┐
      │CacheEngine│ │ModelRunner│ │TP/PP Manager│
      └──────────┘ └─────┬─────┘ └─────────────┘
                         │
                         ▼
                  ┌──────────────┐
                  │    Model     │
                  │  (HuggingFace│
                  │   or Custom) │
                  └───────┬──────┘
                          │
                          ▼
                  ┌───────────────┐
                  │PagedAttention │
                  │    Kernel     │
                  └───────────────┘
```

### 六、性能优势来源

1. **内存效率**: PagedAttention + BlockManager 消除了内存碎片和预留浪费
2. **高吞吐**: Continuous Batching 使GPU利用率最大化
3. **灵活调度**: Scheduler可根据实时状态做抢占和批处理决策
4. **并行友好**: Worker设计天然支持张量并行和流水线并行

### 七、扩展性设计

vLLM的模块化设计使其易于扩展：

- **新的调度策略**: 只需修改Scheduler逻辑
- **新的采样方法**: 在ModelRunner中添加新的sampler
- **新的模型架构**: 实现新的Model类并注册
- **分布式策略**: 扩展Worker的并行管理

---

## 参考文献

1. **Efficient Memory Management for Large Language Model Serving with PagedAttention**  
   Woosuk Kwon, Zhuohan Li, Siyuan Zhuang, et al.  
   SOSP 2023  
   [https://arxiv.org/abs/2309.06180](https://arxiv.org/abs/2309.06180)

2. **vLLM Official Documentation**  
   [https://docs.vllm.ai/en/latest/](https://docs.vllm.ai/en/latest/)

3. **vLLM GitHub Repository**  
   [https://github.com/vllm-project/vllm](https://github.com/vllm-project/vllm)

4. **Orca: A Distributed Serving System for Transformer-Based Generative Models**  
   Gyeong-In Yu, et al.  
   OSDI 2022  
   [https://www.usenix.org/conference/osdi22/presentation/yu](https://www.usenix.org/conference/osdi22/presentation/yu)

5. **vLLM Blog: How continuous batching enables 23x throughput in LLM inference**  
   [https://www.anyscale.com/blog/continuous-batching-llm-inference](https://www.anyscale.com/blog/continuous-batching-llm-inference)

