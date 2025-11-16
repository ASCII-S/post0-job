---
created: '2025-10-19'
last_reviewed: null
next_review: '2025-10-19'
review_count: 0
difficulty: medium
mastery_level: 0.0
tags:
- vllm
- vllm/一条prompt进入后，整个vllm是如何运作的呢？.md
related_outlines: []
---
# 一条prompt进入后，整个vLLM是如何运作的呢？

## 🎯 面试标准答案（可背诵版）

当一条prompt进入vLLM后，主要经历四个关键阶段：

1. **请求接收与调度**：AsyncLLMEngine接收请求并加入调度器队列，调度器根据优先级和资源情况进行批处理调度
2. **Prefill阶段**：对输入的prompt进行并行处理，计算所有token的KV Cache并存储到PagedAttention管理的分页内存中
3. **Decode阶段**：采用连续批处理技术，每次自回归生成一个token，动态更新KV Cache，直到生成结束标志
4. **结果返回**：通过流式或非流式方式将生成的文本返回给用户

核心优化点是**PagedAttention**将KV Cache分页管理（类似操作系统虚拟内存），将显存浪费降到4%以下，以及**连续批处理**动态调度多个请求，将GPU利用率提升到80%以上。

---

## 📚 详细技术讲解

### 一、整体架构

vLLM采用三层架构设计：

```
用户请求 → 主进程（API层）→ 引擎进程（调度层）→ 工作进程（执行层）
```

- **主进程（API Server）**：处理HTTP请求，对prompt进行tokenization等预处理，并将处理后的数据发送给引擎
- **引擎进程（LLMEngine/AsyncLLMEngine）**：核心调度层，负责请求队列管理、资源调度、KV Cache分配
- **工作进程（Worker/Executor）**：在GPU上实际执行模型推理计算

### 二、请求处理完整流程

#### 1. 请求接收与入队（Request Reception）

```python
# 用户发送prompt
prompt = "解释一下量子计算的原理"

# AsyncLLMEngine接收请求
request = {
    "request_id": "uuid-xxx",
    "prompt": prompt,
    "sampling_params": {...},  # 温度、top_p等参数
    "arrival_time": timestamp
}
```

请求被封装成`SequenceGroup`对象，包含：
- 输入token序列
- 采样参数（temperature, top_p, max_tokens等）
- 优先级和时间戳

#### 2. 调度决策（Scheduling）

vLLM的**统一调度器（Unified Scheduler）** 负责：

**a) 资源评估**
- 检查当前可用的GPU显存
- 计算每个请求需要的KV Cache空间
- 评估是否有足够的物理页（Physical Blocks）

**b) 请求优先级排序**
- 考虑请求到达时间（FCFS）
- 考虑prompt长度
- 支持自定义优先级策略

**c) 批处理组装**
- 从等待队列中选择多个请求组成batch
- 使用**连续批处理（Continuous Batching）**技术：不同请求可以处于不同生成阶段
- 一个请求完成时，立即从队列中补充新请求，避免GPU空闲

#### 3. Prefill阶段（并行处理输入）

这是第一次前向传播，处理完整的输入prompt：

**核心特点**：
- **并行计算**：输入的所有token可以并行处理（类似BERT的编码）
- **计算密集型**：需要计算prompt中所有token之间的attention
- **KV Cache生成**：为输入序列的每一层、每个token生成Key和Value向量

**PagedAttention的作用**：
```
传统方式：需要预先分配连续的显存空间
[Token1_KV][Token2_KV][Token3_KV]...[TokenN_KV] ← 必须连续

PagedAttention：分页存储，动态分配
Page1: [Token1_KV][Token2_KV][Token3_KV][Token4_KV]
Page2: [Token5_KV][Token6_KV][Token7_KV][Token8_KV]
...
通过逻辑-物理映射表管理 ← 可以非连续
```

**技术细节**：
- 每个物理页默认包含16个token的KV Cache
- 逻辑页和物理页通过Block Table映射
- 支持Copy-on-Write，多个请求可以共享相同的prompt前缀

#### 4. Decode阶段（自回归生成）

**核心特点**：
- **串行生成**：每次只生成一个新token
- **访存密集型**：需要读取之前所有token的KV Cache进行attention计算
- **动态更新**：每生成一个新token，就追加其KV Cache到对应的页中

**连续批处理的优势**：
```
时刻T1:
Request A: [prefill - 完成]
Request B: [decode - token 5]
Request C: [decode - token 12]

时刻T2: (Request A完成)
Request B: [decode - token 6]
Request C: [decode - token 13]
Request D: [prefill - 开始] ← 立即补充新请求

GPU始终保持高利用率！
```

**生成流程**：
1. 使用最新生成的token作为输入
2. 读取历史KV Cache（通过PagedAttention高效访问）
3. 计算attention，生成logits
4. 应用采样策略（temperature, top_k, top_p等）
5. 得到下一个token
6. 更新KV Cache到对应页
7. 检查停止条件（EOS token或达到max_tokens）

#### 5. 内存管理与调度优化

**PagedAttention的三大优势**：

1. **零碎片化**：显存利用率提升到96%以上
2. **动态扩展**：根据实际生成长度按需分配页
3. **共享机制**：支持Beam Search等需要复制状态的场景

**调度策略**：
- **Preemption（抢占）**：显存不足时，可以将低优先级请求的KV Cache换出到CPU内存
- **Swapping（交换）**：支持KV Cache在GPU和CPU之间迁移
- **Recomputation（重计算）**：极端情况下，可以丢弃KV Cache并重新计算

#### 6. 并行计算策略

vLLM支持多种并行方式组合：

- **张量并行（Tensor Parallel）**：将模型的权重矩阵切分到多张GPU
- **流水线并行（Pipeline Parallel）**：将模型的不同层分配到不同GPU
- **数据并行（Data Parallel）**：多个副本处理不同batch
- **专家并行（Expert Parallel）**：针对MoE模型的特殊并行

#### 7. 结果返回

**流式输出**：
```python
async for output in engine.generate(prompt):
    print(output.text, end='', flush=True)
```

**非流式输出**：
```python
outputs = await engine.generate(prompt)
print(outputs[0].text)
```

### 三、关键技术点总结

| 技术           | 作用              | 性能提升         |
| -------------- | ----------------- | ---------------- |
| PagedAttention | 分页管理KV Cache  | 显存利用率 >96%  |
| 连续批处理     | 动态调度多请求    | GPU利用率 >80%   |
| FlashAttention | 优化attention计算 | 推理速度提升2-4x |
| 统一调度器     | 智能资源分配      | 吞吐量提升10-20x |
| Prefix Caching | 共享公共前缀      | 减少重复计算     |

### 四、性能对比

相比传统推理框架（如HuggingFace Transformers + PyTorch）：
- **吞吐量**：提升10-20倍
- **显存占用**：减少50%以上
- **延迟**：首token延迟（TTFT）降低30-50%
- **并发能力**：可支持数百个并发请求

### 五、实际应用场景

1. **高吞吐场景**：批量文本生成、数据标注
2. **低延迟场景**：在线对话、实时翻译
3. **长文本场景**：文档分析、代码生成（利用PagedAttention高效管理长序列）
4. **多用户服务**：公有云API服务（连续批处理保证高并发）

---

## 🔑 核心要点

1. vLLM的核心创新是**PagedAttention**（分页内存管理）和**Continuous Batching**（连续批处理）
2. 推理分为**Prefill**（并行处理输入）和**Decode**（串行生成输出）两个阶段
3. 统一调度器负责资源分配、请求优先级排序和批处理组装
4. 支持多种并行策略和内存管理机制（抢占、交换、重计算）
5. 相比传统方案，性能提升10-20倍，是目前最高效的LLM推理框架之一


---

## 相关笔记
<!-- 自动生成 -->

- [请描述大语言模型的推理过程，包括prefill和decode阶段的特点？](notes/vllm/请描述大语言模型的推理过程，包括prefill和decode阶段的特点？.md) - 相似度: 31% | 标签: vllm, vllm/请描述大语言模型的推理过程，包括prefill和decode阶段的特点？.md

