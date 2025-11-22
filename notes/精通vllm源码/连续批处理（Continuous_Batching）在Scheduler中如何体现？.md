---
created: '2025-11-23'
last_reviewed: '2025-11-23'
next_review: '2025-11-23'
review_count: 0
difficulty: medium
mastery_level: 0.0
tags:
- 精通vllm源码
- 精通vllm源码/连续批处理（Continuous_Batching）在Scheduler中如何体现？.md
related_outlines: []
---

# 连续批处理（Continuous Batching）在Scheduler中如何体现？

## 1. 什么是连续批处理？

连续批处理（Continuous Batching）是 vLLM 的核心特性之一，它允许在推理过程中动态地添加和移除请求，与传统静态批处理相比，大大提高了 GPU 利用率和整体吞吐量。

### 1.1 传统静态批处理的限制
- **固定批次大小**：必须等待整个批次完成后才能处理下一批
- **低 GPU 利用率**：短序列完成后 GPU 空闲，长序列完成后短序列等待
- **高延迟**：个别长序列会阻塞整个批次

### 1.2 连续批处理的优势
- **动态批次管理**：可以随时添加新请求，移除完成的请求
- **高 GPU 利用率**：保持 GPU 持续工作，减少空闲时间
- **低延迟**：短序列可以快速完成，不等待长序列

## 2. Scheduler 中的连续批处理实现

### 2.1 Scheduler 的核心职责

```python
# vllm/core/scheduler.py
class Scheduler:
    def __init__(self, scheduler_config: SchedulerConfig,
                 cache_engine: CacheEngine):
        self.scheduler_config = scheduler_config
        self.cache_engine = cache_engine

        # 核心数据结构
        self.running: List[SequenceGroup] = []  # 当前正在执行的序列组
        self.waiting: List[SequenceGroup] = []  # 等待执行的序列组
        self.swapped: List[SequenceGroup] = []  # 被交换到内存的序列组

        self.prompt_limit = scheduler_config.max_num_batched_tokens
```

### 2.2 连续批处理的关键方法

#### 2.2.1 schedule() 方法 - 调度核心

```python
def schedule(self) -> SchedulerOutputs:
    """执行调度决策，实现连续批处理的核心逻辑"""

    # 1. 检查已完成的序列
    self._process_finished_sequences()

    # 2. 从内存中恢复被交换的序列
    self._swap_in_sequences()

    # 3. 预填充新的序列（prompt阶段）
    self._prefill_sequences()

    # 4. 解码当前运行的序列（decode阶段）
    self._decode_sequences()

    # 5. 将内存不足的序列交换出去
    self._swap_out_sequences()

    # 6. 准备输出
    return self._prepare_scheduler_outputs()
```

#### 2.2.2 动态批次管理

```python
def _prefill_sequences(self):
    """处理等待队列中的新序列"""
    while self.waiting and not self._is_full():
        seq_group = self.waiting[0]

        # 检查是否有足够的 token 空间
        num_prompt_tokens = seq_group.get_num_prompt_tokens()
        if not self._can_prefill(num_prompt_tokens):
            break

        # 移动到运行队列
        self.waiting.pop(0)
        self.running.append(seq_group)

        # 分配 KV Cache 空间
        self._allocate_kv_cache(seq_group, num_prompt_tokens)
```

#### 2.2.3 序列完成检查

```python
def _process_finished_sequences(self):
    """处理已完成的序列，释放资源"""
    finished_sequences = []

    for seq_group in self.running:
        if seq_group.is_finished():
            finished_sequences.append(seq_group)
            # 释放 KV Cache 空间
            self.cache_engine.free(seq_group)

    # 从运行队列中移除已完成的序列
    for seq_group in finished_sequences:
        self.running.remove(seq_group)
```

### 2.3 KV Cache 管理与连续批处理

#### 2.3.1 KV Cache 分配

```python
# vllm/core/scheduler.py
def _allocate_kv_cache(self, seq_group: SequenceGroup,
                      num_tokens: int) -> bool:
    """为序列分配 KV Cache 空间"""

    # 检查是否有足够的物理块
    num_blocks = self._calculate_blocks_needed(num_tokens)

    if self.cache_engine.has_free_blocks(num_blocks):
        # 分配物理块
        blocks = self.cache_engine.allocate(num_blocks)
        seq_group.allocate(blocks)
        return True
    else:
        return False  # 内存不足，需要交换或拒绝
```

#### 2.3.2 内存交换机制

```python
def _swap_out_sequences(self):
    """当内存不足时，将部分序列交换到内存"""
    if not self.cache_engine.should_swap():
        return

    # 选择候选序列（通常是较长或优先级较低的序列）
    candidates = self._select_swap_candidates()

    for seq_group in candidates:
        if self.cache_engine.swap_out(seq_group):
            self.running.remove(seq_group)
            self.swapped.append(seq_group)
```

## 3. 连续批处理的工作流程

### 3.1 请求到达

```
新请求 → 等待队列 (waiting) → 调度器检查资源 → 进入运行队列 (running)
```

### 3.2 执行过程

```
运行中的序列 → Pre-fill (处理prompt) → Decode (生成token) →
              ↗                           ↘
          检查完成状态              检查内存使用情况
              ↘                           ↗
          释放资源 ←───────────────────────
```

### 3.3 序列完成

```
序列完成 → 释放 KV Cache → 从运行队列移除 → 返回结果
```

## 4. 连续批处理的优化策略

### 4.1 水印机制 (Watermark)

```python
# vllm/core/scheduler.py
def _should_preempt(self, seq_group: SequenceGroup) -> bool:
    """基于水印机制决定是否抢占序列"""

    # 当前 token 数量
    current_tokens = self.get_num_batched_tokens()

    # 水印：防止频繁的序列交换
    watermark = self.scheduler_config.watermark

    return current_tokens > watermark
```

### 4.2 FCFS (First-Come-First-Served) 调度

```python
def _get_next_seq_group(self) -> Optional[SequenceGroup]:
    """FCFS 调度策略"""
    return self.waiting[0] if self.waiting else None
```

### 4.3 优先级调度（扩展）

```python
# 一些实现支持优先级调度
def _get_next_seq_group_priority(self) -> Optional[SequenceGroup]:
    """基于优先级的调度策略"""
    if not self.waiting:
        return None

    # 按优先级排序
    self.waiting.sort(key=lambda x: x.priority, reverse=True)
    return self.waiting[0]
```

## 5. 与传统批处理的对比

### 5.1 批处理示例对比

#### 传统静态批处理：
```
Batch 1: [seq1(100 tokens), seq2(50 tokens), seq3(80 tokens)] → 完成
Batch 2: [seq4(30 tokens), seq5(60 tokens)] → 完成
```
- 所有序列必须同时完成
- 短序列等待长序列
- GPU 利用率低

#### vLLM 连续批处理：
```
Time t1: [seq1, seq2, seq3] → seq2完成 (50 tokens)
Time t2: [seq1, seq3, seq4] → seq4完成 (30 tokens)
Time t3: [seq1, seq3, seq5] → seq5完成 (60 tokens)
Time t4: [seq1, seq3] → seq3完成 (80 tokens)
Time t5: [seq1] → seq1完成 (100 tokens)
```
- 序列可以随时加入和退出
- GPU 持续工作
- 低延迟，高吞吐量

### 5.2 性能提升数据

根据 vLLM 论文和实际测试：

| 指标 | 传统批处理 | vLLM 连续批处理 | 提升 |
|------|------------|-----------------|------|
| GPU 利用率 | 30-60% | 80-95% | 2-3x |
| 吞吐量 | 基准 | 2.2-4x | 2-4x |
| 延迟 | 高 | 低 | 50-80% |
| 请求公平性 | 差 | 好 | 显著改善 |

## 6. 关键代码分析

### 6.1 Scheduler调度状态机

```python
# vllm/core/scheduler.py
class Scheduler:
    def step(self) -> List[SequenceGroupMetadata]:
        """执行一步调度"""

        # 1. 获取完成的序列
        finished = self._get_finished_sequences()

        # 2. 更新运行队列
        self._update_running_queue(finished)

        # 3. 处理等待队列
        self._process_waiting_queue()

        # 4. 内存管理
        self._manage_memory()

        # 5. 准备批次元数据
        return self._prepare_batch_metadata()
```

### 6.2 批次构建

```python
def _build_batch(self) -> SchedulerOutputs:
    """构建输出批次"""

    # 收集当前运行的序列组
    scheduled_seq_groups = []

    for seq_group in self.running:
        # 获取序列组的当前状态
        seq_group_metadata = seq_group.get_seq_group_metadata()
        scheduled_seq_groups.append(seq_group_metadata)

    # 构建调度输出
    return SchedulerOutputs(
        scheduled_seq_groups=scheduled_seq_groups,
        prompt_run=True if any(seq_group.is_prefill()
                             for seq_group in self.running) else False,
        num_batched_tokens=self.get_num_batched_tokens(),
        blocks_to_swap_in=self.blocks_to_swap_in,
        blocks_to_swap_out=self.blocks_to_swap_out,
        blocks_to_copy=self.blocks_to_copy,
    )
```

## 7. 监控和调试

### 7.1 调度器状态监控

```python
def get_scheduler_stats(self) -> Dict[str, Any]:
    """获取调度器统计信息"""
    return {
        "running": len(self.running),
        "waiting": len(self.waiting),
        "swapped": len(self.swapped),
        "num_batched_tokens": self.get_num_batched_tokens(),
        "num_free_gpu_blocks": self.cache_engine.get_num_free_gpu_blocks(),
        "num_free_cpu_blocks": self.cache_engine.get_num_free_cpu_blocks(),
    }
```

### 7.2 性能分析指标

- **吞吐量**：每秒处理的 token 数
- **延迟**：单个请求的平均完成时间
- **GPU 利用率**：GPU 计算资源的使用效率
- **内存利用率**：KV Cache 的使用效率
- **请求公平性**：不同长度请求的处理公平性

## 8. 总结

vLLM 的连续批处理通过 Scheduler 的智能调度实现了：

1. **动态批次管理**：实时添加/移除序列，保持 GPU 持续工作
2. **高效内存管理**：KV Cache 的动态分配和交换机制
3. **智能调度策略**：基于资源状况的调度决策
4. **性能优化**：显著的吞吐量提升和延迟降低

这种设计使得 vLLM 在处理多样化的推理请求时能够充分发挥硬件性能，特别是在实际生产环境中处理不同长度、不同模式的请求时表现出色。连续批处理是 vLLM 成为高性能推理引擎的核心技术之一。