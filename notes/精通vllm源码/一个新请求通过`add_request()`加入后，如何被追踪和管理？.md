---
created: '2025-10-25'
last_reviewed: null
next_review: '2025-10-25'
review_count: 0
difficulty: medium
mastery_level: 0.0
tags:
- 精通vllm源码
- 精通vllm源码/一个新请求通过`add_request()`加入后，如何被追踪和管理？.md
related_outlines: []
---
# 一个新请求通过`add_request()`加入后，如何被追踪和管理？

## 面试标准答案（精简版）

当请求通过`add_request()`加入vLLM后，会经历以下追踪和管理流程：

**核心流程**：
1. **创建数据结构**：创建`Sequence`和`SequenceGroup`对象来封装请求，分配唯一的request_id和sequence_id
2. **加入等待队列**：请求进入Scheduler的`waiting`队列等待调度
3. **状态转换**：根据资源情况在waiting→running→swapped状态间转换，由Scheduler管理
4. **资源分配**：BlockManager为请求分配KV Cache的物理块
5. **执行追踪**：通过sequence_id在各个模块间传递，Worker执行后更新状态
6. **结果回调**：生成完成后触发回调函数，清理资源并返回结果

**关键数据结构**：SequenceGroup（包含采样参数和多个Sequence）、SequenceStatus（WAITING/RUNNING/SWAPPED/FINISHED）、BlockTable（逻辑块到物理块的映射）

---

## 详细解析

### 1. 请求的入口：`add_request()`方法

#### 1.1 方法签名与位置

在`vllm/engine/llm_engine.py`中的`LLMEngine`类中：

```python
def add_request(
    self,
    request_id: str,
    prompt: Optional[str],
    sampling_params: SamplingParams,
    prompt_token_ids: Optional[List[int]] = None,
    arrival_time: Optional[float] = None,
    lora_request: Optional[LoRARequest] = None,
) -> None:
```

#### 1.2 执行流程

**Step 1: 参数预处理**
```python
# 如果没有提供arrival_time，使用当前时间
if arrival_time is None:
    arrival_time = time.time()

# Tokenization（如果提供的是文本而非token ids）
if prompt_token_ids is None:
    prompt_token_ids = self.tokenizer.encode(prompt)
```

**Step 2: 创建核心数据结构**
```python
# 创建SequenceGroup
seq_id = next(self.seq_counter)  # 生成唯一的sequence ID
seq = Sequence(seq_id, prompt, prompt_token_ids, block_size)

# 创建SequenceGroup（一个请求可能包含多个序列，如beam search）
seq_group = SequenceGroup(
    request_id=request_id,
    seqs=[seq],
    sampling_params=sampling_params,
    arrival_time=arrival_time,
    lora_request=lora_request,
)
```

**Step 3: 添加到Scheduler**
```python
# 将SequenceGroup添加到调度器的等待队列
self.scheduler.add_seq_group(seq_group)
```

### 2. 核心数据结构详解

#### 2.1 Sequence类

定义在`vllm/sequence.py`中：

```python
class Sequence:
    def __init__(
        self,
        seq_id: int,
        prompt: str,
        token_ids: List[int],
        block_size: int,
    ):
        self.seq_id = seq_id              # 唯一标识符
        self.prompt = prompt               # 原始文本
        self.data = SequenceData(token_ids)  # Token序列
        self.output_logprobs = []          # 输出token的概率
        self.output_text = ""              # 生成的文本
        self.logical_token_blocks = []     # 逻辑块列表
        self.status = SequenceStatus.WAITING  # 初始状态
```

**关键属性**：
- `seq_id`: 全局唯一的序列ID
- `data`: 包含输入和输出token的`SequenceData`对象
- `logical_token_blocks`: 逻辑块索引列表，用于PagedAttention
- `status`: 当前状态（WAITING/RUNNING/SWAPPED/FINISHED_STOPPED/FINISHED_ABORTED等）

#### 2.2 SequenceGroup类

```python
class SequenceGroup:
    def __init__(
        self,
        request_id: str,
        seqs: List[Sequence],
        sampling_params: SamplingParams,
        arrival_time: float,
        lora_request: Optional[LoRARequest] = None,
    ):
        self.request_id = request_id       # 请求ID
        self.seqs_dict = {seq.seq_id: seq for seq in seqs}  # 序列字典
        self.sampling_params = sampling_params  # 采样参数
        self.arrival_time = arrival_time   # 到达时间
        self.prompt_logprobs = None        # Prompt的logprobs
        self.state = SequenceGroupState()  # 状态信息
```

**设计意图**：
- 一个请求（request）可能产生多个序列（如beam search时）
- SequenceGroup封装了一个请求的所有相关信息
- 便于批量管理和调度

#### 2.3 SequenceStatus枚举

```python
class SequenceStatus(enum.Enum):
    WAITING = enum.auto()           # 等待调度
    RUNNING = enum.auto()           # 正在执行
    SWAPPED = enum.auto()           # 被换出到CPU内存
    FINISHED_STOPPED = enum.auto()  # 正常结束（遇到EOS）
    FINISHED_LENGTH_CAPPED = enum.auto()  # 达到最大长度
    FINISHED_ABORTED = enum.auto()  # 被中止
    FINISHED_IGNORED = enum.auto()  # 被忽略
```

### 3. Scheduler中的追踪管理

#### 3.1 队列管理

在`vllm/core/scheduler.py`中，Scheduler维护三个主要队列：

```python
class Scheduler:
    def __init__(self, scheduler_config, cache_config):
        self.waiting: Deque[SequenceGroup] = deque()   # 等待队列
        self.running: List[SequenceGroup] = []         # 运行队列
        self.swapped: Deque[SequenceGroup] = deque()   # 换出队列
        
        self.block_manager = BlockSpaceManager(...)    # 内存管理器
```

#### 3.2 `add_seq_group()`实现

```python
def add_seq_group(self, seq_group: SequenceGroup) -> None:
    """添加新的序列组到等待队列"""
    # 设置所有序列的状态为WAITING
    for seq in seq_group.get_seqs():
        seq.status = SequenceStatus.WAITING
    
    # 添加到waiting队列的末尾
    self.waiting.append(seq_group)
```

#### 3.3 调度循环：`schedule()`方法

核心调度逻辑：

```python
def schedule(self) -> Tuple[List[SequenceGroupMetadata], SchedulerOutputs]:
    """调度器的核心方法，每个step调用一次"""
    
    # 1. 处理已完成的序列
    self._schedule_finished()
    
    # 2. 优先处理running队列（已有资源分配的序列）
    scheduler_outputs = SchedulerOutputs(
        scheduled_seq_groups=[],
        num_batched_tokens=0,
        blocks_to_swap_in={},
        blocks_to_swap_out={},
        blocks_to_copy={},
    )
    
    # 3. 尝试从waiting队列调度新请求
    while self.waiting:
        seq_group = self.waiting[0]
        
        # 检查是否有足够的KV Cache blocks
        can_allocate = self._can_allocate(seq_group)
        if not can_allocate:
            break  # 资源不足，无法调度更多
        
        # 分配blocks
        self._allocate(seq_group)
        
        # 从waiting移到running
        seq_group = self.waiting.popleft()
        self.running.append(seq_group)
        
        # 更新序列状态
        for seq in seq_group.get_seqs():
            seq.status = SequenceStatus.RUNNING
        
        scheduler_outputs.scheduled_seq_groups.append(seq_group)
    
    # 4. 如果资源不足，可能需要抢占或swap
    if self._need_preemption():
        self._preempt_by_recompute()  # 或 _preempt_by_swap()
    
    return scheduler_outputs
```

### 4. 资源分配：BlockManager

#### 4.1 Block分配流程

当序列从waiting转到running时，BlockManager分配KV Cache：

```python
def allocate(self, seq_group: SequenceGroup) -> None:
    """为序列组分配物理blocks"""
    for seq in seq_group.get_seqs(status=SequenceStatus.WAITING):
        # 分配逻辑块
        block_table = self._allocate_sequence(seq)
        
        # 存储block table（逻辑块 -> 物理块的映射）
        self.block_tables[seq.seq_id] = block_table
```

#### 4.2 Block Table结构

```python
# block_tables: Dict[int, BlockTable]
# 例如：
{
    seq_id_1: [PhysicalBlock(0), PhysicalBlock(5), PhysicalBlock(12)],
    seq_id_2: [PhysicalBlock(1), PhysicalBlock(6), PhysicalBlock(13)],
}
```

每个序列维护一个block table，记录其KV Cache存储在哪些物理块中。

### 5. 请求状态转换图

```
          add_request()
               ↓
          [WAITING]  ←─────┐
               ↓            │
         资源可用？          │
               ↓            │
          [RUNNING]         │
          /    |    \       │
         /     |     \      │
    生成中   资源不足  中止  │
       ↓       ↓       ↓    │
   继续运行 [SWAPPED]  ABORT
       ↓       ↓
   达到EOS  资源可用
       ↓       ↓
   [FINISHED]  ─────────────┘
```

**状态转换详解**：

1. **WAITING → RUNNING**：Scheduler检测到有足够资源，分配blocks
2. **RUNNING → SWAPPED**：内存不足时，低优先级序列被换出到CPU
3. **SWAPPED → RUNNING**：资源释放后，从CPU换回GPU
4. **RUNNING → FINISHED**：生成完成（EOS或达到max_tokens）
5. **任意状态 → ABORTED**：客户端取消或超时

### 6. 执行过程中的追踪

#### 6.1 SequenceGroupMetadata

调度器为每个被调度的序列生成元数据：

```python
@dataclass
class SequenceGroupMetadata:
    request_id: str                      # 请求ID
    is_prompt: bool                      # 是否是prefill阶段
    seq_data: Dict[int, SequenceData]    # 序列数据
    sampling_params: SamplingParams      # 采样参数
    block_tables: Dict[int, List[int]]   # Block映射表
```

这些元数据随着请求传递到Worker进行执行。

#### 6.2 Worker执行

在`vllm/worker/worker.py`中：

```python
def execute_model(
    self,
    seq_group_metadata_list: List[SequenceGroupMetadata],
) -> List[SamplerOutput]:
    """执行模型前向传播"""
    
    # 1. 准备输入
    input_tokens, input_positions, ... = self._prepare_inputs(
        seq_group_metadata_list
    )
    
    # 2. 准备attention metadata（包含block tables）
    attn_metadata = self._prepare_attention_metadata(
        seq_group_metadata_list
    )
    
    # 3. 执行模型
    hidden_states = self.model(
        input_ids=input_tokens,
        positions=input_positions,
        kv_caches=self.gpu_cache,
        attn_metadata=attn_metadata,
    )
    
    # 4. 采样
    sampler_output = self.model.sampler(
        hidden_states,
        sampling_metadata
    )
    
    return sampler_output
```

#### 6.3 结果处理

Engine收到Worker的输出后：

```python
def _process_model_outputs(
    self,
    output: SamplerOutput,
    scheduled_seq_groups: List[SequenceGroup],
) -> List[RequestOutput]:
    """处理模型输出，更新序列状态"""
    
    for seq_group, outputs in zip(scheduled_seq_groups, output):
        for seq, output in zip(seq_group.get_seqs(), outputs):
            # 添加新生成的token
            seq.append_token_id(
                token_id=output.token_id,
                logprobs=output.logprobs,
            )
            
            # 检查是否完成
            if output.token_id == self.tokenizer.eos_token_id:
                seq.status = SequenceStatus.FINISHED_STOPPED
            elif seq.get_len() >= self.max_model_len:
                seq.status = SequenceStatus.FINISHED_LENGTH_CAPPED
```

### 7. 请求完成与资源清理

#### 7.1 完成检测

```python
def _check_finished(self, seq_group: SequenceGroup) -> None:
    """检查序列组是否完成"""
    seqs = seq_group.get_seqs()
    
    # 如果所有序列都完成了
    if all(seq.is_finished() for seq in seqs):
        # 从running队列移除
        self.running.remove(seq_group)
        
        # 释放blocks
        self.block_manager.free(seq_group)
```

#### 7.2 资源释放

```python
def free(self, seq_group: SequenceGroup) -> None:
    """释放序列组占用的所有blocks"""
    for seq in seq_group.get_seqs():
        if seq.seq_id in self.block_tables:
            block_table = self.block_tables[seq.seq_id]
            for block in block_table:
                block.ref_count -= 1
                if block.ref_count == 0:
                    # 回收到free block pool
                    self.free_blocks.append(block)
            
            # 删除block table
            del self.block_tables[seq.seq_id]
```

#### 7.3 回调触发

```python
# 在AsyncLLMEngine中
async def generate(
    self,
    prompt: str,
    sampling_params: SamplingParams,
    request_id: str,
) -> AsyncIterator[RequestOutput]:
    """异步生成接口"""
    
    # 添加请求
    await self.add_request(request_id, prompt, sampling_params)
    
    # 持续获取结果
    while True:
        request_output = await self.request_tracker.get_request_output(
            request_id
        )
        
        yield request_output  # 返回给客户端
        
        if request_output.finished:
            break
```

### 8. 跨模块追踪机制

#### 8.1 ID传递链

```
Request ID (用户提供)
    ↓
Sequence ID (Engine生成)
    ↓
SequenceGroup (Scheduler管理)
    ↓
SequenceGroupMetadata (传递给Worker)
    ↓
Block Tables (BlockManager维护)
    ↓
SamplerOutput (Worker返回)
    ↓
RequestOutput (返回给用户)
```

#### 8.2 关键映射表

Engine维护多个映射表来追踪请求：

```python
class LLMEngine:
    def __init__(self):
        # request_id -> SequenceGroup
        self.request_tracker: Dict[str, SequenceGroup] = {}
        
        # seq_id -> Sequence
        self.sequences: Dict[int, Sequence] = {}
        
        # request_id -> 生成器（用于流式输出）
        self.generators: Dict[str, Generator] = {}
```

### 9. 实际代码示例

#### 9.1 完整的请求生命周期

```python
# 1. 添加请求
engine = LLMEngine(...)
request_id = "req_001"
prompt = "Explain quantum computing"
sampling_params = SamplingParams(temperature=0.7, max_tokens=100)

engine.add_request(request_id, prompt, sampling_params)
# 此时：Sequence对象被创建，状态为WAITING，在scheduler.waiting队列中

# 2. Engine step循环
while has_unfinished_requests():
    # Scheduler调度
    seq_group_metadata, scheduler_outputs = engine.scheduler.schedule()
    # 如果有资源：状态变为RUNNING，分配blocks
    
    # Worker执行
    output = engine.workers[0].execute_model(seq_group_metadata)
    
    # 处理输出
    request_outputs = engine._process_model_outputs(output, ...)
    
    # 更新状态、检查完成
    for request_output in request_outputs:
        if request_output.finished:
            # 状态变为FINISHED，释放资源
            engine._finalize_request(request_output.request_id)

# 3. 返回结果
final_output = engine.get_request_output(request_id)
```

#### 9.2 状态查询

```python
# 在任意时刻查询请求状态
def get_request_status(engine, request_id):
    seq_group = engine.request_tracker.get(request_id)
    if seq_group is None:
        return "NOT_FOUND"
    
    # 获取所有序列的状态
    statuses = [seq.status for seq in seq_group.get_seqs()]
    return statuses[0]  # 简化处理

# 使用
status = get_request_status(engine, "req_001")
# 返回: SequenceStatus.WAITING / RUNNING / FINISHED 等
```

### 10. 高级特性

#### 10.1 优先级调度

```python
class SequenceGroup:
    def __init__(self, ..., priority: int = 0):
        self.priority = priority

# Scheduler中按优先级排序
def schedule(self):
    # 对waiting队列按优先级排序
    self.waiting = deque(sorted(
        self.waiting,
        key=lambda sg: (-sg.priority, sg.arrival_time)
    ))
```

#### 10.2 Prefix Caching

当多个请求共享相同前缀时：

```python
# BlockManager检测共享前缀
def _get_cached_prefix(self, token_ids: List[int]) -> Optional[List[Block]]:
    """查找是否有缓存的前缀blocks"""
    # 在prefix cache中查找
    cached_blocks = self.prefix_cache.get(tuple(token_ids[:prefix_len]))
    if cached_blocks:
        # 增加引用计数（Copy-on-Write）
        for block in cached_blocks:
            block.ref_count += 1
        return cached_blocks
```

#### 10.3 多序列管理（Beam Search）

```python
# 为beam search创建多个序列
seq_group = SequenceGroup(
    request_id="req_002",
    seqs=[
        Sequence(seq_id=1, ...),
        Sequence(seq_id=2, ...),
        Sequence(seq_id=3, ...),  # beam_width=3
    ],
    sampling_params=SamplingParams(use_beam_search=True, best_of=3)
)
```

### 11. 监控与调试

#### 11.1 日志追踪

```python
# 在关键点添加日志
logger.info(f"Request {request_id} added to waiting queue")
logger.debug(f"Sequence {seq_id} status changed: {old_status} -> {new_status}")
logger.info(f"Request {request_id} completed in {elapsed_time:.2f}s")
```

#### 11.2 指标收集

```python
# 收集统计信息
class RequestTracker:
    def __init__(self):
        self.num_waiting = 0
        self.num_running = 0
        self.num_swapped = 0
        self.avg_queue_time = 0.0
    
    def update_metrics(self, scheduler: Scheduler):
        self.num_waiting = len(scheduler.waiting)
        self.num_running = len(scheduler.running)
        self.num_swapped = len(scheduler.swapped)
```

### 12. 性能考虑

#### 12.1 数据结构选择

- **Deque用于waiting/swapped队列**：支持O(1)的两端操作
- **List用于running队列**：需要随机访问和移除
- **Dict用于ID映射**：O(1)查找

#### 12.2 内存效率

- **引用计数**：支持block共享，减少内存使用
- **Lazy allocation**：只在需要时分配blocks
- **及时释放**：请求完成后立即回收资源

## 总结

vLLM的请求追踪和管理机制展现了精妙的系统设计：

**核心要点**：
1. **分层抽象**：Sequence → SequenceGroup → SequenceGroupMetadata
2. **状态机驱动**：清晰的状态转换规则
3. **集中式调度**：Scheduler统一管理所有队列和状态
4. **资源解耦**：BlockManager独立处理内存分配
5. **可追踪性**：通过ID在各模块间传递和关联

这种设计实现了高效的请求管理、灵活的资源调度和良好的系统可扩展性。

## 参考文献

1. **vLLM官方代码仓库**  
   [https://github.com/vllm-project/vllm](https://github.com/vllm-project/vllm)
   - `vllm/engine/llm_engine.py` - Engine实现
   - `vllm/core/scheduler.py` - 调度器实现
   - `vllm/sequence.py` - 序列数据结构

2. **vLLM论文**：Efficient Memory Management for Large Language Model Serving with PagedAttention  
   [https://arxiv.org/abs/2309.06180](https://arxiv.org/abs/2309.06180)

3. **vLLM官方文档 - Architecture Overview**  
   [https://docs.vllm.ai/en/latest/dev/arch_overview.html](https://docs.vllm.ai/en/latest/dev/arch_overview.html)

4. **Understanding vLLM Request Lifecycle (Blog)**  
   [https://vllm-project.github.io/blogs/request-lifecycle](https://vllm-project.github.io/blogs/request-lifecycle)

5. **vLLM源码解析系列 - 请求处理流程**  
   [https://zhuanlan.zhihu.com/p/665607842](https://zhuanlan.zhihu.com/p/665607842)

6. **深入理解vLLM调度机制**  
   [https://arxiv.org/abs/2401.09670](https://arxiv.org/abs/2401.09670) (Orca: A Distributed Serving System for Transformer-Based Generative Models)

