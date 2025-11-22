---
created: '2025-11-23'
last_reviewed: '2025-11-23'
next_review: '2025-11-23'
review_count: 0
difficulty: medium
mastery_level: 0.0
tags:
- 精通vllm源码
- 精通vllm源码/请详细解释`LLMEngine.step()`方法的完整执行流程.md
related_outlines: []
---

# 请详细解释`LLMEngine.step()`方法的完整执行流程

## 概述

`LLMEngine.step()` 是 vLLM 推理引擎的**核心推理循环方法**，负责执行一次完整的推理迭代。每次调用 `step()` 方法，引擎都会：
1. 调度待处理的请求
2. 执行模型前向传播
3. 采样生成 token
4. 更新请求状态并返回输出

这个方法实现了 vLLM 的**连续批处理（Continuous Batching）**机制，使得引擎能够动态地处理不断到来的请求，最大化 GPU 利用率。

## 整体架构图

```
┌─────────────────────────────────────────────────────────────────────────┐
│                         LLMEngine.step()                                 │
├─────────────────────────────────────────────────────────────────────────┤
│                                                                          │
│  ┌──────────────┐    ┌──────────────┐    ┌──────────────┐               │
│  │   Scheduler  │───▶│ ModelRunner  │───▶│   Sampler    │               │
│  │  .schedule() │    │.execute_model│    │  .forward()  │               │
│  └──────┬───────┘    └──────┬───────┘    └──────┬───────┘               │
│         │                   │                   │                        │
│         ▼                   ▼                   ▼                        │
│  ┌──────────────┐    ┌──────────────┐    ┌──────────────┐               │
│  │ BlockManager │    │  Attention   │    │   Output     │               │
│  │ (KV Cache)   │    │  Backends    │    │  Processor   │               │
│  └──────────────┘    └──────────────┘    └──────────────┘               │
│                                                                          │
└─────────────────────────────────────────────────────────────────────────┘
```

## 详细执行流程

### 第一阶段：调度（Scheduling）

**入口：** `Scheduler.schedule()`

调度器是 `step()` 方法的第一个关键组件，负责决定在当前迭代中处理哪些请求。

#### 1.1 请求队列管理

调度器维护三个请求队列：
- **Waiting Queue（等待队列）**：新到达的请求，等待首次处理
- **Running Queue（运行队列）**：正在进行 token 生成的请求
- **Swapped Queue（交换队列）**：因内存不足被暂时换出的请求

```python
# 调度器内部状态（简化示意）
class Scheduler:
    def __init__(self):
        self.waiting: Deque[SequenceGroup] = deque()
        self.running: Deque[SequenceGroup] = deque()
        self.swapped: Deque[SequenceGroup] = deque()
```

#### 1.2 调度决策过程

```
┌─────────────────────────────────────────────────────────────────┐
│                    Scheduler.schedule()                          │
├─────────────────────────────────────────────────────────────────┤
│                                                                  │
│  1. 检查 Running 队列中是否有请求需要抢占                         │
│     ├── 如果 GPU 内存不足 → 抢占部分请求到 Swapped 队列           │
│     └── 释放对应的 KV Cache blocks                               │
│                                                                  │
│  2. 处理 Running 队列中的请求                                    │
│     ├── 为每个序列分配新的 KV Cache block（如需要）              │
│     └── 收集需要执行的序列                                       │
│                                                                  │
│  3. 从 Swapped 队列恢复请求（如果有空闲内存）                     │
│     ├── 将 KV Cache 从 CPU 换回 GPU                              │
│     └── 移动请求到 Running 队列                                  │
│                                                                  │
│  4. 从 Waiting 队列调度新请求                                    │
│     ├── 检查是否有足够的 KV Cache blocks                         │
│     ├── 分配 blocks 并初始化状态                                 │
│     └── 移动请求到 Running 队列                                  │
│                                                                  │
│  5. 返回 SchedulerOutput                                         │
│     ├── scheduled_seq_groups: 要执行的序列组                     │
│     ├── blocks_to_swap_in: 需要换入的 blocks                     │
│     ├── blocks_to_swap_out: 需要换出的 blocks                    │
│     └── blocks_to_copy: 需要复制的 blocks（用于 beam search）    │
│                                                                  │
└─────────────────────────────────────────────────────────────────┘
```

#### 1.3 KV Cache Block 分配

调度器与 `BlockSpaceManager` 协作管理 KV Cache 内存：

```python
# BlockSpaceManager 核心功能
class BlockSpaceManager:
    def can_allocate(self, seq_group: SequenceGroup) -> AllocStatus:
        """检查是否有足够的blocks分配给新请求"""
        pass

    def allocate(self, seq_group: SequenceGroup) -> None:
        """为序列组分配blocks"""
        pass

    def can_append_slots(self, seq_group: SequenceGroup) -> bool:
        """检查是否可以为序列追加新的slot"""
        pass

    def append_slots(self, seq: Sequence) -> Dict[int, List[int]]:
        """追加新slots并返回需要复制的block映射"""
        pass
```

### 第二阶段：模型执行（Model Execution）

**入口：** `ModelRunner.execute_model()`

在获得调度输出后，引擎将执行模型前向传播。

#### 2.1 输入准备

```
┌─────────────────────────────────────────────────────────────────┐
│              ModelRunner._prepare_model_input()                  │
├─────────────────────────────────────────────────────────────────┤
│                                                                  │
│  Prefill（预填充）阶段:                                          │
│  ├── input_tokens: 完整的 prompt token IDs                       │
│  ├── input_positions: [0, 1, 2, ..., prompt_len-1]               │
│  └── slot_mapping: KV Cache 中的存储位置                         │
│                                                                  │
│  Decode（解码）阶段:                                             │
│  ├── input_tokens: 仅最新生成的 token ID                         │
│  ├── input_positions: [current_position]                         │
│  └── slot_mapping: 新 token 的 KV Cache 位置                     │
│                                                                  │
│  Attention Metadata:                                             │
│  ├── block_tables: 每个序列的 KV Cache block 映射表              │
│  ├── seq_lens: 每个序列的当前长度                                │
│  ├── context_lens: 上下文长度（用于attention计算）               │
│  └── max_seq_len: 批次中的最大序列长度                           │
│                                                                  │
└─────────────────────────────────────────────────────────────────┘
```

#### 2.2 模型前向传播

```python
# 简化的模型前向传播流程
def execute_model(self, model_input: ModelInputForGPU) -> SamplerOutput:
    # 1. 嵌入层：将 token IDs 转换为向量
    hidden_states = self.model.embed_tokens(model_input.input_tokens)

    # 2. 位置编码
    positions = model_input.input_positions

    # 3. Transformer 层
    for layer in self.model.layers:
        hidden_states = layer(
            hidden_states,
            positions=positions,
            kv_cache=kv_cache,
            attn_metadata=model_input.attn_metadata,
        )

    # 4. 最终层归一化
    hidden_states = self.model.norm(hidden_states)

    # 5. 计算 logits
    logits = self.lm_head(hidden_states)

    return logits
```

#### 2.3 注意力计算（PagedAttention）

vLLM 的核心创新是 **PagedAttention**，将 KV Cache 组织为固定大小的 blocks：

```
┌─────────────────────────────────────────────────────────────────┐
│                      PagedAttention                              │
├─────────────────────────────────────────────────────────────────┤
│                                                                  │
│  物理内存布局:                                                   │
│  ┌────────┬────────┬────────┬────────┬────────┐                 │
│  │Block 0 │Block 1 │Block 2 │Block 3 │Block 4 │ ...             │
│  │(Seq A) │(Seq B) │(Seq A) │(Seq C) │(Seq B) │                 │
│  └────────┴────────┴────────┴────────┴────────┘                 │
│                                                                  │
│  Block Table (每个序列维护):                                     │
│  ┌────────────────────────────────────────────┐                 │
│  │ Sequence A: [Block 0, Block 2, ...]        │                 │
│  │ Sequence B: [Block 1, Block 4, ...]        │                 │
│  │ Sequence C: [Block 3, ...]                 │                 │
│  └────────────────────────────────────────────┘                 │
│                                                                  │
│  优势:                                                           │
│  - 内存碎片接近零                                                │
│  - 支持动态序列长度                                              │
│  - 高效的内存共享（prefix caching, beam search）                 │
│                                                                  │
└─────────────────────────────────────────────────────────────────┘
```

支持的注意力后端：
- **FlashAttention**: 高效的融合注意力实现
- **FlashInfer**: 针对推理优化的注意力后端
- **XFormers**: Facebook 的高效注意力库
- **PagedAttention (native)**: vLLM 原生实现

### 第三阶段：采样（Sampling）

**入口：** `Sampler.forward()`

获得 logits 后，采样器根据采样参数生成下一个 token。

#### 3.1 采样流程

```
┌─────────────────────────────────────────────────────────────────┐
│                     Sampler.forward()                            │
├─────────────────────────────────────────────────────────────────┤
│                                                                  │
│  1. 应用 Logits 处理器（按顺序）:                                │
│     ├── Temperature scaling: logits / temperature                │
│     ├── Top-p (nucleus) filtering                                │
│     ├── Top-k filtering                                          │
│     ├── Min-p filtering                                          │
│     ├── Repetition penalty                                       │
│     ├── Presence penalty                                         │
│     ├── Frequency penalty                                        │
│     └── 自定义 logits processors                                 │
│                                                                  │
│  2. 计算采样概率:                                                │
│     probs = softmax(processed_logits)                            │
│                                                                  │
│  3. 根据采样方法选择 token:                                      │
│     ├── Greedy: argmax(probs)                                    │
│     ├── Random: multinomial(probs)                               │
│     └── Beam Search: 维护 top-k 候选                             │
│                                                                  │
│  4. 计算 log probabilities（如果请求）:                          │
│     ├── token_logprobs: 选中 token 的 log prob                   │
│     └── top_logprobs: top-k tokens 及其 log probs                │
│                                                                  │
│  5. 返回 SamplerOutput:                                          │
│     ├── sampled_token_ids                                        │
│     ├── sampled_token_probs                                      │
│     └── logprobs (可选)                                          │
│                                                                  │
└─────────────────────────────────────────────────────────────────┘
```

#### 3.2 采样参数

```python
@dataclass
class SamplingParams:
    n: int = 1                          # 生成的序列数
    temperature: float = 1.0            # 温度参数
    top_p: float = 1.0                  # Nucleus sampling 参数
    top_k: int = -1                     # Top-k 采样
    min_p: float = 0.0                  # Min-p 采样
    use_beam_search: bool = False       # 是否使用 beam search
    best_of: int = 1                    # 生成 best_of 个序列，返回最好的 n 个
    presence_penalty: float = 0.0       # 存在惩罚
    frequency_penalty: float = 0.0      # 频率惩罚
    repetition_penalty: float = 1.0     # 重复惩罚
    max_tokens: int = 16                # 最大生成 token 数
    stop: List[str] = None              # 停止词列表
    logprobs: int = None                # 返回的 logprobs 数量
```

### 第四阶段：输出处理（Output Processing）

**入口：** `OutputProcessor` / `_process_model_outputs()`

采样完成后，引擎需要处理输出并更新请求状态。

#### 4.1 输出处理流程

```
┌─────────────────────────────────────────────────────────────────┐
│                   Output Processing                              │
├─────────────────────────────────────────────────────────────────┤
│                                                                  │
│  1. 更新序列状态:                                                │
│     ├── 将新 token 追加到序列                                    │
│     ├── 更新序列长度                                             │
│     └── 更新 KV Cache 元数据                                     │
│                                                                  │
│  2. 检查停止条件:                                                │
│     ├── 达到 max_tokens 限制                                     │
│     ├── 生成了 EOS token                                         │
│     ├── 匹配到 stop 字符串                                       │
│     └── 达到 max_model_len 限制                                  │
│                                                                  │
│  3. 处理完成的请求:                                              │
│     ├── 从 Running 队列移除                                      │
│     ├── 释放 KV Cache blocks                                     │
│     └── 构建 RequestOutput                                       │
│                                                                  │
│  4. Detokenization（异步）:                                      │
│     ├── 将 token IDs 转换为文本                                  │
│     ├── 处理特殊 token                                           │
│     └── 支持增量输出（streaming）                                │
│                                                                  │
└─────────────────────────────────────────────────────────────────┘
```

#### 4.2 RequestOutput 结构

```python
@dataclass
class RequestOutput:
    request_id: str                     # 请求 ID
    prompt: str                         # 原始 prompt
    prompt_token_ids: List[int]         # Prompt token IDs
    outputs: List[CompletionOutput]     # 生成的输出列表
    finished: bool                      # 是否完成

@dataclass
class CompletionOutput:
    index: int                          # 输出索引
    text: str                           # 生成的文本
    token_ids: List[int]                # 生成的 token IDs
    cumulative_logprob: float           # 累计 log probability
    logprobs: List[Dict]                # 每个 token 的 logprobs
    finish_reason: str                  # 完成原因 (stop/length/...)
```

## 完整的 step() 伪代码

```python
def step(self) -> List[RequestOutput]:
    """执行一次推理迭代"""

    # ========== 第一阶段：调度 ==========
    # 获取调度决策
    scheduler_outputs = self.scheduler.schedule()

    if not scheduler_outputs.is_empty():
        # ========== 内存操作 ==========
        # 执行 block 交换操作
        if scheduler_outputs.blocks_to_swap_in:
            self._swap_in(scheduler_outputs.blocks_to_swap_in)
        if scheduler_outputs.blocks_to_swap_out:
            self._swap_out(scheduler_outputs.blocks_to_swap_out)
        if scheduler_outputs.blocks_to_copy:
            self._copy_blocks(scheduler_outputs.blocks_to_copy)

        # ========== 第二阶段：模型执行 ==========
        # 准备模型输入
        model_input = self.model_runner.prepare_model_input(
            scheduler_outputs.scheduled_seq_groups
        )

        # 执行模型前向传播
        # 包含：embedding -> transformer layers -> attention -> lm_head
        hidden_states = self.model_runner.execute_model(model_input)

        # ========== 第三阶段：采样 ==========
        # 对 logits 进行采样，生成下一个 token
        sampler_output = self.model_runner.sample(
            hidden_states,
            model_input.sampling_metadata
        )

        # ========== 第四阶段：输出处理 ==========
        # 处理采样结果，更新序列状态
        request_outputs = self._process_model_outputs(
            sampler_output,
            scheduler_outputs.scheduled_seq_groups
        )
    else:
        request_outputs = []

    # 返回完成或需要流式输出的请求
    return request_outputs
```

## 性能优化机制

### 1. CUDA Graph 优化

对于 decode 阶段（每次只处理一个 token），vLLM 使用 CUDA Graph 消除 kernel 启动开销：

```python
class CUDAGraphRunner:
    """管理 CUDA Graph 的捕获和回放"""

    def capture(self, batch_size: int):
        """捕获特定 batch size 的计算图"""
        # 仅对 decode 阶段使用
        # prefill 阶段由于序列长度变化不使用 CUDA Graph
        pass

    def replay(self, input_ids, positions, kv_caches):
        """回放捕获的计算图"""
        # 极大减少 CPU-GPU 同步开销
        pass
```

### 2. 连续批处理（Continuous Batching）

```
传统静态批处理:
┌────┬────┬────┬────┐
│Req1│Req2│Req3│    │  等待所有请求完成
│████│████│████│    │  才能处理新请求
│████│████│    │    │
│████│    │    │    │
└────┴────┴────┴────┘

vLLM 连续批处理:
┌────┬────┬────┬────┐
│Req1│Req2│Req3│Req4│  请求完成后立即
│████│████│████│████│  被新请求替换
│████│████│Req5│████│
│████│Req6│████│████│
└────┴────┴────┴────┘
```

### 3. Prefix Caching

当启用 `enable_prefix_caching=True` 时，相同前缀的请求可以共享 KV Cache：

```
请求 A: "Translate to French: Hello"
请求 B: "Translate to French: World"

共享前缀: "Translate to French: "
├── KV Cache 只计算一次
└── 后续 token 复用缓存的 KV 值
```

## V0 vs V1 引擎差异

vLLM 有两个引擎版本：

| 特性 | V0 Engine | V1 Engine |
|------|-----------|-----------|
| 架构 | 单进程 | 多进程（EngineCore） |
| 调度器 | 同步调度 | 异步调度 |
| 输出处理 | 同步 | 流水线化 |
| 适用场景 | 调试、简单部署 | 生产环境、高吞吐 |

V1 引擎的 `step()` 方法利用多进程架构实现了更好的并行性：

```
V1 Engine 架构:
┌─────────────────┐    ┌─────────────────┐
│   Frontend      │    │   EngineCore    │
│  (API Server)   │◄──►│   (Scheduler    │
│                 │    │    + Model)     │
└─────────────────┘    └─────────────────┘
        │                      │
        ▼                      ▼
┌─────────────────┐    ┌─────────────────┐
│ OutputProcessor │    │ GPUModelRunner  │
│ (Detokenization)│    │  (Inference)    │
└─────────────────┘    └─────────────────┘
```

## 源码文件参考

| 组件 | 文件路径 |
|------|----------|
| LLMEngine | `vllm/engine/llm_engine.py` |
| AsyncLLMEngine | `vllm/engine/async_llm_engine.py` |
| Scheduler | `vllm/core/scheduler.py` |
| BlockSpaceManager | `vllm/core/block_manager.py` |
| GPUModelRunner | `vllm/worker/model_runner.py` |
| Sampler | `vllm/model_executor/layers/sampler.py` |
| Attention Backends | `vllm/attention/backends/` |
| V1 EngineCore | `vllm/v1/engine/core.py` |
| OutputProcessor | `vllm/v1/engine/output_processor.py` |

## 总结

`LLMEngine.step()` 方法是 vLLM 的核心推理循环，通过精心设计的四个阶段实现高效的 LLM 推理：

1. **调度阶段**：智能管理请求队列和 KV Cache 内存
2. **模型执行阶段**：使用 PagedAttention 高效计算注意力
3. **采样阶段**：支持多种采样策略生成 token
4. **输出处理阶段**：更新状态并返回结果

这种设计使 vLLM 能够实现：
- 高吞吐量（连续批处理）
- 低内存碎片（PagedAttention）
- 低延迟（CUDA Graph 优化）
- 灵活的内存管理（抢占和换页）
