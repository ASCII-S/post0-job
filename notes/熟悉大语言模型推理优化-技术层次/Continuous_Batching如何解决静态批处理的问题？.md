---
created: '2025-10-19'
last_reviewed: null
next_review: '2025-10-19'
review_count: 0
difficulty: medium
mastery_level: 0.0
tags:
- 熟悉大语言模型推理优化-技术层次
- 熟悉大语言模型推理优化-技术层次/Continuous_Batching如何解决静态批处理的问题？.md
related_outlines: []
---
# Continuous Batching如何解决静态批处理的问题？

## 面试标准答案（精简版）

Continuous Batching（连续批处理）解决了静态批处理的两个核心问题：

1. **消除等待浪费**：静态批处理要求等待一个batch中所有序列都生成完成后才能开始处理新请求，导致GPU资源在等待长序列完成时闲置。Continuous Batching允许在任意迭代步骤中，将已完成的序列移除，并动态插入新请求，实现iteration-level的动态调度。

2. **消除padding浪费**：静态批处理需要将batch内所有序列padding到相同长度，造成大量无效计算。Continuous Batching通过动态管理每个序列的实际长度，只对有效token进行计算，避免了padding开销。

通过这种方式，Continuous Batching可以将GPU利用率从静态批处理的30-40%提升到80%以上，吞吐量提升2-3倍。

---

## 详细解析

### 一、静态批处理的问题

#### 1.1 什么是静态批处理

在传统的静态批处理（Static Batching）中，推理系统的工作流程如下：

1. 收集一批请求（如batch_size=8）
2. 将所有请求padding到相同长度
3. 开始自回归生成
4. **等待所有序列都生成完成**（遇到EOS token或达到最大长度）
5. 返回所有结果
6. 处理下一批请求

#### 1.2 核心问题分析

**问题1：强制同步导致的GPU闲置**

在一个batch中，不同请求的生成长度通常差异很大：

```
Batch中的请求:
请求1: "What is AI?" → 生成30个tokens (在第30步完成)
请求2: "Explain quantum computing in detail" → 生成200个tokens (在第200步完成)
请求3: "Hi" → 生成5个tokens (在第5步完成)
...

时间线：
Step 1-5:   所有8个请求都在生成 [GPU利用率: 100%]
Step 6-30:  7个请求在生成 (请求3已完成但必须等待) [GPU利用率: 87.5%]
Step 31-200: 仅1个请求在生成 (其他7个都完成了) [GPU利用率: 12.5%]
```

在这个例子中，从第31步到第200步，有170个迭代步只在处理1个请求，其余7个slot都是空的，GPU资源严重浪费。

**问题2：Padding带来的无效计算**

由于要将batch内所有序列padding到相同长度，会产生大量无效计算：

```
假设batch内的实际序列长度分布：
序列1: 10 tokens
序列2: 25 tokens
序列3: 15 tokens
序列4: 120 tokens (最长)

静态批处理需要：
所有序列都padding到120 tokens

实际有效计算: 10 + 25 + 15 + 120 = 170 tokens
总计算量: 120 × 4 = 480 tokens
计算浪费率: (480 - 170) / 480 = 64.6%
```

**问题3：延迟问题**

新到达的请求必须等待当前batch完全处理完成才能开始，导致：
- 如果当前batch有一个特别长的序列，所有新请求都要等待
- 平均等待时间取决于batch中最长序列的生成时间
- 在高负载场景下，延迟会急剧增加

### 二、Continuous Batching的核心机制

#### 2.1 Iteration-Level调度

Continuous Batching的核心创新是**在每个解码步骤（iteration）动态调整batch组成**：

```
传统静态批处理:
Batch 1: [Req1, Req2, Req3, Req4] → 运行直到全部完成 → Batch 2开始

Continuous Batching:
Step 1:  Batch = [Req1, Req2, Req3, Req4]
Step 5:  Req3完成 → Batch = [Req1, Req2, Req4, Req5(新加入)]
Step 30: Req1完成 → Batch = [Req2, Req4, Req5, Req6(新加入)]
Step 50: Req2完成 → Batch = [Req4, Req5, Req6, Req7(新加入)]
...
```

每个iteration结束后，调度器会：
1. **检查完成状态**：识别已生成EOS token或达到最大长度的请求
2. **移除完成请求**：释放对应的KV Cache和计算资源
3. **插入新请求**：从等待队列中选择新请求加入batch
4. **继续执行**：用新组成的batch进行下一次迭代

#### 2.2 动态长度管理

Continuous Batching不需要padding，每个序列保持自己的实际长度：

```python
# 伪代码示例
class ContinuousBatch:
    def __init__(self):
        self.active_requests = []  # 当前活跃的请求列表
    
    def decode_step(self):
        # 1. 为每个请求生成下一个token（长度各不相同）
        for req in self.active_requests:
            next_token = model.forward(
                input_ids=req.tokens,  # 实际长度，无padding
                kv_cache=req.kv_cache
            )
            req.append_token(next_token)
        
        # 2. 检查并移除完成的请求
        finished = [req for req in self.active_requests if req.is_finished()]
        for req in finished:
            self.active_requests.remove(req)
            req.kv_cache.free()  # 释放KV Cache
            return_result(req)
        
        # 3. 添加新请求（直到达到batch容量或显存限制）
        while len(self.active_requests) < max_batch_size:
            if has_waiting_requests() and has_available_memory():
                new_req = get_next_waiting_request()
                self.active_requests.append(new_req)
                new_req.kv_cache.allocate()  # 分配KV Cache
            else:
                break
        
        # 4. 继续下一次迭代
        return len(self.active_requests) > 0
```

#### 2.3 配合PagedAttention的内存管理

Continuous Batching通常与PagedAttention配合使用，以高效管理动态变化的KV Cache：

- **分页存储**：将KV Cache分成固定大小的页（如每页存储64个token）
- **按需分配**：新请求加入时，动态分配所需页数
- **即时回收**：请求完成时，立即回收页面供新请求使用
- **消除碎片**：通过分页机制避免内存碎片

```
示例：
时刻T1:
  Req1: 使用3个页 (150 tokens)
  Req2: 使用2个页 (80 tokens)
  Req3: 使用1个页 (40 tokens)
  总占用: 6个页

时刻T2 (Req3完成):
  Req1: 使用3个页
  Req2: 使用2个页
  Req4 (新): 使用1个页 ← 复用Req3释放的页
  总占用: 6个页 (无需等待，立即复用)
```

### 三、性能提升分析

#### 3.1 GPU利用率提升

**静态批处理**：
- GPU利用率随时间递减（请求陆续完成）
- 平均利用率：30-40%

**Continuous Batching**：
- GPU利用率保持稳定（持续有新请求补充）
- 平均利用率：70-90%

#### 3.2 吞吐量提升

实际测试数据（以LLaMA-13B为例）：

| 指标              | 静态批处理 | Continuous Batching | 提升    |
| ----------------- | ---------- | ------------------- | ------- |
| 吞吐量 (tokens/s) | 1,200      | 3,600               | 3倍     |
| GPU利用率         | 35%        | 85%                 | 2.4倍   |
| 平均延迟          | 2.5s       | 1.2s                | 降低52% |
| P99延迟           | 8.0s       | 2.5s                | 降低69% |

#### 3.3 内存效率提升

- 消除padding：节省20-40%的KV Cache空间
- 即时回收：内存利用率从60%提升到85%+
- 支持更大batch size或更长上下文

### 四、实现挑战与解决方案

#### 4.1 挑战1：可变batch size的算子支持

**问题**：每次迭代的batch size都在变化，需要算子能高效处理可变batch。

**解决方案**：
- 使用支持动态shape的算子实现
- 预编译多种batch size的kernel（如1, 2, 4, 8, 16...）
- 利用现代推理框架（vLLM, TensorRT-LLM）的原生支持

#### 4.2 挑战2：调度策略设计

**问题**：何时插入新请求？如何平衡延迟和吞吐？

**解决方案**：
```python
# ORCA调度策略示例
def should_add_request(current_batch, waiting_queue):
    # 1. 检查是否有空闲槽位
    if len(current_batch) >= max_batch_size:
        return False
    
    # 2. 估算当前batch的剩余时间
    estimated_finish_time = estimate_completion_time(current_batch)
    
    # 3. 检查内存是否足够
    available_memory = get_available_kv_cache_memory()
    required_memory = estimate_memory_for_new_request(waiting_queue[0])
    if available_memory < required_memory:
        return False
    
    # 4. 优先级判断
    if waiting_queue[0].priority > threshold:
        return True
    
    # 5. 如果当前batch即将全部完成，等待下一轮
    if estimated_finish_time < preemption_threshold:
        return False
    
    return True
```

#### 4.3 挑战3：公平性与饥饿问题

**问题**：短请求可能不断"插队"，导致长请求饥饿。

**解决方案**：
- 为等待时间长的请求提高优先级
- 使用混合调度：FCFS + 优先级
- 设置最大等待时间保证（SLA）

### 五、实际应用

#### 5.1 vLLM的实现

vLLM是Continuous Batching最成功的实现之一：

```python
# vLLM的核心循环（简化版）
class vLLMEngine:
    def step(self):
        # 调度：决定哪些请求进入本次iteration
        scheduled_requests = self.scheduler.schedule(
            running=self.running_requests,
            waiting=self.waiting_queue,
            available_blocks=self.kv_cache_manager.available_blocks
        )
        
        # 执行推理
        outputs = self.model.forward(scheduled_requests)
        
        # 更新状态
        for req, output in zip(scheduled_requests, outputs):
            req.append_output(output)
            if req.is_finished():
                self.running_requests.remove(req)
                self.kv_cache_manager.free(req.kv_cache)
                self.output_queue.put(req)
        
        # 添加新请求
        for new_req in scheduled_requests.new_requests:
            self.running_requests.append(new_req)
            new_req.kv_cache = self.kv_cache_manager.allocate(new_req)
```

#### 5.2 TensorRT-LLM的In-flight Batching

TensorRT-LLM将此技术称为"In-flight Batching"，核心思想相同：

- 使用高度优化的CUDA kernel支持动态batch
- 与TensorRT的编译优化深度结合
- 支持多种并行策略（张量并行、流水线并行）

### 六、总结

Continuous Batching通过以下机制解决静态批处理的问题：

| 静态批处理的问题      | Continuous Batching的解决方案           | 效果               |
| --------------------- | --------------------------------------- | ------------------ |
| GPU在等待长序列时闲置 | Iteration-level动态调度，即时补充新请求 | GPU利用率提升2-3倍 |
| Padding造成无效计算   | 每个序列保持实际长度，无padding         | 计算效率提升20-40% |
| 新请求等待时间长      | 不等待batch完成，持续接受新请求         | 平均延迟降低50%+   |
| 内存碎片和浪费        | 配合PagedAttention，按需分配和即时回收  | 内存利用率提升30%+ |

**关键要点**：
1. Continuous Batching是现代LLM推理服务的标准配置
2. 需要与动态内存管理（如PagedAttention）配合
3. 调度策略需要在延迟、吞吐量、公平性之间权衡
4. 主流框架（vLLM、TensorRT-LLM、TGI）都已实现此技术


---

## 相关笔记
<!-- 自动生成 -->

暂无相关笔记

