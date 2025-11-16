---
created: '2025-10-19'
last_reviewed: null
next_review: '2025-10-19'
review_count: 0
difficulty: medium
mastery_level: 0.0
tags:
- 熟悉大语言模型推理优化-技术层次
- 熟悉大语言模型推理优化-技术层次/如何在流水线中处理动态batch？.md
related_outlines: []
---
# 如何在流水线中处理动态batch？

## 面试标准答案

流水线中处理动态batch的核心是在保持流水线运行的同时动态调整batch组成。主要方法包括：1)Continuous Batching，每个iteration可以添加新请求或移除完成的请求；2)Padding处理，用特殊token填充不同长度的序列；3)Split-fuse策略，将长请求分割、短请求合并；4)优先级调度，平衡延迟和吞吐。关键是维持足够的batch size以摊销流水线气泡，同时避免长尾请求阻塞整体进度。Orca和vLLM等系统实现了高效的动态batch管理。

---

## 详细讲解

### 1. 静态batch的问题

```python
# 传统静态batch
batch = [req1, req2, ..., req32]
# 所有请求一起进入，一起结束
# 问题: 最慢的请求阻塞所有其他请求
```

### 2. Continuous Batching

```python
# 动态调整batch
class DynamicBatchScheduler:
    def __init__(self):
        self.active_batch = []
        
    def step(self):
        # 1. 移除完成的请求
        self.active_batch = [r for r in self.active_batch 
                             if not r.is_finished()]
        
        # 2. 添加新请求
        while len(self.active_batch) < max_batch_size:
            new_req = self.get_waiting_request()
            if new_req:
                self.active_batch.append(new_req)
            else:
                break
        
        # 3. 执行当前batch
        outputs = model.forward(self.active_batch)
        
        # 4. 更新状态
        for req, out in zip(self.active_batch, outputs):
            req.update(out)
```

**优点**:
- 吞吐量高: batch持续满载
- 延迟低: 请求完成即退出
- 资源利用率高

### 3. 流水线中的实现

```python
# 每个stage维护自己的batch
class PipelineStageWithDynamicBatch:
    def __init__(self, stage_id, num_stages):
        self.stage_id = stage_id
        self.active_requests = {}
        
    def forward(self):
        # Stage 0: 接收新请求
        if self.stage_id == 0:
            new_reqs = receive_new_requests()
            for req in new_reqs:
                self.active_requests[req.id] = req
        
        # 所有stage: 处理当前requests
        if self.active_requests:
            batch_ids = list(self.active_requests.keys())
            batch_inputs = [self.active_requests[id].data 
                           for id in batch_ids]
            
            # 执行计算
            outputs = self.compute(batch_inputs)
            
            # 更新数据
            for id, out in zip(batch_ids, outputs):
                self.active_requests[id].data = out
        
        # 传递到下一stage或输出
        if self.stage_id < num_stages - 1:
            # 发送到下一stage
            send_to_next_stage(self.active_requests)
        else:
            # 最后一stage: 检查完成并输出
            finished = [id for id, req in self.active_requests.items()
                       if req.is_finished()]
            for id in finished:
                output_result(self.active_requests[id])
                del self.active_requests[id]
```

### 4. Padding与Masking

```python
# 处理不同长度序列
def dynamic_padding(requests):
    max_len = max(r.current_length for r in requests)
    
    batched_input = []
    attention_mask = []
    
    for req in requests:
        # Padding
        padded = pad(req.tokens, max_len, pad_token=0)
        batched_input.append(padded)
        
        # Mask
        mask = [1] * req.current_length + [0] * (max_len - req.current_length)
        attention_mask.append(mask)
    
    return torch.tensor(batched_input), torch.tensor(attention_mask)
```

### 5. 优先级调度

```python
class PriorityScheduler:
    def select_batch(self, waiting_requests, max_batch_size):
        # 考虑多个因素:
        # 1. 等待时间
        # 2. 预估剩余长度
        # 3. 用户优先级
        
        scores = []
        for req in waiting_requests:
            score = (
                req.wait_time * 1.0 +  # 等待时间权重
                (1.0 / max(req.estimated_remaining, 1)) * 0.5 +  # 快完成的优先
                req.user_priority * 2.0  # 用户优先级
            )
            scores.append((score, req))
        
        # 选择top-k
        scores.sort(reverse=True)
        selected = [req for _, req in scores[:max_batch_size]]
        
        return selected
```

### 6. Split-Fuse策略

```python
# 将长序列分割，短序列合并
def split_fuse_batch(requests, target_tokens):
    """
    target_tokens: 目标总token数
    """
    batch = []
    total_tokens = 0
    
    for req in requests:
        req_tokens = req.current_length
        
        if req_tokens > target_tokens // 2:
            # 长请求: 单独处理或分割
            if total_tokens == 0:
                batch.append(req)
                break
        else:
            # 短请求: 尝试加入batch
            if total_tokens + req_tokens <= target_tokens:
                batch.append(req)
                total_tokens += req_tokens
            else:
                break
    
    return batch
```

### 7. 实际系统实现

**Orca调度**:
```python
# Iteration-level scheduling
每个iteration:
1. 移除finished requests
2. Prefill新requests (如果有空间)
3. Decode existing requests
4. 动态调整batch大小

优化:
- Selective batching
- Preemption (可选)
```

**vLLM**:
```python
# PagedAttention + Continuous Batching
每个step:
1. 分配KV cache pages
2. 动态组batch
3. 执行计算
4. 回收完成请求的pages
5. 接纳新请求

关键: 内存管理与调度解耦
```

### 8. 性能影响

```python
# 静态batch
平均延迟 = 最长请求时间
吞吐量 = batch_size / 最长请求时间

# 动态batch
平均延迟 = 各请求实际完成时间的平均
吞吐量 = 总请求数 / 总时间

# 提升: 30-50% 吞吐量提升, 20-40% 延迟降低
```

### 9. 最佳实践

```python
# 推荐配置
continuous_batching = True
max_batch_size = 64
target_batch_size = 32

# 调度策略
scheduling_policy = "FCFS"  # or "Priority"

# Preemption
enable_preemption = True  # 允许暂停低优先级请求
```

动态batch是现代LLM推理服务的标准特性，显著提升资源利用率。


---

## 相关笔记
<!-- 自动生成 -->

暂无相关笔记

