---
created: '2025-10-19'
last_reviewed: null
next_review: '2025-10-19'
review_count: 0
difficulty: medium
mastery_level: 0.0
tags:
- Transformer
- Transformer/逐token生成的计算过程.md
related_outlines: []
---
# 逐token生成的计算过程

## 面试标准答案（精简版）

Transformer的逐token生成过程包括三个核心步骤：

1. **输入处理**：将已生成的序列进行嵌入和位置编码，输入到Transformer模型
2. **前向计算**：通过编码器-解码器结构（或仅解码器）进行多层自注意力和前馈计算
3. **输出采样**：在最后一个位置的输出上应用softmax得到概率分布，然后采样生成下一个token

关键点是每次只预测一个token，然后将其添加到输入序列中继续下一轮生成，直到遇到结束符或达到最大长度。

---

## 详细技术解析

### 1. 整体流程概述

逐token生成是Transformer模型进行文本生成的核心机制，它采用**自回归（autoregressive）**的方式，每次预测序列中的下一个token。整个过程可以分为以下几个阶段：

```
输入序列 → 模型前向传播 → 概率分布 → 采样策略 → 新token → 更新序列 → 重复
```

### 2. 详细计算步骤

#### 2.1 初始化阶段

```python
# 伪代码示例
def generate_text(model, initial_input, max_length=100):
    # 初始化
    current_sequence = initial_input  # 例如: [BOS, "今天", "天气"]
    generated_tokens = []
    
    for step in range(max_length):
        # 步骤1: 输入预处理
        input_embeddings = embed_tokens(current_sequence)
        position_embeddings = get_position_encoding(len(current_sequence))
        model_input = input_embeddings + position_embeddings
```

#### 2.2 前向传播过程

##### 编码器-解码器架构（如T5）

```python
        # 步骤2: 编码器处理
        encoder_output = encoder(source_sequence)  # 只计算一次
        
        # 步骤3: 解码器处理
        decoder_output = decoder(
            target_sequence=current_sequence,
            encoder_output=encoder_output,
            use_causal_mask=True  # 确保因果性
        )
```

##### 仅解码器架构（如GPT）

```python
        # 步骤2: 解码器处理
        decoder_output = decoder(
            input_sequence=current_sequence,
            use_causal_mask=True  # 关键：防止看到未来token
        )
```

#### 2.3 注意力计算的关键细节

在每一步的注意力计算中：

```python
def causal_attention(Q, K, V, mask):
    # Q, K, V shape: [batch_size, seq_len, d_model]
    scores = torch.matmul(Q, K.transpose(-2, -1)) / sqrt(d_k)
    
    # 应用因果掩码：只能看到当前位置及之前的token
    causal_mask = torch.triu(torch.ones(seq_len, seq_len), diagonal=1)
    scores = scores.masked_fill(causal_mask == 1, -inf)
    
    attention_weights = softmax(scores, dim=-1)
    output = torch.matmul(attention_weights, V)
    return output
```

#### 2.4 输出层计算

```python
        # 步骤4: 获取最后一个位置的输出
        last_hidden_state = decoder_output[:, -1, :]  # [batch_size, d_model]
        
        # 步骤5: 通过输出投影层得到词汇表上的logits
        logits = output_projection(last_hidden_state)  # [batch_size, vocab_size]
        
        # 步骤6: 应用温度参数和softmax
        probabilities = softmax(logits / temperature, dim=-1)
```

#### 2.5 采样策略

```python
        # 步骤7: 根据采样策略选择下一个token
        if sampling_method == "greedy":
            next_token = torch.argmax(probabilities, dim=-1)
        elif sampling_method == "top_k":
            next_token = top_k_sampling(probabilities, k=50)
        elif sampling_method == "nucleus":
            next_token = nucleus_sampling(probabilities, p=0.9)
        
        # 步骤8: 更新序列
        current_sequence = torch.cat([current_sequence, next_token.unsqueeze(-1)], dim=-1)
        generated_tokens.append(next_token.item())
        
        # 步骤9: 检查停止条件
        if next_token == EOS_TOKEN:
            break
```

### 3. 关键技术要点

#### 3.1 KV缓存优化

在实际实现中，为了提高效率，通常会使用KV缓存：

```python
class TransformerWithKVCache:
    def __init__(self):
        self.kv_cache = {}  # 存储每层的K、V矩阵
    
    def forward_with_cache(self, input_ids, past_key_values=None):
        if past_key_values is not None:
            # 只计算新token的K、V
            new_k, new_v = self.compute_kv(input_ids[:, -1:])
            # 与历史K、V拼接
            k = torch.cat([past_key_values['k'], new_k], dim=-2)
            v = torch.cat([past_key_values['v'], new_v], dim=-2)
        else:
            # 第一步：计算完整序列的K、V
            k, v = self.compute_kv(input_ids)
        
        # 计算注意力（只需要计算新token的Q）
        q = self.compute_q(input_ids[:, -1:])
        attention_output = self.attention(q, k, v)
        
        return attention_output, {'k': k, 'v': v}
```

#### 3.2 位置编码的处理

```python
def get_position_encoding(seq_len, d_model, current_position=None):
    if current_position is not None:
        # 增量生成：只计算当前位置的位置编码
        pos = current_position
        pe = torch.zeros(1, d_model)
    else:
        # 完整计算
        pos = torch.arange(seq_len).unsqueeze(1)
        pe = torch.zeros(seq_len, d_model)
    
    # 正弦余弦位置编码
    div_term = torch.exp(torch.arange(0, d_model, 2) * 
                        -(math.log(10000.0) / d_model))
    pe[:, 0::2] = torch.sin(pos * div_term)
    pe[:, 1::2] = torch.cos(pos * div_term)
    
    return pe
```

### 4. 计算复杂度分析

#### 4.1 时间复杂度

- **单步前向传播**：O(n²d + nd²)，其中n是序列长度，d是隐藏维度
- **完整生成过程**：O(L × n²d + L × nd²)，其中L是生成长度
- **使用KV缓存后**：O(L × nd + L × nd²)

#### 4.2 空间复杂度

- **注意力权重**：O(n²)
- **KV缓存**：O(L × layers × d)
- **激活值**：O(nd)

### 5. 实际实现中的优化技术

#### 5.1 批处理生成

```python
def batch_generate(model, batch_inputs, max_length):
    batch_size = len(batch_inputs)
    # 处理不同长度的输入序列
    current_sequences = pad_sequences(batch_inputs)
    attention_masks = create_attention_masks(batch_inputs)
    
    for step in range(max_length):
        # 批量前向传播
        logits = model(current_sequences, attention_mask=attention_masks)
        
        # 批量采样
        next_tokens = sample_batch(logits[:, -1, :])
        
        # 更新序列和掩码
        current_sequences = torch.cat([current_sequences, next_tokens.unsqueeze(-1)], dim=-1)
        attention_masks = update_attention_masks(attention_masks, next_tokens)
```

#### 5.2 动态批处理

```python
class DynamicBatcher:
    def __init__(self, max_batch_size=8):
        self.pending_requests = []
        self.max_batch_size = max_batch_size
    
    def add_request(self, input_text, max_length):
        request = {
            'input': tokenize(input_text),
            'max_length': max_length,
            'current_length': len(tokenize(input_text)),
            'generated': []
        }
        self.pending_requests.append(request)
    
    def process_batch(self):
        # 选择合适的请求组成批次
        batch = self.select_batch()
        
        # 批量处理
        results = self.model.generate_batch(batch)
        
        # 更新请求状态
        self.update_requests(batch, results)
```

### 6. 常见问题与解决方案

#### 6.1 生成质量问题

**问题**：重复生成、逻辑不一致
**解决方案**：
- 使用nucleus sampling或top-k sampling
- 调整温度参数
- 实现重复惩罚机制

```python
def repetition_penalty(logits, input_ids, penalty=1.2):
    for token_id in set(input_ids.tolist()):
        logits[token_id] /= penalty
    return logits
```

#### 6.2 生成速度问题

**问题**：推理速度慢
**解决方案**：
- KV缓存
- 模型量化
- 投机解码（Speculative Decoding）

```python
def speculative_decoding(draft_model, target_model, input_ids, k=4):
    # 使用小模型快速生成k个token
    draft_tokens = draft_model.generate(input_ids, max_new_tokens=k)
    
    # 使用目标模型验证
    target_logits = target_model(torch.cat([input_ids, draft_tokens], dim=-1))
    
    # 接受或拒绝draft tokens
    accepted_tokens = []
    for i, token in enumerate(draft_tokens):
        if should_accept(token, target_logits[len(input_ids) + i]):
            accepted_tokens.append(token)
        else:
            break
    
    return accepted_tokens
```

### 7. 总结

逐token生成是Transformer模型文本生成的核心机制，它通过自回归的方式逐步构建输出序列。理解这个过程的关键在于：

1. **因果性约束**：每个位置只能看到前面的token
2. **增量计算**：利用KV缓存等技术优化重复计算
3. **采样策略**：平衡生成质量和多样性
4. **批处理优化**：提高并发处理能力

这个过程看似简单，但在实际实现中涉及众多优化技术，直接影响模型的生成质量和推理效率。

---

## 相关笔记
<!-- 自动生成 -->

- [KV缓存机制的实现](notes/Transformer/KV缓存机制的实现.md) - 相似度: 31% | 标签: Transformer, Transformer/KV缓存机制的实现.md

