---
created: '2025-10-19'
last_reviewed: null
next_review: '2025-10-19'
review_count: 0
difficulty: medium
mastery_level: 0.0
tags:
- 熟悉大语言模型推理优化-技术层次
- 熟悉大语言模型推理优化-技术层次/ZeRO的三个阶段分别优化什么？.md
related_outlines: []
---
# ZeRO的三个阶段分别优化什么？

## 面试标准答案

ZeRO (Zero Redundancy Optimizer)三阶段分别优化：Stage 1分片优化器状态(如Adam的momentum和variance)，节省4×模型参数量显存；Stage 2额外分片梯度，再节省2×参数量显存；Stage 3进一步分片模型参数，总共节省16×显存(FP16训练)。推理时不需要优化器状态和梯度，主要用ZeRO-3的参数分片思想，结合All-Gather动态加载参数。ZeRO-Infinity扩展支持CPU/NVMe offloading，实现超大模型推理。

---

## 详细讲解

### 显存构成

```python
# FP16混合精度训练
模型参数 (FP16): 2 × M
梯度 (FP16): 2 × M
优化器状态 (FP32): 
  - 参数副本: 4 × M
  - Momentum: 4 × M
  - Variance: 4 × M
总计: 16 × M bytes

# 推理
模型参数 (FP16): 2 × M
激活: 变化
```

### ZeRO Stage 1

```python
# 分片优化器状态
# 每个GPU只存1/N的optimizer states

# GPU 0: optimizer states for params[0:M/N]
# GPU 1: optimizer states for params[M/N:2M/N]
# ...

节省: 12 × M / N
剩余: 2M(参数) + 2M(梯度) + 12M/N(优化器)
```

### ZeRO Stage 2

```python
# 分片梯度
# 每个GPU只存自己负责参数的梯度

节省: 2 × M × (N-1) / N
剩余: 2M(参数) + 2M/N(梯度) + 12M/N(优化器)
```

### ZeRO Stage 3

```python
# 分片模型参数
# 每个GPU只持有1/N参数

# 前向时: All-Gather需要的层参数
# 计算后: 释放
# 反向时: 再次All-Gather

节省: 2 × M × (N-1) / N  
剩余: 2M/N(参数) + 2M/N(梯度) + 12M/N(优化器)
      = 16M/N

总节省: 16× (对比无ZeRO)
```

### 推理应用

```python
# ZeRO-3推理 (ZeRO-Inference)
class ZeROInferenceModel:
    def __init__(self, model, world_size):
        # 分片参数
        self.param_shards = shard_parameters(model, world_size)
        
    def forward(self, x):
        for layer in self.layers:
            # All-Gather layer参数
            full_params = all_gather(self.param_shards[layer])
            
            # 计算
            x = layer.forward(x, full_params)
            
            # 释放(可选)
            del full_params
        
        return x

# 通信换显存
```

### 性能权衡

```
显存节省: 优秀 (16×)
通信增加: 显著 (每层All-Gather)
适用: 显存极度受限场景

推理时通常不用ZeRO-3:
- 增加延迟
- 通信开销大
- 除非模型超大
```

ZeRO主要用于训练，推理时考虑显存极限才使用。


---

## 相关笔记
<!-- 自动生成 -->

- [ZeRO与其他并行策略的关系是什么？](notes/熟悉大语言模型推理优化-技术层次/ZeRO与其他并行策略的关系是什么？.md) - 相似度: 31% | 标签: 熟悉大语言模型推理优化-技术层次, 熟悉大语言模型推理优化-技术层次/ZeRO与其他并行策略的关系是什么？.md

