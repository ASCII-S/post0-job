---
created: '2025-10-19'
last_reviewed: null
next_review: '2025-10-19'
review_count: 0
difficulty: medium
mastery_level: 0.0
tags:
- 熟悉大语言模型推理优化-技术层次
- 熟悉大语言模型推理优化-技术层次/如何划分流水线stage以平衡负载？.md
related_outlines: []
---
# 如何划分流水线stage以平衡负载？

## 面试标准答案

平衡流水线stage负载的关键是使各stage的计算时间尽可能接近。主要方法包括：1)通过Profiling测量每层实际计算时间，基于总计算时间均匀划分；2)考虑不同层的参数量和计算量差异，embedding和lm_head通常单独划分或与较少层组合；3)使用动态规划算法找到最优划分方案；4)考虑显存限制，避免某个stage显存溢出。实践中通常先均匀划分，再通过测量调整，目标是使最慢stage与最快stage的时间差<10%。

---

## 详细讲解

### 1. 负载均衡的重要性

#### 1.1 不平衡的影响

```
不平衡示例（4个stage）:
Stage 0: 10ms
Stage 1: 15ms  ← 瓶颈
Stage 2: 8ms
Stage 3: 12ms

总时间 = max(stage_times) × num_microbatches = 15ms × M
浪费 = (15-10) + (15-8) + (15-12) = 12ms/iteration
效率 = 平均时间/最大时间 = 11.25/15 = 75%
```

#### 1.2 理想情况

```
平衡后:
Stage 0: 11ms
Stage 1: 12ms
Stage 2: 11ms  
Stage 3: 11ms

总时间 = 12ms × M
效率 = 11.25/12 = 94%
```

### 2. Profiling驱动的划分

#### 2.1 测量每层计算时间

```python
def profile_layer_times(model, input_shape, device='cuda'):
    """
    测量每层的实际计算时间
    """
    import torch
    import time
    
    model.eval()
    layer_times = []
    
    # 生成测试输入
    x = torch.randn(input_shape).to(device)
    
    # 预热
    with torch.no_grad():
        for layer in model.layers:
            x = layer(x)
    
    # 测量
    x = torch.randn(input_shape).to(device)
    with torch.no_grad():
        for i, layer in enumerate(model.layers):
            # 同步GPU
            torch.cuda.synchronize()
            
            start = time.time()
            x = layer(x)
            torch.cuda.synchronize()
            
            elapsed = time.time() - start
            layer_times.append(elapsed * 1000)  # 转换为ms
            
            print(f"Layer {i}: {elapsed*1000:.2f} ms")
    
    return layer_times

# 使用
layer_times = profile_layer_times(
    model, 
    input_shape=(4, 2048, 12288)  # [B, S, H]
)
```

#### 2.2 分析Profiling结果

```python
import numpy as np
import matplotlib.pyplot as plt

def analyze_layer_times(layer_times):
    """
    分析层时间分布
    """
    layer_times = np.array(layer_times)
    
    print(f"总层数: {len(layer_times)}")
    print(f"总计算时间: {layer_times.sum():.2f} ms")
    print(f"平均每层: {layer_times.mean():.2f} ms")
    print(f"标准差: {layer_times.std():.2f} ms")
    print(f"最小/最大: {layer_times.min():.2f} / {layer_times.max():.2f} ms")
    
    # 可视化
    plt.figure(figsize=(12, 4))
    plt.bar(range(len(layer_times)), layer_times)
    plt.xlabel('Layer Index')
    plt.ylabel('Time (ms)')
    plt.title('Per-Layer Computation Time')
    plt.show()
    
    return layer_times
```

**典型结果** (GPT-3):
```
Layer 0 (Embedding): 5.2 ms  ← 较快
Layer 1-94 (Transformer): 18.5-19.5 ms  ← 稳定
Layer 95 (LM Head): 22.3 ms  ← 较慢（大矩阵乘）
```

### 3. 基于计算时间的划分算法

#### 3.1 贪心算法

```python
def greedy_partition(layer_times, num_stages):
    """
    贪心算法: 总是将下一层加入当前耗时最小的stage
    """
    stages = [[] for _ in range(num_stages)]
    stage_costs = [0.0] * num_stages
    
    for layer_idx, time in enumerate(layer_times):
        # 找到当前耗时最小的stage
        min_stage = min(range(num_stages), key=lambda i: stage_costs[i])
        
        # 将层加入该stage
        stages[min_stage].append(layer_idx)
        stage_costs[min_stage] += time
    
    # 打印结果
    for i, (stage, cost) in enumerate(zip(stages, stage_costs)):
        print(f"Stage {i}: Layers {stage[0]}-{stage[-1]}, "
              f"Cost: {cost:.2f} ms, "
              f"Num layers: {len(stage)}")
    
    return stages, stage_costs

# 使用
stages, costs = greedy_partition(layer_times, num_stages=8)
print(f"负载不平衡度: {max(costs)/min(costs):.2f}x")
```

#### 3.2 动态规划算法

```python
def dp_partition(layer_times, num_stages):
    """
    动态规划求解最优流水线stage划分方案
    
    核心思想: 这是一个最小化最大值(min-max)问题
    - 目标: 将n层分成num_stages个连续的stage，使得最慢的stage耗时最小
    - 等价于: 最小化 max(stage_1_time, stage_2_time, ..., stage_k_time)
    
    参数:
        layer_times: 每层的计算时间列表，例如 [2, 3, 4, 1, 5, ...]
        num_stages: 要划分的stage数量（通常等于GPU/设备数量）
    
    返回:
        partition: 划分方案，例如 [(0,2), (2,5), (5,8)] 表示层0-2为stage1，层2-5为stage2等
        stage_costs: 每个stage的总耗时
    """
    n = len(layer_times)
    
    # ============ 步骤1: 计算前缀和，用于快速计算区间和 ============
    # prefix_sum[i] = 前i层的总耗时
    # 这样 layers[k:i] 的总耗时 = prefix_sum[i] - prefix_sum[k]
    prefix_sum = [0]
    for t in layer_times:
        prefix_sum.append(prefix_sum[-1] + t)
    # 例如: layer_times=[2,3,4,1] → prefix_sum=[0,2,5,9,10]
    
    # ============ 步骤2: 初始化DP表 ============
    # dp[i][j] 的含义: 将前i层分成j个stage时，所有可能方案中"最慢stage"的最小耗时
    # 这是一个二维优化问题的状态定义
    INF = float('inf')
    dp = [[INF] * (num_stages + 1) for _ in range(n + 1)]
    
    # split[i][j] 记录达到dp[i][j]的最优划分方案
    # 存储格式: [(start1, end1), (start2, end2), ...] 表示每个stage包含的层范围
    split = [[[] for _ in range(num_stages + 1)] for _ in range(n + 1)]
    
    # 边界条件: 0层分成0个stage，耗时为0
    dp[0][0] = 0
    
    # ============ 步骤3: 动态规划主循环 ============
    for i in range(1, n + 1):  # 枚举前i层
        for j in range(1, min(i, num_stages) + 1):  # 枚举分成j个stage
            # min(i, num_stages): 前i层最多只能分成i个stage
            
            # ========= 核心: 枚举第j个stage的起始位置 =========
            for k in range(j - 1, i):  # k是第j个stage的起始层
                # k的范围: [j-1, i)
                # - 最小值j-1: 前面j-1个stage至少需要j-1层
                # - 最大值i-1: 第j个stage至少包含1层
                
                # 计算第j个stage的耗时: layers[k:i]
                stage_cost = prefix_sum[i] - prefix_sum[k]
                
                # 关键: min-max问题的状态转移
                # new_cost = 前j个stage中最慢stage的耗时
                #          = max(前j-1个stage的最慢耗时, 第j个stage的耗时)
                new_cost = max(dp[k][j-1], stage_cost)
                
                # 如果这个划分方案更优（最慢stage更快），则更新
                if new_cost < dp[i][j]:
                    dp[i][j] = new_cost
                    # 记录划分方案: 前k层的最优划分 + 当前stage [k, i)
                    split[i][j] = split[k][j-1] + [(k, i)]
    
    # ============ 步骤4: 提取最优解 ============
    # 前n层分成num_stages个stage的最优划分方案
    partition = split[n][num_stages]
    
    # 计算每个stage的实际耗时（用于验证和分析）
    stage_costs = [prefix_sum[end] - prefix_sum[start] 
                   for start, end in partition]
    
    return partition, stage_costs

# ============ 使用示例 ============
# 假设有32层，每层耗时存储在layer_times中
# 要将其分配到8个GPU上（即8个stage）
partition, costs = dp_partition(layer_times, num_stages=8)

# 结果示例:
# partition = [(0,4), (4,8), (8,12), (12,16), (16,20), (20,24), (24,28), (28,32)]
# costs = [120, 125, 118, 122, 124, 119, 121, 123]  # 每个stage的耗时
# max(costs) = 125  # 瓶颈stage的耗时，这就是整个流水线的吞吐率
```

#### 3.3 近似均分算法

```python
def balanced_partition(layer_times, num_stages):
    """
    目标: 每个stage的总时间 ≈ total_time / num_stages
    """
    total_time = sum(layer_times)
    target_time = total_time / num_stages
    
    stages = []
    current_stage = []
    current_time = 0
    
    for layer_idx, time in enumerate(layer_times):
        current_stage.append(layer_idx)
        current_time += time
        
        # 如果接近目标时间，且还需要更多stage
        if current_time >= target_time and len(stages) < num_stages - 1:
            stages.append(current_stage)
            current_stage = []
            current_time = 0
    
    # 最后一个stage包含剩余所有层
    if current_stage:
        stages.append(current_stage)
    
    # 计算各stage成本
    stage_costs = []
    for stage in stages:
        cost = sum(layer_times[i] for i in stage)
        stage_costs.append(cost)
    
    return stages, stage_costs
```

### 4. 考虑特殊层

#### 4.1 Embedding层

```python
def partition_with_embedding(model_config, num_stages):
    """
    考虑embedding层的特殊处理
    """
    vocab_size = model_config['vocab_size']
    hidden_size = model_config['hidden_size']
    
    # Embedding层的参数量和计算时间
    embedding_params = vocab_size * hidden_size
    embedding_time = estimate_embedding_time(embedding_params)
    
    # 如果embedding很大，可能需要单独一个stage
    if embedding_time > avg_transformer_layer_time * 3:
        # Embedding单独作为Stage 0
        stage_0 = [0]  # 只有embedding
        remaining_layers = list(range(1, num_layers))
        
        # 其余层划分到剩余stage
        partition = [stage_0] + partition_layers(
            remaining_layers, num_stages - 1
        )
    else:
        # Embedding与部分Transformer层在同一stage
        partition = partition_all_layers(num_layers, num_stages)
    
    return partition
```

#### 4.2 LM Head层

```python
def handle_lm_head(partition, lm_head_time, avg_layer_time):
    """
    处理LM Head层的划分
    """
    # LM Head通常计算量大（vocab_size × hidden_size）
    
    if lm_head_time > avg_layer_time * 2:
        # LM Head占用较多时间，最后一个stage层数要少
        # 减少最后stage的Transformer层数
        last_stage = partition[-1]
        reduce_layers = int(lm_head_time / avg_layer_time)
        
        # 将一些层移到倒数第二个stage
        partition[-2].extend(last_stage[:reduce_layers])
        partition[-1] = last_stage[reduce_layers:]
    
    return partition
```

### 5. 显存约束下的划分

#### 5.1 显存感知划分

```python
def memory_constrained_partition(layer_params, layer_activations,
                                  num_stages, memory_limit):
    """
    在显存约束下进行划分
    layer_params: 每层参数显存 (GB)
    layer_activations: 每层激活显存 (GB)
    memory_limit: 每个GPU的显存限制 (GB)
    """
    stages = []
    current_stage = []
    current_memory = 0
    
    for i, (params, acts) in enumerate(zip(layer_params, layer_activations)):
        layer_memory = params + acts
        
        if current_memory + layer_memory > memory_limit:
            # 当前stage已满
            if not current_stage:
                raise ValueError(f"Layer {i} 本身超过显存限制")
            
            stages.append(current_stage)
            current_stage = [i]
            current_memory = layer_memory
        else:
            current_stage.append(i)
            current_memory += layer_memory
    
    if current_stage:
        stages.append(current_stage)
    
    # 检查是否符合stage数量要求
    if len(stages) != num_stages:
        print(f"Warning: 显存约束导致stage数为{len(stages)}，"
              f"而非请求的{num_stages}")
    
    return stages
```

#### 5.2 参数量估算

```python
def estimate_layer_memory(config):
    """
    估算每层的参数量和激活显存
    """
    H = config['hidden_size']
    S = config['seq_length']
    B = config['batch_size']
    
    # Transformer层参数
    # QKV: 3 × H × H
    # O: H × H
    # FFN: 2 × H × 4H
    params_per_layer = (3*H*H + H*H + 2*H*4*H) * 2 / 1e9  # GB (FP16)
    
    # 激活显存（需要保存用于反向传播）
    # Attention: B × S × H (QKV各自)
    # FFN: B × S × 4H
    acts_per_layer = (B*S*H*3 + B*S*4*H) * 2 / 1e9  # GB (FP16)
    
    return params_per_layer, acts_per_layer

# 示例
config = {
    'hidden_size': 12288,
    'seq_length': 2048,
    'batch_size': 4
}
params, acts = estimate_layer_memory(config)
print(f"每层参数: {params:.2f} GB")
print(f"每层激活: {acts:.2f} GB")
```

### 6. 迭代优化

#### 6.1 测量实际性能

```python
def measure_pipeline_balance(model, partition, test_input):
    """
    测量实际流水线的负载平衡
    """
    stage_times = []
    
    for stage_idx, layer_indices in enumerate(partition):
        # 创建当前stage
        stage_layers = [model.layers[i] for i in layer_indices]
        
        # 测量时间
        times = []
        for _ in range(10):  # 多次测量取平均
            start = torch.cuda.Event(enable_timing=True)
            end = torch.cuda.Event(enable_timing=True)
            
            start.record()
            x = test_input
            for layer in stage_layers:
                x = layer(x)
            end.record()
            
            torch.cuda.synchronize()
            times.append(start.elapsed_time(end))
        
        avg_time = np.mean(times)
        stage_times.append(avg_time)
        print(f"Stage {stage_idx}: {avg_time:.2f} ms "
              f"({len(layer_indices)} layers)")
    
    # 分析平衡度
    max_time = max(stage_times)
    min_time = min(stage_times)
    avg_time = np.mean(stage_times)
    
    print(f"\n负载平衡分析:")
    print(f"  最大时间: {max_time:.2f} ms")
    print(f"  最小时间: {min_time:.2f} ms")
    print(f"  平均时间: {avg_time:.2f} ms")
    print(f"  不平衡度: {max_time/min_time:.2f}x")
    print(f"  效率: {avg_time/max_time*100:.1f}%")
    
    return stage_times
```

#### 6.2 调整策略

```python
def adjust_partition(partition, stage_times, threshold=1.15):
    """
    如果不平衡度超过阈值，调整划分
    threshold: 最慢/最快 的阈值
    """
    max_time = max(stage_times)
    min_time = min(stage_times)
    
    if max_time / min_time > threshold:
        # 找到最慢和最快的stage
        slowest_idx = stage_times.index(max_time)
        fastest_idx = stage_times.index(min_time)
        
        # 策略: 从最慢stage移动一些层到最快stage
        # (简化版，实际需要更复杂的逻辑)
        if slowest_idx < fastest_idx:
            # 从慢stage的末尾移到快stage的开头
            layers_to_move = 1
            partition[fastest_idx] = (
                partition[slowest_idx][-layers_to_move:] +
                partition[fastest_idx]
            )
            partition[slowest_idx] = partition[slowest_idx][:-layers_to_move]
        
        print(f"调整: 从Stage {slowest_idx} 移动层到 Stage {fastest_idx}")
        return True
    
    return False  # 不需要调整
```

### 7. 实际案例

#### 7.1 GPT-3 175B (8-way PP)

```python
# Profiling结果
layer_times_gpt3 = {
    'Embedding': 4.5,
    'Layers 1-94': [18.2] * 94,  # 基本一致
    'Layer 95': 18.8,
    'LM Head': 20.1
}

# 基于时间的划分
total_time = 4.5 + 18.2 * 94 + 18.8 + 20.1 = 1755.2 ms
target_per_stage = 1755.2 / 8 = 219.4 ms

# 最优划分
optimal_partition = [
    [0, 1-11],      # Embedding + 11 layers = 204.7 ms
    [12-23],        # 12 layers = 218.4 ms
    [24-35],        # 12 layers = 218.4 ms
    [36-47],        # 12 layers = 218.4 ms
    [48-59],        # 12 layers = 218.4 ms
    [60-71],        # 12 layers = 218.4 ms
    [72-83],        # 12 layers = 218.4 ms
    [84-95, LM]     # 12 layers + LM = 237.5 ms
]

# 负载不平衡度: 237.5 / 204.7 = 1.16x (可接受)
```

#### 7.2 LLaMA-65B (4-way PP)

```python
# 80层，均匀划分
layers_per_stage = 80 / 4 = 20

partition_llama = [
    [0-19],   # Stage 0: 20层
    [20-39],  # Stage 1: 20层
    [40-59],  # Stage 2: 20层
    [60-79],  # Stage 3: 20层
]

# LLaMA各层比较均匀，直接均分即可
```

### 8. 工具和自动化

#### 8.1 使用Megatron的自动划分

```bash
# Megatron-LM会根据模型配置自动划分
python pretrain_gpt.py \
    --num-layers 96 \
    --pipeline-model-parallel-size 8 \
    # 自动均匀划分成8个stage
```

#### 8.2 DeepSpeed的partition方法

```python
from deepspeed.pipe import PipelineModule, LayerSpec

# 定义partition
partition_config = {
    'method': 'parameters',  # 按参数量均分
    # 或 'uniform': 按层数均分
    # 或 'custom': 自定义
}
```

### 9. 最佳实践

**划分原则**：
1. ✅ 先Profiling，基于实际时间
2. ✅ 目标: 不平衡度 < 1.15x
3. ✅ 考虑特殊层（embedding, lm_head）
4. ✅ 验证显存限制
5. ✅ 迭代调整优化

**常见配置**：
- 小模型(<10B): 均匀划分即可
- 大模型(>100B): 需要仔细调整
- 超长序列: 注意attention层的时间

平衡流水线负载是获得高效率的关键，通常需要结合profiling和迭代调优。


---

## 相关笔记
<!-- 自动生成 -->

暂无相关笔记

