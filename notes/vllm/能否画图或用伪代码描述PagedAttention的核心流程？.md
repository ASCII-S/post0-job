---
created: '2025-10-19'
last_reviewed: null
next_review: '2025-10-19'
review_count: 0
difficulty: medium
mastery_level: 0.0
tags:
- vllm
- vllm/能否画图或用伪代码描述PagedAttention的核心流程？.md
related_outlines: []
---
# 能否画图或用伪代码描述PagedAttention的核心流程？

## 标准精简答案 (可背诵版本)

**PagedAttention核心流程：** PagedAttention通过虚拟地址映射将序列token位置转换为物理块位置，按需分配固定大小的KV缓存块，实现动态内存管理。核心步骤包括：地址转换、块分配、KV存储、attention计算、内存回收。

## 详细技术解析

### 1. PagedAttention系统架构图

```
PagedAttention系统架构

┌─────────────────────────────────────────────────────────────────────────┐
│                        序列虚拟地址空间层                                    │
├─────────────────────────────────────────────────────────────────────────┤
│ seq_1: [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, ...] (虚拟token位置)              │
│ seq_2: [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, ...] (虚拟token位置)              │
│ seq_3: [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, ...] (虚拟token位置)              │
└─────────────────────────────────────────────────────────────────────────┘
                                   ↓ 地址转换层 (类似MMU)
┌─────────────────────────────────────────────────────────────────────────┐
│                           块映射表 (页表)                                    │
├─────────────────────────────────────────────────────────────────────────┤
│ seq_1: [block_2, block_5, block_1, ...]  (物理块ID序列)                  │
│ seq_2: [block_7, block_3, block_9, ...]  (物理块ID序列)                  │  
│ seq_3: [block_4, block_8, block_6, ...]  (物理块ID序列)                  │
└─────────────────────────────────────────────────────────────────────────┘
                                   ↓ 物理内存访问
┌─────────────────────────────────────────────────────────────────────────┐
│                         物理内存池 (固定大小块)                              │
├─────────────────────────────────────────────────────────────────────────┤
│ Block_0: [K0,V0,K1,V1,...] │ Block_1: [K16,V16,...] │ Block_2: [K0,V0,...]│
│ Block_3: [K0,V0,K1,V1,...] │ Block_4: [K0,V0,...]   │ Block_5: [K16,V16...]│
│ Block_6: [K0,V0,K1,V1,...] │ Block_7: [K0,V0,...]   │ Block_8: [K0,V0,...]│
│ Block_9: [K16,V16,K17,V17] │ ...                    │ ...                  │
└─────────────────────────────────────────────────────────────────────────┘
```

### 2. PagedAttention核心数据结构

```python
# 核心数据结构定义
class PagedAttentionSystem:
    """PagedAttention核心系统"""
    
    def __init__(self, block_size=16, max_blocks=1024):
        # 系统配置
        self.block_size = block_size  # 每个块的token数量
        self.max_blocks = max_blocks  # 最大块数量
        
        # 物理内存池
        self.physical_blocks = {}  # {block_id: PhysicalBlock}
        self.free_blocks = set(range(max_blocks))  # 空闲块ID集合
        self.used_blocks = set()  # 已使用块ID集合
        
        # 虚拟地址映射
        self.sequence_mappings = {}  # {seq_id: SequenceMapping}
        
        # 内存管理组件
        self.allocator = BlockAllocator(self)
        self.address_translator = AddressTranslator(self)
        self.garbage_collector = GarbageCollector(self)

class PhysicalBlock:
    """物理内存块"""
    def __init__(self, block_id, block_size, hidden_dim):
        self.block_id = block_id
        self.block_size = block_size
        
        # KV缓存存储 (shape: [block_size, hidden_dim])
        self.keys = torch.zeros(block_size, hidden_dim, dtype=torch.float16)
        self.values = torch.zeros(block_size, hidden_dim, dtype=torch.float16)
        
        # 块状态管理
        self.used_slots = 0  # 已使用的slot数量
        self.owners = set()  # 拥有此块的序列ID集合
        self.last_access = time.time()

class SequenceMapping:
    """序列映射表 (类似页表)"""
    def __init__(self, sequence_id):
        self.sequence_id = sequence_id
        self.block_ids = []  # 按顺序存储的物理块ID列表
        self.current_length = 0  # 当前序列长度
        
    def get_block_and_offset(self, token_position):
        """获取token位置对应的块ID和块内偏移"""
        block_index = token_position // self.block_size
        intra_block_offset = token_position % self.block_size
        
        if block_index >= len(self.block_ids):
            return None, None  # 需要分配新块
        
        return self.block_ids[block_index], intra_block_offset
```

### 3. PagedAttention核心流程伪代码

#### 3.1 完整系统流程伪代码

```python
def paged_attention_main_flow():
    """PagedAttention主流程伪代码"""
    
    # ===== 阶段1: 系统初始化 =====
    paged_system = PagedAttentionSystem(
        block_size=16,
        max_blocks=1024,
        hidden_dim=4096
    )
    
    # ===== 阶段2: 序列生命周期管理 =====
    for each_generation_step:
        
        # 2.1 处理新序列请求
        for new_sequence_request in incoming_requests:
            sequence_id = create_sequence_id()
            initialize_sequence_mapping(sequence_id)
        
        # 2.2 为每个活跃序列添加新token
        for sequence_id in active_sequences:
            new_token_kv = compute_new_token_kv(sequence_id)
            append_token_to_sequence(sequence_id, new_token_kv)
        
        # 2.3 执行批量attention计算
        batch_attention_results = compute_batch_paged_attention(active_sequences)
        
        # 2.4 生成下一个token
        next_tokens = generate_next_tokens(batch_attention_results)
        
        # 2.5 清理已完成的序列
        cleanup_completed_sequences()
    
    return generation_results

def append_token_to_sequence(sequence_id, new_key, new_value):
    """为序列添加新token的完整流程"""
    
    # ===== 步骤1: 获取序列映射表 =====
    mapping = paged_system.sequence_mappings[sequence_id]
    current_position = mapping.current_length
    
    # ===== 步骤2: 地址转换 =====
    block_id, offset = mapping.get_block_and_offset(current_position)
    
    if block_id is None:
        # ===== 步骤3: 需要分配新块 =====
        block_id = allocate_new_block_for_sequence(sequence_id)
        mapping.block_ids.append(block_id)
        offset = 0
    
    # ===== 步骤4: 存储KV数据 =====
    physical_block = paged_system.physical_blocks[block_id]
    physical_block.keys[offset] = new_key
    physical_block.values[offset] = new_value
    physical_block.used_slots = max(physical_block.used_slots, offset + 1)
    
    # ===== 步骤5: 更新元数据 =====
    mapping.current_length += 1
    physical_block.last_access = time.time()
    physical_block.owners.add(sequence_id)
    
    return block_id, offset

def allocate_new_block_for_sequence(sequence_id):
    """为序列分配新块的详细流程"""
    
    # ===== 步骤1: 检查空闲块 =====
    if paged_system.free_blocks:
        block_id = paged_system.free_blocks.pop()
        paged_system.used_blocks.add(block_id)
        return block_id
    
    # ===== 步骤2: 内存不足，触发垃圾回收 =====
    freed_blocks = paged_system.garbage_collector.collect()
    
    if freed_blocks:
        block_id = freed_blocks[0]
        paged_system.used_blocks.add(block_id)
        return block_id
    
    # ===== 步骤3: 仍然不足，执行LRU置换 =====
    lru_block_id = find_lru_block()
    if lru_block_id is not None:
        evict_block(lru_block_id)
        return lru_block_id
    
    # ===== 步骤4: 分配失败 =====
    raise OutOfMemoryError("无法分配新的内存块")

def compute_batch_paged_attention(active_sequences):
    """批量PagedAttention计算流程"""
    
    batch_results = []
    
    # ===== 步骤1: 收集所有需要的块 =====
    required_blocks = set()
    for seq_id in active_sequences:
        mapping = paged_system.sequence_mappings[seq_id]
        required_blocks.update(mapping.block_ids)
    
    # ===== 步骤2: 批量预取块到GPU缓存 =====
    prefetch_blocks_to_gpu(required_blocks)
    
    # ===== 步骤3: 并行计算每个序列的attention =====
    for seq_id in active_sequences:
        attention_result = compute_single_sequence_attention(seq_id)
        batch_results.append(attention_result)
    
    return batch_results

def compute_single_sequence_attention(sequence_id):
    """单个序列的attention计算"""
    
    # ===== 步骤1: 获取序列信息 =====
    mapping = paged_system.sequence_mappings[sequence_id]
    query = get_current_query(sequence_id)
    
    # ===== 步骤2: 收集所有KV数据 =====
    all_keys = []
    all_values = []
    
    for block_id in mapping.block_ids:
        physical_block = paged_system.physical_blocks[block_id]
        
        # 只取已使用的部分
        used_keys = physical_block.keys[:physical_block.used_slots]
        used_values = physical_block.values[:physical_block.used_slots]
        
        all_keys.append(used_keys)
        all_values.append(used_values)
    
    # ===== 步骤3: 拼接KV数据 =====
    concatenated_keys = torch.cat(all_keys, dim=0)  # [total_tokens, hidden_dim]
    concatenated_values = torch.cat(all_values, dim=0)
    
    # ===== 步骤4: 计算attention =====
    # 使用FlashAttention优化计算
    attention_output = flash_attention(
        query=query,
        keys=concatenated_keys,
        values=concatenated_values
    )
    
    return attention_output
```

#### 3.2 地址转换流程详解

```python
def virtual_to_physical_translation(sequence_id, token_position):
    """虚拟地址到物理地址的转换流程"""
    
    # ===== 步骤1: 解析虚拟地址 =====
    block_index = token_position // BLOCK_SIZE  # 虚拟块号
    intra_block_offset = token_position % BLOCK_SIZE  # 块内偏移
    
    # ===== 步骤2: 查找映射表 (页表查找) =====
    mapping = paged_system.sequence_mappings[sequence_id]
    
    if block_index >= len(mapping.block_ids):
        # 页面缺失 - 需要分配新块
        trigger_page_fault(sequence_id, block_index)
        return None, None
    
    # ===== 步骤3: 获取物理块ID =====
    physical_block_id = mapping.block_ids[block_index]
    
    # ===== 步骤4: 返回物理地址 =====
    return physical_block_id, intra_block_offset

def trigger_page_fault(sequence_id, missing_block_index):
    """页面缺失处理"""
    
    # ===== 步骤1: 分配新的物理块 =====
    new_block_id = allocate_new_block_for_sequence(sequence_id)
    
    # ===== 步骤2: 扩展映射表 =====
    mapping = paged_system.sequence_mappings[sequence_id]
    
    # 确保映射表有足够长度
    while len(mapping.block_ids) <= missing_block_index:
        if len(mapping.block_ids) == missing_block_index:
            mapping.block_ids.append(new_block_id)
        else:
            mapping.block_ids.append(None)  # 占位符
    
    # ===== 步骤3: 初始化新块 =====
    initialize_physical_block(new_block_id, sequence_id)
    
    return new_block_id
```

### 4. 内存管理流程图示

#### 4.1 块分配流程图

```
块分配决策流程

开始分配请求
      ↓
┌─────────────────┐
│ 检查空闲块池     │
│ free_blocks     │
└─────────────────┘
      ↓
  [有空闲块?]
      ↓ YES
┌─────────────────┐
│ 直接分配空闲块   │
│ 时间: O(1)      │
└─────────────────┘
      ↓
   分配完成
      
  [有空闲块?]
      ↓ NO
┌─────────────────┐
│ 触发垃圾回收     │  
│ gc.collect()    │
└─────────────────┘
      ↓
  [回收到块?]
      ↓ YES
┌─────────────────┐
│ 使用回收的块     │
│ 时间: O(n)      │
└─────────────────┘
      ↓
   分配完成

  [回收到块?]
      ↓ NO
┌─────────────────┐
│ LRU置换算法     │
│ 淘汰最久未用块   │
└─────────────────┘
      ↓
  [找到LRU块?]
      ↓ YES
┌─────────────────┐
│ 换出LRU块       │
│ 分配给新序列     │
└─────────────────┘
      ↓
   分配完成

  [找到LRU块?]
      ↓ NO
┌─────────────────┐
│ 内存不足异常     │
│ OutOfMemoryError│
└─────────────────┘
      ↓
   分配失败
```

#### 4.2 序列生命周期内存变化图

```
序列生命周期的内存使用变化

时间轴: t0 → t1 → t2 → t3 → t4 → t5
        ↓    ↓    ↓    ↓    ↓    ↓

t0: 序列初始化
┌─────────────────────────────────────────────────────────────┐
│ sequence_1: []                                              │
│ 物理内存: [block_0][block_1][block_2][...] (全部空闲)       │
└─────────────────────────────────────────────────────────────┘

t1: 生成前8个token (占用1个块)
┌─────────────────────────────────────────────────────────────┐
│ sequence_1: [block_0] (8/16 slots used)                    │
│ 物理内存: [████████░░░░░░░░][block_1][block_2][...]         │
│           ↑ block_0 部分使用                                │
└─────────────────────────────────────────────────────────────┘

t2: 生成到第20个token (需要2个块)
┌─────────────────────────────────────────────────────────────┐
│ sequence_1: [block_0, block_1] (20 tokens total)           │
│ 物理内存: [████████████████][████░░░░░░░░░░░░][block_2][...] │
│           ↑ block_0 满       ↑ block_1 部分使用             │
└─────────────────────────────────────────────────────────────┘

t3: 新序列sequence_2开始
┌─────────────────────────────────────────────────────────────┐
│ sequence_1: [block_0, block_1] (20 tokens)                 │
│ sequence_2: [block_2] (5 tokens)                           │  
│ 物理内存: [████████████████][████░░░░░░░░░░░░][█████░░░░░░░░...│
└─────────────────────────────────────────────────────────────┘

t4: sequence_1完成，释放内存
┌─────────────────────────────────────────────────────────────┐
│ sequence_1: 已完成，释放block_0, block_1                    │
│ sequence_2: [block_2] (继续生成)                            │
│ 物理内存: [空闲 block_0][空闲 block_1][█████░░░░░░░░][...]   │
└─────────────────────────────────────────────────────────────┘

t5: 新序列sequence_3复用释放的块
┌─────────────────────────────────────────────────────────────┐
│ sequence_2: [block_2] (序列继续)                            │
│ sequence_3: [block_0] (复用sequence_1释放的块)              │
│ 物理内存: [██░░░░░░░░░░░░░░][空闲 block_1][████████░░░░░░░...│
│           ↑ sequence_3使用  ↑ 仍然空闲     ↑ sequence_2使用  │
└─────────────────────────────────────────────────────────────┘
```

### 5. 与传统方式的对比流程

#### 5.1 传统连续分配 vs PagedAttention

```python
# 传统方式的内存分配流程
def traditional_memory_allocation():
    """传统连续内存分配流程"""
    
    def allocate_sequence_traditional(seq_id, estimated_length):
        # 必须一次性分配整个预估长度的连续内存
        required_memory = estimated_length * hidden_dim * 2  # K + V
        
        # 查找足够大的连续内存块
        for memory_region in free_memory_regions:
            if memory_region.size >= required_memory:
                # 分配整个区域
                allocated_memory = memory_region.allocate(required_memory)
                
                # 即使只使用一小部分，也占用整个分配区域
                sequence_memory[seq_id] = allocated_memory
                return allocated_memory
        
        # 如果没有足够大的连续区域，分配失败
        raise OutOfMemoryError("无法找到足够大的连续内存")
    
    def append_token_traditional(seq_id, new_key, new_value):
        # 直接在预分配的连续内存中添加
        memory_region = sequence_memory[seq_id]
        current_pos = sequence_lengths[seq_id]
        
        if current_pos >= memory_region.capacity:
            # 超出预分配容量，必须重新分配更大的连续内存
            new_capacity = memory_region.capacity * 2
            new_memory = allocate_sequence_traditional(seq_id, new_capacity)
            copy_existing_data(memory_region, new_memory)
            free_memory(memory_region)  # 留下内存碎片
        
        memory_region[current_pos] = (new_key, new_value)
        sequence_lengths[seq_id] += 1

# PagedAttention的内存分配流程
def paged_memory_allocation():
    """PagedAttention分块内存分配流程"""
    
    def allocate_sequence_paged(seq_id):
        # 只需要创建空的映射表，不预分配任何物理内存
        paged_system.sequence_mappings[seq_id] = SequenceMapping(seq_id)
        return True  # 总是成功
    
    def append_token_paged(seq_id, new_key, new_value):
        mapping = paged_system.sequence_mappings[seq_id]
        current_pos = mapping.current_length
        
        # 地址转换：虚拟位置 → 块ID + 偏移
        block_id, offset = mapping.get_block_and_offset(current_pos)
        
        if block_id is None:
            # 只在需要时分配新块(16 tokens容量)
            block_id = allocate_single_block()  # 固定大小，无碎片
            mapping.block_ids.append(block_id)
            offset = 0
        
        # 在固定大小的块中存储KV
        physical_block = paged_system.physical_blocks[block_id]
        physical_block.keys[offset] = new_key
        physical_block.values[offset] = new_value
        
        mapping.current_length += 1
```

#### 5.2 性能对比流程分析

```python
def performance_comparison_analysis():
    """性能对比分析"""
    
    comparison_scenarios = {
        "内存分配延迟": {
            "traditional": {
                "best_case": "O(1) - 有合适的连续区域",
                "worst_case": "O(n) - 需要搜索或碎片整理",
                "average_case": "O(log n) - 需要在空闲列表中搜索"
            },
            "paged_attention": {
                "best_case": "O(1) - 从空闲块池直接获取",
                "worst_case": "O(1) - 块大小固定，无搜索开销", 
                "average_case": "O(1) - 一致的性能"
            }
        },
        
        "内存利用率": {
            "traditional": {
                "utilization": "40-70% (大量预分配浪费)",
                "fragmentation": "30-40% (连续分配导致碎片)",
                "scalability": "差 (长序列阻塞短序列)"
            },
            "paged_attention": {
                "utilization": "90-95% (按需分配)",
                "fragmentation": "5-10% (固定块大小)",
                "scalability": "优秀 (灵活的块复用)"
            }
        },
        
        "批处理能力": {
            "traditional": {
                "max_batch_size": "受最长序列限制",
                "memory_efficiency": "按最坏情况分配",
                "dynamic_adjustment": "不支持运行时调整"
            },
            "paged_attention": {
                "max_batch_size": "按实际使用量限制",
                "memory_efficiency": "按实际需求分配",
                "dynamic_adjustment": "完全支持动态调整"
            }
        }
    }
    
    return comparison_scenarios
```

### 6. 关键优化技术的伪代码实现

#### 6.1 Copy-on-Write优化

```python
def copy_on_write_optimization():
    """写时复制优化伪代码"""
    
    def share_prefix_blocks(base_seq_id, new_seq_id, shared_length):
        """共享前缀块"""
        
        # 计算需要共享的块数
        shared_blocks = math.ceil(shared_length / BLOCK_SIZE)
        base_mapping = paged_system.sequence_mappings[base_seq_id]
        
        # 创建新序列的映射表
        new_mapping = SequenceMapping(new_seq_id)
        
        # 共享前缀块
        for i in range(shared_blocks):
            shared_block_id = base_mapping.block_ids[i]
            
            # 增加引用计数
            physical_block = paged_system.physical_blocks[shared_block_id]
            physical_block.ref_count += 1
            physical_block.owners.add(new_seq_id)
            
            # 标记为共享状态
            physical_block.is_shared = True
            
            # 添加到新序列的映射表
            new_mapping.block_ids.append(shared_block_id)
        
        paged_system.sequence_mappings[new_seq_id] = new_mapping
    
    def trigger_copy_on_write(seq_id, block_index):
        """触发写时复制"""
        
        mapping = paged_system.sequence_mappings[seq_id]
        shared_block_id = mapping.block_ids[block_index]
        shared_block = paged_system.physical_blocks[shared_block_id]
        
        if shared_block.ref_count > 1:
            # 需要复制
            new_block_id = allocate_single_block()
            new_block = paged_system.physical_blocks[new_block_id]
            
            # 复制数据
            new_block.keys[:] = shared_block.keys[:]
            new_block.values[:] = shared_block.values[:]
            new_block.used_slots = shared_block.used_slots
            new_block.ref_count = 1
            new_block.owners = {seq_id}
            
            # 更新引用
            shared_block.ref_count -= 1
            shared_block.owners.remove(seq_id)
            
            # 更新映射表
            mapping.block_ids[block_index] = new_block_id
            
            return new_block_id
        
        return shared_block_id
```

#### 6.2 自适应块大小优化

```python
def adaptive_block_size_optimization():
    """自适应块大小优化"""
    
    def analyze_sequence_patterns():
        """分析序列模式来优化块大小"""
        
        # 收集统计数据
        sequence_stats = {}
        for seq_id, mapping in paged_system.sequence_mappings.items():
            stats = {
                "average_length": calculate_average_length(seq_id),
                "variance": calculate_length_variance(seq_id),
                "access_pattern": analyze_access_pattern(seq_id)
            }
            sequence_stats[seq_id] = stats
        
        # 基于统计数据调整块大小
        if should_use_larger_blocks(sequence_stats):
            return 32  # 使用更大的块
        elif should_use_smaller_blocks(sequence_stats):
            return 8   # 使用更小的块
        else:
            return 16  # 保持默认大小
    
    def dynamic_block_size_adjustment():
        """动态调整块大小"""
        
        current_stats = collect_runtime_statistics()
        
        # 基于运行时统计调整策略
        optimization_decisions = {
            "memory_pressure_high": "减小块大小以提高利用率",
            "access_locality_poor": "调整块布局以优化缓存命中",
            "sequence_length_varied": "使用混合块大小策略"
        }
        
        for condition, action in optimization_decisions.items():
            if evaluate_condition(condition, current_stats):
                execute_optimization_action(action)
```

### 7. 总结

PagedAttention的核心流程通过以下关键技术实现了革命性的内存管理优化：

**核心创新点：**
1. **虚拟地址映射** - 将序列token位置抽象为虚拟地址，通过映射表转换为物理块位置
2. **固定块分配** - 使用固定大小的内存块，消除内存碎片和分配延迟
3. **按需分配** - 只在实际需要时分配内存块，避免预分配浪费
4. **动态回收** - 序列完成后立即释放所有块，可被其他序列复用

**流程优势：**
- **O(1)时间复杂度** - 地址转换和块分配都是常数时间
- **高内存利用率** - 从传统的60-70%提升到90-95%
- **优秀可扩展性** - 支持动态长度和大规模批处理
- **低延迟特性** - 消除了内存分配和碎片整理的延迟

通过这套完整的流程设计，PagedAttention成功地将操作系统虚拟内存管理的成熟技术应用到AI推理场景，实现了显著的性能提升。

---

## 相关笔记
<!-- 自动生成 -->

- [请详细解释PagedAttention算法的工作原理，它是如何借鉴操作系统虚拟内存技术的？](notes/vllm/请详细解释PagedAttention算法的工作原理，它是如何借鉴操作系统虚拟内存技术的？.md) - 相似度: 31% | 标签: vllm, vllm/请详细解释PagedAttention算法的工作原理，它是如何借鉴操作系统虚拟内存技术的？.md

