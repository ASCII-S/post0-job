---
created: '2025-10-19'
last_reviewed: null
next_review: '2025-10-19'
review_count: 0
difficulty: medium
mastery_level: 0.0
tags:
- LeetcodeHot100
- LeetcodeHot100/LRU缓存.md
related_outlines: []
---
# LRU 缓存

## 题目描述

请你设计并实现一个满足 **LRU (最近最少使用)** 缓存约束的数据结构。

实现 `LRUCache` 类：
- `LRUCache(int capacity)` 以**正整数**作为容量 `capacity` 初始化 LRU 缓存
- `int get(int key)` 如果关键字 `key` 存在于缓存中，则返回关键字的值，否则返回 `-1`。
- `void put(int key, int value)` 如果关键字 `key` 已经存在，则变更其数据值 `value`；如果不存在，则向缓存中插入该组 `key-value`。如果插入操作导致关键字数量超过 `capacity`，则应该**逐出**最久未使用的关键字。

函数 `get` 和 `put` 必须以 **O(1)** 的平均时间复杂度运行。

**示例：**
```
输入
["LRUCache", "put", "put", "get", "put", "get", "put", "get", "get", "get"]
[[2], [1, 1], [2, 2], [1], [3, 3], [2], [4, 4], [1], [3], [4]]
输出
[null, null, null, 1, null, -1, null, -1, 3, 4]

解释
LRUCache lRUCache = new LRUCache(2);
lRUCache.put(1, 1); // 缓存是 {1=1}
lRUCache.put(2, 2); // 缓存是 {1=1, 2=2}
lRUCache.get(1);    // 返回 1
lRUCache.put(3, 3); // 该操作会使得关键字 2 作废，缓存是 {1=1, 3=3}
lRUCache.get(2);    // 返回 -1 (未找到)
lRUCache.put(4, 4); // 该操作会使得关键字 1 作废，缓存是 {4=4, 3=3}
lRUCache.get(1);    // 返回 -1 (未找到)
lRUCache.get(3);    // 返回 3
lRUCache.get(4);    // 返回 4
```

**提示：**
- 1 <= capacity <= 3000
- 0 <= key <= 10000
- 0 <= value <= 10^5
- 最多调用 2 * 10^5 次 `get` 和 `put`

## 思路讲解

LRU（Least Recently Used，最近最少使用）缓存是一个经典的数据结构设计问题。

### 核心要求

**两个 O(1) 操作：**
1. **查找**：快速判断 key 是否存在，并返回 value
2. **更新**：快速移动/删除/插入节点，维护使用顺序

**单一数据结构的局限性：**
- **哈希表**：查找 O(1)，但无法维护顺序
- **数组**：维护顺序，但查找和删除都是 O(n)
- **链表**：插入删除 O(1)，但查找是 O(n)

### 最优解：哈希表 + 双向链表

**数据结构组合：**
1. **哈希表**：key -> 链表节点，实现 O(1) 查找
2. **双向链表**：维护使用顺序，实现 O(1) 插入/删除

**为什么用双向链表？**
- 需要快速删除节点（知道节点位置时，删除是 O(1)）
- 单向链表删除节点需要知道前驱节点，需要 O(n) 查找
- 双向链表可以直接通过节点找到前驱和后继

**链表节点顺序：**
- 链表头部（head 之后）：最近使用的元素
- 链表尾部（tail 之前）：最久未使用的元素

**核心操作：**
1. **get(key)**：
   - 在哈希表中查找 key
   - 如果存在，将对应节点移到链表头部（更新使用时间）
   - 返回 value

2. **put(key, value)**：
   - 如果 key 已存在：
     - 更新节点的 value
     - 将节点移到链表头部
   - 如果 key 不存在：
     - 创建新节点，插入链表头部
     - 添加到哈希表
     - 如果超出容量，删除链表尾部节点（最久未使用）

## 面试时的快速口述讲解

LRU 缓存要求 get 和 put 操作都是 O(1) 时间复杂度，需要结合哈希表和双向链表。

**数据结构**：哈希表 + 双向链表。

**设计思路**：
1. 哈希表存储 key 到链表节点的映射，实现 O(1) 查找
2. 双向链表维护访问顺序，头部是最近使用的，尾部是最久未使用的
3. 使用虚拟头尾节点简化边界处理

**get 操作**：
1. 在哈希表中查找 key
2. 如果不存在，返回 -1
3. 如果存在，将对应节点移到链表头部，返回 value

**put 操作**：
1. 如果 key 已存在，更新 value，将节点移到头部
2. 如果 key 不存在：
   - 创建新节点，插入链表头部
   - 添加到哈希表
   - 如果超出容量，删除链表尾部节点（最久未使用的）

**时间复杂度**：get 和 put 都是 O(1)。

**空间复杂度**：O(capacity)，存储最多 capacity 个节点。

## 代码实现

```cpp
class LRUCache {
private:
    // 双向链表节点
    struct DLinkedNode {
        int key;
        int value;
        DLinkedNode* prev;
        DLinkedNode* next;
        
        DLinkedNode() : key(0), value(0), prev(nullptr), next(nullptr) {}
        DLinkedNode(int k, int v) : key(k), value(v), prev(nullptr), next(nullptr) {}
    };
    
    // 哈希表：key -> 节点指针
    unordered_map<int, DLinkedNode*> cache;
    
    // 虚拟头尾节点
    DLinkedNode* head;
    DLinkedNode* tail;
    
    // 容量和当前大小
    int capacity;
    int size;
    
public:
    LRUCache(int capacity) : capacity(capacity), size(0) {
        // 创建虚拟头尾节点
        head = new DLinkedNode();
        tail = new DLinkedNode();
        head->next = tail;
        tail->prev = head;
    }
    
    int get(int key) {
        // 如果 key 不存在
        if (cache.find(key) == cache.end()) {
            return -1;
        }
        
        // key 存在，获取节点
        DLinkedNode* node = cache[key];
        
        // 将节点移到头部（表示最近使用）
        moveToHead(node);
        
        return node->value;
    }
    
    void put(int key, int value) {
        // 如果 key 已存在
        if (cache.find(key) != cache.end()) {
            DLinkedNode* node = cache[key];
            // 更新 value
            node->value = value;
            // 移到头部
            moveToHead(node);
        }
        // key 不存在
        else {
            // 创建新节点
            DLinkedNode* node = new DLinkedNode(key, value);
            // 添加到哈希表
            cache[key] = node;
            // 添加到链表头部
            addToHead(node);
            size++;
            
            // 如果超出容量，删除尾部节点
            if (size > capacity) {
                // 删除链表尾部节点
                DLinkedNode* removed = removeTail();
                // 从哈希表中删除
                cache.erase(removed->key);
                // 释放内存
                delete removed;
                size--;
            }
        }
    }
    
private:
    // 将节点添加到头部（head 之后）
    void addToHead(DLinkedNode* node) {
        node->prev = head;
        node->next = head->next;
        head->next->prev = node;
        head->next = node;
    }
    
    // 删除节点
    void removeNode(DLinkedNode* node) {
        node->prev->next = node->next;
        node->next->prev = node->prev;
    }
    
    // 将节点移到头部
    void moveToHead(DLinkedNode* node) {
        removeNode(node);
        addToHead(node);
    }
    
    // 删除尾部节点（tail 之前的节点）
    DLinkedNode* removeTail() {
        DLinkedNode* node = tail->prev;
        removeNode(node);
        return node;
    }
};

/**
 * Your LRUCache object will be instantiated and called as such:
 * LRUCache* obj = new LRUCache(capacity);
 * int param_1 = obj->get(key);
 * obj->put(key,value);
 */
```

**代码说明：**

**数据结构设计：**
```
哈希表：{key -> 节点指针}

双向链表：
head <-> node1 <-> node2 <-> node3 <-> tail
         (最近)                (最久)

虚拟头尾节点的作用：
- 简化边界处理
- 不需要判断节点是否为 null
```

**核心辅助函数：**

1. **addToHead(node)**：将节点添加到链表头部
   ```
   head <-> A <-> B <-> tail
   
   添加 node：
   head <-> node <-> A <-> B <-> tail
   ```

2. **removeNode(node)**：从链表中删除节点
   ```
   A <-> node <-> B
   
   删除后：
   A <-> B
   ```

3. **moveToHead(node)**：将节点移到头部（先删除，再添加到头部）

4. **removeTail()**：删除尾部节点（最久未使用的）

**图解过程（capacity=2）：**
```
put(1, 1):
  链表：head <-> (1,1) <-> tail
  哈希表：{1: node1}

put(2, 2):
  链表：head <-> (2,2) <-> (1,1) <-> tail
  哈希表：{1: node1, 2: node2}

get(1):  返回 1
  将 (1,1) 移到头部
  链表：head <-> (1,1) <-> (2,2) <-> tail

put(3, 3):  容量已满，删除最久未使用的 (2,2)
  链表：head <-> (3,3) <-> (1,1) <-> tail
  哈希表：{1: node1, 3: node3}

get(2):  返回 -1（已被删除）

put(4, 4):  容量已满，删除最久未使用的 (1,1)
  链表：head <-> (4,4) <-> (3,3) <-> tail
  哈希表：{3: node3, 4: node4}
```

**时间复杂度分析：**
- get：O(1)，哈希表查找 + 链表节点移动都是 O(1)
- put：O(1)，哈希表操作 + 链表操作都是 O(1)

**空间复杂度：**
- O(capacity)，最多存储 capacity 个节点

**面试技巧：**
- 强调为什么需要双向链表（单向链表删除需要 O(n)）
- 说明虚拟头尾节点的作用
- 画图辅助说明链表操作
- 这是一道综合性很强的题目，考察数据结构设计能力


---

## 相关笔记
<!-- 自动生成 -->

暂无相关笔记

