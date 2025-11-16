---
created: '2025-10-19'
last_reviewed: null
next_review: '2025-10-19'
review_count: 0
difficulty: medium
mastery_level: 0.0
tags:
- C++
- C++/map、unordered_map的实现原理和性能差异.md
related_outlines: []
---
## 标准答案（可背诵）

### map的实现原理

**底层结构**：**红黑树**（自平衡二叉搜索树）

**特点**：
1. **有序性**：元素按键自动排序（默认升序）
2. **时间复杂度**：查找、插入、删除均为 **O(log n)**
3. **内存布局**：节点分散在堆上，需要额外存储父节点、左右子节点指针和颜色信息
4. **迭代器**：双向迭代器，支持++和--操作

---

### unordered_map的实现原理

**底层结构**：**哈希表**（散列表 + 链表/开放寻址）

**特点**：
1. **无序性**：元素无序存储，遍历顺序不确定
2. **时间复杂度**：查找、插入、删除平均为 **O(1)**，最坏 **O(n)**（哈希冲突严重时）
3. **内存布局**：桶数组 + 链表（或红黑树，C++11后冲突过多时）
4. **迭代器**：前向迭代器，只支持++操作

---

### 性能差异对比

| 特性       | map                | unordered_map       |
| ---------- | ------------------ | ------------------- |
| 底层实现   | 红黑树             | 哈希表              |
| 时间复杂度 | O(log n)           | 平均O(1)，最坏O(n)  |
| 是否有序   | 有序（按键排序）   | 无序                |
| 内存开销   | 较小（节点+指针）  | 较大（桶数组+链表） |
| 缓存友好性 | 差（节点分散）     | 中等（桶连续）      |
| 插入性能   | 稳定O(log n)       | 通常更快，但不稳定  |
| 查找性能   | 稳定O(log n)       | 通常最快            |
| 适用场景   | 需要有序、范围查询 | 需要快速查找        |

**选择建议**：
- 需要**有序**或**范围查询** → 使用 `map`
- 需要**最快查找**且不关心顺序 → 使用 `unordered_map`
- 数据量小或性能差异不明显 → 优先 `unordered_map`

---

## 详细讲解

### 1. map的实现原理

#### 1.1 红黑树基础

红黑树是一种自平衡的二叉搜索树，满足以下性质：
1. 每个节点是红色或黑色
2. 根节点是黑色
3. 所有叶子节点（NULL）是黑色
4. 红色节点的子节点必须是黑色（不能有连续的红节点）
5. 从任一节点到其叶子节点的所有路径包含相同数量的黑色节点

```cpp
// 红黑树节点结构（简化版）
template<typename Key, typename Value>
struct RBTreeNode {
    std::pair<const Key, Value> data;  // 键值对
    RBTreeNode* parent;                // 父节点
    RBTreeNode* left;                  // 左子节点
    RBTreeNode* right;                 // 右子节点
    enum Color { RED, BLACK } color;   // 节点颜色
    
    RBTreeNode(const Key& k, const Value& v)
        : data(k, v), parent(nullptr), 
          left(nullptr), right(nullptr), color(RED) {}
};
```

#### 1.2 map的内部结构

```cpp
// map的实现原理（简化版）
template<typename Key, typename Value, typename Compare = std::less<Key>>
class map {
private:
    RBTreeNode<Key, Value>* root;  // 红黑树根节点
    size_t nodeCount;               // 元素数量
    Compare comp;                   // 比较函数对象
    
public:
    // 插入操作
    std::pair<iterator, bool> insert(const std::pair<Key, Value>& kv) {
        // 1. 按BST规则找到插入位置
        // 2. 插入新节点（初始为红色）
        // 3. 修复红黑树性质（旋转+变色）
        // 时间复杂度：O(log n)
    }
    
    // 查找操作
    iterator find(const Key& key) {
        // 1. 从根节点开始
        // 2. 根据比较结果向左或向右查找
        // 3. 找到返回迭代器，否则返回end()
        // 时间复杂度：O(log n)
    }
    
    // 删除操作
    void erase(const Key& key) {
        // 1. 找到要删除的节点
        // 2. 删除节点
        // 3. 修复红黑树性质
        // 时间复杂度：O(log n)
    }
};
```

#### 1.3 map的插入过程示例

```cpp
#include <map>
#include <iostream>

void demonstrateMapInsertion() {
    std::map<int, std::string> m;
    
    // 插入元素
    m.insert({3, "three"});  // 成为根节点
    m.insert({1, "one"});    // 插入左子树
    m.insert({5, "five"});   // 插入右子树
    m.insert({2, "two"});    // 插入后可能需要旋转
    m.insert({4, "four"});   // 插入后可能需要旋转
    
    // 红黑树会自动平衡，保证高度为O(log n)
    
    // 遍历：按键的升序
    for (const auto& [key, value] : m) {
        std::cout << key << ": " << value << std::endl;
    }
    // 输出：1: one, 2: two, 3: three, 4: four, 5: five
}
```

#### 1.4 map的时间复杂度分析

```cpp
void mapTimeComplexity() {
    std::map<int, std::string> m;
    
    // 插入：O(log n)
    // - 查找插入位置：O(log n)
    // - 插入节点：O(1)
    // - 旋转和重新着色：O(1)（最多旋转2次）
    m.insert({1, "one"});
    
    // 查找：O(log n)
    // - 二叉搜索树查找
    auto it = m.find(1);
    
    // 删除：O(log n)
    // - 查找节点：O(log n)
    // - 删除节点：O(1)
    // - 旋转和重新着色：O(1)（最多旋转3次）
    m.erase(1);
    
    // 访问：O(log n)
    // operator[] 如果键不存在会插入
    m[2] = "two";  // 可能需要插入，O(log n)
    
    // 范围查询：O(log n + k)，k是结果数量
    auto lower = m.lower_bound(1);  // O(log n)
    auto upper = m.upper_bound(5);  // O(log n)
    for (auto it = lower; it != upper; ++it) {
        // 遍历k个元素：O(k)
    }
}
```

#### 1.5 map的优缺点

**优点**：
```cpp
// 1. 自动排序
std::map<int, std::string> m = {{3, "c"}, {1, "a"}, {2, "b"}};
// 自动按键排序：1, 2, 3

// 2. 支持范围查询
auto lower = m.lower_bound(2);  // >= 2的第一个元素
auto upper = m.upper_bound(4);  // > 4的第一个元素

// 3. 性能稳定
// 无论什么数据，都是O(log n)，不会退化

// 4. 迭代器稳定
// 插入、删除不会使其他迭代器失效（除了被删除的元素）
```

**缺点**：
```cpp
// 1. 性能不是最优
// O(log n) 慢于 O(1)

// 2. 内存开销大
// 每个节点需要存储3个指针+颜色信息

// 3. 缓存不友好
// 节点分散在内存中，缓存命中率低
```

---

### 2. unordered_map的实现原理

#### 2.1 哈希表基础

哈希表通过哈希函数将键映射到桶（bucket）中，实现快速查找。

```cpp
// 哈希表节点结构（简化版）
template<typename Key, typename Value>
struct HashNode {
    std::pair<const Key, Value> data;  // 键值对
    HashNode* next;                     // 链表指针（处理冲突）
    
    HashNode(const Key& k, const Value& v)
        : data(k, v), next(nullptr) {}
};
```

#### 2.2 unordered_map的内部结构

```cpp
// unordered_map的实现原理（简化版）
template<typename Key, typename Value, 
         typename Hash = std::hash<Key>,
         typename KeyEqual = std::equal_to<Key>>
class unordered_map {
private:
    std::vector<HashNode<Key, Value>*> buckets;  // 桶数组
    size_t elementCount;                          // 元素数量
    float maxLoadFactor;                          // 最大负载因子
    Hash hashFunction;                            // 哈希函数
    KeyEqual keyEqual;                            // 键比较函数
    
public:
    // 插入操作
    std::pair<iterator, bool> insert(const std::pair<Key, Value>& kv) {
        // 1. 计算哈希值
        size_t hashValue = hashFunction(kv.first);
        size_t bucketIndex = hashValue % buckets.size();
        
        // 2. 检查是否已存在
        for (auto* node = buckets[bucketIndex]; node; node = node->next) {
            if (keyEqual(node->data.first, kv.first)) {
                return {iterator(node), false};  // 已存在
            }
        }
        
        // 3. 插入新节点到链表头部
        auto* newNode = new HashNode<Key, Value>(kv.first, kv.second);
        newNode->next = buckets[bucketIndex];
        buckets[bucketIndex] = newNode;
        ++elementCount;
        
        // 4. 检查是否需要rehash
        if (loadFactor() > maxLoadFactor) {
            rehash(buckets.size() * 2);
        }
        
        return {iterator(newNode), true};
        // 平均时间复杂度：O(1)
    }
    
    // 查找操作
    iterator find(const Key& key) {
        size_t hashValue = hashFunction(key);
        size_t bucketIndex = hashValue % buckets.size();
        
        // 遍历该桶的链表
        for (auto* node = buckets[bucketIndex]; node; node = node->next) {
            if (keyEqual(node->data.first, key)) {
                return iterator(node);
            }
        }
        
        return end();
        // 平均时间复杂度：O(1)
        // 最坏时间复杂度：O(n)（所有元素在一个桶）
    }
    
    float loadFactor() const {
        return static_cast<float>(elementCount) / buckets.size();
    }
    
    void rehash(size_t newBucketCount) {
        // 重新分配桶，并重新插入所有元素
        // 时间复杂度：O(n)
    }
};
```

#### 2.3 哈希冲突的处理

```cpp
void demonstrateHashCollision() {
    // 假设简单的哈希函数：key % 10
    std::unordered_map<int, std::string> m;
    
    m[11] = "eleven";   // hash: 11 % 10 = 1
    m[21] = "twenty-one"; // hash: 21 % 10 = 1（冲突！）
    m[31] = "thirty-one"; // hash: 31 % 10 = 1（冲突！）
    
    // 处理方式1：链地址法（C++常用）
    // bucket[1] -> [11, "eleven"] -> [21, "twenty-one"] -> [31, "thirty-one"]
    
    // 处理方式2：开放寻址（线性探测、二次探测、双重哈希）
    // 如果bucket[1]已占用，尝试bucket[2], bucket[3]...
    
    // C++11标准：当链表过长时，转换为红黑树（提升性能）
}
```

#### 2.4 负载因子和rehash

```cpp
void demonstrateRehash() {
    std::unordered_map<int, std::string> m;
    
    // 默认最大负载因子：1.0
    std::cout << "Max load factor: " << m.max_load_factor() << std::endl;
    
    // 插入元素
    for (int i = 0; i < 100; ++i) {
        m[i] = std::to_string(i);
        
        // 当 元素数量/桶数量 > 最大负载因子时，触发rehash
        if (m.load_factor() > m.max_load_factor()) {
            // 桶数量翻倍
            // 重新计算所有元素的桶位置
            // 时间复杂度：O(n)
        }
    }
    
    std::cout << "Bucket count: " << m.bucket_count() << std::endl;
    std::cout << "Load factor: " << m.load_factor() << std::endl;
    
    // 可以手动设置
    m.max_load_factor(0.5);  // 降低负载因子，减少冲突但增加内存
    m.reserve(200);          // 预留空间，避免频繁rehash
}
```

#### 2.5 unordered_map的时间复杂度

```cpp
void unorderedMapTimeComplexity() {
    std::unordered_map<int, std::string> m;
    
    // 插入：平均O(1)，最坏O(n)
    // - 计算哈希：O(1)
    // - 查找桶：O(1)
    // - 遍历链表：平均O(1)，最坏O(n)
    // - 如果需要rehash：O(n)（摊还后仍是O(1)）
    m.insert({1, "one"});
    
    // 查找：平均O(1)，最坏O(n)
    // - 计算哈希：O(1)
    // - 查找桶：O(1)
    // - 遍历链表：平均O(1)，最坏O(n)
    auto it = m.find(1);
    
    // 删除：平均O(1)，最坏O(n)
    // - 查找：平均O(1)
    // - 删除节点：O(1)
    m.erase(1);
    
    // 访问：平均O(1)，最坏O(n)
    m[2] = "two";
    
    // 注意：不支持范围查询
    // 因为元素无序，没有lower_bound/upper_bound
}
```

#### 2.6 unordered_map的优缺点

**优点**：
```cpp
// 1. 查找速度快
// 平均O(1)，通常比map快

// 2. 插入速度快
// 平均O(1)，不需要维护树结构

// 3. 简单场景性能更好
std::unordered_map<int, int> cache;
cache[key] = value;  // 非常快
```

**缺点**：
```cpp
// 1. 无序
// 不能依赖遍历顺序

// 2. 内存开销大
// 需要桶数组 + 链表节点
// 负载因子通常 < 1，有空闲桶

// 3. 性能不稳定
// 最坏情况O(n)（所有元素哈希到同一个桶）
// rehash时性能突降

// 4. 不支持范围查询
// 没有lower_bound/upper_bound

// 5. 需要好的哈希函数
// 自定义类型需要实现哈希函数
struct CustomKey {
    int x, y;
};

// 需要提供哈希函数
struct CustomKeyHash {
    size_t operator()(const CustomKey& k) const {
        return std::hash<int>()(k.x) ^ (std::hash<int>()(k.y) << 1);
    }
};

std::unordered_map<CustomKey, int, CustomKeyHash> m;
```

---

### 3. 性能对比实验

#### 3.1 插入性能对比

```cpp
#include <chrono>
#include <map>
#include <unordered_map>
#include <iostream>
#include <random>

void compareInsertionPerformance() {
    const int N = 100000;
    std::vector<int> randomKeys(N);
    
    // 生成随机键
    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_int_distribution<> dis(0, 1000000);
    for (int& key : randomKeys) {
        key = dis(gen);
    }
    
    // 测试map
    {
        std::map<int, int> m;
        auto start = std::chrono::high_resolution_clock::now();
        
        for (int key : randomKeys) {
            m[key] = key;
        }
        
        auto end = std::chrono::high_resolution_clock::now();
        auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end - start);
        std::cout << "map insertion: " << duration.count() << " ms" << std::endl;
    }
    
    // 测试unordered_map
    {
        std::unordered_map<int, int> um;
        auto start = std::chrono::high_resolution_clock::now();
        
        for (int key : randomKeys) {
            um[key] = key;
        }
        
        auto end = std::chrono::high_resolution_clock::now();
        auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end - start);
        std::cout << "unordered_map insertion: " << duration.count() << " ms" << std::endl;
    }
    
    // 通常结果：unordered_map更快（约2-3倍）
}
```

#### 3.2 查找性能对比

```cpp
void compareLookupPerformance() {
    const int N = 100000;
    std::map<int, int> m;
    std::unordered_map<int, int> um;
    
    // 插入相同数据
    for (int i = 0; i < N; ++i) {
        m[i] = i;
        um[i] = i;
    }
    
    // 准备查找的键
    std::vector<int> searchKeys(10000);
    for (int& key : searchKeys) {
        key = rand() % N;
    }
    
    // 测试map查找
    {
        auto start = std::chrono::high_resolution_clock::now();
        int sum = 0;
        for (int key : searchKeys) {
            auto it = m.find(key);
            if (it != m.end()) {
                sum += it->second;
            }
        }
        auto end = std::chrono::high_resolution_clock::now();
        auto duration = std::chrono::duration_cast<std::chrono::microseconds>(end - start);
        std::cout << "map lookup: " << duration.count() << " μs" << std::endl;
    }
    
    // 测试unordered_map查找
    {
        auto start = std::chrono::high_resolution_clock::now();
        int sum = 0;
        for (int key : searchKeys) {
            auto it = um.find(key);
            if (it != um.end()) {
                sum += it->second;
            }
        }
        auto end = std::chrono::high_resolution_clock::now();
        auto duration = std::chrono::duration_cast<std::chrono::microseconds>(end - start);
        std::cout << "unordered_map lookup: " << duration.count() << " μs" << std::endl;
    }
    
    // 通常结果：unordered_map更快（约3-5倍）
}
```

#### 3.3 内存占用对比

```cpp
void compareMemoryUsage() {
    const int N = 100000;
    
    // map的内存占用
    {
        std::map<int, int> m;
        for (int i = 0; i < N; ++i) {
            m[i] = i;
        }
        
        // 每个节点大小（64位系统）：
        // - pair<int, int>: 8字节
        // - parent指针: 8字节
        // - left指针: 8字节
        // - right指针: 8字节
        // - color: 1字节（通常对齐到8字节）
        // 总计：约40字节/节点
        
        size_t estimatedSize = N * 40;
        std::cout << "map estimated memory: " << estimatedSize / 1024 << " KB" << std::endl;
    }
    
    // unordered_map的内存占用
    {
        std::unordered_map<int, int> um;
        for (int i = 0; i < N; ++i) {
            um[i] = i;
        }
        
        // 桶数组 + 节点
        // - 桶数组：bucket_count * 8字节
        // - 每个节点：pair<int, int> (8字节) + next指针 (8字节) = 16字节
        // - 负载因子约1.0，所以bucket_count ≈ N
        
        size_t bucketArraySize = um.bucket_count() * 8;
        size_t nodesSize = N * 16;
        size_t estimatedSize = bucketArraySize + nodesSize;
        
        std::cout << "unordered_map bucket count: " << um.bucket_count() << std::endl;
        std::cout << "unordered_map estimated memory: " << estimatedSize / 1024 << " KB" << std::endl;
    }
    
    // 通常结果：map内存占用更大（约2倍）
    // 但unordered_map的桶数组会有空闲空间
}
```

---

### 4. 使用场景和选择建议

#### 4.1 使用map的场景

```cpp
// 场景1：需要有序遍历
void orderedTraversal() {
    std::map<std::string, int> wordCount;
    wordCount["apple"] = 5;
    wordCount["banana"] = 3;
    wordCount["cherry"] = 7;
    
    // 按字母顺序遍历
    for (const auto& [word, count] : wordCount) {
        std::cout << word << ": " << count << std::endl;
    }
    // 输出：apple: 5, banana: 3, cherry: 7（字母顺序）
}

// 场景2：需要范围查询
void rangeQuery() {
    std::map<int, std::string> students;
    students[95] = "Alice";
    students[87] = "Bob";
    students[92] = "Charlie";
    students[78] = "David";
    
    // 查找分数在80-90之间的学生
    auto lower = students.lower_bound(80);
    auto upper = students.upper_bound(90);
    
    std::cout << "Students with scores 80-90:" << std::endl;
    for (auto it = lower; it != upper; ++it) {
        std::cout << it->second << ": " << it->first << std::endl;
    }
    // 输出：Bob: 87
}

// 场景3：需要找最小/最大元素
void minMaxElement() {
    std::map<int, std::string> m = {{3, "c"}, {1, "a"}, {5, "e"}, {2, "b"}};
    
    // 最小元素
    auto min = m.begin();  // O(log n)找到最左节点
    std::cout << "Min: " << min->first << std::endl;  // 1
    
    // 最大元素
    auto max = m.rbegin();  // O(log n)找到最右节点
    std::cout << "Max: " << max->first << std::endl;  // 5
}

// 场景4：需要稳定的性能
void stablePerformance() {
    std::map<int, int> m;
    
    // 无论什么数据，都是O(log n)
    // 不会因为哈希冲突而退化
    for (int i = 0; i < 1000000; ++i) {
        m[i] = i;  // 稳定的O(log n)
    }
}
```

#### 4.2 使用unordered_map的场景

```cpp
// 场景1：缓存/查找表
void cacheUsage() {
    std::unordered_map<std::string, int> cache;
    
    // 快速查找
    auto it = cache.find("key");
    if (it != cache.end()) {
        // 命中缓存
        return it->second;
    } else {
        // 计算并缓存
        int value = expensiveComputation();
        cache["key"] = value;
        return value;
    }
}

// 场景2：统计频率
void frequencyCount() {
    std::vector<std::string> words = {"apple", "banana", "apple", "cherry", "banana", "apple"};
    std::unordered_map<std::string, int> freq;
    
    for (const auto& word : words) {
        ++freq[word];  // O(1)平均时间
    }
    
    // 不需要有序，只需要快速统计
}

// 场景3：去重
void deduplication() {
    std::vector<int> nums = {1, 2, 3, 2, 4, 1, 5};
    std::unordered_map<int, bool> seen;
    std::vector<int> unique;
    
    for (int num : nums) {
        if (seen.find(num) == seen.end()) {
            unique.push_back(num);
            seen[num] = true;
        }
    }
    // unique: {1, 2, 3, 4, 5}
}

// 场景4：两数之和等算法题
void twoSum(const std::vector<int>& nums, int target) {
    std::unordered_map<int, int> indexMap;
    
    for (int i = 0; i < nums.size(); ++i) {
        int complement = target - nums[i];
        auto it = indexMap.find(complement);
        
        if (it != indexMap.end()) {
            std::cout << "Found: " << it->second << ", " << i << std::endl;
            return;
        }
        
        indexMap[nums[i]] = i;
    }
    // 平均O(n)时间复杂度，使用map是O(n log n)
}
```

#### 4.3 选择决策树

```
需要有序遍历？
├─ 是 → 使用 map
└─ 否 → 需要范围查询？
    ├─ 是 → 使用 map
    └─ 否 → 数据量很大且性能关键？
        ├─ 是 → 使用 unordered_map（注意内存）
        └─ 否 → 优先使用 unordered_map（通常更快）
```

---

### 5. 自定义类型的使用

#### 5.1 map的自定义类型

```cpp
// 需要提供比较函数
struct Person {
    std::string name;
    int age;
    
    // 方法1：重载operator<
    bool operator<(const Person& other) const {
        if (name != other.name) {
            return name < other.name;
        }
        return age < other.age;
    }
};

void useMapWithCustomType() {
    std::map<Person, int> scores;
    
    Person p1{"Alice", 25};
    Person p2{"Bob", 30};
    
    scores[p1] = 95;
    scores[p2] = 87;
}

// 方法2：自定义比较函数
struct PersonCompare {
    bool operator()(const Person& a, const Person& b) const {
        return a.age < b.age;  // 按年龄排序
    }
};

void useMapWithCustomComparator() {
    std::map<Person, int, PersonCompare> scores;
    // 按年龄排序
}
```

#### 5.2 unordered_map的自定义类型

```cpp
// 需要提供哈希函数和相等比较
struct Person {
    std::string name;
    int age;
    
    // 相等比较
    bool operator==(const Person& other) const {
        return name == other.name && age == other.age;
    }
};

// 哈希函数
struct PersonHash {
    size_t operator()(const Person& p) const {
        // 组合name和age的哈希值
        size_t h1 = std::hash<std::string>()(p.name);
        size_t h2 = std::hash<int>()(p.age);
        return h1 ^ (h2 << 1);  // 简单的组合方式
    }
};

void useUnorderedMapWithCustomType() {
    std::unordered_map<Person, int, PersonHash> scores;
    
    Person p1{"Alice", 25};
    Person p2{"Bob", 30};
    
    scores[p1] = 95;
    scores[p2] = 87;
}
```

---

### 6. 常见面试延伸问题

#### 6.1 为什么map比unordered_map慢但还要用？

```cpp
// 原因1：有序性是必需的
std::map<int, std::string> sortedMap;
// 需要按键排序时，unordered_map无法满足

// 原因2：性能稳定性
// map：始终O(log n)
// unordered_map：平均O(1)，但最坏O(n)

// 原因3：范围查询
// map支持lower_bound、upper_bound
// unordered_map不支持

// 原因4：迭代器稳定性
// map的迭代器在插入、删除其他元素时保持有效
// unordered_map在rehash时所有迭代器失效
```

#### 6.2 如何优化unordered_map的性能？

```cpp
void optimizeUnorderedMap() {
    std::unordered_map<int, int> m;
    
    // 1. 预留空间（避免频繁rehash）
    m.reserve(10000);
    
    // 2. 调整最大负载因子
    m.max_load_factor(0.75);  // 默认1.0
    // 降低负载因子：减少冲突，增加内存
    // 提高负载因子：节省内存，增加冲突
    
    // 3. 提供好的哈希函数
    // 避免大量冲突
    
    // 4. 使用emplace而非insert
    m.emplace(1, 100);  // 直接构造，避免临时对象
    
    // 5. 批量操作时预先reserve
    std::vector<std::pair<int, int>> data(10000);
    m.reserve(data.size());
    for (const auto& [k, v] : data) {
        m[k] = v;
    }
}
```

#### 6.3 map和unordered_map的迭代器失效情况

```cpp
void iteratorInvalidation() {
    // map的迭代器失效
    std::map<int, int> m = {{1, 1}, {2, 2}, {3, 3}};
    auto it1 = m.find(1);
    auto it2 = m.find(2);
    
    m.erase(1);  // it1失效，it2仍有效
    m.insert({4, 4});  // 所有迭代器仍有效
    
    // unordered_map的迭代器失效
    std::unordered_map<int, int> um = {{1, 1}, {2, 2}, {3, 3}};
    auto uit1 = um.find(1);
    auto uit2 = um.find(2);
    
    um.erase(1);  // uit1失效，uit2仍有效
    um.insert({4, 4});  // 可能触发rehash，所有迭代器失效！
    
    // 避免迭代器失效
    um.reserve(100);  // 预留足够空间
    um.insert({4, 4});  // 不会rehash，迭代器有效
}
```

---

### 7. 性能优化技巧

#### 7.1 选择合适的容器

```cpp
// 小数据量：差异不明显，优先unordered_map（代码简洁）
std::unordered_map<int, int> smallMap;  // < 1000个元素

// 大数据量 + 频繁查找：unordered_map
std::unordered_map<int, int> cache(1000000);

// 需要有序 + 频繁范围查询：map
std::map<int, int> orderedData;

// 需要稳定性能 + 避免最坏情况：map
std::map<int, int> stablePerformance;
```

#### 7.2 避免不必要的拷贝

```cpp
void avoidCopy() {
    std::map<std::string, std::vector<int>> m;
    
    // 错误：多次拷贝
    std::vector<int> data = {1, 2, 3};
    m["key"] = data;  // 拷贝
    
    // 正确：使用移动语义
    m["key"] = std::move(data);  // 移动
    
    // 更好：使用emplace
    m.emplace("key", std::vector<int>{1, 2, 3});  // 原地构造
    
    // 访问时使用引用
    const auto& value = m["key"];  // 引用，不拷贝
}
```

---

### 8. 总结

**核心差异**：
- **map**：有序、稳定、支持范围查询，时间复杂度O(log n)
- **unordered_map**：无序、通常更快，时间复杂度平均O(1)

**选择原则**：
1. 需要有序或范围查询 → `map`
2. 纯查找性能最重要 → `unordered_map`
3. 性能要求不高 → 优先 `unordered_map`（通常更快）
4. 需要稳定性能 → `map`

**性能对比**（通常情况）：
- 插入：unordered_map 快 2-3倍
- 查找：unordered_map 快 3-5倍
- 遍历：map 更稳定（有序）
- 内存：unordered_map 通常用得更多

**实践建议**：
- 优先考虑业务需求（有序性）
- 性能关键路径用unordered_map
- 预留空间避免rehash
- 自定义类型提供好的哈希函数
- 使用性能分析工具验证选择

记住：没有绝对的"更好"，只有"更适合"的场景。


---

## 相关笔记
<!-- 自动生成 -->

暂无相关笔记

