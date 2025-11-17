# 列表的时间复杂度：append、insert、pop、index

## 面试标准答案（可背诵）

Python列表基于**动态数组**实现，不同操作的时间复杂度差异显著：**append()** 在末尾添加元素是 **O(1) 平摊时间**，因为预留了额外空间；**insert(i, x)** 在指定位置插入是 **O(n)**，需要移动后续所有元素；**pop()** 从末尾删除是 **O(1)**，但 **pop(i)** 从中间删除是 **O(n)**；**index(x)** 查找元素是 **O(n)**，因为需要线性遍历。实际应用中应**优先在列表末尾操作**，避免频繁在开头或中间插入删除，频繁查找时应使用字典或集合。

---

## 详细讲解

### 1. append() - O(1) 平摊时间复杂度

#### 1.1 基本用法

```python
lst = [1, 2, 3]
lst.append(4)  # O(1) 平摊
# 结果: [1, 2, 3, 4]
```

#### 1.2 时间复杂度分析

- **大多数情况**：O(1) - 直接在预留空间中添加元素
- **容量不足时**：O(n) - 需要扩容并复制所有元素
- **平摊复杂度**：O(1) - 扩容操作分摊到多次append中

#### 1.3 底层原理

Python列表在创建时会预留额外的内存空间。当容量不足时，会按照以下策略扩容：

```python
# CPython的扩容策略（简化版）
new_allocated = (current_size >> 3) + (current_size < 9 ? 3 : 6) + current_size
# 大约是原容量的 1.125 倍
```

**扩容示例：**
```python
import sys

lst = []
for i in range(10):
    lst.append(i)
    print(f"长度: {len(lst)}, 容量: {sys.getsizeof(lst)}")
# 可以观察到容量的跳跃式增长
```

#### 1.4 使用建议

- 构建列表时优先使用 `append()` 而不是 `insert()`
- 如果知道最终大小，可以预分配空间避免多次扩容

```python
# 预分配空间
lst = [None] * 1000000
for i in range(1000000):
    lst[i] = i  # O(1)，无需扩容
```

### 2. insert() - O(n)

#### 2.1 基本用法

```python
lst = [1, 2, 3, 4]
lst.insert(1, 99)  # 在索引1处插入99
# 结果: [1, 99, 2, 3, 4]
```

#### 2.2 时间复杂度分析

- `insert(0, x)`：**O(n)** - 最坏情况，需要移动所有元素
- `insert(i, x)`：**O(n-i)** - 需要移动索引i之后的所有元素
- `insert(len(lst), x)`：**O(1)** - 等同于append

#### 2.3 底层原理

插入操作需要：
1. 检查容量是否足够，不足则扩容（O(n)）
2. 将索引i及之后的元素向后移动一位（O(n-i)）
3. 在索引i处放置新元素（O(1)）

```python
# 模拟insert的底层行为
def simulate_insert(lst, index, value):
    # 假设容量足够
    lst.append(None)  # 增加一个位置
    # 从后往前移动元素
    for i in range(len(lst) - 1, index, -1):
        lst[i] = lst[i - 1]
    lst[index] = value
```

#### 2.4 性能对比

```python
import time

# 在开头插入 - 最慢
lst = list(range(100000))
start = time.time()
lst.insert(0, -1)
print(f"insert(0): {time.time() - start:.6f}s")

# 在中间插入
lst = list(range(100000))
start = time.time()
lst.insert(50000, -1)
print(f"insert(50000): {time.time() - start:.6f}s")

# 在末尾插入 - 最快
lst = list(range(100000))
start = time.time()
lst.insert(len(lst), -1)
print(f"insert(end): {time.time() - start:.6f}s")
```

#### 2.5 替代方案

如果需要频繁在开头插入，使用 `collections.deque`：

```python
from collections import deque

dq = deque([1, 2, 3])
dq.appendleft(0)  # O(1)
# 结果: deque([0, 1, 2, 3])
```

### 3. pop() - O(1) 或 O(n)

#### 3.1 基本用法

```python
lst = [1, 2, 3, 4, 5]

# 从末尾删除 - O(1)
last = lst.pop()  # 返回5，lst变为[1, 2, 3, 4]

# 从指定位置删除 - O(n)
first = lst.pop(0)  # 返回1，lst变为[2, 3, 4]
```

#### 3.2 时间复杂度分析

- `pop()` 或 `pop(-1)`：**O(1)** - 删除末尾元素
- `pop(i)`：**O(n-i)** - 删除索引i处的元素
- `pop(0)`：**O(n)** - 最坏情况，删除第一个元素

#### 3.3 底层原理

删除操作需要：
1. 保存要删除的元素（O(1)）
2. 将索引i之后的元素向前移动一位（O(n-i)）
3. 减少列表长度（O(1)）

```python
# 模拟pop的底层行为
def simulate_pop(lst, index=-1):
    if index < 0:
        index = len(lst) + index
    value = lst[index]
    # 将后续元素前移
    for i in range(index, len(lst) - 1):
        lst[i] = lst[i + 1]
    # 实际实现中会减少长度，这里简化
    return value
```

#### 3.4 性能对比

```python
import time

# 从末尾删除 - 最快
lst = list(range(100000))
start = time.time()
for _ in range(1000):
    if lst:
        lst.pop()
print(f"pop(): {time.time() - start:.6f}s")

# 从开头删除 - 最慢
lst = list(range(100000))
start = time.time()
for _ in range(1000):
    if lst:
        lst.pop(0)
print(f"pop(0): {time.time() - start:.6f}s")
```

#### 3.5 使用建议

```python
# 不好的做法：从前往后删除
lst = list(range(1000))
while lst:
    lst.pop(0)  # O(n) 每次

# 好的做法1：从后往前删除
lst = list(range(1000))
while lst:
    lst.pop()  # O(1) 每次

# 好的做法2：使用deque
from collections import deque
dq = deque(range(1000))
while dq:
    dq.popleft()  # O(1) 每次
```

### 4. index() - O(n)

#### 4.1 基本用法

```python
lst = [1, 2, 3, 4, 5]
idx = lst.index(3)  # 返回2
# lst.index(10)  # ValueError: 10 is not in list

# 指定搜索范围
idx = lst.index(3, 1, 4)  # 在索引1到4之间查找3
```

#### 4.2 时间复杂度分析

- **最好情况**：O(1) - 元素在第一个位置
- **最坏情况**：O(n) - 元素在最后或不存在
- **平均情况**：O(n/2) = O(n) - 平均需要遍历一半元素

#### 4.3 底层原理

`index()` 使用线性搜索：

```python
# 模拟index的底层实现
def simulate_index(lst, value, start=0, end=None):
    if end is None:
        end = len(lst)
    for i in range(start, end):
        if lst[i] == value:
            return i
    raise ValueError(f"{value} is not in list")
```

#### 4.4 性能对比

```python
import time

lst = list(range(100000))

# 查找第一个元素 - 最快
start = time.time()
for _ in range(1000):
    lst.index(0)
print(f"index(first): {time.time() - start:.6f}s")

# 查找中间元素
start = time.time()
for _ in range(1000):
    lst.index(50000)
print(f"index(middle): {time.time() - start:.6f}s")

# 查找最后元素 - 最慢
start = time.time()
for _ in range(1000):
    lst.index(99999)
print(f"index(last): {time.time() - start:.6f}s")
```

#### 4.5 替代方案

如果需要频繁查找，使用字典或集合：

```python
# 方案1：使用字典存储索引映射
lst = list(range(10000))
index_map = {val: idx for idx, val in enumerate(lst)}
idx = index_map[5000]  # O(1)

# 方案2：使用集合判断存在性
lst = list(range(10000))
lst_set = set(lst)
if 5000 in lst_set:  # O(1)
    idx = lst.index(5000)

# 方案3：对于有序列表，使用二分查找
import bisect
sorted_lst = list(range(10000))
idx = bisect.bisect_left(sorted_lst, 5000)  # O(log n)
if idx < len(sorted_lst) and sorted_lst[idx] == 5000:
    print(f"找到，索引为 {idx}")
```

### 5. 其他常见操作的时间复杂度

#### 5.1 访问和修改

| 操作         | 时间复杂度 | 说明             |
| ------------ | ---------- | ---------------- |
| `len(lst)`   | O(1)       | 长度存储在对象中 |
| `lst[i]`     | O(1)       | 直接通过索引访问 |
| `lst[i] = x` | O(1)       | 直接通过索引赋值 |

#### 5.2 查找和删除

| 操作            | 时间复杂度 | 说明             |
| --------------- | ---------- | ---------------- |
| `x in lst`      | O(n)       | 线性搜索         |
| `lst.count(x)`  | O(n)       | 需要遍历整个列表 |
| `lst.remove(x)` | O(n)       | 需要搜索+删除    |
| `del lst[i]`    | O(n)       | 等同于pop(i)     |

#### 5.3 列表操作

| 操作               | 时间复杂度 | 说明                |
| ------------------ | ---------- | ------------------- |
| `lst.extend(lst2)` | O(k)       | k为lst2的长度       |
| `lst + lst2`       | O(n+m)     | 创建新列表          |
| `lst * k`          | O(nk)      | 创建新列表，重复k次 |
| `lst.copy()`       | O(n)       | 浅拷贝              |
| `lst[:]`           | O(n)       | 切片复制            |

#### 5.4 排序和反转

| 操作            | 时间复杂度 | 说明                  |
| --------------- | ---------- | --------------------- |
| `lst.sort()`    | O(n log n) | Timsort算法，原地排序 |
| `sorted(lst)`   | O(n log n) | 返回新列表            |
| `lst.reverse()` | O(n)       | 原地反转              |
| `reversed(lst)` | O(1)       | 返回迭代器            |

### 6. 性能优化实践

#### 6.1 构建大列表的最佳实践

```python
import time

n = 1000000

# 方法1：使用append（较慢）
start = time.time()
lst1 = []
for i in range(n):
    lst1.append(i)
print(f"append: {time.time() - start:.4f}s")

# 方法2：使用列表推导式（更快）
start = time.time()
lst2 = [i for i in range(n)]
print(f"list comprehension: {time.time() - start:.4f}s")

# 方法3：使用list()转换（最快）
start = time.time()
lst3 = list(range(n))
print(f"list(range()): {time.time() - start:.4f}s")

# 方法4：预分配空间（适用于已知大小）
start = time.time()
lst4 = [0] * n
for i in range(n):
    lst4[i] = i
print(f"pre-allocate: {time.time() - start:.4f}s")
```

#### 6.2 批量操作优于单个操作

```python
# 不好的做法
lst = []
for x in range(1000):
    lst.append(x)  # 1000次函数调用

# 好的做法
lst = []
lst.extend(range(1000))  # 1次函数调用

# 更好的做法
lst = list(range(1000))  # 最快
```

#### 6.3 删除多个元素的策略

```python
# 场景：删除列表中所有偶数

# 不好的做法：边遍历边删除
lst = list(range(1000))
for i in range(len(lst) - 1, -1, -1):
    if lst[i] % 2 == 0:
        lst.pop(i)  # O(n) 每次

# 好的做法：列表推导式
lst = list(range(1000))
lst = [x for x in lst if x % 2 != 0]  # O(n) 一次

# 或者使用filter
lst = list(range(1000))
lst = list(filter(lambda x: x % 2 != 0, lst))
```

#### 6.4 频繁查找的优化

```python
# 场景：需要多次查找元素

# 不好的做法
lst = list(range(10000))
targets = [100, 200, 300, 400, 500]
for target in targets:
    if target in lst:  # O(n) 每次
        idx = lst.index(target)  # O(n) 每次
        print(f"{target} at index {idx}")

# 好的做法：使用字典
lst = list(range(10000))
index_map = {val: idx for idx, val in enumerate(lst)}  # O(n) 一次
targets = [100, 200, 300, 400, 500]
for target in targets:
    if target in index_map:  # O(1) 每次
        idx = index_map[target]  # O(1) 每次
        print(f"{target} at index {idx}")
```

### 7. 选择合适的数据结构

#### 7.1 需要频繁在两端操作 - 使用 deque

```python
from collections import deque

# 列表在开头操作很慢
lst = [1, 2, 3]
lst.insert(0, 0)  # O(n)
lst.pop(0)        # O(n)

# deque在两端操作都很快
dq = deque([1, 2, 3])
dq.appendleft(0)  # O(1)
dq.popleft()      # O(1)
dq.append(4)      # O(1)
dq.pop()          # O(1)
```

#### 7.2 需要频繁查找 - 使用 set 或 dict

```python
# 列表查找慢
lst = list(range(10000))
5000 in lst  # O(n)

# 集合查找快
s = set(range(10000))
5000 in s  # O(1) 平均

# 字典查找快且可以存储额外信息
d = {i: f"value_{i}" for i in range(10000)}
5000 in d  # O(1) 平均
value = d[5000]  # O(1)
```

#### 7.3 需要保持有序且频繁插入 - 使用 bisect

```python
import bisect

# 维护有序列表
sorted_lst = []
for value in [5, 2, 8, 1, 9, 3]:
    bisect.insort(sorted_lst, value)  # O(n)，但保持有序
print(sorted_lst)  # [1, 2, 3, 5, 8, 9]

# 二分查找
idx = bisect.bisect_left(sorted_lst, 5)  # O(log n)
```

### 8. 常见误区和陷阱

#### 8.1 误区1：认为所有操作都是O(1)

```python
# 错误认知：列表所有操作都很快
lst = list(range(100000))
lst.insert(0, -1)  # 实际是O(n)，很慢！
```

#### 8.2 误区2：在循环中修改列表

```python
# 危险：边遍历边删除
lst = [1, 2, 3, 4, 5]
for i in range(len(lst)):
    if lst[i] % 2 == 0:
        lst.pop(i)  # 会导致索引错误！

# 正确做法
lst = [1, 2, 3, 4, 5]
lst = [x for x in lst if x % 2 != 0]
```

#### 8.3 误区3：忽略扩容成本

```python
# 虽然append是O(1)平摊，但单次扩容可能很慢
import time

lst = []
for i in range(1000000):
    start = time.time()
    lst.append(i)
    elapsed = time.time() - start
    if elapsed > 0.0001:  # 检测到扩容
        print(f"扩容发生在索引 {i}，耗时 {elapsed:.6f}s")
```

---

## 总结

### 核心要点

1. **append()**: O(1) 平摊 - 在末尾添加元素的最佳选择，偶尔需要扩容
2. **insert()**: O(n) - 需要移动后续元素，避免频繁使用，特别是在开头插入
3. **pop()**: 末尾O(1)，其他位置O(n) - 优先从末尾删除，从开头删除需要移动所有元素
4. **index()**: O(n) - 线性搜索，频繁查找时考虑使用字典或集合

### 最佳实践

- **优先在列表末尾操作**（append、pop）以获得O(1)性能
- **避免在开头或中间频繁插入/删除**，考虑使用deque
- **频繁查找时使用字典或集合**，而不是列表
- **预分配空间**可以避免多次扩容
- **使用列表推导式**通常比循环append更快
- **批量操作**优于多次单个操作

### 数据结构选择指南

| 使用场景          | 推荐数据结构  | 原因             |
| ----------------- | ------------- | ---------------- |
| 只在末尾添加/删除 | list          | append/pop是O(1) |
| 需要在两端操作    | deque         | 两端操作都是O(1) |
| 频繁查找元素      | set/dict      | 查找是O(1)       |
| 需要保持有序      | list + bisect | 二分查找O(log n) |
| 随机访问          | list          | 索引访问O(1)     |

---

## 参考文献

1. **Python官方文档 - Time Complexity**
   - https://wiki.python.org/moin/TimeComplexity
   - Python各种数据结构操作的时间复杂度官方说明

2. **CPython源码 - listobject.c**
   - https://github.com/python/cpython/blob/main/Objects/listobject.c
   - Python列表的底层C实现，包含扩容策略

3. **Python官方文档 - Data Structures**
   - https://docs.python.org/3/tutorial/datastructures.html
   - 列表和其他数据结构的使用指南

4. **collections模块文档**
   - https://docs.python.org/3/library/collections.html
   - deque等高性能数据结构的说明

5. **bisect模块文档**
   - https://docs.python.org/3/library/bisect.html
   - 有序列表的二分查找和插入
