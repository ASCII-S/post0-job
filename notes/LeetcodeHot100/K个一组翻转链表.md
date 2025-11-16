---
created: '2025-10-19'
last_reviewed: null
next_review: '2025-10-19'
review_count: 0
difficulty: medium
mastery_level: 0.0
tags:
- LeetcodeHot100
- LeetcodeHot100/K个一组翻转链表.md
related_outlines: []
---
# K 个一组翻转链表

## 题目描述

给你链表的头节点 `head`，每 `k` 个节点一组进行翻转，请你返回修改后的链表。

`k` 是一个正整数，它的值小于或等于链表的长度。如果节点总数不是 `k` 的整数倍，那么请将最后剩余的节点保持原有顺序。

你不能只是单纯的改变节点内部的值，而是需要实际进行节点交换。

**示例 1：**
```
输入：head = [1,2,3,4,5], k = 2
输出：[2,1,4,3,5]
```

**示例 2：**
```
输入：head = [1,2,3,4,5], k = 3
输出：[3,2,1,4,5]
```

**提示：**
- 链表中的节点数目为 n
- 1 <= k <= n <= 5000
- 0 <= Node.val <= 1000

**进阶：** 你可以设计一个只用 O(1) 额外内存空间的算法解决此问题吗？

## 思路讲解

这道题是"反转链表"的进阶版本，需要分组进行反转。

### 核心思路

**算法步骤：**
1. 检查剩余节点数量是否足够 k 个
2. 如果足够，反转这 k 个节点
3. 将反转后的部分连接到前面已处理的部分
4. 继续处理下一组
5. 如果剩余节点不足 k 个，保持原序

**关键难点：**
- 需要记录每一组的前驱节点（用于连接）
- 反转 k 个节点后，需要正确连接前后部分
- 需要准确找到下一组的起始位置

**指针管理：**
- `prev`：当前组的前驱节点
- `start`：当前组的第一个节点
- `end`：当前组的最后一个节点
- `next`：下一组的第一个节点

### 方法一：迭代（推荐）

**详细步骤：**
1. 使用虚拟头节点简化操作
2. 检查是否还有 k 个节点可以反转
3. 找到这一组的最后一个节点
4. 保存下一组的起始位置
5. 反转当前这 k 个节点
6. 将反转后的部分连接到链表中
7. 更新指针，处理下一组

### 方法二：递归

递归地处理每一组，代码更简洁但空间复杂度较高。

## 面试时的快速口述讲解

这道题要求将链表每 k 个节点一组进行反转，不足 k 个的部分保持原序。

**数据结构**：使用虚拟头节点，需要记录每组的前驱、起始和结束位置。

**实现方式（迭代法）**：
1. 创建虚拟头节点，初始化 prev 指针
2. 循环处理每一组：
   - 找到当前组的结束位置（第 k 个节点）
   - 如果不足 k 个，结束循环
   - 保存下一组的起始位置
   - 反转当前这 k 个节点
   - 连接反转后的部分：prev -> 反转后的头 -> ... -> 反转后的尾 -> 下一组
   - 更新 prev 为当前组反转后的尾部
3. 返回 dummy->next

**时间复杂度**：O(n)，每个节点只会被访问常数次。

**空间复杂度**：O(1)，只使用了几个指针变量。

## 代码实现

### 方法一：迭代（推荐）

```cpp
/**
 * Definition for singly-linked list.
 * struct ListNode {
 *     int val;
 *     ListNode *next;
 *     ListNode() : val(0), next(nullptr) {}
 *     ListNode(int x) : val(x), next(nullptr) {}
 *     ListNode(int x, ListNode *next) : val(x), next(next) {}
 * };
 */
class Solution {
public:
    ListNode* reverseKGroup(ListNode* head, int k) {
        // 创建虚拟头节点
        ListNode* dummy = new ListNode(0, head);
        
        // prev 指向当前组的前一个节点
        ListNode* prev = dummy;
        ListNode* end = dummy;
        
        while (end->next != nullptr) {
            // 找到当前组的结束位置（第 k 个节点）
            for (int i = 0; i < k && end != nullptr; i++) {
                end = end->next;
            }
            
            // 如果不足 k 个节点，不需要反转，直接结束
            if (end == nullptr) {
                break;
            }
            
            // 记录当前组的起始节点和下一组的起始节点
            ListNode* start = prev->next;
            ListNode* next = end->next;
            
            // 断开当前组（准备反转）
            end->next = nullptr;
            
            // 反转当前这 k 个节点，并连接到 prev 后面
            prev->next = reverse(start);
            
            // 反转后，start 变成了这一组的最后一个节点
            // 将 start 连接到下一组
            start->next = next;
            
            // 更新 prev 和 end，准备处理下一组
            prev = start;
            end = start;
        }
        
        return dummy->next;
    }
    
private:
    // 反转链表的辅助函数
    ListNode* reverse(ListNode* head) {
        ListNode* prev = nullptr;
        ListNode* curr = head;
        
        while (curr != nullptr) {
            ListNode* next = curr->next;
            curr->next = prev;
            prev = curr;
            curr = next;
        }
        
        return prev;
    }
};
```

**代码说明：**

**图解过程（[1,2,3,4,5], k=2）：**
```
初始状态：
dummy -> 1 -> 2 -> 3 -> 4 -> 5
↑
prev, end

第一组 [1,2]：
1. 找到 end（节点2）
   dummy -> 1 -> 2 -> 3 -> 4 -> 5
   ↑        ↑    ↑
   prev   start end

2. 保存 next = 3，断开：end->next = null
   dummy -> 1 -> 2    3 -> 4 -> 5
                      ↑
                     next

3. 反转 [1,2]，得到 2 -> 1
   连接：prev->next = 2
   dummy -> 2 -> 1
   
4. 连接下一组：start->next = next
   dummy -> 2 -> 1 -> 3 -> 4 -> 5
                 ↑
              prev, end

第二组 [3,4]：
1. 找到 end（节点4）
   dummy -> 2 -> 1 -> 3 -> 4 -> 5
                      ↑    ↑
                    start end

2. 保存 next = 5，断开
   dummy -> 2 -> 1 -> 3 -> 4    5
                                ↑
                               next

3. 反转 [3,4]，得到 4 -> 3
   连接：prev->next = 4
   dummy -> 2 -> 1 -> 4 -> 3

4. 连接下一组：start->next = next
   dummy -> 2 -> 1 -> 4 -> 3 -> 5
                           ↑
                        prev, end

第三组：
1. 找 end，发现只有1个节点（不足k个）
2. 结束循环

最终结果：2 -> 1 -> 4 -> 3 -> 5
```

**关键步骤解析：**

1. **找到当前组的结束位置**：
   ```cpp
   for (int i = 0; i < k && end != nullptr; i++) {
       end = end->next;
   }
   ```
   - 循环 k 次，end 指向第 k 个节点
   - 如果提前遇到 nullptr，说明不足 k 个

2. **断开当前组**：
   ```cpp
   ListNode* next = end->next;
   end->next = nullptr;
   ```
   - 保存下一组的起始位置
   - 断开当前组，准备反转

3. **反转并连接**：
   ```cpp
   prev->next = reverse(start);
   start->next = next;
   ```
   - 反转后，原来的 start 变成了尾部
   - 需要将尾部连接到下一组

4. **更新指针**：
   ```cpp
   prev = start;
   end = start;
   ```
   - prev 移动到当前组的尾部（为下一组做准备）
   - end 也重置到相同位置

### 方法二：递归

```cpp
class Solution {
public:
    ListNode* reverseKGroup(ListNode* head, int k) {
        // 检查是否有 k 个节点
        ListNode* curr = head;
        int count = 0;
        while (curr != nullptr && count < k) {
            curr = curr->next;
            count++;
        }
        
        // 如果不足 k 个节点，不反转
        if (count < k) {
            return head;
        }
        
        // 反转前 k 个节点
        ListNode* prev = nullptr;
        curr = head;
        for (int i = 0; i < k; i++) {
            ListNode* next = curr->next;
            curr->next = prev;
            prev = curr;
            curr = next;
        }
        
        // 递归处理剩余部分，并连接
        // prev 是反转后的新头节点
        // head 是反转后的尾节点
        // curr 是下一组的头节点
        head->next = reverseKGroup(curr, k);
        
        return prev;
    }
};
```

**递归过程图解（[1,2,3,4,5], k=2）：**
```
reverseKGroup([1,2,3,4,5], 2)
  ├─ 反转前2个：[2,1]
  └─ 1->next = reverseKGroup([3,4,5], 2)
       ├─ 反转前2个：[4,3]
       └─ 3->next = reverseKGroup([5], 2)
            └─ 不足2个，返回 [5]
       
       返回：4 -> 3 -> 5
  
  返回：2 -> 1 -> 4 -> 3 -> 5
```

### 方法三：更清晰的迭代写法

```cpp
class Solution {
public:
    ListNode* reverseKGroup(ListNode* head, int k) {
        ListNode* dummy = new ListNode(0, head);
        ListNode* prevGroupEnd = dummy;
        
        while (true) {
            // 检查是否还有 k 个节点
            ListNode* kthNode = getKthNode(prevGroupEnd, k);
            if (kthNode == nullptr) {
                break;
            }
            
            // 保存下一组的起始位置
            ListNode* nextGroupStart = kthNode->next;
            
            // 反转当前组
            ListNode* prev = nextGroupStart;  // 反转后会连接到下一组
            ListNode* curr = prevGroupEnd->next;
            
            while (curr != nextGroupStart) {
                ListNode* next = curr->next;
                curr->next = prev;
                prev = curr;
                curr = next;
            }
            
            // 连接反转后的部分
            ListNode* originalGroupStart = prevGroupEnd->next;
            prevGroupEnd->next = kthNode;  // 连接到反转后的头
            prevGroupEnd = originalGroupStart;  // 更新为反转后的尾
        }
        
        return dummy->next;
    }
    
private:
    // 获取从 start 开始的第 k 个节点
    ListNode* getKthNode(ListNode* start, int k) {
        while (start != nullptr && k > 0) {
            start = start->next;
            k--;
        }
        return start;
    }
};
```

**复杂度分析：**
- 时间复杂度：O(n)，每个节点被访问常数次
- 空间复杂度：O(1)（迭代），O(n/k)（递归，调用栈深度）

**常见错误：**
1. 忘记处理不足 k 个节点的情况
2. 反转后连接错误，丢失节点
3. 指针更新顺序错误

**面试技巧：**
- 这是一道难度较高的题目，需要清晰的思路
- 建议画图辅助说明每一步的指针变化
- 强调边界情况的处理（不足 k 个节点）
- 可以先写出反转整个链表的代码，再扩展到 k 个一组
- 优先使用迭代法，空间复杂度更优


---

## 相关笔记
<!-- 自动生成 -->

- [反转链表](notes/LeetcodeHot100/反转链表.md) - 相似度: 31% | 标签: LeetcodeHot100, LeetcodeHot100/反转链表.md
- [两两交换链表中的节点](notes/LeetcodeHot100/两两交换链表中的节点.md) - 相似度: 31% | 标签: LeetcodeHot100, LeetcodeHot100/两两交换链表中的节点.md

