---
created: '2025-10-19'
last_reviewed: null
next_review: '2025-10-19'
review_count: 0
difficulty: medium
mastery_level: 0.0
tags:
- LeetcodeHot100
- LeetcodeHot100/合并K个升序链表.md
related_outlines: []
---
# 合并 K 个升序链表

## 题目描述

给你一个链表数组，每个链表都已经按升序排列。

请你将所有链表合并到一个升序链表中，返回合并后的链表。

**示例 1：**
```
输入：lists = [[1,4,5],[1,3,4],[2,6]]
输出：[1,1,2,3,4,4,5,6]
解释：链表数组如下：
[
  1->4->5,
  1->3->4,
  2->6
]
将它们合并到一个有序链表中得到。
1->1->2->3->4->4->5->6
```

**示例 2：**
```
输入：lists = []
输出：[]
```

**示例 3：**
```
输入：lists = [[]]
输出：[]
```

**提示：**
- k == lists.length
- 0 <= k <= 10^4
- 0 <= lists[i].length <= 500
- -10^4 <= lists[i][j] <= 10^4
- lists[i] 按**升序**排列
- lists[i].length 的总和不超过 10^4

## 思路讲解

这道题是"合并两个有序链表"的扩展版本，需要合并 k 个链表。

### 方法对比

**方法一：顺序合并**
- 依次合并第 1 和第 2 个，再合并结果和第 3 个，以此类推
- 时间复杂度：O(k²n)，每次合并的链表长度递增
- 不推荐

**方法二：分治合并（归并，推荐）**
- 类似归并排序，两两配对合并
- 时间复杂度：O(kn log k)
- 最优的非堆解法

**方法三：优先队列（最小堆）**
- 使用最小堆维护 k 个链表的当前最小节点
- 时间复杂度：O(kn log k)
- 代码简洁，容易理解

### 方法一：分治合并（推荐）

**核心思路：**
使用分治法，两两合并链表，类似归并排序的合并过程。

**算法步骤：**
1. 将 k 个链表配对，两两合并
2. 第一轮后得到 k/2 个链表
3. 继续配对合并，直到只剩一个链表

**优势：**
- 时间复杂度优于顺序合并
- 空间复杂度 O(log k)（递归调用栈）
- 思路清晰，易于实现

**图解过程（k=4）：**
```
第0轮：[L1, L2, L3, L4]

第1轮：两两合并
  merge(L1, L2) -> L12
  merge(L3, L4) -> L34
  结果：[L12, L34]

第2轮：继续两两合并
  merge(L12, L34) -> L1234
  结果：[L1234]

返回 L1234
```

### 方法二：优先队列（最小堆）

**核心思路：**
使用最小堆维护每个链表的当前节点，每次取出最小的节点。

**算法步骤：**
1. 将 k 个链表的头节点加入最小堆
2. 循环直到堆为空：
   - 取出堆顶节点（最小值）
   - 将该节点加入结果链表
   - 如果该节点有 next，将 next 加入堆
3. 返回结果链表

**优势：**
- 代码简洁直观
- 时间复杂度 O(kn log k)
- 空间复杂度 O(k)（堆的大小）

## 面试时的快速口述讲解

这道题要求合并 k 个升序链表，有两种高效的解法。

**方法一（分治合并）：**

**数据结构**：递归地两两合并链表。

**实现方式**：
1. 使用分治思想，将 k 个链表两两配对
2. 递归地合并每一对链表
3. 类似归并排序的合并过程
4. 最终得到一个完整的有序链表

**时间复杂度**：O(kn log k)，其中 k 是链表个数，n 是所有节点总数。分治有 log k 层，每层需要合并所有 n 个节点。

**空间复杂度**：O(log k)，递归调用栈的深度。

**方法二（优先队列）：**

**数据结构**：使用最小堆维护 k 个链表的当前最小节点。

**实现方式**：
1. 将所有链表的头节点加入最小堆
2. 每次从堆中取出最小节点，加入结果链表
3. 如果该节点有后继节点，将后继节点加入堆
4. 重复直到堆为空

**时间复杂度**：O(kn log k)，需要处理 n 个节点，每次堆操作是 O(log k)。

**空间复杂度**：O(k)，堆中最多有 k 个节点。

## 代码实现

### 方法一：分治合并（推荐）

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
    ListNode* mergeKLists(vector<ListNode*>& lists) {
        if (lists.empty()) {
            return nullptr;
        }
        
        // 分治合并
        return merge(lists, 0, lists.size() - 1);
    }
    
private:
    // 分治合并：合并 lists[left...right] 的所有链表
    ListNode* merge(vector<ListNode*>& lists, int left, int right) {
        // 递归终止条件
        if (left == right) {
            return lists[left];
        }
        
        if (left > right) {
            return nullptr;
        }
        
        // 分治：找中点
        int mid = left + (right - left) / 2;
        
        // 递归合并左右两部分
        ListNode* l1 = merge(lists, left, mid);
        ListNode* l2 = merge(lists, mid + 1, right);
        
        // 合并两个有序链表
        return mergeTwoLists(l1, l2);
    }
    
    // 合并两个有序链表
    ListNode* mergeTwoLists(ListNode* l1, ListNode* l2) {
        ListNode* dummy = new ListNode(0);
        ListNode* curr = dummy;
        
        while (l1 != nullptr && l2 != nullptr) {
            if (l1->val <= l2->val) {
                curr->next = l1;
                l1 = l1->next;
            } else {
                curr->next = l2;
                l2 = l2->next;
            }
            curr = curr->next;
        }
        
        curr->next = (l1 != nullptr) ? l1 : l2;
        
        return dummy->next;
    }
};
```

**代码说明：**

**分治过程图解：**
```
lists = [L1, L2, L3, L4]

merge(lists, 0, 3)
├─ merge(lists, 0, 1)
│  ├─ merge(lists, 0, 0) -> L1
│  ├─ merge(lists, 1, 1) -> L2
│  └─ mergeTwoLists(L1, L2) -> L12
└─ merge(lists, 2, 3)
   ├─ merge(lists, 2, 2) -> L3
   ├─ merge(lists, 3, 3) -> L4
   └─ mergeTwoLists(L3, L4) -> L34

mergeTwoLists(L12, L34) -> L1234
```

**时间复杂度分析：**
- 分治树的高度：log k
- 每一层合并的总节点数：n（所有节点）
- 总时间：O(n log k)

**空间复杂度：**
- 递归调用栈深度：O(log k)

### 方法二：优先队列（最小堆）

```cpp
class Solution {
public:
    ListNode* mergeKLists(vector<ListNode*>& lists) {
        // 定义最小堆的比较函数
        auto cmp = [](ListNode* a, ListNode* b) {
            return a->val > b->val;  // 最小堆
        };
        
        // 创建优先队列（最小堆）
        priority_queue<ListNode*, vector<ListNode*>, decltype(cmp)> pq(cmp);
        
        // 将所有链表的头节点加入堆
        for (ListNode* head : lists) {
            if (head != nullptr) {
                pq.push(head);
            }
        }
        
        // 创建虚拟头节点
        ListNode* dummy = new ListNode(0);
        ListNode* curr = dummy;
        
        // 从堆中依次取出最小节点
        while (!pq.empty()) {
            // 取出堆顶（最小节点）
            ListNode* node = pq.top();
            pq.pop();
            
            // 加入结果链表
            curr->next = node;
            curr = curr->next;
            
            // 如果该节点有后继，将后继加入堆
            if (node->next != nullptr) {
                pq.push(node->next);
            }
        }
        
        return dummy->next;
    }
};
```

**代码说明：**

**优先队列过程图解：**
```
lists = [1->4->5, 1->3->4, 2->6]

初始堆：[1(L1), 1(L2), 2(L3)]

第1步：取出 1(L1)，加入 4(L1)
  结果：1
  堆：[1(L2), 2(L3), 4(L1)]

第2步：取出 1(L2)，加入 3(L2)
  结果：1 -> 1
  堆：[2(L3), 3(L2), 4(L1)]

第3步：取出 2(L3)，加入 6(L3)
  结果：1 -> 1 -> 2
  堆：[3(L2), 4(L1), 6(L3)]

第4步：取出 3(L2)，加入 4(L2)
  结果：1 -> 1 -> 2 -> 3
  堆：[4(L1), 4(L2), 6(L3)]

第5步：取出 4(L1)，加入 5(L1)
  结果：1 -> 1 -> 2 -> 3 -> 4
  堆：[4(L2), 5(L1), 6(L3)]

第6步：取出 4(L2)，L2 结束
  结果：1 -> 1 -> 2 -> 3 -> 4 -> 4
  堆：[5(L1), 6(L3)]

第7步：取出 5(L1)，L1 结束
  结果：1 -> 1 -> 2 -> 3 -> 4 -> 4 -> 5
  堆：[6(L3)]

第8步：取出 6(L3)，L3 结束
  结果：1 -> 1 -> 2 -> 3 -> 4 -> 4 -> 5 -> 6
  堆：[]

完成！
```

**时间复杂度分析：**
- 总共有 n 个节点
- 每个节点入堆和出堆各一次
- 堆操作的时间复杂度：O(log k)
- 总时间：O(n log k)

**空间复杂度：**
- 堆中最多有 k 个节点：O(k)

### 方法三：迭代两两合并

```cpp
class Solution {
public:
    ListNode* mergeKLists(vector<ListNode*>& lists) {
        if (lists.empty()) {
            return nullptr;
        }
        
        // 迭代地两两合并
        while (lists.size() > 1) {
            vector<ListNode*> merged;
            
            // 两两配对合并
            for (int i = 0; i < lists.size(); i += 2) {
                if (i + 1 < lists.size()) {
                    // 合并 lists[i] 和 lists[i+1]
                    merged.push_back(mergeTwoLists(lists[i], lists[i + 1]));
                } else {
                    // 如果是奇数个，最后一个直接加入
                    merged.push_back(lists[i]);
                }
            }
            
            // 更新 lists 为合并后的结果
            lists = merged;
        }
        
        return lists[0];
    }
    
private:
    ListNode* mergeTwoLists(ListNode* l1, ListNode* l2) {
        ListNode* dummy = new ListNode(0);
        ListNode* curr = dummy;
        
        while (l1 && l2) {
            if (l1->val <= l2->val) {
                curr->next = l1;
                l1 = l1->next;
            } else {
                curr->next = l2;
                l2 = l2->next;
            }
            curr = curr->next;
        }
        
        curr->next = l1 ? l1 : l2;
        return dummy->next;
    }
};
```

**迭代过程图解（k=4）：**
```
初始：[L1, L2, L3, L4]

第1轮：
  merge(L1, L2) -> L12
  merge(L3, L4) -> L34
  结果：[L12, L34]

第2轮：
  merge(L12, L34) -> L1234
  结果：[L1234]

返回 L1234
```

**复杂度对比：**
- **分治合并（递归）**：时间 O(kn log k)，空间 O(log k)，代码清晰
- **优先队列**：时间 O(kn log k)，空间 O(k)，代码简洁
- **迭代合并**：时间 O(kn log k)，空间 O(k)（存储中间结果）

**面试技巧：**
- 优先推荐分治合并，思路清晰，复杂度最优
- 优先队列解法代码更简洁，也是不错的选择
- 强调时间复杂度的分析（为什么是 kn log k）
- 可以对比顺序合并的劣势（k²n）
- 这道题综合考察了归并思想和堆的应用


---

## 相关笔记
<!-- 自动生成 -->

暂无相关笔记

