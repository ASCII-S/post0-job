---
created: '2025-10-19'
last_reviewed: null
next_review: '2025-10-19'
review_count: 0
difficulty: medium
mastery_level: 0.0
tags:
- LeetcodeHot100
- LeetcodeHot100/删除链表的倒数第N个结点.md
related_outlines: []
---
# 删除链表的倒数第 N 个结点

## 题目描述

给你一个链表，删除链表的倒数第 `n` 个结点，并且返回链表的头结点。

**示例 1：**
```
输入：head = [1,2,3,4,5], n = 2
输出：[1,2,3,5]
```

**示例 2：**
```
输入：head = [1], n = 1
输出：[]
```

**示例 3：**
```
输入：head = [1,2], n = 1
输出：[1]
```

**提示：**
- 链表中结点的数目为 sz
- 1 <= sz <= 30
- 0 <= Node.val <= 100
- 1 <= n <= sz

**进阶：** 你能尝试使用一趟扫描实现吗？

## 思路讲解

这道题的关键是如何在一次遍历中找到倒数第 n 个节点。

### 方法一：计算链表长度

**核心思路：**
1. 第一次遍历：计算链表长度 L
2. 第二次遍历：找到第 L - n 个节点（倒数第 n 个节点的前一个）
3. 删除目标节点

**复杂度：**
- 时间复杂度：O(L)，需要两次遍历
- 空间复杂度：O(1)

### 方法二：双指针（快慢指针，最优解）

这是满足进阶要求的一次遍历解法。

**核心思路：**
使用两个指针，让它们之间保持 n 个节点的距离。当快指针到达链表末尾时，慢指针正好指向倒数第 n 个节点的前一个节点。

**关键步骤：**
1. 使用虚拟头节点（简化删除头节点的情况）
2. 快指针先走 n+1 步（为了让慢指针停在待删除节点的前一个）
3. 快慢指针同时移动，直到快指针到达末尾
4. 此时慢指针指向倒数第 n+1 个节点
5. 删除慢指针的下一个节点

**为什么快指针要先走 n+1 步？**
- 如果走 n 步，慢指针会停在倒数第 n 个节点上
- 但删除节点需要知道它的前一个节点
- 所以让快指针多走一步，慢指针就会停在目标节点的前一个位置

**虚拟头节点的作用：**
- 统一处理删除头节点和非头节点的情况
- 避免特殊判断

## 面试时的快速口述讲解

这道题要求删除链表的倒数第 n 个节点，使用双指针可以一次遍历完成。

**数据结构**：使用虚拟头节点和快慢双指针。

**实现方式**：
1. 创建虚拟头节点 dummy，指向 head
2. 初始化快慢指针都指向 dummy
3. 让快指针先走 n+1 步
4. 然后快慢指针同时移动，每次走一步
5. 当快指针到达 nullptr 时，慢指针指向倒数第 n+1 个节点
6. 删除慢指针的下一个节点：slow->next = slow->next->next
7. 返回 dummy->next

**为什么快指针走 n+1 步**：这样慢指针会停在待删除节点的前一个位置，方便删除操作。

**时间复杂度**：O(L)，只需要一次遍历，其中 L 是链表长度。

**空间复杂度**：O(1)，只使用了两个指针。

## 代码实现

### 方法一：双指针（一次遍历，推荐）

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
    ListNode* removeNthFromEnd(ListNode* head, int n) {
        // 创建虚拟头节点，简化删除头节点的情况
        ListNode* dummy = new ListNode(0, head);
        
        // 初始化快慢指针
        ListNode* fast = dummy;
        ListNode* slow = dummy;
        
        // 快指针先走 n+1 步
        // 为什么是 n+1？因为要让 slow 停在待删除节点的前一个位置
        for (int i = 0; i <= n; i++) {
            fast = fast->next;
        }
        
        // 快慢指针同时移动，直到 fast 到达末尾
        while (fast != nullptr) {
            fast = fast->next;
            slow = slow->next;
        }
        
        // 此时 slow 指向倒数第 n+1 个节点
        // 删除 slow 的下一个节点（倒数第 n 个节点）
        ListNode* toDelete = slow->next;
        slow->next = slow->next->next;
        delete toDelete;  // 释放内存
        
        // 返回真正的头节点
        ListNode* result = dummy->next;
        delete dummy;  // 释放虚拟头节点
        return result;
    }
};
```

**代码说明：**

1. **虚拟头节点**：
   - 创建 dummy 节点，指向 head
   - 避免特殊处理删除头节点的情况
   - dummy->next 始终指向链表的真正头节点

2. **快指针先走 n+1 步**：
   ```
   示例：[1,2,3,4,5], n=2
   删除倒数第2个节点（值为4）
   
   初始状态：
   dummy -> 1 -> 2 -> 3 -> 4 -> 5 -> null
   ↑
   fast, slow
   
   fast 先走 n+1=3 步：
   dummy -> 1 -> 2 -> 3 -> 4 -> 5 -> null
   ↑              ↑
   slow          fast
   
   然后同时移动：
   dummy -> 1 -> 2 -> 3 -> 4 -> 5 -> null
            ↑              ↑
           slow          fast
   
   继续移动：
   dummy -> 1 -> 2 -> 3 -> 4 -> 5 -> null
                 ↑              ↑
                slow          fast
   
   继续移动：
   dummy -> 1 -> 2 -> 3 -> 4 -> 5 -> null
                      ↑              ↑
                     slow       fast(null)
   
   此时 slow 指向节点3，slow->next 是节点4（待删除）
   ```

3. **删除节点**：
   - `slow->next = slow->next->next`
   - 将前一个节点的 next 指向下一个节点的 next
   - 可以选择释放被删除节点的内存

4. **返回结果**：
   - 返回 `dummy->next`（真正的头节点）
   - 可以选择释放 dummy 节点的内存

**图解过程（[1,2,3,4,5], n=2）：**
```
目标：删除倒数第2个节点（节点4）

1. 创建虚拟头节点
   dummy -> 1 -> 2 -> 3 -> 4 -> 5 -> null

2. fast 先走 n+1 = 3 步
   dummy -> 1 -> 2 -> 3 -> 4 -> 5 -> null
   ↑                   ↑
   slow               fast

3. 同时移动到 fast 为 null
   dummy -> 1 -> 2 -> 3 -> 4 -> 5 -> null
                        ↑              ↑
                       slow          fast

4. 删除 slow->next（节点4）
   dummy -> 1 -> 2 -> 3 -> 5 -> null
                        ↑
                       slow

5. 返回 dummy->next
   结果：1 -> 2 -> 3 -> 5
```

### 方法二：计算长度（两次遍历）

```cpp
class Solution {
public:
    ListNode* removeNthFromEnd(ListNode* head, int n) {
        // 创建虚拟头节点
        ListNode* dummy = new ListNode(0, head);
        
        // 第一次遍历：计算链表长度
        int length = 0;
        ListNode* curr = head;
        while (curr != nullptr) {
            length++;
            curr = curr->next;
        }
        
        // 第二次遍历：找到倒数第 n+1 个节点
        curr = dummy;
        for (int i = 0; i < length - n; i++) {
            curr = curr->next;
        }
        
        // 删除倒数第 n 个节点
        curr->next = curr->next->next;
        
        return dummy->next;
    }
};
```

**代码说明：**
1. 第一次遍历计算链表总长度 L
2. 倒数第 n 个节点就是正数第 L-n+1 个节点
3. 从 dummy 开始走 L-n 步，到达倒数第 n+1 个节点
4. 删除其下一个节点

**复杂度对比：**
- 方法一（双指针）：时间 O(L)，一次遍历，推荐
- 方法二（计算长度）：时间 O(L)，两次遍历，直观

**面试技巧：**
- 优先说双指针解法（一次遍历）
- 强调虚拟头节点的作用
- 说明为什么快指针要先走 n+1 步
- 可以画图辅助说明指针移动过程


---

## 相关笔记
<!-- 自动生成 -->

- [合并两个有序链表](notes/LeetcodeHot100/合并两个有序链表.md) - 相似度: 36% | 标签: LeetcodeHot100, LeetcodeHot100/合并两个有序链表.md
- [反转链表](notes/LeetcodeHot100/反转链表.md) - 相似度: 36% | 标签: LeetcodeHot100, LeetcodeHot100/反转链表.md
- [回文链表](notes/LeetcodeHot100/回文链表.md) - 相似度: 31% | 标签: LeetcodeHot100, LeetcodeHot100/回文链表.md
- [随机链表的复制](notes/LeetcodeHot100/随机链表的复制.md) - 相似度: 31% | 标签: LeetcodeHot100, LeetcodeHot100/随机链表的复制.md
- [环形链表](notes/LeetcodeHot100/环形链表.md) - 相似度: 31% | 标签: LeetcodeHot100, LeetcodeHot100/环形链表.md

