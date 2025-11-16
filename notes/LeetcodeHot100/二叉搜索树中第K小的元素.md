---
created: '2025-10-19'
last_reviewed: null
next_review: '2025-10-19'
review_count: 0
difficulty: medium
mastery_level: 0.0
tags:
- LeetcodeHot100
- LeetcodeHot100/二叉搜索树中第K小的元素.md
related_outlines: []
---
# 二叉搜索树中第 K 小的元素

## 题目描述

给定一个二叉搜索树的根节点 `root`，和一个整数 `k`，请你设计一个算法查找其中第 `k` 小的元素（从 1 开始计数）。

**示例 1：**
```
输入：root = [3,1,4,null,2], k = 1
输出：1
```

**示例 2：**
```
输入：root = [5,3,6,2,4,null,null,1], k = 3
输出：3
```

**提示：**
- 树中的节点数为 n
- 1 <= k <= n <= 10^4
- 0 <= Node.val <= 10^4

**进阶：** 如果二叉搜索树经常被修改（插入/删除操作）并且你需要频繁地查找第 k 小的值，你将如何优化算法？

## 思路讲解

这道题利用了 BST 的核心性质：中序遍历是升序的。

### 核心思想

BST 的中序遍历结果是**升序序列**，因此第 k 小的元素就是中序遍历的第 k 个节点。

### 方法一：中序遍历（递归）

1. 进行中序遍历（左-根-右）
2. 使用计数器记录已访问的节点数
3. 当计数器等于 k 时，当前节点就是答案

### 方法二：中序遍历（迭代）

使用栈模拟递归的中序遍历：
1. 一直往左走，将路径上的节点压入栈
2. 弹出栈顶节点，计数器加 1
3. 如果计数器等于 k，返回当前节点
4. 转向右子树，重复上述过程

### 进阶优化

如果需要频繁查询：
1. **方法一**：在每个节点存储其左子树的节点数
   - 查询时可以快速判断第 k 小的元素在哪个子树
   - 时间复杂度：O(h)，h 是树的高度
   
2. **方法二**：使用平衡二叉搜索树（如 AVL 树、红黑树）
   - 在节点中维护子树大小信息
   - 支持高效的插入、删除和查询操作

## 面试时的快速口述讲解

这道题要求找到二叉搜索树中第 k 小的元素。

**数据结构**：
- 递归法：递归调用栈
- 迭代法：显式栈

**实现方式**：
利用 BST 的中序遍历是升序序列的性质，进行中序遍历，用计数器记录已访问的节点数，当计数到 k 时，当前节点就是答案。可以用递归或迭代实现中序遍历。

**时间复杂度**：O(H + k)，H 是树的高度。最坏情况下 O(n)，当 k = n 时需要遍历所有节点。

**空间复杂度**：O(H)，递归调用栈或显式栈的深度。

## 代码实现

### 方法一：递归

```cpp
/**
 * Definition for a binary tree node.
 * struct TreeNode {
 *     int val;
 *     TreeNode *left;
 *     TreeNode *right;
 *     TreeNode() : val(0), left(nullptr), right(nullptr) {}
 *     TreeNode(int x) : val(x), left(nullptr), right(nullptr) {}
 *     TreeNode(int x, TreeNode *left, TreeNode *right) : val(x), left(left), right(right) {}
 * };
 */
class Solution {
public:
    int kthSmallest(TreeNode* root, int k) {
        int count = 0;
        int result = 0;
        inorder(root, k, count, result);
        return result;
    }
    
private:
    void inorder(TreeNode* node, int k, int& count, int& result) {
        if (node == nullptr) {
            return;
        }
        
        // 中序遍历：左 -> 根 -> 右
        
        // 遍历左子树
        inorder(node->left, k, count, result);
        
        // 访问当前节点
        count++;
        if (count == k) {
            result = node->val;
            return;  // 找到答案，提前返回
        }
        
        // 遍历右子树
        inorder(node->right, k, count, result);
    }
};
```

### 方法二：迭代（使用栈）

```cpp
class Solution {
public:
    int kthSmallest(TreeNode* root, int k) {
        stack<TreeNode*> stk;
        TreeNode* curr = root;
        int count = 0;
        
        while (curr != nullptr || !stk.empty()) {
            // 一直往左走
            while (curr != nullptr) {
                stk.push(curr);
                curr = curr->left;
            }
            
            // 弹出栈顶节点（此时左子树已处理完）
            curr = stk.top();
            stk.pop();
            
            // 访问当前节点
            count++;
            if (count == k) {
                return curr->val;
            }
            
            // 转向右子树
            curr = curr->right;
        }
        
        return -1;  // 不会执行到这里
    }
};
```

**代码说明：**

**递归法：**
1. 使用 `count` 记录已访问的节点数
2. 使用 `result` 存储第 k 小的元素值
3. 中序遍历顺序：左 -> 根 -> 右
4. 访问当前节点时：
   - `count++`
   - 如果 `count == k`，记录结果并提前返回
5. 找到答案后可以提前返回，不需要遍历剩余节点

**迭代法：**
1. 使用栈 `stk` 和当前指针 `curr`
2. 模拟中序遍历：
   - 一直向左走，将路径上的节点压入栈
   - 弹出栈顶节点，访问它（计数）
   - 如果 `count == k`，返回当前节点值
   - 转向右子树
3. 这种方法可以在找到答案后立即返回，效率更高

**执行过程示例**（以 [5,3,6,2,4,null,null,1], k = 3 为例）：

```
树的结构：
      5
     / \
    3   6
   / \
  2   4
 /
1

中序遍历序列：1, 2, 3, 4, 5, 6
第 3 小的元素是 3

迭代过程：
1. 一路向左：stk = [5, 3, 2, 1], curr = null
2. 弹出 1，count = 1，curr = null（1 无右子树）
3. 弹出 2，count = 2，curr = null（2 无右子树）
4. 弹出 3，count = 3 == k，返回 3
```

**优化技巧：**
- 两种方法都可以在找到第 k 个元素后立即返回
- 不需要遍历整棵树，只需要遍历到第 k 个节点
- 平均情况下比完整遍历要快

**进阶问题的解答：**

如果 BST 经常被修改，可以在每个节点中维护一个 `leftCount` 字段，表示其左子树的节点数：

```cpp
struct TreeNode {
    int val;
    int leftCount;  // 左子树的节点数
    TreeNode* left;
    TreeNode* right;
};

int kthSmallest(TreeNode* root, int k) {
    if (root == nullptr) return -1;
    
    int leftCount = (root->left != nullptr) ? root->leftCount : 0;
    
    if (k <= leftCount) {
        // 第 k 小在左子树
        return kthSmallest(root->left, k);
    } else if (k == leftCount + 1) {
        // 当前节点就是第 k 小
        return root->val;
    } else {
        // 第 k 小在右子树
        return kthSmallest(root->right, k - leftCount - 1);
    }
}
```

这样查询的时间复杂度降为 O(h)，但需要在插入/删除时维护 `leftCount`。


---

## 相关笔记
<!-- 自动生成 -->

- [验证二叉搜索树](notes/LeetcodeHot100/验证二叉搜索树.md) - 相似度: 36% | 标签: LeetcodeHot100, LeetcodeHot100/验证二叉搜索树.md
- [二叉树的中序遍历](notes/LeetcodeHot100/二叉树的中序遍历.md) - 相似度: 31% | 标签: LeetcodeHot100, LeetcodeHot100/二叉树的中序遍历.md

