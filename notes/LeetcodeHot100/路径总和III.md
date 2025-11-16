---
created: '2025-10-19'
last_reviewed: null
next_review: '2025-10-19'
review_count: 0
difficulty: medium
mastery_level: 0.0
tags:
- LeetcodeHot100
- LeetcodeHot100/路径总和III.md
related_outlines: []
---
# 路径总和 III

## 题目描述

给定一个二叉树的根节点 `root`，和一个整数 `targetSum`，求该二叉树里节点值之和等于 `targetSum` 的**路径**的数目。

**路径**不需要从根节点开始，也不需要在叶子节点结束，但是路径方向必须是向下的（只能从父节点到子节点）。

**示例 1：**
```
输入：root = [10,5,-3,3,2,null,11,3,-2,null,1], targetSum = 8
输出：3
解释：和等于 8 的路径有 3 条：
1. 5 -> 3
2. 5 -> 2 -> 1
3. -3 -> 11
```

**示例 2：**
```
输入：root = [5,4,8,11,null,13,4,7,2,null,null,5,1], targetSum = 22
输出：3
```

**提示：**
- 二叉树的节点个数的范围是 [0, 1000]
- -10^9 <= Node.val <= 10^9
- -1000 <= targetSum <= 1000

## 思路讲解

这道题的难点在于路径可以从任意节点开始，到任意节点结束。

### 方法一：双重递归（暴力法）

最直观的思路：
1. 遍历每个节点，将其作为路径的起点
2. 对于每个起点，使用 DFS 计算从该节点开始的所有满足条件的路径数

**时间复杂度**：O(n²)，对于每个节点，都要 DFS 遍历其子树。

### 方法二：前缀和 + 哈希表

更优化的方法，类似于数组的"和为 K 的子数组"问题：
1. 使用前缀和思想：记录从根节点到当前节点的路径和
2. 使用哈希表存储前缀和的出现次数
3. 对于当前节点，如果 `prefixSum - targetSum` 在哈希表中存在，说明存在满足条件的路径

### 核心思想（前缀和）

如果从根节点到节点 A 的路径和为 `sumA`，到节点 B 的路径和为 `sumB`（B 在 A 的下方），那么从 A 到 B 的路径和为 `sumB - sumA`。

因此，如果 `sumB - sumA == targetSum`，即 `sumA == sumB - targetSum`，就找到了一条路径。

### 关键点

- **前缀和**：从根节点到当前节点的路径和
- **哈希表**：存储前缀和及其出现次数
- **回溯**：离开节点时，需要将该节点从哈希表中移除（恢复状态）
- **初始化**：`prefixSumCount[0] = 1`，表示前缀和为 0 的路径有 1 条（空路径）

## 面试时的快速口述讲解

这道题要求找出所有路径和等于目标值的路径数量，路径可以从任意节点开始和结束。

**数据结构**：
- 方法一：双重递归
- 方法二：哈希表存储前缀和

**实现方式**：
- **方法一（双重递归）**：对每个节点，以它为起点 DFS 计算满足条件的路径数。外层递归遍历所有节点，内层递归计算从当前节点开始的路径。时间复杂度 O(n²)
- **方法二（前缀和+哈希表）**：使用前缀和思想，维护从根到当前节点的路径和。如果当前前缀和减去目标值在哈希表中存在，说明存在满足条件的路径。类似于"和为 K 的子数组"问题。时间复杂度 O(n)

**时间复杂度**：
- 方法一：O(n²)
- 方法二：O(n)

**空间复杂度**：
- 方法一：O(h)，递归调用栈
- 方法二：O(n)，哈希表存储前缀和

## 代码实现

### 方法一：双重递归

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
    int pathSum(TreeNode* root, int targetSum) {
        if (root == nullptr) {
            return 0;
        }
        
        // 以当前节点为起点的路径数 +
        // 以左子树中某节点为起点的路径数 +
        // 以右子树中某节点为起点的路径数
        return countPaths(root, targetSum) +
               pathSum(root->left, targetSum) +
               pathSum(root->right, targetSum);
    }
    
private:
    // 计算从 node 开始的满足条件的路径数
    int countPaths(TreeNode* node, long long targetSum) {
        if (node == nullptr) {
            return 0;
        }
        
        int count = 0;
        
        // 如果当前节点的值等于目标和，找到一条路径
        if (node->val == targetSum) {
            count++;
        }
        
        // 继续向下寻找（目标和减去当前节点值）
        count += countPaths(node->left, targetSum - node->val);
        count += countPaths(node->right, targetSum - node->val);
        
        return count;
    }
};
```

### 方法二：前缀和 + 哈希表（最优解）

```cpp
class Solution {
public:
    int pathSum(TreeNode* root, int targetSum) {
        unordered_map<long long, int> prefixSumCount;
        // 初始化：前缀和为 0 的路径有 1 条（空路径）
        prefixSumCount[0] = 1;
        return dfs(root, 0, targetSum, prefixSumCount);
    }
    
private:
    int dfs(TreeNode* node, long long currentSum, int targetSum,
            unordered_map<long long, int>& prefixSumCount) {
        if (node == nullptr) {
            return 0;
        }
        
        // 更新当前路径和
        currentSum += node->val;
        
        // 查找是否存在前缀和为 currentSum - targetSum 的路径
        // 如果存在，说明从那个节点到当前节点的路径和为 targetSum
        int count = prefixSumCount[currentSum - targetSum];
        
        // 将当前前缀和加入哈希表
        prefixSumCount[currentSum]++;
        
        // 递归处理左右子树
        count += dfs(node->left, currentSum, targetSum, prefixSumCount);
        count += dfs(node->right, currentSum, targetSum, prefixSumCount);
        
        // 回溯：移除当前节点的前缀和（恢复状态）
        prefixSumCount[currentSum]--;
        
        return count;
    }
};
```

**代码说明：**

**方法一（双重递归）：**
1. **外层递归 `pathSum`**：
   - 遍历树的每个节点
   - 对每个节点，计算以它为起点的满足条件的路径数
   - 递归处理左右子树

2. **内层递归 `countPaths`**：
   - 从指定节点开始，向下 DFS 寻找路径
   - 如果当前节点值等于目标和，找到一条路径
   - 递归到子节点，目标和减去当前节点值

3. 使用 `long long` 避免整数溢出

**方法二（前缀和+哈希表）：**

这是最优解，核心思想是前缀和：

1. **prefixSumCount**：哈希表，存储前缀和及其出现次数
   - Key: 前缀和
   - Value: 该前缀和出现的次数

2. **初始化**：`prefixSumCount[0] = 1`
   - 表示前缀和为 0 的路径有 1 条（从根节点到根节点之前，空路径）
   - 这样可以正确处理从根节点开始的路径

3. **DFS 过程**：
   - 更新当前路径和：`currentSum += node->val`
   - 查找是否存在前缀和为 `currentSum - targetSum` 的路径
     - 如果存在，说明从那些节点到当前节点的路径和为 `targetSum`
   - 将当前前缀和加入哈希表
   - 递归处理左右子树
   - **回溯**：移除当前节点的前缀和（重要！）

4. **为什么需要回溯？**
   - 因为路径必须是向下的，不能跨越不同的子树
   - 离开当前节点时，需要将其从哈希表中移除
   - 这样保证哈希表中只包含从根节点到当前节点路径上的前缀和

**执行过程示例**（简化版）：

```
树：  10
     /  \
    5   -3
   / \    \
  3   2   11
 / \   \
3  -2   1

targetSum = 8

DFS 遍历（部分）：
节点 10: currentSum=10, 查找10-8=2（不存在），count=0
         map={0:1, 10:1}
节点 5:  currentSum=15, 查找15-8=7（不存在），count=0
         map={0:1, 10:1, 15:1}
节点 3:  currentSum=18, 查找18-8=10（存在！count=1）
         这意味着路径 5->3 的和为8
         map={0:1, 10:1, 15:1, 18:1}
...
```

**两种方法的对比：**
- **方法一**：简单直观，但时间复杂度高 O(n²)
- **方法二**：使用前缀和优化，时间复杂度 O(n)，是最优解

面试中推荐使用**方法二**，展示了对前缀和技巧的掌握，也体现了从暴力解到优化解的思考过程。

**关键技巧：**
- 将树的路径问题转化为数组的子数组和问题
- 使用前缀和 + 哈希表优化
- 注意回溯恢复状态


---

## 相关笔记
<!-- 自动生成 -->

暂无相关笔记

