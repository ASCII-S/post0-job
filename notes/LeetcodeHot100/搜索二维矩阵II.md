---
created: '2025-10-19'
last_reviewed: null
next_review: '2025-10-19'
review_count: 0
difficulty: medium
mastery_level: 0.0
tags:
- LeetcodeHot100
- LeetcodeHot100/搜索二维矩阵II.md
related_outlines: []
---
# 搜索二维矩阵 II

## 题目描述

编写一个高效的算法来搜索 `m x n` 矩阵 `matrix` 中的一个目标值 `target`。该矩阵具有以下特性：

- 每行的元素从左到右升序排列。
- 每列的元素从上到下升序排列。

**示例 1：**
```
输入：matrix = [[1,4,7,11,15],
               [2,5,8,12,19],
               [3,6,9,16,22],
               [10,13,14,17,24],
               [18,21,23,26,30]], target = 5
输出：true
```

**示例 2：**
```
输入：matrix = [[1,4,7,11,15],
               [2,5,8,12,19],
               [3,6,9,16,22],
               [10,13,14,17,24],
               [18,21,23,26,30]], target = 20
输出：false
```

**提示：**
- m == matrix.length
- n == matrix[i].length
- 1 <= n, m <= 300
- -10^9 <= matrix[i][j] <= 10^9
- 每行的所有元素从左到右升序排列
- 每列的所有元素从上到下升序排列
- -10^9 <= target <= 10^9

## 思路讲解

这道题的关键是利用矩阵的特性：行列都有序。

### 方法一：从右上角（或左下角）开始搜索

**核心思想：**
从右上角开始，根据比较结果决定向左还是向下移动。

**为什么选择右上角？**
- 右上角元素是该行最大、该列最小
- 如果 `target < 当前值`，向左移动（减小）
- 如果 `target > 当前值`，向下移动（增大）
- 这样可以排除整行或整列

**算法步骤：**
1. 从右上角开始（或左下角）
2. 比较当前值与 target：
   - 相等：返回 true
   - 大于 target：向左移动（col--）
   - 小于 target：向下移动（row++）
3. 超出边界：返回 false

**示例演示：**
```
matrix = [[1,  4,  7,  11, 15],
          [2,  5,  8,  12, 19],
          [3,  6,  9,  16, 22],
          [10, 13, 14, 17, 24],
          [18, 21, 23, 26, 30]]
target = 5

从右上角 (0, 4) 开始：
(0,4): 15 > 5, 向左 → (0,3)
(0,3): 11 > 5, 向左 → (0,2)
(0,2): 7 > 5, 向左 → (0,1)
(0,1): 4 < 5, 向下 → (1,1)
(1,1): 5 == 5, 找到！返回 true
```

**为什么从左上角或右下角不行？**
- 左上角：既是行最小又是列最小，无法确定方向
- 右下角：既是行最大又是列最大，无法确定方向
- 只有右上角和左下角才有明确的搜索方向

### 方法二：二分查找

对每一行或每一列进行二分查找，时间复杂度 O(m log n) 或 O(n log m)，不如方法一优秀。

### 方法三：分治

将矩阵分为四个象限递归搜索，但实现复杂且常数较大。

## 面试时的快速口述讲解

这道题在行列都有序的矩阵中搜索目标值。

**数据结构**：不需要额外数据结构，只用两个指针表示当前位置。

**实现方式**：
1. 从右上角（或左下角）开始
2. 根据当前值与目标值的比较决定移动方向：
   - 如果当前值大于目标，向左移动（减小）
   - 如果当前值小于目标，向下移动（增大）
   - 如果相等，返回 true
3. 超出边界返回 false

**关键点**：
- 选择右上角或左下角作为起点
- 每次比较都能排除一行或一列
- 类似二叉搜索树的搜索过程

**时间复杂度**：O(m + n)，最多走 m+n 步。

**空间复杂度**：O(1)，只使用常数额外空间。

## 代码实现

### 方法一：从右上角开始（推荐）

```cpp
class Solution {
public:
    bool searchMatrix(vector<vector<int>>& matrix, int target) {
        if (matrix.empty() || matrix[0].empty()) return false;
        
        int m = matrix.size();
        int n = matrix[0].size();
        
        // 从右上角开始
        int row = 0;
        int col = n - 1;
        
        while (row < m && col >= 0) {
            if (matrix[row][col] == target) {
                return true;
            } else if (matrix[row][col] > target) {
                // 当前值太大，向左移动
                col--;
            } else {
                // 当前值太小，向下移动
                row++;
            }
        }
        
        return false;
    }
};
```

**代码说明：**
1. 从右上角 `(0, n-1)` 开始
2. 循环条件：`row < m && col >= 0`（在矩阵范围内）
3. 根据比较结果移动：
   - 等于：找到，返回 true
   - 大于：左移（col--）
   - 小于：下移（row++）
4. 超出边界：未找到，返回 false

### 方法二：从左下角开始

```cpp
class Solution {
public:
    bool searchMatrix(vector<vector<int>>& matrix, int target) {
        if (matrix.empty() || matrix[0].empty()) return false;
        
        int m = matrix.size();
        int n = matrix[0].size();
        
        // 从左下角开始
        int row = m - 1;
        int col = 0;
        
        while (row >= 0 && col < n) {
            if (matrix[row][col] == target) {
                return true;
            } else if (matrix[row][col] > target) {
                // 当前值太大，向上移动
                row--;
            } else {
                // 当前值太小，向右移动
                col++;
            }
        }
        
        return false;
    }
};
```

左下角的逻辑：
- 大于 target：向上（row--）
- 小于 target：向右（col++）

### 方法三：每行二分查找

```cpp
class Solution {
public:
    bool searchMatrix(vector<vector<int>>& matrix, int target) {
        if (matrix.empty() || matrix[0].empty()) return false;
        
        for (const auto& row : matrix) {
            // 对每一行进行二分查找
            if (binary_search(row.begin(), row.end(), target)) {
                return true;
            }
        }
        
        return false;
    }
};
```

**手动实现二分查找：**
```cpp
class Solution {
public:
    bool searchMatrix(vector<vector<int>>& matrix, int target) {
        if (matrix.empty() || matrix[0].empty()) return false;
        
        int m = matrix.size();
        int n = matrix[0].size();
        
        for (int i = 0; i < m; i++) {
            // 剪枝：如果该行第一个元素大于target，后续行也不用找了
            if (matrix[i][0] > target) break;
            
            // 剪枝：如果该行最后一个元素小于target，继续下一行
            if (matrix[i][n - 1] < target) continue;
            
            // 二分查找
            int left = 0, right = n - 1;
            while (left <= right) {
                int mid = left + (right - left) / 2;
                if (matrix[i][mid] == target) {
                    return true;
                } else if (matrix[i][mid] < target) {
                    left = mid + 1;
                } else {
                    right = mid - 1;
                }
            }
        }
        
        return false;
    }
};
```

**复杂度分析：**

| 方法          | 时间复杂度 | 空间复杂度 | 难度 |
| ------------- | ---------- | ---------- | ---- |
| 右上角/左下角 | O(m + n)   | O(1)       | 简单 |
| 每行二分查找  | O(m log n) | O(1)       | 简单 |
| 每列二分查找  | O(n log m) | O(1)       | 简单 |

**易错点：**
1. 起点选择：只能是右上角或左下角
2. 移动方向：
   - 右上角：大了左移，小了下移
   - 左下角：大了上移，小了右移
3. 边界条件：`row < m && col >= 0`
4. 空矩阵的处理

**为什么时间复杂度是 O(m + n)？**
- 每次移动都会排除一行或一列
- 最多向下移动 m 次，向左移动 n 次
- 总共最多 m + n 次移动

**类比二叉搜索树：**
这个搜索过程类似于在 BST 中查找：
- 右上角相当于根节点
- 向下相当于去右子树（更大）
- 向左相当于去左子树（更小）

**相关问题：**
- 搜索二维矩阵（整个矩阵严格递增）
- 有序矩阵中第K小的元素


---

## 相关笔记
<!-- 自动生成 -->

暂无相关笔记

