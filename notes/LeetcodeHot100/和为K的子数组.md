---
created: '2025-10-19'
last_reviewed: null
next_review: '2025-10-19'
review_count: 0
difficulty: medium
mastery_level: 0.0
tags:
- LeetcodeHot100
- LeetcodeHot100/和为K的子数组.md
related_outlines: []
---
# 和为 K 的子数组

## 题目描述

给你一个整数数组 `nums` 和一个整数 `k`，请你统计并返回该数组中和为 `k` 的连续子数组的个数。

**示例 1：**
```
输入：nums = [1,1,1], k = 2
输出：2
```

**示例 2：**
```
输入：nums = [1,2,3], k = 3
输出：2
```

**提示：**
- 1 <= nums.length <= 2 × 10^4
- -1000 <= nums[i] <= 1000
- -10^7 <= k <= 10^7

## 思路讲解

这道题虽然归类在滑动窗口，但实际上更适合用**前缀和 + 哈希表**的方法。

### 为什么不能用普通滑动窗口？

滑动窗口通常适用于：
- 数组元素都是正数
- 窗口大小可以明确增减

本题中数组可能包含负数，窗口无法简单地通过和的大小来决定收缩或扩展。

### 前缀和 + 哈希表

**核心思想：**
- 使用前缀和将问题转化为：找有多少对 `(i, j)` 满足 `sum[j] - sum[i] = k`
- 即 `sum[i] = sum[j] - k`
- 使用哈希表记录前缀和出现的次数

**前缀和定义：**
- `preSum[i]` 表示 `nums[0...i-1]` 的和
- 子数组 `nums[i...j]` 的和 = `preSum[j+1] - preSum[i]`

**算法步骤：**
1. 遍历数组，计算前缀和
2. 对于每个位置 j，查找之前有多少个前缀和等于 `preSum[j] - k`
3. 使用哈希表记录每个前缀和出现的次数
4. 累加符合条件的子数组数量

**为什么这样可行？**
- 如果 `preSum[j] - preSum[i] = k`，那么子数组 `nums[i...j-1]` 的和为 k
- 我们在遍历到位置 j 时，查找之前有多少个位置 i 满足 `preSum[i] = preSum[j] - k`
- 这个数量就是以 j-1 结尾、和为 k 的子数组数量

**示例演示：**
```
nums = [1, 2, 3], k = 3

前缀和计算：
preSum[0] = 0 (空数组)
preSum[1] = 1 (nums[0])
preSum[2] = 3 (nums[0..1])
preSum[3] = 6 (nums[0..2])

遍历过程：
j=0, preSum=1, 需要找preSum=1-3=-2, count=0, 记录{0:1, 1:1}
j=1, preSum=3, 需要找preSum=3-3=0, count=1, 记录{0:1, 1:1, 3:1}
j=2, preSum=6, 需要找preSum=6-3=3, count=1, 记录{0:1, 1:1, 3:1, 6:1}

结果: 2 (子数组[2]和[0,1])
```

## 面试时的快速口述讲解

这道题要统计和为 k 的连续子数组个数，数组中可能有负数。

**数据结构**：前缀和 + 哈希表（unordered_map）。

**实现方式**：
1. 维护前缀和，表示从数组开头到当前位置的累加和
2. 使用哈希表记录每个前缀和出现的次数
3. 对于每个位置，查找之前是否存在前缀和等于"当前前缀和 - k"
4. 如果存在，说明中间的子数组和为 k，累加这样的前缀和出现次数

**关键点**：
- 子数组和 = 当前前缀和 - 之前某个前缀和
- 转化为查找：`preSum[i] = preSum[j] - k`
- 初始化哈希表时要加入 `{0: 1}`，表示前缀和为 0 的情况（空数组）

**时间复杂度**：O(n)，遍历一次数组，哈希表操作 O(1)。

**空间复杂度**：O(n)，哈希表最多存储 n 个不同的前缀和。

## 代码实现

```cpp
class Solution {
public:
    int subarraySum(vector<int>& nums, int k) {
        // 哈希表：前缀和 -> 出现次数
        unordered_map<int, int> preSumCount;
        preSumCount[0] = 1;  // 初始化：前缀和为0出现1次（空数组）
        
        int preSum = 0;  // 当前前缀和
        int count = 0;   // 结果计数
        
        for (int num : nums) {
            // 计算当前前缀和
            preSum += num;
            
            // 查找是否存在前缀和 = preSum - k
            // 如果存在，说明存在子数组和为k
            if (preSumCount.find(preSum - k) != preSumCount.end()) {
                count += preSumCount[preSum - k];
            }
            
            // 记录当前前缀和
            preSumCount[preSum]++;
        }
        
        return count;
    }
};
```

**代码说明：**
1. 初始化哈希表，`preSumCount[0] = 1` 表示前缀和为 0 出现了 1 次
   - 这是为了处理从索引 0 开始的子数组
2. 遍历数组：
   - 累加当前元素到前缀和
   - 查找哈希表中是否存在 `preSum - k`
   - 如果存在，说明从那些位置到当前位置的子数组和为 k
   - 将当前前缀和记录到哈希表
3. 返回总计数

**为什么要初始化 preSumCount[0] = 1？**

考虑数组 `[1, 2]`，k = 3：
- 当遍历到索引 1 时，preSum = 3
- 我们需要找 preSum - k = 0
- 这个 0 代表"空数组"，表示从索引 0 开始的子数组 [1, 2] 和为 3
- 所以初始化时要加入 `{0: 1}`

**简化版本：**
```cpp
class Solution {
public:
    int subarraySum(vector<int>& nums, int k) {
        unordered_map<int, int> mp;
        mp[0] = 1;
        
        int sum = 0, count = 0;
        for (int num : nums) {
            sum += num;
            count += mp[sum - k];
            mp[sum]++;
        }
        
        return count;
    }
};
```

**复杂度分析：**
- 时间复杂度：O(n)，遍历数组一次
- 空间复杂度：O(n)，哈希表最多存储 n 个前缀和

**与滑动窗口的区别：**
| 特点       | 滑动窗口     | 前缀和+哈希表  |
| ---------- | ------------ | -------------- |
| 适用条件   | 元素都是正数 | 可以有负数     |
| 窗口大小   | 可变         | 不需要维护窗口 |
| 时间复杂度 | O(n)         | O(n)           |
| 空间复杂度 | O(1)         | O(n)           |

**易错点：**
1. 不要忘记初始化 `preSumCount[0] = 1`
2. 要先查找再更新哈希表，顺序不能颠倒
3. 累加的是 `preSumCount[preSum - k]` 的值，不是简单的 +1
4. 理解前缀和的含义：`preSum[i]` 是前 i 个元素的和


---

## 相关笔记
<!-- 自动生成 -->

- [无重复字符的最长子串](notes/LeetcodeHot100/无重复字符的最长子串.md) - 相似度: 31% | 标签: LeetcodeHot100, LeetcodeHot100/无重复字符的最长子串.md

