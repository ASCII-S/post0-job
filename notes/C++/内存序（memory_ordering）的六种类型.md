---
created: '2025-10-19'
last_reviewed: null
next_review: '2025-10-19'
review_count: 0
difficulty: medium
mastery_level: 0.0
tags:
- C++
- C++/内存序（memory_ordering）的六种类型.md
related_outlines: []
---
# 内存序（memory_ordering）的六种类型

### 面试标准表述（精炼版）

内存序是 C++ 中的一个概念，用于描述多线程环境下变量操作的顺序。

主要有六种内存序：

memory_order_relaxed：最宽松的内存序，不保证操作的顺序。
memory_order_consume：保证操作的顺序，但只能用于依赖关系。
memory_order_acquire：保证操作的顺序，但只能用于依赖关系。

好，这就是并发编程里最“硬核”的一块：**C++ 内存模型的六种内存序（memory order）**。`std::atomic` 的每个操作都可以指定内存序，来决定编译器和 CPU 在指令重排、缓存一致性上的约束。

---

### 内存序是什么

内存序（memory ordering）是 C++ 并发编程中用来控制**多线程对内存操作顺序**的一套规则。

在多线程环境下，编译器和 CPU 可能会为了优化性能而重新排列指令顺序（指令重排），但这可能导致一个线程看到的操作顺序与另一个线程不一致，从而引发数据竞争和逻辑错误。

内存序就是告诉编译器和硬件：**哪些操作不能被重排，以及线程间的可见性要求**。

例如：


---

### 六种内存序（从最弱到最强）

1. **`memory_order_relaxed`**

   * 只保证操作本身是原子的，不做任何同步或顺序保证。
   * 用途：计数器、统计类变量，不依赖于线程间的可见顺序。

2. **`memory_order_consume`**（已弃用，等同于 acquire）

   * 原本设计是：只保证依赖于该原子值的操作不会被重排到前面。
   * 因为实现太复杂，C++17 开始基本当作 `acquire` 处理。

3. **`memory_order_acquire`**

   * 保证当前线程中，所有**在 acquire 之后的操作**不会被重排到 acquire 之前。
   * 常用于：加载（`load`），确保读到的数据，以及依赖于它的后续操作都是最新的。

4. **`memory_order_release`**

   * 保证当前线程中，所有**在 release 之前的操作**不会被重排到 release 之后。
   * 常用于：存储（`store`），确保在 release 前写入的数据对其他 acquire 可见。

5. **`memory_order_acq_rel`**

   * 同时具备 acquire + release 的语义。
   * 常用于 `compare_exchange` 这种既读又写的原子操作。

6. **`memory_order_seq_cst`**（Sequentially Consistent，顺序一致性，默认）

   * 最强语义：所有线程看到的原子操作顺序一致，禁止重排。
   * 简单安全，但性能开销可能较大。

---

### 小总结（面试可背版）

* **relaxed**：只保证原子性，不保证顺序。
* **acquire**：读操作，保证之后的不会跑到前面。
* **release**：写操作，保证之前的不会跑到后面。
* **acq\_rel**：读+写都加约束。
* **seq\_cst**：最严格，全局统一顺序。
* **consume**：理论上弱于 acquire，但标准里基本弃用。

---

### 一句话总结

C++ 的六种内存序提供了不同强度的同步保证，从性能最好的 `relaxed` 到语义最强的 `seq_cst`。在面试时只要能清晰说出 acquire / release / seq\_cst 三个主要语义，就已经超过大部分候选人了。

---

要不要我接着帮你整理一张 **“内存序语义 + 典型使用场景”对照表**，方便你快速记忆？

---

## 相关笔记
<!-- 自动生成 -->

暂无相关笔记

