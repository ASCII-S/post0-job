---
created: '2025-10-19'
last_reviewed: null
next_review: '2025-10-19'
review_count: 0
difficulty: medium
mastery_level: 0.0
tags:
- C++
- C++/lock_guard和std::unique_lock的区别.md
related_outlines: []
---
# std::lock_guard和std::unique_lock的区别

### 面试标准表述（精炼版）

std::lock_guard 和 std::unique_lock 都是 RAII 风格的锁管理类，用来避免手动 lock/unlock 带来的错误。

lock_guard：最轻量、最简单。构造时立即加锁，析构时自动解锁，不能延迟加锁，也不能手动解锁。适合大多数简单的临界区保护。

unique_lock：更灵活。支持延迟加锁（std::defer_lock）、手动 lock()/unlock()，可以多次加解锁，并且是 std::condition_variable 必须搭配的锁类型。适合复杂并发控制场景。

一句话对比：
lock_guard = 傻瓜锁，简单安全；
unique_lock = 高级锁，灵活强大。


---

### `std::lock_guard`

* **定位**：最轻量的 RAII 封装。
* **特征**：

  * 构造时立刻加锁，析构时立刻解锁。
  * 没有其他操作，不可复制、不可移动。
* **使用场景**：

  * 代码块里临时保护一段共享数据。
  * 没有复杂需求时，推荐用它（性能开销最小）。

```cpp
{
    std::lock_guard<std::mutex> lg(m); // 构造时加锁
    counter++;                         // 作用域内安全访问
} // 作用域结束自动解锁
```

---

### `std::unique_lock`

* **定位**：功能更强的 RAII 封装。
* **特征**：

  * 可以选择 **延迟加锁**（`std::defer_lock`）。
  * 可以 **手动 unlock()/lock()**，支持多次开关锁。
  * 可以和 `std::condition_variable` 搭配使用（必须用 `unique_lock`，因为条件变量需要能解锁再重新加锁）。
  * 可移动，不可复制。
* **使用场景**：

  * 需要灵活控制锁的生命周期。
  * 需要和条件变量一起用。

```cpp
std::unique_lock<std::mutex> ul(m, std::defer_lock); 
// 此时还没加锁
ul.lock(); 
counter++;
ul.unlock(); 
// 以后还可以再次 lock()
```

---

### 总结对比（面试答法）

* `lock_guard`：轻量、简单，作用域即生命周期，推荐默认使用。
* `unique_lock`：灵活，可延迟加锁、手动解锁、多次加锁，并且是条件变量的必需搭档。

**一句话**：
`lock_guard` 是“傻瓜式锁”，适合大多数场景；`unique_lock` 是“高级锁”，适合需要灵活控制和条件变量的场景。

---

要不要我顺着抛给你下一个并发必考点：**条件变量（`std::condition_variable`）的作用和用法**？

---

## 相关笔记
<!-- 自动生成 -->

暂无相关笔记

