---
created: '2025-10-19'
last_reviewed: null
next_review: '2025-10-19'
review_count: 0
difficulty: medium
mastery_level: 0.0
tags:
- C++
- C++/atomic的基本使用.md
related_outlines: []
---
# std::atomic的基本使用

### 面试标准表述（精炼版）

std::atomic 是 C++11 引入的一个类模板，用于对变量进行原子操作。所谓原子操作，是指在硬件层面不可分割的指令序列，保证在多线程环境下不会产生数据竞争。

使用 std::atomic，我们可以对变量执行安全的 load、store、自增自减，甚至 compare_exchange（CAS）操作。
这些操作通常由 CPU 的原子指令直接支持，因此大多数原子类型是 lock-free 的，比用互斥锁来保护单个变量更高效。

不过，atomic 只能保证单个变量的操作安全，如果要保证多个操作的整体一致性，还是需要 mutex。

---

### 1. 什么是原子操作

所谓 **原子操作**，就是一段操作在硬件层面不可分割，不会被线程切换打断。对多线程环境下的共享变量来说，这能避免数据竞争。

例如普通的 `counter++` 在汇编层面其实是三步：读 → 加 → 写。如果两个线程同时操作，就会出现丢失更新。
而 `std::atomic<int> counter; counter++;` 保证了整个自增是一个原子操作，线程安全。

---

### 2. 基本用法

```cpp
#include <atomic>
#include <thread>
#include <iostream>

std::atomic<int> counter{0};

void work() {
    for (int i = 0; i < 100000; i++) {
        counter++;  // 原子自增
    }
}

int main() {
    std::thread t1(work), t2(work);
    t1.join();
    t2.join();
    std::cout << counter << "\n";  // 一定是 200000
}
```

---

### 3. 常见原子操作接口

* **读写**：

  ```cpp
  a.store(10);       // 原子写
  int x = a.load();  // 原子读
  ```
* **算术和逻辑操作**：

  ```cpp
  a++;  --a;
  a.fetch_add(5);   // 返回旧值
  a.fetch_sub(2);
  a.exchange(100);  // 原子交换
  ```
* **CAS（Compare-And-Swap）**：

  ```cpp
  int expected = 10;
  if (a.compare_exchange_strong(expected, 20)) {
      // a 从 10 改到 20 成功
  }
  ```

---

### 4. 注意点

* `std::atomic` 默认是 **无锁实现**，直接映射到 CPU 原子指令（如 x86 的 `lock` 前缀），效率很高。
* 但不是所有类型都能做 lock-free，标准库里有个 `is_lock_free()` 可以检测。
* 原子操作解决的是数据竞争，但如果多个操作需要组合成“事务”，还是要靠 `mutex`。

---

### 一句话总结

`std::atomic` 提供了对共享变量的原子操作，避免数据竞争，常见操作有 `load/store`、`fetch_add`、`compare_exchange` 等，适合替代简单的锁来实现高效线程安全。


---

## 相关笔记
<!-- 自动生成 -->

暂无相关笔记

