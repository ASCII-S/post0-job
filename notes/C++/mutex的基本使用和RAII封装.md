---
created: '2025-10-19'
last_reviewed: null
next_review: '2025-10-19'
review_count: 0
difficulty: medium
mastery_level: 0.0
tags:
- C++
- C++/mutex的基本使用和RAII封装.md
related_outlines: []
---
# std::mutex的基本使用和RAII封装

### 面试标准表述（精炼版）
std::mutex 是 C++11 提供的互斥量，用于保证同一时间只有一个线程能访问共享资源。最基本的用法是 lock() 和 unlock()，但这样容易忘记解锁，导致死锁。

RAII 封装解决了这个问题：std::lock_guard 在构造时自动加锁，析构时自动解锁；std::unique_lock 则更灵活，可以延迟加锁、提前解锁，还能和条件变量结合使用。这样能保证即使发生异常或提前返回，锁也一定会被正确释放。

---

### 1. `std::mutex` 的基本使用

`std::mutex` 是 C++11 提供的互斥量，用于保证同一时间只有一个线程能访问共享资源。

最基本用法：

```cpp
#include <iostream>
#include <thread>
#include <mutex>

int counter = 0;
std::mutex m;

void work() {
    for (int i = 0; i < 100000; i++) {
        m.lock();      // 加锁
        counter++;
        m.unlock();    // 解锁
    }
}

int main() {
    std::thread t1(work), t2(work);
    t1.join();
    t2.join();
    std::cout << "counter = " << counter << "\n";
}
```

**缺点**：必须手动 `lock()` 和 `unlock()`，如果中途抛异常或函数提前返回，很容易忘记 `unlock()`，导致死锁。

---

### 2. RAII 封装

RAII（Resource Acquisition Is Initialization）思想：把资源的获取和释放绑定到对象的生命周期里。

C++ 提供了两种现成的 RAII 封装：

* `std::lock_guard<std::mutex>`（最简单的，作用域结束时自动解锁）
* `std::unique_lock<std::mutex>`（功能更灵活，可以延迟加锁、提前解锁、支持条件变量）

示例：

```cpp
void work() {
    for (int i = 0; i < 100000; i++) {
        std::lock_guard<std::mutex> lg(m);  // 构造时加锁，析构时解锁
        counter++;
    }
}
```

再比如 `unique_lock`：

```cpp
void work() {
    for (int i = 0; i < 100000; i++) {
        std::unique_lock<std::mutex> ul(m);
        counter++;
        // 可以手动提前释放
        ul.unlock();
    }
}
```

---

### 3. 总结回答（面试版）

* `std::mutex` 用于保护共享资源，防止多个线程并发修改导致竞态条件。
* 基本方式是 `lock()`/`unlock()`，但容易出错。
* 推荐用 RAII 封装：`std::lock_guard` 或 `std::unique_lock`，保证作用域结束时自动释放锁，避免忘记解锁导致死锁。

---

## 相关笔记
<!-- 自动生成 -->

暂无相关笔记

