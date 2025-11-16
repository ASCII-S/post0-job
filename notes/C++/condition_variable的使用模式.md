---
created: '2025-10-19'
last_reviewed: null
next_review: '2025-10-19'
review_count: 0
difficulty: medium
mastery_level: 0.0
tags:
- C++
- C++/condition_variable的使用模式.md
related_outlines: []
---
# std::condition_variable的使用模式

### 面试标准表述（精炼版）

条件变量是 C++ 提供的一种线程间同步机制，常和 std::unique_lock<std::mutex> 搭配使用。它允许一个线程在条件不满足时阻塞等待，同时释放锁，让其他线程可以修改条件；当条件满足时，另一个线程通过 notify_one() 或 notify_all() 唤醒等待线程。

使用模式通常是：

```cpp
std::unique_lock<std::mutex> lock(m);
cv.wait(lock, []{ return 条件满足; });
// 被唤醒后继续执行临界区代码
```
这样能避免线程忙等，提高效率。注意要用 wait(lock, predicate) 来防止 虚假唤醒。

一句话总结：条件变量提供了线程间的等待–通知机制，保证线程只在条件满足时继续执行，并避免锁资源被无谓占用。

---

### 1. 基本概念

* 条件变量不是锁，它需要和 **互斥锁** 一起使用。
* 等待方线程在 `wait()` 时会：

  1. 自动释放传入的锁（保证其他线程能修改条件）；
  2. 进入阻塞等待；
  3. 当被唤醒时，再自动重新获得锁，然后继续往下执行。

---

### 2. 基本用法

典型生产者–消费者模型：

```cpp
#include <iostream>
#include <thread>
#include <mutex>
#include <condition_variable>
#include <queue>

std::mutex m;
std::condition_variable cv;
std::queue<int> q;

void producer() {
    for (int i = 0; i < 5; i++) {
        std::unique_lock<std::mutex> lock(m);
        q.push(i);
        std::cout << "produce " << i << "\n";
        cv.notify_one();  // 通知一个等待线程
    }
}

void consumer() {
    for (int i = 0; i < 5; i++) {
        std::unique_lock<std::mutex> lock(m);
        cv.wait(lock, []{ return !q.empty(); });  // 条件不满足就阻塞
        int val = q.front();
        q.pop();
        std::cout << "consume " << val << "\n";
    }
}

int main() {
    std::thread t1(producer);
    std::thread t2(consumer);
    t1.join();
    t2.join();
}
```

---

### 3. 关键点

* **必须用 `unique_lock`** 而不能用 `lock_guard`，因为 `wait()` 内部需要先解锁再加锁。
* `wait(lock, pred)` 的好处是：会在被唤醒时重新检查条件，避免 **虚假唤醒**。
* `notify_one()` 唤醒一个等待线程，`notify_all()` 唤醒所有等待线程。

---

### 4. 面试一句话总结

条件变量用来在线程间进行**等待–通知机制**，避免无效轮询。它必须和 `unique_lock` 配合使用，`wait` 会自动释放和重新获取锁，常见场景是生产者–消费者模型。

---

## 相关笔记
<!-- 自动生成 -->

- [实现两个线程交替打印0-1000](notes/C++/实现两个线程交替打印0-1000.md) - 相似度: 31% | 标签: C++, C++/实现两个线程交替打印0-1000.md

