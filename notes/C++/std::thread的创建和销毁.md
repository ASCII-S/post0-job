---
created: '2025-10-19'
last_reviewed: null
next_review: '2025-10-19'
review_count: 0
difficulty: medium
mastery_level: 0.0
tags:
- C++
- C++/std::thread的创建和销毁.md
related_outlines: []
---
# std::thread的创建和销毁

### 面试标准表述（精炼版）
std::thread 构造时传入可调用对象就会启动一个新线程。

必须在 std::thread 对象销毁前调用 join() 或 detach()，否则会触发 std::terminate。

join()：阻塞当前线程，等待子线程结束，并收回资源。

detach()：将子线程与 std::thread 对象分离，让它在后台运行，结束时系统自动清理资源，但主线程无法再跟踪它。
---

### 1. 创建方式

`std::thread` 的构造函数接收一个**可调用对象**（callable）：函数、lambda、函数对象都行。

```cpp
#include <iostream>
#include <thread>

void func(int x) {
    std::cout << "thread " << x << "\n";
}

int main() {
    std::thread t1(func, 42);         // 用普通函数
    std::thread t2([]{ std::cout << "lambda thread\n"; });  // 用 lambda
    std::thread t3([](int a, int b){ std::cout << a+b << "\n"; }, 1, 2);

    t1.join();
    t2.join();
    t3.join();
}
```

**注意**：构造时，参数会按值拷贝到线程函数。如果要传引用，要用 `std::ref`。

---

### 2. 销毁方式

* **join()**：阻塞当前线程，直到子线程执行完毕。调用后，该 `thread` 对象不再和系统线程关联。
* **detach()**：让子线程在后台独立运行，与 `thread` 对象分离。调用后线程成为守护线程，不能再 `join`。

```cpp
std::thread t(func);
t.join();   // 等待执行结束，最常用
// 或者
t.detach(); // 后台运行，主线程退出时可能导致资源未清理
```

* **析构函数规则**：

  * 如果一个 `std::thread` 在析构时还是“joinable”的（即还没 join/detach），程序会直接调用 `std::terminate()` 崩溃。
  * 因此一定要在对象销毁前显式调用 `join()` 或 `detach()`。

---

### 3. 小结

* `std::thread` 构造时传入可调用对象即可启动线程。
* 必须在析构前 **join 或 detach**，否则会 terminate。
* `join()` 是常用的安全做法，`detach()` 适合后台守护线程。

---

要不要我下一步抛给你一个常见考点：**join 和 detach 的区别，以及为什么析构时没处理会 terminate**？

---

## 相关笔记
<!-- 自动生成 -->

暂无相关笔记

