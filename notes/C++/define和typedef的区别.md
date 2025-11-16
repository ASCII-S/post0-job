---
created: '2025-10-19'
last_reviewed: null
next_review: '2025-10-19'
review_count: 0
difficulty: medium
mastery_level: 0.0
tags:
- C++
- C++/define和typedef的区别.md
related_outlines: []
---
# define和typedef的区别

### 面试标准回答

“`#define` 是预处理指令，在编译前做文本替换，没有类型检查，容易出错；而 `typedef` 是编译器处理的关键字，定义真正的类型别名，有类型检查，也能保留调试信息。两者在简单场景下效果相似，但在指针、复杂声明等地方差别很大。现代 C++ 推荐用 `using` 代替 `typedef`。”
---

### 1. 本质区别

* **`#define`**

  * 属于预处理指令，在 **编译前** 由预处理器做文本替换。
  * 没有类型检查，只是“字符串替换”。
* **`typedef`**

  * 属于 C/C++ 的 **关键字**，在编译阶段由编译器处理。
  * 有严格的类型检查，真正引入了一个“类型别名”。

---

### 2. 使用方式

**`#define` 定义别名**：

```c
#define UINT unsigned int
UINT a;  // 实际展开成 unsigned int a;
```

**`typedef` 定义别名**：

```c
typedef unsigned int UINT;
UINT a;  // 真正的新类型别名
```

表面看起来类似，但语义完全不同：

* `#define` 完全是文本替换，编译器看到的只是 `unsigned int`。
* `typedef` 让 `UINT` 成为一个类型标识符，能参与类型检查。

---

### 3. 指针类型的区别（常见坑点）

```c
#define PINT int*
typedef int* PINT2;

PINT a, b;   // a 是 int*，但 b 只是 int （因为 #define 只是替换成 int* a, int b）
PINT2 c, d;  // c 和 d 都是 int* （因为 typedef 定义了一个真正的指针类型别名）
```

👉 这正是 `typedef` 的优势，避免了宏替换带来的歧义。

---

### 4. 调试与编译支持

* 宏替换后调试器里只看到替换后的原始类型。
* typedef 的别名在调试信息里能保留，更清晰。

---

### 5. C++ 中的扩展

在现代 C++ 里，还可以用 `using` 代替 `typedef`，语义更直观：

```cpp
using UINT = unsigned int;
using PFunc = void(*)(int);
```

---

---

## 相关笔记
<!-- 自动生成 -->

暂无相关笔记

