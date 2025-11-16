---
created: '2025-10-19'
last_reviewed: null
next_review: '2025-10-19'
review_count: 0
difficulty: medium
mastery_level: 0.0
tags:
- C++
- C++/const关键字.md
related_outlines: []
---
# const关键字

“C++ 里的 const 关键字用于限定不可修改。

- 修饰普通变量：表示该变量只读，例如 const int a = 10;。

- 修饰指针：分顶层 const 和底层 const：

  - int* const p：顶层 const，p 不能改。

  - const int* p：底层 const，*p 不能改。

- 修饰函数参数：保证函数内部不修改传入对象，比如 void f(const int& x)。

- 修饰返回值：常见于返回引用或指针，防止调用者修改返回的对象，比如 const std::string& getName() const;。

- 修饰成员函数：在函数声明末尾加 const，保证该成员函数不修改类的成员变量，相当于 this 是一个 const 指针。

## const修饰成员函数

这个问题切到了精髓，面试官如果追问，你要能把 `this` 指针和 `const` 的关系说清楚。

---

### 1. 普通成员函数里的 `this`

在一个普通成员函数里，编译器会悄悄地给你传一个隐含参数 `this`：

```cpp
struct Foo {
    int x;
    void bar() { x = 42; } 
};
```

上面的 `bar`，其实编译器会翻译成：

```cpp
void bar(Foo* const this) { this->x = 42; }
```

注意这里的 `this` 是 **顶层 const**：它本身是个指针，不能改成指向别的对象，但通过它仍然可以修改所指向的对象。

---

### 2. const 成员函数里的 `this`

当你写成：

```cpp
struct Foo {
    int x;
    void baz() const { /* ... */ }
};
```

编译器会翻译成：

```cpp
void baz(const Foo* const this) { /* ... */ }
```

这里的区别是：

* **指向对象的类型变成了 `const Foo*`**。
* 也就是说，在函数体内，把 `this` 当成指向常量对象的指针。
* 因此你不能通过 `this->x` 去修改成员变量（除非成员被声明为 `mutable`）。

---

### 3. 这意味着什么？

* **常对象**只能调用 const 成员函数：

  ```cpp
  const Foo f;
  f.baz(); // ✅
  f.bar(); // ❌ 不允许，非 const 成员函数可能会修改对象
  ```
* const 成员函数就是在语义上向编译器和调用者承诺：
  “我不会修改对象的状态”。

---

### 4. 面试标准表述

“在 const 成员函数中，`this` 的类型会从 `Foo* const this` 变成 `const Foo* const this`。这意味着函数内部不能修改对象的非 mutable 成员。这样常对象也能调用这些函数，保证了对象的逻辑只读性。”

---

## 相关笔记
<!-- 自动生成 -->

暂无相关笔记

