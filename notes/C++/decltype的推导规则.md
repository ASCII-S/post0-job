---
created: '2025-10-19'
last_reviewed: null
next_review: '2025-10-19'
review_count: 0
difficulty: medium
mastery_level: 0.0
tags:
- C++
- C++/decltype的推导规则.md
related_outlines: []
---
在 C++11 中，`decltype` 是一个用于类型推导的关键字。它和 `auto` 不同，`decltype` 不会丢掉 const、不会丢掉引用，而是会**完整地按照表达式的类型来推导**。

核心规则：

1. **变量名**：如果传给 `decltype` 的是一个变量名，它会得到该变量的类型，包括顶层 const 和引用。

   ```cpp
   int x = 0;  
   const int cx = 1;  
   int& rx = x;  
   decltype(x) a;    // int  
   decltype(cx) b;   // const int  
   decltype(rx) c = x; // int&  
   ```

2. **表达式**：

   * 如果表达式是一个**纯右值**（比如算术表达式），`decltype(expr)` 会得到对应的值类型。

     ```cpp
     decltype(x + 0) d; // int  
     ```
   * 如果表达式是**左值**，推导结果会是该类型的引用。

     ```cpp
     decltype((x)) e = x; // int&   注意：双括号让它变成一个左值表达式  
     ```

3. **和 auto 的区别**：

   * `auto` 在推导时会去掉顶层 const 和引用。
   * `decltype` 保留完整的类型信息，更适合在模板和泛型编程中获取精确类型。

4. **常见使用场景**：

   * 在泛型函数中推导返回类型：

     ```cpp
     template <typename T, typename U>
     auto add(T t, U u) -> decltype(t + u) {
         return t + u;
     }
     ```
   * 定义和某个表达式完全一致的变量类型。

---

---

## 相关笔记
<!-- 自动生成 -->

暂无相关笔记

