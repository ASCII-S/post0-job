---
created: '2025-10-19'
last_reviewed: null
next_review: '2025-10-19'
review_count: 0
difficulty: medium
mastery_level: 0.0
tags:
- C++
- C++/include的工作原理.md
related_outlines: []
---
# #include的工作原理

`#include` 是 C/C++ 里的一个 **预处理指令**，它的工作原理就是在 **预处理阶段**，把目标头文件的内容原封不动地拷贝到源文件中对应的位置。

它有两种常见形式：

* `#include "file.h"`：优先在当前源文件所在目录查找，如果找不到再去系统头文件目录查找。
* `#include <file.h>`：只在编译器设置的系统头文件路径里查找，比如 `/usr/include` 或者编译器内置的路径。

在编译器实现上，预处理器会根据包含路径搜索到目标文件，然后把内容展开。比如：

```cpp
#include <stdio.h>
```

经过预处理后，你的源文件里就真的被替换成了 `<stdio.h>` 文件里的几千行声明。

所以编译器在后续编译阶段就能识别 `printf`、`scanf` 这些函数的声明。

---

需要注意的是：

* `#include` 只是做 **文本替换**，不会做语义检查。如果你重复包含一个头文件，没有防护措施就会导致“重复定义”。因此头文件里通常要加 **include guard**：

  ```cpp
  #ifndef HEADER_H
  #define HEADER_H
  // 内容
  #endif
  ```

  或者直接用 `#pragma once`。
* 编译器通过命令行参数 `-I` 可以指定额外的头文件搜索路径。

---

所以一句话总结：
**`#include` 在预处理阶段把头文件展开到源代码里，相当于拼接源代码文件。编译器借此能获得函数和类型的声明，从而保证后续编译和链接的正确性。**

---

## 相关笔记
<!-- 自动生成 -->

暂无相关笔记

