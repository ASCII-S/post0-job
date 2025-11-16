---
created: '2025-10-19'
last_reviewed: null
next_review: '2025-10-19'
review_count: 0
difficulty: medium
mastery_level: 0.0
tags:
- C++
- C++/int、long、long_long在不同平台上的差异.md
related_outlines: []
---
# int、long、long_long在不同平台上的差异

 **C/C++ 标准并没有强制规定 `int/long/long long` 的确切字节数**，只规定了它们的相对大小关系：

* `sizeof(short) ≤ sizeof(int) ≤ sizeof(long) ≤ sizeof(long long)`
* `short ≥ 16 bit`，`int ≥ 16 bit`，`long ≥ 32 bit`，`long long ≥ 64 bit`
* 但具体多少字节由 **编译器和 ABI（应用二进制接口）** 决定。

---

### 常见平台上的实际情况

**1. Windows (LLP64 模型, 64 位 / 32 位 MSVC)**

* `int` = 4 字节 (32 位)
* `long` = 4 字节 (32 位)
* `long long` = 8 字节 (64 位)
  👉 在 Windows 下，哪怕是 64 位程序，`long` 依旧是 4 字节。

**2. Linux/Unix (LP64 模型, x86-64, LoongArch64, AArch64)**

* `int` = 4 字节 (32 位)
* `long` = 8 字节 (64 位)
* `long long` = 8 字节 (64 位)
  👉 在主流 Unix/Linux 64 位平台上，`long` 已经是 64 位。

**3. 32 位平台 (ILP32 模型, x86 32-bit, ARM 32-bit)**

* `int` = 4 字节 (32 位)
* `long` = 4 字节 (32 位)
* `long long` = 8 字节 (64 位)

---

### 标准面试回答可以这样说

“C/C++ 里整型的大小不是完全固定的，标准只保证相对大小关系。常见 ABI 模型有：

* **ILP32 (32 位平台)**：int/long = 32 bit，long long = 64 bit。
* **LP64 (Linux/Unix 64 位)**：int = 32 bit，long/long long = 64 bit。
* **LLP64 (Windows 64 位)**：int = 32 bit，long = 32 bit，long long = 64 bit。

所以在 64 位 Linux 上 `long` 是 8 字节，而在 64 位 Windows 上 `long` 还是 4 字节。为了跨平台写安全代码，通常推荐用 `<cstdint>` 里的 `int32_t`, `int64_t` 这样的定长类型。”

---

---

## 相关笔记
<!-- 自动生成 -->

暂无相关笔记

