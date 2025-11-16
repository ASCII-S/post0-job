---
created: '2025-10-19'
last_reviewed: null
next_review: '2025-10-19'
review_count: 0
difficulty: medium
mastery_level: 0.0
tags:
- cuda
- cuda/Kernel启动配置.md
related_outlines: []
---
CUDA kernel 的启动配置用 <<<gridDim, blockDim, sharedMem, stream>>> 语法，其中：

gridDim：定义网格规模，即有多少个线程块。可以是一维、二维或三维（dim3 类型）。

blockDim：定义单个线程块中的线程数，也可以是一维、二维或三维。

sharedMem（可选）：为每个线程块动态分配的共享内存大小（字节数），默认是 0。

stream（可选）：指定 kernel 属于哪个 CUDA 流，没写时默认是 0 号流。

---

## 相关笔记
<!-- 自动生成 -->

暂无相关笔记

