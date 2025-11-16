---
created: '2025-10-19'
last_reviewed: null
next_review: '2025-10-19'
review_count: 0
difficulty: medium
mastery_level: 0.0
tags:
- cuda
- cuda/统一内存（Unified_Memory）.md
related_outlines: []
---
理想答案应该是：
统一内存是 CUDA 提供的一种编程模型，通过 cudaMallocManaged 分配，得到的内存可以同时被 CPU 和 GPU 访问。开发者只需要维护一个指针，CUDA 运行时会根据需要自动在 Host 和 Device 之间迁移数据。

优点：编程更简单，统一的地址空间让代码更直观；不需要显式写 cudaMemcpy。

缺点：自动迁移带来额外开销，性能可能比手动拷贝差；在多 GPU 环境下可能引发复杂的跨设备迁移问题。

---

## 相关笔记
<!-- 自动生成 -->

暂无相关笔记

