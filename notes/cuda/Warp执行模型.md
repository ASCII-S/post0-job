---
created: '2025-10-19'
last_reviewed: null
next_review: '2025-10-19'
review_count: 0
difficulty: medium
mastery_level: 0.0
tags:
- cuda
- cuda/Warp执行模型.md
related_outlines: []
---
在一个 SM 中，warp 是最小调度单元。warp 内 32 个线程要么一起执行指令，要么一起等待。
Warp 调度器只负责在每个 cycle 挑选一个就绪 warp，发射它的指令到执行单元。
如果某个 warp 在等待 global memory，它会挂起，调度器就去调度别的 warp 执行计算。这样通过 warp 切换来掩盖访存延迟，而真正的计算和等待动作是 warp 内线程完成的，不是调度器本身。

在现代 GPU 中，一个 SM 通常有 4 个 warp 调度器，每个调度器每个周期可以挑一个就绪 warp，发射一条指令到执行单元。
这样，一个 SM 每个周期最多能同时发射 4 条 warp 指令，分别跑在 FP32、INT、Tensor、访存等不同执行单元上。
这保证了 SM 内部既有 warp 切换隐藏延迟，又有 执行单元层面的指令并行。

---

## 相关笔记
<!-- 自动生成 -->

暂无相关笔记

