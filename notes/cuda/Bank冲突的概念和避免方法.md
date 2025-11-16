---
created: '2025-10-19'
last_reviewed: null
next_review: '2025-10-19'
review_count: 0
difficulty: medium
mastery_level: 0.0
tags:
- cuda
- cuda/Bank冲突的概念和避免方法.md
related_outlines: []
---
Bank 是共享内存的并行存储单元。Warp 内 32 个线程同时访问共享内存时，如果访问地址落在不同 Bank，就能并行执行；如果落在同一个 Bank，就会发生 Bank 冲突，访问会被串行化，延迟增加。
避免方法包括：让线程按连续地址访问、在数组维度上做 padding 打破冲突、或者利用广播机制。

---

## 相关笔记
<!-- 自动生成 -->

- [Shared_Memory](notes/cuda/Shared_Memory.md) - 相似度: 31% | 标签: cuda, cuda/Shared_Memory.md
- [什么是Bank_Conflict？它如何影响性能？](notes/cuda/什么是Bank_Conflict？它如何影响性能？.md) - 相似度: 31% | 标签: cuda, cuda/什么是Bank_Conflict？它如何影响性能？.md

