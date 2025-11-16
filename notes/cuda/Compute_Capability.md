---
created: '2025-10-19'
last_reviewed: null
next_review: '2025-10-19'
review_count: 0
difficulty: medium
mastery_level: 0.0
tags:
- cuda
- cuda/Compute_Capability.md
related_outlines: []
---
Compute Capability 是 GPU 架构的功能版本号，标识支持的硬件特性，而不是算力大小。
不同版本有不同特性：Volta (7.0) 引入 Tensor Core，Turing (7.5) 加强推理 INT8/INT4，Ampere (8.x) 支持 BF16、TF32 和稀疏性，Hopper (9.0) 支持 FP8、Transformer Engine、DPX。
编译低版本 CC 可以在新 GPU 上跑，但用不到新特性；编译高版本 CC 则旧 GPU 可能不兼容。
实际工程里我们用 -gencode 同时生成目标架构的 SASS 并保留 PTX，既能在当前 GPU 上高效运行，也能保证未来 GPU 的兼容性。

---

## 相关笔记
<!-- 自动生成 -->

暂无相关笔记

