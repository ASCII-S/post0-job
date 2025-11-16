---
created: '2025-10-19'
last_reviewed: null
next_review: '2025-10-19'
review_count: 0
difficulty: medium
mastery_level: 0.0
tags:
- Transformer
- Transformer/简述Seq2Seq训练流程.md
related_outlines: []
---
很好，你的问题实际上在追问 **Seq2Seq 输入是什么、token 和 embedding 的关系、训练到底在优化谁**。我来把这几个环节捋清楚：

---

### 1. 输入 $x_1, x_2, \dots, x_n$ 代表什么？

* 在自然语言处理里，原始输入是句子的文本，比如 *“猫喜欢鱼”*。
* 经过 **分词 (tokenization)**，句子被切分成 token 序列：$猫, 喜欢, 鱼$。
* 每个 token 通过 **embedding 层** 映射为一个向量，通常是 $d_{\text{model}}$ 维。
* 所以 **$x_1, x_2$** 就是输入序列第 1、2 个 token 的 embedding 向量。

---

### 2. Seq2Seq 模型如何处理这些输入？

* **编码器 (Encoder)**：把 token embedding 序列 $(x_1, x_2, \dots, x_n)$ 输入 RNN/Transformer，得到一系列隐藏状态 $(h_1, h_2, \dots, h_n)$。这些隐藏状态包含了输入 token 的语义和上下文信息。
* **解码器 (Decoder)**：在训练阶段，解码器会逐步生成输出序列（比如目标语言的翻译）。输入包括：

  * 上一步生成的 token embedding；
  * 编码器提供的上下文（传统 Seq2Seq 里是最后一个隐藏状态，有了 Attention 后就是加权和）。

---

### 3. 训练时到底在优化谁？

训练目标是让解码器生成正确的目标序列。

* 输入：源语言 token 序列 → embedding → 编码器。
* 输出：目标语言 token 序列（比如中文翻译）。
* **损失函数**：通常是交叉熵 (cross-entropy)，比较解码器预测的概率分布和真实的目标 token。
* **优化对象**：整个模型的参数，包括

  * embedding 矩阵，
  * 编码器的权重，
  * 解码器的权重，
  * 注意力机制里的参数。

所以训练本质上是 **最小化预测序列和真实序列之间的差距**，通过梯度下降更新所有可学习的参数。

---

✅ **一句话总结**：

* $x_i$ 是输入序列中第 i 个 token 的 embedding。
* 编码器把它们转成隐藏状态序列，解码器再利用这些状态生成目标序列。
* 训练时我们优化的是 embedding、编码器、解码器和注意力的参数，使得预测输出尽可能接近真实标签。

---

要不要我接下来帮你补充 **Bahdanau Attention 的计算公式**，这样你能更直观地看到训练时注意力权重是怎么学到的？

---

## 相关笔记
<!-- 自动生成 -->

暂无相关笔记

