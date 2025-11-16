---
created: '2025-10-19'
last_reviewed: null
next_review: '2025-10-19'
review_count: 0
difficulty: medium
mastery_level: 0.0
tags:
- Transformer
- Transformer/简单说说transformer从输入到输出的过程.md
related_outlines: []
---
好，给你一条从“文本 → 输出表示”的**Self-Attention全流程**（含形状、掩码、数值稳定与多头融合）。我用最常见的张量维度符号：

* 批大小 $B$，序列长度 $n$，模型维度 $d_{\text{model}}$，头数 $h$，单头维度 $d_k=d_v=\frac{d_{\text{model}}}{h}$。

---

# 从输入到输出：一步到位

## 1) 文本 → token → 向量

* 分词后得到 token 序列；查表得到嵌入：

  $$
  X \in \mathbb{R}^{B\times n\times d_{\text{model}}}
  $$

## 2) 加位置信号（绝对或相对）

* 绝对位置（如正弦）常见做法是逐位置相加：

  $$
  \tilde X = X + P \quad (P\text{与}X\text{同形状})
  $$
* 相对位置（如RoPE/ALiBi）通常影响打分阶段（见第4步）。

## 3) 线性投影得到 Q/K/V（参数化学习）

对每个头 $h=1..H$ 各有独立权重：

$$
W_Q^{(h)},W_K^{(h)},W_V^{(h)}\in\mathbb{R}^{d_{\text{model}}\times d_k}
$$

把 $\tilde X$ 投影并把头维显式展开：

$$
Q,K,V \in \mathbb{R}^{B\times h\times n\times d_k}
$$

（实现里常先做一次大矩阵乘法得到 $B\times n\times (3d_{\text{model}})$，再reshape/split成 $Q,K,V$ 的多头形状。）

## 4) 计算注意力分数（含掩码与相对位置偏置）

* 基础分数：

  $$
  S = \frac{QK^\top}{\sqrt{d_k}}
  \quad\Rightarrow\quad
  S \in \mathbb{R}^{B\times h\times n\times n}
  $$

  其中 $K^\top$ 指最后一维与倒数第二维做矩阵乘（对每个头、每个样本独立）。
* **掩码**（可选）：

  * 因果掩码（解码器自回归）：上三角置 $-\infty$。
  * padding 掩码：把 padding 位置对应的列置 $-\infty$。

  $$
  S \leftarrow S + \text{mask}
  $$
* **相对位置**（可选）：如 ALiBi 直接在 $S$ 上加线性偏置；RoPE通过对 $Q,K$ 做旋转嵌入影响点积结果。

## 5) Softmax 与数值稳定

* 对最后一维（keys 维）做 softmax：

  $$
  A=\text{Softmax}(S)\in \mathbb{R}^{B\times h\times n\times n}
  $$
* $\div \sqrt{d_k}$ 的目的是防止 $S$ 过大导致 softmax 饱和（梯度消失）。

## 6) 加权求和得到各头输出

$$
Z^{(h)} = A\,V \quad\Rightarrow\quad
Z \in \mathbb{R}^{B\times h\times n\times d_k}
$$

## 7) 多头拼接与输出投影

* 先沿头维拼接：

  $$
  \text{Concat}(Z^{(1)},\dots,Z^{(h)}) \in \mathbb{R}^{B\times n\times (h d_k)}=\mathbb{R}^{B\times n\times d_{\text{model}}}
  $$
* 再经输出投影：

  $$
  Y = \text{Concat}(Z)\, W_O,\quad W_O\in\mathbb{R}^{d_{\text{model}}\times d_{\text{model}}}
  $$

  $W_O$ 的作用是**融合各头特征**并**对齐维度**到 $d_{\text{model}}$。

## 8) 残差与归一化（Pre-LN形式）

$$
\hat Y = \text{LayerNorm}(X + Y)
$$

（现代Transformer多用 **Pre-LN**：先LN再子层或子层后残差再LN，按实现略有差异，但目标是稳定深层训练。）

## 9) 前馈网络（FFN）与再次残差/LN

典型两层 MLP（含GELU/SwiGLU等激活），中间维度一般是 $4\,d_{\text{model}}$：

$$
\text{FFN}(\hat Y)=\sigma(\hat Y W_1 + b_1) W_2 + b_2
$$

再做残差+LayerNorm：

$$
\text{Out}=\text{LayerNorm}(\hat Y + \text{FFN}(\hat Y))
$$

这就是一个完整的 Transformer 层输出，堆叠多层后进入任务头（分类、LM 头等）。

---

## 复杂度一眼看懂

* **时间复杂度**：注意力主耗时在 $QK^\top$ 与 $AV$。

  $$
  O(B\cdot h\cdot n^2\cdot d_k) \;\approx\; O(B\cdot n^2\cdot d_{\text{model}})
  $$
* **空间复杂度**：显式保存注意力权重 $A$ 需要

  $$
  O(B\cdot h\cdot n^2)
  $$

  这正是长序列的内存瓶颈（Flash-Attention 等优化就是围绕这点做的）。

---

**轮到你（只需回答一个问题）**：
给定批大小 $B$，序列长度 $n$，模型维度 $d_{\text{model}}$，头数 $h$（单头维度 $d_k=d_{\text{model}}/h$），请写出 $Q,K,V,S(=QK^\top/\sqrt{d_k}),A(=\text{Softmax}(S)),Z(=AV)$ 的**张量形状**；并说明 **注意力的时间复杂度** 与 **显式注意力矩阵的空间复杂度** 分别为何。

---

## 相关笔记
<!-- 自动生成 -->

暂无相关笔记

