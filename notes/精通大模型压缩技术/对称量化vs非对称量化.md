---
created: '2025-10-19'
last_reviewed: null
next_review: '2025-10-19'
review_count: 0
difficulty: medium
mastery_level: 0.0
tags:
- 精通大模型压缩技术
- 精通大模型压缩技术/对称量化vs非对称量化.md
related_outlines: []
---
# 对称量化 vs 非对称量化

## 面试标准答案（精简版）

**对称量化**假设数据分布以零为中心对称，零点固定为0，量化公式为 \(x_q = \text{round}(x/s)\)，计算更简单高效。**非对称量化**不要求数据对称分布，引入零点参数 \(z\)，可以更充分利用量化范围，适用于ReLU后的激活值等非对称数据。权重通常适合对称量化，激活值更适合非对称量化。

---

## 详细讲解

### 1. 对称量化（Symmetric Quantization）

#### 定义
对称量化假设浮点数的取值范围关于零点对称，即 \([-\alpha, \alpha]\)，将其映射到整数范围 \([-2^{b-1}, 2^{b-1}-1]\)（b为比特数）。

#### 量化公式

$$
s = \frac{\max(|x_{\max}|, |x_{\min}|)}{2^{b-1}}
$$

$$
x_q = \text{round}\left(\frac{x}{s}\right)
$$

$$
\tilde{x} = s \cdot x_q
$$

**关键特点**：零点 \(z = 0\)

#### 示例
- 浮点范围：[-5.0, 5.0]
- INT8量化：[-128, 127]
- Scale: \(s = 5.0 / 128 \approx 0.039\)
- 浮点值 2.0 → 量化值 \(\text{round}(2.0/0.039) = 51\)

#### 优点
- **计算极简**：只需一次除法和取整，反量化只需一次乘法
- **硬件高效**：减少一个加法操作（无零点偏移）
- **矩阵乘法友好**：\(Y = WX\) 的量化形式更简单
  $$
  Y_q = W_q \times X_q
  $$
  无需处理零点项

#### 缺点
- **范围利用不充分**：当数据不对称时（如[0, 10]），负半轴的量化点被浪费
- **精度损失**：非对称数据被迫对称映射，损失精度

### 2. 非对称量化（Asymmetric Quantization）

#### 定义
非对称量化不假设数据对称分布，引入零点参数，可以将任意范围 \([x_{\min}, x_{\max}]\) 映射到整数范围。

#### 量化公式

$$
s = \frac{x_{\max} - x_{\min}}{2^b - 1}
$$

$$
z = \text{round}\left(-\frac{x_{\min}}{s}\right)
$$

$$
x_q = \text{round}\left(\frac{x}{s} + z\right)
$$

$$
\tilde{x} = s \cdot (x_q - z)
$$

#### 示例
- 浮点范围：[0, 10.0]（ReLU后的激活值）
- INT8量化：[0, 255]
- Scale: \(s = 10.0 / 255 \approx 0.039\)
- Zero point: \(z = 0\)
- 浮点值 5.0 → 量化值 \(\text{round}(5.0/0.039) = 128\)

#### 优点
- **充分利用量化范围**：所有量化点都被有效使用
- **适应任意分布**：不要求数据对称性
- **精度更高**：对于非对称数据（如ReLU激活），精度明显优于对称量化

#### 缺点
- **计算复杂**：需要额外处理零点
- **矩阵乘法复杂**：量化矩阵乘法 \(Y = WX\) 变为
  $$
  Y_q = s_W \cdot s_X \cdot [(W_q - z_W)(X_q - z_X)] / s_Y + z_Y
  $$
  需要额外计算零点项
- **硬件开销**：需要更多计算和存储

### 3. 对比分析

| 特性                   | 对称量化 | 非对称量化 |
| ---------------------- | -------- | ---------- |
| **零点**               | z = 0    | z ≠ 0      |
| **数据要求**           | 对称分布 | 任意分布   |
| **量化范围利用率**     | 可能浪费 | 充分利用   |
| **计算复杂度**         | 低       | 高         |
| **精度（对称数据）**   | 高       | 相当       |
| **精度（非对称数据）** | 低       | 高         |
| **适用场景**           | 权重     | 激活值     |

### 4. 矩阵乘法的量化推导

#### 对称量化
$$
Y = W \cdot X
$$

$$
s_Y \cdot Y_q = (s_W \cdot W_q) \cdot (s_X \cdot X_q)
$$

$$
Y_q = \frac{s_W \cdot s_X}{s_Y} \cdot (W_q \cdot X_q)
$$

**简单！** 只需整数矩阵乘法 + 一次缩放

#### 非对称量化
$$
Y = W \cdot X
$$

$$
s_Y(Y_q - z_Y) = s_W(W_q - z_W) \cdot s_X(X_q - z_X)
$$

$$
Y_q = \frac{s_W \cdot s_X}{s_Y}[W_q X_q - z_W X_q - W_q z_X + z_W z_X] + z_Y
$$

**复杂！** 需要处理三个额外项

### 5. 实践中的选择

#### 权重量化
**推荐：对称量化**
- 权重分布通常接近对称（均值接近0）
- 可以通过批归一化使权重更对称
- 计算效率优先

#### 激活值量化
**推荐：非对称量化**
- ReLU后激活值范围为 [0, +∞)，严重非对称
- 充分利用量化范围带来的精度提升值得额外计算开销
- 某些框架（如TensorFlow Lite）默认使用非对称量化

#### 混合方案（常用）
- **权重**：对称量化（INT8）
- **激活**：非对称量化（UINT8）
- 平衡精度和性能

### 6. 代码示例

```python
import numpy as np

# 对称量化
def symmetric_quantize(x, bits=8):
    max_val = max(abs(x.max()), abs(x.min()))
    scale = max_val / (2 ** (bits - 1))
    x_q = np.round(x / scale).astype(np.int8)
    return x_q, scale

# 非对称量化
def asymmetric_quantize(x, bits=8):
    x_min, x_max = x.min(), x.max()
    scale = (x_max - x_min) / (2 ** bits - 1)
    zero_point = int(np.round(-x_min / scale))
    x_q = np.round(x / scale + zero_point).astype(np.uint8)
    return x_q, scale, zero_point

# 示例
weights = np.random.randn(100) * 2  # 对称分布，均值~0
activations = np.abs(np.random.randn(100))  # ReLU后，[0, +∞)

w_q, w_s = symmetric_quantize(weights)
a_q, a_s, a_z = asymmetric_quantize(activations)

print(f"权重范围: [{weights.min():.2f}, {weights.max():.2f}]")
print(f"激活范围: [{activations.min():.2f}, {activations.max():.2f}]")
```

---

**总结**：对称量化更高效，适合权重；非对称量化更精确，适合激活值。实际部署中常采用混合策略。


---

## 相关笔记
<!-- 自动生成 -->

暂无相关笔记

