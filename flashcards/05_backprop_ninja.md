---
tags:
---
# Lecture 5: Becoming a Backprop Ninja — 手动反向传播
#flashcards/nn-zero-to-hero


反向传播的本质是什么？每一步的计算模式是什么？
?
反向传播 = **链式法则的递归应用**。
每一步的模式：**上游梯度 × 局部导数**。
$$\frac{\partial L}{\partial x} = \frac{\partial L}{\partial g} \cdot \frac{\partial g}{\partial x}$$


`cmp()` 中 exact 和 approximate 分别检查什么？为什么 exact=False 但 approximate=True 仍说明推导正确？
?
- **exact**：`torch.all(dt == t.grad)`，bit-for-bit 完全相等
- **approximate**：`torch.allclose(dt, t.grad)`，允许浮点误差（|a-b| ≤ 1e-8 + 1e-5×|b|）
手动反传和 autograd 的计算路径不同，浮点舍入会导致极微小差异，所以 exact 可能为 False 但 approximate 为 True——推导仍然正确


为什么需要对中间变量调用 `retain_grad()`？
?
PyTorch 默认只保留**叶子节点**（参数）的 `.grad`，中间变量的梯度在 `backward()` 后会被释放以省内存。
调用 `retain_grad()` 使中间变量的梯度保留下来，方便用 `cmp()` 对比验证手动梯度


同一个变量被多条计算路径使用时，梯度怎么处理？
?
**梯度累加**。例如 `counts` 同时参与了 `probs = counts * counts_sum_inv` 和 `counts_sum = counts.sum(1)`，
反传时要把两条路径的梯度加起来：`dcounts = dcounts_1 + dcounts_2`


矩阵乘法 Y = A @ B 的反传梯度公式？
?
$$\frac{\partial L}{\partial A} = \frac{\partial L}{\partial Y} @ B^T, \qquad \frac{\partial L}{\partial B} = A^T @ \frac{\partial L}{\partial Y}$$
记忆技巧：结果形状必须和原矩阵匹配，转置哪个由维度对齐决定


前向传播中 `sum(0)` 把 (32,64) → (1,64)，反传时梯度怎么走？反过来呢？
?
- **前向 sum → 反向广播复制**：梯度从 (1,64) 广播回 (32,64)，每行得到同样的梯度
- **前向广播 → 反向 sum**：梯度沿广播维度求和
这是因为 sum 和 broadcast 互为逆运算


loss = -mean(logprobs[range(n), Yb]) 对 logprobs 的梯度是什么？为什么？
?
dlogprobs 是一个全零矩阵 (32,27)，只在 `[i, Yb[i]]` 位置为 **-1/n**。
因为 loss 只用到了每行中正确类别那一个元素的 log 概率，其余位置的偏导为 0；
负号来自 `-log`，1/n 来自 `mean`


tanh(x) 的导数是什么？如果已知 h = tanh(x)，如何用 h 表示导数？
?
$$\frac{d}{dx}\tanh(x) = 1 - \tanh^2(x) = 1 - h^2$$
代码：`dhpreact = (1 - h**2) * dh`
这避免了重新计算 tanh，直接用前向传播已有的 h


嵌入层反传为什么用 `index_add_` 而不是直接赋值？
?
同一个字符可能在 batch 中**出现多次**（比如字符 'a' 可能出现 10 次），
每次出现都产生一份梯度，这些梯度需要**累加**到 dC 的同一行上。
`dC.index_add_(0, Xb.view(-1), demb.view(-1, n_embd))` 就是按索引累加


Softmax + Cross-Entropy 合并反传的公式是什么？
?
$$\frac{\partial\,\text{loss}}{\partial\, l_i} = \begin{cases} P_i & i \neq y \\ P_i - 1 & i = y \end{cases}$$
向量形式：$\nabla_{\mathbf{l}}\text{loss} = \mathbf{P} - \mathbf{e}_y$
代码：`dlogits = softmax(logits); dlogits[range(n), Yb] -= 1; dlogits /= n`


为什么 PyTorch 的 `F.cross_entropy` 接收 logits 而不是 probs？
?
将 softmax 和 cross-entropy 合并计算有两个好处：
1. **梯度简洁**：合并后梯度 = P - e_y，不需要中间变量
2. **数值稳定**：内部使用 log-sum-exp 技巧避免 exp 溢出（先减最大值再取 exp）


BatchNorm 反传中，x_i 通过哪三条路径影响 loss？
?
1. **直接路径**：x_i → x̂_i → y_i（标准化公式的分子）
2. **通过均值 μ**：x_i → μ → x̂_i → y_i
3. **通过方差 σ²**：x_i → σ² → x̂_i → y_i
三条路径的梯度必须累加


BatchNorm 反传中，为什么 ∂σ²/∂μ = 0？
?
$$\frac{\partial \sigma^2}{\partial \mu} = \frac{-2}{m-1}\sum_i(x_i - \mu) = 0$$
因为均值的定义就是让 $\sum(x_i - \mu) \equiv 0$。
所以虽然 μ 出现在方差公式里，但方差对均值的一阶导数恒为零——均值路径和方差路径互不干扰


BatchNorm 合并反传公式中，三项分别对应什么？
?
$$\frac{\partial L}{\partial x_i} = \frac{\gamma \cdot (\sigma^2+\epsilon)^{-1/2}}{m}\left[\underbrace{m \cdot \frac{\partial L}{\partial y_i}}_{\text{直接反传}} - \underbrace{\sum_j \frac{\partial L}{\partial y_j}}_{\text{减去梯度均值（中心化）}} - \underbrace{\frac{m}{m-1}\hat{x}_i \sum_j \frac{\partial L}{\partial y_j}\hat{x}_j}_{\text{减去与 x̂ 相关分量（标准化）}}\right]$$
直觉：BN 反传在做**对梯度的"标准化"**——与前向传播对称


logits_maxes 的梯度为什么需要 one-hot scatter？
?
`logits_maxes = logits.max(1).values`，max 操作只从每行选了**最大值那一个**元素，
所以反传时梯度只流回那一个位置，其余位置梯度为 0。
用 `one_hot_max = zeros.scatter(1, max_indices, 1)` 构造 mask，乘以上游梯度实现选择


dlogits 可视化热力图中，为什么每行恰好只有一个黑点？黑点代表什么？
?
黑点 = 正确类别位置，梯度值为 $(P_y - 1)/n$（负数，所以在灰度图中显示为黑）。
含义：梯度下降时 $l_y \leftarrow l_y - lr \times (\text{负数})$，即**把正确答案的 logit 往上推**。
每行只有一个黑点是因为每个样本只有一个正确答案


Softmax+CE 合并反传公式的推导，$i=y$ 和 $i \neq y$ 分别怎么得来的？
?
令 $f = e^{l_y}/\sum e^{l_j}$，$\text{loss} = -\log f$：
- **$i \neq y$**：$e^{l_y}$ 不含 $l_i$（分子导数=0），只有分母贡献 → 结果 = $P_i$
- **$i = y$**：分子和分母都含 $l_i$，用乘积法则 → 结果 = $P_i - 1$
关键技巧：把分式写成 $e^{l_y} \cdot (\sum e^{l_j})^{-1}$ 再用乘积法则+幂法则


"先拆后合"的训练方法论是什么？为什么不直接用合并公式？
?
先把前向传播**拆成 20+ 原子操作**，逐个推导梯度并用 `cmp()` 验证（理解阶段）；
再把多步**合并为一行**高效公式（工程阶段）。
如果直接用合并公式，出了 bug 不知道该怀疑哪里——只有亲手实现过原子操作，才知道合并公式里每一项的来历


用手动梯度训练时，为什么可以用 `torch.no_grad()` 上下文管理器？
?
手动反传不需要 PyTorch 构建计算图（不调用 `loss.backward()`），
所以用 `torch.no_grad()` 关闭自动求导追踪可以**节省内存和加速**——
前向传播中不会记录任何操作到计算图中，因为梯度全由你自己算
