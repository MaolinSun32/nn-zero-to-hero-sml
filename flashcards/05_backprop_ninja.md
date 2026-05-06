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
<!--SR:!2026-05-07,19,250-->


`cmp()` 中 exact 和 approximate 分别检查什么？为什么 exact=False 但 approximate=True 仍说明推导正确？
?
- **exact**：`torch.all(dt == t.grad)`，bit-for-bit 完全相等
- **approximate**：`torch.allclose(dt, t.grad)`，允许浮点误差（|a-b| ≤ 1e-8 + 1e-5×|b|）
手动反传和 autograd 的计算路径不同，浮点舍入会导致极微小差异，所以 exact 可能为 False 但 approximate 为 True——推导仍然正确
<!--SR:!2026-05-10,22,250-->


为什么需要对中间变量调用 `retain_grad()`？
?
PyTorch 默认只保留**叶子节点**（参数）的 `.grad`，中间变量的梯度在 `backward()` 后会被释放以省内存。
调用 `retain_grad()` 使中间变量的梯度保留下来，方便用 `cmp()` 对比验证手动梯度
<!--SR:!2026-05-09,21,250-->


同一个变量被多条计算路径使用时，梯度怎么处理？
?
**梯度累加**。例如 `counts` 同时参与了 `probs = counts * counts_sum_inv` 和 `counts_sum = counts.sum(1)`，
反传时要把两条路径的梯度加起来：`dcounts = dcounts_1 + dcounts_2`
<!--SR:!2026-05-12,24,250-->


矩阵乘法 Y = A @ B 的反传梯度公式？
?
$$\frac{\partial L}{\partial A} = \frac{\partial L}{\partial Y} @ B^T, \qquad \frac{\partial L}{\partial B} = A^T @ \frac{\partial L}{\partial Y}$$
记忆技巧：结果形状必须和原矩阵匹配，转置哪个由维度对齐决定
<!--SR:!2026-05-08,20,250-->


前向传播中 `sum(0)` 把 (32,64) → (1,64)，反传时梯度怎么走？反过来呢？
?
- **前向 sum → 反向广播复制**：梯度从 (1,64) 广播回 (32,64)，每行得到同样的梯度
- **前向广播 → 反向 sum**：梯度沿广播维度求和
这是因为 sum 和 broadcast 互为逆运算
<!--SR:!2026-05-11,23,250-->


loss = -mean(logprobs[range(n), Yb]) 对 logprobs 的梯度是什么？为什么？
?
dlogprobs 是一个全零矩阵 (32,27)，只在 `[i, Yb[i]]` 位置为 **-1/n**。
因为 loss 只用到了每行中正确类别那一个元素的 log 概率，其余位置的偏导为 0；
负号来自 `-log`，1/n 来自 `mean`
<!--SR:!2026-05-07,19,250-->


tanh(x) 的导数是什么？如果已知 h = tanh(x)，如何用 h 表示导数？
?
$$\frac{d}{dx}\tanh(x) = 1 - \tanh^2(x) = 1 - h^2$$
代码：`dhpreact = (1 - h**2) * dh`
这避免了重新计算 tanh，直接用前向传播已有的 h
<!--SR:!2026-05-10,22,250-->


嵌入层反传为什么用 `index_add_` 而不是直接赋值？
?
同一个字符可能在 batch 中**出现多次**（比如字符 'a' 可能出现 10 次），
每次出现都产生一份梯度，这些梯度需要**累加**到 dC 的同一行上。
`dC.index_add_(0, Xb.view(-1), demb.view(-1, n_embd))` 就是按索引累加
<!--SR:!2026-05-09,21,250-->


Softmax + Cross-Entropy 合并反传的公式是什么？
?
$$\frac{\partial\,\text{loss}}{\partial\, l_i} = \begin{cases} P_i & i \neq y \\ P_i - 1 & i = y \end{cases}$$
向量形式：$\nabla_{\mathbf{l}}\text{loss} = \mathbf{P} - \mathbf{e}_y$
代码：`dlogits = softmax(logits); dlogits[range(n), Yb] -= 1; dlogits /= n`
<!--SR:!2026-05-12,24,250-->


为什么 PyTorch 的 `F.cross_entropy` 接收 logits 而不是 probs？
?
将 softmax 和 cross-entropy 合并计算有两个好处：
1. **梯度简洁**：合并后梯度 = P - e_y，不需要中间变量
2. **数值稳定**：内部使用 log-sum-exp 技巧避免 exp 溢出（先减最大值再取 exp）
<!--SR:!2026-05-08,20,250-->


BatchNorm 反传中，x_i 通过哪三条路径影响 loss？
?
1. **直接路径**：x_i → x̂_i → y_i（标准化公式的分子）
2. **通过均值 μ**：x_i → μ → x̂_i → y_i
3. **通过方差 σ²**：x_i → σ² → x̂_i → y_i
三条路径的梯度必须累加
<!--SR:!2026-06-16,41,250-->


BatchNorm 反传中，为什么 ∂σ²/∂μ = 0？
?
$$\frac{\partial \sigma^2}{\partial \mu} = \frac{-2}{m-1}\sum_i(x_i - \mu) = 0$$
因为均值的定义就是让 $\sum(x_i - \mu) \equiv 0$。
所以虽然 μ 出现在方差公式里，但方差对均值的一阶导数恒为零——均值路径和方差路径互不干扰
<!--SR:!2026-05-11,23,250-->


BatchNorm 合并反传公式中，三项分别对应什么？
?
$$\frac{\partial L}{\partial x_i} = \frac{\gamma \cdot (\sigma^2+\epsilon)^{-1/2}}{m}\left[\underbrace{m \cdot \frac{\partial L}{\partial y_i}}_{\text{直接反传}}- \underbrace{\sum_j \frac{\partial L}{\partial y_j}}_{\text{减去梯度均值（中心化）}}- \underbrace{\frac{m}{m-1}\hat{x}_i \sum_j \frac{\partial L}{\partial y_j}\hat{x}_j}_{\text{减去与 x̂ 相关分量（标准化）}}\right]$$
直觉：BN 反传在做**对梯度的"标准化"**——与前向传播对称
<!--SR:!2026-05-07,19,250-->



logits_maxes 的梯度为什么需要 one-hot scatter？
?
`logits_maxes = logits.max(1).values`，max 操作只从每行选了**最大值那一个**元素，
所以反传时梯度只流回那一个位置，其余位置梯度为 0。
用 `one_hot_max = zeros.scatter(1, max_indices, 1)` 构造 mask，乘以上游梯度实现选择
<!--SR:!2026-05-16,10,230-->


dlogits 可视化热力图中，为什么每行恰好只有一个黑点？黑点代表什么？
?
黑点 = 正确类别位置，梯度值为 $(P_y - 1)/n$（负数，所以在灰度图中显示为黑）。
含义：梯度下降时 $l_y \leftarrow l_y - lr \times (\text{负数})$，即**把正确答案的 logit 往上推**。
每行只有一个黑点是因为每个样本只有一个正确答案
<!--SR:!2026-05-10,22,250-->


Softmax+CE 合并反传公式的推导，$i=y$ 和 $i \neq y$ 分别怎么得来的？
?
令 $f = e^{l_y}/\sum e^{l_j}$，$\text{loss} = -\log f$：
- **$i \neq y$**：$e^{l_y}$ 不含 $l_i$（分子导数=0），只有分母贡献 → 结果 = $P_i$
- **$i = y$**：分子和分母都含 $l_i$，用乘积法则 → 结果 = $P_i - 1$
关键技巧：把分式写成 $e^{l_y} \cdot (\sum e^{l_j})^{-1}$ 再用乘积法则+幂法则
<!--SR:!2026-05-09,21,250-->


"先拆后合"的训练方法论是什么？为什么不直接用合并公式？
?
先把前向传播**拆成 20+ 原子操作**，逐个推导梯度并用 `cmp()` 验证（理解阶段）；
再把多步**合并为一行**高效公式（工程阶段）。
如果直接用合并公式，出了 bug 不知道该怀疑哪里——只有亲手实现过原子操作，才知道合并公式里每一项的来历
<!--SR:!2026-06-18,43,250-->


用手动梯度训练时，为什么可以用 `torch.no_grad()` 上下文管理器？
?
手动反传不需要 PyTorch 构建计算图（不调用 `loss.backward()`），
所以用 `torch.no_grad()` 关闭自动求导追踪可以**节省内存和加速**——
前向传播中不会记录任何操作到计算图中，因为梯度全由你自己算
<!--SR:!2026-05-12,24,250-->


---

## 代码实现

写出 Softmax + Cross-Entropy 合并反传的 3 行代码（已知 `logits` 形状 (n,27)，`Yb` 是目标索引）：
?
```python
dlogits = F.softmax(logits, 1)
dlogits[range(n), Yb] -= 1
dlogits /= n
```
<!--SR:!2026-05-08,20,250-->



已知前向 `logits = h @ W2 + b2`，写出 `dh`、`dW2`、`db2` 的反传代码：
?
```python
dh = dlogits @ W2.t()
dW2 = h.t() @ dlogits
db2 = dlogits.sum(0)
```
<!--SR:!2026-06-21,46,250-->


已知前向 `h = torch.tanh(hpreact)`，写出 `dhpreact` 的反传代码：
?
```python
dhpreact = (1 - h**2) * dh
```
关键：tanh 导数 = 1 - tanh²，直接用前向已有的 `h` 避免重新计算
<!--SR:!2026-05-11,23,250-->


已知前向 `hpreact = bngain * bnraw + bnbias`，写出 `dbngain` 和 `dbnbias` 的反传代码：
?
```python
dbngain = (dhpreact * bnraw).sum(0, keepdim=True)
dbnbias = dhpreact.sum(0, keepdim=True)
```
γ 的梯度需要乘以 x̂ 再沿 batch 维求和；β 的梯度直接沿 batch 维求和
<!--SR:!2026-05-07,19,250-->


写出 BatchNorm 合并反传的一行代码（从 `dhpreact` 得到 `dhprebn`），括号内有三项分别对应什么？
?
```python
dhprebn = bngain*bnvar_inv/n * (n*dhpreact - dhpreact.sum(0) - n/(n-1)*bnraw*(dhpreact*bnraw).sum(0))
```
三项：`n*dhpreact`（直接反传）、`-dhpreact.sum(0)`（减梯度均值 → μ 路径）、`-n/(n-1)*bnraw*(...)`（减与 x̂ 相关分量 → σ² 路径）
<!--SR:!2026-06-17,42,250-->


嵌入层反传：如何将梯度 `demb` (32,3,10) 累加回嵌入矩阵 `dC` (27,10)？写出代码：
?
```python
dC = torch.zeros_like(C)
dC.index_add_(0, Xb.view(-1), demb.view(-1, n_embd))
```
`Xb.view(-1)` 将 (32,3) 展平为 96 个索引；`demb.view(-1, n_embd)` 展平为 96 个梯度向量；同一字符出现多次时梯度自动累加
<!--SR:!2026-05-10,22,250-->


已知前向 `hprebn = embcat @ W1 + b1`，写出 `dembcat`、`dW1`、`db1` 的反传代码：
?
```python
dembcat = dhprebn @ W1.t()
dW1 = embcat.t() @ dhprebn
db1 = dhprebn.sum(0)
```
<!--SR:!2026-05-09,21,250-->


emb 形状 (32,3,10)，如何展平为 embcat 以及反传时如何还原？
?
```python
# 前向展平：(32,3,10) → (32,30)
embcat = emb.view(emb.shape[0], -1)
# 反传还原：(32,30) → (32,3,10)
demb = dembcat.view(emb.shape)
```
<!--SR:!2026-06-22,47,250-->


写出交叉熵损失的手动前向计算代码（NLL，已知 `logprobs` 形状 (n,27)，`Yb` 是目标索引）：
?
```python
loss = -logprobs[range(n), Yb].mean()
```
`range(n)` 选行，`Yb` 选列 → fancy indexing 取出每个样本正确类别的 log 概率
<!--SR:!2026-05-12,24,250-->


`logprobs` 的梯度是什么样的？写出代码：
?
```python
dlogprobs = torch.zeros_like(logprobs)
dlogprobs[range(n), Yb] = -1/n
```
全零矩阵 (n,27)，只在正确类别位置为 -1/n（负号来自 `-log`，1/n 来自 `mean`）
<!--SR:!2026-05-08,20,250-->


`probs` 的梯度怎么从 `dlogprobs` 得到？（已知前向 `logprobs = probs.log()`）
?
```python
dprobs = dlogprobs * (1/probs)
```
log 的导数是 1/x，再乘上游梯度（链式法则）
<!--SR:!2026-06-19,44,250-->


`counts` 同时参与了 `probs = counts * counts_sum_inv` 和 `counts_sum = counts.sum(1)`，写出两条路径的梯度及累加：
?
```python
dcounts_1 = dcounts_sum * torch.ones_like(counts)  # 来自 sum 路径
dcounts_2 = dprobs * counts_sum_inv                 # 来自 probs 路径
dcounts = dcounts_1 + dcounts_2                     # 多路径梯度累加
```
<!--SR:!2026-05-11,23,250-->


已知前向 `counts = norm_logits.exp()`，写出 `dnorm_logits` 的反传代码：
?
```python
dnorm_logits = dcounts * counts
```
exp 的导数是自身：d/dx(e^x) = e^x，所以局部导数就是 `counts` 本身
<!--SR:!2026-05-07,19,250-->


写出 BatchNorm 中方差逆平方根的计算代码（需要加 ε 防除零）：
?
```python
bnvar_inv = (bnvar + 1e-5)**-0.5
```
即 1/√(σ² + ε)，ε = 1e-5 防止除以零
<!--SR:!2026-06-20,45,250-->


已知前向 `bnvar_inv = (bnvar + 1e-5)^(-0.5)`，用幂法则写出 `dbnvar` 的反传代码：
?
```python
dbnvar = -0.5 * (bnvar + 1e-5)**(-1.5) * dbnvar_inv
```
幂法则：d/dx(x^n) = n·x^(n-1)，这里 n = -0.5
<!--SR:!2026-05-10,22,250-->


`max` 操作的反传中，如何用 `scatter` 构造 one-hot mask 使梯度只流回最大值位置？
?
```python
max_indices = logits.max(1, keepdim=True).indices
one_hot_max = torch.zeros_like(logits).scatter(1, max_indices, 1)
dlogits_2 = dlogits_maxes * one_hot_max
```
<!--SR:!2026-05-09,21,250-->


Kaiming 初始化中，针对 tanh 激活函数的缩放系数是什么？写出 W1 的初始化代码：
?
```python
W1 = torch.randn((fan_in, n_hidden)) * (5/3) / (fan_in**0.5)
```
5/3 是 tanh 的增益系数（补偿压缩效应），除以 √fan_in 保持输出方差稳定
<!--SR:!2026-06-16,41,250-->


写出 BatchNorm 前向传播的合并一行代码（从 `hprebn` 直接得到 `hpreact`）：
?
```python
hpreact = bngain * (hprebn - hprebn.mean(0, keepdim=True)) / torch.sqrt(hprebn.var(0, keepdim=True, unbiased=True) + 1e-5) + bnbias
```
<!--SR:!2026-05-12,24,250-->


`cmp()` 中精确匹配和近似匹配分别用什么 PyTorch 函数？
?
精确：`torch.all(dt == t.grad)` — 逐元素比较后检查全部为 True
近似：`torch.allclose(dt, t.grad)` — 允许浮点误差 |a-b| ≤ 1e-8 + 1e-5×|b|
<!--SR:!2026-05-08,20,250-->
