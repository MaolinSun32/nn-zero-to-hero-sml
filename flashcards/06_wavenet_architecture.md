---
tags:
---
# Lecture 6: WaveNet 风格的层级融合网络
#flashcards/nn-zero-to-hero


WaveNet 风格的层级融合和 Lecture 4 的 MLP 在处理上下文时有什么本质区别？
?
- **MLP**：把所有嵌入向量一次性拼接成长向量 `(30,)`，全连接处理——所有字符地位平等，丢失了局部顺序信息
- **WaveNet**：相邻字符先两两融合，再逐层向上汇聚，形成二叉树结构——先学局部模式（如 "th"），再组合成全局模式（如 "tion"）
类比：MLP 像让 8 个人同时发言；WaveNet 像淘汰赛，两两对决逐轮汇总
<!--SR:!2026-04-05,3,250-->


`FlattenConsecutive(2)` 对形状为 `(B, 8, 24)` 的张量做了什么？经过 3 次 FC(2) 后形状如何变化？
?
将相邻 2 个 token 的特征拼接起来：序列长度减半，特征维度翻倍。
`(B, 8, 24)` → `(B, 4, 48)` → `(B, 2, 400)` → `(B, 1, 400)` → squeeze → `(B, 400)`
第 3 次后 T=1，squeeze 掉序列维度变成 2D
<!--SR:!2026-04-05,3,250-->


`block_size` 和 `FlattenConsecutive(2)` 的层数之间是什么关系？为什么 block_size=8 最多只能用 3 层 FC(2)？
?
层数 = $\log_2(\text{block\_size})$。
每次 FC(2) 将序列长度减半：$8 \to 4 \to 2 \to 1$，3 次后就到底了。
第 4 次 FC(2) 会拿到 2D 张量，`B, T, C = x.shape` 解包会报错（只有 2 个维度）
<!--SR:!2026-04-05,3,250-->


`FlattenConsecutive` 和 1D 卷积（kernel_size=2, stride=2）有什么关系？
?
本质做的事一样：都是把相邻元素的特征拼起来再做线性变换。
区别在于卷积用**共享权重的滑动窗口**，FC 是**不重叠的分组**。
WaveNet 论文正是用空洞卷积（dilated convolution）实现类似的层级融合
<!--SR:!2026-04-05,3,250-->


为什么 BN 处理 3D 输入 `(B, T, C)` 时用 `dim=(0,1)` 而不是 `dim=0`？
?
3D 张量有 batch 维和序列维，都不是特征维。BN 需要对**特征维以外**的所有维度求统计量：
- 2D `(B, C)`：`dim=0`（只跨 batch）
- 3D `(B, T, C)`：`dim=(0,1)`（跨 batch 和序列位置）
保证每个特征通道 $c$ 有一个标量均值和方差
<!--SR:!2026-04-05,3,250-->


为什么模型构建时最后一层的权重要乘以 0.1（`model.layers[-1].weight *= 0.1`）？
?
让初始 logits 接近 0 → softmax 输出接近均匀分布 $1/27$ → 初始 loss $\approx -\log(1/27) \approx 3.30$。
如果权重太大，初始 logits 极端 → softmax "自信地猜错" → loss 爆炸，训练不稳定
<!--SR:!2026-04-05,3,250-->


train loss 下降但 dev loss 上升，说明什么？在这个实验中何时出现的？
?
典型的**过拟合**信号：模型用多余容量记忆训练数据而非学习泛化规律。
实验中在"增加第 4 层"后出现：train 1.76→1.64（↓），dev 2.00→2.03（↑），gap 从 0.24 扩大到 0.39
<!--SR:!2026-04-05,3,250-->


`FlattenConsecutive.__call__` 中 `squeeze(1)` 的触发条件和作用？
?
当 `T // n == 1` 时触发（序列维度只剩 1 个元素）。
作用：去掉多余的序列维，`(B, 1, C*n)` → `(B, C*n)`，变成 2D 张量。
这样后续 Linear 层既能处理 3D（中间层）也能处理 2D（最后一层）
<!--SR:!2026-04-05,3,250-->


`Sequential` 容器封装的好处是什么？
?
1. **调用简化**：`model(Xb)` 一行代替手动循环多个层
2. **参数收集**：`model.parameters()` 自动汇总所有层的可学习参数
3. **架构解耦**：改模型结构只改 Sequential 的列表，训练代码一行不用改
<!--SR:!2026-04-05,3,250-->


---


写出 `FlattenConsecutive.__call__` 的完整实现（包括 squeeze 逻辑）：
?
```python
def __call__(self, x):
    B, T, C = x.shape
    x = x.view(B, T // self.n, C * self.n)
    if x.shape[1] == 1:
        x = x.squeeze(1)
    self.out = x
    return self.out
```
`view` 把相邻 n 个 token 拼接；`squeeze` 在序列维只剩 1 时去掉它
<!--SR:!2026-04-05,3,250-->


写出 `Embedding` 层的前向传播实现（输入是索引张量 IX）：
?
```python
def __call__(self, IX):
    self.out = self.weight[IX]
    return self.out
```
利用 PyTorch 高级索引：`(27, 24)[IX]` 中 IX 形状 `(B, 8)` → 输出 `(B, 8, 24)`
<!--SR:!2026-04-05,3,250-->


用 `Sequential` 构建一个 block_size=8 的 WaveNet 模型（3 层 FC + 1 层普通 Linear + 输出层）：
?
```python
model = Sequential([
    Embedding(vocab_size, n_embd),
    FlattenConsecutive(2), Linear(n_embd*2, n_hidden, bias=False), BatchNorm1d(n_hidden), Tanh(),
    FlattenConsecutive(2), Linear(n_hidden*2, n_hidden, bias=False), BatchNorm1d(n_hidden), Tanh(),
    FlattenConsecutive(2), Linear(n_hidden*2, n_hidden, bias=False), BatchNorm1d(n_hidden), Tanh(),
    Linear(n_hidden, n_hidden, bias=False), BatchNorm1d(n_hidden), Tanh(),
    Linear(n_hidden, vocab_size)
])
```
3 组 FC(2) 对应 $8 \to 4 \to 2 \to 1$；第 4 组无 FC 因为已是 2D
<!--SR:!2026-04-05,3,250-->


训练完成后，如何切换 BN 层到推理模式？
?
```python
for layer in model.layers:
    layer.training = False
```
遍历所有层，将 `training` 标志设为 False。BN 改用 EMA 积累的 `running_mean/var`，不再用当前 batch 统计量
<!--SR:!2026-04-05,3,250-->
