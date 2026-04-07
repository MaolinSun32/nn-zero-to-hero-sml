---
tags:
---
# Micrograd — 从零构建自动微分引擎
#flashcards/nn-zero-to-hero


导数（derivative）的极限定义公式是什么？
?
$f'(x) = \lim_{h \to 0} \frac{f(x+h) - f(x)}{h}$
即：当自变量产生微小变化 h 时，函数值变化量与 h 的比值在 h 趋近于 0 时的极限
<!--SR:!2026-04-29,22,250-->


在函数的极小值点（minimum），导数等于多少？为什么？
?
导数等于 0。因为极小值点处函数既不增也不减，切线水平，斜率为零
<!--SR:!2026-05-01,24,250-->


用 Python 数值近似求导数的代码模式是什么？
?
```python
h = 0.0001
slope = (f(x + h) - f(x)) / h
```
用一个很小的 h 代替极限中的无穷小
<!--SR:!2026-04-26,19,250-->


偏导数（partial derivative）的核心思想是什么？
?
固定其他所有变量不变，只让一个变量产生微小变化，观察输出的变化率。例如 `d = a*b + c`，则 `dd/da = b`，`dd/db = a`，`dd/dc = 1`
<!--SR:!2026-04-30,23,250-->


Value 类中存储了哪五个关键属性？各自的作用是什么？
?
1. `self.data` — 存储标量数值
2. `self.grad` — 梯度值（初始为 0.0，backward 填充）
3. `self._backward` — 一个闭包（closure），计算局部梯度传播
4. `self._prev` — 父节点集合（产生该节点的输入）
5. `self._op` — 操作符字符串（'+', '*', 'tanh' 等），用于可视化
<!--SR:!2026-04-28,21,250-->


为什么 Value 类的 `_backward` 中梯度要用 `+=` 而不是 `=`？
?
因为一个节点可能被多条路径使用（multivariate chain rule）。用 `+=` 才能把来自所有路径的梯度正确**累加**（accumulate），否则后来的路径会覆盖前面的梯度
<!--SR:!2026-04-29,22,250-->


Value 加法运算的 backward 规则是什么？
?
`out = a + b` 时：
`a.grad += 1.0 * out.grad`
`b.grad += 1.0 * out.grad`
加法的局部导数都是 1.0，乘以上游梯度 out.grad
<!--SR:!2026-04-27,20,250-->


Value 乘法运算的 backward 规则是什么？
?
`out = a * b` 时：
`a.grad += b.data * out.grad`
`b.grad += a.data * out.grad`
对 a 求导得 b，对 b 求导得 a，再分别乘以上游梯度
<!--SR:!2026-05-01,24,250-->


Value 幂运算的 backward 规则是什么？
?
`out = a ** n` 时：
`a.grad += n * a.data**(n-1) * out.grad`
即幂函数求导法则 $\frac{d}{da}a^n = n \cdot a^{n-1}$，再乘以上游梯度
<!--SR:!2026-04-26,19,250-->


tanh 激活函数的公式和导数分别是什么？
?
公式：$\tanh(x) = \frac{e^{2x} - 1}{e^{2x} + 1}$
导数：$\frac{d}{dx}\tanh(x) = 1 - \tanh^2(x)$
所以 backward 中：`a.grad += (1 - t**2) * out.grad`
<!--SR:!2026-04-30,23,250-->


exp 运算的 backward 规则为什么可以直接用 `out.data`？
?
因为 $\frac{d}{dx}e^x = e^x$，而 `out.data` 就是 $e^x$ 的值，所以：
`a.grad += out.data * out.grad`
<!--SR:!2026-04-28,21,250-->


Python 中 `__radd__` 和 `__rmul__` 的作用是什么？在 Value 类中为什么需要它们？
?
当左操作数不支持该运算时（如 `2 + value`），Python 会调用右操作数的 `__radd__`。Value 类需要它们来处理 `int/float + Value` 的情况，否则 `2 + a` 会报错
<!--SR:!2026-04-29,22,250-->


Value 类中的 type coercion 模式是什么？为什么需要？
?
`other = other if isinstance(other, Value) else Value(other)`
在每个运算符方法开头将普通数字包装成 Value 对象，这样 `a + 2` 和 `a * 3` 等混合运算才能正常工作
<!--SR:!2026-04-27,20,250-->


backpropagation 算法的两个核心步骤是什么？
?
**Step 1 — Topological sort**：用 DFS 构建拓扑排序，确保每个节点在其所有子节点之后
**Step 2 — Reverse traversal**：设 `self.grad = 1.0`，然后按拓扑排序的逆序依次调用每个节点的 `_backward()`
<!--SR:!2026-05-01,24,250-->


为什么 backward 开始时要设 `self.grad = 1.0`？
?
因为输出节点对自身的导数 $\frac{\partial L}{\partial L} = 1$，这是链式法则（chain rule）的起点，所有后续梯度都从这个 1.0 开始向后传播
<!--SR:!2026-04-26,19,250-->


拓扑排序（topological sort）在 backpropagation 中的作用是什么？
?
保证在计算某个节点的梯度之前，所有依赖它的下游节点的梯度已经计算完毕。这样每个节点的 `out.grad`（上游梯度）在调用 `_backward()` 时已经是完整的
<!--SR:!2026-04-30,23,250-->


拓扑排序的 DFS 实现代码模式？
?
```python
topo = []
visited = set()
def build_topo(v):
    if v not in visited:
        visited.add(v)
        for child in v._prev:
            build_topo(child)
        topo.append(v)
build_topo(root)
```
后序遍历：先递归处理所有子节点，最后才 append 自己
<!--SR:!2026-04-28,21,250-->


单个神经元（Neuron）的前向计算公式是什么？
?
$y = \tanh(w_1 x_1 + w_2 x_2 + \cdots + w_n x_n + b)$
即所有输入的加权求和加上偏置（bias），再通过 tanh 激活函数
<!--SR:!2026-04-29,22,250-->


`sum((wi*xi for wi,xi in zip(self.w, x)), self.b)` 中第二个参数 `self.b` 的作用是什么？
?
`sum()` 的第二个参数是**起始值**（start value），所以实际计算的是 `b + w1*x1 + w2*x2 + ...`，巧妙地把 bias 加入了求和过程
<!--SR:!2026-04-27,20,250-->


为什么神经网络需要激活函数（activation function）如 tanh？
?
没有非线性激活函数，多层线性变换的堆叠等价于单层线性变换（矩阵乘法可以合并）。tanh 引入**非线性**（nonlinearity），使网络能逼近复杂的非线性函数
<!--SR:!2026-05-01,24,250-->


Layer 类的结构是什么？
?
一个 Layer 包含 `nout` 个并行的 Neuron，每个 Neuron 接收相同的 `nin` 个输入。
当 `nout == 1` 时返回单个 Value（而非列表），方便使用
<!--SR:!2026-04-26,19,250-->


MLP 的构造函数 `MLP(3, [4, 4, 1])` 创建了怎样的网络结构？
?
`sz = [3, 4, 4, 1]`，创建三个 Layer：
- `Layer(3, 4)` — 3 个输入，4 个神经元
- `Layer(4, 4)` — 4 个输入，4 个神经元
- `Layer(4, 1)` — 4 个输入，1 个神经元
输入依次流过各层
<!--SR:!2026-04-30,23,250-->


如何计算 `MLP(3, [4, 4, 1])` 的参数总数？
?
每个 Neuron 有 `nin` 个权重 + 1 个 bias：
- Layer(3,4): 4 × (3+1) = **16**
- Layer(4,4): 4 × (4+1) = **20**
- Layer(4,1): 1 × (4+1) = **5**
总计：16 + 20 + 5 = **41** 个参数
<!--SR:!2026-04-28,21,250-->


MSE（Mean Squared Error）损失函数的公式和代码？
?
公式：$L = \sum_i (y_{pred,i} - y_{target,i})^2$
代码：
```python
loss = sum((yout - ygt)**2 for yout, ygt in zip(ypreds, ys))
```
`**2` 和 `sum` 都是 Value 运算，所以计算图自动延伸到 loss
<!--SR:!2026-04-29,22,250-->


训练循环（training loop）的五个步骤是什么？
?
1. **Forward pass**：计算所有样本的预测值
2. **Compute loss**：计算 MSE 损失
3. **Zero gradients**：`p.grad = 0.0`（必须在 backward 前清零）
4. **Backward pass**：`loss.backward()` 计算所有梯度
5. **Gradient descent update**：`p.data += -lr * p.grad`
<!--SR:!2026-04-27,20,250-->


为什么每次 backward 之前必须把梯度清零（zero gradients）？
?
因为 `_backward` 中使用 `+=` 累加梯度。如果不清零，新一轮的梯度会叠加到上一轮的旧梯度上，导致梯度不正确
<!--SR:!2026-05-01,24,250-->


梯度下降（gradient descent）更新公式中为什么是负号？
?
`p.data += -lr * p.grad`
梯度指向 loss **增大**最快的方向。要**减小** loss，需要沿梯度的**反方向**移动，所以加负号
<!--SR:!2026-04-26,19,250-->


学习率（learning rate）太大或太小分别会怎样？
?
- **太大**：参数更新步幅过大，loss 来回震荡（oscillation），无法收敛
- **太小**：参数更新步幅过小，收敛极慢，训练效率低
<!--SR:!2026-04-30,23,250-->


为什么可以用分解后的基本运算替代 tanh 而得到相同梯度？
?
`o = ((2*n).exp() - 1) / ((2*n).exp() + 1)` 与直接调用 `tanh` 的数学结果相同。autograd 引擎无论计算图的粒度如何（粗粒度的 tanh 或细粒度的 exp/add/div），只要每个操作的局部导数正确，链式法则就能给出相同的最终梯度
<!--SR:!2026-04-28,21,250-->


如何用 PyTorch 验证自定义 Value 类的梯度？
?
```python
x = torch.Tensor([2.0]).double()
x.requires_grad = True
o = torch.tanh(x * w + b)
o.backward()
print(x.grad.item())  # .item() 取出 Python float
```
对比 Value 和 PyTorch 的梯度值是否一致
<!--SR:!2026-04-25,18,250-->


`__repr__` 和 `__str__` 的区别是什么？
?
- `__repr__`：给开发者看的，应精确无歧义，REPL 和 `repr()` 调用
- `__str__`：给用户看的，应可读，`print()` 和 `str()` 调用
- 如果只实现一个，优先实现 `__repr__`，因为 Python 在没有 `__str__` 时会 fallback 到 `__repr__`
<!--SR:!2026-04-29,22,250-->


Python 中单下划线 `_name` 和双下划线 `__name__` 命名的区别？
?
- `_name`（单下划线前缀）：约定俗成的"私有"属性，表示内部使用，不建议外部访问
- `__name__`（双下划线包裹，dunder）：Python 魔术方法（magic method），有特殊语法含义，如 `__init__`、`__add__`、`__call__`
<!--SR:!2026-04-27,20,250-->


`__call__` 魔术方法的作用是什么？在神经网络中如何使用？
?
定义了 `__call__` 的对象可以像函数一样被调用。
在 Neuron 类中定义 `__call__(self, x)` 后，可以直接写 `neuron(x)` 来执行前向传播，而不需要 `neuron.forward(x)`
<!--SR:!2026-04-24,17,250-->


Neuron 类的权重和偏置如何初始化？
?
```python
self.w = [Value(np.random.uniform(-1, 1)) for _ in range(nin)]
self.b = Value(np.random.uniform(-1, 1))
```
权重和偏置都从 `[-1, 1]` 均匀分布中随机采样
<!--SR:!2026-05-01,24,250-->


链式法则（chain rule）的核心思想是什么？在 autograd 中如何体现？
?
若 $y = f(g(x))$，则 $\frac{dy}{dx} = \frac{dy}{dg} \cdot \frac{dg}{dx}$
在 autograd 中，每个节点的 `_backward` 将上游梯度 `out.grad`（$\frac{dL}{dy}$）乘以局部导数（$\frac{dy}{dx}$），逐层传播回输入节点
<!--SR:!2026-04-26,19,250-->


computation graph 可视化中，`trace` 函数和 `draw_dot` 函数各自的作用？
?
- `trace(root)`：从根节点 DFS 遍历，返回 `(nodes, edges)` 集合
- `draw_dot(root)`：用 graphviz 将 Value 画成 record 形状节点（显示 label | data | grad），操作符画成圆形节点，箭头从输入指向操作再指向输出，布局为从左到右（`rankdir='LR'`）
<!--SR:!2026-04-25,18,250-->
