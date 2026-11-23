# **mini-llm 数学基础与神经网络原理**

第一部分：从数学基础理解大语言模型的计算机制

要理解大语言模型（LLM），不能只从 Transformer 开始。

Transformer 本质上是一个由大量数学运算构成的神经网络。它使用线性代数表示数据，使用微积分计算梯度，使用概率统计描述预测结果，再通过优化算法不断调整参数。

因此，我们需要先建立一套数学基础。

这部分文档分为六章：

| 章节 | 内容     | 对应 LLM 技术             |
| -- | ------ | --------------------- |
| 01 | 线性代数   | Tensor、Embedding、矩阵运算 |
| 02 | 微积分    | 梯度、反向传播、Autograd      |
| 03 | 概率与统计  | Token 概率、Softmax、交叉熵  |
| 04 | 优化理论   | SGD、Adam、模型训练         |
| 05 | 神经网络数学 | Linear、MLP、激活函数       |
| 06 | 综合练习   | 手算、代码验证、原理理解          |

# **第一章：线性代数（Linear Algebra）**

## **1.1 为什么大语言模型需要线性代数？**

先看一个问题：

假设我们希望 AI 理解一句话：

> 今天的天气非常好。

计算机并不能直接理解这句话。

它首先需要将文本转换成数字，然后通过数学运算处理这些数字。

例如，经过 Tokenizer 和 Embedding 后，可能得到这样的向量：

\\[ x = \begin{bmatrix} 0.12\\\ 0.35\\\ -0.21\\\ 0.87 \end{bmatrix} \\]

这个向量就是计算机内部表示文本的一种形式。

如果一句话包含多个 Token，就可以形成一个矩阵：

\\[ X = \begin{bmatrix} 0.12 & 0.35 & -0.21 & 0.87\\\ 0.42 & 0.18 & 0.63 & 0.11\\\ 0.27 & -0.15 & 0.31 & 0.72 \end{bmatrix} \\]

这里：

- 每一行代表一个 Token。
- 每一列代表向量的一个维度。
- 整个矩阵代表一段文本的向量表示。

Transformer 后续的大部分计算，都建立在这些矩阵之上。

所以，线性代数是理解 LLM 的基础。

## **1.2 标量、向量、矩阵和张量**

数学中有四种常见的数据结构。

**1. 标量（Scalar）**

# **5**

只有一个数值，没有维度。

**2. 向量（Vector）**

\\[ \mathbf{x}= \begin{bmatrix} 1\\\2\\\3 \end{bmatrix} \\]

一组有顺序的数字，可以表示一个对象的特征。

**3. 矩阵（Matrix）**

\\[ A= \begin{bmatrix} 1&2&3\\\ 4&5&6 \end{bmatrix} \\]

由行和列组成的二维数字结构。

**4. 张量（Tensor）**

\\[ T\in\mathbb{R}^{2\times3\times4} \\]

可以理解为多维数组，是深度学习中最常见的数据容器。

在 LLM 中，通常可以这样理解：

| 数据结构 | LLM 中的例子              |
| ---- | --------------------- |
| 标量   | Loss、学习率              |
| 向量   | 一个 Token 的 Embedding  |
| 矩阵   | 一层 Linear 的权重         |
| 三维张量 | 一批文本的 Token Embedding |
| 四维张量 | 多头 Attention 的 Q/K/V  |

例如：

\\[ X\in\mathbb{R}^{B\times S\times D} \\]

其中：

- \\(B\\)：Batch Size，一次训练多少条样本。
- \\(S\\)：Sequence Length，每条样本包含多少个 Token。
- \\(D\\)：Hidden Size，每个 Token 的向量维度。

假设：

\\[ B=2,\quad S=3,\quad D=4 \\]

那么：

\\[ X\in\mathbb{R}^{2\times3\times4} \\]

意味着一次输入两条文本，每条文本三个 Token，每个 Token 使用四维向量表示。

**关键理解：Tensor 是深度学习中承载数据和参数的基本结构。**

## **1.3 向量的点积（Dot Product）**

点积是 Transformer 中非常重要的运算。

假设有两个向量：

\\[ a= \begin{bmatrix} 1\\\2\\\3 \end{bmatrix} ,\quad b= \begin{bmatrix} 4\\\5\\\6 \end{bmatrix} \\]

它们的点积为：

\\[ a\cdot b=1\times4+2\times5+3\times6 \\]

\\[ =4+10+18=32 \\]

点积可以衡量两个向量在数值上的关联程度。

从几何角度：

\\[ a\cdot b=|a||b|\cos\theta \\]

其中：

- \\(|a|\\)：向量 a 的长度。
- \\(|b|\\)：向量 b 的长度。
- \\(\theta\\)：两个向量之间的夹角。

当两个向量方向相同时，点积较大；当两个向量正交时，点积为零；方向相反时，点积为负。

[ベクトルの内積とコサインの関係｜澁谷直樹](https://images.openai.com/static-rsc-4/cp0nw76vrYB4SaAbk5T0KwChRScZZYniJBYtI_7GiLN9lmgAqhDl-t-sxAYof4zmKcwbyKiJ--nxDH9o8wGFeWcRV49zxXgjANonB3QFsP5v7l59Ci_F1iMTiSF4uA-uUUv-E_q7UL34zZH8umU9YOrMGA9hp_DuxmW7uUj2pNo?purpose=inline)

[note.com](https://note.com/kikaben/n/n4a2124743027?utm_source=chatgpt.com)

### **点积与 Attention 的关系**

Transformer 中，Attention 需要计算 Query 和 Key 的相似程度：

\\[ Score(Q,K)=QK^T \\]

这里的矩阵乘法，本质上就是批量计算向量点积。

例如：

- Query 表示当前 Token 正在寻找什么信息。
- Key 表示其他 Token 可以提供什么信息。
- 点积用于计算二者的匹配程度。

因此，Attention 并不是凭空产生的机制，而是建立在向量点积之上的。

## **1.4 矩阵乘法（Matrix Multiplication）**

矩阵乘法是神经网络中最重要的计算之一。

假设：

\\[ A= \begin{bmatrix} 1&2\\\ 3&4 \end{bmatrix} \\]

\\[ B= \begin{bmatrix} 5&6\\\ 7&8 \end{bmatrix} \\]

那么：

\\[ C=AB \\]

计算：

\\[ C= \begin{bmatrix} 1\times5+2\times7 & 1\times6+2\times8\\\ 3\times5+4\times7 & 3\times6+4\times8 \end{bmatrix} \\]

最终：

\\[ C= \begin{bmatrix} 19&22\\\ 43&50 \end{bmatrix} \\]

### **为什么神经网络需要矩阵乘法？**

考虑一个简单的神经网络层：

\\[ Y=XW+b \\]

其中：

- \\(X\\)：输入数据。
- \\(W\\)：模型权重。
- \\(b\\)：偏置。
- \\(Y\\)：输出数据。

假设输入有三个特征，输出有两个特征：

\\[ X\in\mathbb{R}^{1\times3} \\]

\\[ W\in\mathbb{R}^{3\times2} \\]

\\[ b\in\mathbb{R}^{1\times2} \\]

那么：

\\[ Y=XW+b \\]

这就是神经网络中的 Linear 层。

Linear 层并不意味着输出一定与输入成比例，而是通过一组可学习的权重，将输入特征重新组合。

## **1.5 转置（Transpose）**

矩阵转置就是交换行和列。

例如：

\\[ A= \begin{bmatrix} 1&2&3\\\ 4&5&6 \end{bmatrix} \\]

转置之后：

\\[ A^T= \begin{bmatrix} 1&4\\\ 2&5\\\ 3&6 \end{bmatrix} \\]

原始矩阵：

\\[ A\in\mathbb{R}^{2\times3} \\]

转置后：

\\[ A^T\in\mathbb{R}^{3\times2} \\]

### **为什么 Attention 需要转置？**

假设：

\\[ Q\in\mathbb{R}^{S\times D} \\]

\\[ K\in\mathbb{R}^{S\times D} \\]

为了计算每个 Query 与所有 Key 的匹配程度：

\\[ Scores=QK^T \\]

维度：

\\[ (S\times D)(D\times S) \\]

结果：

\\[ Scores\in\mathbb{R}^{S\times S} \\]

这意味着每个 Token 都可以得到它与其他 Token 的关联分数。

## **1.6 向量范数（Norm）**

范数用于衡量向量的大小。

最常见的是 L2 范数：

\\[ \\|x\\|\_2=\sqrt{x_1^2+x_2^2+\cdots+x_n^2} \\]

例如：

\\[ x= \begin{bmatrix} 3\\\4 \end{bmatrix} \\]

那么：

\\[ \\|x\\|\_2=\sqrt{3^2+4^2}=5 \\]

范数在神经网络中的应用：

- 梯度裁剪。
- 权重正则化。
- 向量归一化。
- 相似度计算。
- LayerNorm 等归一化方法。

# **第二章：微积分（Calculus）**

线性代数告诉我们如何计算。

但神经网络还需要解决一个更重要的问题：

**如何让模型变得更准确？**

这就需要微积分。

## **2.1 什么是函数？**

函数描述输入和输出之间的关系。

例如：

\\[ y=f(x)=x^2 \\]

输入不同的 x，会得到不同的 y。

| x | y  |
| - | -- |
| 1 | 1  |
| 2 | 4  |
| 3 | 9  |
| 4 | 16 |

神经网络可以理解为一个非常复杂的函数：

\\[ y=f(x;\theta) \\]

其中：

- \\(x\\)：输入。
- \\(\theta\\)：模型参数。
- \\(y\\)：模型输出。

对于大语言模型：

\\[ Logits=f(Token;\theta) \\]

模型通过大量参数，将输入 Token 转换为下一个 Token 的预测分数。

## **2.2 什么是导数？**

导数描述函数变化的速度。

例如：

\\[ f(x)=x^2 \\]

它的导数为：

\\[ f'(x)=2x \\]

当：

\\[ x=3 \\]

那么：

\\[ f'(3)=6 \\]

这意味着在 x=3 附近，x 每增加一个很小的量，函数值大约增加该小量的 6 倍。

从几何角度看，导数就是曲线在某一点的切线斜率。

[Gilbert Strang’s Calculus: Big Picture on Derivatives – ∇ ickma.dev](https://images.openai.com/static-rsc-4/4CEhvjvlF6HJMZ-2ugoHl2Sz-YsojogmEzqmRr4D_oTXgAMGkYcm2cHC5mhgWT9tSEmRFTI5yGzBskv12bKOAmmEUljvmBjdjEf65myoZc1xbuvWynXaUoAPANaLWKkaPVW9DpEeuFFZSmxH49kFl7owSKN6jz6sCLAZfj9FUR8?purpose=inline)

[ickma2311.github.io](https://ickma2311.github.io/Math/Calculus/big-picture-derivatives.html?utm_source=chatgpt.com)

### **导数为什么对 AI 重要？**

因为模型需要知道：

> 修改一个参数，会让预测误差增加还是减少？

导数可以回答这个问题。

## **2.3 偏导数（Partial Derivative）**

神经网络通常有数百万甚至数十亿个参数。

假设一个模型的损失函数是：

\\[ L=f(w_1,w_2,w_3) \\]

它同时依赖三个参数。

那么我们可以分别计算：

\\[ \frac{\partial L}{\partial w_1} \\]

\\[ \frac{\partial L}{\partial w_2} \\]

\\[ \frac{\partial L}{\partial w_3} \\]

这就是偏导数。

它表示：

在其他参数保持不变的情况下，单独改变某个参数，Loss 会如何变化。

## **2.4 梯度（Gradient）**

将所有参数的偏导数组合起来，就得到梯度：

\\[ \nabla L= \begin{bmatrix} \frac{\partial L}{\partial w_1}\\\ \frac{\partial L}{\partial w_2}\\\ \frac{\partial L}{\partial w_3} \end{bmatrix} \\]

梯度是一个向量。

它有两个重要属性：

1. 梯度方向指向函数增长最快的方向。
2. 梯度大小描述函数在该方向上的局部变化速度。

如果我们希望减少 Loss，就应该沿着梯度的反方向调整参数。

这就是梯度下降的基础。

## **2.5 链式法则（Chain Rule）**

这是理解神经网络反向传播最重要的数学知识之一。

假设：

\\[ y=f(u) \\]

\\[ u=g(x) \\]

那么：

\\[ y=f(g(x)) \\]

根据链式法则：

\\[ \frac{dy}{dx} = \frac{dy}{du} \cdot \frac{du}{dx} \\]

例如：

\\[ u=x^2 \\]

\\[ y=3u+1 \\]

那么：

\\[ \frac{du}{dx}=2x \\]

\\[ \frac{dy}{du}=3 \\]

所以：

\\[ \frac{dy}{dx}=3\times2x=6x \\]

### **神经网络为什么需要链式法则？**

因为神经网络由很多层组成：

\\[ x\rightarrow Layer_1\rightarrow Layer_2\rightarrow Layer_3\rightarrow Loss \\]

Loss 是通过多个函数连续计算得到的。

我们希望知道每一层参数对最终 Loss 的影响。

这就必须将梯度从最后一层逐步传递到前面的层。

这就是反向传播（Backpropagation）。

[42. Backpropagation: How Neural Networks Learn | by Kiran vutukuri | Medium](https://images.openai.com/static-rsc-4/j2QITLyW3cmpxR-XHSzGVg3qvlvljk8FS2aZ5l-JKsw8KznEHmkX7UCkgh5S7odeKffx9LES3SmtAS3QPSwVgkpKGW_3RCuoxjJchTtkW0r11HUjkkPuAm6YHF8bWew3852gfGdTgRWOKYhguyu0i_RbJgVGGjT7QzjXeFpw_KI?purpose=inline)

[medium.com](https://medium.com/%40kiranvutukuri/42-backpropagation-how-neural-networks-learn-f4a4a1270dd5?utm_source=chatgpt.com)

## **2.6 反向传播的简单例子**

假设：

\\[ x=2 \\]

\\[ w=3 \\]

\\[ y=wx \\]

目标值：

\\[ y\_{target}=10 \\]

使用平方损失：

\\[ L=(y-y\_{target})^2 \\]

首先计算预测：

\\[ y=3\times2=6 \\]

损失：

\\[ L=(6-10)^2=16 \\]

接下来计算参数 w 的梯度：

\\[ \frac{\partial L}{\partial w} = \frac{\partial L}{\partial y} \cdot \frac{\partial y}{\partial w} \\]

因为：

\\[ \frac{\partial L}{\partial y}=2(y-10) \\]

\\[ \frac{\partial y}{\partial w}=x \\]

所以：

\\[ \frac{\partial L}{\partial w} =2(6-10)\times2=-16 \\]

梯度为 -16。

这意味着在当前位置，增加 w 会使 Loss 下降。

使用学习率：

\\[ \eta=0.1 \\]

梯度下降：

\\[ w\_{new}=w-\eta\frac{\partial L}{\partial w} \\]

\\[ w\_{new}=3-0.1(-16)=4.6 \\]

更新后：

\\[ y=4.6\times2=9.2 \\]

新的 Loss：

\\[ L=(9.2-10)^2=0.64 \\]

Loss 从 16 降低到 0.64

这就是神经网络学习的最基本过程。

# **第三章：概率与统计（Probability & Statistics）**

## **3.1 为什么大语言模型需要概率？**

假设用户输入：

> 今天天气非常……

模型需要预测下一个 Token。

可能的候选结果：

| Token | 概率   |
| ----- | ---- |
| 好     | 0.65 |
| 热     | 0.20 |
| 冷     | 0.10 |
| 糟糕    | 0.05 |

模型并不是简单地判断哪个词一定正确。

它实际上是在计算一个概率分布：

\\[ P(x\_{t+1}|x_1,x_2,\ldots,x_t) \\]

意思是：

在已经知道前面所有 Token 的情况下，下一个 Token 出现的条件概率。

这就是自回归语言模型的核心。

## **3.2 什么是概率分布？**

概率分布描述不同事件发生的可能性。

一个离散概率分布需要满足：

\\[ 0\leq P(x_i)\leq1 \\]

以及：

\\[ \sum_iP(x_i)=1 \\]

例如：

\\[ P=[0.2,0.3,0.5] \\]

三个概率之和为：

\\[ 0.2+0.3+0.5=1 \\]

在 LLM 中，Softmax 可以将模型输出的分数转换成概率分布。

## **3.3 Softmax**

假设模型输出三个 Logits：

\\[ z=[2,1,0] \\]

Softmax 公式：

\\[ Softmax(z_i)= \frac{e^{z_i}}{\sum_je^{z_j}} \\]

先计算指数：

\\[ e^2\approx7.389 \\]

\\[ e^1\approx2.718 \\]

\\[ e^0=1 \\]

总和：

\\[ 7.389+2.718+1=11.107 \\]

最终概率：

\\[ P\approx[0.665,0.245,0.090] \\]

模型将原始分数转换成了概率。

[RNNs | Why Positional embeddings Transformer Part-1 | by Asifrazartu | Medium](https://images.openai.com/static-rsc-4/5xRbRhnJTMXMDoXsE09B5Dl0TkGq9Ev0wEYyHFnlJEg1_14pIHHvgGGjCQoYyV4XtW7uqvV8I-UtM6Emu1XjjzLGjOronP1wcHq2QWYXHfonQUlVKmjGcViCFD5e-8-azi7f8x6J7tcW5EC6QLCoV6Y3Rn3lbwBYnQI98yb4Iug?purpose=inline)

[medium.com](https://medium.com/%40asifrazartu/rnns-why-positional-embeddings-transformer-part-1-3c9c7939ea6d?utm_source=chatgpt.com)

### **为什么不能直接使用 Logits？**

Logits 可以是任意实数：

\\[ [-5,2,10] \\]

它们不是概率。

Softmax 将它们转换成：

- 非负数。
- 总和为 1。
- 可以解释为一个离散概率分布。

注意：Softmax 的输出是模型定义的预测概率，不代表现实世界中已经得到校准的真实概率。

## **3.4 交叉熵（Cross Entropy）**

模型预测概率之后，还需要判断预测是否准确。

假设真实答案是：

```
天气很好
```

对于其中一个预测位置，正确 Token 是“好”。

模型预测：

\\[ P=[0.7,0.2,0.1] \\]

其中正确 Token 的概率是 0.7。

交叉熵：

\\[ L=-\sum_i y_i\log p_i \\]

其中：

- \\(y_i\\)：真实标签分布。
- \\(p_i\\)：模型预测概率。

如果正确答案对应第一个类别：

\\[ y=[1,0,0] \\]

那么：

\\[ L=-\log(0.7) \\]

\\[ L\approx0.357 \\]

如果模型只给正确答案 0.01 的概率：

\\[ L=-\log(0.01)\approx4.605 \\]

可见，模型越不相信正确答案，损失通常越大。

### **交叉熵为什么适合训练 LLM？**

语言模型训练的目标是：

> 让正确的下一个 Token 获得尽可能高的概率。

交叉熵正好衡量真实分布与预测分布之间的差异。

GPT 的基础训练目标通常可以表示为：

\\[ L=-\sum\_{t=1}^{T}\log P(x_t|x\_{\<t}) \\]

也就是让模型尽可能准确地预测训练文本中的每一个下一个 Token。

## **3.5 熵（Entropy）**

熵用于衡量概率分布的不确定性。

公式：

\\[ H(X)=-\sum_iP(x_i)\log P(x_i) \\]

例如：

分布 A：

\\[ P_A=[0.99,0.01] \\]

分布 B：

\\[ P_B=[0.5,0.5] \\]

A 的概率高度集中，预测比较确定。

B 的概率平均分散，预测更加不确定。

因此：

\\[ H(P_B)>H(P_A) \\]

### **熵与 LLM 的关系**

熵可以帮助理解：

- Token 预测的不确定性。
- 模型输出的随机性。
- Temperature 对采样的影响。
- 模型预测分布的集中程度。

注意：熵高不等于模型能力强，熵低也不等于模型能力弱。它描述的是概率分布的不确定性。

## **3.6 KL 散度（KL Divergence）**

KL 散度用于衡量两个概率分布之间的差异。

\\[ D\_{KL}(P\\|Q)= \sum_iP(x_i)\log\frac{P(x_i)}{Q(x_i)} \\]

其中：

- P：参考分布。
- Q：需要比较的分布。

KL 散度常用于：

- 知识蒸馏。
- VAE。
- 强化学习中的策略约束。
- DPO 等偏好优化方法的理论分析。

需要注意，KL 散度通常不对称：

\\[ D\_{KL}(P\\|Q)\neq D\_{KL}(Q\\|P) \\]

因此它不是普通几何意义上的距离。

# **第四章：优化理论（Optimization）**

## **4.1 什么是优化？**

训练神经网络的目标可以概括为：

\\[ \min\_\theta L(\theta) \\]

其中：

- \\(\theta\\)：模型所有可训练参数。
- \\(L\\)：损失函数。
- \\(\min\\)：寻找使 Loss 尽可能小的参数。

假设一个模型只有一个参数 w：

\\[ L(w)=(w-5)^2 \\]

最优点：

\\[ w=5 \\]

此时：

\\[ L(5)=0 \\]

但真实的大语言模型拥有大量参数，不可能直接通过简单公式求出最优解。

所以需要迭代优化算法。

## **4.2 梯度下降（Gradient Descent）**

基本公式：

\\[ \theta\_{t+1} = \theta_t-\eta\nabla L(\theta_t) \\]

其中：

- \\(\theta_t\\)：当前参数。
- \\(\eta\\)：学习率。
- \\(\nabla L\\)：当前梯度。
- \\(\theta\_{t+1}\\)：更新后的参数。

可以理解为：

1. 使用当前模型计算预测。
2. 计算 Loss。
3. 计算每个参数的梯度。
4. 根据梯度调整参数。
5. 重复以上过程。

## **4.3 SGD**

随机梯度下降（Stochastic Gradient Descent）通常使用一个 Mini-batch 估计梯度。

\\[ \theta\_{t+1}=\theta_t-\eta g_t \\]

这里的 \\(g_t\\) 是当前批次样本估计出来的梯度。

相比每次都使用整个训练集，SGD 更适合大规模数据训练。

## **4.4 Momentum**

普通 SGD 可能在优化过程中出现震荡。

Momentum 引入历史梯度：

\\[ v_t=\beta v\_{t-1}+g_t \\]

\\[ \theta_t=\theta\_{t-1}-\eta v_t \\]

它相当于给参数更新增加了惯性。

这样可以在方向相对稳定时加速，在梯度方向频繁变化时缓解震荡。

## **4.5 Adam**

Adam 同时维护梯度的一阶矩和二阶矩。

一阶矩：

\\[ m_t=\beta_1m\_{t-1}+(1-\beta_1)g_t \\]

二阶矩：

\\[ v_t=\beta_2v\_{t-1}+(1-\beta_2)g_t^2 \\]

偏差修正：

\\[ \hat m_t=\frac{m_t}{1-\beta_1^t} \\]

\\[ \hat v_t=\frac{v_t}{1-\beta_2^t} \\]

参数更新：

\\[ \theta_t= \theta\_{t-1} -\eta\frac{\hat m_t}{\sqrt{\hat v_t}+\epsilon} \\]

Adam 会根据历史梯度调整不同参数的更新幅度。

## **4.6 AdamW**

AdamW 将权重衰减与 Adam 的梯度更新解耦。

其常见更新形式：

\\[ \theta_t= \theta\_{t-1} -\eta\frac{\hat m_t}{\sqrt{\hat v_t}+\epsilon} -\eta\lambda\theta\_{t-1} \\]

其中：

- \\(\lambda\\)：权重衰减系数。
- \\(\eta\\)：学习率。

AdamW 是 Transformer 训练中常用的优化器之一。

## **4.7 学习率为什么重要？**

学习率决定每一次参数更新的幅度。

[Gradyan İniş (Gradient Descent): Makine Öğreniminde Optimizasyon | by Murat Kose | Devops & AI Türkiye☁️ 🐧 🐳 ☸️ | Medium](https://images.openai.com/static-rsc-4/TG4M38cMLQs_mEpv4uOO0264CtBaRP8WrYa0uuSapQBJ1Npjv2PndIlRCoHIwopxl_2kYgqsrSjSq2a5ORiYSfFk8vIC7SWEY5JwnBMIocfj8gl4zsAwqJynaE9qf6H0PRY09lihZ-bAEHeQPqCWBxbjeVf_06gLHUc6IaGOGG4?purpose=inline)

[medium.com](https://medium.com/devopsturkiye/gradient-descent-gradyan-i%CC%87ni%C5%9F-57548540e0a4?utm_source=chatgpt.com)

- 学习率太大：可能越过最优区域，甚至无法收敛。
- 学习率太小：训练速度非常慢。
- 学习率合适：能够逐步接近较好的参数区域。

常见学习率策略：

- Warmup：训练初期逐渐增大学习率。
- Cosine Decay：按照余弦曲线逐渐降低学习率。
- Constant：保持学习率不变。
- Step Decay：按照训练阶段降低学习率。

# **第五章：神经网络数学原理**

现在，我们将前面四章的知识组合起来。

## **5.1 神经网络是什么？**

神经网络可以看成一个由多个可学习函数组成的计算系统。

最简单的结构：

\\[ Input\rightarrow Hidden\rightarrow Output \\]

例如：

\\[ x\rightarrow Linear_1\rightarrow ReLU\rightarrow Linear_2\rightarrow y \\]

神经网络通过训练调整内部参数，使输出越来越接近目标结果。

[High-dimensional aerodynamic data modeling using a machine learning method based on a convolutional neural network | Advances in Aerodynamics | Springer Nature Link](https://images.openai.com/static-rsc-4/Ne1cBl0rzI0rW7XHiimfiBX_7YYAKoUq3gXAyhS7UySoYbzfiMVLhQp6LjqwMRGG4Dkkoi_A5VbtputeBuSH2HkUdiUSDkemQZgrlOA14OXiyuirFCom3KRPYwy_F8XMgKZCatL43sWJyWmKAznc_b3TSNLxp7U8npbykpVXZzM?purpose=inline)

[link.springer.com](https://link.springer.com/article/10.1186/s42774-022-00128-8?utm_source=chatgpt.com)

## **5.2 Linear 层**

Linear 层的公式：

\\[ y=xW+b \\]

假设：

\\[ x=[1,2] \\]

\\[ W= \begin{bmatrix} 2&1\\\ 3&4 \end{bmatrix} \\]

\\[ b=[1,1] \\]

计算：

\\[ xW=[1\times2+2\times3,\ 1\times1+2\times4] \\]

\\[ xW=[8,9] \\]

加入偏置：

\\[ y=[9,10] \\]

Linear 层的核心就是矩阵乘法和偏置加法。

### **Linear 层为什么需要可训练参数？**

因为 W 和 b 是可以通过训练更新的。

不同的 W 和 b 会产生不同的输入输出映射。

模型学习的过程，本质上就是寻找一组更合适的参数。

## **5.3 激活函数（Activation Function）**

如果神经网络只有 Linear 层：

\\[ y=xW_1W_2W_3 \\]

多个 Linear 层可以合并成一个 Linear 变换。

这意味着单纯堆叠 Linear 层无法形成足够丰富的非线性表达能力。

所以需要激活函数。

### **ReLU**

\\[ ReLU(x)=\max(0,x) \\]

例如：

\\[ x=[-2,-1,0,1,2] \\]

经过 ReLU：

\\[ y=[0,0,0,1,2] \\]

### **GELU**

GELU 的常见定义：

\\[ GELU(x)=x\Phi(x) \\]

其中 \\(\Phi(x)\\) 是标准正态分布的累积分布函数。

GELU 在 Transformer 的前馈网络中被广泛使用。

### **SiLU**

\\[ SiLU(x)=x\cdot sigmoid(x) \\]

其中：

\\[ sigmoid(x)=\frac{1}{1+e^{-x}} \\]

SiLU 也常见于现代神经网络架构。

## **5.4 MLP**

MLP（Multi-Layer Perceptron）是多层感知机。

一个简单 MLP：

\\[ h=Activation(xW_1+b_1) \\]

\\[ y=hW_2+b_2 \\]

它通过两次线性变换和一次非线性激活，完成输入特征到输出特征的转换。

### **MLP 在 Transformer 中的作用**

Transformer 不仅需要 Attention，还需要 MLP。

Attention 负责根据上下文混合不同 Token 的信息。

MLP 负责对每个 Token 的特征进行进一步的非线性变换。

二者配合，构成 Transformer Block 的核心计算结构。

## **5.5 神经网络如何学习？**

将前面知识串起来：

## **神经网络训练流程**

1. **输入数据**

   将文本转换为 Token，再通过 Embedding 转换成向量。
2. **前向传播**

   输入经过 Linear、Activation、Attention、MLP 等计算。
3. **计算预测结果**

   输出 Logits，并计算预测概率。
4. **计算 Loss**

   使用交叉熵等损失函数衡量预测结果。
5. **反向传播**

   使用链式法则计算各个参数的梯度。
6. **优化参数**

   使用 AdamW 等优化器更新参数。
7. **重复训练**

   多次迭代，让模型逐步学习数据中的规律。

# **第六章：把数学知识与 mini-llm 联系起来**

现在我们可以将数学基础映射到项目架构。

| 数学知识      | 对应模块           | 主要功能           |
| --------- | -------------- | -------------- |
| 向量、矩阵、张量  | Tensor Engine  | 存储和运算          |
| 矩阵乘法      | Linear         | 特征变换           |
| 导数、偏导数    | Autograd       | 计算梯度           |
| 链式法则      | Backward       | 反向传播           |
| 梯度下降      | Optimizer      | 更新参数           |
| 概率分布      | Softmax        | 生成预测概率         |
| 交叉熵       | Loss           | 衡量预测误差         |
| 非线性函数     | Activation     | 增强表达能力         |
| 多层函数组合    | Neural Network | 构建神经网络         |
| Attention | Transformer    | 建模 Token 之间的关系 |

## **6.1 mini-llm 的数学计算链**

**输入数据**

Token IDs / Tensor

**Tensor Engine**

加法、乘法、矩阵运算

**Neural Network**

Linear、Activation、MLP、Transformer

**Loss**

交叉熵、预测误差

**Autograd**

链式法则、梯度计算

**Optimizer**

SGD、Adam、AdamW

更新参数，进入下一轮训练

## **6.2 本章必须真正理解的十个问题**

不要急着进入 Transformer。先尝试回答以下问题。

**自测：理解这十个问题**

0/10 已查看答案

**1. 为什么神经网络使用矩阵乘法，而不是简单地逐个处理数字？**

查看答案

**2. 为什么 Attention 使用 QKᵀ？**

查看答案

**3. 导数和梯度有什么区别？**

查看答案

**4. 为什么反向传播需要链式法则？**

查看答案

**5. 为什么梯度下降使用负梯度？**

查看答案

**6. Softmax 和交叉熵分别解决什么问题？**

查看答案

**7. 为什么神经网络需要非线性激活函数？**

查看答案

**8. Adam 为什么同时维护一阶矩和二阶矩？**

查看答案

**9. Loss 降低是否一定代表模型真正理解了知识？**

查看答案

**10. 大语言模型训练究竟在优化什么？**

查看答案

展开全部答案

# **第七章：mini-llm 下一阶段学习路线**

数学基础建立之后，下一步应该深入研究 Tensor 和 Autograd 的内部机制。

推荐顺序：

| 阶段 | 学习主题            | 核心问题                  |
| -- | --------------- | --------------------- |
| 1  | Tensor Engine   | 数字如何组织成高效的张量？         |
| 2  | Autograd Engine | 如何自动计算梯度？             |
| 3  | Neural Network  | 如何组合数学运算构建网络？         |
| 4  | Transformer     | 如何让 Token 建立上下文关系？    |
| 5  | GPT             | 如何通过预测下一个 Token 学习语言？ |
| 6  | Training        | 如何训练一个完整语言模型？         |

**下一篇文档建议：**

《mini-llm Tensor Engine 深度解析：从内存布局、Shape、Stride 到矩阵乘法》

重点解释：

- Tensor 为什么不只是一个多维数组？
- Shape、Stride、Dtype、Device 分别是什么？
- 连续内存与非连续内存有什么区别？
- 为什么矩阵乘法能够成为 GPU 计算的核心？
- 如何设计 Rust Tensor 的底层数据结构？
- 如何从 CPU 参考实现逐步扩展到 GPU？

这将直接连接数学基础与 mini-llm 的第一个核心模块。

