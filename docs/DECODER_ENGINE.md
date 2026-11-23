# **mini-llm Decoder Engine 深度解析：从 Transformer Stack、LM Head 到 Next Token Prediction**

**—— 理解 GPT 如何将隐藏状态转换成词表概率，以及一个语言模型如何逐个 Token 生成文本**

## **一、引言：Transformer 已经理解上下文，接下来做什么？**

在前面的 mini-llm 系列中，我们已经完成了 Transformer 核心结构的学习。

目前已经理解：

| Engine / 模块           | 核心能力                        |
| --------------------- | --------------------------- |
| Tensor Engine         | 张量、Shape、Stride、矩阵运算        |
| Autograd Engine       | 计算图、梯度、反向传播                 |
| Neural Network Engine | Linear、MLP、Loss、Optimizer   |
| Embedding             | Token 向量化                   |
| Positional Encoding   | 位置表示                        |
| Self-Attention        | Token 信息交互                  |
| Multi-Head Attention  | 多头注意力                       |
| Transformer Block     | Attention、FFN、Residual、Norm |

现在出现一个关键问题：

假设模型输入：

```
我 喜欢
```

经过多个 Transformer Block 后，模型已经获得了上下文相关的隐藏状态。

但是：

**隐藏状态是一个向量，它如何变成下一个汉字或者 Token？**

例如，模型需要预测：

```
我 喜欢 人工智能
```

模型是如何从一个向量中判断下一个 Token 可能是：

- 人工智能
- 编程
- 音乐
- 运动

这就是 Decoder Engine 要解决的问题。

我们今天重点研究：

1. Transformer Stack：多个 Transformer Block 如何组成完整模型？
2. LM Head：隐藏状态如何转换成词表分数？
3. Logits：模型如何表示每个 Token 的预测分数？
4. Softmax：如何将分数转换为概率？
5. Next Token Prediction：模型如何预测下一个 Token？
6. Autoregressive Generation：模型如何逐步生成完整文本？

# **二、Decoder-only：为什么 GPT 使用 Decoder 架构？**

## **2.1 Transformer 原始架构**

2017 年，Google 在论文《Attention Is All You Need》中提出 Transformer。

原始 Transformer 采用 Encoder-Decoder 架构。

[Encoder, Decoder, Both: The Three Transformer Architectures | Daniel Fridljand](https://images.openai.com/static-rsc-4/oG2P1MUz1gweH9cxtiACTfM7ZukXzqemMBde-R4qUvFpG-oH9lyQ0ijWMOW1As9K2pl_mZB1tx-Vuq5pEOGRzOzgeEWViwt_FoS0De2y-MutXVHDmhH-vXM7dkyBRTS-iEgZs3SvGN6OcobmZ_pUq4d7ccF87_c4IrDCFMJYTb4?purpose=inline)

[danielfridljand.de](https://danielfridljand.de/post/transformer-architectures?utm_source=chatgpt.com)

它包含两个部分：

**Encoder：编码器**

负责理解输入序列，将输入转换成上下文表示。

**Decoder：解码器**

结合 Encoder 的输出和已经生成的 Token，预测下一个 Token。

这种结构适合机器翻译等序列到序列任务。

例如：

```
英文输入：
I love artificial intelligence.

       │
       ▼
    Encoder
       │
       ▼
上下文表示
       │
       ▼
    Decoder
       │
       ▼
中文输出：
我喜欢人工智能。
```

## **2.2 GPT 为什么采用 Decoder-only？**

GPT 使用了另一种设计：

**只保留 Transformer Decoder 中的自回归语言建模部分。**

这里的 Decoder-only 并不是说它只使用了一个 Decoder Block，而是说它使用多层带因果掩码的 Transformer Block，不再使用独立的 Encoder 和 Encoder-Decoder Cross-Attention。

结构：

[NLP (6): GPT and Generative Language Models · Chen Kai Blog](https://images.openai.com/static-rsc-4/qnBLFmlmGKbJTnCFMNy3p-Mv2pof9eYdB22u7KPkYqE1SfBo_YG4Y2dEJ5Dd5yYTV8Ueq1eSJ4iyEzBFgKMS6qPKIgAp4kKOE8jb6A4uPfk60Rq4EIsSoF6tD3_Swrrh4Qq04oxfSwR838S35S8T8XSNQgbUpaUNaX-zVeTn6oU?purpose=inline)

[chenk.top](https://www.chenk.top/en/nlp/gpt-generative-models/?utm_source=chatgpt.com)

输入：

```
我 喜欢
```

模型预测：

```
人工智能
```

然后把预测结果追加到输入中：

```
我 喜欢 人工智能
```

继续预测：

```
我 喜欢 人工智能 技术
```

如此反复。

GPT 的核心训练目标是：

**根据前面的 Token，预测下一个 Token。**

数学表达：

\\[ P(x_1,x_2,\ldots,x_T) = \prod\_{t=1}^{T}P(x_t|x_1,\ldots,x\_{t-1}) \\]

这叫作自回归语言模型（Autoregressive Language Model）。

## **2.3 Decoder-only 的优势**

- 架构相对统一。
- 训练目标简单明确。
- 可以通过统一的自回归机制生成文本。
- 可以利用大规模无标注文本进行预训练。
- 适合扩展到对话、代码生成和推理任务。

# **三、Transformer Stack：多个 Transformer Block 如何组成模型？**

在上一篇文章中，我们已经理解了单个 Transformer Block。

但一个 Block 通常不足以形成足够复杂的上下文表示。

因此，Transformer 会堆叠多个 Block。

## **3.1 什么是 Transformer Stack？**

假设我们有一个模型：

```
num_layers = 12d_model = 512num_heads = 8
```

那么模型包含 12 个 Transformer Block。

结构：

[Generalized Language Models](https://images.openai.com/static-rsc-4/RzKEmtj9D6Jbj_2GMlVVdpzZNU06xojykb3-0dOLuMCQYJGCdyZyT_bpt3bVEWXqZXMSMl9gMY90PJGA7oAFJgwbs9HqmiX5fl7n6SUwCz8d5a-zRxPI8zdG3VIouHemu3IoaT-kE-aqmXck5KNAWAiiHCFiC1GMjC9xjVTfXq0?purpose=inline)

[lilianweng.github.io](https://lilianweng.github.io/lil-log/2019/01/31/generalized-language-models.html?utm_source=chatgpt.com)

计算过程：

```
Token IDs
    │
    ▼
Embedding
    │
    ▼
Position Encoding
    │
    ▼
Transformer Block 1
    │
    ▼
Transformer Block 2
    │
    ▼
Transformer Block 3
    │
    ▼
      ...
    │
    ▼
Transformer Block 12
    │
    ▼
Final Norm
    │
    ▼
LM Head
    │
    ▼
Logits
```

每一层都会对输入隐藏状态进行更新。

数学表示：

\\[ H_0=Embedding(X)+PositionEncoding(X) \\]

\\[ H_1=Block_1(H_0) \\]

\\[ H_2=Block_2(H_1) \\]

\\[ \cdots \\]

\\[ H_N=Block_N(H\_{N-1}) \\]

最终：

\\[ H_N\in\mathbb{R}^{B\times T\times D} \\]

其中：

- B：Batch Size。
- T：Sequence Length。
- D：Hidden Dimension。

## **3.2 不同层到底有什么区别？**

假设输入：

```
苹果公司发布了一款新手机。
```

我们可以从功能角度理解不同层：

| 层级           | 可能形成的表示特征         |
| ------------ | ----------------- |
| 初始 Embedding | Token 基础表示        |
| 浅层           | 局部上下文、词语关系        |
| 中层           | 句法、语义和更复杂的关系      |
| 深层           | 更抽象的上下文特征         |
| 最终隐藏状态       | 用于下一个 Token 预测的表示 |

这里是帮助理解的抽象划分，并不是每一层都有固定、独立的语义职责。

实际上，不同层之间会存在复杂的特征交互。

## **3.3 为什么模型需要很多层？**

因为每个 Block 都会对当前表示进行一次信息交互和非线性变换。

可以把它理解为：

\\[ H\_{n+1}=F_n(H_n) \\]

多个 Block 组合：

\\[ H_N=F_N(F\_{N-1}(\cdots F_1(H_0))) \\]

随着深度增加，模型可以构建更复杂的上下文表示。

但要注意：

增加层数不意味着模型能力一定线性增加。

模型效果还受到参数规模、数据质量、训练计算量、优化方法和架构设计影响。

# **四、Final Norm：为什么 Transformer 最后还需要归一化？**

现代 Decoder-only 模型通常在 Transformer Stack 后面增加一个最终归一化层。

例如 LLaMA 风格：

```
Transformer Block 1
        │
Transformer Block 2
        │
       ...
        │
Transformer Block N
        │
        ▼
     RMSNorm
        │
        ▼
     LM Head
```

数学表达：

\\[ H\_{final}=RMSNorm(H_N) \\]

为什么需要？

因为经过很多层计算之后，隐藏状态的数值尺度可能发生变化。

Final Norm 对最终表示进行归一化，使后续预测层接收到经过规范化处理的特征。

注意：

Final Norm 并不是把隐藏状态转换成概率。

它仍然输出隐藏向量。

真正负责转换成词表分数的是 LM Head。

# **五、LM Head：隐藏向量如何变成 Token 分数？**

这是 Decoder Engine 中最关键的模块之一。

## **5.1 什么是 LM Head？**

LM Head 全称 Language Modeling Head。

它负责：

**将 Transformer 输出的隐藏状态投影到词表空间。**

假设：

```
vocab_size = 10000d_model = 128
```

最终隐藏状态：

\\[ H\in\mathbb{R}^{B\times T\times128} \\]

LM Head 权重：

\\[ W\_{lm}\in\mathbb{R}^{128\times10000} \\]

计算：

\\[ Logits=H W\_{lm} \\]

输出：

\\[ Logits\in\mathbb{R}^{B\times T\times10000} \\]

这意味着：

每个 Token 位置都可以得到一个长度为 10000 的分数向量。

[A Visual Guide to Gemma 4 - by Maarten Grootendorst](https://images.openai.com/static-rsc-4/fEbfZnPxv6x-yrNbxyT8Zc16qi1XeiMx8BrhxMq8nYz0rNgUKsn5JtSoENNpj4gtRJpN_IdOA19Ib3SmjEL_qtqK1IF4k_jTAg085YpGPDRisy0OqVeIM7g_BibdCq1nfaLEtB7-OAl5WrLmTGQzquZ3tE-cSAM_BzLobSxHUMw?purpose=inline)

[newsletter.maartengrootendorst.com](https://newsletter.maartengrootendorst.com/p/a-visual-guide-to-gemma-4?utm_source=chatgpt.com)

## **5.2 为什么要映射到词表空间？**

假设词表只有 5 个 Token：

```
0: 我
1: 喜欢
2: 人工智能
3: 编程
4: 音乐
```

模型经过 Transformer 后，得到一个隐藏向量：

\\[ h=[0.2,0.8,0.1,0.5] \\]

LM Head 将它投影成：

```
[2.1, 0.7, 4.5, 3.2, 1.4]
```

这就是 Logits。

每个分数对应一个候选 Token。

| Token | Logit |
| ----- | ----- |
| 我     | 2.1   |
| 喜欢    | 0.7   |
| 人工智能  | 4.5   |
| 编程    | 3.2   |
| 音乐    | 1.4   |

从这个结果看：

“人工智能”的分数最高。

但这里还不是概率。

## **5.3 LM Head 的代码实现**

```
class LMHead:    def __init__(self, d_model, vocab_size):        self.linear = Linear(            d_model,            vocab_size,            bias=False        )    def forward(self, hidden_states):        return self.linear(hidden_states)
```

使用：

```
hidden_states = transformer(x)logits = lm_head(hidden_states)
```

假设：

```
hidden_states: [2, 10, 128]
logits:        [2, 10, 10000]
```

代表：

- 2 个样本。
- 每个样本 10 个 Token。
- 每个位置对 10000 个 Token 进行预测。

# **六、Weight Tying：为什么 LM Head 可以共享 Embedding 权重？**

这是一个值得深入理解的设计。

## **6.1 两个矩阵之间的关系**

Embedding：

\\[ E\in\mathbb{R}^{V\times D} \\]

LM Head：

\\[ W\_{lm}\in\mathbb{R}^{D\times V} \\]

可以发现：

二者的维度刚好可以通过转置对应。

因此，可以让 LM Head 直接使用 Embedding 矩阵的转置：

\\[ Logits=HE^T \\]

这种设计叫作 Weight Tying（权重共享）。

[Weight tying in language models: when and why LLMs share embeddings | by VISHAL SINGH | Jun, 2026 | Medium](https://images.openai.com/static-rsc-4/8W06YIrOZB-SIQl2KQkRor15NMa9vsOFda_O7rBcqrCUO8mBMtHNfozkQ5KSr1ZAO43WIsm26oAHHcgLzGq2_qaNRqDZCCDv1owP4L5Ur_FIJcXK8bWa4vtmi02alWFiqUE1pENJqlCAWjgRT7Y9QQY5izlKfteIjXweclTRKHs?purpose=inline)

[medium.com](https://medium.com/%40vishal09vns/weight-tying-in-language-models-when-and-why-llms-share-embeddings-7f5f5376c625?utm_source=chatgpt.com)

## **6.2 为什么可以共享？**

Embedding 的工作：

```
Token ID → 向量
```

LM Head 的工作：

```
向量 → Token 分数
```

两者分别处于输入和输出方向。

共享权重意味着：

输入时使用某个 Token 对应的向量；

输出时使用相同的向量作为预测该 Token 的投影方向。

这有助于减少参数数量，也可能起到一定的正则化作用。

不过，并不是所有模型都使用 Weight Tying。

在 mini-llm 中，可以把它设计为可配置功能：

```
config.tie_word_embeddings = True
```

如果启用：

```
logits = hidden_states @ embedding.weight.T
```

否则：

```
logits = lm_head(hidden_states)
```

# **七、Logits：模型真正输出的是什么？**

## **7.1 Logits 不是概率**

这是初学大语言模型时非常容易混淆的概念。

模型输出：

```
[2.1, 0.7, 4.5, 3.2, 1.4]
```

这些数字可以是正数，也可以是负数。

它们没有必须满足的范围。

例如：

```
[-3.2, 1.5, 8.7, -0.4, 2.1]
```

仍然是合法的 Logits。

Logits 只表示模型对不同候选 Token 的未归一化预测分数。

## **7.2 Logits 如何变成概率？**

使用 Softmax。

数学公式：

\\[ P_i= \frac{e^{z_i}}{\sum_j e^{z_j}} \\]

其中：

- \\(z_i\\)：第 i 个 Token 的 Logit。
- \\(P_i\\)：第 i 个 Token 的预测概率。

假设：

```
Logits:
[2.1, 0.7, 4.5, 3.2, 1.4]
```

经过 Softmax：

```
概率:
[0.073, 0.018, 0.803, 0.218, 0.036]
```

以上为示意数值，实际 Softmax 结果应归一化为总和 1。

更准确地说，Softmax 会让所有概率相加等于 1。

[GitHub - FareedKhan-dev/Understanding-Transformers-Step-by-Step-math-example: Understanding Large Language Transformer Architecture like a child](https://images.openai.com/static-rsc-4/GjoBBiXppkRxxRwl5UUnI9nUY1xG3dG_u01HxSEKLsC1thPPsOmi-P81mrvupmAd2zjQjHCsnH6vWngnmOwwCUCWK9CzbKCKYbhbGdej4N_f_FQSyviQbCTz4AB_qI4YiP6AkXkFDHKSiQT3Id1_aUNWjCmgLZipcALBy0R_mlQ?purpose=inline)

[github.com](https://github.com/FareedKhan-dev/Understanding-Transformers-Step-by-Step-math-example?utm_source=chatgpt.com)

## **7.3 Softmax 的数值稳定性**

实际工程中，Softmax 通常先减去最大 Logit：

\\[ Softmax(z_i)= \frac{e^{z_i-\max(z)}}{ \sum_j e^{z_j-\max(z)} } \\]

这样可以避免指数运算产生过大的数值。

mini-llm 的 Softmax 必须实现这种稳定版本。

# **八、Next Token Prediction：模型究竟如何预测下一个 Token？**

现在终于来到语言模型最核心的任务。

假设输入：

```
我 喜欢
```

经过完整模型：

```
Token IDs
    │
    ▼
Embedding
    │
    ▼
Transformer Stack
    │
    ▼
Final Norm
    │
    ▼
LM Head
    │
    ▼
Logits
    │
    ▼
Softmax
    │
    ▼
Token Probabilities
```

模型得到一个词表概率分布：

| 候选 Token | 概率   |
| -------- | ---- |
| 人工智能     | 0.42 |
| 编程       | 0.25 |
| 音乐       | 0.15 |
| 学习       | 0.12 |
| 其他       | 0.06 |

注意，这里的概率只是为了说明机制而构造的示例。

如果选择最大概率：

```
人工智能
```

那么：

```
我 喜欢
```

就变成：

```
我 喜欢 人工智能
```

模型再根据新的上下文预测下一个 Token。

这就是 Next Token Prediction。

# **九、为什么模型不总是选择概率最大的 Token？**

如果每次都选择概率最大的 Token，会出现一个问题：

生成结果可能过于单一、重复。

因此，语言模型通常支持多种采样策略。

## **9.1 Greedy Decoding**

贪心解码。

每次选择概率最大的 Token：

\\[ x_t=\arg\max_i P(x_i|context) \\]

优点：

- 简单。
- 结果确定。
- 不需要随机采样。

缺点：

- 容易产生重复。
- 缺少多样性。
- 可能过早进入局部最优生成路径。

## **9.2 Temperature：温度采样**

Temperature 用于调整概率分布的尖锐程度。

公式：

\\[ P_i= Softmax\left(\frac{z_i}{T}\right) \\]

其中 T 是温度参数。

[Understanding OpenAI's Temperature Parameter | Colt Steele](https://images.openai.com/static-rsc-4/ocKOg_bGFAMbH1JzekF_VJGQ2XGiAGAtf0Ui4ciOIXGEdIkLsx8aLGDRQsZ_7HrTnu1Whra3TjBBHjtF5fCdJ778lfW3h98OFpYeeotthrtg4AJTqq2MA3VxrQYGxuqxzGxDqKLUxTcArrBDTe9LkauTNSzu0AS_Xl0__GJK-Gg?purpose=inline)

[coltsteele.com](https://www.coltsteele.com/tips/understanding-openai-s-temperature-parameter?utm_source=chatgpt.com)

| Temperature | 影响               |
| ----------- | ---------------- |
| T 接近 0      | 更倾向于高概率 Token    |
| T = 1       | 保持原始 Softmax 分布  |
| T 大于 1      | 分布更加平滑           |
| T 非常大       | 不同 Token 的概率趋于接近 |

需要注意：

Temperature 并不是直接改变模型参数，而是改变推理时的采样分布。

## **9.3 Top-K Sampling**

Top-K 的思想：

只保留概率最高的 K 个 Token。

例如：

```
原始候选：

人工智能 0.42
编程     0.25
音乐     0.15
学习     0.12
其他     0.06
```

设置：

```
top_k = 3
```

那么：

```
人工智能
编程
音乐
```

进入采样候选集合。

其他 Token 被屏蔽。

然后对保留的分数重新归一化。

## **9.4 Top-P Sampling**

Top-P 又叫 Nucleus Sampling。

它不固定保留多少个 Token，而是按照概率从高到低排列，保留累计概率达到 P 的最小候选集合。

例如：

```
top_p = 0.9
```

假设：

```
人工智能 0.42
编程     0.25
音乐     0.15
学习     0.12
其他     0.06
```

前四项累计概率为：

\\[ 0.42+0.25+0.15+0.12=0.94 \\]

因此，保留前四项。

## **9.5 不同采样策略对比**

| 方法          | 核心机制   | 特点     |
| ----------- | ------ | ------ |
| Greedy      | 最大概率   | 确定性    |
| Temperature | 调整分布   | 控制随机程度 |
| Top-K       | 固定候选数量 | 限制候选范围 |
| Top-P       | 累计概率阈值 | 动态候选范围 |

在 mini-llm 中，建议先实现 Greedy，再依次实现 Temperature、Top-K 和 Top-P。

# **十、Autoregressive Generation：模型如何生成一整段文本？**

现在把前面的内容组合起来。

输入：

```
今天天气
```

模型预测：

```
很好
```

然后将新 Token 追加到序列：

```
今天天气很好
```

再次预测：

```
。
```

继续：

```
今天天气很好。
```

这就是自回归生成。

## **10.1 生成过程**

## **Next Token 生成循环**

1. **输入 Prompt**

   今天天气
2. **模型前向计算**

   Embedding → Transformer Stack → LM Head
3. **计算概率分布**

   Softmax → Temperature → Sampling
4. **选择下一个 Token**

   很好
5. **追加到上下文**

   今天天气很好
6. **重复计算**

   直到遇到 EOS、停止条件或达到最大生成长度。

## **10.2 生成算法伪代码**

```
def generate(    model,    input_ids,    max_new_tokens=50,    temperature=1.0):    for _ in range(max_new_tokens):        # 模型前向计算        logits = model(input_ids)        # 获取最后一个位置的预测分数        next_logits = logits[:, -1, :]        # 温度缩放        next_logits = next_logits / temperature        # 转换成概率        probs = softmax(next_logits)        # 采样下一个 Token        next_token = sample(probs)        # 追加 Token        input_ids = concat(            input_ids,            next_token        )        # 检查结束条件        if is_eos(next_token):            break    return input_ids
```

这段代码已经表达了一个最简单的语言模型生成过程。

但它还存在一个非常大的性能问题。

# **十一、KV Cache：为什么大模型推理需要缓存？**

这是 Decoder Engine 与 Inference Engine 之间的重要连接点。

## **11.1 不使用 KV Cache 会发生什么？**

假设模型已经输入：

```
我 喜欢 人工智能
```

现在需要预测下一个 Token。

模型需要计算 Attention。

但如果每次生成一个新 Token，都重新计算整个序列的所有 Query、Key、Value，就会产生大量重复计算。

例如：

第一轮：

```
我
```

第二轮：

```
我 喜欢
```

第三轮：

```
我 喜欢 人工智能
```

每轮都重新计算之前 Token 的 K、V。

这非常浪费。

## **11.2 KV Cache 的核心思想**

由于自回归模型不会改变历史 Token，因此历史位置计算出来的 Key 和 Value 可以缓存起来。

下一轮只需要计算新 Token 的 Query、Key、Value。

[KV Caching in Autoregressive Transformers - FeynmanWiki](https://images.openai.com/static-rsc-4/WQwWpbZZdMkQshIElpsHZek1rkI2JlQpA4KwjcUJseoity3vBoztsrezdGK3PJUDGmLfMssQEvSZ1MHwicIkquT2k576DLFZeuWMfpecxgGdwRu9IEIJ8A-ZY_7-Rec-h1q1WuPWGyQN3GS48qrDX1PwQlvJ2U92O_IDLEcxLaY?purpose=inline)

[feynmanwiki.com](https://www.feynmanwiki.com/library/kv-caching-pvo5?utm_source=chatgpt.com)

例如：

第一轮：

```
Token 1:
计算 K1、V1
缓存 K1、V1
```

第二轮：

```
Token 2:
计算 K2、V2

缓存：
K1、V1
K2、V2
```

第三轮：

```
Token 3:
计算 K3、V3

缓存：
K1、V1
K2、V2
K3、V3
```

下一轮只需要计算新 Token 的 K、V，并使用缓存的历史 K、V 参与 Attention。

## **11.3 KV Cache 的内存开销**

假设：

```
num_layers = 32num_heads = 32head_dim = 128seq_len = 8192
```

使用 FP16，每个元素占 2 字节。

普通 MHA 的 KV Cache 估算：

\\[ Memory= 2\times L\times H\times S\times D\times2 \\]

其中第一个 2 代表 Key 和 Value 两份缓存，最后一个 2 代表 FP16 字节数。

代入：

\\[ 2\times32\times32\times8192\times128\times2 \\]

约为：

4 GiB

这只是单个请求的 KV Cache 估算，不包含模型权重、激活值和其他运行时开销。

因此，KV Cache 是大模型推理的重要内存组成部分。

## **11.4 GQA 为什么能够减少 KV Cache？**

在 GQA 架构中，多个 Query Head 共享 Key、Value Head。

例如：

```
32 Query Heads
8 KV Heads
```

KV Cache 主要按照 8 个 KV Head 计算，而不是 32 个。

理想情况下，KV Cache 可以缩小到原来的四分之一。

这也是现代 LLM 推理优化的重要方向。

# **十二、mini-llm Decoder Engine 应该如何实现？**

现在可以正式设计 Decoder Engine。

## **12.1 推荐项目结构**

```
mini-llm/
│
├── tensor/
│   ├── tensor.py
│   ├── ops.py
│   └── autograd.py
│
├── nn/
│   ├── linear.py
│   ├── embedding.py
│   ├── normalization.py
│   └── activation.py
│
├── transformer/
│   ├── attention.py
│   ├── rope.py
│   ├── multi_head_attention.py
│   ├── transformer_block.py
│   └── causal_mask.py
│
├── model/
│   ├── decoder.py
│   ├── transformer_stack.py
│   ├── lm_head.py
│   └── model_config.py
│
├── generation/
│   ├── sampler.py
│   ├── temperature.py
│   ├── top_k.py
│   ├── top_p.py
│   └── generate.py
│
├── inference/
│   ├── kv_cache.py
│   └── inference_engine.py
│
└── tokenizer/
    ├── tokenizer.py
    └── vocabulary.py
```

## **12.2 Decoder Model 的结构**

```
class DecoderModel:    def __init__(self, config):        self.embedding = Embedding(            config.vocab_size,            config.d_model        )        self.layers = [            TransformerBlock(config)            for _ in range(config.num_layers)        ]        self.norm = RMSNorm(            config.d_model        )        self.lm_head = LMHead(            config.d_model,            config.vocab_size        )    def forward(self, input_ids):        x = self.embedding(input_ids)        for layer in self.layers:            x = layer(x)        x = self.norm(x)        logits = self.lm_head(x)        return logits
```

这是一个简化的 Decoder-only 模型结构。

完整实现还需要根据模型配置处理 RoPE、Causal Mask、KV Cache、Padding Mask、权重共享以及训练和推理模式。

# **十三、训练阶段：模型如何学会 Next Token Prediction？**

现在我们已经知道模型如何生成 Token。

但是模型最初并不知道应该生成什么。

它需要训练。

假设训练数据：

```
我 喜欢 人工智能
```

构造输入和目标：

| 输入 Token  | 目标 Token |
| --------- | -------- |
| 我         | 喜欢       |
| 我、喜欢      | 人工智能     |
| 我、喜欢、人工智能 | EOS      |

这里需要特别注意：

训练时通常一次性将完整序列输入模型，通过 Causal Mask 实现并行预测。

不需要像推理阶段一样逐个 Token 执行模型前向计算。

## **13.1 Cross Entropy Loss**

模型输出 Logits：

\\[ z=[2.1,0.7,4.5,3.2,1.4] \\]

假设正确答案是 Token 2。

模型需要提高正确 Token 的预测概率。

交叉熵损失：

\\[ L=-\log P(y\_{true}) \\]

如果正确 Token 的概率是：

\\[ P(y\_{true})=0.8 \\]

那么：

\\[ L=-\log(0.8) \\]

约为：

\\[ 0.223 \\]

如果正确 Token 的概率只有 0.01：

\\[ L=-\log(0.01) \\]

约为：

\\[ 4.605 \\]

概率越低，损失越大。

## **13.2 训练完整流程**

[Andrej Karpathy's microGPT Architecture — Complete Guide - DEV Community](https://images.openai.com/static-rsc-4/JWp9yD2gQBnc_qMXYagE-BaDbTR0TPET4y5QdBQvvtGnYx0Ti5SosAmMMgk2YYSvNxhhAcNo-asj-DZ-CooBVAoDQXXc84seF1CNGxndDx0MsvWtOF86wXCIpOtVjb94Az4TqlRQUJ3IyYEi2Z8D-AuYQYqATSYD1L2RKF1p-Nc?purpose=inline)

[dev.to](https://dev.to/rsrini7/andrej-karpathys-microgpt-architecture-complete-guide-em8?utm_source=chatgpt.com)

```
训练文本
   │
   ▼
Tokenizer
   │
   ▼
Input IDs
   │
   ▼
Decoder Model
   │
   ▼
Logits
   │
   ▼
Cross Entropy Loss
   │
   ▼
Backward
   │
   ▼
Gradient
   │
   ▼
Optimizer
   │
   ▼
更新模型参数
```

不断重复这个过程。

模型逐渐学会根据前面的 Token 预测下一个 Token。

# **十四、从 mini-llm 看完整大语言模型**

现在可以将整个系统串联起来。

## **mini-llm 完整计算架构**

**Tokenizer**

文本 → Token IDs

**Embedding + Position Encoding**

离散 Token → 向量表示

**Transformer Stack**

Transformer Block 1

Attention + FFN + Residual + Norm

Transformer Block 2

Attention + FFN + Residual + Norm

Transformer Block 3

Attention + FFN + Residual + Norm

…… N 层

**Final Norm**

**LM Head**

隐藏向量 → 词表 Logits

**Softmax + Sampling**

概率分布 → 下一个 Token

**Autoregressive Generation**

追加 Token → 继续预测

底层由此前设计的几个 Engine 提供支撑：

| Engine                | 职责       |
| --------------------- | -------- |
| Tensor Engine         | 数学计算     |
| Autograd Engine       | 自动求导     |
| Neural Network Engine | 神经网络算子   |
| Transformer Engine    | 上下文信息交互  |
| Decoder Engine        | 组织完整语言模型 |
| Training Engine       | 模型训练     |
| Inference Engine      | 模型推理和生成  |

# **十五、最重要的思考：为什么预测下一个 Token 能够产生语言能力？**

这是整个 mini-llm 系列最值得思考的问题。

一个模型只是在做：

```
输入前面的 Token
       │
       ▼
预测下一个 Token
       │
       ▼
计算预测误差
       │
       ▼
更新模型参数
```

看起来只是一个简单的概率预测任务。

但是，大规模语言数据中包含大量复杂结构：

- 语言语法；
- 人类知识；
- 逻辑关系；
- 因果关系；
- 数学规律；
- 编程规则；
- 世界事件；
- 人类行为模式。

如果模型想要准确预测下一个 Token，就必须学习能够解释这些数据规律的内部表示。

例如：

```
水在标准大气压下的沸点是
```

模型需要学习：

```
100℃
```

但如果上下文变成：

```
水在海拔较高的地区沸点通常会
```

模型需要给出不同的预测。

这意味着模型不能只记住固定答案，还需要学习上下文条件与预测结果之间的关系。

因此：

**Next Token Prediction 虽然是训练目标，但为了完成这个目标，模型可能逐渐形成复杂的语言、知识、关系和世界状态表示。**

不过，这不意味着模型必然拥有与人类相同的理解机制，也不意味着预测能力本身就等于完整的世界理解。

它是研究语言模型智能的重要入口。

# **十六、mini-llm 下一阶段：从 Decoder Engine 进入 Training Engine**

到这里，我们已经完成了从 Tensor 到 Decoder 的主要架构知识。

下一步不应该继续停留在单个神经网络模块，而应该开始研究：

**模型如何真正训练起来？**

下一篇推荐

# **《mini-llm Training Engine 深度解析：从 Dataset、Batch、Cross Entropy 到 Backpropagation、AdamW》**

**核心研究内容：**

1. Dataset：训练文本如何组织？
2. Tokenizer：如何构建训练样本？
3. Sliding Window：如何切分长文本？
4. Batch：为什么需要批量训练？
5. Input / Target：如何构造下一个 Token 预测任务？
6. Cross Entropy：如何衡量预测误差？
7. Backpropagation：如何计算整个 Transformer 的梯度？
8. AdamW：如何更新数十亿个模型参数？
9. Learning Rate Scheduler：如何控制训练速度？
10. Checkpoint：如何保存和恢复模型训练？
11. Gradient Accumulation：显存不足时如何训练？
12. Mixed Precision：FP16、BF16 如何降低显存消耗？

**到下一篇，我们就会从“理解一个大语言模型如何计算”，正式进入“如何从零训练一个大语言模型”。**

