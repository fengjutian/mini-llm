# **《mini-llm Agent Learning Engine 深度解析：从 Experience Replay、Credit Assignment 到 Policy Optimization、Continual Learning 与 Self-improvement》**

前面的 mini-llm 系列，我们已经构建了 Agent Engine。

Agent 能够：

- 理解用户目标。
- 拆解复杂任务。
- 制定执行计划。
- 调用外部工具。
- 观察环境变化。
- 根据执行结果调整行动。
- 通过多个 Agent 协同完成任务。

但这里还存在一个关键问题：

**Agent 完成了一次任务，并不意味着它获得了新的能力。**

例如，一个 Agent 第一次修复代码失败，第二次通过调整策略成功。

下一次遇到类似问题，它是否能够：

1. 记住上一次的失败原因？
2. 识别上一次成功的关键行动？
3. 将经验迁移到相似任务？
4. 改进自己的决策策略？
5. 在未来减少相同错误？

如果不能，那么它可能只是一个能够重复执行任务的系统，而不是一个能够从经验中持续改进的学习系统。

这就是 Agent Learning Engine 的研究目标。

# **一、从 Agent Execution 到 Agent Learning**

## **1.1 执行与学习有什么区别？**

先看一个简单例子。

用户要求：

```
修复项目中的 TypeScript 类型错误。
```

Agent 第一次执行：

```
1. 扫描项目。
2. 找到 15 个类型错误。
3. 修改代码。
4. 运行测试。
5. 发现仍有 5 个错误。
6. 修改失败。
```

第二次执行：

```
1. 检查错误日志。
2. 发现问题来自公共类型定义。
3. 修改公共类型。
4. 重新运行测试。
5. 全部通过。
```

现在有两种可能。

**情况 A：只有执行，没有学习**

下一次遇到类似问题，Agent 仍然从头开始，可能再次采用错误策略。

**情况 B：执行后形成经验**

Agent 保存：

```
经验：
遇到大量关联的 TypeScript 类型错误时，
优先检查公共类型定义和依赖关系，
而不是逐个修改报错文件。
```

下一次遇到类似问题，Agent 可以检索这条经验。

但还需要注意：

记住一条经验，不代表一定学会了。

真正的学习应该表现为未来任务中的可测量能力改善。

## **1.2 Agent Learning 的定义**

可以将 Agent Learning 定义为：

> Agent 根据与环境交互产生的经验，更新其内部状态、策略、知识或模型参数，从而改善未来任务表现的过程。

抽象为：

\\[ \pi\_{t+1}=Learning(\pi_t,E_t) \\]

其中：

- \\(\pi_t\\)：当前策略。
- \\(E_t\\)：当前或历史经验。
- \\(\pi\_{t+1}\\)：更新后的策略。

这里的策略不一定是神经网络权重。

它也可以是：

- 更好的任务分解规则。
- 更合理的工具选择方式。
- 更准确的环境模型。
- 更有效的记忆检索策略。
- 更可靠的执行流程。

因此，Agent Learning 不等于模型训练。

它包含多个层次。

# **二、Experience：经验是什么？**

Agent 要学习，首先需要定义什么是经验。

## **2.1 从一次行动到一条经验**

Agent 执行任务时，会产生一系列信息：

```
用户目标
    ↓
当前观察
    ↓
模型推理
    ↓
任务规划
    ↓
工具调用
    ↓
工具结果
    ↓
环境变化
    ↓
任务评估
```

这些信息组合起来，形成一条经验。

在强化学习中，常见的基本经验表示是：

\\[ e_t=(s_t,a_t,r_t,s\_{t+1}) \\]

其中：

| 符号             | 含义       |
| -------------- | -------- |
| \\(s_t\\)      | 当前状态     |
| \\(a_t\\)      | 当前行动     |
| \\(r_t\\)      | 行动获得的奖励  |
| \\(s\_{t+1}\\) | 行动之后的新状态 |

例如：

```
experience = {    "state": {        "task": "修复类型错误",        "error_count": 15    },    "action": {        "tool": "search_code",        "query": "type definitions"    },    "reward": 0.8,    "next_state": {        "error_count": 5    }}
```

这是简化的强化学习经验表示。

但对于实际 Agent，单纯的四元组通常不够。

还需要记录：

- 用户目标。
- 任务上下文。
- 行动理由。
- 工具参数。
- 执行结果。
- 失败原因。
- 环境约束。
- 结果证据。
- 模型和工具版本。

## **2.2 Trajectory：行动轨迹**

一条经验描述一次状态转移。

而 Agent 完成一个复杂任务，通常需要很多次状态转移。

因此需要 Trajectory。

\\[ \tau=(s_0,a_0,r_0,s_1,\ldots,s_T) \\]

例如：

## **Agent 执行轨迹**

1. **State 0：发现性能问题**

   页面加载时间 4.8 秒
2. **Action 0：运行性能分析**

   调用 Performance Profiler
3. **State 1：发现图片加载瓶颈**

   图片资源占用大量加载时间
4. **Action 1：压缩图片资源**

   修改图片格式和尺寸
5. **State 2：重新运行测试**

   加载时间下降到 1.9 秒
6. **Final Evaluation：任务通过**

   目标达成

Trajectory 对 Agent Learning 特别重要，因为复杂任务的结果往往不是某一个行动单独造成的。

例如：

- 第一步选择正确的分析工具。
- 第二步获取正确的证据。
- 第三步制定合理的修改方案。
- 第四步执行修改。
- 第五步通过测试。

最终成功可能依赖整条行动链。

## **2.3 Experience Schema：经验数据结构**

建议 mini-llm 使用结构化经验：

```
from dataclasses import dataclassfrom typing import Any@dataclassclass Experience:    experience_id: str    task_id: str    state: dict[str, Any]    action: dict[str, Any]    result: dict[str, Any]    next_state: dict[str, Any]    reward: float | None = None    success: bool | None = None    evidence: list[str] | None = None
```

实际系统还应增加：

```
created_at
agent_version
model_version
tool_version
policy_version
environment_version
source
confidence
review_status
```

为什么需要版本？

因为同样的行动，在不同模型、不同工具版本和不同环境中，结果可能完全不同。

如果不记录版本，就很难判断性能变化究竟来自学习，还是来自外部环境变化。

# **三、Experience Replay：经验回放**

经验产生之后，如何利用它们？

一个经典方法是 Experience Replay。

## **3.1 为什么需要经验回放？**

假设 Agent 执行了 1000 个任务。

如果每次任务结束后就丢弃历史过程，那么每次面对新任务，都可能需要重新探索。

Experience Replay 的核心思想是：

**保存过去的经验，在后续学习过程中重复使用。**

在强化学习中，Experience Replay 是一种经典训练机制。

例如 DQN 会将历史转移存入 Replay Buffer，然后随机抽取一批经验训练价值网络。

## **3.2 Replay Buffer**

Replay Buffer 是经验缓存区。

## **Experience Replay**

**Agent Environment Interaction**

不断产生经验

**Replay Buffer**

Experience 001

成功经验

Experience 002

失败经验

Experience 003

成功经验

Experience 004

未完成经验

**Sample Batch**

从历史经验中抽取训练样本

**Learning / Policy Update**

更新策略或模型参数

## **3.3 为什么不直接使用最新经验？**

如果 Agent 只使用最新产生的经验，可能出现：

- 数据分布过于集中。
- 近期任务主导学习。
- 忘记过去的重要能力。
- 训练过程不稳定。

经验回放通过混合历史经验，能够在一定程度上缓解这些问题。

在 DQN 中，随机采样历史转移可以降低连续样本之间的相关性，但不能保证样本完全独立，也不能自动消除分布偏差。

## **3.4 Replay Buffer 实现**

```
import randomfrom collections import dequeclass ReplayBuffer:    def __init__(self, capacity=10000):        self.buffer = deque(            maxlen=capacity        )    def add(self, experience):        self.buffer.append(experience)    def sample(self, batch_size):        return random.sample(            self.buffer,            min(batch_size, len(self.buffer))        )    def __len__(self):        return len(self.buffer)
```

使用：

```
buffer = ReplayBuffer()buffer.add({    "state": "state_001",    "action": "search_code",    "reward": 0.5,    "next_state": "state_002"})batch = buffer.sample(32)
```

这是通用经验缓存示例，并不是完整的 DQN 训练器。

## **3.5 Agent 的经验回放与传统强化学习有什么区别？**

这里需要区分两种用途。

| 类型                     | 目的               |
| ---------------------- | ---------------- |
| RL Replay Buffer       | 用于训练策略或价值网络      |
| Agent Experience Store | 用于检索历史任务、案例和执行经验 |

例如，Agent Experience Store 可以保存：

```
任务类型：
TypeScript 类型错误修复

问题：
多个模块出现关联错误

成功策略：
优先检查公共类型定义

适用条件：
错误集中在共享接口

验证结果：
相关测试全部通过
```

未来遇到相似任务时，Agent 可以检索这条经验。

但这种检索属于经验利用，不一定发生了模型参数更新。

这也是 mini-llm 需要区分的两种能力：

**经验记忆（Experience Memory）与经验学习（Experience Learning）。**

# **四、Credit Assignment：信用分配**

经验回放解决的是如何重用经验。

但还有一个更困难的问题：

**一次复杂任务成功了，究竟是哪些行动带来了成功？**

这就是 Credit Assignment。

## **4.1 为什么需要信用分配？**

假设 Agent 执行了以下任务：

```
1. 搜索代码
2. 读取文件
3. 修改代码
4. 删除旧文件
5. 运行测试
```

最终测试通过。

如果给每个行动都奖励 1：

```
搜索代码：+1
读取文件：+1
修改代码：+1
删除旧文件：+1
运行测试：+1
```

就会产生一个问题：

删除旧文件真的有助于任务成功吗？

可能完全没有关系。

甚至可能是危险操作。

如果系统只根据最终成功给所有行动奖励，就可能强化无关甚至有害的行为。

## **4.2 Terminal Reward：终端奖励**

最简单的奖励方式是在任务结束时才提供奖励。

例如：

\\[ R= \begin{cases} 1 & \text{任务成功}\\\ 0 & \text{任务失败} \end{cases} \\]

问题是：

中间行动没有直接反馈。

这就是稀疏奖励问题（Sparse Reward）。

## **4.3 Reward Shaping：奖励塑形**

可以为中间过程增加奖励。

例如：

\\[ R_t= R\_{\text{progress}} - R\_{\text{cost}} - R\_{\text{risk}} \\]

其中：

- \\(R\_{\text{progress}}\\)：任务进展。
- \\(R\_{\text{cost}}\\)：执行成本。
- \\(R\_{\text{risk}}\\)：行动风险。

例如：

| 行动   | 进展奖励 | 成本惩罚  | 风险惩罚  |
| ---- | ---- | ----- | ----- |
| 搜索代码 | +0.2 | -0.01 | 0     |
| 读取文件 | +0.1 | -0.01 | 0     |
| 修改代码 | +0.3 | -0.05 | -0.02 |
| 删除文件 | 0    | -0.02 | -0.5  |
| 测试通过 | +0.5 | -0.02 | 0     |

这些数字只是示例，不代表通用奖励标准。

奖励塑形可以改善学习信号，但如果设计不当，也可能让 Agent 学会钻奖励规则的空子。

## **4.4 Temporal Difference：时序差分学习**

在强化学习中，一个经典方法是 Temporal Difference（TD）。

以状态价值函数为例：

\\[ V(s_t) \\]

表示从状态 \\(s_t\\) 出发，未来能够获得的期望累计回报。

TD 误差：

\\[ \delta_t= r_t+\gamma V(s\_{t+1})-V(s_t) \\]

其中：

- \\(r_t\\)：当前奖励。
- \\(\gamma\\)：折扣因子。
- \\(V(s\_{t+1})\\)：下一状态价值。
- \\(V(s_t)\\)：当前状态价值。

通过 TD 误差，模型可以逐步调整对状态价值的估计。

对于终止状态，下一状态价值应为零，不能继续 bootstrap。

## **4.5 Advantage：优势函数**

还有一个重要概念：Advantage。

\\[ A(s,a)=Q(s,a)-V(s) \\]

其中：

- \\(Q(s,a)\\)：在状态 \\(s\\) 执行动作 \\(a\\) 的价值。
- \\(V(s)\\)：状态 \\(s\\) 的平均价值。

如果：

\\[ A(s,a)>0 \\]

表示这个行动相对于该状态下的平均水平更有价值。

如果：

\\[ A(s,a)<0 \\]

表示这个行动相对较差。

这为策略优化提供了重要学习信号。

需要注意：价值函数只是对未来回报的估计，不代表行动结果的因果证明。

# **五、Reward Modeling：奖励模型**

Agent 如何判断一次行动是好还是坏？

这就涉及 Reward Modeling。

## **5.1 什么是 Reward Model？**

Reward Model 是一个用于评价模型输出或行动质量的模型。

例如：

```
任务：
修复 TypeScript 类型错误。

候选方案 A：
直接使用 any 消除类型错误。

候选方案 B：
修正接口类型定义。

候选方案 C：
删除发生错误的模块。
```

奖励模型可以根据：

- 正确性。
- 完整性。
- 安全性。
- 代码质量。
- 任务约束。
- 测试结果。

对候选方案进行评分。

例如：

| 方案 | 评分   | 主要问题   |
| -- | ---- | ------ |
| A  | 0.35 | 掩盖类型问题 |
| B  | 0.92 | 修复根本原因 |
| C  | 0.05 | 破坏功能   |

这些分数仅用于演示。

## **5.2 Reward Model 的基本形式**

可以构建：

\\[ R\_\phi(s,a) \\]

其中：

- \\(s\\)：当前状态。
- \\(a\\)：候选行动。
- \\(\phi\\)：奖励模型参数。

输入状态和行动，输出一个标量分数。

例如：

```
class RewardModel:    def score(self, state, action):        features = {            "correctness": 0.9,            "safety": 0.8,            "cost": 0.7,            "constraint_match": 1.0        }        return (            0.4 * features["correctness"]            + 0.3 * features["safety"]            + 0.1 * features["cost"]            + 0.2 * features["constraint_match"]        )
```

这只是规则评分示例，不是真正训练出来的神经网络奖励模型。

## **5.3 Preference Learning：偏好学习**

奖励模型可以通过人类偏好数据训练。

例如：

```
同一个问题：

方案 A：使用 any 快速消除错误。
方案 B：修复正确的接口类型。

人类偏好：B > A
```

将大量偏好数据收集起来：

\\[ D=\\{(x,y_w,y_l)\\} \\]

其中：

- \\(x\\)：输入上下文。
- \\(y_w\\)：更受偏好的方案。
- \\(y_l\\)：较不受偏好的方案。

一种常见的 Bradley-Terry 偏好模型是：

\\[ P(y_w\succ y_l) = \frac{e^{r\_\phi(x,y_w)}} {e^{r\_\phi(x,y_w)}+e^{r\_\phi(x,y_l)}} \\]

等价于：

\\[ P(y_w\succ y_l) = \sigma(r\_\phi(x,y_w)-r\_\phi(x,y_l)) \\]

其中 \\(\sigma\\) 是 Sigmoid 函数。

训练时，可以最小化：

\\[ \mathcal{L}\_{RM} = -\log\sigma( r\_\phi(x,y_w)-r\_\phi(x,y_l) ) \\]

这样，奖励模型逐渐学会预测人类偏好。

但需要记住：

**人类偏好不一定等于客观正确性。**

奖励模型学到的是偏好数据中的评价规律，可能继承偏见、遗漏约束，也可能被模型利用。

# **六、Policy Optimization：策略优化**

有了经验和奖励信号，就可以尝试改进 Agent 的策略。

## **6.1 什么是 Policy？**

Policy 表示 Agent 在给定状态下选择行动的规则。

\\[ \pi(a|s) \\]

它表示：

在状态 \\(s\\) 下，Agent 选择行动 \\(a\\) 的概率。

例如：

```
当前状态：
项目出现多个关联类型错误。

策略：

搜索公共类型定义：0.70
逐个修改错误文件：0.20
直接使用 any：0.10
```

这些概率表示策略分布，而不是模型对客观事实的置信度。

## **6.2 策略优化的目标**

强化学习通常希望最大化期望累计回报：

\\[ J(\theta)= \mathbb{E}\_{\tau\sim\pi\_\theta} \left[ \sum\_{t=0}^{T} \gamma^t r_t \right] \\]

其中：

- \\(\theta\\)：策略参数。
- \\(\tau\\)：行动轨迹。
- \\(\gamma\\)：折扣因子。
- \\(r_t\\)：每一步奖励。

策略优化就是调整 \\(\theta\\)，使期望回报增加。

## **6.3 Policy Gradient**

一个经典策略梯度表达式：

\\[ \nabla\_\theta J(\theta) = \mathbb{E} \left[ \nabla\_\theta\log\pi\_\theta(a_t|s_t) A_t \right] \\]

其中 \\(A_t\\) 是优势估计。

直观理解：

- 如果行动优于平均水平，提高它的选择概率。
- 如果行动低于平均水平，降低它的选择概率。

这就是 Policy Gradient 的基本思想。

## **6.4 PPO：Proximal Policy Optimization**

PPO 是一种常用的策略优化算法。

它通过限制新旧策略的变化幅度，避免一次更新造成过大的策略变化。

定义概率比率：

\\[ r_t(\theta)= \frac{\pi\_\theta(a_t|s_t)} {\pi\_{\theta\_{\text{old}}}(a_t|s_t)} \\]

PPO 的 Clipped Objective：

\\[ L^{CLIP}(\theta)= \mathbb{E}\_t \left[ \min \left( r_t(\theta)A_t, \operatorname{clip}(r_t(\theta),1-\epsilon,1+\epsilon)A_t \right) \right] \\]

其中 \\(\epsilon\\) 控制更新幅度。

PPO 并不是保证策略永远不会退化，而是通过裁剪目标限制部分更新行为。

## **6.5 Agent 策略优化不一定要修改 LLM 权重**

这是工程上非常重要的区别。

Agent 的策略可以由多个部分组成：

```
Agent Policy
   │
   ├── Tool Selection Policy
   ├── Task Decomposition Policy
   ├── Planning Policy
   ├── Retry Policy
   ├── Memory Retrieval Policy
   └── Model Generation Policy
```

其中：

- Tool Selection Policy 可以通过历史成功率优化。
- Retry Policy 可以使用统计方法调整。
- Planning Policy 可以优化搜索策略。
- Memory Retrieval Policy 可以优化召回和排序。
- Model Generation Policy 才可能涉及神经网络参数训练。

因此，Agent 的学习不一定需要频繁微调大模型。

# **七、Reinforcement Learning：强化学习与 Agent**

前面介绍的 Reward、Value、Policy，最终都可以连接到强化学习。

## **7.1 强化学习的基本结构**

强化学习通常包括：

- Agent：智能体。
- Environment：环境。
- State：状态。
- Action：行动。
- Reward：奖励。
- Policy：策略。

## **Reinforcement Learning Loop**

**Agent**

根据策略选择行动

**Action**

**Environment**

环境状态发生变化

State + Reward

**Agent Update**

更新价值估计或策略

## **7.2 MDP：马尔可夫决策过程**

强化学习常用 MDP 建模：

\\[ MDP=(S,A,P,R,\gamma) \\]

其中：

| 符号           | 含义     |
| ------------ | ------ |
| \\(S\\)      | 状态空间   |
| \\(A\\)      | 行动空间   |
| \\(P\\)      | 状态转移概率 |
| \\(R\\)      | 奖励函数   |
| \\(\gamma\\) | 折扣因子   |

状态转移：

\\[ P(s\_{t+1}|s_t,a_t) \\]

奖励：

\\[ R(s_t,a_t,s\_{t+1}) \\]

MDP 假设当前状态包含预测未来所需的信息，即具有 Markov 性质。

如果 Agent 只能观察环境的一部分，可能需要 POMDP（部分可观测马尔可夫决策过程），并维护 Belief State。

## **7.3 Agent 与 RL 的关系**

不是所有 Agent 都需要强化学习。

例如：

一个使用 LLM 进行任务规划、通过工具调用完成工作的 Agent，可以没有任何在线策略梯度训练。

它仍然是 Agent。

强化学习是 Agent 改进策略的一种方法，不是 Agent 的必要条件。

# **八、Online Learning：在线学习**

传统训练通常先准备数据，再集中训练。

但 Agent 运行过程中，会不断产生新经验。

这就引出了 Online Learning。

## **8.1 什么是在线学习？**

Online Learning 指系统在数据逐步到达时，持续更新模型或预测策略。

与 Batch Learning 相比：

| Batch Learning | Online Learning |
| -------------- | --------------- |
| 一次收集大量数据       | 数据持续到达          |
| 集中训练           | 增量更新            |
| 更新周期较长         | 更新频率可以更高        |
| 依赖固定数据集        | 面对动态数据流         |

例如：

```
第 1 天：
Agent 修复了 10 个代码问题。

第 2 天：
Agent 修复了 20 个代码问题。

第 3 天：
Agent 发现某类问题的旧策略效果下降。

第 4 天：
系统更新工具选择策略。
```

## **8.2 在线学习的数学形式**

假设每次收到样本：

\\[ (x_t,y_t) \\]

系统更新参数：

\\[ \theta\_{t+1} = \theta_t-\eta\nabla\_\theta \mathcal{L}(\theta_t;x_t,y_t) \\]

其中：

- \\(\theta_t\\)：当前参数。
- \\(\eta\\)：学习率。
- \\(\mathcal{L}\\)：损失函数。

但真实系统中，新样本可能有噪声、错误标签或恶意输入。

所以不能将 Agent 所有执行结果直接用于训练。

## **8.3 在线学习的安全问题**

假设 Agent 错误地删除了一个重要文件。

如果系统把这次行为当成成功经验保存，并在后续学习中强化，就可能持续放大错误。

建议建立数据审核流程：

```
Agent Experience
       │
       ▼
Evidence Collection
       │
       ▼
Quality Evaluation
       │
       ▼
Human / Automated Review
       │
       ▼
Approved Learning Data
       │
       ▼
Model / Policy Update
       │
       ▼
Offline Evaluation
       │
       ▼
Controlled Deployment
```

在线产生经验，不意味着应该立即更新模型权重。

对于高风险 Agent，采用离线训练、离线评估和受控发布通常更可靠。

# **九、Continual Learning：持续学习**

Online Learning 关注数据持续到达。

Continual Learning 更关注：

**系统如何不断学习新任务，同时尽量保留过去已经获得的能力？**

## **9.1 灾难性遗忘**

假设模型先学习：

```
Task A：Python 编程
Task B：SQL 查询
Task C：React 开发
```

如果模型只在 Task C 数据上持续训练，可能导致 Task A 和 Task B 的表现下降。

这种现象叫：

Catastrophic Forgetting（灾难性遗忘）。

## **9.2 为什么会遗忘？**

神经网络使用共享参数学习多个任务。

假设模型参数：

\\[ \theta=(\theta_1,\theta_2,\ldots,\theta_n) \\]

新任务训练会改变这些参数。

但某些参数可能同时承担旧任务的重要功能。

更新新任务时：

\\[ \theta\_{new} = \theta\_{old} - \eta\nabla\_\theta L\_{new} \\]

可能破坏旧任务已经学到的表示。

## **9.3 Continual Learning 的常见方法**

### **方法一：Replay**

保存历史数据，训练新任务时混合旧任务样本。

```
New Task Data
      +
Historical Replay Data
      │
      ▼
Combined Training
```

优点：

- 思路简单。
- 通常有较好的效果。

缺点：

- 需要保存历史数据。
- 数据规模可能越来越大。
- 隐私和数据治理复杂。

### **方法二：Regularization**

限制重要参数发生过大变化。

例如 EWC（Elastic Weight Consolidation）。

其基本思想是：

\\[ L(\theta)= L\_{new}(\theta) + \frac{\lambda}{2} \sum_i F_i(\theta_i-\theta_i^\*)^2 \\]

其中：

- \\(\theta_i^\*\\)：旧任务训练得到的参数。
- \\(F_i\\)：参数重要性估计，经典 EWC 使用 Fisher Information 的对角近似。
- \\(\lambda\\)：保护旧知识的强度。

参数越重要，偏离旧参数的代价越大。

### **方法三：Parameter Isolation**

为不同任务保留独立或部分独立的参数。

例如：

- Adapter。
- LoRA。
- 独立任务头。
- 模块化网络。

这样可以降低任务之间的参数干扰。

但它们也会增加模型管理复杂度，并不能自动解决所有持续学习问题。

## **9.4 Agent 层面的持续学习**

对于 Agent，持续学习不仅是神经网络参数更新。

还可以是：

```
Agent Continual Learning
   │
   ├── Memory Update
   ├── Knowledge Update
   ├── Tool Experience Update
   ├── Planning Strategy Update
   ├── Reward Model Update
   └── Model Parameter Update
```

这意味着 Agent 可以在不修改基础 LLM 权重的情况下，逐步提高系统能力。

# **十、Self-improvement：自我改进**

前面介绍了经验回放、策略优化和持续学习。

现在讨论一个更复杂的问题：

**Agent 能否自己发现能力不足，设计改进方法，并验证改进是否有效？**

这就是 Self-improvement。

## **10.1 什么是自我改进？**

可以将自我改进抽象为：

\\[ System\_{t+1} = Improve(System_t,Experience_t) \\]

但关键在于：

系统必须能够判断：

- 自己哪里做得不好。
- 应该改进什么。
- 改进方案是否合理。
- 改进后是否真的更好。

否则，自我改进可能只是一种自我评价。

## **10.2 Self-Reflection 与 Self-improvement**

二者有明显区别。

| Self-Reflection | Self-improvement |
| --------------- | ---------------- |
| 识别错误            | 改变系统             |
| 分析原因            | 实施改进             |
| 生成建议            | 验证改进             |
| 可能没有实际变化        | 必须产生可测量变化        |

例如：

```
Reflection：

我发现过去经常在修改代码之前，
没有充分理解项目依赖关系。
```

这只是反思。

如果随后：

1. 增加依赖关系分析步骤。
2. 更新任务规划策略。
3. 在测试任务中应用新策略。
4. 发现错误率下降。

这才构成有证据支持的改进过程。

## **10.3 Self-improvement Loop**

**Experience Collection**

收集任务经验

**Performance Analysis**

分析能力缺陷

**Improvement Proposal**

提出改进方案

**Controlled Update**

在隔离环境中实施更新

**Evaluation**

使用独立测试集验证

**Deployment / Rollback**

通过验证则发布，否则回滚

## **10.4 一个重要问题：如何避免自我欺骗？**

假设 Agent 修改了自己的代码。

修改后，它使用自己设计的测试证明修改成功。

但测试可能遗漏关键问题，甚至被修改代码本身绕过。

这会形成评价闭环：

```
Agent 修改自己
      ↓
Agent 设计测试
      ↓
Agent 评价自己
      ↓
Agent 宣布成功
```

这样的结果不一定可信。

更可靠的方案是：

- 使用独立测试集。
- 使用外部评估器。
- 保留基准版本。
- 进行 A/B 对比。
- 设置回归测试。
- 对高风险变更进行人工审核。
- 记录所有改进的来源和效果。

**自我改进必须接受独立验证，而不能只依靠 Agent 自己宣布成功。**

# **十一、Learning Engine 与 Memory Engine 的关系**

前面我们已经研究过 Memory Architecture。

现在需要明确二者的边界。

Memory 负责保存和检索信息。

Learning 负责利用经验改变未来行为。

**Memory Engine**

保存经验、知识、事件和技能

**Learning Engine**

分析经验、评价策略、更新能力

**Agent Engine**

应用改进后的能力执行任务

举个例子：

Memory 保存：

```
过去 20 次任务中，
先分析依赖关系的任务成功率较高。
```

Learning 分析：

```
在依赖复杂的项目中，
先分析依赖关系可能降低修改错误率。
```

Agent 应用：

```
遇到复杂项目时，
优先执行依赖分析。
```

这三个过程分别是：

- 记住经验。
- 从经验中提取规律。
- 将规律应用到未来任务。

它们不是同一个过程。

# **十二、构建 mini-llm Learning Engine**

现在把前面的理论转化为工程架构。

## **12.1 模块目录**

```
mini-llm/
│
├── engines/
│   │
│   ├── agent/
│   │
│   ├── learning/
│   │   ├── experience/
│   │   │   ├── experience.py
│   │   │   ├── trajectory.py
│   │   │   └── collector.py
│   │   │
│   │   ├── replay/
│   │   │   ├── buffer.py
│   │   │   ├── sampler.py
│   │   │   └── prioritizer.py
│   │   │
│   │   ├── credit_assignment/
│   │   │   ├── reward.py
│   │   │   ├── advantage.py
│   │   │   └── evaluator.py
│   │   │
│   │   ├── reward_model/
│   │   │   ├── model.py
│   │   │   └── preference.py
│   │   │
│   │   ├── policy/
│   │   │   ├── policy.py
│   │   │   ├── optimizer.py
│   │   │   └── ppo.py
│   │   │
│   │   ├── continual/
│   │   │   ├── replay_learning.py
│   │   │   ├── regularization.py
│   │   │   └── task_manager.py
│   │   │
│   │   ├── improvement/
│   │   │   ├── analyzer.py
│   │   │   ├── proposal.py
│   │   │   └── validator.py
│   │   │
│   │   └── evaluation/
│   │       ├── benchmark.py
│   │       ├── regression.py
│   │       └── report.py
│   │
└── tests/
    ├── test_experience.py
    ├── test_replay.py
    ├── test_reward.py
    ├── test_policy.py
    └── test_learning.py
```

## **12.2 Learning Engine 核心流程**

```
class LearningEngine:    def __init__(        self,        experience_store,        evaluator,        strategy_updater,        benchmark    ):        self.experience_store = experience_store        self.evaluator = evaluator        self.strategy_updater = strategy_updater        self.benchmark = benchmark    def learn(self, task_id):        # 1. 收集经验        experiences = (            self.experience_store            .get_by_task(task_id)        )        # 2. 评估经验质量        evaluated = self.evaluator.evaluate(            experiences        )        # 3. 提取改进建议        proposal = self.strategy_updater.propose(            evaluated        )        # 4. 在隔离环境中测试        baseline = self.benchmark.run(            version="current"        )        candidate = self.benchmark.run(            proposal=proposal        )        # 5. 对比结果        if candidate.is_better_than(baseline):            return {                "status": "candidate_approved",                "proposal": proposal            }        return {            "status": "rejected",            "reason": "No measurable improvement"        }
```

这是一个架构级简化示例。

真正的系统还需要定义：

- 经验质量标准。
- 训练数据审核。
- 策略版本管理。
- 独立评估数据集。
- 回滚机制。
- 部署审批。
- 统计显著性和置信区间。

尤其不能仅凭一次测试得分上升就认为系统能力已经提升。

# **十三、如何证明 Agent 真的学会了？**

这是整个 Learning Engine 最重要的工程问题。

假设 Agent 第一次完成任务用了 30 步。

第二次用了 20 步。

第三次用了 15 步。

能否证明 Agent 越来越聪明？

不能直接这样判断。

因为：

- 第二次任务可能更简单。
- 环境可能发生变化。
- 工具可能升级。
- 模型可能换了版本。
- 任务数据可能泄漏。

因此需要建立系统化评估。

## **13.1 核心评价指标**

| 指标                    | 含义     |
| --------------------- | ------ |
| Task Success Rate     | 任务成功率  |
| Generalization        | 泛化能力   |
| Sample Efficiency     | 样本效率   |
| Learning Curve        | 学习曲线   |
| Regression Rate       | 旧任务退化率 |
| Recovery Rate         | 故障恢复率  |
| Cost per Task         | 单任务成本  |
| Safety Violation Rate | 安全违规率  |

## **13.2 Learning Curve**

学习曲线表示系统随着经验增加，任务表现如何变化。

Agent Learning Curve（示意）

横轴为累计训练经验，纵轴为测试任务成功率；数据仅用于展示学习曲线概念。

0%25%50%75%100%0100200300400500600700800

如果经验增加，但独立测试任务的表现没有提升，就不能证明学习有效。

## **13.3 Generalization：泛化能力**

例如：

训练经验：

```
修复 React 项目中的类型错误。
```

测试任务：

```
修复 Vue 项目中的类型错误。
```

如果 Agent 只记住 React 项目的具体解决方案，就不是真正的泛化。

如果它能够迁移：

- 类型依赖分析。
- 错误定位。
- 测试驱动修复。
- 修改后验证。

那么它可能学到了更通用的解决问题策略。

但需要通过足够多的不同项目和独立任务验证。

## **13.4 Regression：旧能力是否退化？**

学习新能力时，还要测试过去的能力。

例如：

| 任务            | 学习前 | 学习后 |
| ------------- | --- | --- |
| Python 修复     | 90% | 89% |
| SQL 查询        | 85% | 86% |
| React 开发      | 80% | 88% |
| TypeScript 修复 | 70% | 85% |

这是一种理想的结果：新能力提升，旧能力基本保留。

如果新任务成功率提高，但旧任务成功率大幅下降，就需要重新考虑学习策略。

# **十四、Agent Learning 的完整架构**

现在将 Agent Engine 与 Learning Engine 组合起来。

## **mini-llm Agent Learning Architecture**

**Agent Engine**

理解目标、规划任务、调用工具、执行行动

**Environment Interaction**

获取观察、行动结果和环境反馈

**Experience Collection**

Experience、Trajectory、Execution Trace

**Memory**

保存与检索经验

**Evaluation**

评价行动质量

**Learning Engine**

Replay、Credit Assignment、Policy Optimization、Continual Learning

**Improvement Validation**

独立评估、回归测试、风险审核

**Updated Agent Policy**

应用经过验证的新策略

进入下一轮任务，继续积累经验

这就形成了一个完整的闭环：

\\[ \boxed{ Experience \rightarrow Evaluation \rightarrow Learning \rightarrow Improvement \rightarrow New\ Experience } \\]

# **十五、mini-llm 下一阶段的研究方向**

到目前为止，mini-llm 已经覆盖了从底层张量计算到高层智能体学习的一系列模块。

可以把整个项目划分为三个层次。

**第一层：模型计算与生成**

Tensor、Autograd、Neural Network、Transformer、Decoder、Training、Evaluation

**第二层：认知与世界建模**

Representation、Memory、Knowledge、World Model、Causal Reasoning、Decision、Planning

**第三层：行动与持续改进**

Agent、Tool Calling、Runtime、Sandbox、Multi-agent、Learning、Self-improvement

但还有一个重要方向值得进一步研究：

**Agent 如何管理自己的计算资源、模型选择、工具调用和执行成本？**

例如：

- 简单任务是否应该使用小模型？
- 复杂推理是否需要调用大模型？
- 是否需要搜索外部知识？
- 是否需要启动多个 Agent？
- 什么时候应该继续推理？
- 什么时候应该停止？
- 如何在有限预算内提高任务完成率？

这就涉及智能系统的资源分配和计算控制。

下一篇建议：

# **《mini-llm Meta-Controller 深度解析：从 Metacognition、Dynamic Model Routing 到 Compute Allocation、Inference-time Scaling 与 Adaptive Reasoning》**

建议研究：

1. Meta-controller：元控制器是什么？
2. Metacognition：如何监控自身认知过程？
3. Dynamic Model Routing：如何动态选择模型？
4. Model Cascading：如何从小模型逐步升级到大模型？
5. Compute Allocation：如何分配推理计算资源？
6. Inference-time Scaling：如何通过增加推理计算改善结果？
7. Adaptive Reasoning：如何动态调整推理深度？
8. Tool Selection：何时使用工具？
9. Stop Decision：何时停止推理和执行？
10. Budget-aware Planning：如何在成本预算内规划任务？
11. Multi-agent Orchestration：如何决定是否启用多个 Agent？
12. Meta-Controller Architecture：如何实现整个系统的元控制层？

**这一步将让 mini-llm 从一个能够执行和学习的 Agent 系统，进一步发展为能够动态管理自身计算、推理和行动资源的智能系统。**

