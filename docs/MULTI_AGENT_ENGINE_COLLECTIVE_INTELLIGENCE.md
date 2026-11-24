# **《mini-llm Multi-Agent Engine 深度解析：从 Agent Collaboration、Task Delegation 到 Consensus、Negotiation 与 Collective Intelligence》**

## **一、Multi-Agent System：为什么需要多个 Agent？**

前面我们研究了 Agent Engine。

一个 Agent 可以理解任务、制定计划、调用工具、执行操作，并根据反馈调整行为。

但当任务越来越复杂时，单个 Agent 会遇到一些问题：

- 任务涉及多个专业领域，单个 Agent 难以兼顾。
- 上下文越来越长，信息管理成本增加。
- 多个子任务可以并行执行，但单个 Agent 通常需要协调执行顺序。
- 复杂决策需要不同视角进行分析。
- 一个 Agent 可能产生错误判断，需要其他 Agent 进行验证。
- 不同任务需要不同模型、工具和执行权限。

例如，让 AI 开发一个完整的软件系统：

```
用户需求：
开发一个企业级 SaaS 管理平台

涉及工作：

1. 需求分析
2. 系统架构设计
3. 数据库设计
4. 前端开发
5. 后端开发
6. 测试
7. 安全审查
8. 部署
```

如果全部交给一个 Agent：

```
User
  |
  v
Single Agent
  |
  +-- Requirement Analysis
  +-- Architecture
  +-- Database
  +-- Frontend
  +-- Backend
  +-- Testing
  +-- Security
  +-- Deployment
```

随着任务复杂度增加，Agent 需要维护越来越多的上下文和任务状态。

另一种方式是将任务交给多个专业 Agent：

```
                   User
                    |
                    v
              Manager Agent
                    |
        +-----------+-----------+
        |           |           |
        v           v           v
   Architect     Backend     Frontend
     Agent        Agent        Agent
        |           |           |
        v           v           v
   Architecture  API Code    UI Code
        |           |           |
        +-----------+-----------+
                    |
                    v
                QA Agent
                    |
                    v
              Security Agent
                    |
                    v
               Final Result
```

这就是 Multi-Agent System（多智能体系统）。

**Multi-Agent 的核心不是让多个 AI 同时聊天，而是让多个具有不同职责、上下文、工具和执行权限的智能体，通过协作完成单个 Agent 难以完成的复杂任务。**

## **二、什么是 Agent？**

首先明确 Agent 的基本结构。

一个 Agent 不只是一个 LLM。

它通常由以下组件组成：

\\[ Agent = Model + Context + Policy + Tools + Memory \\]

其中：

| 组件      | 作用      |
| ------- | ------- |
| Model   | 理解和生成信息 |
| Context | 当前任务上下文 |
| Policy  | 决定如何行动  |
| Tools   | 与外部系统交互 |
| Memory  | 保存历史信息  |

一个 Agent 可以抽象成：

```
class Agent:    def __init__(        self,        model,        policy,        tools,        memory    ):        self.model = model        self.policy = policy        self.tools = tools        self.memory = memory    def run(self, task):        context = self.memory.retrieve(task)        action = self.policy.decide(            task,            context        )        result = self.execute(action)        self.memory.store(result)        return result
```

实际系统还需要处理任务状态、超时、权限、错误恢复和执行审计。

### **2.1 Single-Agent 与 Multi-Agent**

| 对比   | Single-Agent   | Multi-Agent |
| ---- | -------------- | ----------- |
| 执行主体 | 一个 Agent       | 多个 Agent    |
| 专业分工 | 通过 Prompt 切换角色 | 可以独立配置      |
| 上下文  | 通常集中管理         | 可以分布管理      |
| 并行能力 | 取决于内部调度        | 更容易实现独立任务并行 |
| 协作成本 | 较低             | 较高          |
| 一致性  | 相对容易维护         | 需要协调        |
| 故障影响 | 可能影响整个任务       | 可以隔离部分任务    |
| 资源成本 | 通常较低           | 可能更高        |

需要特别注意：

多个 Agent 不一定比一个 Agent 更智能。

如果任务本身简单，增加 Agent 只会增加：

- Token 消耗。
- 通信成本。
- 状态同步成本。
- 任务协调复杂度。
- 错误传播风险。

因此，Multi-Agent 的价值在于合理分工，而不是 Agent 数量。

## **三、Agent Collaboration：多个 Agent 如何协作？**

Agent Collaboration 是多智能体系统最核心的能力之一。

多个 Agent 需要共享目标、交换信息、协调任务和整合结果。

常见协作模式有五种。

### **3.1 Sequential Collaboration：顺序协作**

一个 Agent 完成任务后，将结果交给下一个 Agent。

**Sequential Collaboration**

**Requirement Agent**

分析用户需求

**Architect Agent**

设计系统架构

**Developer Agent**

实现功能

**QA Agent**

测试和验证

适合：

- 文档生成。
- 软件开发流程。
- 数据处理 Pipeline。
- 代码审查与发布。

优点是流程清晰，容易追踪。

缺点是前一个 Agent 失败可能阻塞后续任务。

### **3.2 Parallel Collaboration：并行协作**

多个 Agent 同时执行相互独立的任务。

例如：

```
                 Manager Agent
                       |
          +------------+------------+
          |            |            |
          v            v            v
     Frontend       Backend       Security
       Agent          Agent         Agent
          |            |            |
          v            v            v
      UI Module     API Module   Security Report
          |            |            |
          +------------+------------+
                       |
                       v
                 Integration
```

适合：

- 独立模块开发。
- 多源信息收集。
- 多方案分析。
- 并行代码审查。

但是并行任务必须具备明确边界。

例如，Frontend Agent 和 Backend Agent 同时修改同一个 API 契约，就可能产生冲突。

因此，并行协作需要：

- 任务依赖图。
- 文件所有权。
- 接口契约。
- 共享资源锁。
- 合并和验证机制。

### **3.3 Hierarchical Collaboration：层级协作**

多个 Agent 按照管理层级组织。

**Hierarchical Multi-Agent**

**Executive Agent**

负责总体目标与全局决策

**Planning Agent**

任务拆解

**Review Agent**

结果审查

**Frontend**

开发

**Backend**

开发

**Testing**

测试

适合大型任务和多层级组织。

优点是容易控制整体流程。

缺点是容易形成过度中心化，顶层 Agent 可能成为性能瓶颈。

### **3.4 Debate Collaboration：辩论协作**

多个 Agent 从不同角度分析同一个问题。

例如：

用户问：

```
这个项目应该使用 PostgreSQL 还是 MySQL？
```

可以安排：

- Agent A：分析 PostgreSQL。
- Agent B：分析 MySQL。
- Agent C：分析运维和成本。
- Agent D：作为评审者综合比较。

最终输出：

```
PostgreSQL：
优点：复杂查询、JSONB、扩展能力。

MySQL：
优点：生态成熟、团队熟悉度高。

综合建议：
根据项目的数据模型、团队经验和运维条件选择。
```

辩论的价值是增加不同观点和错误检查机会。

但多个 Agent 达成一致，不代表结论就一定正确。

如果它们使用相同模型、相同 Prompt 和相同错误信息，很可能只是重复同一种偏差。

### **3.5 Blackboard Collaboration：黑板协作**

多个 Agent 共享一个公共工作空间。

每个 Agent 可以读取已有信息，并提交自己的结果。

**Blackboard Architecture**

**Agent A**

需求分析

**Agent B**

技术分析

**Agent C**

数据分析

**Agent D**

风险分析

**Shared Blackboard**

公共任务状态、事实、证据、决策和产物

**Coordinator**

判断信息是否完整，决定下一步工作

Blackboard 的优势是多个 Agent 不需要彼此直接建立复杂通信关系。

但必须设计：

- 数据版本。
- 写入权限。
- 冲突解决。
- 信息来源。
- 状态一致性。

## **四、Task Delegation：如何把复杂任务分配给多个 Agent？**

多 Agent 系统首先要解决一个问题：

**谁负责什么任务？**

这就是 Task Delegation。

### **4.1 从 Goal 到 Subtask**

例如：

```
Goal：
开发一个用户管理系统
```

Manager Agent 可以将其拆解为：

```
Task Graph

T1：需求分析
 |
 +--> T2：数据库设计
 |
 +--> T3：API 设计
          |
          +--> T4：后端实现
          |
          +--> T5：前端实现
                    |
                    +--> T6：集成测试
                              |
                              +--> T7：安全检查
                                        |
                                        +--> T8：交付
```

但真实依赖关系应该由任务内容决定。

例如前端和后端实现可能并行，但双方需要先确定 API 契约。

因此，任务结构通常不是简单的树，而是 DAG（有向无环图）。

### **4.2 Task DAG**

定义：

\\[ G=(V,E) \\]

其中：

- \\(V\\)：任务集合。
- \\(E\\)：依赖关系集合。

如果：

\\[ (T_1,T_2)\in E \\]

表示 \\(T_2\\) 必须等待 \\(T_1\\) 完成。

例如：

```
T1: Requirement Analysis
 |
 v
T2: API Contract
 |          |
 v          v
T3: Backend T4: Frontend
 |          |
 +-----+----+
       |
       v
    T5: Integration
       |
       v
    T6: Testing
```

这类 DAG 可以交给 Runtime Engine 的 Scheduler 调度。

### **4.3 Task Schema**

每个任务应该具备统一的数据结构。

```
class AgentTask:    id: str    title: str    description: str    assigned_agent: str    dependencies: list[str]    input_context: dict    expected_output: dict    priority: int    deadline: str | None    status: str    retry_limit: int
```

这里最重要的是 `expected_output`。

例如：

```
{
  "task": "设计用户管理 API",
  "expected_output": {
    "type": "api_contract",
    "required": [
      "endpoints",
      "request_schema",
      "response_schema",
      "error_codes"
    ]
  }
}
```

这样 Manager Agent 才能判断任务是否真正完成，而不是仅仅收到一段看起来合理的文字。

### **4.4 Agent Selection**

如何选择合适的 Agent？

可以考虑：

\\[ Score(A_i,T)= w_1 C_i+ w_2 S_i+ w_3 R_i- w_4 Cost_i \\]

其中：

- \\(C_i\\)：任务能力匹配度。
- \\(S_i\\)：当前可用性。
- \\(R_i\\)：历史可靠性。
- \\(Cost_i\\)：执行成本。

例如：

| Agent          | 专业能力 | 历史成功率 | 成本 |
| -------------- | ---- | ----- | -- |
| Frontend Agent | 0.95 | 0.91  | 低  |
| Backend Agent  | 0.92 | 0.89  | 中  |
| General Agent  | 0.72 | 0.80  | 低  |

对于前端任务，Frontend Agent 显然更合适。

不过这些分数只是工程评价指标，不应直接等同于真实能力概率。需要使用历史任务和独立评估数据校准。

## **五、Agent Communication：多个 Agent 如何通信？**

多个 Agent 不能只依靠自然语言聊天。

大型系统需要明确的通信协议。

### **5.1 Point-to-Point Communication**

Agent 之间直接发送消息。

```
Agent A  ------> Agent B
Agent B  ------> Agent C
Agent C  ------> Agent A
```

优点：

- 简单。
- 直接。
- 适合小规模系统。

缺点：

- Agent 数量增加后，通信关系迅速复杂化。
- 难以统一审计。
- 难以处理离线 Agent 和消息重试。

如果 \\(N\\) 个 Agent 之间建立完全连接：

\\[ E=\frac{N(N-1)}{2} \\]

当 Agent 数量增加时，通信关系会迅速增长。

### **5.2 Message Bus**

另一种方式是引入消息总线。

**Message Bus Architecture**

Agent A

Agent B

Agent C

Agent D

**Message Bus**

路由、订阅、消息持久化、重试和追踪

**Task / Event / Result Channels**

可以使用：

- Redis Streams。
- RabbitMQ。
- NATS。
- Kafka。

小规模 V1 可以先使用进程内消息队列和数据库持久化，不必一开始引入完整分布式消息基础设施。

### **5.3 Agent Message Schema**

可以定义：

```
class AgentMessage:    message_id: str    sender: str    receiver: str | None    message_type: str    task_id: str    correlation_id: str    payload: dict    created_at: str
```

消息类型可以包括：

| 类型            | 作用     |
| ------------- | ------ |
| TaskRequest   | 请求执行任务 |
| TaskAccepted  | 接受任务   |
| TaskRejected  | 拒绝任务   |
| TaskProgress  | 报告进度   |
| TaskResult    | 返回结果   |
| TaskFailed    | 报告失败   |
| AgentEvent    | 发布事件   |
| ReviewRequest | 请求审查   |

注意区分：

- Command：要求某个 Agent 执行动作。
- Event：描述已经发生的事情。
- Request/Response：请求与响应。

消息可能重复投递，因此消费端必须设计幂等处理。

### **5.4 MCP 与 A2A 的关系**

在 mini-llm 中，值得区分两个协议方向。

**MCP（Model Context Protocol）**

主要解决 AI 应用如何连接工具、资源和 Prompt。

例如：

```
Agent
  |
  v
MCP Client
  |
  v
MCP Server
  |
  +-- Database
  +-- Git
  +-- Filesystem
  +-- Browser
```

**A2A（Agent2Agent）**

主要解决不同 Agent 系统之间如何发现、委派任务和交换任务状态。

例如：

```
Agent Platform A
       |
       v
   A2A Protocol
       |
       v
Agent Platform B
```

两者不是替代关系：

- MCP：Agent 如何使用工具和资源。
- A2A：Agent 如何与其他 Agent 协作。

如果 mini-llm 后续希望接入外部 Agent 平台，就应该考虑 A2A，而不应把所有 Agent 通信都设计成私有协议。

## **六、Consensus：多个 Agent 如何达成共识？**

假设三个 Agent 对同一个问题提出不同结论：

```
问题：是否应该引入 Redis？

Agent A：应该引入。
Agent B：没有必要。
Agent C：应该先使用数据库缓存。
```

系统如何决定？

这涉及 Consensus（共识）。

### **6.1 Majority Voting：多数投票**

最简单的方法：

\\[ Decision=\operatorname{Mode}(a_1,a_2,\ldots,a_n) \\]

例如：

```
Agent A：方案 X
Agent B：方案 X
Agent C：方案 Y

最终：方案 X
```

优点：

- 简单。
- 容易实现。
- 可以降低单个 Agent 偶然判断错误的影响。

缺点：

- 多数意见不一定正确。
- Agent 可能存在共同偏差。
- 无法充分表达不同意见的质量差异。

### **6.2 Weighted Voting：加权投票**

不同 Agent 根据其在特定任务上的历史表现获得不同权重。

\\[ Score(a)= \sum\_{i=1}^{n}w_i I(a_i=a) \\]

其中：

- \\(w_i\\)：Agent 的权重。
- \\(I\\)：指示函数。

例如：

| Agent          | 权重  | 方案 |
| -------------- | --- | -- |
| Database Agent | 0.5 | X  |
| Backend Agent  | 0.3 | X  |
| General Agent  | 0.2 | Y  |

方案 X 获得 0.8 的加权支持。

但不能简单认为某个 Agent 永远更可靠。

一个 Agent 可能擅长数据库设计，却不擅长安全分析。

因此，权重应与具体任务领域相关。

### **6.3 Evidence-based Consensus：基于证据的共识**

我更建议 mini-llm 优先采用这种方式。

不是统计有多少 Agent 同意，而是检查：

1. 结论是否有证据？
2. 证据是否来自可信来源？
3. 推理过程是否存在逻辑错误？
4. 是否有反例？
5. 是否经过实际测试？

例如：

```
方案 X：

支持证据：
- 当前并发量较低
- PostgreSQL 已支持所需查询
- 现有系统没有 Redis 依赖

反对证据：
- 未来需要跨服务共享缓存
- 可能出现热点查询

最终决策：
V1 暂不引入 Redis。
达到性能阈值后重新评估。
```

这种共识机制比简单投票更适合工程系统。

### **6.4 Consensus 不等于分布式共识算法**

这里还需要区分两种完全不同的概念。

**AI Agent Consensus：**

多个智能体通过讨论、投票和证据审查，形成一个决策。

**Distributed Consensus：**

分布式节点通过协议，在网络延迟、节点故障等条件下，对系统状态达成一致。

例如：

- Raft。
- Paxos。

前者解决意见和决策问题。

后者解决分布式系统中的状态一致性问题。

不要将多个 LLM Agent 投票称为 Raft 或 Paxos 意义上的分布式共识。

## **七、Negotiation：多个 Agent 如何处理利益冲突？**

共识解决的是意见不一致。

Negotiation 更进一步：

**当多个 Agent 的目标或资源需求发生冲突时，如何协商？**

例如：

一个 SaaS 项目中：

- Frontend Agent 希望增加更多页面交互。
- Backend Agent 希望减少 API 复杂度。
- Security Agent 希望限制数据访问。
- Performance Agent 希望减少请求次数。

这些目标可能相互冲突。

### **7.1 多目标优化**

可以将 Agent 的目标表示为：

\\[ U_i(x) \\]

其中：

- \\(i\\)：Agent。
- \\(x\\)：系统方案。
- \\(U_i\\)：该 Agent 对方案的效用评价。

例如：

| Agent       | 关注目标    |
| ----------- | ------- |
| Frontend    | 交互体验    |
| Backend     | 开发和维护成本 |
| Security    | 安全风险    |
| Performance | 响应时间    |

一个方案可能对前端有利，却增加后端复杂度。

因此，不一定存在让所有 Agent 都满意的方案。

### **7.2 Pareto Optimality**

如果不存在另一个方案能够让至少一个 Agent 变得更好，同时不让其他 Agent 变差，那么当前方案就是 Pareto 最优方案。

可以写成：

\\[ \nexists x': U_i(x')\geq U_i(x),\ \forall i \\]

并且至少一个 Agent 严格改善。

这有助于分析多目标之间的权衡。

但 Pareto 最优不代表方案公平，也不代表方案符合安全要求。

安全和权限依然应当是硬约束。

### **7.3 Nash Bargaining**

在协商理论中，可以通过 Nash Bargaining 思想寻找双方都能接受的方案。

一种常见形式：

\\[ x^\*= \arg\max_x \prod_i \left(U_i(x)-d_i\right) \\]

其中：

- \\(U_i(x)\\)：Agent 在方案中的效用。
- \\(d_i\\)：无法达成协议时的保留效用。

直观理解：

寻找一个让参与者相对其最低可接受结果都有改善的方案。

在 mini-llm 中，可以用它作为多目标方案选择的参考，而不是要求 LLM 直接计算出真实的博弈均衡。

### **7.4 Negotiation Workflow**

```
Conflict Detection
       |
       v
Goal Identification
       |
       v
Constraint Collection
       |
       v
Proposal Generation
       |
       v
Counter Proposal
       |
       v
Evidence Evaluation
       |
       v
Final Decision
```

例如：

```
冲突：
前端需要实时更新订单状态。
后端不希望增加高频轮询。

协商：
前端提出 WebSocket。
后端评估连接维护成本。
安全 Agent 检查身份验证。
运维 Agent 评估部署成本。

最终：
采用 WebSocket + 心跳 + 断线重连。
```

## **八、Collective Intelligence：多个 Agent 如何形成群体智能？**

这是 Multi-Agent 最值得研究的理论问题。

单个 Agent 具有一定能力。

多个 Agent 通过协作，是否能够产生超过单体能力的系统效果？

这就是 Collective Intelligence（群体智能）。

### **8.1 个体智能与群体智能**

可以把群体智能理解为：

\\[ C=f(A_1,A_2,\ldots,A_n,G) \\]

其中：

- \\(A_i\\)：单个 Agent 的能力。
- \\(G\\)：协作结构。
- \\(C\\)：群体协作后的整体能力。

这里最重要的是 \\(G\\)。

同样的十个 Agent：

- 如果没有任务分工，可能重复工作。
- 如果缺少协调，可能互相冲突。
- 如果共享错误信息，可能集体犯错。
- 如果具有合理分工和验证机制，则可能获得更好的整体结果。

因此，群体智能并不是 Agent 数量的简单累加。

### **8.2 协作产生的能力增益**

可以建立一个概念性指标：

\\[ G_c = Q\_{\text{multi}}- Q\_{\text{single}} \\]

其中：

- \\(Q\_{\text{multi}}\\)：多 Agent 系统的任务质量。
- \\(Q\_{\text{single}}\\)：单 Agent 基线的任务质量。

如果：

\\[ G_c>0 \\]

说明在当前评价条件下，多 Agent 系统表现更好。

但还需要计算额外成本：

\\[ Efficiency= \frac{Q\_{\text{multi}}} {Cost\_{\text{multi}}} \\]

如果多 Agent 系统的质量只提高 2%，成本却增加 5 倍，那么是否值得使用，需要根据业务场景决定。

### **8.3 Diversity：多样性的重要性**

群体智能需要一定程度的差异性。

例如：

- 不同专业知识。
- 不同模型架构。
- 不同 Prompt。
- 不同工具。
- 不同信息来源。
- 不同评价标准。

如果所有 Agent 都使用同一个模型、相同上下文和相同策略，那么它们的判断可能高度相关。

这种情况下，增加 Agent 数量不一定能显著降低整体错误率。

一个值得研究的问题是：

**如何让多个 Agent 在保持协作能力的同时，拥有足够的独立判断能力？**

可以从以下方向实现：

1. 专业领域分工。
2. 独立信息检索。
3. 不同模型交叉验证。
4. 对抗性审查。
5. 证据来源隔离。
6. 独立评估。

## **九、Multi-Agent Coordination：如何解决冲突、死锁和重复执行？**

当多个 Agent 同时运行时，系统会出现一些传统分布式系统也会遇到的问题。

### **9.1 Task Conflict**

例如：

```
Agent A：修改 user.ts
Agent B：修改 user.ts
```

两个 Agent 同时修改同一个文件。

可能产生：

- Git 冲突。
- 代码覆盖。
- 逻辑不一致。
- 测试结果失效。

解决方案：

- 文件级任务所有权。
- Git Worktree 隔离。
- Patch-based 修改。
- 集成阶段统一合并。
- 合并后重新测试。

### **9.2 Deadlock**

假设：

```
Agent A 等待 Agent B 的结果。
Agent B 等待 Agent C 的结果。
Agent C 等待 Agent A 的结果。
```

形成循环等待：

\\[ A\rightarrow B\rightarrow C\rightarrow A \\]

这就是死锁。

解决方法：

- 任务 DAG 检测。
- 超时机制。
- 任务取消。
- 依赖关系检查。
- 避免循环等待。
- 为每个任务设置最大等待时间。

### **9.3 Duplicate Execution**

消息可能重复投递。

例如：

```
Agent A 请求支付。
消息发送成功。
响应丢失。
Agent A 重试。
Agent B 再次执行支付。
```

结果可能是重复支付。

因此，不能依赖消息系统保证所有业务动作只执行一次。

需要：

- 幂等键。
- 操作状态持久化。
- 去重记录。
- 事务边界。
- 对外部副作用进行结果核验。

例如：

```
def execute_task(task):    if already_completed(task.idempotency_key):        return get_previous_result(            task.idempotency_key        )    result = perform_action(task)    save_result(        task.idempotency_key,        result    )    return result
```

这只是简化示例。真实支付等操作需要依赖业务服务端的原子幂等处理，不能仅靠 Agent 侧的检查。

### **9.4 Agent Failure**

某个 Agent 可能：

- 超时。
- 模型服务不可用。
- 输出格式错误。
- 工具执行失败。
- 返回不可信信息。

因此 Multi-Agent Engine 需要：

| 机制              | 作用           |
| --------------- | ------------ |
| Timeout         | 防止无限等待       |
| Retry           | 处理临时错误       |
| Circuit Breaker | 隔离持续失败的服务    |
| Fallback        | 使用备用 Agent   |
| Checkpoint      | 保存任务进度       |
| Compensation    | 对可补偿副作用执行补偿  |
| Human Approval  | 对高风险任务进行人工审批 |

注意，重试不能解决所有错误。

例如，参数错误、权限拒绝和业务规则冲突通常不应该无限重试。

## **十、Multi-Agent Evaluation：如何评价群体协作效果？**

一个 Multi-Agent 系统可能看起来很复杂，但复杂并不代表有效。

需要回答几个问题：

1. 多个 Agent 是否真正提升了结果质量？
2. 是否减少了错误？
3. 是否提高了执行效率？
4. 是否产生了更多不必要的通信？
5. 是否增加了任务失败的概率？

### **10.1 核心指标**

| 指标                    | 含义           |
| --------------------- | ------------ |
| Task Success Rate     | 整体任务成功率      |
| Agent Contribution    | 单个 Agent 的贡献 |
| Coordination Overhead | 协调开销         |
| Message Count         | 消息数量         |
| Duplicate Work Rate   | 重复工作比例       |
| Conflict Rate         | 任务冲突比例       |
| Deadlock Rate         | 死锁比例         |
| Recovery Rate         | 故障恢复成功率      |
| Cost per Task         | 单任务总成本       |
| Latency               | 端到端延迟        |

### **10.2 单 Agent 与多 Agent 对照实验**

例如：

单 Agent 与多 Agent 任务成功率

假设实验数据，用于说明对照评估方法。

0%25%50%75%100%Single AgentSequentialParallelHierarchical

这些数值只是示意，不代表某种架构必然优于另一种。

真实评估应控制：

- 使用相同任务集。
- 尽可能控制模型和模型版本。
- 保持相同工具权限。
- 使用相同评价标准。
- 多次运行。
- 统计置信区间。
- 同时比较成本和延迟。

尤其需要避免一种常见误区：

如果多 Agent 系统调用了更多模型、进行了更多轮验证，那么它的结果可能更好，但这不一定意味着协作机制本身更有效。

## **十一、mini-llm Multi-Agent Engine 架构设计**

现在把前面的内容整合为一个工程架构。

**mini-llm Multi-Agent Engine**

**Agent Orchestrator**

整体任务协调与生命周期管理

**Task Planner**

任务拆解与 DAG

**Agent Registry**

Agent 注册与发现

**Agent Selector**

能力匹配与分配

**Message Router**

消息路由与通信

**Consensus Engine**

意见整合与证据审查

**Negotiation Engine**

冲突处理与多目标协商

**Coordination & State Manager**

任务状态、并发控制、超时、重试、恢复

**Agent Pool**

**Agent A**

**Agent B**

**Agent C**

**Evaluation & Result Aggregator**

验证子任务与整合最终结果

### **11.1 工程目录**

```
mini-llm/
├── engines/
│   ├── multi_agent/
│   │   ├── orchestrator/
│   │   │   ├── manager.py
│   │   │   ├── lifecycle.py
│   │   │   └── coordinator.py
│   │   │
│   │   ├── planner/
│   │   │   ├── task_decomposer.py
│   │   │   ├── dag.py
│   │   │   └── scheduler.py
│   │   │
│   │   ├── registry/
│   │   │   ├── agent_registry.py
│   │   │   └── capability.py
│   │   │
│   │   ├── delegation/
│   │   │   ├── selector.py
│   │   │   ├── dispatcher.py
│   │   │   └── assignment.py
│   │   │
│   │   ├── communication/
│   │   │   ├── message.py
│   │   │   ├── router.py
│   │   │   ├── bus.py
│   │   │   └── protocol.py
│   │   │
│   │   ├── collaboration/
│   │   │   ├── sequential.py
│   │   │   ├── parallel.py
│   │   │   ├── hierarchical.py
│   │   │   ├── debate.py
│   │   │   └── blackboard.py
│   │   │
│   │   ├── consensus/
│   │   │   ├── voting.py
│   │   │   ├── weighted.py
│   │   │   └── evidence.py
│   │   │
│   │   ├── negotiation/
│   │   │   ├── conflict.py
│   │   │   ├── proposal.py
│   │   │   └── resolution.py
│   │   │
│   │   ├── coordination/
│   │   │   ├── state_manager.py
│   │   │   ├── lock_manager.py
│   │   │   ├── timeout.py
│   │   │   ├── retry.py
│   │   │   └── recovery.py
│   │   │
│   │   └── evaluation/
│   │       ├── evaluator.py
│   │       ├── contribution.py
│   │       └── benchmark.py
│   │
│   └── agent/
│       ├── agent.py
│       ├── model.py
│       ├── tools.py
│       └── memory.py
│
├── protocols/
│   ├── mcp/
│   └── a2a/
│
├── tests/
│   ├── test_delegation.py
│   ├── test_collaboration.py
│   ├── test_consensus.py
│   ├── test_coordination.py
│   └── test_multi_agent.py
│
└── examples/
    ├── sequential_agents.py
    ├── parallel_agents.py
    └── software_team.py
```

## **十二、Multi-Agent Engine 的最小实现**

先实现一个 Manager Agent 和三个专业 Agent。

```
class MultiAgentEngine:    def __init__(        self,        manager,        agents,        scheduler,        evaluator    ):        self.manager = manager        self.agents = agents        self.scheduler = scheduler        self.evaluator = evaluator    async def run(self, goal):        tasks = await self.manager.decompose(            goal        )        assignments = self.manager.assign(            tasks,            self.agents        )        results = await self.scheduler.execute(            assignments        )        evaluation = await self.evaluator.evaluate(            goal,            results        )        if not evaluation.passed:            return {
```

专业 Agent：

```
class SpecializedAgent:    def __init__(        self,        name,        model,        capabilities,        tools    ):        self.name = name        self.model = model        self.capabilities = capabilities        self.tools = tools    async def execute(self, task):        context = await self.build_context(            task        )        result = await self.model.generate(            context        )        return await self.validate(            result,            task        )
```

需要强调，这只是系统骨架。

生产环境还需要：

- Agent 状态持久化。
- 任务 DAG 调度。
- 消息幂等。
- 权限隔离。
- 超时和取消。
- 并发限制。
- 资源预算。
- 结果验证。
- 审计和 Trace。
- 故障恢复。

## **十三、V1 实施路线：先构建一个真正有用的多 Agent 团队**

对于 mini-llm，我不建议一开始实现十几个 Agent。

建议从软件工程任务入手，因为它容易获得客观验证结果。

### **V1：Software Development Multi-Agent**

**四个核心 Agent**

[Agentic Design Patterns: Connecting the Dots | by Gaurav Caprihan, Ph.D | Medium](https://images.openai.com/static-rsc-4/JiL9g3q_H_pu0PmfQ7hhHZLR3U-GuwWdnM1PlqlQxVDHkkC3o1V41I7Epof9S5n0DSpujYY_C4PXjWcoF4WdagjveJsRxTSomxX5PQOKoiUjgeHdWaOPeFb2WCcTn5toUxayoeAfE6wn_Zwyc8cWaAQBGhvtTwlDTdHP1JASH9A?purpose=inline)

**1. Architect Agent**

分析需求、设计架构、定义接口契约。

[Backend development coding and programming. Woman write, test and optimize code, work with different program languages. Vector illustration with line people for web design.](https://images.openai.com/static-rsc-4/Sz0oiR7Sw-HIHEqN4_f1BiUMlTZdKD4xK1SEIDd2fZhHCdATNZjB5bBazg2dhj9ffJoWjlenjmqzw3x5dyI9fZ1ZuzDwFzz2l-DA93v2k7SRrOqpuVU-rywxYbFRnOfqHoiESmr3PSTvwQUKEs2iWssXDreFIavX9uKYqdzdgS28tiSozbfZ_5CkSVJcp8qF?purpose=inline)

**2. Developer Agent**

实现代码、修改文件、调用开发工具。

[Automation testing abstract concept vector illustration.](https://images.openai.com/static-rsc-4/DLrbOlH8aEXJGa15EfPfEwGqeBSVD9zeeDzCfSGMG0COfXS0U1JAeIF_HoWzJiPmvfmIYLvtAlW9vE9OdKlYL7yVbrkSXMDUw_oSFkBrGJbWuV8QRIDRixQRPSBKrt8HeNO4PATu0frO8ib1pSfhUhNDUl0r-oI4zz8c6c7TEkJKldzMdSiQP9uB_dhVCzHm?purpose=inline)

**3. QA Agent**

运行测试、检查结果、定位问题。

[What is a secure code review? A complete guide for developers | Sonar](https://images.openai.com/static-rsc-4/jZYAgekICmXmTp1LUpxCxDASMosRfnvDpyOUANpPUHws6VO32gi0Gnpp2jzyE4Mth_MGYg0UtPvPjEANyezn0mpXeUwj_3_tuSO7G39ZPeYguLpKzhjv54PkfaumX3IqAIDvvxCz0-h6nEzVjlnIW1KbfWAQfU1pG63O2XJfUus?purpose=inline)

**4. Reviewer Agent**

检查代码质量、架构一致性和安全问题。

### **V1 执行流程**

```
User Requirement
       |
       v
Manager Agent
       |
       v
Architect Agent
       |
       v
Architecture Contract
       |
       +------------------+
       |                  |
       v                  v
Developer Agent      Reviewer Agent
       |
       v
Code Changes
       |
       v
QA Agent
       |
       v
Test Results
       |
       v
Reviewer Agent
       |
       v
Final Evaluation
       |
       v
Human Approval
       |
       v
Delivery
```

其中 Reviewer 可以在代码修改完成后再执行独立审查，不必在开发开始时就与 Developer 并行。

### **V1 验收标准**

1. 能够将一个复杂开发需求拆解成多个子任务。
2. 能够将任务分配给不同 Agent。
3. 能够按照 DAG 关系执行任务。
4. 能够并行运行没有依赖关系的任务。
5. 能够保存每个 Agent 的输入和输出。
6. 能够发现任务失败并执行受控重试。
7. 能够检测代码冲突。
8. 能够独立验证最终代码。
9. 能够记录每个 Agent 的 Token、成本和执行时间。
10. 能够追溯完整任务执行链路。

## **十四、Multi-Agent 的核心工程原则**

最后总结几个重要原则。

**原则一：Agent 数量不是能力指标。**

更多 Agent 可能意味着更多成本和通信开销。

**原则二：任务必须有明确边界。**

没有明确输入、输出和完成条件的任务，不适合直接分配给 Agent。

**原则三：共享信息不等于共享所有上下文。**

应该使用受控的 Context Sharing，而不是让所有 Agent 共享完整对话历史。

**原则四：协作结果必须独立验证。**

多个 Agent 一致同意，不代表结论正确。

**原则五：并行执行必须有状态协调。**

任务锁、幂等、超时、取消、恢复缺一不可。

**原则六：安全权限不能通过 Agent 之间的协商决定。**

每次工具执行都必须由独立的权限和策略系统检查。

**原则七：Multi-Agent 必须证明自身价值。**

与 Single-Agent 基线相比，必须在质量、成本、速度或可靠性方面体现明确收益。

## **十五、总结：从单体智能到群体智能**

**mini-llm Multi-Agent 能力结构**

1. **Agent Collaboration**

   多个 Agent 如何分工协作。
2. **Task Delegation**

   如何将复杂目标分解为可执行任务。
3. **Agent Communication**

   如何交换任务、事件和结果。
4. **Consensus**

   如何基于证据形成共同决策。
5. **Negotiation**

   如何解决目标和资源冲突。
6. **Collective Intelligence**

   如何通过协作获得整体能力增益。
7. **Multi-Agent Coordination**

   如何保证并发、状态一致性和故障恢复。
8. **Multi-Agent Evaluation**

   如何证明群体协作真正提高了系统能力。

从整个 mini-llm 的发展来看：

- Transformer Engine 负责信息处理。
- Representation Engine 负责内部表示。
- World Model Engine 负责世界状态和未来预测。
- Agent Engine 负责自主行动。
- Self-Improvement Engine 负责经验驱动的持续改进。
- Multi-Agent Engine 负责多个智能体之间的协作。

这里出现了一个重要变化：

**AI 不再只是一个模型回答问题，而是由多个具有不同能力、职责和执行边界的智能体组成的协作系统。**

而当多个 Agent 能够进行有效分工、共享证据、独立验证、协调行动和持续学习时，系统才有机会从单体智能走向群体智能。

## **下一篇建议**

**《mini-llm Distributed Agent Engine 深度解析：从 Agent Cluster、Distributed Scheduling 到 State Synchronization、Fault Tolerance 与 Scalable Execution》**

重点研究：

1. Agent Cluster：如何构建分布式 Agent 集群？
2. Distributed Scheduling：如何跨机器调度 Agent？
3. Agent Worker：如何构建无状态和有状态 Worker？
4. State Synchronization：如何同步 Agent 状态？
5. Distributed Memory：如何共享分布式记忆？
6. Fault Tolerance：如何处理节点故障？
7. Load Balancing：如何平衡模型和 Agent 负载？
8. Resource Scheduling：如何管理 GPU、CPU、Token 和并发额度？
9. Distributed Tracing：如何追踪跨机器的 Agent 执行链？
10. Distributed Agent Architecture：如何构建可扩展的 mini-llm 分布式智能体平台？
