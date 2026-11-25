# **《mini-llm System Integration Engine 深度解析：从 Engine Orchestration、Dependency Graph 到 Inter-Engine Communication、Data Contracts 与 Unified AI Runtime》**

前面我们已经从底层计算开始，逐步构建了 mini-llm 的各个 Engine：

- Tensor Engine：张量计算。
- Autograd Engine：自动微分。
- Neural Network Engine：神经网络计算。
- Transformer Engine：Transformer 架构。
- Training Engine：模型训练。
- Representation Engine：内部表示。
- World Model Engine：世界模型。
- Reasoning Engine：推理。
- Memory Engine：记忆。
- Knowledge Engine：知识管理。
- Agent Engine：智能体。
- Runtime Engine：执行运行时。
- Protocol Engine：通信协议。
- Security Engine：安全控制。
- Observability Engine：可观测性。
- Simulation Engine：环境模拟。
- Multimodal Engine：多模态处理。
- Generation Engine：内容生成。
- Trust & Verification Engine：可信验证。

现在出现了一个新的问题：

**这些 Engine 如何组合成一个真正能够运行的 AI 系统？**

如果每个 Engine 都独立开发，即使它们各自具备很强的能力，也不意味着整个系统能够正常协作。

例如：

Reasoning Engine 生成了一个任务计划，但 Planning Engine 不知道如何读取。

Planning Engine 生成了一个任务 DAG，但 Runtime Engine 不知道如何执行。

Runtime Engine 调用了工具，但 Security Engine 没有参与权限检查。

任务执行结束后，Observability Engine 无法追踪完整调用链。

这说明：

单个 Engine 的能力只是基础，Engine 之间的协作才决定系统的整体能力。

System Integration Engine 就是负责解决这个问题的系统集成层。

# **一、System Integration Engine：为什么需要统一的系统集成层？**

## **1.1 从单体模型到多 Engine 系统**

传统大模型系统通常可以简化为：

```
User
  |
  v
LLM
  |
  v
Response
```

这种结构比较简单。

但当我们开始构建复杂 AI 系统时，实际执行过程会变成：

```
User
  |
  v
Agent Engine
  |
  +----> Retrieval Engine
  |
  +----> Knowledge Engine
  |
  +----> Reasoning Engine
  |
  +----> Planning Engine
  |
  +----> Tool Engine
  |
  +----> Trust Engine
  |
  +----> Runtime Engine
  |
  v
Response
```

这时就出现了几个问题：

1. 谁负责决定调用哪个 Engine？
2. Engine 之间如何传递数据？
3. 如何管理任务依赖？
4. 如何处理并行执行？
5. 如何处理 Engine 失败？
6. 如何管理跨 Engine 的状态？
7. 如何追踪整个任务的执行过程？

如果每个 Engine 自己调用其他 Engine，系统很快就会形成复杂的依赖关系。

例如：

```
Agent Engine
  |
  v
Reasoning Engine
  |
  v
Planning Engine
  |
  v
Runtime Engine
  |
  v
Agent Engine
```

这种循环依赖会让系统难以维护。

因此，需要一个统一的协调机制。

## **1.2 System Integration Engine 的定义**

System Integration Engine 是一个负责组织、协调和管理多个 Engine 的系统层。

它不负责替代各个 Engine 的专业能力。

它主要负责：

- Engine 注册与发现。
- Engine 生命周期管理。
- Engine 依赖管理。
- Engine 调度。
- Engine 间通信。
- 数据契约管理。
- 共享上下文管理。
- 任务状态管理。
- 故障隔离。
- 版本兼容。
- 系统级可观测性。

可以将它理解为：

**AI 系统内部的操作系统式协调层。**

这里的“操作系统式”是架构类比，不意味着它就是传统操作系统内核。

## **1.3 System Integration 的核心架构**

## **mini-llm System Architecture**

**Application Layer**

Chat / Coding Agent / Research Agent / Multimodal Agent

## **System Integration Engine**

Orchestrator

Dependency Manager

Engine Registry

Communication Bus

Data Contract Manager

Context Manager

Lifecycle Manager

Failure Manager

**Engine Layer**

Reasoning

Planning

Memory

Knowledge

Retrieval

Generation

Trust

Security

Runtime

Simulation

Multimodal

Evaluation

**Infrastructure Layer**

Model Gateway / Database / Message Broker / Storage / Container / GPU Runtime

这里最重要的是：

System Integration Engine 位于应用和专业 Engine 之间，负责系统级协调。

它不是一个新的超级模型，而是一套工程运行机制。

# **二、Engine Orchestration：如何协调多个 Engine？**

Orchestration 是整个系统集成层的核心。

## **2.1 什么是 Orchestration？**

Orchestration，编排。

它解决的问题是：

**一个复杂任务应该调用哪些 Engine，以什么顺序执行，以及如何根据执行结果调整流程？**

例如用户提出：

> 分析一个 GitHub 项目，判断它是否适合构建企业级分布式 Agent 平台。

系统可能需要执行：

1. 获取 GitHub 仓库信息。
2. 分析项目目录。
3. 提取依赖关系。
4. 分析核心架构。
5. 检查是否支持分布式执行。
6. 验证技术事实。
7. 形成技术评价。
8. 输出架构建议。

这些步骤对应不同 Engine。

Orchestrator 负责组织整个执行流程。

## **2.2 三种基本编排模式**

### **模式一：Sequential Orchestration**

顺序编排。

```
Input
  |
  v
Retrieval
  |
  v
Knowledge
  |
  v
Reasoning
  |
  v
Trust
  |
  v
Generation
  |
  v
Output
```

特点：

- 执行顺序明确。
- 实现简单。
- 适合依赖关系固定的任务。
- 前一个步骤失败可能阻塞后续步骤。

例如：

文档解析 → 知识提取 → 知识验证 → 报告生成。

### **模式二：Parallel Orchestration**

并行编排。

```
             Input
               |
               v
         Orchestrator
          /    |    \
         /     |     \
        v      v      v
   Retrieval  Code   Docs
     Engine  Engine  Engine
        \      |      /
         \     |     /
          v    v    v
         Result Aggregator
               |
               v
          Reasoning
```

三个 Engine 可以同时执行。

例如：

- Retrieval Engine 搜索 README。
- Code Engine 分析源代码。
- Dependency Engine 分析依赖关系。

当三者完成后，再交给 Reasoning Engine 综合分析。

需要注意：并行执行要求任务之间不存在尚未满足的数据依赖，而且需要考虑资源限制、超时和部分失败。

### **模式三：Dynamic Orchestration**

动态编排。

```
User Task
    |
    v
Orchestrator
    |
    v
Reasoning
    |
    v
Select Engine
    |
    v
Execute
    |
    v
Observe Result
    |
    +------> Continue
    |
    +------> Retry
    |
    +------> Select Another Engine
    |
    +------> Complete
```

这种模式更适合 Agent 系统。

例如：

用户要求分析一个项目。

系统首先调用 Retrieval Engine。

发现 README 信息不足，于是调用 Code Engine。

Code Engine 发现项目使用多个微服务，再调用 Dependency Engine。

Dependency Engine 发现存在循环依赖，系统进一步调用 Architecture Analysis Engine。

这不是预先写死的固定流程，而是根据中间结果动态决定后续步骤。

## **2.3 Orchestration 与 Choreography 的区别**

这两个概念容易混淆。

| 维度    | Orchestration   | Choreography |
| ----- | --------------- | ------------ |
| 控制方式  | 中央协调            | 事件驱动的分布式协作   |
| 流程管理  | Orchestrator 负责 | 各参与者根据事件响应   |
| 流程可见性 | 相对集中            | 需要事件追踪       |
| 适合场景  | 复杂任务工作流         | 松耦合事件系统      |
| 主要风险  | 中央协调器成为瓶颈       | 流程分散、难以理解    |

例如：

在 Orchestration 中：

```
Orchestrator:
    调用 A
    等待 A
    调用 B
    等待 B
    调用 C
```

在 Choreography 中：

```
A 完成
  |
  v
发布 Event A.Completed
  |
  v
B 订阅并执行
  |
  v
发布 Event B.Completed
  |
  v
C 订阅并执行
```

对于 mini-llm：

建议以 Orchestration 作为主要任务协调方式，以 Event-Driven Architecture 作为模块之间的辅助通信方式。

这样既能保持任务流程清晰，又能减少模块之间的直接耦合。

# **三、Engine Dependency Graph：如何描述 Engine 之间的依赖关系？**

## **3.1 为什么需要 Dependency Graph？**

假设一个任务需要：

```
Retrieval
Knowledge
Reasoning
Trust
Generation
```

这些 Engine 并不是任意顺序执行的。

例如：

Reasoning Engine 需要先获取检索结果和知识信息。

Trust Engine 需要先获取待验证的声明及相关证据。

Generation Engine 需要等到推理和验证完成后，才能生成最终回答。

因此，需要明确 Engine 之间的依赖关系。

## **3.2 用 DAG 表示执行依赖**

DAG（Directed Acyclic Graph），有向无环图。

例如：

### **Research Task DAG**

**User Task**

**Retrieval**

**Code Analysis**

**Knowledge Integration**

**Reasoning**

**Trust Verification**

**Generation**

这是一个简化的 DAG。

实际系统中，Retrieval 和 Code Analysis 可以并行，Knowledge Integration 需要等待二者完成。

## **3.3 DAG 的数学表示**

可以将 Engine 依赖关系表示为：

\\[ G=(V,E) \\]

其中：

- \\(V\\)：Engine 或 Task 节点。
- \\(E\\)：有向依赖边。

如果：

\\[ (A,B)\in E \\]

表示 B 依赖 A 的执行结果。

例如：

\\[ Retrieval\rightarrow Reasoning \\]

意味着 Reasoning 需要 Retrieval 的输出。

一个有效 DAG 必须满足：

\\[ G\text{ contains no directed cycles} \\]

也就是不能存在有向循环。

为什么？

如果：

```
A → B → C → A
```

那么每个节点都在等待另一个节点，系统无法通过正常拓扑排序确定执行顺序。

## **3.4 Topological Sorting**

拓扑排序用于确定符合依赖关系的执行顺序。

例如：

```
from collections import defaultdict, dequedef topological_sort(nodes, edges):    graph = defaultdict(list)    indegree = {node: 0 for node in nodes}    for source, target in edges:        graph[source].append(target)        indegree[target] += 1    queue = deque(        node        for node in nodes        if indegree[node] == 0    )    result = []    while queue:        node = queue.popleft()        result.append(node)        for neighbor in graph[node]:            indegree[neighbor] -= 1            if indegree[neighbor] == 0:                queue.append(neighbor)    if len(result) != len(nodes):        raise ValueError("Dependency cycle detected")    return result
```

示例：

```
nodes = [    "retrieval",    "code_analysis",    "knowledge",    "reasoning",    "trust",    "generation"]edges = [    ("retrieval", "knowledge"),    ("code_analysis", "knowledge"),    ("knowledge", "reasoning"),    ("reasoning", "trust"),    ("trust", "generation")]print(topological_sort(nodes, edges))
```

一个可能的结果：

```
[
  retrieval,
  code_analysis,
  knowledge,
  reasoning,
  trust,
  generation
]
```

Retrieval 和 Code Analysis 的先后顺序可以不同，因为二者之间没有依赖关系。

## **3.5 Dependency Graph 不等于 Execution Graph**

这两个概念需要区分。

**Dependency Graph** 描述系统组件之间的依赖关系。

**Execution Graph** 描述某一次任务实际发生的执行过程。

例如：

系统中存在：

```
Reasoning → Retrieval
Reasoning → Trust
Agent → Runtime
```

这是组件依赖关系。

但某次具体任务可能只调用了：

```
Agent
  |
  v
Retrieval
  |
  v
Reasoning
  |
  v
Trust
```

这是实际执行图。

一个 Engine 可以存在于系统依赖图中，但某次任务不一定调用它。

这个区分对于后续的动态编排、故障分析和执行回放非常重要。

# **四、Inter-Engine Communication：Engine 之间如何通信？**

Engine 之间不仅需要依赖关系，还需要通信机制。

## **4.1 为什么不能直接互相调用？**

最简单的方式是：

```
class AgentEngine:    def __init__(        self,        reasoning_engine,        planning_engine,        runtime_engine    ):        self.reasoning = reasoning_engine        self.planning = planning_engine        self.runtime = runtime_engine
```

这种方式在小项目中没有问题。

但随着 Engine 数量增加：

```
Agent
  ├── Reasoning
  ├── Planning
  ├── Runtime
  ├── Memory
  ├── Knowledge
  ├── Trust
  └── Security
```

会出现：

- 依赖关系越来越复杂。
- Engine 修改影响其他模块。
- 测试需要加载大量依赖。
- 进程边界难以建立。
- 分布式部署困难。

因此需要抽象通信机制。

## **4.2 三种主要通信方式**

### **方式一：Direct Function Call**

直接函数调用。

```
result = reasoning_engine.execute(context)
```

优点：

- 简单。
- 延迟低。
- 调试方便。
- 适合单进程模块。

缺点：

- 调用双方强耦合。
- 不适合直接跨机器。
- 调用链容易复杂化。

适合 V1 的进程内 Engine。

### **方式二：Request-Response**

请求响应模式。

```
Engine A
   |
   | Request
   v
Engine B
   |
   | Response
   v
Engine A
```

例如：

```
POST /api/reasoning/execute
Content-Type: application/json
```

适合：

- 独立服务。
- 跨进程调用。
- 需要明确返回结果的操作。

可以使用 HTTP、gRPC 等实现。

### **方式三：Event-Driven Communication**

事件驱动通信。

```
Engine A
   |
   v
Message Broker
   |
   v
Engine B
```

例如：

```
{
  "event_type": "document.parsed",
  "event_id": "evt-001",
  "payload": {
    "document_id": "doc-001"
  }
}
```

Engine A 发布事件。

Engine B 订阅事件。

两者不需要直接知道对方的内部实现。

适合：

- 异步任务。
- 长时间运行任务。
- 多个消费者。
- 分布式协作。

但事件驱动会引入最终一致性、重复投递、事件顺序和消息丢失等问题，需要额外设计。

## **4.3 Command、Event 与 Query**

这是设计通信协议时非常重要的概念。

| 类型      | 含义        | 示例            |
| ------- | --------- | ------------- |
| Command | 请求系统执行操作  | ExecuteTask   |
| Event   | 描述已经发生的事实 | TaskCompleted |
| Query   | 请求读取信息    | GetTaskStatus |

例如：

```
Command:
    ExecuteReasoningTask

Event:
    ReasoningTaskCompleted

Query:
    GetReasoningResult
```

三者不能混淆。

Command 表达意图，不代表操作已经成功。

Event 表达已经发生的事实。

Query 用于读取状态，不应产生业务副作用。

## **4.4 建议的通信架构**

对于 mini-llm，建议采用混合通信模式：

**Hybrid Communication Architecture**

**System Orchestrator**

**Synchronous**

同步通信

- Function Call
- HTTP
- gRPC

适合需要立即获得结果的调用

**Asynchronous**

异步通信

- Event Bus
- Message Queue
- Pub/Sub

适合长任务和事件驱动协作

**Engine Registry + Communication Adapter**

V1 不需要立即引入复杂消息中间件。

可以先实现进程内 Event Bus，并为未来跨进程通信预留接口。

# **五、Data Contracts：如何统一不同 Engine 的数据结构？**

这是整个 System Integration Engine 最容易被低估的问题。

假设：

Reasoning Engine 输出：

```
{
  "answer": "需要进一步分析",
  "confidence": 0.8
}
```

Planning Engine 期待：

```
{
  "reasoning_result": "...",
  "score": 0.8
}
```

两个 Engine 使用不同的数据字段。

即使它们各自工作正常，也无法直接协作。

这就是 Data Contract 要解决的问题。

## **5.1 什么是 Data Contract？**

Data Contract，数据契约。

它定义：

- 数据结构。
- 字段名称。
- 数据类型。
- 字段约束。
- 版本。
- 语义。
- 错误处理方式。
- 兼容规则。

简单来说：

**Data Contract 是 Engine 之间共同遵守的数据接口标准。**

## **5.2 统一 Engine Input / Output**

建议所有 Engine 遵循统一的执行接口。

例如：

```
interface EngineInput<T> {
  taskId: string;

  executionId: string;

  contextId: string;

  traceId: string;

  payload: T;

  metadata: {
    sourceEngine?: string;
    createdAt: string;
    schemaVersion: string;
  };
}
```

统一输出：

```
interface EngineOutput<T> {
  taskId: string;

  executionId: string;

  engineId: string;

  status:
    | "success"
    | "failed"
    | "partial";

  payload?: T;

  error?: {
    code: string;
    message: string;
    retryable: boolean;
  };

  metadata: {
    startedAt: string;
    completedAt: string;
    schemaVersion: string;
  };
}
```

这样，Orchestrator 不需要了解每个 Engine 的内部实现。

它只需要理解统一输入输出协议。

## **5.3 Data Contract 与业务数据不能混为一谈**

例如：

```
interface ReasoningResult {
  conclusion: string;

  assumptions: string[];

  evidenceIds: string[];

  confidence?: number;
}
```

这属于 Reasoning Engine 的业务输出。

而：

```
interface EngineOutput<T> {
  taskId: string;
  status: string;
  payload?: T;
}
```

属于统一执行协议。

两者应该分离。

否则统一接口会变得越来越庞大，所有 Engine 都被迫适应同一个业务数据结构。

## **5.4 Schema Versioning**

假设 Reasoning Engine V1：

```
{
  "conclusion": "方案 A"
}
```

V2：

```
{
  "conclusion": "方案 A",
  "alternatives": [
    "方案 B",
    "方案 C"
  ]
}
```

如果直接删除旧字段或修改字段含义，就可能破坏其他 Engine。

因此需要版本管理。

例如：

```
reasoning.result.v1
reasoning.result.v2
```

建议遵循：

- 新增可选字段通常较容易兼容。
- 删除字段属于破坏性变更。
- 修改字段类型通常属于破坏性变更。
- 修改字段语义即使类型不变，也可能破坏兼容。
- 旧版本消费者需要有明确的迁移策略。

可以使用 JSON Schema、Protocol Buffers 或 TypeScript Schema 工具管理契约。

# **六、Shared Context：如何共享上下文而不造成数据污染？**

这是多 Engine AI 系统中非常关键的问题。

## **6.1 什么是 Shared Context？**

假设用户提出：

> 分析这个项目，并结合我之前的技术架构设计提出改进方案。

系统需要共享：

- 当前用户请求。
- 用户提供的项目资料。
- 前面 Engine 的分析结果。
- 历史对话信息。
- 当前任务的执行状态。
- 已经获取的证据。

如果每个 Engine 都独立维护上下文，就会出现信息不一致。

例如：

```
Reasoning Engine:
    项目使用 Go

Planning Engine:
    项目使用 Java

Generation Engine:
    项目使用 Node.js
```

这就是上下文不一致问题。

## **6.2 为什么不能把所有上下文塞进一个 Prompt？**

一种简单做法：

```
把所有用户信息
+ 所有历史对话
+ 所有检索结果
+ 所有 Engine 输出
全部塞进 Prompt
```

但这会带来：

1. Token 成本增加。
2. 上下文窗口被占满。
3. 无关信息干扰。
4. 历史错误信息被重复传播。
5. 不同来源的可信度无法区分。
6. 敏感信息可能进入不必要的模块。

因此，Shared Context 不应该是一个无限增长的文本字符串。

它应该是一个结构化的上下文管理系统。

## **6.3 Context 分层**

建议将上下文分为五类：

| Context 类型         | 内容             | 生命周期 |
| ------------------ | -------------- | ---- |
| Request Context    | 当前用户请求         | 单次请求 |
| Task Context       | 当前任务状态和中间结果    | 单次任务 |
| Execution Context  | 某次 Engine 执行信息 | 单次执行 |
| Session Context    | 会话级上下文         | 多轮对话 |
| Persistent Context | 持久化知识与记忆       | 长期   |

需要特别区分：

Execution Context 与 Task Context 不是一回事。

一个 Task 可能包含多个 Execution。

例如：

```
Task: Analyze Repository
  |
  +---- Execution 1: Retrieval
  |
  +---- Execution 2: Code Analysis
  |
  +---- Execution 3: Reasoning
  |
  +---- Execution 4: Trust Verification
```

## **6.4 Context 数据结构**

```
interface SharedContext {
  contextId: string;

  taskId: string;

  sessionId?: string;

  request: {
    userInput: string;
    timestamp: string;
  };

  artifacts: Array<{
    artifactId: string;
    type: string;
    reference: string;
    sourceEngine: string;
  }>;

  evidenceIds: string[];

  metadata: {
    contextVersion: string;
    createdAt: string;
    updatedAt: string;
  };
}
```

这里使用 `reference` 而不是把所有内容直接放进 Context。

例如：

```
{
  "artifactId": "artifact-001",
  "type": "repository_analysis",
  "reference": "artifact://repository/001",
  "sourceEngine": "code-analysis"
}
```

这样可以避免上下文无限膨胀。

## **6.5 Context 必须有权限边界**

不是所有 Engine 都应该访问所有上下文。

例如：

- Generation Engine 需要读取已验证的结论。
- Retrieval Engine 需要读取搜索目标。
- Security Engine 需要读取授权信息。
- 外部工具不应该直接访问用户完整 Memory。
- 不可信文档不能获得修改系统策略的权限。

因此：

\\[ ContextAccess = f(Identity, Role, Scope, Policy) \\]

Shared Context 必须和 Security Engine 配合。

**共享数据不代表共享权限。**

# **七、Event-Driven Architecture：如何通过事件驱动 Engine 协作？**

当系统变得复杂后，很多操作不需要立即返回结果。

例如：

- 文档解析。
- 大规模代码分析。
- 模型微调。
- 长时间运行的 Agent 任务。
- 数据库索引。
- 向量索引重建。

这些任务适合事件驱动架构。

## **7.1 Event 的基本结构**

建议：

```
interface EngineEvent<T> {
  eventId: string;

  eventType: string;

  source: string;

  taskId: string;

  executionId?: string;

  traceId: string;

  timestamp: string;

  version: string;

  payload: T;
}
```

例如：

```
{
  "eventId": "evt-001",
  "eventType": "document.parsed",
  "source": "document-engine",
  "taskId": "task-001",
  "traceId": "trace-001",
  "timestamp": "2026-11-25T10:00:00Z",
  "version": "1.0",
  "payload": {
    "documentId": "doc-001",
    "pageCount": 20
  }
}
```

## **7.2 Event Bus**

Event Bus 负责事件的发布和订阅。

```
type EventHandler<T> = (
  event: EngineEvent<T>
) => Promise<void>;

class EventBus {

  private handlers = new Map<
    string,
    EventHandler<any>[]
  >();

  subscribe<T>(
    eventType: string,
    handler: EventHandler<T>
  ) {

    const list =
      this.handlers.get(eventType) ?? [];

    list.push(handler);

    this.handlers.set(eventType, list);
  }

  async publish<T>(
    event: EngineEvent<T>
  ) {

    const handlers =
      this.handlers.get(event.eventType) ?? [];

    await Promise.all(
      handlers.map(handler => handler(event))
    );
  }
}
```

这是一个进程内的简化版本。

生产环境还需要：

- 消息持久化。
- 消费者确认。
- 重试机制。
- 死信队列。
- 消费者幂等。
- 事件顺序控制。
- 消息过期策略。

## **7.3 Event Sourcing**

这里还有一个重要概念：

Event Sourcing，事件溯源。

传统方式：

```
Task Status = completed
```

系统只保存最终状态。

事件溯源方式：

```
TaskCreated
TaskStarted
EngineSelected
EngineExecutionStarted
EngineExecutionCompleted
TrustVerificationPassed
TaskCompleted
```

系统通过事件序列重建任务状态。

这对于 AI Agent 特别有价值。

因为 Agent 的执行往往不是简单的一次函数调用，而是多轮感知、推理、工具调用和结果观察。

但要注意：

Event Sourcing 不等于所有事件都必须永久保存。

需要设计：

- 数据保留期限。
- 敏感数据脱敏。
- 快照机制。
- 事件版本升级。
- 访问控制。

另外，事件日志可以帮助重建系统状态，但不能保证重新运行 LLM 时得到完全相同的输出。

# **八、Distributed Engine Execution：如何跨进程、跨机器运行？**

当 Engine 数量增加后，可能需要将不同 Engine 部署到不同机器。

例如：

```
Machine A:
    API Gateway
    System Orchestrator

Machine B:
    Reasoning Engine
    Generation Engine

Machine C:
    Retrieval Engine
    Knowledge Engine

Machine D:
    Training Engine
    GPU Runtime
```

为什么需要这样设计？

因为不同 Engine 的资源需求差异很大。

| Engine     | 主要资源需求         |
| ---------- | -------------- |
| Retrieval  | CPU、内存、磁盘、网络   |
| Knowledge  | CPU、内存、数据库     |
| Reasoning  | GPU、模型推理资源     |
| Generation | GPU、KV Cache   |
| Training   | GPU、显存、存储带宽    |
| Simulation | CPU/GPU、计算资源   |
| Multimodal | GPU、图像和音视频处理资源 |

如果全部运行在一台机器上，可能出现资源竞争。

## **8.1 Worker Architecture**

建议把 Engine 的实际执行封装成 Worker。

```
System Orchestrator
       |
       v
   Task Queue
       |
       +----------------+
       |                |
       v                v
   Worker A         Worker B
       |                |
       v                v
 Reasoning          Retrieval
 Engine             Engine
```

Worker 负责：

- 接收任务。
- 校验输入。
- 执行 Engine。
- 记录状态。
- 返回结果。
- 上报指标。
- 处理异常。

## **8.2 Stateless Worker 与 Stateful Worker**

### **Stateless Worker**

无状态 Worker。

每次执行都从外部获取所需数据，不依赖本地长期状态。

优点：

- 容易扩容。
- 容易迁移。
- 容易恢复。
- 负载均衡简单。

### **Stateful Worker**

有状态 Worker。

需要维护一些执行状态。

例如：

- 长时间运行的 Agent Session。
- 流式生成连接。
- 本地模型 KV Cache。
- 正在执行的仿真环境。

有状态 Worker 并不意味着所有状态只能存在进程内。

生产系统应将需要恢复的状态持久化或外置，并明确 Worker 崩溃后的恢复机制。

## **8.3 Worker 注册与心跳**

可以设计：

```
interface WorkerInfo {
  workerId: string;

  engineTypes: string[];

  endpoint: string;

  status:
    | "ready"
    | "busy"
    | "unhealthy"
    | "offline";

  capabilities: {
    cpu?: number;
    memory?: number;
    gpu?: number;
  };

  lastHeartbeat: string;
}
```

Orchestrator 根据：

- Engine 类型。
- Worker 状态。
- 资源情况。
- 任务优先级。
- 延迟要求。

选择合适的 Worker。

## **8.4 不要让每个 Engine 都变成微服务**

这是一个非常重要的工程决策。

如果有 30 个 Engine，就创建 30 个微服务，可能导致：

- 部署复杂度增加。
- 网络通信增加。
- 运维成本上升。
- 日志和链路追踪更加复杂。
- 开发效率下降。

建议：

**逻辑上 Engine 独立，部署上允许多个 Engine 组合。**

例如：

```
Core Service
├── Context Engine
├── Knowledge Engine
└── Trust Engine

Agent Service
├── Agent Engine
├── Planning Engine
└── Reasoning Engine

Inference Service
├── Generation Engine
├── Multimodal Engine
└── Model Runtime
```

这是一种从模块化单体逐渐向分布式架构演进的方式。

# **九、Failure Isolation：一个 Engine 失败后如何避免整个系统崩溃？**

多 Engine 系统最大的挑战之一，就是故障传播。

## **9.1 故障传播问题**

例如：

```
Agent
  |
  v
Retrieval
  |
  v
Knowledge
  |
  v
Reasoning
  |
  v
Generation
```

如果 Retrieval Engine 超时，后面的所有步骤都可能被阻塞。

更复杂的是：

一个 Engine 可能不断重试，消耗大量资源，最终影响其他 Engine。

这就是级联故障。

## **9.2 Timeout**

每个 Engine 都应该有超时限制。

```
interface ExecutionPolicy {
  timeoutMs: number;

  maxRetries: number;

  retryDelayMs: number;

  concurrencyLimit: number;
}
```

例如：

```
const policy: ExecutionPolicy = {
  timeoutMs: 30000,
  maxRetries: 2,
  retryDelayMs: 1000,
  concurrencyLimit: 10
};
```

超时意味着系统不再无限等待。

但超时并不一定代表远端操作已经停止。

因此，对具有副作用的操作还需要取消协议、执行状态查询和幂等控制。

## **9.3 Retry**

重试机制。

常见方式：

- Fixed Delay。
- Exponential Backoff。
- Exponential Backoff with Jitter。

指数退避：

\\[ delay_n=\min(d\_{max},d_0\cdot 2^n) \\]

例如：

```
第一次：1 秒
第二次：2 秒
第三次：4 秒
第四次：8 秒
```

加入随机抖动可以降低大量 Worker 同时重试造成的流量尖峰。

但不是所有错误都应该重试。

例如：

| 错误      | 是否重试   |
| ------- | ------ |
| 网络暂时不可用 | 可以     |
| 服务过载    | 可以，需退避 |
| 参数格式错误  | 不应该    |
| 权限拒绝    | 不应该    |
| 资源暂时不足  | 视情况    |
| 外部服务限流  | 遵循限流策略 |

## **9.4 Circuit Breaker**

熔断器。

当某个 Engine 持续失败时，暂时停止向它发送请求。

基本状态：

### **&#x20;Circuit Breaker State Machine&#x20;**

**Closed**

正常

允许请求

**Open**

熔断

快速失败

**Half-Open**

试探

允许有限请求

Closed 在失败率达到阈值后进入 Open；经过冷却时间进入 Half-Open；试探成功恢复 Closed，失败则回到 Open。

例如：

Reasoning Engine 连续失败 10 次。

系统暂时不再向它发送普通请求。

经过冷却时间后，发送少量探测请求。

如果恢复，则重新开放调用。

## **9.5 Bulkhead**

Bulkhead，舱壁隔离。

它的思想来自船舶结构：

一个舱室进水，不应该导致整个船沉没。

应用到 AI 系统：

```
Worker Pool A:
    Reasoning

Worker Pool B:
    Retrieval

Worker Pool C:
    Generation
```

如果 Generation Engine 消耗大量 GPU 资源，不应该直接拖垮 Retrieval Engine。

可以通过：

- 独立 Worker Pool。
- 独立资源配额。
- 并发限制。
- 队列隔离。
- 优先级调度。

实现故障和资源隔离。

# **十、Unified Observability：如何追踪跨 Engine 的完整执行链路？**

之前我们已经研究过 Observability Engine。

System Integration Engine 需要把它真正连接到所有 Engine。

## **10.1 为什么需要统一 Trace？**

假设一次 Agent 任务经历：

```
Agent
  |
  v
Retrieval
  |
  v
Reasoning
  |
  v
Tool Calling
  |
  v
Runtime
  |
  v
Trust
  |
  v
Generation
```

如果每个 Engine 使用自己的日志 ID，就很难还原整个执行过程。

因此需要统一 Trace ID。

## **10.2 Trace、Span 与 Event**

- Trace：一次完整任务的调用链。
- Span：一次具体操作或执行阶段。
- Event：Span 内发生的特定事件。

例如：

```
Trace: trace-001

  Span: agent.execute
    |
    +-- Span: retrieval.search
    |
    +-- Span: reasoning.run
    |
    +-- Span: runtime.tool_call
    |
    +-- Span: trust.verify
    |
    +-- Span: generation.run
```

每个 Span 都应该保留：

- Trace ID。
- Span ID。
- Parent Span ID。
- Engine ID。
- Task ID。
- Execution ID。
- 开始时间。
- 结束时间。
- 执行状态。

跨服务通信时，需要传播 Trace Context。

## **10.3 统一可观测性架构**

```
All Engines
    |
    v
Telemetry SDK
    |
    +----> Logs
    |
    +----> Metrics
    |
    +----> Traces
    |
    +----> Events
    |
    v
Observability Backend
    |
    +----> Dashboard
    |
    +----> Alerting
    |
    +----> Execution Replay
```

可以使用 OpenTelemetry 作为统一遥测标准。

但要注意：

Observability 记录的是系统可观测的执行事件，不意味着系统可以完整读取或还原模型内部所有隐式推理过程。

# **十一、Engine Lifecycle：如何管理 Engine 的启动、停止和升级？**

当 Engine 越来越多，就需要统一生命周期管理。

## **11.1 Engine Lifecycle**

一个 Engine 不应该只有简单的：

```
engine.start()
```

还应该具备完整生命周期。

### **Engine Lifecycle**

**Registered**

已经注册

**Initializing**

初始化配置和依赖

**Ready**

健康检查通过，可以接收任务

**Running**

执行任务

**Draining**

停止接收新任务，处理已有任务

**Stopped**

正常停止

实际实现中，Ready 和 Running 可以同时存在：Ready 描述健康和可接收任务的状态，Running 描述正在执行任务的状态。上图只是简化的生命周期路径。

还应该考虑：

- Failed：初始化或运行失败。
- Unhealthy：健康检查异常。
- Updating：版本升级中。

## **11.2 Engine Interface**

建议：

```
interface Engine {
  id: string;

  name: string;

  version: string;

  initialize(): Promise<void>;

  healthCheck(): Promise<boolean>;

  execute(
    input: EngineInput<any>
  ): Promise<EngineOutput<any>>;

  shutdown(): Promise<void>;
}
```

但实际工程中，最好将生命周期、执行能力和健康检查接口分开定义。

例如：

```
interface EngineExecutor<TInput, TOutput> {
  execute(
    input: EngineInput<TInput>
  ): Promise<EngineOutput<TOutput>>;
}

interface EngineLifecycle {
  initialize(): Promise<void>;
  shutdown(): Promise<void>;
}

interface EngineHealth {
  healthCheck(): Promise<{
    healthy: boolean;
    details?: Record<string, unknown>;
  }>;
}
```

这样可以让某些远程 Engine 只暴露执行接口，而不需要让 Orchestrator 直接管理其进程生命周期。

## **11.3 Version Compatibility**

假设：

```
Reasoning Engine: 2.0
Planning Engine: 1.0
Trust Engine: 3.0
```

版本不一致不一定意味着不能协作。

真正重要的是：

- 接口是否兼容。
- 数据契约是否兼容。
- 模型输入输出是否兼容。
- 是否满足最低能力要求。
- 是否存在已知问题。

建议 Engine Registry 记录：

```
interface EngineDescriptor {
  engineId: string;

  version: string;

  apiVersion: string;

  supportedContracts: string[];

  capabilities: string[];

  dependencies: string[];
}
```

系统在注册和部署阶段进行兼容性检查，而不是等到运行时才发现数据格式不匹配。

# **十二、mini-llm System Integration Engine 的完整代码架构**

前面介绍了架构原理，现在将其组织成一个可实际开发的工程结构。

## **12.1 推荐目录结构**

考虑到 mini-llm 的目标是长期扩展，不建议把所有功能都写进一个 `engine.py`。

推荐：

```
mini-llm/
│
├── apps/
│   ├── api-server/
│   ├── cli/
│   └── worker/
│
├── core/
│   │
│   ├── integration/
│   │   ├── orchestrator/
│   │   ├── dependency/
│   │   ├── registry/
│   │   ├── scheduler/
│   │   ├── lifecycle/
│   │   ├── communication/
│   │   ├── contracts/
│   │   ├── context/
│   │   ├── failure/
│   │   └── integration_engine.py
│   │
│   ├── runtime/
│   ├── security/
│   ├── observability/
│   └── event_bus/
│
├── engines/
│   ├── tensor/
│   ├── autograd/
│   ├── neural_network/
│   ├── transformer/
│   ├── training/
│   ├── reasoning/
│   ├── planning/
│   ├── memory/
│   ├── knowledge/
│   ├── retrieval/
│   ├── agent/
│   ├── generation/
│   ├── multimodal/
│   ├── trust/
│   └── evaluation/
│
├── contracts/
│   ├── engine/
│   ├── task/
│   ├── event/
│   └── artifact/
│
├── infrastructure/
│   ├── database/
│   ├── message_broker/
│   ├── storage/
│   └── model_gateway/
│
├── tests/
│   ├── unit/
│   ├── integration/
│   ├── contract/
│   ├── failure/
│   └── end_to_end/
│
└── docs/
    ├── architecture/
    ├── engine_specs/
    ├── contracts/
    └── api/
```

这里的 `core/integration` 是本篇的核心。

其他 Engine 保持自己的业务边界。

## **12.2 Engine Registry**

Engine Registry 负责注册、查找和管理 Engine 元信息。

```
class EngineRegistry {

  private engines =
    new Map<string, Engine>();

  register(engine: Engine) {

    if (this.engines.has(engine.id)) {
      throw new Error(
        `Engine already registered: ${engine.id}`
      );
    }

    this.engines.set(
      engine.id,
      engine
    );
  }

  get(engineId: string): Engine {

    const engine =
      this.engines.get(engineId);

    if (!engine) {
      throw new Error(
        `Engine not found: ${engineId}`
      );
    }

    return engine;
  }

  list(): Engine[] {
    return [...this.engines.values()];
  }
}
```

这是进程内 Registry。

分布式部署时，还需要服务发现、健康状态同步和 Worker 注册机制。

## **12.3 Orchestrator**

Orchestrator 负责协调 Engine 执行。

```
class Orchestrator {

  constructor(
    private registry: EngineRegistry
  ) {}

  async execute(
    engineId: string,
    input: EngineInput<any>
  ) {

    const engine =
      this.registry.get(engineId);

    return await engine.execute(input);
  }
}
```

这是最小版本。

后续需要增加：

- DAG 调度。
- 并行执行。
- 超时。
- 重试。
- 取消。
- 资源配额。
- 执行状态持久化。
- 故障隔离。

注意，实际任务编排不应长期依赖这个简单的单节点接口，而应该有独立的 Task Graph 和执行状态模型。

## **12.4 一个完整任务示例**

用户请求：

```
分析一个 GitHub 项目，
判断它是否适合企业级 Agent 平台。
```

Orchestrator 生成任务图：

```
const workflow = {
  taskId: "task-001",

  nodes: [
    {
      id: "retrieval",
      engine: "retrieval",
      dependsOn: []
    },
    {
      id: "code",
      engine: "code-analysis",
      dependsOn: []
    },
    {
      id: "knowledge",
      engine: "knowledge",
      dependsOn: [
        "retrieval",
        "code"
      ]
    },
    {
      id: "reasoning",
      engine: "reasoning",
      dependsOn: [
        "knowledge"
      ]
    },
    {
      id: "trust",
      engine: "trust",
      dependsOn: [
        "reasoning"
      ]
    },
    {
      id: "generation",
      engine: "generation",
      dependsOn: [
        "trust"
      ]
    }
  ]
};
```

执行流程：

```
                Task
                 |
          +------+------+
          |             |
          v             v
      Retrieval      Code
          |             |
          +------+------+
                 |
                 v
              Knowledge
                 |
                 v
              Reasoning
                 |
                 v
                Trust
                 |
                 v
             Generation
                 |
                 v
               Result
```

这样，系统就能够将多个独立 Engine 组织成一个完整任务。

# **十三、System Integration Engine 的 V1 开发路线**

这里不建议一开始就开发分布式调度、复杂事件系统和自动故障恢复。

应该先实现可靠的模块化协作，再逐步演进。

## **V1 实施路线**

1. **阶段一：Engine Interface**

   第一优先级
   - 统一 Engine Input / Output
   - Engine ID 和 Version
   - Engine Lifecycle
   - Error Contract
   - 基础 Registry
2. **阶段二：Orchestration**
   - Sequential Orchestration
   - DAG Representation
   - Topological Sorting
   - Task State
   - Execution Result
3. **阶段三：Communication**
   - Direct Function Call
   - Request-Response
   - In-Process Event Bus
   - Event Contract
4. **阶段四：Shared Context**
   - Request Context
   - Task Context
   - Execution Context
   - Artifact Reference
   - Context Access Policy
5. **阶段五：Reliability**
   - Timeout
   - Retry
   - Cancellation
   - Circuit Breaker
   - Failure Isolation
6. **阶段六：Distributed Execution**
   - Worker
   - Worker Registry
   - Task Queue
   - Heartbeat
   - Remote Engine Adapter
7. **阶段七：Production Integration**
   - OpenTelemetry
   - Contract Testing
   - Engine Version Compatibility
   - Deployment Management
   - Execution Replay

## **13.1 V1 验收标准**

| 测试项目              | 验收标准                    |
| ----------------- | ----------------------- |
| Engine Registry   | 能够注册、发现和查询 Engine       |
| Unified Interface | Engine 遵循统一输入输出协议       |
| Dependency Graph  | 支持 DAG 和循环依赖检测          |
| Orchestration     | 能够顺序和并行执行任务             |
| Data Contract     | 能够验证输入输出结构              |
| Event Bus         | 支持事件发布和订阅               |
| Shared Context    | 支持跨 Engine 共享结构化上下文     |
| Timeout           | Engine 超时不会无限阻塞         |
| Retry             | 支持按错误类型执行重试策略           |
| Failure Isolation | 一个 Engine 失败不会无条件拖垮其他任务 |
| Observability     | 完整任务具有统一 Trace ID       |
| Integration Test  | 能够完成多 Engine 端到端任务      |

# **十四、进一步思考：System Integration Engine 是否就是 AI 操作系统？**

这是一个值得深入研究的问题。

当一个 AI 系统具有：

- Engine Registry。
- Dependency Manager。
- Task Scheduler。
- Runtime。
- Memory。
- Context。
- Event Bus。
- Security。
- Observability。
- Resource Manager。
- Lifecycle Manager。

它看起来已经具有了某些操作系统的特征。

可以进行一个概念对比：

| 传统操作系统              | mini-llm                    |
| ------------------- | --------------------------- |
| Process             | Engine / Worker             |
| Thread              | Task Execution              |
| Scheduler           | Task Scheduler              |
| IPC                 | Inter-Engine Communication  |
| File System         | Artifact / Storage          |
| Memory Management   | Context / Memory Management |
| Permission          | Security Engine             |
| System Call         | Engine API / Tool Interface |
| Process Monitoring  | Observability               |
| Resource Management | Cost & Resource Engine      |

但是，两者不能简单等同。

传统操作系统管理的是 CPU、内存、设备和进程等计算资源。

mini-llm 还需要管理：

- 模型推理。
- 语义上下文。
- 任务目标。
- 证据。
- 知识。
- Agent 行为。
- 不确定性。
- 决策和执行结果。

因此，更准确的定位是：

**System Integration Engine 是 AI 操作系统架构中的核心协调层。**

如果未来把 Runtime、Security、Memory、Context、Scheduler、Resource Manager 等进一步统一起来，就可以形成更完整的 AI Operating System。

## **14.1 从 Engine 到 AI Operating System**

## **mini-llm Architecture Evolution**

**Stage 1：Independent Engines**

独立实现各种 AI 能力

**Stage 2：Integrated Engine System**

统一接口、数据契约和编排机制

**Stage 3：Unified AI Runtime**

统一任务、资源、状态和执行生命周期

**Stage 4：AI Operating System**

完整管理模型、Agent、工具、知识与环境

这里有一个重要的架构原则：

**不要为了构建 AI Operating System，就把所有能力做成一个巨大的核心模块。**

真正好的系统应该具备：

- 内核足够小。
- 模块边界清晰。
- 接口稳定。
- 能力可插拔。
- 执行可追踪。
- 权限可控制。
- 资源可调度。
- 故障可隔离。

# **十五、总结：从单个 Engine 到完整 AI 系统**

我们可以把 mini-llm 的发展过程总结为四个层次。

| 层次                | 解决的问题                 |
| ----------------- | --------------------- |
| Engine Layer      | 如何实现单项 AI 能力？         |
| Integration Layer | 如何让不同 Engine 协作？      |
| Runtime Layer     | 如何执行和管理复杂任务？          |
| System Layer      | 如何形成完整、可靠、可扩展的 AI 系统？ |

最初我们研究：

> Transformer 如何产生智能？

随后研究：

> 模型如何形成世界表示、进行推理、记忆和决策？

再往后研究：

> Agent 如何使用工具、规划任务、与环境交互？

现在进入了一个更高的工程层次：

> 如何把这些能力组织成一个可运行、可扩展、可管理的系统？

这就是 System Integration Engine 的意义。

最终，mini-llm 不应该只是一个 Transformer 实现，也不应该只是一个 Agent 框架。

它可以逐步演进为一个由多个专业 Engine 构成的统一 AI 系统。

# **下一篇：mini-llm Unified AI Runtime Engine**

System Integration Engine 解决了多个 Engine 如何协作的问题。

但还有一个问题没有彻底解决：

**当几十个 Engine 同时运行，数百个 Task 并行执行，模型、工具、Worker 和资源不断变化时，系统如何统一管理执行过程？**

下一篇建议：

**《mini-llm Unified AI Runtime Engine 深度解析：从 Task Execution、Resource Scheduling 到 State Management、Checkpoint、Recovery 与 Distributed Runtime》**

重点研究：

1. Unified AI Runtime：为什么需要统一运行时？
2. Task Execution：如何统一执行不同类型的任务？
3. Execution State Machine：如何管理复杂任务状态？
4. Resource Scheduling：如何调度 CPU、GPU、Memory 和 Model？
5. Worker Runtime：如何管理不同 Engine 的执行 Worker？
6. Async Runtime：如何实现异步、并发和取消？
7. Checkpoint：如何保存 Agent 和 Task 的执行状态？
8. Recovery：如何从故障中恢复？
9. Distributed Runtime：如何跨机器执行任务？
10. Runtime Isolation：如何隔离不同 Agent 和任务？
11. Runtime Observability：如何统一追踪任务执行？
12. Runtime Architecture：如何构建完整的 mini-llm Unified Runtime？

这篇将进一步把 mini-llm 从一个多 Engine 架构，推进到真正具备统一执行能力的 AI Runtime。
