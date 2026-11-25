# **《mini-llm Unified AI Runtime Engine 深度解析：从 Task Execution、Resource Scheduling 到 State Management、Checkpoint、Recovery 与 Distributed Runtime》**

在前面的 mini-llm 系列中，我们已经研究了 Runtime Engine、System Integration Engine、Agent Engine、Planning Engine、Decision Engine、Memory Architecture、Observability Engine 等核心模块。

这些模块解决了 AI 系统的不同问题：

- Agent Engine：决定由谁执行任务。
- Planning Engine：决定任务应该如何完成。
- System Integration Engine：协调不同 Engine 之间的协作。
- Observability Engine：记录和分析系统运行过程。
- Runtime Engine：提供任务执行的基础设施。

但这里还有一个更深层的问题：

**当一个 AI Agent 同时执行多个任务，调用不同模型和工具，使用 CPU、GPU、内存等资源，并且可能遇到超时、崩溃、网络中断时，谁来保证整个系统能够持续、可靠地运行？**

这就是 Unified AI Runtime Engine 要解决的问题。

# **一、Unified AI Runtime：为什么需要统一运行时？**

## **1.1 从简单的模型调用到复杂的 AI 系统**

最简单的 LLM 应用可能只有一个模型调用：

```
response = llm.generate("什么是 Transformer？")
```

执行过程非常简单：

```
User Input
    |
    v
LLM
    |
    v
Response
```

但是，真正的 Agent 系统远不止如此。

例如，用户提出：

> 分析这个 GitHub 项目，理解它的架构，检查代码问题，生成改进方案，并提交修改。

系统可能需要执行：

```
User Request
    |
    v
Task Decomposition
    |
    v
Planning
    |
    v
Task Scheduler
    |
    +-------------------+
    |                   |
    v                   v
Repository Analysis    Dependency Analysis
    |                   |
    v                   v
Code Analysis          Architecture Analysis
    |                   |
    +---------+---------+
              |
              v
       Problem Evaluation
              |
              v
        Code Modification
              |
              v
         Test Execution
              |
              v
        Git Diff Review
              |
              v
          Final Result
```

这里出现了很多运行时问题：

1. 哪些任务可以并行执行？
2. 哪些任务必须等待前置任务完成？
3. 如何分配 CPU、GPU 和内存？
4. 如何管理不同任务的执行状态？
5. 某个 Worker 崩溃后怎么办？
6. 如何暂停和恢复一个 Agent？
7. 如何避免重复执行有副作用的操作？
8. 如何在多台机器之间分配任务？

这些问题已经超出了单纯的模型推理范围。

因此，我们需要一个统一运行时。

## **1.2 Unified AI Runtime 的定义**

Unified AI Runtime Engine 是 AI 系统的统一执行基础设施。

它负责：

- 任务接收与执行。
- 任务生命周期管理。
- 执行状态管理。
- 资源分配和调度。
- Worker 生命周期管理。
- 异步执行与并发控制。
- Checkpoint 状态保存。
- 故障恢复。
- 分布式执行。
- 执行隔离。
- 运行时监控。

可以将其理解为：

**AI 系统的 Runtime Kernel。**

这里的 Kernel 是架构层面的比喻，并不意味着它必须实现成操作系统内核。

## **1.3 Runtime Engine 与 Unified AI Runtime 的区别**

| 对比维度   | Runtime Engine | Unified AI Runtime    |
| ------ | -------------- | --------------------- |
| 核心目标   | 提供执行机制         | 统一管理整个执行环境            |
| 任务执行   | 基础执行           | 完整生命周期管理              |
| 调度     | 基础任务调度         | 多任务、多资源调度             |
| 状态     | 执行上下文          | 持久化状态管理               |
| Worker | 执行单元           | Worker 池与生命周期         |
| 故障恢复   | 基础错误处理         | Checkpoint 与 Recovery |
| 分布式    | 可选             | 统一分布式执行模型             |
| 资源管理   | 基础资源使用         | CPU、GPU、内存、模型等资源管理    |

两者并不是完全独立的两个系统。

Unified AI Runtime 可以看作 Runtime Engine 向完整运行时基础设施的扩展。

# **二、Unified AI Runtime 的整体架构**

先看整体结构。

## **Unified AI Runtime Architecture**

**AI Application Layer**

Agent · Workflow · Planning · Model · Tool

**Unified Runtime API**

Submit · Execute · Pause · Resume · Cancel · Query

**Task Execution**

任务执行

**State Manager**

状态管理

**Resource Scheduler**

资源调度

**Worker Runtime**

Worker 管理

**Async Runtime**

异步并发

**Checkpoint Manager**

状态快照

**Recovery Manager**

故障恢复

**Distributed Runtime**

分布式执行

**Infrastructure Layer**

CPU · GPU · Memory · Model Server · Storage · Message Queue · Network

整体分为四层：

1. Application Layer：Agent、Workflow 和模型应用。
2. Runtime API：统一执行接口。
3. Runtime Core：任务、状态、调度、Worker、恢复等核心模块。
4. Infrastructure Layer：实际计算资源和外部基础设施。

核心思想是：

**上层不应该直接管理底层执行细节，而是通过统一 Runtime API 提交和控制任务。**

# **三、Task Execution：如何统一执行不同类型的任务？**

## **3.1 为什么需要统一任务模型？**

AI 系统中存在很多不同类型的任务：

| 任务类型            | 示例             | 主要资源       |
| --------------- | -------------- | ---------- |
| LLM Task        | 文本生成、推理        | GPU、模型服务   |
| Tool Task       | 调用搜索 API       | 网络、CPU     |
| Code Task       | 执行 Python、编译代码 | CPU、内存     |
| Agent Task      | 多轮工具协作         | 综合资源       |
| Data Task       | 文档解析、向量化       | CPU、GPU、内存 |
| Workflow Task   | 执行完整工作流        | 综合资源       |
| Evaluation Task | 模型评估、结果评分      | CPU、GPU    |

如果每种任务都使用完全不同的执行接口，系统会越来越难以维护。

因此需要统一抽象。

## **3.2 定义 Task**

一个 Task 可以包含：

```
interface RuntimeTask {
  id: string;

  type: TaskType;

  status: TaskStatus;

  input: unknown;

  output?: unknown;

  dependencies: string[];

  priority: number;

  resourceRequirements: ResourceRequirements;

  timeout?: number;

  retryPolicy?: RetryPolicy;

  createdAt: number;

  updatedAt: number;
}
```

任务类型：

```
type TaskType =
  | "llm"
  | "tool"
  | "code"
  | "agent"
  | "data"
  | "workflow"
  | "evaluation";
```

任务状态：

```
type TaskStatus =
  | "created"
  | "queued"
  | "scheduled"
  | "running"
  | "waiting"
  | "paused"
  | "succeeded"
  | "failed"
  | "cancelled"
  | "timed_out";
```

注意：这里的 `waiting` 可以表示等待依赖、外部事件或异步结果；具体原因应通过额外字段记录，不能只靠状态名称判断。

## **3.3 Task Execution Pipeline**

任务执行一般包含以下阶段：

```
Task Submission
      |
      v
Task Validation
      |
      v
Dependency Resolution
      |
      v
Resource Scheduling
      |
      v
Worker Assignment
      |
      v
Task Execution
      |
      v
Result Collection
      |
      v
State Update
      |
      v
Task Completion
```

每个阶段都有明确职责。

例如：

- Validation：验证任务结构、权限和输入。
- Dependency Resolution：检查前置任务是否完成。
- Scheduling：选择适合的执行资源。
- Worker Assignment：分配执行 Worker。
- Execution：实际运行任务。
- Result Collection：收集执行结果。
- State Update：持久化任务状态。

## **3.4 Task 与 Job 的区别**

在不同系统中，Task 和 Job 的定义并不统一。

在 mini-llm 中，可以采用以下约定：

- Job：用户提交的一次完整工作请求。
- Task：Job 拆分出的一个执行单元。
- Attempt：Task 的一次具体执行尝试。

例如：

```
Job: 分析一个 GitHub 项目

  Task 1: 下载仓库
    Attempt 1

  Task 2: 分析依赖
    Attempt 1

  Task 3: 分析代码
    Attempt 1
    Attempt 2 (第一次失败后重试)

  Task 4: 生成报告
    Attempt 1
```

这种划分非常重要。

如果没有 Attempt 概念，系统可能无法区分：

- 任务本身失败。
- 某一次执行尝试失败。
- 任务重试后成功。

生产级 Runtime 必须保留这些信息。

# **四、Execution State Machine：如何管理复杂任务状态？**

## **4.1 为什么不能只使用一个布尔值？**

假设：

```
const running = true;
```

这只能表示任务正在运行，无法表达：

- 任务是否已经进入队列？
- 是否正在等待模型？
- 是否暂停？
- 是否失败？
- 是否正在恢复？
- 是否已经取消？

复杂 AI 系统需要状态机。

## **4.2 Task State Machine**

## **Task Lifecycle**

**Created**

**Queued**

**Scheduled**

**Running**

**Succeeded**

**Failed**

**Waiting / Paused**

**Cancelled**

示意状态图。真实状态机还需要定义合法转换、重试、恢复和终态规则。

状态转换需要受到约束。

例如：

```
Created -> Queued
Queued -> Scheduled
Scheduled -> Running

Running -> Waiting
Running -> Paused
Running -> Succeeded
Running -> Failed
Running -> Cancelled

Failed -> Queued       (满足重试条件)
Paused -> Queued       (恢复执行)
Waiting -> Queued      (等待条件满足)
```

终态通常包括：

- Succeeded
- Failed
- Cancelled

如果支持人工恢复失败任务，可以设计显式的恢复操作，而不是随意修改状态。

## **4.3 状态机实现**

```
const transitions: Record<string, string[]> = {
  created: ["queued"],
  queued: ["scheduled", "cancelled"],
  scheduled: ["running", "cancelled"],
  running: [
    "waiting",
    "paused",
    "succeeded",
    "failed",
    "cancelled",
    "timed_out"
  ],
  waiting: ["queued", "cancelled"],
  paused: ["queued", "cancelled"],
  failed: ["queued"],
};

function transition(
  current: string,
  next: string
) {
  if (!transitions[current]?.includes(next)) {
    throw new Error(
      `Invalid transition: ${current} -> ${next}`
    );
  }

  return next;
}
```

真实系统还应该考虑：

1. 并发状态更新。
2. 状态版本号。
3. 数据库事务。
4. 重复事件。
5. Worker 失联。
6. 取消和完成同时发生的竞争条件。

一个常见做法是使用乐观并发控制：

```
UPDATE tasks
SET status = 'running',
    version = version + 1
WHERE id = ?
  AND status = 'scheduled'
  AND version = ?;
```

只有更新成功的 Worker 才获得对应状态转换的所有权。

# **五、Resource Scheduling：如何调度 CPU、GPU、Memory 和 Model？**

这是 Unified AI Runtime 最重要的模块之一。

## **5.1 为什么 AI Runtime 需要资源调度？**

传统任务调度可能只需要考虑 CPU。

但 AI 系统存在不同类型的资源：

- CPU：代码执行、数据处理。
- GPU：模型推理、训练、向量计算。
- Memory：模型加载、上下文、数据缓存。
- VRAM：GPU 模型权重、KV Cache、激活值。
- Model：具体模型及其服务容量。
- Network：外部 API、远程模型调用。
- Storage：文件读取、Checkpoint、数据集。

例如，同时提交三个任务：

| Task   | CPU | GPU 显存 | 内存    |
| ------ | --- | ------ | ----- |
| 文档解析   | 4 核 | 0 GB   | 4 GB  |
| LLM 推理 | 2 核 | 12 GB  | 8 GB  |
| 视频分析   | 6 核 | 8 GB   | 12 GB |

如果机器只有：

- 8 核 CPU。
- 16 GB GPU 显存。
- 16 GB 系统内存。

那么三个任务显然不能全部同时执行。

Runtime 必须决定：

- 哪个任务优先？
- 哪个任务等待？
- 哪个任务可以降级？
- 是否可以选择其他模型？
- 是否可以转移到其他 Worker？

这就是 Resource Scheduling。

## **5.2 Resource Requirements**

定义资源需求：

```
interface ResourceRequirements {
  cpuCores?: number;

  memoryMB?: number;

  gpuMemoryMB?: number;

  gpuCount?: number;

  requiredModel?: string;

  network?: boolean;
}
```

Worker 资源：

```
interface WorkerResources {
  cpuTotal: number;

  cpuAvailable: number;

  memoryTotalMB: number;

  memoryAvailableMB: number;

  gpuMemoryTotalMB: number;

  gpuMemoryAvailableMB: number;

  loadedModels: string[];
}
```

调度器根据任务需求和 Worker 能力进行匹配。

## **5.3 Resource Scheduling Algorithm**

一个简单的调度过程：

```
Task Queue
    |
    v
Read Task Requirements
    |
    v
Discover Available Workers
    |
    v
Filter Incompatible Workers
    |
    v
Calculate Scheduling Score
    |
    v
Select Best Worker
    |
    v
Reserve Resources
    |
    v
Dispatch Task
```

可以定义一个简单的评分模型：

\\[ Score(w,t)= \alpha C(w,t) +\beta M(w,t) +\gamma G(w,t) +\delta L(w,t) \\]

其中：

- \\(C\\)：CPU 适配度。
- \\(M\\)：内存适配度。
- \\(G\\)：GPU 适配度。
- \\(L\\)：模型本地加载带来的性能收益。
- \\(\alpha,\beta,\gamma,\delta\\)：权重。

但需要注意：评分只是候选 Worker 的排序方法。

**资源硬约束必须先满足，不能通过高评分绕过显存不足或权限限制。**

## **5.4 GPU 调度为什么特别复杂？**

GPU 不只是一个简单的可用/不可用资源。

实际情况包括：

1. GPU 显存碎片。
2. 不同模型的显存占用。
3. KV Cache 动态增长。
4. 多模型共享 GPU。
5. CUDA Context 开销。
6. 模型加载和卸载成本。
7. 推理批处理。
8. GPU 多进程竞争。

例如，一个 14GB 显存的 GPU：

```
GPU Memory = 14GB

Model A = 8GB
Model B = 5GB
KV Cache = 2GB
```

总需求：

\\[ 8+5+2=15GB \\]

已经超过显存容量。

即使 CPU 和系统内存还有大量空间，也不能直接认为 GPU 任务能够运行。

可以通过以下策略缓解：

- 模型按需加载。
- 模型驻留与卸载策略。
- Dynamic Batching。
- KV Cache 管理。
- GPU Memory Reservation。
- 模型量化。
- CPU Offloading。
- 多 GPU 分配。

但这些通常需要与具体推理引擎配合，Runtime 不应该假设自己能够直接控制所有 GPU 内部细节。

## **5.5 Model Scheduling：模型也需要调度**

假设系统同时部署：

- 小型模型：快速分类。
- 中型模型：普通对话。
- 大型模型：复杂推理。
- Embedding Model：向量化。

Runtime 可以根据任务类型选择模型。

```
Incoming Task
      |
      v
Model Router
      |
      +------------------+
      |         |        |
      v         v        v
   Small      Medium    Large
   Model      Model     Model
      |         |        |
      +---------+--------+
                |
                v
           Task Result
```

模型路由可以考虑：

- 任务复杂度。
- 模型能力。
- Token 限制。
- 延迟要求。
- 成本预算。
- 模型可用性。
- 数据隐私要求。

这里要区分两个概念：

**Model Gateway 负责模型访问和路由；Resource Scheduler 负责运行资源的分配与容量控制。**

两者可以协作，但不应混为一个模块。

# **六、Worker Runtime：如何管理不同 Engine 的执行 Worker？**

## **6.1 什么是 Worker？**

Worker 是实际执行任务的运行单元。

例如：

```
Unified Runtime
       |
       v
Worker Pool
       |
       +---- LLM Worker
       |
       +---- Tool Worker
       |
       +---- Code Worker
       |
       +---- Data Worker
       |
       +---- Evaluation Worker
```

不同 Worker 处理不同类型的任务。

## **6.2 Worker 接口**

```
interface RuntimeWorker {
  id: string;

  type: string;

  initialize(): Promise<void>;

  execute(
    task: RuntimeTask,
    context: RuntimeContext
  ): Promise<TaskResult>;

  cancel(taskId: string): Promise<void>;

  healthCheck(): Promise<boolean>;

  shutdown(): Promise<void>;
}
```

其中：

- initialize：初始化 Worker。
- execute：执行任务。
- cancel：请求取消任务。
- healthCheck：检查健康状态。
- shutdown：安全关闭。

## **6.3 Worker 生命周期**

```
Created
   |
   v
Initializing
   |
   v
Ready
   |
   v
Busy
   |
   v
Ready
   |
   v
Draining
   |
   v
Stopped
```

如果 Worker 崩溃：

```
Busy
  |
  v
Worker Lost
  |
  v
Task Lease Expired
  |
  v
Recovery Manager
  |
  v
Reschedule Task
```

这里有一个重要概念：

**Worker 失联不代表原任务一定停止。**

例如 Worker 向远程服务发起付款请求后网络断开，Runtime 不知道付款是否成功。

因此，重新调度前必须考虑：

- 是否有幂等键？
- 是否可以查询原操作结果？
- 是否存在重复执行副作用？
- 是否需要人工确认？

## **6.4 Stateless Worker 与 Stateful Worker**

| 类型                | 特征        | 适用场景      |
| ----------------- | --------- | --------- |
| Stateless Worker  | 不依赖本地持久状态 | API、文本处理  |
| Stateful Worker   | 执行依赖持续状态  | 长任务、交互式环境 |
| Ephemeral Worker  | 一次性创建并销毁  | 沙箱执行      |
| Persistent Worker | 长期驻留      | 模型服务      |

推荐的设计原则：

**尽可能将 Worker 设计为无状态，把必要状态放在外部持久化存储中。**

这样更容易扩容、迁移和故障恢复。

不过，GPU 模型 Worker 为了避免频繁加载模型，可能需要长期驻留。这属于有意的资源优化，而不是把任务状态随意留在进程内。

# **七、Async Runtime：如何实现异步、并发和取消？**

## **7.1 为什么 AI Runtime 需要异步执行？**

一个 Agent 可能同时调用：

- 搜索 API。
- 数据库查询。
- 文档解析。
- Embedding 服务。
- LLM 推理服务。

如果全部采用同步阻塞调用，系统吞吐量会受到严重影响。

例如：

```
search_result = search()db_result = query_database()embedding = generate_embedding()llm_result = call_llm()
```

假设每个操作耗时：

| 操作        | 时间  |
| --------- | --- |
| Search    | 2 秒 |
| Database  | 1 秒 |
| Embedding | 3 秒 |
| LLM       | 5 秒 |

串行执行：

\\[ T=2+1+3+5=11s \\]

如果四个操作相互独立，可以并行：

\\[ T\approx \max(2,1,3,5)=5s \\]

实际耗时还会包含调度、网络和资源竞争开销。

## **7.2 Async Execution**

Python 示例：

```
import asyncioasync def execute_task(task):    return await task.execute()async def execute_parallel(tasks):    results = await asyncio.gather(        *(execute_task(task) for task in tasks)    )    return results
```

但这里有一个重要问题：

`asyncio` 并不意味着 CPU 密集型任务自动获得并行计算能力。

对于不同任务，应采用不同执行方式：

| 任务        | 执行机制                           |
| --------- | ------------------------------ |
| 网络请求      | Async I/O                      |
| 数据库请求     | Async I/O / Connection Pool    |
| CPU 密集型计算 | Process Pool                   |
| GPU 推理    | GPU Runtime / Inference Server |
| 外部命令      | Subprocess                     |
| 长时间任务     | Worker Queue                   |

## **7.3 Cancellation：任务取消机制**

用户可能在 Agent 执行过程中点击：

```
Cancel Task
```

Runtime 必须向对应 Worker 发送取消请求。

```
User
 |
 v
Runtime API
 |
 v
Cancellation Manager
 |
 v
Worker
 |
 v
Stop Execution
```

但取消请求不等于任务立即停止。

例如：

- HTTP 请求可能已经发送。
- 数据库事务可能已经提交。
- GPU Kernel 可能已经进入执行。
- 子进程可能需要显式终止。
- 外部工具可能不支持取消。

因此，需要区分：

1. Cancellation Requested：收到取消请求。
2. Cancellation Accepted：执行方接受取消。
3. Cancellation Completed：执行已经停止。
4. Cancellation Failed：无法确认或停止执行。

**取消是一个协作式控制过程，不是简单地把数据库中的 status 改成 cancelled。**

# **八、Checkpoint：如何保存 Agent 和 Task 的执行状态？**

## **8.1 为什么需要 Checkpoint？**

考虑一个长时间运行的任务：

```
读取 1000 个文档
      |
      v
提取文本
      |
      v
生成 Embedding
      |
      v
构建向量索引
      |
      v
生成分析报告
```

如果执行到第 900 个文档时机器崩溃，怎么办？

没有 Checkpoint，就可能需要从头执行。

有 Checkpoint，则可以恢复到最近一次可靠保存的状态。

## **8.2 Checkpoint 的定义**

Checkpoint 是某个任务或执行过程在特定时刻的可恢复状态快照。

可以保存：

- Task ID。
- Job ID。
- 当前执行阶段。
- 已完成的子任务。
- 未完成任务。
- 中间结果引用。
- Agent Session 状态。
- Memory 引用。
- 模型和 Prompt 版本。
- Worker 执行元数据。
- 状态版本号。

示例：

```
interface TaskCheckpoint {
  id: string;

  taskId: string;

  sequence: number;

  status: string;

  currentStep: string;

  completedSteps: string[];

  pendingSteps: string[];

  stateSnapshot: unknown;

  createdAt: number;
}
```

## **8.3 Checkpoint Architecture**

## **Checkpoint Architecture**

**Task / Agent Runtime**

**Checkpoint Manager**

Capture · Validate · Version · Restore

**Metadata Store**

任务状态、版本、索引

**Snapshot Storage**

快照、中间结果、文件

可以采用：

- PostgreSQL / MySQL：Checkpoint 元数据。
- S3 / MinIO：大体积快照。
- Redis：短期状态缓存。
- 本地文件系统：单机开发环境。

不建议将大型模型权重、完整文件内容和所有中间结果直接塞进数据库 JSON 字段。

## **8.4 Checkpoint 的三种策略**

**策略一：Step-based Checkpoint**

每完成一个步骤就保存状态。

```
Step 1 -> Save
Step 2 -> Save
Step 3 -> Save
Step 4 -> Save
```

优点：恢复粒度细。

缺点：写入开销较大。

**策略二：Periodic Checkpoint**

每隔一定时间或完成一定工作量保存一次。

```
Execute
   |
   v
Checkpoint
   |
   v
Execute
   |
   v
Checkpoint
```

优点：降低保存频率。

缺点：故障时可能丢失最近一段进度。

**策略三：Event-based Checkpoint**

在关键状态变化时保存。

例如：

- Agent 完成 Planning。
- Tool 执行成功。
- 重要数据写入。
- 子任务完成。
- 工作流进入新阶段。

对于 mini-llm，建议先采用 Step-based 和关键事件触发的混合方式。

## **8.5 Checkpoint 不等于完整进程快照**

这是一个容易误解的地方。

保存 Agent 的上下文、工具结果和执行进度，并不意味着可以恢复原来的 Python 线程、GPU Kernel 或任意操作系统进程。

例如：

```
Checkpoint:
  completed_steps = ["search", "analysis"]
  pending_steps = ["code_review"]
```

Runtime 可以重新创建 Worker，从 `code_review` 继续执行。

但它不能凭这份 JSON 快照自动恢复任意进程的 CPU 寄存器、线程栈和 GPU 内部状态。

**工程上更实用的是逻辑状态恢复，而不是通用进程级快照。**

# **九、Recovery：如何从故障中恢复？**

Checkpoint 解决的是状态保存。

Recovery 解决的是故障后如何继续运行。

## **9.1 故障有哪些类型？**

| 故障                | 典型表现   | 恢复策略    |
| ----------------- | ------ | ------- |
| Worker Crash      | 进程崩溃   | 重新调度    |
| Model Failure     | 推理服务异常 | 重试或切换模型 |
| Network Failure   | 网络超时   | 退避重试    |
| Memory Exhaustion | 内存不足   | 降级或释放资源 |
| GPU OOM           | 显存不足   | 调整批次或模型 |
| Host Failure      | 主机宕机   | 跨节点恢复   |
| Storage Failure   | 快照不可用  | 副本或备份恢复 |
| Task Timeout      | 超过时间限制 | 终止或补偿   |

## **9.2 Recovery Pipeline**

```
Failure Detected
      |
      v
Failure Classification
      |
      v
Check Retry Policy
      |
      v
Load Latest Checkpoint
      |
      v
Validate Recovery State
      |
      v
Select New Worker
      |
      v
Restore Task Context
      |
      v
Resume Execution
      |
      v
Verify Result
```

## **9.3 Retry Policy**

一个常见策略是 Exponential Backoff：

\\[ T_n=\min(T\_{max},T_0 \cdot 2^n) \\]

例如：

| 重试次数 | 等待时间 |
| ---- | ---- |
| 1    | 1 秒  |
| 2    | 2 秒  |
| 3    | 4 秒  |
| 4    | 8 秒  |
| 5    | 16 秒 |

实际生产环境通常还会加入随机抖动（Jitter），避免大量任务同时重试造成流量尖峰。

```
import randomdef retry_delay(attempt):    base = min(30, 2 ** attempt)    return random.uniform(base * 0.5, base * 1.5)
```

但是，并不是所有错误都应该重试。

例如：

- 参数错误：通常不重试。
- 权限拒绝：通常不重试。
- 临时网络错误：可以重试。
- 模型服务暂时不可用：可以重试。
- GPU OOM：应该先调整资源策略。
- 外部付款结果未知：不能盲目重试。

## **9.4 Exactly-once Execution 是一个重要陷阱**

很多系统希望实现：

> 一个任务只执行一次。

但分布式环境中，这个目标非常困难。

例如：

```
Worker A
   |
   v
调用外部支付 API
   |
   v
支付成功
   |
   X
网络断开
```

Runtime 没有收到成功响应，于是重新分配给 Worker B。

```
Worker B
   |
   v
再次调用支付 API
   |
   v
可能重复支付
```

因此，需要区分：

- At-most-once：最多执行一次，可能丢失。
- At-least-once：至少执行一次，可能重复。
- Effectively-once：通过幂等键、去重、事务和结果查询，使业务效果尽量等价于一次。

对于涉及外部副作用的任务，建议：

1. 每个业务操作生成稳定的 Idempotency Key。
2. 保存操作执行记录。
3. 外部服务支持结果查询时，优先查询。
4. 对未知结果进入 reconciliation 状态。
5. 无法自动确认时交给人工处理。

# **十、Distributed Runtime：如何跨机器执行任务？**

## **10.1 为什么需要分布式 Runtime？**

单机 Runtime 有资源上限。

例如：

```
Machine A
CPU: 16 Cores
RAM: 64 GB
GPU: 24 GB
```

如果需要执行大量模型推理、代码分析和数据处理任务，可以扩展为：

```
                 Runtime Coordinator
                         |
          +--------------+--------------+
          |              |              |
          v              v              v
       Worker A       Worker B       Worker C
       CPU Tasks      GPU Tasks      Data Tasks
```

Coordinator 负责统一调度，Worker 负责实际执行。

## **10.2 Coordinator 与 Worker**

**Coordinator：**

- 接收任务。
- 管理任务状态。
- 维护依赖关系。
- 分配资源。
- 选择 Worker。
- 处理超时。
- 触发恢复。

**Worker：**

- 注册自身能力。
- 接收任务。
- 执行任务。
- 汇报心跳。
- 上报执行结果。
- 接受取消和关闭指令。

## **10.3 Worker Registration**

Worker 启动时向 Coordinator 注册：

```
{
  "worker_id": "worker-gpu-01",
  "capabilities": {
    "cpu_cores": 16,
    "memory_mb": 65536,
    "gpu_memory_mb": 24576,
    "models": [
      "embedding-model",
      "llm-model"
    ]
  }
}
```

Coordinator 根据这些信息判断 Worker 是否适合执行某个任务。

但 Worker 上报的能力不能直接视为永远准确的事实。

运行过程中，资源会变化，因此还需要：

- 周期性心跳。
- 动态资源报告。
- 任务租约。
- Worker 健康检查。
- 资源预留与释放。
- 失联检测。

## **10.4 Task Lease**

假设 Coordinator 将任务分配给 Worker A。

如果 Worker A 崩溃，Coordinator 怎么知道？

可以引入 Lease（租约）。

```
Task Assigned
      |
      v
Lease Created
      |
      v
Worker Heartbeat
      |
      v
Lease Renewed
      |
      v
Worker Lost
      |
      v
Lease Expired
      |
      v
Recovery
```

Lease 不是永久锁。

Worker 必须在租约过期之前持续证明自己仍然拥有执行资格。

但即使 Lease 过期，也不能保证旧 Worker 已经停止执行。因此，任务状态更新应通过 fencing token 或版本号限制旧 Worker，外部副作用仍需幂等保护。

## **10.5 分布式 Runtime 的核心挑战**

| 问题        | 解决方向                        |
| --------- | --------------------------- |
| Worker 失联 | Heartbeat + Lease           |
| 任务重复执行    | Idempotency + Deduplication |
| 状态不一致     | Versioning + Transaction    |
| 消息丢失      | Durable Queue               |
| 任务分配冲突    | Lease + Fencing Token       |
| 节点资源不足    | Resource Scheduler          |
| 网络分区      | 超时、隔离、恢复策略                  |
| 跨节点状态恢复   | Shared Checkpoint Storage   |

分布式 Runtime 不只是把任务发送到其他机器。

它需要处理网络不可靠、节点失效、重复执行和状态一致性。

# **十一、Runtime Isolation：如何隔离不同 Agent 和任务？**

AI Agent 可能执行不可信代码、访问文件和调用外部工具。

因此 Runtime 需要隔离机制。

## **11.1 为什么需要隔离？**

假设系统有三个 Agent：

```
Agent A: 分析 GitHub 仓库
Agent B: 执行 Python 代码
Agent C: 处理用户上传文件
```

如果三个 Agent 共享同一个执行环境，可能出现：

- 文件互相覆盖。
- 环境变量泄漏。
- 进程互相干扰。
- CPU 和内存竞争。
- 恶意代码影响其他任务。

## **11.2 Isolation Levels**

| 隔离方式                | 隔离强度 | 典型用途      |
| ------------------- | ---- | --------- |
| Logical Isolation   | 低    | 普通任务上下文隔离 |
| Process Isolation   | 中    | 独立进程执行    |
| Container Isolation | 较高   | 代码执行环境    |
| VM Isolation        | 更高   | 强隔离工作负载   |
| Hardware Isolation  | 依赖平台 | 特定高安全环境   |

容器通常共享宿主机内核，因此不能简单认为容器与虚拟机具有相同的安全边界。

## **11.3 Runtime Sandbox**

可以为每个 Code Task 创建独立 Sandbox：

```
Code Task
    |
    v
Sandbox Manager
    |
    v
Container Runtime
    |
    +-- CPU Limit
    +-- Memory Limit
    +-- Process Limit
    +-- Filesystem Isolation
    +-- Network Policy
    +-- Execution Timeout
    |
    v
Execute Code
```

对于高风险任务，还应考虑：

- 禁止默认访问宿主机敏感路径。
- 限制网络出口。
- 使用非特权用户。
- 删除不必要的 Linux Capabilities。
- 限制系统调用。
- 只读挂载。
- 独立的临时工作目录。
- 清理执行后的资源。

如果使用 Youki 等 OCI Runtime 构建自己的容器基础设施，Runtime Engine 可以通过统一 Sandbox API 调用容器执行层，而不需要将容器实现细节混入任务调度器。

这也意味着：

**AI Runtime 与 Container Runtime 应当保持清晰的分层。**

AI Runtime 管理任务和 Agent；Container Runtime 提供进程隔离和执行环境。

# **十二、Runtime Observability：如何统一追踪任务执行？**

Unified Runtime 还必须回答：

- 当前有多少任务？
- 哪些任务正在执行？
- 哪些任务排队时间过长？
- 哪个 Worker 出现故障？
- GPU 利用率是多少？
- 任务为什么失败？
- Checkpoint 恢复了多少次？
- 单个 Agent 花费了多少 Token？

可以建立以下指标：

| 指标                  | 说明         |
| ------------------- | ---------- |
| Task Queue Length   | 等待执行的任务数量  |
| Task Execution Time | 任务执行耗时     |
| Task Success Rate   | 任务成功率      |
| Worker Utilization  | Worker 利用率 |
| CPU Usage           | CPU 使用率    |
| GPU Memory Usage    | GPU 显存使用   |
| Recovery Count      | 恢复次数       |
| Retry Count         | 重试次数       |
| Checkpoint Duration | 快照保存耗时     |
| Scheduling Latency  | 调度延迟       |

建议使用统一 Trace ID：

```
Job ID
  |
  v
Task ID
  |
  v
Attempt ID
  |
  v
Worker ID
  |
  v
Engine ID
  |
  v
Tool Call ID
```

这样才能把一个复杂 Agent 的整个执行过程关联起来。

需要特别注意：Runtime Trace 记录的是可审计的执行事件、输入输出摘要、状态变化和资源信息，并不等于暴露模型完整的内部思维过程。

# **十三、mini-llm Unified Runtime 的完整工程架构**

经过上面的分析，可以设计如下工程目录：

```
mini-llm/
│
├── apps/
│   ├── runtime-server/
│   ├── runtime-worker/
│   └── runtime-dashboard/
│
├── packages/
│   │
│   ├── runtime-core/
│   │   ├── task/
│   │   ├── job/
│   │   ├── execution/
│   │   └── lifecycle/
│   │
│   ├── scheduler/
│   │   ├── task-scheduler/
│   │   ├── resource-scheduler/
│   │   ├── model-scheduler/
│   │   └── priority-queue/
│   │
│   ├── worker-runtime/
│   │   ├── worker/
│   │   ├── worker-pool/
│   │   ├── registration/
│   │   └── heartbeat/
│   │
│   ├── async-runtime/
│   │   ├── executor/
│   │   ├── concurrency/
│   │   └── cancellation/
│   │
│   ├── state-manager/
│   │   ├── state-machine/
│   │   ├── persistence/
│   │   └── versioning/
│   │
│   ├── checkpoint-manager/
│   │   ├── snapshot/
│   │   ├── storage/
│   │   └── restore/
│   │
│   ├── recovery-manager/
│   │   ├── retry/
│   │   ├── failure-detection/
│   │   └── reconciliation/
│   │
│   ├── distributed-runtime/
│   │   ├── coordinator/
│   │   ├── worker-client/
│   │   ├── lease/
│   │   └── message-transport/
│   │
│   ├── isolation/
│   │   ├── sandbox/
│   │   ├── container/
│   │   └── resource-limit/
│   │
│   └── runtime-observability/
│       ├── metrics/
│       ├── tracing/
│       └── events/
│
└── infrastructure/
    ├── database/
    ├── message-queue/
    ├── object-storage/
    └── deployment/
```

这里的目录是逻辑模块划分，不意味着 V1 就要把所有模块拆成独立微服务。

我更建议采用：

**模块化单体 + 独立 Worker 进程。**

这样既能保持模块边界，也不会过早引入复杂的分布式部署成本。

# **十四、Unified Runtime API 设计**

上层 Agent 和 Workflow 通过统一 API 控制任务。

```
interface UnifiedRuntime {
  submit(
    task: RuntimeTask
  ): Promise<string>;

  getTask(
    taskId: string
  ): Promise<RuntimeTask>;

  pause(
    taskId: string
  ): Promise<void>;

  resume(
    taskId: string
  ): Promise<void>;

  cancel(
    taskId: string
  ): Promise<void>;

  checkpoint(
    taskId: string
  ): Promise<string>;

  recover(
    taskId: string
  ): Promise<void>;

  getExecutionTrace(
    taskId: string
  ): Promise<ExecutionTrace>;
}
```

简单使用：

```
const taskId = await runtime.submit({
  id: "task-001",
  type: "agent",
  status: "created",
  input: {
    prompt: "分析 mini-llm 项目架构"
  },
  dependencies: [],
  priority: 5,
  resourceRequirements: {
    cpuCores: 2,
    memoryMB: 4096
  },
  createdAt: Date.now(),
  updatedAt: Date.now()
});

const task = await runtime.getTask(taskId);

console.log(task.status);
```

需要注意：这是概念性 API。生产实现还要定义鉴权、租户隔离、请求幂等、并发控制、错误码、事件订阅和持久化契约。

# **十五、V1 应该如何实现？**

Unified Runtime 看起来非常复杂，但第一版没有必要实现所有能力。

我建议分四个阶段。

## **mini-llm Unified Runtime V1 Roadmap**

1. **阶段一：单机任务运行时**

   优先开发
   - Task / Job / Attempt
   - Task State Machine
   - Sequential / Parallel Execution
   - Async Worker Pool
   - Timeout / Cancellation
2. **阶段二：持久化与恢复**
   - Task Persistence
   - Checkpoint Manager
   - Retry Policy
   - Worker Heartbeat
   - Failure Recovery
3. **阶段三：资源调度**
   - CPU / Memory 限制
   - GPU 资源感知
   - Model Routing Integration
   - Worker Capability
   - Priority Scheduling
4. **阶段四：分布式运行时**
   - Runtime Coordinator
   - Remote Worker
   - Task Lease
   - Distributed Checkpoint
   - Cross-node Recovery

## **V1 验收标准**

| 测试项           | 验收要求                   |
| ------------- | ---------------------- |
| 任务执行          | 能执行 LLM、Tool、Code 三种任务 |
| 并发执行          | 能控制最大并发数量              |
| 状态管理          | 非法状态转换被拒绝              |
| 依赖调度          | 前置任务未完成时不执行后续任务        |
| 超时            | 能识别超时任务                |
| 取消            | 能发送取消请求并记录最终结果         |
| 持久化           | 服务重启后可以恢复任务状态          |
| Checkpoint    | 能从最近一个有效检查点恢复          |
| Retry         | 可配置重试次数和退避策略           |
| Observability | 能查询完整任务执行事件            |

# **十六、从 Unified Runtime 到 AI Operating System**

到这里，我们可以看到 mini-llm 的架构已经不再是一个简单的 LLM 项目。

它正在逐渐形成一个完整的 AI 系统：

## **mini-llm AI Operating System**

**AI Application**

Chat · Coding · Research · Automation · Robotics

Agent Engine

Planning Engine

Decision Engine

Cognitive Architecture

Memory Architecture

Knowledge Architecture

**Unified AI Runtime**

Execution · Scheduling · State · Checkpoint · Recovery

Model Gateway

System Integration

Protocol Engine

Observability Engine

Security Engine

Evaluation Engine

**Infrastructure**

CPU · GPU · Memory · Storage · Network · Container · Distributed Nodes

从系统工程角度看，这里出现了一个重要变化：

过去：

```
Application -> LLM API -> Response
```

现在：

```
Application
    |
    v
Cognitive Architecture
    |
    v
Agent / Planning / Decision
    |
    v
Unified Runtime
    |
    v
Model / Tool / Memory / Environment
    |
    v
Observable and Recoverable Execution
```

AI 不仅需要生成内容，还需要具备持续执行任务的能力。

而持续执行任务，就必须拥有状态管理、资源管理、故障恢复和执行控制。

这也是 Unified AI Runtime 的核心价值。

# **十七、总结：Unified Runtime 是 AI 系统的执行基础设施**

可以用一句话概括：

**Agent 决定做什么，Planning 决定怎么做，Unified Runtime 负责让这些计划在真实计算环境中可靠地执行。**

三者的职责：

| 模块                        | 核心问题            |
| ------------------------- | --------------- |
| Agent Engine              | 谁来完成任务？         |
| Planning Engine           | 任务应该如何分解？       |
| Decision Engine           | 当前应该选择什么行动？     |
| System Integration Engine | 不同 Engine 如何协作？ |
| Unified Runtime           | 如何可靠地执行？        |
| Observability Engine      | 执行过程发生了什么？      |
| Security Engine           | 哪些操作被允许？        |

真正成熟的 AI Runtime，不只是一个任务队列，也不是一个简单的异步执行器。

它应该成为整个 AI 系统的统一执行底座。

下一篇建议研究：

**《mini-llm Model Gateway Engine 深度解析：从 Model Registry、Model Routing 到 Inference Serving、Dynamic Batching、KV Cache 与 Model Lifecycle》**

重点研究模型注册、模型路由、多模型服务、推理请求调度、动态批处理、KV Cache 管理、模型加载卸载、模型降级与推理服务的高可用架构。
