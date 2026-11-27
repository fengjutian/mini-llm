# **《mini-llm Constraint Engine 深度解析（上）：从 Constraint Modeling、Constraint Graph 到 Constraint Hierarchy》**

## **—— AI 如何理解、管理并遵守现实软件工程中的复杂约束？**

在前面的讨论中，我们提出了一个重要观点：

**代码生成只是能力，约束满足才是工程。**

AI 可以生成一个功能完整的模块，但这个模块不一定符合现有架构；可以修改数据库，却可能破坏历史数据；可以优化性能，却可能牺牲事务一致性。

这说明，AI 编程真正需要解决的问题，不只是如何生成正确的代码，而是：

- 如何理解现实项目中的规则？
- 如何知道哪些规则不能违反？
- 如何识别规则之间的依赖关系？
- 如何处理相互冲突的工程要求？
- 如何证明生成的代码满足这些要求？
- 如何在项目持续演进时保持规则有效？

因此，我们需要引入一个独立的工程引擎：

**Constraint Engine（约束引擎）。**

本篇重点研究它的基础理论、数据模型和核心架构。

# **第一章：Constraint Engine 是什么？**

## **1.1 从 Code Generation 到 Constraint Satisfaction**

传统 AI 编程主要采用这样的工作模式：

**User Prompt**

用户描述开发需求

**LLM Code Generation**

模型生成代码

**Source Code**

输出代码

这种模式有一个明显的问题：

LLM 主要根据当前上下文预测合理的代码实现，却没有天然保证代码符合项目的全部工程约束。

我们希望升级为：

**User Requirement**

**Constraint Modeling**

提取需求、规则、依赖和边界

**Constraint Evaluation**

分析可行性和规则冲突

**AI Planning & Coding**

在约束范围内规划和生成代码

**Constraint Validation**

静态分析、测试、策略和证据验证

**Verified Code Changes**

两者的根本区别：

| 传统代码生成 | 约束驱动编程 |
| ----------- | ----------- |
| 关注生成什么代码 | 关注允许生成什么代码 |
| 主要依赖 Prompt | 依赖结构化约束 |
| 局部实现导向 | 全局规则导向 |
| 生成后检查 | 生成前分析、生成后验证 |
| 关注功能完成 | 关注工程正确性 |

Constraint Engine 的目标不是取代 LLM，而是为 LLM 建立工程边界。

## **1.2 Constraint Engine 的定义**

可以将其定义为：

> Constraint Engine 是负责提取、建模、组织、分析、验证和管理软件工程约束的独立系统，为 AI Agent 提供可计算、可验证和可追溯的工程规则体系。

它主要承担六种职责：

1. Constraint Extraction：约束提取。
2. Constraint Modeling：约束建模。
3. Constraint Dependency：约束依赖分析。
4. Constraint Conflict Detection：冲突检测。
5. Constraint Validation：约束验证。
6. Constraint Evolution：约束演进管理。

# **第二章：Constraint Modeling——如何把现实规则转换成机器能够理解的模型？**

这是整个 Constraint Engine 最基础的能力。

现实项目中的规则往往以不同形式存在：

- 产品需求文档。
- 技术设计文档。
- 数据库 Schema。
- API 文档。
- TypeScript 类型。
- ESLint 规则。
- 单元测试。
- CI/CD 配置。
- 开发人员的经验。
- 历史代码中的隐含约定。

这些信息并不是统一格式。

AI 首先需要将它们转换成结构化的约束模型。

## **2.1 什么是 Constraint？**

在数学中，约束可以理解为限制变量取值范围或限制变量之间关系的条件。

例如：

\\[ x > 0 \\]

表示变量 \\(x\\) 必须大于零。

软件工程中也有类似约束：

```
Controller cannot directly access Database.
```

它可以表达为：

\\[ Controller \not\rightarrow Database \\]

再例如：

```
Order amount must be greater than or equal to zero.
```

可以表达为：

\\[ Order.amount \geq 0 \\]

更复杂的规则：

```
A paid order cannot be deleted.
```

可以表示为：

\\[ Order.status = PAID \Rightarrow Delete(Order) = Forbidden \\]

这意味着现实中的工程规则可以被转换成不同类型的逻辑表达式。

## **2.2 Constraint 的基本数据结构**

我们可以先设计统一的 Constraint Model。

```
interface Constraint {
  id: string;

  name: string;

  description: string;

  type: ConstraintType;

  severity: ConstraintSeverity;

  scope: ConstraintScope;

  expression?: ConstraintExpression;

  source: ConstraintSource;

  validator?: ValidatorConfig;

  status: ConstraintStatus;

  version: number;
}
```

定义约束类型：

```
type ConstraintType =
  | "requirement"
  | "architecture"
  | "dependency"
  | "api"
  | "data"
  | "business"
  | "security"
  | "performance"
  | "environment"
  | "compatibility"
  | "quality"
  | "maintenance";
```

定义严重等级：

```
type ConstraintSeverity =
  | "critical"
  | "high"
  | "medium"
  | "low";
```

定义约束状态：

```
type ConstraintStatus =
  | "active"
  | "deprecated"
  | "suspended"
  | "proposed";
```

这样就可以将原来分散在文档、代码和开发人员经验中的规则，逐步统一管理。

## **2.3 Constraint Expression：约束表达式**

仅仅保存规则描述还不够。

例如：

```
Service must not access another module's database.
```

这句话对人类很容易理解，但计算机无法直接判断是否违反。

因此需要设计约束表达式。

例如：

```
{
  "operator": "FORBIDDEN_DEPENDENCY",
  "source": "modules/*/service",
  "target": "modules/*/repository",
  "condition": "cross_module"
}
```

又例如：

```
{
  "operator": "REQUIRE",
  "condition": {
    "field": "order.status",
    "equals": "PAID"
  },
  "effect": {
    "operation": "DELETE",
    "allowed": false
  }
}
```

约束表达式需要支持：

| 表达式类别 | 含义 |
| ------------ | ----- |
| REQUIRE | 必须满足 |
| FORBID | 禁止发生 |
| ALLOW | 允许发生 |
| EQUALS | 相等 |
| NOT_EQUALS | 不相等 |
| GREATER_THAN | 大于 |
| LESS_THAN | 小于 |
| DEPENDS_ON | 依赖 |
| BEFORE | 顺序约束 |
| AFTER | 后置条件 |
| EXISTS | 必须存在 |
| NOT_EXISTS | 不允许存在 |

需要注意：这些是我们设计的 DSL 操作符，不是某个现成的统一行业标准。

## **2.4 Constraint Source：约束来源**

这是一个非常重要的设计。

现实项目中，规则不是凭空产生的。

例如：

| 来源 | 示例 |
| --------------------- | --------------------------- |
| PRD | 已支付订单不可删除 |
| Architecture Document | Controller 不允许访问 Repository |
| OpenAPI | API 响应必须遵循统一格式 |
| Database Schema | 用户 ID 不允许重复 |
| ESLint | 禁止使用 any |
| Test | 删除订单必须保留审计记录 |
| Developer Decision | 不允许引入新的 UI 框架 |

每条约束都必须记录来源。

否则，当两条规则发生冲突时，系统不知道应该相信哪一条。

建议定义：

```
interface ConstraintSource {
  sourceType:
    | "document"
    | "code"
    | "schema"
    | "test"
    | "policy"
    | "human_decision";

  sourceId: string;

  location?: string;

  createdAt: string;

  confidence?: number;
}
```

这里的 `confidence` 特别值得注意。

例如：

- 从明确的架构文档提取：高置信度。
- 从多个现有代码模式推断：中等置信度。
- 从单个函数推测开发习惯：低置信度。

AI 不应该把推测出来的规则当成正式的强制规则。

# **第三章：Constraint Graph——如何理解约束之间的关系？**

单独管理每条约束是不够的。

因为现实项目中的规则通常不是孤立存在的。

例如：

```
订单不能直接删除
       |
       v
必须执行逻辑删除
       |
       v
必须保留历史记录
       |
       v
必须记录操作人
       |
       v
必须写入审计日志
```

一条业务规则可能衍生出多条技术约束。

这就需要 Constraint Graph。

## **3.1 什么是 Constraint Graph？**

Constraint Graph 是用于描述约束之间关联关系的图结构。

我们可以把每条 Constraint 看成一个 Node。

约束之间的关系看成 Edge。

例如：

**Constraint Dependency Graph**

**C1：已支付订单不可直接删除**

Business Constraint

REQUIRES

**C2：逻辑删除**

Data Constraint

**C3：记录操作人**

Audit Constraint

**C4：保留历史数据**

**C5：写入审计日志**

这样，AI 不仅知道订单不能删除，还能理解这个规则背后的技术要求。

## **3.2 Constraint Graph 的节点类型**

建议支持以下节点：

```
type ConstraintNodeType =
  | "constraint"
  | "requirement"
  | "module"
  | "component"
  | "api"
  | "database"
  | "entity"
  | "test"
  | "policy"
  | "decision";
```

为什么需要 Module、API、Database 等节点？

因为约束最终需要映射到具体的软件对象。

例如：

```
Constraint:
  Controller cannot access Database

关联节点:
  UserController
  UserRepository
  UserDatabase
```

这样，当 AI 修改 `UserController` 时，系统就能找到相关约束。

## **3.3 Constraint Graph 的边类型**

```
type ConstraintRelation =
  | "REQUIRES"
  | "DEPENDS_ON"
  | "CONFLICTS_WITH"
  | "IMPLEMENTS"
  | "VALIDATES"
  | "SUPERSEDES"
  | "APPLIES_TO"
  | "DERIVED_FROM";
```

这些关系有不同含义。

| 关系 | 含义 |
| -------------- | ----------- |
| REQUIRES | 一条约束要求另一条约束 |
| DEPENDS_ON | 一条约束依赖另一条约束 |
| CONFLICTS_WITH | 两条约束存在冲突 |
| IMPLEMENTS | 某个模块实现约束 |
| VALIDATES | 某个测试验证约束 |
| SUPERSEDES | 新规则替代旧规则 |
| APPLIES_TO | 约束适用于某个模块 |
| DERIVED_FROM | 约束由其他规则推导而来 |

## **3.4 为什么不能只使用普通依赖图？**

这是一个值得注意的问题。

软件依赖图通常表达：

```
Module A -> Module B
```

而 Constraint Graph 表达：

```
Constraint A
    |
    +---- requires ----> Constraint B
    |
    +---- conflicts ---> Constraint C
    |
    +---- applies to --> Module D
    |
    +---- validated by -> Test E
```

二者的语义不同。

依赖图主要关注模块之间的结构关系。

约束图关注规则之间的逻辑关系、适用范围和验证证据。

实际系统中，两张图需要互相关联，但不应该混成一张图。

# **第四章：Constraint Hierarchy——如何确定规则的优先级？**

现实软件项目存在很多规则。

但它们并不是平等的。

例如：

- 代码必须通过 ESLint。
- 用户数据不能泄露。
- API 必须兼容旧版本。
- 新代码尽量遵守团队命名规范。

假设 AI 为了通过 ESLint，修改了一个旧接口，导致历史客户端无法使用。

这显然是错误的。

问题在于：AI 没有正确理解规则的优先级。

## **4.1 三层基础约束体系**

我们先设计三层。

**Hard Constraints**

强制约束

不允许违反。

例如数据完整性、安全边界、强制业务规则。

**Soft Constraints**

软约束

可以权衡，但必须说明原因。

例如性能目标、复杂度、可维护性。

**Preferences**

偏好规则

尽量遵守。

例如命名习惯、注释风格、代码格式。

## **4.2 Hard Constraints：强制约束**

例如：

```
id: SEC-001

name: User Data Isolation

type: security

level: hard

rule:
  operator: FORBID
  expression: cross_tenant_data_access

enforcement:
  mode: blocking
```

如果 AI 生成的代码存在跨租户数据访问，系统必须阻止变更进入集成阶段。

注意：Hard 并不意味着所有情况都永远不能改变，而是不能由 AI 擅自突破。修改规则本身必须经过授权的治理流程。

## **4.3 Soft Constraints：软约束**

例如：

```
id: PERF-001

name: API Response Time

level: soft

target:
  p95: 200ms

evaluation:
  type: benchmark

weight: 0.8
```

如果 API 响应时间为 220ms，系统不一定立即拒绝代码。

它可以进一步分析：

- 是否超过实际业务容忍度？
- 是否影响核心用户路径？
- 是否增加了服务器成本？
- 是否存在更好的优化方案？

软约束需要优化，而不是简单地通过或拒绝。

## **4.4 Preferences：偏好规则**

例如：

```
id: STYLE-001

name: Service Naming Convention

level: preference

rule:
  pattern: "*Service"

enforcement:
  mode: advisory
```

如果 AI 使用了其他命名方式，可以给出提示，但不一定阻止变更。

## **4.5 约束优先级不等于简单数字排序**

一个容易出现的设计错误：

```
Hard = 100
Soft = 50
Preference = 10
```

然后认为只要数字大，规则就一定优先。

实际上，优先级还涉及：

1. 规则的来源。
2. 规则的适用范围。
3. 规则的生效时间。
4. 规则的授权主体。
5. 规则之间的明确覆盖关系。
6. 法律、组织和项目治理要求。

例如，一条较新的项目规则不一定能覆盖更高层级的安全要求。

因此需要引入：

**Constraint Governance（约束治理）。**

它负责确定规则的权威性、作用域和修改权限。

# **第五章：Constraint Engine 的核心工作流程**

前面已经介绍了三个基础模块：

- Constraint Modeling。
- Constraint Graph。
- Constraint Hierarchy。

现在把它们组合起来。

## **Constraint Engine Pipeline**

1. **Constraint Extraction**：从 PRD、代码、数据库、测试和架构文档提取规则。
2. **Constraint Normalization**：将自然语言和分散配置转换为统一数据结构。
3. **Constraint Classification**：判断约束类型、级别和适用范围。
4. **Constraint Graph Construction**：建立依赖、冲突、实现和验证关系。
5. **Constraint Conflict Analysis**：检测规则冲突，识别不可满足条件。
6. **Constraint Evaluation**：根据当前任务筛选需要执行的约束。
7. **Constraint Validation**：对 AI 生成的代码进行自动化验证。
8. **Evidence Recording**：保存验证结果、证据和版本信息。

这里有一个关键设计：

不要每次让 AI 读取整个项目的所有规则。

应该通过任务范围、代码变更范围和依赖关系，筛选出当前任务相关的约束子图。

例如：

AI 正在修改 `OrderService.ts`。

系统只需要优先加载：

- Order 相关业务规则。
- OrderService 所在模块的架构规则。
- 涉及的 API 契约。
- 相关数据库约束。
- 对应测试要求。
- 安全和权限规则。

这能够减少上下文开销，同时降低遗漏规则的概率。

# **第六章：mini-llm Constraint Engine 的架构设计**

我们可以把它拆成七个核心子系统。

## **Constraint Engine Architecture**

- **Constraint API**：统一入口与任务上下文。
- **Extraction Engine**：约束提取。
- **Modeling Engine**：规则结构化。
- **Graph Engine**：关系构建。
- **Hierarchy Engine**：优先级治理。
- **Conflict Engine**：冲突分析。
- **Validation Engine**：约束验证。
- **Evidence & Governance Engine**：证据记录、规则授权、版本管理。
- **Constraint Storage**：规则库、关系库、验证记录。

建议的目录结构：

```
mini-llm/
└── engines/
    └── constraint-engine/
        ├── core/
        │   ├── constraint.ts
        │   ├── expression.ts
        │   ├── hierarchy.ts
        │   └── context.ts
        │
        ├── extraction/
        │   ├── document-parser.ts
        │   ├── code-analyzer.ts
        │   ├── schema-parser.ts
        │   └── llm-extractor.ts
        │
        ├── modeling/
        │   ├── normalizer.ts
        │   ├── classifier.ts
        │   └── rule-compiler.ts
        │
        ├── graph/
        │   ├── graph.ts
        │   ├── node.ts
        │   ├── edge.ts
        │   └── traversal.ts
        │
        ├── conflict/
        │   ├── detector.ts
        │   ├── resolver.ts
        │   └── analyzer.ts
        │
        ├── validation/
        │   ├── static-validator.ts
        │   ├── test-validator.ts
        │   ├── policy-validator.ts
        │   └── benchmark-validator.ts
        │
        ├── governance/
        │   ├── approval.ts
        │   ├── versioning.ts
        │   └── audit.ts
        │
        ├── evidence/
        │   ├── evidence.ts
        │   └── recorder.ts
        │
        └── index.ts
```

这个目录只是逻辑模块划分，V1 不需要一开始就把每个目录实现成独立服务。

# **第七章：一个完整案例——AI 修改订单模块**

假设我们要求 AI：

> 为订单系统增加取消订单功能。

项目已经存在以下规则：

```
C1：已支付订单不能直接取消。

C2：未支付订单允许取消。

C3：取消订单必须记录操作人。

C4：取消订单必须保留审计日志。

C5：订单金额必须保持不变。

C6：不能破坏原有订单 API。

C7：Controller 不能直接操作数据库。
```

AI 开始生成代码之前，Constraint Engine 首先分析规则。

## **7.1 建立约束模型**

```
const constraints = [
  {
    id: "C1",
    level: "hard",
    type: "business"
  },
  {
    id: "C2",
    level: "hard",
    type: "business"
  },
  {
    id: "C3",
    level: "hard",
    type: "audit"
  },
  {
    id: "C4",
    level: "hard",
    type: "audit"
  },
  {
    id: "C5",
    level: "hard",
    type: "data"
  },
  {
    id: "C6",
    level: "hard",
    type: "compatibility"
  },
  {
    id: "C7",
    level: "hard",
    type: "architecture"
  }
];
```

## **7.2 AI 生成实现**

```
async function cancelOrder(orderId: string) {
  const order = await repository.findById(orderId);

  if (!order) {
    throw new Error("Order not found");
  }

  if (order.status === "PAID") {
    throw new Error("Paid order cannot be cancelled");
  }

  order.status = "CANCELLED";

  await repository.save(order);
}
```

这段代码看起来没问题。

但它没有记录操作人，也没有记录审计日志。

## **7.3 Constraint Validation**

**Validation Report**

| 约束 | 校验项 | 结果 |
| ---- | -------------- | ------ |
| C1 | 已支付订单限制 | 通过 |
| C2 | 未支付订单取消 | 通过 |
| C3 | 操作人记录 | 失败 |
| C4 | 审计日志 | 失败 |
| C5 | 金额不变 | 待验证 |
| C6 | API 兼容 | 通过 |
| C7 | 架构分层 | 通过 |

示意报告：真实通过状态必须来自实际执行的验证器，不能仅凭代码外观判定。

系统将验证失败的信息返回给 AI：

```
Constraint Violation:

C3:
Missing operator attribution.

C4:
Missing audit log.

Required Action:
Update cancellation workflow.
```

AI 根据反馈修改：

```
async function cancelOrder(
  orderId: string,
  operatorId: string
) {
  return database.transaction(async (tx) => {
    const order = await tx.orders.findById(orderId);

    if (!order) {
      throw new Error("Order not found");
    }

    if (order.status === "PAID") {
      throw new Error("Paid order cannot be cancelled");
    }

    await tx.orders.update(orderId, {
      status: "CANCELLED"
    });

    await tx.auditLogs.create({
      entityId: orderId,
      operatorId,
      action: "ORDER_CANCELLED"
    });
  });
}
```

再次执行验证。

需要注意，这仍然只是示意实现。真实项目还需要验证事务隔离、并发状态变更、审计失败时的回滚行为、权限校验以及实际数据库适配情况。

这个案例说明：

**Constraint Engine 不负责告诉 AI 如何写出最漂亮的代码，而是负责告诉 AI 哪些工程条件必须得到满足。**

# **第八章：本篇核心结论**

我们已经建立了 Constraint Engine 的三个基础理论：

| 核心模块 | 解决的问题 |
| -------------------- | ------------------- |
| Constraint Modeling | 如何把现实规则转换为机器可理解的模型？ |
| Constraint Graph | 如何理解规则之间的关联？ |
| Constraint Hierarchy | 如何确定规则的优先级和权威性？ |

它们共同构成了约束管理的基础。

但还有一个更困难的问题没有解决：

当两个合理的工程规则发生冲突时，AI 应该怎么办？

例如：

- 新架构要求修改 API。
- 兼容规则要求保留旧 API。
- 性能优化要求改变数据库结构。
- 数据约束要求保留历史结构。
- 用户要求快速交付。
- 安全策略要求增加额外验证。

这不是简单的代码生成问题，而是一个约束冲突求解问题。

# **下一篇：Constraint Conflict Resolution**

我建议下一篇重点研究：

**《mini-llm Constraint Engine 深度解析（中）：从 Constraint Conflict Detection、SAT/SMT Solving 到 Constraint Optimization 与 Resolution Strategy》**

重点展开：

1. Constraint Conflict：为什么现实工程规则会相互冲突？
2. Conflict Detection：如何自动发现约束冲突？
3. SAT Solver：如何判断一组逻辑约束是否存在可行解？
4. SMT Solver：如何处理整数、字符串、数据类型和复杂条件？
5. Constraint Optimization：如何在多个软约束之间寻找最优解？
6. Conflict Resolution：如何确定规则的处理顺序？
7. Human-in-the-loop：哪些冲突应该交给人类决策？
8. AI Self-Correction：AI 如何根据冲突分析结果重新规划代码？

**特别值得研究的是 SAT/SMT Solver：它们能够为 AI 编程增加传统 LLM 不具备的形式化约束求解能力。**

这将是 Constraint Engine 从规则管理系统走向真正可计算工程系统的关键一步。
