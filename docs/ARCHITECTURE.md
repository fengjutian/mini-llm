# mini-llm 系统架构

## 1. 架构原则

- Rust-first，核心功能不依赖 Python。
- CPU 教学实现作为数值参考实现。
- 加速后端可替换，不把模型逻辑绑定到 Candle。
- 模型、训练器、数据集、后端分层。
- Inspector 通过显式 Trace / Hook / Event 获取数据，避免默认复制所有中间张量。
- 小步演进，先保证可测试，再追求性能。

## 2. 逻辑架构

```text
CLI / Future Research Studio
            |
     Application Layer
            |
  Experiment / Training / Inference
            |
       Model Interface
            |
  Tensor & Operator Abstraction
       /             \
 Educational CPU   Accelerated Backend
       |                 |
   Vec<f32>            Candle
            |
     Device / Memory
```

## 3. Workspace crate 责任

| Crate | 责任 | 不应负责 |
|---|---|---|
| mini-llm-tensor | CPU Tensor 与基础算子 | 训练任务管理 |
| mini-llm-autograd | 计算图、反向传播、梯度累积 | 优化器状态、模型结构 |
| mini-llm-nn | Linear / Embedding / LayerNorm / 优化器 | 具体任务与数据加载 |
| mini-llm-data | 字符 tokenizer、窗口批采样 | 模型结构与训练 |
| mini-llm-train | 训练循环、checkpoint 序列化 | 模型定义、UI |
| mini-llm-model | 模型配置、Attention 等模型原语 | 文件持久化、UI |
| mini-llm-training（planned） | 训练循环与优化器 | GPU 驱动细节 |
| mini-llm-data（planned） | Dataset、Batch、Tokenizer 适配 | 模型结构 |
| mini-llm-backend（planned） | CPU/Candle 后端抽象 | 实验业务规则 |
| mini-llm-inspector（planned） | Trace、Hook、张量快照 | 修改模型权重 |
| mini-llm-cli | 命令行应用 | 核心算法实现 |

## 4. Tensor 设计

MVP Tensor 使用连续 row-major `Vec<f32>` 存储。

不变量：

- `shape` 的维度乘积必须等于 `data.len()`。
- 标量可用空 shape 表示（后续确定是否支持）。
- MVP 首先支持二维矩阵运算。
- 非法维度通过 `Result` 返回错误。
- 逐元素算子采用 NumPy 风格的右对齐广播：维度相等或其中一个为 1 时兼容，
  否则返回 `ShapeMismatch`；反向传播用 `sum_to` 把梯度归约回原形状。

Autograd 只依赖 tensor：节点保存 `value`、`requires_grad`、可选 `grad`、
父节点以及一个 `Fn(&Tensor) -> Vec<Tensor>` 的 backward 闭包。图按需构建，
没有全局 tape；`backward()` 每次做拓扑排序，因此共享子表达式会正确累积梯度。

## 5. Attention 数据约定

MVP 采用单头 scaled dot-product attention：

- Q: `[query_len, head_dim]`
- K: `[key_len, head_dim]`
- V: `[key_len, value_dim]`
- Scores: `Q × Kᵀ / sqrt(head_dim)`
- Weights: 对 Scores 每行做 Softmax
- Output: `Weights × V`

当前示例不包含 causal mask、multi-head reshape、dropout 或 KV cache。这些属于后续 Transformer 里程碑。

## 6. 训练与可观测性设计

未来训练循环应发布结构化事件，例如：

- `RunStarted`
- `StepStarted`
- `MetricsReported`
- `TensorCaptured`
- `CheckpointSaved`
- `RunCompleted`
- `RunFailed`

事件中应包含 run_id、step、timestamp、metric name/value、模型层路径等元数据。大张量不应直接通过 WebSocket 高频传输；应先落盘或转成采样、降维后的可视化数据。

## 7. 持久化约定

- 模型权重：独立二进制文件。
- 模型配置：TOML / JSON。
- 实验元数据：SQLite（后续）。
- 数据集：用户指定路径，记录 hash 和来源。
- Checkpoint：权重、优化器状态、step、随机状态和配置版本。

## 8. 依赖方向

`cli -> model -> tensor`，`cli -> autograd -> tensor`，`nn -> autograd -> tensor`，
`model -> nn -> autograd -> tensor`，`data -> nn`（仅复用 `Rng`），
`train -> model + data + nn + autograd`，`cli -> train`

后续建议：

- `training -> model + nn + autograd + backend + data`
- `inspector -> trace protocol`
- `backend -> tensor abstraction`
- `model` 不应反向依赖 `cli` 或 UI。
