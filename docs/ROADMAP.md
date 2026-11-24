# mini-llm 开发路线图

## Milestone 0 — 可运行骨架（已完成）

- [x] Cargo workspace
- [x] CPU Tensor
- [x] Add / MatMul / ReLU / Softmax
- [x] Scaled dot-product attention
- [x] CLI demo
- [x] 基础测试
- [x] PRD / 架构文档

## Milestone 1 — Autograd（已完成）

- [x] Tensor 节点与计算图
- [x] backward 机制
- [x] Add / Mul / MatMul 梯度
- [x] ReLU / Softmax / CrossEntropy 梯度
- [x] finite-difference 梯度检查

实现位于 `crates/mini-llm-autograd`：

- `TensorValue` 包装计算图节点，记录 op 名称、父节点和 backward 闭包；
- `backward()` 先做拓扑排序，再按逆序累积梯度，共享子表达式会正确相加；
- 额外支持 `sub` / `transpose` / `scale` / `add_scalar` / `powf` / `sum` /
  `mean` / `mse`，以及 NumPy 风格的右对齐广播（梯度通过 `sum_to` 还原）；
- `crates/mini-llm-autograd/tests/gradients.rs` 用中心差分校验每个算子；
- 演示命令：`cargo run -p mini-llm-cli -- autograd`（计算图 + 梯度）和
  `cargo run -p mini-llm-cli -- train-line`（手写 SGD 拟合直线）。

当前限制：仅 `f32`、单线程、无原地更新、无 `no_grad`、无高阶梯度。

## Milestone 2 — Neural Network Basics（已完成）

- [x] Linear
- [x] Embedding
- [x] LayerNorm
- [x] SGD / AdamW
- [x] XOR 与小型分类任务

实现位于 `crates/mini-llm-nn`：

- `Linear`（`y = xW + b`，Xavier 初始化，可选 bias）、`Embedding`（行 gather，
  重复 token 的梯度会累加）、`LayerNorm`（由 autograd 原语组合而成，反向传播
  自动推导）、`Relu` / `Tanh` / `Sigmoid`、`Sequential` 容器；
- `Sgd`（可选 momentum）与 `AdamW`（解耦权重衰减 + 偏差修正），通过
  `TensorValue::set_value` 原地更新参数；
- 确定性 `Rng`（xorshift64*），相同 seed 得到相同的初始化；
- 测试覆盖形状、LayerNorm 归一化性质、Embedding 梯度散射、SGD/AdamW 数值、
  XOR 端到端收敛；
- 演示命令：`cargo run -p mini-llm-cli -- xor` 与
  `cargo run -p mini-llm-cli -- classify`。

## Milestone 3 — Tiny GPT（已完成）

- [x] 字符级 Tokenizer
- [x] 数据批处理
- [x] Causal Attention
- [x] Transformer Decoder
- [x] Language Modeling Loss
- [x] 训练循环与 checkpoint
- [x] 文本生成

实现拆分：

- `mini-llm-data`：`CharTokenizer`（按码点排序、确定性的字符表）与
  `TextDataset`（随机窗口采样，target 右移一位，显式 `batch_at` 便于测试）；
- `mini-llm-model`：`CausalSelfAttention`（按 head 切片，块对角 causal mask）、
  `Mlp`、`TransformerBlock`（pre-norm + 残差）、`TinyGpt`
  （token/position embedding → blocks → final norm → lm head）；
- `mini-llm-train`：`Trainer`（采样、前向、反向、AdamW 更新、评估）与
  自描述的二进制 checkpoint（配置 + 字符表 + 全部参数）；
- 命令行：`train [steps]` 在 420 字符的内置语料上训练约 6 万参数的模型，
  `generate [prompt] [tokens] [temp]` 加载 checkpoint 采样；
- 测试覆盖：causal mask 不泄漏未来 token、不同样本不互相注意、梯度到达全部参数、
  训练可复现且 loss 下降、checkpoint 往返与损坏检测。

为了让纯标量 kernel 在 debug 构建下可用，workspace 的 `[profile.dev]` 与
`[profile.test]` 设为 `opt-level = 2`。

## Milestone 4 — Modern LLM（已完成）

- [x] RoPE
- [x] RMSNorm
- [x] SwiGLU
- [x] GQA
- [x] KV Cache
- [x] MoE

实现要点：

- `mini-llm-tensor`：`rope_rotate(head_dim, positions, base)` 按 head 内
  `(2i, 2i+1)` 配对旋转，负位置即逆旋转；`mini-llm-autograd` 的 `rope`
  正是用它做反向传播，梯度已通过有限差分校验。
- `mini-llm-nn`：`RmsNorm`（无均值中心、无 bias）、`SwiGlu`
  （`W2(SiLU(W1x) ⊙ W3x)`）、`Moe`（router + 多个 SwiGLU 专家，top-k 门控；
  当前为稠密执行，便于观察路由概率与专家占用）。
- `mini-llm-model`：`ModelConfig` 新增 `num_kv_heads`、`norm`、`position`、
  `feed_forward`、`num_experts`；`ModelConfig::modern()` 组合
  RMSNorm + RoPE + SwiGLU + GQA。
- KV Cache：`AttentionCache` / `BlockCache` 保存每层每头的历史 K/V，
  `generate()` 每步只前向一个 token；`generate_uncached()` 保留为对照，
  测试断言两者贪心解码完全一致。
- checkpoint 升级为 v2（10 个配置字段），v1 文件仍可读取并回退到经典架构。

已知限制：MoE 为稠密执行（所有专家都计算）；RoPE 超出训练上下文后质量下降；
缓存只支持 batch=1 的生成。

## Milestone 5 — Training Recipes（下一步）

- [ ] Pretraining
- [ ] SFT / Loss Masking
- [ ] LoRA
- [ ] DPO
- [ ] GRPO toy environment

## Milestone 6 — Accelerated & Observable

- [ ] Candle backend
- [ ] CPU/GPU parity tests
- [ ] Training event protocol
- [ ] Attention / activation / logits capture
- [ ] React Research Studio

## Definition of Done

每个 milestone 至少需要：
1. 可运行示例。
2. 单元测试或集成测试。
3. 文档说明算法和张量形状。
4. 明确列出当前限制。
5. 可复现的命令行运行方式。
