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

## Milestone 3 — Tiny GPT（下一步）

- [ ] 字符级 Tokenizer
- [ ] 数据批处理
- [ ] Causal Attention
- [ ] Transformer Decoder
- [ ] Language Modeling Loss
- [ ] 训练循环与 checkpoint
- [ ] 文本生成

## Milestone 4 — Modern LLM

- [ ] RoPE
- [ ] RMSNorm
- [ ] SwiGLU
- [ ] GQA
- [ ] KV Cache
- [ ] MoE

## Milestone 5 — Training Recipes

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
