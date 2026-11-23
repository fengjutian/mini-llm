# mini-llm 开发路线图

## Milestone 0 — 可运行骨架（当前）

- [x] Cargo workspace
- [x] CPU Tensor
- [x] Add / MatMul / ReLU / Softmax
- [x] Scaled dot-product attention
- [x] CLI demo
- [x] 基础测试
- [x] PRD / 架构文档

## Milestone 1 — Autograd

- [ ] Tensor 节点与计算图
- [ ] backward 机制
- [ ] Add / Mul / MatMul 梯度
- [ ] ReLU / Softmax / CrossEntropy 梯度
- [ ] finite-difference 梯度检查

## Milestone 2 — Neural Network Basics

- [ ] Linear
- [ ] Embedding
- [ ] LayerNorm
- [ ] SGD / AdamW
- [ ] XOR 与小型分类任务

## Milestone 3 — Tiny GPT

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
