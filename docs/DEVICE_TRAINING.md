# 设备驻留训练示例

`crates/mini-llm-backend/examples/device_attention_block.rs` 是一个完整的小型训练循环：
**参数、梯度、优化器状态都留在 `DeviceTensor` 上**，只有数据准备、dropout mask 生成和
日志输出使用普通 Rust。它的目的不是刷指标，而是验证设备抽象层（`DeviceTensor` / `DeviceValue` /
`DeviceAdamW` / DEVT checkpoint）能在真实训练里闭环。

## 它验证了什么

- 自定义反向图 `DeviceValue`：`matmul`/`add`/`sub`/`mul`/`relu`/`silu`/`mean`/`layer_norm`/
  `softmax`/`cross_entropy`/`rope` 都能在设备张量上前向与反传；
- 设备侧注意力：GQA + 因果掩码的前向与手工反向（Candle 存储当前委托 host 数学，见“限制”）；
- `DeviceAdamW`：一阶/二阶矩、偏差校正、解耦权重衰减都在设备上；
- 断点续训：checkpoint 保存参数 + 矩 + step + 超参 + 语料指纹，`--resume` 原样接续；
- 可复现：`--seed` 同时控制初始化与 dropout。

## 快速开始

```bash
# 默认 1 层、seq 8、batch 4、1600 步
cargo run -p mini-llm-backend --example device_attention_block

# 目前最好的配置（2 层，seq 64，batch 16，3000 步）
cargo run -p mini-llm-backend --example device_attention_block -- \
  --layers 2 --seq 64 --batch 16 --steps 3000

# 保存 checkpoint 后只做评估，或从它继续训练
cargo run -p mini-llm-backend --example device_attention_block -- \
  --layers 2 --seq 64 --batch 16 --steps 3000 --checkpoint target/run.bin
cargo run -p mini-llm-backend --example device_attention_block -- --load target/run.bin
cargo run -p mini-llm-backend --example device_attention_block -- \
  --layers 2 --seq 64 --batch 16 --steps 6000 --resume target/run.bin
```

CLI 入口（同一套训练逻辑）：

```bash
cargo run -p mini-llm-cli -- device-train --layers 2 --seq 64 --batch 16 --steps 3000

# 训练后采样
cargo run -p mini-llm-cli -- device-train --layers 1 --seq 32 --batch 8 --steps 600 \
  --sample 120 --prompt "The " --temperature 0.8 --top-k 20
cargo run -p mini-llm-cli --features candle -- device-train \
  --device cuda --layers 1 --seq 32 --batch 8 --steps 40
```

常用子命令：

| 参数 | 作用 |
| --- | --- |
| `--sweep` | weight decay × dropout 四组对照 |
| `--lr-sweep` | 0.005 / 0.01 / 0.02 / 0.04 学习率对照 |
| `--layer-sweep` | 1/2/3 层对照 |
| `--load PATH` | 加载 checkpoint 直接评估（自动恢复超参） |
| `--resume PATH` | 从 checkpoint 继续训练（含 Adam 矩与 step） |
| `--checkpoint PATH` | 最佳 checkpoint 输出位置（默认临时目录） |
| `--patience N` | 连续 N 次评估无改善则早停（0 关闭） |
| `--device cpu\|cuda[:N]` | 训练设备（CUDA 需 `candle` 特性） |
| `--sample N` | 训练后用最佳参数生成 N 个字符 |
| `--prompt TEXT` | 生成前缀 |
| `--temperature T` | 采样温度（0 = argmax） |
| `--top-k K` | 只从 top-k 采样（0 = 全部） |
| `--corpus PATH...` | 外部语料（可重复，ASCII 过滤） |
| `--corpus-chars N` | 读入字符上限（默认 600,000） |
| `--bpe` | 字节级 BPE（基础词表 256） |
| `--bpe-vocab N` | BPE 目标词表（默认 512，实际 ≥ 257） |
| `--checkpoint-format f32\|f16\|q8` | checkpoint 存储精度（默认 f32） |


## 公开语料

仓库不内置大数据集；`scripts/fetch-corpora.ps1`（Windows / PowerShell）与
`scripts/fetch-corpora.sh`（Linux / macOS）会把公开语料下载到 `datasets/`
（已在 `.gitignore` 中忽略）：

- TinyShakespeare（约 1 MB，GitHub raw）
- TinyStories valid split（约 22 MB，HuggingFace）
- Project Gutenberg / WikiText-2 也列在脚本里，但源站可能维护或重定向失败，
  脚本只打印警告。

```bash
./scripts/fetch-corpora.sh            # 或 powershell -File scripts/fetch-corpora.ps1
cargo run -p mini-llm-cli -- device-train \
  --corpus datasets/tinystories-valid.txt --corpus-chars 300000 \
  --layers 1 --seq 32 --batch 8 --steps 400 --sample 120
```

`--corpus` 可重复传入多个文件；`--corpus-chars N` 限制读入的字符数
（默认 600,000），避免大文件把训练拖慢。语料会被转成 ASCII 并做块切分，
checkpoint 里记录的 `corpus_hash` 也基于同样的内容。

实测（CPU，200 步）：

| 语料 | 字符 | vocab | held-out |
| --- | --- | --- | --- |
| TinyShakespeare | 200k | 62 | 2.61 |
| TinyStories valid | 300k | 66 | 2.39 |
| Shakespeare + BPE（117k 字符） | 117k | 257 | 2.65 |
## 结构

```text
ids → embedding(gather_rows)
    → N × [ LayerNorm → QKV → GQA attention(causal) → out-proj → residual
            LayerNorm → SwiGLU → down-proj → residual ]
    → final LayerNorm → LM head → cross_entropy
```

- 每一层是 `crates/mini-llm-backend/src/device.rs` 里的 `DeviceTensor` 算子组合；
- 反向由 `tape.rs` 的 `DeviceValue` 拓扑排序完成，注意力是融合算子（手工反向）；
- 优化器 `optim.rs` 的 `DeviceAdamW` 持有设备上的 `m`/`v`。

## 数据与切分

语料在运行时由三部分组成（只保留 ASCII，避免 CJK 把字符词表撑到上千）：

1. 仓库 `README.md`；
2. 所有 crate 的 `//!` / `///` 文档注释；
3. `crates/*/tests/*.rs` 源码与会话中 `web/src` 的 TS/Astro 源码。

当前约 **45.5 万字符、词表 127**。held-out 是**末尾 20% 的连续文本块**，训练与验证窗口
不重叠；这比随机窗口更严格，也避免了相邻窗口的上下文泄漏。

## 训练配置

- 优化器：`DeviceAdamW(lr=0.02, wd=0.01)`，默认 warmup + cosine（warmup = steps/10 上限 200，
  cosine 降到 0.1×lr）；
- 梯度：无裁剪（示例规模下未出现不稳定）；
- dropout：加在 attention 与 FFN 的残差分支，评估时关闭；
- 日志：训练 loss 用 EMA（0.98/0.02）平滑，每 25 步评估 held-out，并记录最佳点；
- 早停：`--patience N` 连续 N 次评估无改善即停止（默认 0=关闭）。

## 结果（2 层，seq 64，warmup + cosine）

| 配置 | 最终 held-out | 最佳 held-out |
| --- | --- | --- |
| batch 16, 3000 步 | 2.65 | 2.26 |
| batch 16, 3000 步, lr 0.01 | 2.56 | 2.25 |
| batch 32, 3000 步 | 2.18 | 2.18 |
| batch 16, 6000 步 | 2.10 | 2.10 |
| batch 32, 6000 步 | 2.04 | 2.03 |

其他结论：

- 块切分与随机窗口的结果一致（2.26），说明早前的数字没有明显泄漏；
- `--hidden 48` 反而更差（2.39 vs 2.26），加宽不如加 batch/步数；
- 3 层在该预算下劣于 2 层；warmup+cosine 显著优于常数学习率。

## Checkpoint（DEVT v3）

```text
magic "DEVT" | version 3 | metadata 长度 + UTF-8 元数据
              | 每个张量: 存储格式(u8) + rank/dims + 数据
              | 参数(N) | Adam 一阶矩(N) | 二阶矩(N) | step(1)
```

存储格式（`--checkpoint-format`）：

| 格式 | 每值字节 | 说明 |
| --- | --- | --- |
| `f32` | 4 | 默认，与训练一致 |
| `f16` | 2 | IEEE 754 半精度 |
| `q8` | 1 + 4（每张量一个 f32 scale） | 对称 int8，`x ≈ q·scale` |

加载时统一反量化为 f32，**训练/推理仍用 f32 计算**，所以只是存储压缩，
不是 f16 计算 kernel。实测（60k Shakespeare，1 层，80 步）：

| 格式 | 文件大小 | reload 后 held-out |
| --- | --- | --- |
| f32 | 163,080 B | 2.9578 |
| f16 | 82,084 B（50%） | 2.9578 |
| q8 | 41,866 B（26%） | 2.9580 |

元数据示例：

```text
layers=2 hidden=32 seq=64 batch=16 steps=6000 lr=0.02 wd=0.01
dropout=0 min_ratio=0.1 warmup=0 corpus_chars=455406
corpus_hash=09a13f86ca35b65c seed=7 bpe=0 bpe_vocab=512
```

加载行为：

- `--load` / `--resume` 会恢复 `layers/hidden/seq/batch/seed/bpe/bpe_vocab`，
  命令行不一致时以 checkpoint 为准；
- 语料哈希与当前语料不一致时打印警告（此时 held-out loss 不可跨语料比较）；
- `--resume` 从保存的 step 继续，学习率调度按绝对步数恢复。

## 限制

- 目前只在 `Device::Cpu` 上运行；`Candle(Cpu)` 存储上的 attention 走 host fallback（正确但有
  一次 host↔device 往返，实测前向 1.4×、反向 2.2×（host 注意力已按 batch 并行）），原生 4D kernel 仍未实现；
- 注意力是 O(seq²) 的 host 循环（按 batch 并行），seq 64 / batch 16 / 3000 步约 162 s、batch 32 约 266 s；
- 语料是仓库自身文本，指标只用于相对比较，不具通用性。

## 相关代码

- `crates/mini-llm-backend/src/device.rs`：`DeviceTensor` 算子与 attention；
- `crates/mini-llm-backend/src/tape.rs`：`DeviceValue` 反向图；
- `crates/mini-llm-backend/src/optim.rs`：`DeviceAdamW`；
- `crates/mini-llm-backend/src/device_io.rs`：DEVT checkpoint 读写。
