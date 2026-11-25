# mini-llm

**A Rust-native learning and research platform for understanding, training, inspecting, and experimenting with small language models.**

mini-llm is inspired by the educational approach of MiniMind, but is designed as an independent Rust-first project. The initial goal is not to compete with production LLM frameworks. It is to make the mechanisms behind modern language models understandable, executable, measurable, and extensible.

## Project principles

1. **Learn from first principles:** begin with transparent CPU tensor operations and build toward neural networks.
2. **Separate learning from acceleration:** keep a small educational CPU backend while adding an accelerated backend later.
3. **Make experiments reproducible:** configurations, seeds, datasets, metrics, and checkpoints should be traceable.
4. **Make internals observable:** expose tensors, activations, attention maps, logits, gradients, and training events.
5. **Keep architecture modular:** model definitions must not be tightly coupled to a specific compute backend.

## Current status

This repository is a working learning stack, not a production framework:

- Rust Cargo workspace
- CPU tensor foundation with shape validation, broadcasting, addition, matrix multiplication, ReLU, and softmax
- `mini-llm-autograd`: a reverse-mode autodiff engine (computation graph, reverse topological backward pass, gradient accumulation)
- `mini-llm-nn`: `Linear`, `Embedding`, `LayerNorm`, `RmsNorm`, `SwiGlu`, `Moe`, activations, `Sequential`, SGD and AdamW
- `mini-llm-random`: seeded xorshift64\* RNG and Xavier/normal initialization
- `mini-llm-model`: causal multi-head / grouped-query attention with optional RoPE, a configurable pre-norm block, KV-cache generation, and `TinyGpt`
- `mini-llm-data`: character and byte-level BPE tokenizers, plus next-token window sampling
- `mini-llm-train`: training loop and a self-describing binary checkpoint format
- `mini-llm-trace`: structured training events (JSONL) and tensor capture
- `mini-llm-backend`: `Backend` seam with a CPU reference and an optional Candle implementation
- CLI demos: hand-checked gradients, linear regression, XOR, a 2D classifier, MoE routing, and Tiny GPT training plus text generation
- Astro site in `web/`: three interactive concept lessons plus the repository docs
- Product requirements and architecture documents

The tensor implementation is deliberately simple: `f32` plus an `Arc`-backed
`Vec<f32>`, scalar loops, with large matrix products split across cores. It is an
educational baseline, not a high-performance library. GPU acceleration and
higher-order gradients remain future work.

## Quick start

Requirements: Rust stable toolchain.

```bash
cargo test
cargo run -p mini-llm-cli -- demo
cargo run -p mini-llm-cli -- inspect-config
cargo run -p mini-llm-cli -- autograd    # graph + gradients for y = wx + b
cargo run -p mini-llm-cli -- train-line  # fit y = 2x - 1 with manual SGD
cargo run -p mini-llm-cli -- xor         # Linear + Tanh MLP trained with AdamW
cargo run -p mini-llm-cli -- classify    # softmax + cross-entropy on two blobs
cargo run -p mini-llm-cli -- moe         # mixture-of-experts routing
cargo run -p mini-llm-cli -- train       # modern Tiny GPT (RMSNorm/RoPE/SwiGLU/GQA)
cargo run -p mini-llm-cli -- train 200 --corpus examples/corpus.txt   # train on a text file
cargo run -p mini-llm-cli -- train 200 --dropout 0.1   # residual dropout
cargo run -p mini-llm-cli -- train 200 --early-stop 3  # keep the best checkpoint
cargo run -p mini-llm-cli -- train 200 --hidden 64 --layers 3 --block 64
cargo run -p mini-llm-cli -- train 200 --f16        # half-precision checkpoint
cargo run -p mini-llm-cli -- generate "The" 20 0.8 --seed 7
cargo run -p mini-llm-cli -- train 200 --bpe --vocab 300 --corpus examples/corpus.txt
cargo run -p mini-llm-cli -- train 200 --resume   # continue from the saved step
cargo run -p mini-llm-cli -- eval        # validation loss of the saved checkpoint
cargo run -p mini-llm-cli -- generate "The little language model learns " 14 0.3
cargo run -p mini-llm-cli -- generate "The " 20 0.8 --top-k 20 --top-p 0.9
cargo run -p mini-llm-cli -- sft         # supervised fine-tuning with prompt masking
cargo run -p mini-llm-cli -- sft 200 --jsonl examples/instructions.jsonl
cargo run -p mini-llm-cli -- lora        # adapt a frozen classifier with low-rank adapters
cargo run -p mini-llm-cli -- dpo         # preference optimization on a toy policy
cargo run -p mini-llm-cli -- grpo        # group relative policy optimization bandit
cargo run -p mini-llm-cli -- trace       # training events + attention capture (JSONL)
cargo run --release -p mini-llm-cli -- bench  # kernel + training micro-benchmarks
```

`train` writes `checkpoints/tiny-gpt.bin`; `generate` loads it and samples with
the KV cache. A full 800-step run takes roughly 20 seconds on a laptop CPU.

The training loop includes what a real run needs:

- **Warmup + cosine schedule** (`LrSchedule::WarmupCosine`), printed per step
  together with the current learning rate.
- **Global gradient-norm clipping** (`Trainer::with_grad_clip`); the reported
  `|g|` is post-clip.
- **Held-out validation loss** — the corpus is split 90/10, validation is
  evaluated every 50 steps and once at the end, and `eval` reproduces the final
  number from the checkpoint.
- **Exact resumption** — the version-3 checkpoint also stores the step counter,
  RNG state and Adam first/second moments, so `--resume` continues the same run
  instead of restarting the optimizer. A test asserts that a resumed step is
  bit-identical to the uninterrupted one.
- **Gradient accumulation** — `--accum N` (`train` and `sft`) averages N
  micro-batches before one optimizer update, trading time for a larger effective
  batch (`batch_size × N`); a test checks that accumulating two identical batches
  matches a plain single-batch step.
- **Model shape and seed flags** — `--hidden`, `--layers`, `--heads`,
  `--kv-heads`, `--block` and `--seed` override the demo configuration; an
  invalid shape exits with the configuration error instead of panicking.
- **Best checkpoint and early stopping** — validation loss is evaluated every
  50 steps; with `--early-stop N` the checkpoint is rewritten on every
  improvement and training stops after `N` evaluations without one, leaving the
  best model on disk instead of the last one.
- **Any UTF-8 text corpus** — `--corpus PATH` builds the vocabulary from that
  file instead of the built-in paragraph; the checkpoint records the vocabulary,
  so `generate` works from it and `eval --corpus PATH` reproduces the held-out
  number. `--corpus` can be repeated or point at a directory (its `.txt`/`.md`
  files are concatenated in sorted order). `examples/corpus.txt` is a small
  pangram corpus to try.
- **Character or byte-level BPE tokenizer** — `--bpe --vocab N` learns `N` byte
  merges from the corpus (default 512) instead of one token per character. BPE
  covers every byte, so unknown characters never fail; the checkpoint tags which
  tokenizer it used (format version 4) and `generate`/`eval` decode accordingly.
  Encoding keeps the sequence in a linked list and candidates in a min-heap, so
  it is O(n log n) rather than rescanning per merge.

## Autograd engine

`crates/mini-llm-autograd` implements reverse-mode differentiation on top of the
tensor engine. Every operation records its inputs and a closure that maps the
upstream gradient to the gradients of those inputs; `backward()` walks the graph
in reverse topological order and accumulates gradients into the leaves that
require them.

```rust
use mini_llm_autograd::TensorValue;

let x = TensorValue::constant(vec![2.0], vec![1, 1]).unwrap();
let w = TensorValue::parameter(vec![3.0], vec![1, 1]).unwrap();
let b = TensorValue::parameter(vec![1.0], vec![1]).unwrap();
let target = TensorValue::constant(vec![10.0], vec![1]).unwrap();

let y = x.matmul(&w).unwrap().add(&b).unwrap();
let loss = y.sub(&target).unwrap().powf(2.0).unwrap().mean().unwrap();

loss.backward().unwrap();
assert!((w.grad().unwrap().data()[0] + 12.0).abs() < 1e-4); // dL/dw = -12
assert!((b.grad().unwrap().data()[0] + 6.0).abs() < 1e-4);  // dL/db = -6
```

Supported operations: `add`, `sub`, `mul`, `matmul`, `transpose`, `relu`, `tanh`,
`sigmoid`, `softmax_rows`, `scale`, `add_scalar`, `powf`, `sum`, `mean`,
`mean_rows_keepdim`, `gather_rows`, `scatter_add_rows`, `mse`, and
`cross_entropy`. Binary operations follow NumPy-style right-aligned
broadcasting, and gradients are summed back to the original shape. Every analytic
gradient is verified against central finite differences in
`crates/mini-llm-autograd/tests/gradients.rs`.

Inference can skip graph construction:

```rust
use mini_llm_autograd::{no_grad, TensorValue};

let x = TensorValue::constant(vec![1.0, 2.0], vec![1, 2]).unwrap();
let y = no_grad::with_no_grad(|| x.scale(2.0)).unwrap();
assert!(!y.requires_grad());
```

The flag is thread-local, nestable and restored on drop; `TinyGpt::generate*`
uses it internally, so sampling no longer allocates graph nodes.

Known limitations: `f32` only, no higher-order gradients, and only `matmul` is
parallel (element-wise ops stay on the calling thread). `no_grad::with_no_grad`
provides a scoped inference mode, which generation uses internally.

## Neural network building blocks

`crates/mini-llm-nn` adds the pieces needed to train small models. Parameters are
`TensorValue` handles updated in place, so modules only need `&self` in `forward`
and optimizers can mutate them between steps.

```rust
use mini_llm_nn::{AdamW, Linear, Module, Optimizer, Rng, Sequential, Tanh};

let mut rng = Rng::new(42); // deterministic initialization
let model = Sequential::new(vec![
    Box::new(Linear::new(2, 8, &mut rng).unwrap()),
    Box::new(Tanh),
    Box::new(Linear::new(8, 1, &mut rng).unwrap()),
]);

let parameters = model.parameters();
let mut optimizer = AdamW::new(0.02);

// forward -> loss -> backward -> step -> zero_grad
```

| Piece | Notes |
| --- | --- |
| `Linear` | `y = xW + b`, Xavier-uniform weights, optional bias |
| `Embedding` | row gather; repeated tokens accumulate gradient in the table |
| `LayerNorm` / `RmsNorm` | composed from autograd primitives, so backprop is automatic |
| `SwiGlu` | `W2(SiLU(W1x) ⊙ W3x)` gated feed-forward |
| `Moe` | top-k router over SwiGLU experts (sparse dispatch) |
| `LoraLinear` | frozen base + trainable low-rank `A`/`B` adapters |
| `Relu` / `Tanh` / `Sigmoid` | stateless activation modules |
| `Sequential` | chains modules and concatenates their parameters |
| `Dropout` | inverted dropout, disabled in `mode::with_eval` |
| `Sgd` / `AdamW` | momentum SGD and AdamW with decoupled weight decay |

`Rng` is a seeded xorshift64\* generator, so the same seed reproduces the same
initialization. Normalization, embedding lookups, SwiGLU shapes, MoE routing,
optimizer updates and XOR convergence are covered in
`crates/mini-llm-nn/tests/modules.rs`.

Known limitations: no fused kernels.

## Tiny GPT

`mini-llm-model` contains a decoder-only Transformer that trains end to end on a
CPU:

```text
tokens ──► token embedding ─┐
                            ├─► [+ position embedding] ─► block × N ─► norm ─► lm head ─► logits
positions ─► RoPE / embed ──┘
```

Each block is pre-norm with residual connections, and the parts are selectable
through `ModelConfig`:

```text
x = x + Attn(Norm(x))          # Norm: LayerNorm | RMSNorm
x = x + FFN(Norm(x))           # FFN: ReLU-MLP | SwiGLU | top-k MoE
```

| Switch | Options | Default in `ModelConfig::modern()` |
| --- | --- | --- |
| `norm` | `LayerNorm`, `RmsNorm` | `RmsNorm` |
| `position` | `Learned`, `Rope` | `Rope` |
| `feed_forward` | `Relu`, `SwiGlu` | `SwiGlu` |
| `num_kv_heads` | any divisor of `num_heads` | half the query heads (GQA) |
| `num_experts` | `1` = dense, `>1` = MoE per block | `1` |
| `dropout` | residual-branch dropout probability, `0.0` disables | `0.0` |

Attention flattens the batch to `[batch·seq, hidden]` and applies a **block-diagonal
causal mask** of shape `[batch·seq, batch·seq]`, so a token can attend only to
earlier tokens of its own sample. With GQA several query heads share one
key/value head; the KV projections shrink accordingly.

Generation uses a per-layer KV cache: `generate()` feeds one token at a time and
appends to `AttentionCache` instead of recomputing the window.
`generate_uncached()` is kept as the reference path, and a test asserts that both
produce identical greedy tokens for both the classic and the modern stack.

The training stack:

| Crate | Role |
| --- | --- |
| `mini-llm-data` | `CharTokenizer` (sorted, deterministic vocabulary), `BpeTokenizer` (byte-level merges) and `TextDataset` (random windows, targets shifted by one) |
| `mini-llm-train` | `Trainer` (sample → forward → cross-entropy → backward → AdamW), LR schedule, gradient clipping, validation loss, checkpoint save/load with resumable state |

Checkpoints are a dependency-free little-endian format (version 5 adds the
parameter dtype, version 4 the tokenizer tag, version 3 the training state;
older versions still load):

```text
magic "MINILLM1" | version u32 | dtype u32 (0 = f32, 1 = f16) | 10 config u32s
tokenizer_kind u32 (0 = char, 1 = BPE)
  char: vocab_size u32 | vocab_size × (char as u32)
  bpe:  merge_count u32 | merge_count × (left u32, right u32)
parameter_count u32 | parameter_count × { rank u32 | dims | data (f32 or f16) }
has_state u32 (1 = the training state follows)
  step u32 | rng_state u64 | optimizer_step u32
  first_moments, second_moments: count u32 | count × { len u32 | f32 data }
```

`train --f16` writes f16 parameters, which roughly halves the weight bytes;
training and inference still compute in f32, and loading dequantizes back to
f32 (so a resumed f16 run sees slightly rounded weights).

A 420-character corpus, 31-character vocabulary, 2 layers, 4 heads (2 KV heads),
hidden size 32 and a 48-token context give a ~34k-parameter model. After 800
steps (about 20 s) it continues the corpus verbatim inside its context:

```text
prompt: The little language model learns
output: The little language model learns one character
```

Known limitations: MoE dispatch is sparse (only experts with assigned rows run;
orward_dense keeps a dense reference for parity tests) and RoPE generation
degrades past the trained context. Sampling supports
temperature, top-k and top-p (`Sampling`, exposed as `generate --top-k/--top-p`).
`generate_batch_cached` keeps one KV-cache row per sequence and updates them in
a single batched step. With RoPE it accepts unequal prompt lengths by
right-padding and masking the padding out of attention; learned position
embeddings require equal lengths. `generate_batch` handles unequal lengths
without a cache.

> Note: because the kernels are scalar loops, the workspace sets
> `[profile.dev] opt-level = 2`. Without it the Tiny GPT demo is roughly an order
> of magnitude slower.

## Training recipes

Milestone 5 adds the supervised and preference-optimization recipes on top of the
same autograd engine.

| Recipe | Building blocks | Demo output |
| --- | --- | --- |
| Pretraining | `TextDataset` + `cross_entropy` | Tiny GPT (see above) |
| SFT / loss masking | `Dataset`, `Batch.weights`, `cross_entropy_weighted`, `SftDataset` | prompt masked, losses fall to ~0.002 |
| LoRA | `LoraLinear` (frozen base, rank-4 adapters) | 54% → 100% on the new task, base weights unchanged |
| DPO | `sequence_log_prob`, `log_sigmoid`, `dpo_loss` | margin −1.0 → +9.7, chosen prob 0.99 |
| GRPO | `grpo_loss` (group-normalized advantage) | P(target arm) 0.15 → 0.998 |

Loss masking is a weighted cross-entropy: a weight of `0.0` removes a position
from both the loss and its gradient, and the loss is normalized by the weight
sum. `batch.weights` carries the mask through the trainer, so SFT and
pretraining share one training loop.

```rust
use mini_llm_data::{Tokenizer, SftDataset};
use mini_llm_model::{ModelConfig, TinyGpt};
use mini_llm_nn::Rng;
use mini_llm_train::{TrainConfig, Trainer};

let tokenizer = Tokenizer::from(CharTokenizer::from_text(corpus)); // or a BpeTokenizer
let (tokens, weights) = tokenizer.encode_supervised(prompt, response)?;
let dataset = SftDataset::new(tokens, weights, block_size)?;
let mut trainer = Trainer::new(model, dataset, TrainConfig::default());
trainer.train(400, |_| {})?;
```

`Tokenizer::encode_supervised` works for both vocabularies, so SFT can run with a
BPE checkpoint; `cargo run -p mini-llm-cli -- sft --bpe` exercises that path.

SFT can train on instruction data instead of the built-in dialogue:

```bash
cargo run -p mini-llm-cli -- sft 200 --jsonl examples/instructions.jsonl
```

Each line is a flat `{"prompt": "...", "response": "..."}` object (comments and
blank lines are skipped, unknown string fields are ignored). The parser is in
`mini-llm-data` and has no JSON dependency; malformed lines fail with the line
number.

`LoraLinear` starts as the identity update (`B = 0`) and reports only `A` and
`B` from `parameters()`, so an optimizer that receives `model.parameters()`
leaves the base weights untouched — the LoRA demo asserts this explicitly.

## Observability and backends

`mini-llm-trace` defines a small event protocol; `Trainer::train_observed` emits
`RunStarted`, one `Step` per update, and `RunCompleted` (or `RunFailed`) to any
`EventSink`. `JsonlSink` writes one JSON object per line, which is enough for a
UI to tail:

```json
{"run":"run-1795482837097","event":"step","step":1,"loss":4.017231,"elapsed_ms":3.620,"t_ms":4.038}
```

`TinyGpt::forward_traced` returns the logits plus one `AttentionRecord` per
layer and head, so attention maps can be captured after training:

```rust
let (logits, records) = model.forward_traced(&tokens, 1)?;
for record in &records {
    // record.layer, record.head, record.weights.shape() == [seq, seq]
}
```

The `trace` command trains for 30 steps, captures 8 attention matrices and the
logits of one batch, saves a checkpoint, and writes everything to
`traces/<run_id>.jsonl`:

```text
model: 19464 params | 30 steps | sink: traces/run-1795482837097.jsonl
captured 8 attention matrices (seq 16) and logits 16x31
layer 0 head 0: mean max attention 0.424
wrote 42 events to traces/run-1795482837097.jsonl
```

For acceleration, `mini-llm-backend` defines a `Backend` seam. `CpuBackend`
always works; `CandleBackend` is behind a feature flag and is compared against the
reference kernels by `tests/parity.rs`. The trait covers matrix multiplication,
row-wise softmax and RMS normalization; backends that do not override a kernel
fall back to the CPU implementation.

```bash
cargo test -p mini-llm-backend               # CPU reference checks
cargo test -p mini-llm-backend --features candle   # CPU vs Candle (tolerance 1e-4)
```

The parity suite currently compares the in-repo CPU kernels with Candle's CPU
device; `Device::Cuda` can be passed to `CandleBackend::new` where a GPU is
available. Candle 0.11 keeps softmax in `candle-nn`, so the backend composes it
from `max_keepdim` / `exp` / `sum_keepdim` rather than pulling in that crate.

Autograd ops can be routed through a backend for a scoped region:

```rust
use std::sync::Arc;
use mini_llm_autograd::backend::with_backend;
use mini_llm_backend::CandleBackend;

let logits = with_backend(Arc::new(CandleBackend::cpu()), || {
    model.forward(&tokens, 1)
})?;
```

The selection is thread-local and restored when the closure returns (even on
panic), and the default remains the CPU reference. Both the forward pass and the
`matmul` backward closures call through the backend, so wrapping a whole
`forward + backward` step accelerates both directions; softmax is forward-only.

From the CLI, build once with the feature and pass the flag:

```bash
cargo run --features candle -p mini-llm-cli -- train 200 --backend candle
cargo run --features candle -p mini-llm-cli -- generate "the " 64 --backend candle
```

Without the feature, `--backend candle` exits with a message telling you to
rebuild; `--backend cpu` (the default) always works. Candle on the CPU device is
a correctness/smoke path (it hops through a host copy per kernel); the speedup
comes from `CandleBackend::new(Device::Cuda)` where a GPU is available.

## Performance notes

`cargo run --release -p mini-llm-cli -- bench` prints the numbers this project is
optimized against. Timings are the best of three rounds of ten iterations, which
removes most of the background-load noise on a laptop:

```text
matmul  64x64 x64 :     0.02 ms  27754.8 MFLOP/s
matmul 128x128x128:     0.21 ms  19795.7 MFLOP/s
matmul 256x256x256:     0.36 ms  92943.4 MFLOP/s
forward (hidden  32, block 16, batch 4):     2.03 ms   1.6 MB alloc
forward (hidden 128, block 32, batch 4):    16.14 ms  10.0 MB alloc
forward (hidden 256, block 32, batch 4):    32.29 ms  18.1 MB alloc
forward (hidden 512, block 32, batch 4):    71.48 ms  35.0 MB alloc
train step (hidden  32, block 16, batch 2):  2.84 ms   2.0 MB alloc
train step (hidden 128, block 32, batch 2): 26.00 ms  16.6 MB alloc
train step (hidden 256, block 32, batch 2): 80.23 ms  41.9 MB alloc
generate 32 tokens (KV cache):                5.27 ms
clone 1024x1024 (4 MB shared buffer):        0.087 us
```

Even with best-of-three, values still move with background load: `matmul 256³`
has measured 0.36–0.45 ms and the 512-wide forward 53–72 ms on the same machine.
Compare a change against the previous run on the same machine rather than across
machines.

Design choices behind those numbers:

- **Large matmuls run row-parallel on a rayon pool.** Products above ~1M
  multiply-adds are split into row chunks that rayon schedules on a persistent
  worker pool, so a matmul pays no thread-spawn cost. The chunk count is four per
  thread and the thread count is capped by `work / 2M`, so medium matrices do not
  oversubscribe. Each output element keeps its original accumulation order, so
  results are bit-identical to the sequential kernel (covered by a test) and
  training stays reproducible. Below the threshold everything stays on the
  calling thread. Four-row register blocking means each row of `right` is loaded
  once and reused four times. An eight-row variant was tried and rejected: the
  extra accumulators spill and it showed no consistent gain.
- **Tensor buffers are `Arc<Vec<f32>>`.** Cloning a tensor is a refcount bump
  (~0.08 µs for 4 MB) instead of a copy. The autograd engine clones node values
  on every op, so this turns an O(size) cost into O(1).
- **KV cache appends in place.** `Tensor::append_row` grows the shared buffer with
  `Arc::make_mut`, so a decode step costs O(1) amortized instead of copying the
  whole cache. The GQA path also appends one row per key/value head and shares it
  across its query heads; previously it appended once per query head, doubling
  the cache and the attention work.
- **The causal mask is cached per `(batch, seq)`.** Training reuses the same
  shape every step, so the `[N, N]` mask is built once instead of per forward.
- **Optimizers update parameters in place.** `Trainer::step` drops the graph
  after `backward()` (that graph is what keeps parameter buffers shared), then
  `AdamW`/`Sgd` call `TensorValue::update_value`, which mutates the buffer via
  `Arc::make_mut`. The allocation counter in `bench` shows the effect
  (18.7 → 16.7 MB per step at hidden 128); wall-clock time at this scale is
  compute-bound, so the gain is not visible there.
- **matmul keeps i-k-j ordering.** A transposed dot-product variant was measured
  ~4× slower (3.7 vs 15.8 GFLOP/s) at these sizes because the inner loop stops
  vectorizing, so it is deliberately not used. The comment in
  `crates/mini-llm-tensor/src/matrix.rs` records the experiment.
- **`Trainer` caches parameter handles.** Collecting parameters walks the graph;
  doing it once at construction removes a per-step allocation.
- **Softmax reuses one scratch row** instead of allocating a `Vec` per row.
- Scaling the benchmark to hidden 256/512 shows allocation stays proportional to
  activations (35.5 MB for a 512-wide forward) while time grows faster
  (187 ms), i.e. the model is compute-bound rather than allocator-bound. That is
  why a per-step workspace buffer is not implemented yet: it would trim a
  minority of the allocation volume without changing the runtime. The numbers
  above are the baseline to revisit when a real model size is in play.
- The tiny demo model is dominated by per-op overhead and graph bookkeeping, not
  floating-point work, so micro-optimizing kernels further has diminishing
  returns until the model grows.

## Web site (Astro)

`web/` contains a static Astro site whose goal is to make the concepts behind the
code easier to grasp:

- `/learn/tensor-shape` — shape, strides and the linear offset of a tensor
- `/learn/softmax` — logits, temperature, entropy and the probability distribution
- `/learn/attention` — scores, row-wise softmax, weighted values and a causal-mask preview
- `/studio` — the Research Studio: load a JSONL trace and inspect the loss curve,
  every attention head and the final logits
- `/docs` — the Markdown files in `docs/`, rendered with LaTeX support
- `/docs` is also a searchable **knowledge map**: four knowledge tracks
  (implementation, cognition, planning/learning, agents) plus project design,
  numbered parts, per-document reading time, extracted knowledge points, and
  prev/next navigation

The site is a presentation layer only; it does not train or run the model. The
`docs/*.md` files remain the single source of truth: `web/scripts/sync-docs.mjs`
discovers them, adds the frontmatter (including series position and reading
time), strips the duplicated H1, and normalizes the escaped LaTeX dialect used in
those documents. New documents appear automatically.

The Research Studio is a React island (`client:load`) that consumes the trace
protocol. It ships with `web/src/data/sample-trace.json` (produced by
`cargo run -p mini-llm-cli -- trace`) and accepts any `.jsonl` written by the
same command via the file picker.

Requirements: Node.js 22+ and pnpm.

```bash
cd web
pnpm install
pnpm dev      # http://localhost:4321
pnpm build    # static output in web/dist
```

### Deploying to Netlify

The repository ships a [`netlify.toml`](netlify.toml) that points Netlify at the
`web/` base directory, runs `pnpm run build`, and publishes `web/dist` with
`NODE_VERSION=22`. Netlify's file configuration overrides the UI build settings,
so connect the repository and leave the build fields empty (or set Base directory
to `web`), then trigger a deploy.

For a one-off deploy without Git integration:

```bash
cd web
pnpm build
npx netlify deploy --prod --dir=dist
```

Or drag the generated `web/dist` folder onto <https://app.netlify.com/drop>.
`SITE_URL` is optional on Netlify: the build reads Netlify's own `URL` variable
for canonical and Open Graph tags.

## Workspace

```text
mini-llm/
├── Cargo.toml
├── docs/
│   ├── PRD.md
│   ├── ARCHITECTURE.md
│   └── ROADMAP.md
├── crates/
│   ├── mini-llm-tensor/   # CPU tensor: error, broadcast, tensor, matrix kernels
│   ├── mini-llm-random/   # deterministic RNG and initialization sampling
│   ├── mini-llm-autograd/ # graph, value and ops (arith / activation / shape)
│   ├── mini-llm-nn/       # one file per layer + module trait + optimizers
│   ├── mini-llm-data/     # char/BPE tokenizers and batch sampling
│   ├── mini-llm-model/    # config, attention, blocks, Tiny GPT
│   ├── mini-llm-train/    # training loop and checkpoints
│   ├── mini-llm-trace/    # event protocol, JSONL sink, tensor records
│   ├── mini-llm-backend/  # Backend seam + optional Candle backend
│   └── mini-llm-cli/      # dispatch in main.rs + one module per command
└── web/                  # Astro concept site and rendered docs
```

Dependencies point upwards only: `cli -> train -> model -> nn -> autograd -> tensor`,
while `data` depends on `random` (not on `nn`), and `random` only needs `tensor`.
`trace` is dependency-free, `train` emits events into it, and `backend` is a
standalone parity seam. Every crate's `lib.rs` is limited to module declarations
and re-exports.

## Compute backends

- `cpu-edu`: first-principles CPU implementation for learning and correctness.
- `candle` (feature-gated): the accelerated seam in `mini-llm-backend`; parity
  with the reference kernel is checked in `crates/mini-llm-backend/tests/parity.rs`.
- Future backends implement `Backend` and reuse the same parity harness.

## Continuous integration

`.github/workflows/ci.yml` runs three jobs on every push and pull request:

- `cargo fmt --all -- --check`, `cargo clippy --workspace --all-targets -- -D warnings`
  and `cargo test --workspace`;
- `cargo test -p mini-llm-backend --features candle` for the Candle parity suite;
- `pnpm install --frozen-lockfile && pnpm run build` for the Astro site.

The same commands can be run locally; `cargo test --workspace` is the quickest
gate before committing.

## Non-goals for v0.1

- Training frontier-scale models
- Replacing PyTorch, Candle, Burn, or llama.cpp
- Production distributed training
- A complete multimodal stack
- Claiming that next-token prediction alone proves a model has a grounded world model

## License

Apache-2.0. See `LICENSE`.
