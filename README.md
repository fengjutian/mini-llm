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

This repository is an initial scaffold:

- Rust Cargo workspace
- CPU tensor foundation with shape validation, broadcasting, addition, matrix multiplication, ReLU, and softmax
- `mini-llm-autograd`: a reverse-mode autodiff engine (computation graph, reverse topological backward pass, gradient accumulation)
- `mini-llm-nn`: `Linear`, `Embedding`, `LayerNorm`, activations, a `Sequential` container, SGD and AdamW
- Small model crate with configuration and a transparent scaled dot-product attention implementation
- CLI demos: hand-checked `y = wx + b` gradients, linear regression, XOR, and a 2D classifier
- Astro site in `web/`: three interactive concept lessons plus the repository docs
- Product requirements and architecture documents

The initial tensor implementation is deliberately simple and uses `f32` plus `Vec<f32>`. It is an educational baseline, not a high-performance tensor library. Tokenization, complete Transformer blocks, training infrastructure, GPU acceleration, and higher-order gradients are planned milestones, not yet implemented.

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
```

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
`mean_rows_keepdim`, `gather_rows`, `mse`, and `cross_entropy`. Binary operations
follow NumPy-style right-aligned broadcasting, and gradients are summed back to
the original shape. Every analytic gradient is verified against central finite
differences in `crates/mini-llm-autograd/tests/gradients.rs`.

Known limitations: `f32` only, single-threaded, no `no_grad` region, no
higher-order gradients.

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
| `LayerNorm` | composed from autograd primitives, so backprop is automatic |
| `Relu` / `Tanh` / `Sigmoid` | stateless activation modules |
| `Sequential` | chains modules and concatenates their parameters |
| `Sgd` / `AdamW` | momentum SGD and AdamW with decoupled weight decay |

`Rng` is a seeded xorshift64\* generator, so the same seed reproduces the same
initialization. Layer normalization, embedding lookups, optimizer updates and XOR
convergence are covered in `crates/mini-llm-nn/tests/modules.rs`.

Known limitations: no dropout, no fused kernels, no learning-rate schedule, no
save/load of parameters yet.

## Web site (Astro)

`web/` contains a static Astro site whose goal is to make the concepts behind the
code easier to grasp:

- `/learn/tensor-shape` — shape, strides and the linear offset of a tensor
- `/learn/softmax` — logits, temperature, entropy and the probability distribution
- `/learn/attention` — scores, row-wise softmax, weighted values and a causal-mask preview
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
│   ├── mini-llm-tensor/   # transparent CPU tensor operations
│   ├── mini-llm-autograd/ # reverse-mode autodiff over the tensor engine
│   ├── mini-llm-nn/       # layers, activations and optimizers
│   ├── mini-llm-model/    # model configuration and attention primitives
│   └── mini-llm-cli/      # command-line entry point
└── web/                  # Astro concept site and rendered docs
```

## Planned compute backends

- `cpu-edu`: first-principles CPU implementation for learning and correctness.
- `candle`: accelerated backend using Hugging Face Candle, introduced after the CPU path is tested.
- Future backends should implement the same backend-facing interfaces where practical.

## Non-goals for v0.1

- Training frontier-scale models
- Replacing PyTorch, Candle, Burn, or llama.cpp
- Production distributed training
- A complete multimodal stack
- Claiming that next-token prediction alone proves a model has a grounded world model

## License

Apache-2.0. See `LICENSE`.
