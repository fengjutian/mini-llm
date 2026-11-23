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
- CPU tensor foundation with shape validation, addition, matrix multiplication, ReLU, and softmax
- Small model crate with configuration and a transparent scaled dot-product attention implementation
- CLI demo
- Astro site in `web/`: three interactive concept lessons plus the repository docs
- Product requirements and architecture documents

The initial tensor implementation is deliberately simple and uses `f32` plus `Vec<f32>`. It is an educational baseline, not a high-performance tensor library. Automatic differentiation, tokenization, complete Transformer blocks, training, and GPU acceleration are planned milestones, not yet implemented.

## Quick start

Requirements: Rust stable toolchain.

```bash
cargo test
cargo run -p mini-llm-cli -- demo
cargo run -p mini-llm-cli -- inspect-config
```

## Web site (Astro)

`web/` contains a static Astro site whose goal is to make the concepts behind the
code easier to grasp:

- `/learn/tensor-shape` — shape, strides and the linear offset of a tensor
- `/learn/softmax` — logits, temperature, entropy and the probability distribution
- `/learn/attention` — scores, row-wise softmax, weighted values and a causal-mask preview
- `/docs` — the Markdown files in `docs/`, rendered with LaTeX support
- `/docs` also presents the nine-part **engine series** (math, tensor, autograd,
  neural network, Transformer, block, decoder, training, evaluation) as a numbered
  reading path, with per-document reading time and prev/next navigation

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

## Workspace

```text
mini-llm/
├── Cargo.toml
├── docs/
│   ├── PRD.md
│   ├── ARCHITECTURE.md
│   └── ROADMAP.md
├── crates/
│   ├── mini-llm-tensor/  # transparent CPU tensor operations
│   ├── mini-llm-model/   # model configuration and attention primitives
│   └── mini-llm-cli/     # command-line entry point
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
