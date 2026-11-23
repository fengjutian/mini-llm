//! # mini-llm-autograd
//!
//! A small, readable reverse-mode automatic differentiation engine built on top
//! of `mini-llm-tensor`.
//!
//! Each operation records the tensors it consumed and a closure that turns the
//! upstream gradient into the gradients of those inputs. Calling
//! [`TensorValue::backward`] on a scalar loss walks the recorded graph in
//! reverse topological order and accumulates gradients into every leaf that
//! requires them.
//!
//! ```no_run
//! use mini_llm_autograd::TensorValue;
//! use mini_llm_tensor::Tensor;
//!
//! let x = TensorValue::constant(vec![1.0, 2.0, 3.0, 4.0], vec![2, 2]).unwrap();
//! let w = TensorValue::parameter(vec![0.5, -0.5, 0.5, -0.5], vec![2, 2]).unwrap();
//! let loss = x.matmul(&w).unwrap().mean().unwrap();
//! loss.backward().unwrap();
//!
//! assert!(w.grad().is_some());
//! ```
//!
//! ## Supported operations
//!
//! | Operation | Forward | Gradient |
//! | --- | --- | --- |
//! | `add` / `sub` | `a ± b` | `G`, `±G` |
//! | `mul` | `a ⊙ b` | `G ⊙ b`, `G ⊙ a` |
//! | `matmul` | `A B` | `G Bᵀ`, `Aᵀ G` |
//! | `transpose` | `Aᵀ` | `Gᵀ` |
//! | `relu` | `max(x, 0)` | `G ⊙ 1[x > 0]` |
//! | `tanh` / `sigmoid` | `tanh(x)`, `σ(x)` | `(1 - y²) ⊙ G`, `y(1 - y) ⊙ G` |
//! | `softmax_rows` | row-wise softmax | `s ⊙ (G - ⟨G, s⟩)` |
//! | `scale` / `add_scalar` / `powf` | `cx`, `x + c`, `x^p` | `cG`, `G`, `p x^{p-1} ⊙ G` |
//! | `sum` / `mean` | full reduction | broadcast of `G` |
//! | `mean_rows_keepdim` | row mean, `[r, 1]` | `G / cols` per row |
//! | `gather_rows` | row lookup | scatter-add into the source rows |
//! | `cross_entropy` | mean NLL | `(softmax - onehot) / batch` |
//!
//! ## Limitations
//!
//! - `f32` only, single-threaded, CPU only.
//! - Binary operations broadcast NumPy-style, but broadcasting is materialized
//!   in the forward pass rather than fused with the kernel.
//! - Parameters can be replaced in place with `set_value`, but there is no
//!   in-place tensor arithmetic and no `no_grad` context.
//! - No higher-order gradients.
//! - Graph nodes live as long as any [`TensorValue`] handle referencing them.

mod error;
mod value;

pub use error::AutogradError;
pub use value::TensorValue;
