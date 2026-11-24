//! # mini-llm-nn
//!
//! Small, explicit neural-network building blocks on top of `mini-llm-autograd`:
//! [`Linear`], [`Embedding`], [`LayerNorm`], [`RmsNorm`], [`SwiGlu`], [`Moe`],
//! activations, a [`Sequential`] container and two optimizers ([`Sgd`],
//! [`AdamW`]).
//!
//! Everything is deterministic given a seed: parameter initialization draws from
//! [`Rng`], a tiny xorshift generator, so experiments can be reproduced exactly.
//!
//! ```no_run
//! use mini_llm_nn::{AdamW, Linear, Module, Optimizer, Rng, Sequential, Tanh};
//!
//! let mut rng = Rng::new(7);
//! let model = Sequential::new(vec![
//!     Box::new(Linear::new(2, 8, &mut rng).unwrap()),
//!     Box::new(Tanh::default()),
//!     Box::new(Linear::new(8, 1, &mut rng).unwrap()),
//! ]);
//! let parameters = model.parameters();
//! let mut optimizer = AdamW::new(0.03);
//! // ... build a loss with `model.forward(&input)`, run `backward()`, then:
//! optimizer.step(&parameters).unwrap();
//! optimizer.zero_grad(&parameters);
//! ```
//!
//! ## Design notes
//!
//! - Parameters are `TensorValue` handles. Optimizers update them in place via
//!   `TensorValue::set_value`, so module structs only need `&self` in `forward`.
//! - Every layer is explicit about the shapes it accepts; rank and width
//!   mismatches return [`NnError`] instead of panicking.
//! - `Rng` and `xavier_uniform` live in `mini-llm-random` and are re-exported
//!   here for convenience; the data layer depends on that crate directly.
//! - No dropout, no multi-head reshape and no fused kernels yet.

pub mod activation;
mod embedding;
mod error;
mod layernorm;
mod linear;
mod module;
mod moe;
mod optim;
mod rmsnorm;
mod sequential;
mod swiglu;

pub use activation::{Relu, Sigmoid, Tanh};
pub use embedding::Embedding;
pub use error::NnError;
pub use layernorm::LayerNorm;
pub use linear::Linear;
pub use mini_llm_random::{xavier_uniform, Rng};
pub use module::Module;
pub use moe::Moe;
pub use optim::{AdamW, Optimizer, Sgd};
pub use rmsnorm::RmsNorm;
pub use sequential::Sequential;
pub use swiglu::SwiGlu;

pub(crate) use module::expect_matrix;
