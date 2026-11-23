//! # mini-llm-nn
//!
//! Small, explicit neural-network building blocks on top of `mini-llm-autograd`:
//! [`Linear`], [`Embedding`], [`LayerNorm`], activations, a [`Sequential`]
//! container and two optimizers ([`Sgd`], [`AdamW`]).
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
//! - No dropout, no multi-head reshape and no fused kernels yet.

pub mod activation;
mod embedding;
mod layernorm;
mod linear;
mod optim;
mod rng;
mod sequential;

pub use activation::{Relu, Sigmoid, Tanh};
pub use embedding::Embedding;
pub use layernorm::LayerNorm;
pub use linear::Linear;
pub use optim::{AdamW, Optimizer, Sgd};
pub use rng::{xavier_uniform, Rng};
pub use sequential::Sequential;

use std::error::Error;
use std::fmt::{Display, Formatter};

use mini_llm_autograd::{AutogradError, TensorValue};

/// A composable piece of a model: a forward function plus its parameters.
pub trait Module {
    /// Runs the forward pass on a rank-2 activation tensor.
    fn forward(&self, input: &TensorValue) -> Result<TensorValue, NnError>;

    /// Every learnable tensor owned by this module, in a stable order.
    fn parameters(&self) -> Vec<TensorValue>;

    /// Human-readable layer name, used in summaries.
    fn name(&self) -> &'static str {
        "module"
    }

    /// Total number of scalar parameters.
    fn num_parameters(&self) -> usize {
        self.parameters()
            .iter()
            .map(|parameter| parameter.value().len())
            .sum()
    }

    /// A one-line description of the module and its parameter count.
    fn summary(&self) -> String {
        format!("{} ({} params)", self.name(), self.num_parameters())
    }
}

#[derive(Debug, Clone, PartialEq)]
pub enum NnError {
    /// An operation inside the autograd graph failed.
    Autograd(AutogradError),
    /// A layer was configured with a zero (or otherwise invalid) dimension.
    InvalidDimension { name: &'static str, value: usize },
    /// The input rank does not match what the layer expects.
    InputRank { expected: usize, actual: usize },
    /// The input width does not match the layer's `in_features`.
    InputWidth { expected: usize, actual: usize },
    /// A token id is outside the embedding vocabulary.
    TokenOutOfRange { index: usize, vocab_size: usize },
}

impl Display for NnError {
    fn fmt(&self, f: &mut Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Autograd(err) => Display::fmt(err, f),
            Self::InvalidDimension { name, value } => {
                write!(f, "{} must be greater than zero, got {}", name, value)
            }
            Self::InputRank { expected, actual } => {
                write!(f, "expected a rank-{} input, got rank {}", expected, actual)
            }
            Self::InputWidth { expected, actual } => {
                write!(f, "expected input width {}, got {}", expected, actual)
            }
            Self::TokenOutOfRange { index, vocab_size } => write!(
                f,
                "token id {} is outside the vocabulary of size {}",
                index, vocab_size
            ),
        }
    }
}

impl Error for NnError {}

impl From<AutogradError> for NnError {
    fn from(value: AutogradError) -> Self {
        Self::Autograd(value)
    }
}

impl From<mini_llm_tensor::TensorError> for NnError {
    fn from(value: mini_llm_tensor::TensorError) -> Self {
        Self::Autograd(AutogradError::Tensor(value))
    }
}

/// Validates that `input` is rank-2 with `expected_width` columns.
pub(crate) fn expect_matrix(input: &TensorValue, expected_width: usize) -> Result<usize, NnError> {
    let shape = input.shape();
    if shape.len() != 2 {
        return Err(NnError::InputRank {
            expected: 2,
            actual: shape.len(),
        });
    }
    if shape[1] != expected_width {
        return Err(NnError::InputWidth {
            expected: expected_width,
            actual: shape[1],
        });
    }
    Ok(shape[0])
}
