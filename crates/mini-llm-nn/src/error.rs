//! Errors raised by the neural-network layers.

use std::error::Error;
use std::fmt::{Display, Formatter};

use mini_llm_autograd::AutogradError;

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
    /// A mixture-of-experts layer got a top-k outside `1..=num_experts`.
    InvalidTopK { top_k: usize, num_experts: usize },
    /// Restored optimizer state does not match a parameter's length.
    StateShape {
        index: usize,
        expected: usize,
        actual: usize,
    },
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
            Self::InvalidTopK { top_k, num_experts } => write!(
                f,
                "top-k must be between 1 and {}, got {}",
                num_experts, top_k
            ),
            Self::StateShape {
                index,
                expected,
                actual,
            } => write!(
                f,
                "optimizer state for parameter {} has {} values, expected {}",
                index, actual, expected
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
