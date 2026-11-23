//! Errors raised by the mini-llm autograd engine.

use mini_llm_tensor::TensorError;
use std::error::Error;
use std::fmt::{Display, Formatter};

#[derive(Debug, Clone, PartialEq)]
pub enum AutogradError {
    /// A tensor/shape operation failed.
    Tensor(TensorError),
    /// `backward()` was called on a value that was never tracked.
    NoGradientTracked,
    /// A cross-entropy target index is outside the vocabulary.
    TargetOutOfRange { index: usize, vocab_size: usize },
    /// The number of targets does not match the number of logits rows.
    TargetCountMismatch { logits_rows: usize, targets: usize },
    /// `set_value` was called on a value that is not a learnable parameter.
    NotAParameter,
}

impl Display for AutogradError {
    fn fmt(&self, f: &mut Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Tensor(err) => Display::fmt(err, f),
            Self::NoGradientTracked => write!(
                f,
                "cannot run backward on a value that does not require gradients"
            ),
            Self::TargetOutOfRange { index, vocab_size } => write!(
                f,
                "target index {} is outside the vocabulary of size {}",
                index, vocab_size
            ),
            Self::TargetCountMismatch {
                logits_rows,
                targets,
            } => write!(
                f,
                "cross-entropy expects {} targets for {} logits rows, got {}",
                logits_rows, logits_rows, targets
            ),
            Self::NotAParameter => write!(f, "only learnable parameters can be updated in place"),
        }
    }
}

impl Error for AutogradError {}

impl From<TensorError> for AutogradError {
    fn from(value: TensorError) -> Self {
        Self::Tensor(value)
    }
}
