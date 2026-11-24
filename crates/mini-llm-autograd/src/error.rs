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
    /// The number of per-token weights does not match the number of targets.
    WeightCountMismatch { targets: usize, weights: usize },
    /// Every per-token weight is zero, so the loss is undefined.
    ZeroTotalWeight,
    /// A per-token weight is negative or not finite.
    InvalidWeight { index: usize, value: f32 },
    /// `set_value` was called on a value that is not a learnable parameter.
    NotAParameter,
    /// A row/column slice extends past the tensor bounds.
    SliceOutOfBounds {
        axis: &'static str,
        start: usize,
        len: usize,
        size: usize,
    },
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
            Self::WeightCountMismatch { targets, weights } => {
                write!(f, "expected {} per-token weights, got {}", targets, weights)
            }
            Self::ZeroTotalWeight => write!(f, "the loss mask has no active token"),
            Self::InvalidWeight { index, value } => {
                write!(f, "weight {} is invalid: {}", index, value)
            }
            Self::SliceOutOfBounds {
                axis,
                start,
                len,
                size,
            } => write!(
                f,
                "{} slice [{}, {}) is out of bounds for size {}",
                axis,
                start,
                start + len,
                size
            ),
        }
    }
}

impl Error for AutogradError {}

impl From<TensorError> for AutogradError {
    fn from(value: TensorError) -> Self {
        Self::Tensor(value)
    }
}
