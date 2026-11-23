//! Errors raised by the model layer.

use mini_llm_autograd::AutogradError;
use mini_llm_nn::NnError;
use std::error::Error;
use std::fmt::{Display, Formatter};

use crate::ConfigError;

#[derive(Debug, Clone, PartialEq)]
pub enum ModelError {
    Config(ConfigError),
    Nn(NnError),
    Autograd(AutogradError),
    /// Inputs to attention had the wrong shape.
    AttentionInputShape {
        expected: Vec<usize>,
        actual: Vec<usize>,
    },
    EmptyHeadDimension,
    /// A token id is outside the vocabulary.
    TokenOutOfRange {
        index: usize,
        vocab_size: usize,
    },
    /// The requested sequence is longer than `max_seq_len`.
    SequenceTooLong {
        length: usize,
        max_seq_len: usize,
    },
    /// `tokens.len()` is not divisible by the batch size.
    BatchMismatch {
        tokens: usize,
        batch: usize,
    },
    EmptyPrompt,
}

impl Display for ModelError {
    fn fmt(&self, f: &mut Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Config(err) => Display::fmt(err, f),
            Self::Nn(err) => Display::fmt(err, f),
            Self::Autograd(err) => Display::fmt(err, f),
            Self::AttentionInputShape { expected, actual } => write!(
                f,
                "attention expected input shape {:?}, got {:?}",
                expected, actual
            ),
            Self::EmptyHeadDimension => write!(f, "head dimension must be greater than zero"),
            Self::TokenOutOfRange { index, vocab_size } => write!(
                f,
                "token id {} is outside the vocabulary of size {}",
                index, vocab_size
            ),
            Self::SequenceTooLong {
                length,
                max_seq_len,
            } => write!(
                f,
                "sequence length {} exceeds the maximum of {}",
                length, max_seq_len
            ),
            Self::BatchMismatch { tokens, batch } => write!(
                f,
                "{} tokens cannot be split into batches of {}",
                tokens, batch
            ),
            Self::EmptyPrompt => write!(f, "generation requires a non-empty prompt"),
        }
    }
}

impl Error for ModelError {}

impl From<ConfigError> for ModelError {
    fn from(value: ConfigError) -> Self {
        Self::Config(value)
    }
}

impl From<NnError> for ModelError {
    fn from(value: NnError) -> Self {
        Self::Nn(value)
    }
}

impl From<AutogradError> for ModelError {
    fn from(value: AutogradError) -> Self {
        Self::Autograd(value)
    }
}

impl From<mini_llm_tensor::TensorError> for ModelError {
    fn from(value: mini_llm_tensor::TensorError) -> Self {
        Self::Autograd(AutogradError::Tensor(value))
    }
}
