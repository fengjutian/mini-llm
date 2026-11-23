//! Model primitives for mini-llm.
//!
//! - [`TinyGpt`]: a decoder-only Transformer language model built from autograd
//!   ops, [`TransformerBlock`]s and causal multi-head attention.
//! - [`scaled_dot_product_attention`]: the original transparent single-head
//!   primitive used by the CLI demo and the concept lessons on the web site.

mod attention;
mod block;
mod error;
mod gpt;

pub use attention::{causal_batch_mask, CausalSelfAttention};
pub use block::{Mlp, TransformerBlock};
pub use error::ModelError;
pub use gpt::TinyGpt;

use mini_llm_tensor::{Tensor, TensorError};
use std::error::Error;
use std::fmt::{Display, Formatter};

#[derive(Debug, Clone)]
pub struct ModelConfig {
    pub vocab_size: usize,
    pub hidden_size: usize,
    pub num_layers: usize,
    pub num_heads: usize,
    pub max_seq_len: usize,
}

impl Default for ModelConfig {
    fn default() -> Self {
        Self {
            vocab_size: 8_192,
            hidden_size: 256,
            num_layers: 4,
            num_heads: 4,
            max_seq_len: 512,
        }
    }
}

impl ModelConfig {
    /// A small, trainable configuration for character-level experiments.
    pub fn tiny(vocab_size: usize) -> Self {
        Self {
            vocab_size,
            hidden_size: 64,
            num_layers: 2,
            num_heads: 4,
            max_seq_len: 32,
        }
    }

    pub fn validate(&self) -> Result<(), ConfigError> {
        if self.vocab_size == 0
            || self.hidden_size == 0
            || self.num_layers == 0
            || self.num_heads == 0
            || self.max_seq_len == 0
        {
            return Err(ConfigError::ZeroDimension);
        }
        if !self.hidden_size.is_multiple_of(self.num_heads) {
            return Err(ConfigError::HiddenSizeNotDivisible {
                hidden_size: self.hidden_size,
                num_heads: self.num_heads,
            });
        }
        Ok(())
    }
}

#[derive(Debug, Clone, PartialEq)]
pub enum ConfigError {
    ZeroDimension,
    HiddenSizeNotDivisible {
        hidden_size: usize,
        num_heads: usize,
    },
}

impl Display for ConfigError {
    fn fmt(&self, f: &mut Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::ZeroDimension => write!(f, "model dimensions must all be greater than zero"),
            Self::HiddenSizeNotDivisible {
                hidden_size,
                num_heads,
            } => write!(
                f,
                "hidden_size ({}) must be divisible by num_heads ({})",
                hidden_size, num_heads
            ),
        }
    }
}

impl Error for ConfigError {}

#[derive(Debug)]
pub enum AttentionError {
    Tensor(TensorError),
    InvalidHeadDimension,
}

impl Display for AttentionError {
    fn fmt(&self, f: &mut Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Tensor(err) => Display::fmt(err, f),
            Self::InvalidHeadDimension => write!(f, "head dimension must be greater than zero"),
        }
    }
}

impl Error for AttentionError {}

impl From<TensorError> for AttentionError {
    fn from(value: TensorError) -> Self {
        Self::Tensor(value)
    }
}

/// Computes single-head scaled dot-product attention.
///
/// Shapes:
/// Q = [query_len, head_dim]
/// K = [key_len, head_dim]
/// V = [key_len, value_dim]
///
/// This educational primitive does not apply a causal mask.
pub fn scaled_dot_product_attention(
    q: &Tensor,
    k: &Tensor,
    v: &Tensor,
) -> Result<(Tensor, Tensor), AttentionError> {
    if q.rank() != 2 || k.rank() != 2 || v.rank() != 2 {
        return Err(AttentionError::Tensor(TensorError::ExpectedRank {
            expected: 2,
            actual: q.rank().max(k.rank()).max(v.rank()),
        }));
    }
    let head_dim = q.shape()[1];
    if head_dim == 0 {
        return Err(AttentionError::InvalidHeadDimension);
    }

    let k_t = k.transpose_2d()?;
    let scores = q.matmul(&k_t)?;
    let scale = (head_dim as f32).sqrt();
    let scaled = Tensor::new(
        scores.data().iter().map(|x| x / scale).collect(),
        scores.shape().to_vec(),
    )?;
    let weights = scaled.softmax_rows()?;
    let output = weights.matmul(v)?;
    Ok((output, weights))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn default_config_is_valid() {
        ModelConfig::default().validate().unwrap();
    }

    #[test]
    fn attention_shapes_are_correct() {
        let q = Tensor::from_2d(2, 2, vec![1., 0., 0., 1.]).unwrap();
        let k = Tensor::from_2d(2, 2, vec![1., 0., 0., 1.]).unwrap();
        let v = Tensor::from_2d(2, 2, vec![10., 0., 0., 10.]).unwrap();

        let (out, weights) = scaled_dot_product_attention(&q, &k, &v).unwrap();
        assert_eq!(out.shape(), &[2, 2]);
        assert_eq!(weights.shape(), &[2, 2]);
        assert!((weights.data()[0] + weights.data()[1] - 1.0).abs() < 1e-5);
    }
}
