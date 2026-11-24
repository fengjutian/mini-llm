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
mod norm;

pub use attention::{causal_batch_mask, AttentionCache, CausalSelfAttention};
pub use block::{BlockCache, Mlp, TransformerBlock};
pub use error::ModelError;
pub use gpt::TinyGpt;

use mini_llm_tensor::{Tensor, TensorError};
use std::error::Error;
use std::fmt::{Display, Formatter};

/// Normalization used inside the blocks.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum NormKind {
    #[default]
    LayerNorm,
    RmsNorm,
}

/// How positions are injected into the model.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum PositionKind {
    /// A learned embedding table of size `[max_seq_len, hidden]`.
    #[default]
    Learned,
    /// Rotary position embedding applied to queries and keys.
    Rope,
}

/// Which feed-forward network each block uses.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum FeedForwardKind {
    #[default]
    Relu,
    SwiGlu,
}

/// Architecture of a [`TinyGpt`].
#[derive(Debug, Clone)]
pub struct ModelConfig {
    pub vocab_size: usize,
    pub hidden_size: usize,
    pub num_layers: usize,
    pub num_heads: usize,
    pub max_seq_len: usize,
    /// Key/value heads for grouped-query attention; must divide `num_heads`.
    pub num_kv_heads: usize,
    pub norm: NormKind,
    pub position: PositionKind,
    pub feed_forward: FeedForwardKind,
    /// `1` uses one dense feed-forward per block; `>1` swaps in a top-k MoE.
    pub num_experts: usize,
}

impl Default for ModelConfig {
    fn default() -> Self {
        Self {
            vocab_size: 8_192,
            hidden_size: 256,
            num_layers: 4,
            num_heads: 4,
            max_seq_len: 512,
            num_kv_heads: 4,
            norm: NormKind::LayerNorm,
            position: PositionKind::Learned,
            feed_forward: FeedForwardKind::Relu,
            num_experts: 1,
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
            num_kv_heads: 4,
            norm: NormKind::LayerNorm,
            position: PositionKind::Learned,
            feed_forward: FeedForwardKind::Relu,
            num_experts: 1,
        }
    }

    /// The same budget as [`ModelConfig::tiny`] but with the modern stack:
    /// RMSNorm, RoPE, SwiGLU and grouped-query attention.
    pub fn modern(vocab_size: usize) -> Self {
        Self {
            vocab_size,
            hidden_size: 64,
            num_layers: 2,
            num_heads: 4,
            max_seq_len: 64,
            num_kv_heads: 2,
            norm: NormKind::RmsNorm,
            position: PositionKind::Rope,
            feed_forward: FeedForwardKind::SwiGlu,
            num_experts: 1,
        }
    }

    pub fn validate(&self) -> Result<(), ConfigError> {
        if self.vocab_size == 0
            || self.hidden_size == 0
            || self.num_layers == 0
            || self.num_heads == 0
            || self.max_seq_len == 0
            || self.num_kv_heads == 0
            || self.num_experts == 0
        {
            return Err(ConfigError::ZeroDimension);
        }
        if !self.hidden_size.is_multiple_of(self.num_heads) {
            return Err(ConfigError::HiddenSizeNotDivisible {
                hidden_size: self.hidden_size,
                num_heads: self.num_heads,
            });
        }
        if !self.num_heads.is_multiple_of(self.num_kv_heads) {
            return Err(ConfigError::HeadsNotDivisible {
                num_heads: self.num_heads,
                num_kv_heads: self.num_kv_heads,
            });
        }
        if (self.hidden_size / self.num_heads) % 2 != 0 && self.position == PositionKind::Rope {
            return Err(ConfigError::OddHeadDimensionForRope {
                head_dim: self.hidden_size / self.num_heads,
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
    HeadsNotDivisible {
        num_heads: usize,
        num_kv_heads: usize,
    },
    OddHeadDimensionForRope {
        head_dim: usize,
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
            Self::HeadsNotDivisible {
                num_heads,
                num_kv_heads,
            } => write!(
                f,
                "num_heads ({}) must be divisible by num_kv_heads ({})",
                num_heads, num_kv_heads
            ),
            Self::OddHeadDimensionForRope { head_dim } => write!(
                f,
                "RoPE needs an even head dimension, got {}",
                head_dim
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
