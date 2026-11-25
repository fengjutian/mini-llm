//! Architecture configuration: dimensions, switches and their on-disk codes.

use std::error::Error;
use std::fmt::{Display, Formatter};

/// Normalization used inside the blocks.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum NormKind {
    #[default]
    LayerNorm,
    RmsNorm,
}

impl NormKind {
    pub const fn code(self) -> u32 {
        match self {
            Self::LayerNorm => 0,
            Self::RmsNorm => 1,
        }
    }

    pub const fn from_code(code: u32) -> Option<Self> {
        match code {
            0 => Some(Self::LayerNorm),
            1 => Some(Self::RmsNorm),
            _ => None,
        }
    }
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

impl PositionKind {
    pub const fn code(self) -> u32 {
        match self {
            Self::Learned => 0,
            Self::Rope => 1,
        }
    }

    pub const fn from_code(code: u32) -> Option<Self> {
        match code {
            0 => Some(Self::Learned),
            1 => Some(Self::Rope),
            _ => None,
        }
    }
}

/// Which feed-forward network each block uses.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum FeedForwardKind {
    #[default]
    Relu,
    SwiGlu,
}

impl FeedForwardKind {
    pub const fn code(self) -> u32 {
        match self {
            Self::Relu => 0,
            Self::SwiGlu => 1,
        }
    }

    pub const fn from_code(code: u32) -> Option<Self> {
        match code {
            0 => Some(Self::Relu),
            1 => Some(Self::SwiGlu),
            _ => None,
        }
    }
}

/// Architecture of a [`crate::TinyGpt`].
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
    /// Dropout probability on the residual branches (`0.0` disables it).
    pub dropout: f32,
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
            dropout: 0.0,
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
            dropout: 0.0,
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
            dropout: 0.0,
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
        if !(0.0..=1.0).contains(&self.dropout) {
            return Err(ConfigError::InvalidDropout {
                value: self.dropout,
            });
        }
        let head_dim = self.hidden_size / self.num_heads;
        if self.position == PositionKind::Rope && !head_dim.is_multiple_of(2) {
            return Err(ConfigError::OddHeadDimensionForRope { head_dim });
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
    InvalidDropout {
        value: f32,
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
            Self::OddHeadDimensionForRope { head_dim } => {
                write!(f, "RoPE needs an even head dimension, got {}", head_dim)
            }
            Self::InvalidDropout { value } => {
                write!(f, "dropout must be in [0, 1], got {}", value)
            }
        }
    }
}

impl Error for ConfigError {}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn default_config_is_valid() {
        ModelConfig::default().validate().unwrap();
        ModelConfig::tiny(32).validate().unwrap();
        ModelConfig::modern(32).validate().unwrap();
    }

    #[test]
    fn dropout_must_be_a_probability() {
        let valid = ModelConfig {
            dropout: 0.5,
            ..ModelConfig::default()
        };
        valid.validate().unwrap();

        for invalid in [-0.1, 1.5] {
            let config = ModelConfig {
                dropout: invalid,
                ..ModelConfig::default()
            };
            assert!(matches!(
                config.validate(),
                Err(ConfigError::InvalidDropout { .. })
            ));
        }
    }

    #[test]
    fn architecture_codes_roundtrip() {
        for kind in [NormKind::LayerNorm, NormKind::RmsNorm] {
            assert_eq!(NormKind::from_code(kind.code()), Some(kind));
        }
        for kind in [PositionKind::Learned, PositionKind::Rope] {
            assert_eq!(PositionKind::from_code(kind.code()), Some(kind));
        }
        for kind in [FeedForwardKind::Relu, FeedForwardKind::SwiGlu] {
            assert_eq!(FeedForwardKind::from_code(kind.code()), Some(kind));
        }
        assert_eq!(NormKind::from_code(9), None);
    }
}
