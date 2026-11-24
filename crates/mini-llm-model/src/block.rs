//! A pre-norm Transformer decoder block with a selectable feed-forward network.

use mini_llm_autograd::TensorValue;
use mini_llm_nn::{Linear, Module, Moe, Relu, Rng, SwiGlu};

use crate::attention::{AttentionCache, CausalSelfAttention};
use crate::norm::NormLayer;
use crate::{FeedForwardKind, ModelConfig, ModelError};

/// `Linear(hidden, 4*hidden) -> ReLU -> Linear(4*hidden, hidden)`.
pub struct Mlp {
    expand: Linear,
    contract: Linear,
}

impl Mlp {
    pub fn new(config: &ModelConfig, rng: &mut Rng) -> Result<Self, ModelError> {
        let hidden = config.hidden_size;
        let inner = hidden * 4;
        Ok(Self {
            expand: Linear::new(hidden, inner, rng)?,
            contract: Linear::new(inner, hidden, rng)?,
        })
    }

    pub fn forward(&self, input: &TensorValue) -> Result<TensorValue, ModelError> {
        let hidden = Relu.forward(&self.expand.forward(input)?)?;
        Ok(self.contract.forward(&hidden)?)
    }

    pub fn parameters(&self) -> Vec<TensorValue> {
        let mut parameters = self.expand.parameters();
        parameters.extend(self.contract.parameters());
        parameters
    }
}

enum FeedForward {
    Dense(Mlp),
    Gated(SwiGlu),
    Mixture(Moe),
}

impl FeedForward {
    fn new(config: &ModelConfig, rng: &mut Rng) -> Result<Self, ModelError> {
        let hidden = config.hidden_size;
        let inner = hidden * 4;
        if config.num_experts > 1 {
            let top_k = if config.num_experts >= 4 { 2 } else { 1 };
            return Ok(Self::Mixture(Moe::new(
                hidden,
                inner,
                config.num_experts,
                top_k,
                rng,
            )?));
        }
        Ok(match config.feed_forward {
            FeedForwardKind::Relu => Self::Dense(Mlp::new(config, rng)?),
            FeedForwardKind::SwiGlu => Self::Gated(SwiGlu::new(hidden, inner, rng)?),
        })
    }

    fn forward(&self, input: &TensorValue) -> Result<TensorValue, ModelError> {
        Ok(match self {
            Self::Dense(layer) => layer.forward(input)?,
            Self::Gated(layer) => layer.forward(input)?,
            Self::Mixture(layer) => layer.forward(input)?,
        })
    }

    fn parameters(&self) -> Vec<TensorValue> {
        match self {
            Self::Dense(layer) => layer.parameters(),
            Self::Gated(layer) => layer.parameters(),
            Self::Mixture(layer) => layer.parameters(),
        }
    }
}

/// Inference state for one block.
pub struct BlockCache {
    pub attention: AttentionCache,
}

impl BlockCache {
    pub fn new(config: &ModelConfig) -> Self {
        Self {
            attention: AttentionCache::new(
                config.num_kv_heads,
                config.hidden_size / config.num_heads,
            ),
        }
    }
}

/// Pre-norm block: `x = x + attn(norm(x)); x = x + ffn(norm(x))`.
pub struct TransformerBlock {
    norm1: NormLayer,
    attention: CausalSelfAttention,
    norm2: NormLayer,
    feed_forward: FeedForward,
    hidden_size: usize,
    num_heads: usize,
}

impl TransformerBlock {
    pub fn new(config: &ModelConfig, rng: &mut Rng) -> Result<Self, ModelError> {
        Ok(Self {
            norm1: NormLayer::new(config.norm, config.hidden_size)?,
            attention: CausalSelfAttention::new(config, rng)?,
            norm2: NormLayer::new(config.norm, config.hidden_size)?,
            feed_forward: FeedForward::new(config, rng)?,
            hidden_size: config.hidden_size,
            num_heads: config.num_heads,
        })
    }

    pub fn forward(
        &self,
        input: &TensorValue,
        batch: usize,
        seq: usize,
        mask: &TensorValue,
        positions: &[usize],
    ) -> Result<TensorValue, ModelError> {
        let normalized = self.norm1.forward(input)?;
        let attended = self
            .attention
            .forward(&normalized, batch, seq, mask, positions)?;
        let residual = input.add(&attended)?;

        let normalized = self.norm2.forward(&residual)?;
        let transformed = self.feed_forward.forward(&normalized)?;
        Ok(residual.add(&transformed)?)
    }

    /// Single-token forward pass that appends to the KV cache.
    pub fn forward_step(
        &self,
        input: &TensorValue,
        position: usize,
        cache: &mut BlockCache,
    ) -> Result<TensorValue, ModelError> {
        let normalized = self.norm1.forward(input)?;
        let attended = self
            .attention
            .forward_step(&normalized, position, &mut cache.attention)?;
        let residual = input.add(&attended)?;

        let normalized = self.norm2.forward(&residual)?;
        let transformed = self.feed_forward.forward(&normalized)?;
        Ok(residual.add(&transformed)?)
    }

    pub fn parameters(&self) -> Vec<TensorValue> {
        let mut parameters = self.norm1.parameters();
        parameters.extend(self.attention.parameters());
        parameters.extend(self.norm2.parameters());
        parameters.extend(self.feed_forward.parameters());
        parameters
    }

    pub fn hidden_size(&self) -> usize {
        self.hidden_size
    }

    pub fn num_heads(&self) -> usize {
        self.num_heads
    }
}
