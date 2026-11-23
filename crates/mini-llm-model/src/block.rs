//! A pre-norm Transformer decoder block and its feed-forward network.

use mini_llm_autograd::TensorValue;
use mini_llm_nn::{LayerNorm, Linear, Module, Relu, Rng};

use crate::attention::CausalSelfAttention;
use crate::{ModelConfig, ModelError};

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

/// Pre-norm block: `x = x + attn(norm(x)); x = x + mlp(norm(x))`.
pub struct TransformerBlock {
    norm1: LayerNorm,
    attention: CausalSelfAttention,
    norm2: LayerNorm,
    mlp: Mlp,
    hidden_size: usize,
    num_heads: usize,
}

impl TransformerBlock {
    pub fn new(config: &ModelConfig, rng: &mut Rng) -> Result<Self, ModelError> {
        Ok(Self {
            norm1: LayerNorm::new(config.hidden_size)?,
            attention: CausalSelfAttention::new(config, rng)?,
            norm2: LayerNorm::new(config.hidden_size)?,
            mlp: Mlp::new(config, rng)?,
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
    ) -> Result<TensorValue, ModelError> {
        let normalized = self.norm1.forward(input)?;
        let attended = self.attention.forward(&normalized, batch, seq, mask)?;
        let residual = input.add(&attended)?;

        let normalized = self.norm2.forward(&residual)?;
        let transformed = self.mlp.forward(&normalized)?;
        Ok(residual.add(&transformed)?)
    }

    pub fn parameters(&self) -> Vec<TensorValue> {
        let mut parameters = self.norm1.parameters();
        parameters.extend(self.attention.parameters());
        parameters.extend(self.norm2.parameters());
        parameters.extend(self.mlp.parameters());
        parameters
    }

    pub fn hidden_size(&self) -> usize {
        self.hidden_size
    }

    pub fn num_heads(&self) -> usize {
        self.num_heads
    }
}
