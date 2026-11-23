//! Causal multi-head self-attention.

use mini_llm_autograd::TensorValue;
use mini_llm_nn::{Linear, Module, Rng};

use crate::{ModelConfig, ModelError};

/// Multi-head self-attention with a causal, per-sample mask.
///
/// The attention matrix is built over the *flattened* batch: rows are
/// `[batch * seq]`, and a block-diagonal causal mask prevents a token from
/// attending to other samples or to future positions. This keeps the whole
/// operation as two 2-D matmuls per head, which is easy to inspect.
pub struct CausalSelfAttention {
    query: Linear,
    key: Linear,
    value: Linear,
    output: Linear,
    num_heads: usize,
    head_dim: usize,
}

impl CausalSelfAttention {
    pub fn new(config: &ModelConfig, rng: &mut Rng) -> Result<Self, ModelError> {
        let hidden = config.hidden_size;
        let head_dim = hidden / config.num_heads;
        Ok(Self {
            query: Linear::new(hidden, hidden, rng)?,
            key: Linear::new(hidden, hidden, rng)?,
            value: Linear::new(hidden, hidden, rng)?,
            output: Linear::new(hidden, hidden, rng)?,
            num_heads: config.num_heads,
            head_dim,
        })
    }

    pub fn num_heads(&self) -> usize {
        self.num_heads
    }

    pub fn head_dim(&self) -> usize {
        self.head_dim
    }

    /// `input` is `[batch * seq, hidden]`; `mask` is `[batch * seq, batch * seq]`.
    pub fn forward(
        &self,
        input: &TensorValue,
        batch: usize,
        seq: usize,
        mask: &TensorValue,
    ) -> Result<TensorValue, ModelError> {
        let rows = batch * seq;
        if input.shape() != [rows, self.num_heads * self.head_dim] {
            return Err(ModelError::AttentionInputShape {
                expected: vec![rows, self.num_heads * self.head_dim],
                actual: input.shape(),
            });
        }
        if self.head_dim == 0 {
            return Err(ModelError::EmptyHeadDimension);
        }

        let query = self.query.forward(input)?;
        let key = self.key.forward(input)?;
        let value = self.value.forward(input)?;
        let scale = 1.0 / (self.head_dim as f32).sqrt();

        let mut merged: Option<TensorValue> = None;
        for head in 0..self.num_heads {
            let start = head * self.head_dim;

            let q = query.slice_cols(start, self.head_dim)?;
            let k = key.slice_cols(start, self.head_dim)?;
            let v = value.slice_cols(start, self.head_dim)?;

            // [rows, head_dim] x [head_dim, rows] -> [rows, rows]
            let scores = q.matmul(&k.transpose()?)?.scale(scale)?;
            let masked = scores.add(mask)?;
            let weights = masked.softmax_rows()?;
            let attended = weights.matmul(&v)?;

            // Project this head back with its own slice of the output weights.
            let out_weight = self.output.weight().slice_rows(start, self.head_dim)?;
            let projected = attended.matmul(&out_weight)?;
            merged = Some(match merged {
                Some(accumulated) => accumulated.add(&projected)?,
                None => projected,
            });
        }

        let merged = merged.ok_or(ModelError::EmptyHeadDimension)?;
        match self.output.bias() {
            Some(bias) => Ok(merged.add(bias)?),
            None => Ok(merged),
        }
    }

    pub fn parameters(&self) -> Vec<TensorValue> {
        let mut parameters = Vec::new();
        parameters.extend(self.query.parameters());
        parameters.extend(self.key.parameters());
        parameters.extend(self.value.parameters());
        parameters.extend(self.output.parameters());
        parameters
    }
}

/// Builds the `[batch * seq, batch * seq]` mask: `0` for allowed positions and
/// a large negative value for other samples or future tokens.
pub fn causal_batch_mask(batch: usize, seq: usize) -> Vec<f32> {
    let rows = batch * seq;
    let mut data = vec![-1e9f32; rows * rows];
    for sample in 0..batch {
        for row in 0..seq {
            for column in 0..=row {
                data[(sample * seq + row) * rows + sample * seq + column] = 0.0;
            }
        }
    }
    data
}
