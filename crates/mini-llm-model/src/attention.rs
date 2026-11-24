//! Causal grouped-query attention with optional rotary positions.

use mini_llm_autograd::TensorValue;
use mini_llm_nn::{Linear, Module, Rng};
use mini_llm_tensor::Tensor;

use crate::{ModelConfig, ModelError, PositionKind};

/// Rotary base from the RoPE paper.
const ROPE_BASE: f32 = 10_000.0;

/// Key/value cache for one attention layer, one `[len, head_dim]` tensor per
/// key/value head. Only used for inference.
pub struct AttentionCache {
    keys: Vec<Tensor>,
    values: Vec<Tensor>,
}

impl AttentionCache {
    pub fn new(num_kv_heads: usize, head_dim: usize) -> Self {
        let empty = Tensor::zeros(vec![0, head_dim]).expect("valid empty cache");
        Self {
            keys: vec![empty; num_kv_heads],
            values: vec![Tensor::zeros(vec![0, head_dim]).expect("valid empty cache"); num_kv_heads],
        }
    }

    pub fn len(&self) -> usize {
        self.keys.first().map(|key| key.shape()[0]).unwrap_or(0)
    }

    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }
}

/// Multi-head self-attention with a causal, per-sample mask.
///
/// Queries always use `num_heads` heads; keys and values use `num_kv_heads`,
/// so several query heads share one key/value head (grouped-query attention).
/// The attention matrix is built over the flattened batch: rows are
/// `[batch * seq]`, and a block-diagonal causal mask prevents a token from
/// attending to other samples or to future positions.
pub struct CausalSelfAttention {
    query: Linear,
    key: Linear,
    value: Linear,
    output: Linear,
    num_heads: usize,
    num_kv_heads: usize,
    head_dim: usize,
    use_rope: bool,
}

impl CausalSelfAttention {
    pub fn new(config: &ModelConfig, rng: &mut Rng) -> Result<Self, ModelError> {
        let hidden = config.hidden_size;
        let head_dim = hidden / config.num_heads;
        let kv_dim = config.num_kv_heads * head_dim;
        Ok(Self {
            query: Linear::new(hidden, hidden, rng)?,
            key: Linear::new(hidden, kv_dim, rng)?,
            value: Linear::new(hidden, kv_dim, rng)?,
            output: Linear::new(hidden, hidden, rng)?,
            num_heads: config.num_heads,
            num_kv_heads: config.num_kv_heads,
            head_dim,
            use_rope: config.position == PositionKind::Rope,
        })
    }

    pub fn num_heads(&self) -> usize {
        self.num_heads
    }

    pub fn num_kv_heads(&self) -> usize {
        self.num_kv_heads
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
        positions: &[usize],
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

        let (query, key) = if self.use_rope {
            (
                query.rope(self.head_dim, positions, ROPE_BASE)?,
                key.rope(self.head_dim, positions, ROPE_BASE)?,
            )
        } else {
            (query, key)
        };

        let scale = 1.0 / (self.head_dim as f32).sqrt();
        let group = self.num_heads / self.num_kv_heads;
        let mut merged: Option<TensorValue> = None;

        for head in 0..self.num_heads {
            let kv_head = head / group;
            let q = query.slice_cols(head * self.head_dim, self.head_dim)?;
            let k = key.slice_cols(kv_head * self.head_dim, self.head_dim)?;
            let v = value.slice_cols(kv_head * self.head_dim, self.head_dim)?;

            let scores = q.matmul(&k.transpose()?)?.scale(scale)?;
            let masked = scores.add(mask)?;
            let weights = masked.softmax_rows()?;
            let attended = weights.matmul(&v)?;

            let out_weight = self.output.weight().slice_rows(head * self.head_dim, self.head_dim)?;
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

    /// Single-token step used by cached generation (`output.len() == 1`).
    pub fn forward_step(
        &self,
        input: &TensorValue,
        position: usize,
        cache: &mut AttentionCache,
    ) -> Result<TensorValue, ModelError> {
        if input.shape() != [1, self.num_heads * self.head_dim] {
            return Err(ModelError::AttentionInputShape {
                expected: vec![1, self.num_heads * self.head_dim],
                actual: input.shape(),
            });
        }
        if cache.keys.len() != self.num_kv_heads {
            return Err(ModelError::CacheHeadMismatch {
                expected: self.num_kv_heads,
                actual: cache.keys.len(),
            });
        }

        let query = self.query.forward(input)?;
        let key = self.key.forward(input)?;
        let value = self.value.forward(input)?;

        let (query, key) = if self.use_rope {
            (
                query.rope(self.head_dim, &[position], ROPE_BASE)?,
                key.rope(self.head_dim, &[position], ROPE_BASE)?,
            )
        } else {
            (query, key)
        };

        let scale = 1.0 / (self.head_dim as f32).sqrt();
        let group = self.num_heads / self.num_kv_heads;
        let mut merged: Option<TensorValue> = None;

        for head in 0..self.num_heads {
            let kv_head = head / group;
            let q = query.slice_cols(head * self.head_dim, self.head_dim)?;
            let k = key.slice_cols(kv_head * self.head_dim, self.head_dim)?;
            let v = value.slice_cols(kv_head * self.head_dim, self.head_dim)?;

            cache.keys[kv_head] = append_row(&cache.keys[kv_head], &k.value())?;
            cache.values[kv_head] = append_row(&cache.values[kv_head], &v.value())?;

            let keys = TensorValue::new(cache.keys[kv_head].clone());
            let values = TensorValue::new(cache.values[kv_head].clone());

            // All cached positions are in the past, so no mask is needed.
            let scores = q.matmul(&keys.transpose()?)?.scale(scale)?;
            let weights = scores.softmax_rows()?;
            let attended = weights.matmul(&values)?;

            let out_weight = self.output.weight().slice_rows(head * self.head_dim, self.head_dim)?;
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

/// Appends a `[1, cols]` row to a `[rows, cols]` tensor.
fn append_row(tensor: &Tensor, row: &Tensor) -> Result<Tensor, mini_llm_tensor::TensorError> {
    let (rows, cols) = (tensor.shape()[0], tensor.shape()[1]);
    let mut data = tensor.data().to_vec();
    data.extend_from_slice(row.data());
    Tensor::new(data, vec![rows + 1, cols])
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
