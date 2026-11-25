//! Attention: causal grouped-query attention, KV cache and the original
//! single-head educational primitive.

use std::error::Error;
use std::fmt::{Display, Formatter};

use mini_llm_autograd::TensorValue;
use mini_llm_nn::{Linear, Module, Rng};
use mini_llm_tensor::{Tensor, TensorError};

use crate::{ModelConfig, ModelError, PositionKind};

/// Rotary base from the RoPE paper.
const ROPE_BASE: f32 = 10_000.0;

/// Key/value cache for one attention layer: one `[len, head_dim]` tensor per
/// batch row and key/value head. Only used for inference.
pub struct AttentionCache {
    keys: Vec<Vec<Tensor>>,
    values: Vec<Vec<Tensor>>,
    batch: usize,
    /// Per row and cached position: whether the token is real or padding.
    real: Vec<Vec<bool>>,
    /// Flags for the token about to be appended by the next step.
    pending: Vec<bool>,
}

impl AttentionCache {
    pub fn new(batch: usize, num_kv_heads: usize, head_dim: usize) -> Self {
        let batch = batch.max(1);
        let empty = Tensor::zeros(vec![0, head_dim]).expect("valid empty cache");
        Self {
            keys: vec![vec![empty.clone(); num_kv_heads]; batch],
            values: vec![vec![empty; num_kv_heads]; batch],
            batch,
            real: vec![Vec::new(); batch],
            pending: vec![true; batch],
        }
    }

    /// Number of sequences held by this cache.
    pub fn batch(&self) -> usize {
        self.batch
    }

    /// Marks whether the token appended by the next step is real per row.
    ///
    /// Padding tokens are masked out of the attention scores.
    pub fn set_pending_real(&mut self, real: &[bool]) {
        for (slot, &flag) in self.pending.iter_mut().zip(real) {
            *slot = flag;
        }
    }

    /// Per-position real flags for `row` (true = attendable).
    pub fn real_flags(&self, row: usize) -> &[bool] {
        &self.real[row]
    }

    pub fn len(&self) -> usize {
        self.keys
            .first()
            .and_then(|heads| heads.first())
            .map(|key| key.shape()[0])
            .unwrap_or(0)
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
        self.forward_impl(input, batch, seq, mask, positions, None)
    }

    /// Like [`CausalSelfAttention::forward`], also recording one attention
    /// matrix per head (shape `[batch * seq, batch * seq]`).
    pub fn forward_traced(
        &self,
        input: &TensorValue,
        batch: usize,
        seq: usize,
        mask: &TensorValue,
        positions: &[usize],
        capture: &mut Vec<Tensor>,
    ) -> Result<TensorValue, ModelError> {
        self.forward_impl(input, batch, seq, mask, positions, Some(capture))
    }

    fn forward_impl(
        &self,
        input: &TensorValue,
        batch: usize,
        seq: usize,
        mask: &TensorValue,
        positions: &[usize],
        mut capture: Option<&mut Vec<Tensor>>,
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
            if let Some(capture) = capture.as_deref_mut() {
                capture.push(weights.value());
            }
            let attended = weights.matmul(&v)?;

            let out_weight = self
                .output
                .weight()
                .slice_rows(head * self.head_dim, self.head_dim)?;
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

    /// One-token step used by cached generation.
    ///
    /// `input` is `[batch, hidden]`; `positions[row]` is the absolute position of
    /// that row (rows may be at different positions). Padding tokens beyond each
    /// row's valid length are masked out of the attention scores.
    pub fn forward_step(
        &self,
        input: &TensorValue,
        positions: &[usize],
        cache: &mut AttentionCache,
    ) -> Result<TensorValue, ModelError> {
        let batch = cache.batch();
        if input.shape() != [batch, self.num_heads * self.head_dim] {
            return Err(ModelError::AttentionInputShape {
                expected: vec![batch, self.num_heads * self.head_dim],
                actual: input.shape(),
            });
        }
        if positions.len() != batch {
            return Err(ModelError::AttentionInputShape {
                expected: vec![batch],
                actual: vec![positions.len()],
            });
        }
        if cache.keys.len() != batch || cache.keys[0].len() != self.num_kv_heads {
            return Err(ModelError::CacheHeadMismatch {
                expected: self.num_kv_heads,
                actual: cache.keys[0].len(),
            });
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

        // Append one row per batch row and key/value head, then reuse it for
        // every query head in the group. The per-position flag records whether
        // the appended token is real or padding.
        for row in 0..batch {
            let key_row = key.slice_rows(row, 1)?;
            let value_row = value.slice_rows(row, 1)?;
            for kv_head in 0..self.num_kv_heads {
                let k = key_row.slice_cols(kv_head * self.head_dim, self.head_dim)?;
                let v = value_row.slice_cols(kv_head * self.head_dim, self.head_dim)?;
                let k_value = k.value();
                cache.keys[row][kv_head].append_row(k_value.data())?;
                let v_value = v.value();
                cache.values[row][kv_head].append_row(v_value.data())?;
            }
            let flag = cache.pending[row];
            cache.real[row].push(flag);
        }

        let mut output: Option<TensorValue> = None;
        let total = cache.len();
        for row in 0..batch {
            let mut merged: Option<TensorValue> = None;
            for head in 0..self.num_heads {
                let kv_head = head / group;
                let q = query
                    .slice_rows(row, 1)?
                    .slice_cols(head * self.head_dim, self.head_dim)?;
                let keys = TensorValue::new(cache.keys[row][kv_head].clone());
                let values = TensorValue::new(cache.values[row][kv_head].clone());

                // Cached positions are in the past; padding positions are
                // masked out using the per-position real flags.
                let mut scores = q.matmul(&keys.transpose()?)?.scale(scale)?;
                let flags = cache.real_flags(row);
                if flags.iter().any(|flag| !flag) {
                    let mask: Vec<f32> = flags
                        .iter()
                        .map(|flag| if *flag { 0.0 } else { -1e9 })
                        .collect();
                    let mask = TensorValue::new(Tensor::from_2d(1, total, mask)?);
                    scores = scores.add(&mask)?;
                }
                let weights = scores.softmax_rows()?;
                let attended = weights.matmul(&values)?;

                let out_weight = self
                    .output
                    .weight()
                    .slice_rows(head * self.head_dim, self.head_dim)?;
                let projected = attended.matmul(&out_weight)?;
                merged = Some(match merged {
                    Some(accumulated) => accumulated.add(&projected)?,
                    None => projected,
                });
            }

            let row_output = merged.ok_or(ModelError::EmptyHeadDimension)?;
            let scattered = row_output.scatter_add_rows(&[row], batch)?;
            output = Some(match output {
                Some(accumulated) => accumulated.add(&scattered)?,
                None => scattered,
            });
        }

        let merged = output.ok_or(ModelError::EmptyHeadDimension)?;
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
/// This educational primitive does not apply a causal mask. It predates the
/// autograd engine and is kept for the CLI demo and the concept lessons.
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
    fn attention_shapes_are_correct() {
        let q = Tensor::from_2d(2, 2, vec![1., 0., 0., 1.]).unwrap();
        let k = Tensor::from_2d(2, 2, vec![1., 0., 0., 1.]).unwrap();
        let v = Tensor::from_2d(2, 2, vec![10., 0., 0., 10.]).unwrap();

        let (out, weights) = scaled_dot_product_attention(&q, &k, &v).unwrap();
        assert_eq!(out.shape(), &[2, 2]);
        assert_eq!(weights.shape(), &[2, 2]);
        assert!((weights.data()[0] + weights.data()[1] - 1.0).abs() < 1e-5);
    }

    #[test]
    fn mask_blocks_other_samples_and_future_tokens() {
        let mask = causal_batch_mask(2, 2);
        // row 0 (sample 0, position 0) may only see column 0
        assert_eq!(mask[0], 0.0);
        assert!(mask[1] < -1e8);
        assert!(mask[2] < -1e8);
        // row 3 (sample 1, position 1) may see columns 2 and 3
        assert_eq!(mask[3 * 4 + 2], 0.0);
        assert_eq!(mask[3 * 4 + 3], 0.0);
        assert!(mask[3 * 4] < -1e8);
    }
}
