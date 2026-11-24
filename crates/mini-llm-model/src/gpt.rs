//! Tiny GPT: a character-level decoder-only Transformer.
//!
//! The architecture is configurable: LayerNorm or RMSNorm, learned positions or
//! RoPE, a ReLU MLP or SwiGLU, and dense or grouped-query attention. Generation
//! uses a KV cache so each new token costs one token of compute instead of a
//! full window.

use std::cell::RefCell;

use mini_llm_autograd::TensorValue;
use mini_llm_nn::{Embedding, Linear, Module, Rng};
use mini_llm_tensor::Tensor;

use crate::attention::causal_batch_mask;
use crate::block::{BlockCache, TransformerBlock};
use crate::norm::NormLayer;
use crate::{ModelConfig, ModelError, PositionKind};

/// One captured attention matrix, for inspection and visualization.
#[derive(Debug, Clone)]
pub struct AttentionRecord {
    pub layer: usize,
    pub head: usize,
    /// Shape `[batch * seq, batch * seq]`.
    pub weights: Tensor,
}

/// A decoder-only Transformer language model.
pub struct TinyGpt {
    config: ModelConfig,
    token_embedding: Embedding,
    position_embedding: Option<Embedding>,
    blocks: Vec<TransformerBlock>,
    final_norm: NormLayer,
    lm_head: Linear,
    /// Causal mask for the last `(batch, seq)` pair; rebuilt only on change.
    mask_cache: RefCell<Option<((usize, usize), Tensor)>>,
}

impl TinyGpt {
    /// Builds the model with deterministic initialization from `rng`.
    pub fn new(config: ModelConfig, rng: &mut Rng) -> Result<Self, ModelError> {
        config.validate()?;
        let hidden = config.hidden_size;

        let token_embedding = Embedding::new(config.vocab_size, hidden, 0.02, rng)?;
        let position_embedding = match config.position {
            PositionKind::Learned => Some(Embedding::new(config.max_seq_len, hidden, 0.01, rng)?),
            PositionKind::Rope => None,
        };

        let mut blocks = Vec::with_capacity(config.num_layers);
        for _ in 0..config.num_layers {
            blocks.push(TransformerBlock::new(&config, rng)?);
        }

        let final_norm = NormLayer::new(config.norm, hidden)?;
        let lm_head = Linear::without_bias(hidden, config.vocab_size, rng)?;

        Ok(Self {
            config,
            token_embedding,
            position_embedding,
            blocks,
            final_norm,
            lm_head,
            mask_cache: RefCell::new(None),
        })
    }

    /// Returns the causal mask for `(batch, seq)`, reusing the cached tensor.
    fn causal_mask(&self, batch: usize, seq: usize) -> Result<TensorValue, ModelError> {
        let rows = batch * seq;
        {
            let cache = self.mask_cache.borrow();
            if let Some((key, tensor)) = cache.as_ref() {
                if *key == (batch, seq) {
                    return Ok(TensorValue::new(tensor.clone()));
                }
            }
        }
        let tensor = Tensor::new(causal_batch_mask(batch, seq), vec![rows, rows])?;
        *self.mask_cache.borrow_mut() = Some(((batch, seq), tensor.clone()));
        Ok(TensorValue::new(tensor))
    }

    pub fn config(&self) -> &ModelConfig {
        &self.config
    }

    pub fn parameters(&self) -> Vec<TensorValue> {
        let mut parameters = self.token_embedding.parameters();
        if let Some(position_embedding) = &self.position_embedding {
            parameters.extend(position_embedding.parameters());
        }
        for block in &self.blocks {
            parameters.extend(block.parameters());
        }
        parameters.extend(self.final_norm.parameters());
        parameters.extend(self.lm_head.parameters());
        parameters
    }

    pub fn num_parameters(&self) -> usize {
        self.parameters()
            .iter()
            .map(|parameter| parameter.value().len())
            .sum()
    }

    /// Runs the model on `tokens` shaped as `[batch, seq]` flattened row-major.
    ///
    /// Returns logits of shape `[batch * seq, vocab_size]`.
    pub fn forward(&self, tokens: &[usize], batch: usize) -> Result<TensorValue, ModelError> {
        self.forward_impl(tokens, batch, None)
    }

    /// Runs the model and records every attention matrix.
    ///
    /// Returns the logits plus one [`AttentionRecord`] per layer and head, in
    /// layer-major order. Matrices have shape `[batch * seq, batch * seq]`.
    pub fn forward_traced(
        &self,
        tokens: &[usize],
        batch: usize,
    ) -> Result<(TensorValue, Vec<AttentionRecord>), ModelError> {
        let mut records = Vec::new();
        let logits = self.forward_impl(tokens, batch, Some(&mut records))?;
        Ok((logits, records))
    }

    fn forward_impl(
        &self,
        tokens: &[usize],
        batch: usize,
        mut capture: Option<&mut Vec<AttentionRecord>>,
    ) -> Result<TensorValue, ModelError> {
        if batch == 0 {
            return Err(ModelError::BatchMismatch {
                tokens: tokens.len(),
                batch,
            });
        }
        if !tokens.len().is_multiple_of(batch) {
            return Err(ModelError::BatchMismatch {
                tokens: tokens.len(),
                batch,
            });
        }
        let seq = tokens.len() / batch;
        if seq == 0 {
            return Err(ModelError::BatchMismatch {
                tokens: tokens.len(),
                batch,
            });
        }
        if seq > self.config.max_seq_len {
            return Err(ModelError::SequenceTooLong {
                length: seq,
                max_seq_len: self.config.max_seq_len,
            });
        }
        if let Some(&index) = tokens
            .iter()
            .find(|&&index| index >= self.config.vocab_size)
        {
            return Err(ModelError::TokenOutOfRange {
                index,
                vocab_size: self.config.vocab_size,
            });
        }

        let positions: Vec<usize> = (0..batch).flat_map(|_| 0..seq).collect();
        let mut embedded = self.token_embedding.lookup(tokens)?;
        if let Some(position_embedding) = &self.position_embedding {
            embedded = embedded.add(&position_embedding.lookup(&positions)?)?;
        }

        let mask = self.causal_mask(batch, seq)?;

        let mut hidden = embedded;
        for (layer, block) in self.blocks.iter().enumerate() {
            match capture.as_deref_mut() {
                Some(records) => {
                    let mut weights = Vec::new();
                    hidden = block.forward_traced(
                        &hidden,
                        batch,
                        seq,
                        &mask,
                        &positions,
                        &mut weights,
                    )?;
                    for (head, matrix) in weights.into_iter().enumerate() {
                        records.push(AttentionRecord {
                            layer,
                            head,
                            weights: matrix,
                        });
                    }
                }
                None => {
                    hidden = block.forward(&hidden, batch, seq, &mask, &positions)?;
                }
            }
        }
        let normalized = self.final_norm.forward(&hidden)?;
        Ok(self.lm_head.forward(&normalized)?)
    }

    /// Fresh KV caches, one per block.
    pub fn empty_caches(&self) -> Vec<BlockCache> {
        self.blocks
            .iter()
            .map(|_| BlockCache::new(&self.config))
            .collect()
    }

    /// Runs one token through the model at `position`, updating the KV caches.
    pub fn step_token(
        &self,
        token: usize,
        position: usize,
        caches: &mut [BlockCache],
    ) -> Result<TensorValue, ModelError> {
        if token >= self.config.vocab_size {
            return Err(ModelError::TokenOutOfRange {
                index: token,
                vocab_size: self.config.vocab_size,
            });
        }
        if caches.len() != self.blocks.len() {
            return Err(ModelError::CacheDepthMismatch {
                expected: self.blocks.len(),
                actual: caches.len(),
            });
        }

        let mut hidden = self.token_embedding.lookup(&[token])?;
        if let Some(position_embedding) = &self.position_embedding {
            hidden = hidden.add(&position_embedding.lookup(&[position])?)?;
        }
        for (block, cache) in self.blocks.iter().zip(caches.iter_mut()) {
            hidden = block.forward_step(&hidden, position, cache)?;
        }
        let normalized = self.final_norm.forward(&hidden)?;
        Ok(self.lm_head.forward(&normalized)?)
    }

    /// Autoregressive sampling with a KV cache.
    ///
    /// With learned position embeddings the output stops at `max_seq_len`;
    /// with RoPE the position index keeps growing (quality degrades past the
    /// trained context, but the mechanics stay valid).
    pub fn generate(
        &self,
        prompt: &[usize],
        max_new_tokens: usize,
        temperature: f32,
        rng: &mut Rng,
    ) -> Result<Vec<usize>, ModelError> {
        if prompt.is_empty() {
            return Err(ModelError::EmptyPrompt);
        }
        if let Some(&index) = prompt
            .iter()
            .find(|&&index| index >= self.config.vocab_size)
        {
            return Err(ModelError::TokenOutOfRange {
                index,
                vocab_size: self.config.vocab_size,
            });
        }

        let start = prompt.len().saturating_sub(self.config.max_seq_len);
        let window = &prompt[start..];
        let mut caches = self.empty_caches();

        let mut logits = None;
        for (offset, &token) in window.iter().enumerate() {
            logits = Some(self.step_token(token, offset, &mut caches)?);
        }
        let mut logits = logits.ok_or(ModelError::EmptyPrompt)?;

        let mut tokens = prompt.to_vec();
        for _ in 0..max_new_tokens {
            let next = sample_next(&logits, self.config.vocab_size, temperature, rng)?;
            tokens.push(next);

            if self.position_embedding.is_some() && tokens.len() >= self.config.max_seq_len {
                break;
            }
            logits = self.step_token(next, tokens.len() - 1, &mut caches)?;
        }
        Ok(tokens)
    }

    /// Reference generation without a KV cache: recomputes the whole window at
    /// every step. Kept for tests and for comparison with [`TinyGpt::generate`].
    pub fn generate_uncached(
        &self,
        prompt: &[usize],
        max_new_tokens: usize,
        temperature: f32,
        rng: &mut Rng,
    ) -> Result<Vec<usize>, ModelError> {
        if prompt.is_empty() {
            return Err(ModelError::EmptyPrompt);
        }
        let mut tokens = prompt.to_vec();
        for _ in 0..max_new_tokens {
            let start = tokens.len().saturating_sub(self.config.max_seq_len);
            let window = &tokens[start..];
            let logits = self.forward(window, 1)?;
            let next = sample_next(&logits, self.config.vocab_size, temperature, rng)?;
            tokens.push(next);
        }
        Ok(tokens)
    }
}

fn sample_next(
    logits: &TensorValue,
    vocab: usize,
    temperature: f32,
    rng: &mut Rng,
) -> Result<usize, ModelError> {
    let values = logits.value();
    let rows = values.shape()[0];
    let last = &values.data()[(rows - 1) * vocab..rows * vocab];
    if temperature <= 0.0 {
        return Ok(argmax(last));
    }
    let scaled: Vec<f32> = last.iter().map(|value| value / temperature).collect();
    let probabilities = Tensor::from_2d(1, vocab, scaled)?
        .softmax_rows()?
        .into_data();
    Ok(sample(&probabilities, rng))
}

fn argmax(values: &[f32]) -> usize {
    let mut best = 0usize;
    for (index, &value) in values.iter().enumerate() {
        if value > values[best] {
            best = index;
        }
    }
    best
}

fn sample(probabilities: &[f32], rng: &mut Rng) -> usize {
    let mut threshold = rng.next_f32();
    for (index, &probability) in probabilities.iter().enumerate() {
        threshold -= probability;
        if threshold <= 0.0 {
            return index;
        }
    }
    probabilities.len().saturating_sub(1)
}
