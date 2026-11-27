//! Tiny GPT: a character-level decoder-only Transformer.
//!
//! The architecture is configurable: LayerNorm or RMSNorm, learned positions or
//! RoPE, a ReLU MLP or SwiGLU, and dense or grouped-query attention. Generation
//! uses a KV cache so each new token costs one token of compute instead of a
//! full window. [Sampling] controls temperature, top-k and top-p.

use std::cell::RefCell;

use mini_llm_autograd::TensorValue;
use mini_llm_nn::{Embedding, Linear, Module, Rng};
use mini_llm_tensor::Tensor;

use crate::attention::causal_batch_mask;
use crate::block::{BlockCache, TransformerBlock};
use crate::language_model::LanguageModel;
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

/// Sampling controls for [`TinyGpt::generate_with`].
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Sampling {
    /// `<= 0.0` selects the argmax (greedy decoding).
    pub temperature: f32,
    /// Keep only the `k` most likely tokens before sampling.
    pub top_k: Option<usize>,
    /// Keep the smallest prefix of tokens whose probability mass reaches `p`.
    pub top_p: Option<f32>,
}

impl Sampling {
    pub fn greedy() -> Self {
        Self {
            temperature: 0.0,
            top_k: None,
            top_p: None,
        }
    }

    pub fn with_temperature(temperature: f32) -> Self {
        Self {
            temperature,
            ..Self::greedy()
        }
    }

    pub fn top_k(mut self, k: usize) -> Self {
        self.top_k = Some(k.max(1));
        self
    }

    pub fn top_p(mut self, p: f32) -> Self {
        self.top_p = Some(p.clamp(0.0, 1.0));
        self
    }
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
        if tokens.len() % batch != 0 {
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

    /// Fresh KV caches for a single sequence.
    pub fn empty_caches(&self) -> Vec<BlockCache> {
        self.empty_caches_batch(1)
    }

    /// Fresh KV caches for `batch` sequences.
    pub fn empty_caches_batch(&self, batch: usize) -> Vec<BlockCache> {
        self.blocks
            .iter()
            .map(|_| BlockCache::new(&self.config, batch))
            .collect()
    }

    /// Runs one token through the model at `position`, updating the KV caches.
    pub fn step_token(
        &self,
        token: usize,
        position: usize,
        caches: &mut [BlockCache],
    ) -> Result<TensorValue, ModelError> {
        self.step_tokens(&[token], &[position], &[true], caches)
    }

    /// Runs one token per batch row, each at its own absolute position.
    ///
    /// `real[row]` says whether the appended token is part of the sequence or
    /// padding; padding tokens are masked out of attention.
    pub fn step_tokens(
        &self,
        tokens: &[usize],
        positions: &[usize],
        real: &[bool],
        caches: &mut [BlockCache],
    ) -> Result<TensorValue, ModelError> {
        for &token in tokens {
            if token >= self.config.vocab_size {
                return Err(ModelError::TokenOutOfRange {
                    index: token,
                    vocab_size: self.config.vocab_size,
                });
            }
        }
        if tokens.is_empty() || tokens.len() != positions.len() || tokens.len() != real.len() {
            return Err(ModelError::EmptyPrompt);
        }
        if caches.len() != self.blocks.len() {
            return Err(ModelError::CacheDepthMismatch {
                expected: self.blocks.len(),
                actual: caches.len(),
            });
        }

        let mut hidden = self.token_embedding.lookup(tokens)?;
        if let Some(position_embedding) = &self.position_embedding {
            hidden = hidden.add(&position_embedding.lookup(positions)?)?;
        }
        for (block, cache) in self.blocks.iter().zip(caches.iter_mut()) {
            cache.attention.set_pending_real(real);
            hidden = block.forward_step(&hidden, positions, cache)?;
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
        self.generate_with(
            prompt,
            max_new_tokens,
            Sampling::with_temperature(temperature),
            rng,
        )
    }

    /// Autoregressive sampling with explicit temperature, top-k and top-p.
    pub fn generate_with(
        &self,
        prompt: &[usize],
        max_new_tokens: usize,
        sampling: Sampling,
        rng: &mut Rng,
    ) -> Result<Vec<usize>, ModelError> {
        let _no_grad = mini_llm_autograd::no_grad::guard();
        let _eval = mini_llm_nn::mode::guard();
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
            let next = sample_next(&logits, self.config.vocab_size, sampling, rng)?;
            tokens.push(next);

            if self.position_embedding.is_some() && tokens.len() >= self.config.max_seq_len {
                break;
            }
            logits = self.step_token(next, tokens.len() - 1, &mut caches)?;
        }
        Ok(tokens)
    }

    /// Generates continuations for several prompts in one batched forward pass.
    ///
    /// The causal mask keeps samples independent; shorter prompts are padded on
    /// the right with token 0, and padding cannot affect a real position because
    /// attention only looks left. Each sequence stops at `max_seq_len`, so keep
    /// `prompt length + max_new_tokens` within that limit. This path does not use
    /// the KV cache: every step recomputes the window (the cached path supports
    /// one sequence).
    pub fn generate_batch(
        &self,
        prompts: &[Vec<usize>],
        max_new_tokens: usize,
        sampling: Sampling,
        rng: &mut Rng,
    ) -> Result<Vec<Vec<usize>>, ModelError> {
        let _no_grad = mini_llm_autograd::no_grad::guard();
        let _eval = mini_llm_nn::mode::guard();
        if prompts.is_empty() {
            return Err(ModelError::EmptyPrompt);
        }
        let mut width = 0usize;
        for prompt in prompts {
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
            width = width.max(prompt.len());
        }

        let batch = prompts.len();
        let mut sequences: Vec<Vec<usize>> = prompts
            .iter()
            .map(|prompt| {
                let mut sequence = prompt.clone();
                sequence.resize(width, 0);
                sequence
            })
            .collect();
        let mut lengths: Vec<usize> = prompts.iter().map(Vec::len).collect();
        let mut active = vec![true; batch];

        for _ in 0..max_new_tokens {
            if !active.iter().any(|flag| *flag) {
                break;
            }

            let inputs: Vec<usize> = sequences.iter().flatten().copied().collect();
            let logits = self.forward(&inputs, batch)?;
            let data = logits.data();
            let sampled_width = width;

            for index in 0..batch {
                if !active[index] {
                    continue;
                }
                let row = (index * sampled_width + lengths[index] - 1) * self.config.vocab_size;
                let last = &data[row..row + self.config.vocab_size];
                let next = sample_from_last(last, sampling, rng)?;

                if lengths[index] == width {
                    for sequence in &mut sequences {
                        sequence.push(0);
                    }
                    width += 1;
                }
                sequences[index][lengths[index]] = next;
                lengths[index] += 1;
                if lengths[index] >= self.config.max_seq_len {
                    active[index] = false;
                }
            }
        }

        Ok(sequences
            .into_iter()
            .zip(lengths)
            .map(|(sequence, length)| sequence[..length].to_vec())
            .collect())
    }

    /// Cached batched generation, one KV cache row per sequence.
    ///
    /// Prompts may have different lengths when the model uses RoPE: shorter
    /// prompts are right-padded with token 0, the padding is masked out of the
    /// attention scores, and each row keeps its own position. With learned
    /// position embeddings all prompts must have the same length.
    pub fn generate_batch_cached(
        &self,
        prompts: &[Vec<usize>],
        max_new_tokens: usize,
        sampling: Sampling,
        rng: &mut Rng,
    ) -> Result<Vec<Vec<usize>>, ModelError> {
        let _no_grad = mini_llm_autograd::no_grad::guard();
        let _eval = mini_llm_nn::mode::guard();
        if prompts.is_empty() {
            return Err(ModelError::EmptyPrompt);
        }
        for prompt in prompts {
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
        }

        let width = prompts.iter().map(Vec::len).max().unwrap_or(0);
        let unequal = prompts.iter().any(|prompt| prompt.len() != width);
        if unequal && self.position_embedding.is_some() {
            return Err(ModelError::SequenceLengthMismatch {
                expected: width,
                actual: prompts
                    .iter()
                    .map(Vec::len)
                    .find(|&length| length != width)
                    .unwrap_or(width),
            });
        }

        let batch = prompts.len();
        let mut caches = self.empty_caches_batch(batch);
        let mut tokens: Vec<Vec<usize>> = prompts.to_vec();
        let mut lengths: Vec<usize> = prompts.iter().map(Vec::len).collect();
        let mut last_logits: Vec<Option<Vec<f32>>> = vec![None; batch];

        // Prefill: short prompts are padded on the right; their padding is
        // masked out by `valid_lengths`.
        for position in 0..width {
            let step: Vec<usize> = prompts
                .iter()
                .map(|prompt| prompt.get(position).copied().unwrap_or(0))
                .collect();
            let positions = vec![position; batch];
            let real: Vec<bool> = lengths.iter().map(|&length| position < length).collect();
            let logits = self.step_tokens(&step, &positions, &real, &mut caches)?;

            let data = logits.data();
            for row in 0..batch {
                if position + 1 == lengths[row] {
                    let start = row * self.config.vocab_size;
                    last_logits[row] = Some(data[start..start + self.config.vocab_size].to_vec());
                }
            }
        }

        for _ in 0..max_new_tokens {
            let mut next_tokens = Vec::with_capacity(batch);
            for logits in &last_logits {
                let logits = logits.as_ref().ok_or(ModelError::EmptyPrompt)?;
                next_tokens.push(sample_from_last(logits, sampling, rng)?);
            }

            if self.position_embedding.is_some() && lengths[0] + 1 > self.config.max_seq_len {
                break;
            }

            for (row, token) in next_tokens.iter().enumerate() {
                tokens[row].push(*token);
                lengths[row] += 1;
            }
            let positions: Vec<usize> = lengths.iter().map(|length| length - 1).collect();
            let real = vec![true; batch];
            let logits = self.step_tokens(&next_tokens, &positions, &real, &mut caches)?;

            let data = logits.data();
            for (row, slot) in last_logits.iter_mut().enumerate() {
                let start = row * self.config.vocab_size;
                *slot = Some(data[start..start + self.config.vocab_size].to_vec());
            }
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
        let _no_grad = mini_llm_autograd::no_grad::guard();
        let _eval = mini_llm_nn::mode::guard();
        if prompt.is_empty() {
            return Err(ModelError::EmptyPrompt);
        }
        let mut tokens = prompt.to_vec();
        for _ in 0..max_new_tokens {
            let start = tokens.len().saturating_sub(self.config.max_seq_len);
            let window = &tokens[start..];
            let logits = self.forward(window, 1)?;
            let next = sample_next(
                &logits,
                self.config.vocab_size,
                Sampling::with_temperature(temperature),
                rng,
            )?;
            tokens.push(next);
        }
        Ok(tokens)
    }
}

impl LanguageModel for TinyGpt {
    fn forward(&self, tokens: &[usize], batch: usize) -> Result<TensorValue, ModelError> {
        TinyGpt::forward(self, tokens, batch)
    }

    fn parameters(&self) -> Vec<TensorValue> {
        TinyGpt::parameters(self)
    }

    fn vocab_size(&self) -> usize {
        self.config.vocab_size
    }

    fn context_len(&self) -> usize {
        self.config.max_seq_len
    }
}

pub fn sample_next(
    logits: &TensorValue,
    vocab: usize,
    sampling: Sampling,
    rng: &mut Rng,
) -> Result<usize, ModelError> {
    let values = logits.value();
    let rows = values.shape()[0];
    let last = &values.data()[(rows - 1) * vocab..rows * vocab];
    sample_from_last(last, sampling, rng)
}

/// Samples one token from a row of next-token logits.
fn sample_from_last(last: &[f32], sampling: Sampling, rng: &mut Rng) -> Result<usize, ModelError> {
    if sampling.temperature <= 0.0 {
        return Ok(argmax(last));
    }
    let vocab = last.len();
    let scaled: Vec<f32> = last
        .iter()
        .map(|value| value / sampling.temperature)
        .collect();
    let probabilities = Tensor::from_2d(1, vocab, scaled)?
        .softmax_rows()?
        .into_data();

    let candidates = truncate(&probabilities, sampling);
    if candidates.len() == probabilities.len() && sampling.top_k.is_none() {
        // Fast path: keep the original order so plain temperature sampling is
        // bit-identical to the pre-top-k implementation.
        return Ok(sample(&probabilities, rng));
    }
    Ok(sample_truncated(&candidates, rng))
}

/// Applies top-k then top-p, returning `(index, probability)` pairs.
fn truncate(probabilities: &[f32], sampling: Sampling) -> Vec<(usize, f32)> {
    let mut items: Vec<(usize, f32)> = probabilities.iter().copied().enumerate().collect();
    if let Some(top_k) = sampling.top_k {
        if top_k < items.len() {
            items.sort_by(|left, right| right.1.total_cmp(&left.1));
            items.truncate(top_k);
        }
    }
    if let Some(top_p) = sampling.top_p {
        if !items.is_empty() {
            items.sort_by(|left, right| right.1.total_cmp(&left.1));
            let mut cumulative = 0.0;
            let mut keep = 1;
            for (index, &(_, probability)) in items.iter().enumerate() {
                cumulative += probability;
                keep = index + 1;
                if cumulative >= top_p {
                    break;
                }
            }
            items.truncate(keep.max(1));
        }
    }
    items
}

fn sample_truncated(candidates: &[(usize, f32)], rng: &mut Rng) -> usize {
    let total: f32 = candidates.iter().map(|(_, probability)| probability).sum();
    if total <= 0.0 {
        return candidates.first().map(|(index, _)| *index).unwrap_or(0);
    }
    let mut threshold = rng.next_f32() * total;
    for &(index, probability) in candidates {
        threshold -= probability;
        if threshold <= 0.0 {
            return index;
        }
    }
    candidates
        .last()
        .map(|(index, _)| *index)
        .unwrap_or_default()
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
