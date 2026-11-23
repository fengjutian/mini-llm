//! Tiny GPT: a character-level decoder-only Transformer.
//!
//! The model is deliberately small and readable. It uses learned token and
//! position embeddings, pre-norm blocks, multi-head causal attention and a
//! tied-free language-model head.

use mini_llm_autograd::TensorValue;
use mini_llm_nn::{Embedding, LayerNorm, Linear, Module, Rng};
use mini_llm_tensor::Tensor;

use crate::attention::causal_batch_mask;
use crate::block::TransformerBlock;
use crate::{ModelConfig, ModelError};

/// A decoder-only Transformer language model.
pub struct TinyGpt {
    config: ModelConfig,
    token_embedding: Embedding,
    position_embedding: Embedding,
    blocks: Vec<TransformerBlock>,
    final_norm: LayerNorm,
    lm_head: Linear,
}

impl TinyGpt {
    /// Builds the model with deterministic initialization from `rng`.
    pub fn new(config: ModelConfig, rng: &mut Rng) -> Result<Self, ModelError> {
        config.validate()?;
        let hidden = config.hidden_size;

        let token_embedding = Embedding::new(config.vocab_size, hidden, 0.02, rng)?;
        let position_embedding = Embedding::new(config.max_seq_len, hidden, 0.01, rng)?;

        let mut blocks = Vec::with_capacity(config.num_layers);
        for _ in 0..config.num_layers {
            blocks.push(TransformerBlock::new(&config, rng)?);
        }

        let final_norm = LayerNorm::new(hidden)?;
        let lm_head = Linear::without_bias(hidden, config.vocab_size, rng)?;

        Ok(Self {
            config,
            token_embedding,
            position_embedding,
            blocks,
            final_norm,
            lm_head,
        })
    }

    pub fn config(&self) -> &ModelConfig {
        &self.config
    }

    pub fn parameters(&self) -> Vec<TensorValue> {
        let mut parameters = self.token_embedding.parameters();
        parameters.extend(self.position_embedding.parameters());
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
        let embedded = self
            .token_embedding
            .lookup(tokens)?
            .add(&self.position_embedding.lookup(&positions)?)?;

        let mask = TensorValue::new(Tensor::new(
            causal_batch_mask(batch, seq),
            vec![batch * seq, batch * seq],
        )?);

        let mut hidden = embedded;
        for block in &self.blocks {
            hidden = block.forward(&hidden, batch, seq, &mask)?;
        }
        let normalized = self.final_norm.forward(&hidden)?;
        Ok(self.lm_head.forward(&normalized)?)
    }

    /// Autoregressive sampling. The context is truncated to `max_seq_len`.
    ///
    /// `temperature == 0` is treated as greedy-decoding (`argmax`).
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

        let mut tokens = prompt.to_vec();
        for _ in 0..max_new_tokens {
            let start = tokens.len().saturating_sub(self.config.max_seq_len);
            let window = &tokens[start..];
            let logits = self.forward(window, 1)?;
            let rows = window.len();
            let vocab = self.config.vocab_size;
            let logits_tensor = logits.value();
            let last = &logits_tensor.data()[(rows - 1) * vocab..rows * vocab];

            let next = if temperature <= 0.0 {
                argmax(last)
            } else {
                let scaled: Vec<f32> = last.iter().map(|value| value / temperature).collect();
                let probabilities = Tensor::from_2d(1, vocab, scaled)?
                    .softmax_rows()?
                    .into_data();
                sample(&probabilities, rng)
            };
            tokens.push(next);
        }
        Ok(tokens)
    }
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
