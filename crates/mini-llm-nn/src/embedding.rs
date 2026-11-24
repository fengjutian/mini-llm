//! An embedding table: token id -> learned vector.

use mini_llm_autograd::TensorValue;

use crate::NnError;
use mini_llm_random::Rng;

/// A lookup table of shape `[vocab_size, embedding_dim]`.
pub struct Embedding {
    weight: TensorValue,
    vocab_size: usize,
    embedding_dim: usize,
}

impl Embedding {
    /// Creates a table initialized with `N(0, std)`.
    pub fn new(
        vocab_size: usize,
        embedding_dim: usize,
        std: f32,
        rng: &mut Rng,
    ) -> Result<Self, NnError> {
        if vocab_size == 0 {
            return Err(NnError::InvalidDimension {
                name: "vocab_size",
                value: vocab_size,
            });
        }
        if embedding_dim == 0 {
            return Err(NnError::InvalidDimension {
                name: "embedding_dim",
                value: embedding_dim,
            });
        }
        let weight = TensorValue::param(rng.normal_tensor(vec![vocab_size, embedding_dim], std)?);
        Ok(Self {
            weight,
            vocab_size,
            embedding_dim,
        })
    }

    /// Looks up `tokens`, producing a `[tokens.len(), embedding_dim]` tensor.
    ///
    /// Repeated tokens are allowed; their gradients accumulate in the table.
    pub fn lookup(&self, tokens: &[usize]) -> Result<TensorValue, NnError> {
        if let Some(&index) = tokens.iter().find(|&&index| index >= self.vocab_size) {
            return Err(NnError::TokenOutOfRange {
                index,
                vocab_size: self.vocab_size,
            });
        }
        Ok(self.weight.gather_rows(tokens)?)
    }

    pub fn weight(&self) -> &TensorValue {
        &self.weight
    }

    pub fn vocab_size(&self) -> usize {
        self.vocab_size
    }

    pub fn embedding_dim(&self) -> usize {
        self.embedding_dim
    }

    pub fn parameters(&self) -> Vec<TensorValue> {
        vec![self.weight.clone()]
    }

    pub fn summary(&self) -> String {
        format!(
            "embedding({} x {}) ({} params)",
            self.vocab_size,
            self.embedding_dim,
            self.vocab_size * self.embedding_dim
        )
    }
}
