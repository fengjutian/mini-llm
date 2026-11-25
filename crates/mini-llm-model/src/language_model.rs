//! The model-level abstraction the training loop depends on.
//!
//! `mini-llm-train`'s `Trainer` is generic over [`LanguageModel`], so the loop
//! only needs logits, parameters and two shape numbers; swapping in a different
//! architecture does not touch the training code.

use mini_llm_autograd::TensorValue;

use crate::ModelError;

/// What a trainer needs from a language model.
///
/// [`TinyGpt`](crate::TinyGpt) is the reference implementation. Any other
/// architecture qualifies as long as it can produce next-token logits from a
/// flat token buffer and list its trainable tensors in a stable order.
pub trait LanguageModel {
    /// Logits of shape `[batch * seq, vocab_size]` for `tokens` shaped
    /// `[batch, seq]` flattened row-major.
    fn forward(&self, tokens: &[usize], batch: usize) -> Result<TensorValue, ModelError>;

    /// Every learnable tensor, in a stable order.
    ///
    /// The optimizer mutates these handles in place, and the checkpoint writer
    /// serializes them in this order, so the order is part of the contract.
    fn parameters(&self) -> Vec<TensorValue>;

    /// Width of the logits axis.
    fn vocab_size(&self) -> usize;

    /// Longest sequence the model accepts, in tokens.
    fn context_len(&self) -> usize;

    /// Total number of scalar parameters; walks [`Self::parameters`] by default.
    fn num_parameters(&self) -> usize {
        self.parameters()
            .iter()
            .map(|parameter| parameter.value().len())
            .sum()
    }
}
