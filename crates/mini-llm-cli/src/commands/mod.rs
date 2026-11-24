//! One module per command, so each demo owns its imports and dependencies.

pub mod autograd;
pub mod basics;
pub mod classify;
pub mod dpo;
pub mod generate;
pub mod grpo;
pub mod lora;
pub mod moe;
pub mod regression;
pub mod sft;
pub mod trace;
pub mod train;
pub mod xor;

/// Where `train` writes and `generate` reads the Tiny GPT checkpoint.
pub(crate) const CHECKPOINT_PATH: &str = "checkpoints/tiny-gpt.bin";

/// A short public-domain-style paragraph used for the character-level demo.
pub(crate) const TINY_CORPUS: &str = "\
The little language model learns one character at a time. It reads a short story, \
predicts the next letter, and slowly discovers words, spaces and punctuation. \
Small models can still surprise you: they learn rhythm before meaning, and grammar \
before grace. Training is cheap here, so experiments are fast. The model does not \
know what a word is; it only knows which letter tends to follow which. That is \
enough to begin.";
