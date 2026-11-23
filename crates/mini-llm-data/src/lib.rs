//! # mini-llm-data
//!
//! Character-level tokenization and next-token batch sampling.
//!
//! The tokenizer is intentionally trivial: every distinct character in the
//! corpus becomes one token, sorted by code point so the vocabulary is
//! deterministic and independent of iteration order.
//!
//! ```no_run
//! use mini_llm_data::{CharTokenizer, TextDataset};
//! use mini_llm_nn::Rng;
//!
//! let tokenizer = CharTokenizer::from_text("hello world");
//! let tokens = tokenizer.encode("hello").unwrap();
//! let dataset = TextDataset::new(tokens, 4).unwrap();
//! let mut rng = Rng::new(1);
//! let batch = dataset.sample_batch(2, &mut rng).unwrap();
//! assert_eq!(batch.inputs.len(), 2 * 4);
//! ```

mod dataset;
mod tokenizer;

pub use dataset::{Batch, TextDataset};
pub use tokenizer::CharTokenizer;

use std::error::Error;
use std::fmt::{Display, Formatter};

#[derive(Debug, Clone, PartialEq)]
pub enum DataError {
    /// The corpus produced no tokens.
    EmptyCorpus,
    /// The requested context window is larger than the corpus.
    BlockTooLarge { block_size: usize, tokens: usize },
    /// A character is not part of the vocabulary.
    UnknownCharacter { character: char },
    /// A token id is not part of the vocabulary.
    TokenOutOfRange { token: usize, vocab_size: usize },
    /// A batch must contain at least one sample.
    EmptyBatch,
    /// A window start index is out of range.
    WindowOutOfRange { start: usize, windows: usize },
}

impl Display for DataError {
    fn fmt(&self, f: &mut Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::EmptyCorpus => write!(f, "corpus must contain at least one character"),
            Self::BlockTooLarge { block_size, tokens } => write!(
                f,
                "block size {} needs at least {} tokens, corpus has {}",
                block_size,
                block_size + 1,
                tokens
            ),
            Self::UnknownCharacter { character } => {
                write!(f, "character {:?} is not in the vocabulary", character)
            }
            Self::TokenOutOfRange { token, vocab_size } => write!(
                f,
                "token id {} is outside the vocabulary of size {}",
                token, vocab_size
            ),
            Self::EmptyBatch => write!(f, "batch size must be greater than zero"),
            Self::WindowOutOfRange { start, windows } => write!(
                f,
                "window start {} is out of range for {} windows",
                start, windows
            ),
        }
    }
}

impl Error for DataError {}
