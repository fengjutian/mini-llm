//! # mini-llm-data
//!
//! Tokenization and next-token batch sampling.
//!
//! [`CharTokenizer`] maps every distinct character to one token (sorted by code
//! point, so the vocabulary is deterministic). [`BpeTokenizer`] is a byte-level
//! byte-pair encoder: it starts from the 256 bytes and learns merges from the
//! corpus, so any UTF-8 text is representable. [`Tokenizer`] dispatches over
//! both and is what checkpoints store.
//!
//! ```no_run
//! use mini_llm_data::{CharTokenizer, Dataset, TextDataset};
//! use mini_llm_random::Rng;
//!
//! let tokenizer = CharTokenizer::from_text("hello world");
//! let tokens = tokenizer.encode("hello").unwrap();
//! let dataset = TextDataset::new(tokens, 4).unwrap();
//! let mut rng = Rng::new(1);
//! let batch = dataset.sample_batch(2, &mut rng).unwrap();
//! assert_eq!(batch.inputs.len(), 2 * 4);
//! ```
//!
//! ```no_run
//! use mini_llm_data::BpeTokenizer;
//!
//! let tokenizer = BpeTokenizer::train("the quick brown fox", 300);
//! let tokens = tokenizer.encode("quick fox");
//! assert_eq!(tokenizer.decode(&tokens).unwrap(), "quick fox");
//! ```

mod bpe;
mod dataset;
mod instructions;
mod tokenizer;

pub use bpe::BpeTokenizer;
pub use instructions::{parse_instructions, InstructionPair};
pub use dataset::{Batch, Dataset, SftDataset, TextDataset};
pub use tokenizer::{CharTokenizer, Tokenizer};

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
    /// Per-token weights and tokens have different lengths.
    MaskLengthMismatch { tokens: usize, weights: usize },
    /// A sampled window has no supervised (unmasked) target.
    EmptyMaskWindow,
    /// A supervised example needs a non-empty response to learn from.
    EmptyResponse,
    /// A BPE merge refers to a token id that does not exist yet.
    InvalidMerge { index: usize },
    /// A JSONL instruction line could not be parsed.
    InvalidInstruction { line: usize, reason: &'static str },
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
            Self::MaskLengthMismatch { tokens, weights } => {
                write!(f, "{} tokens but {} weights", tokens, weights)
            }
            Self::EmptyMaskWindow => write!(f, "window contains no supervised tokens"),
            Self::EmptyResponse => write!(f, "supervised example needs a non-empty response"),
            Self::InvalidMerge { index } => write!(
                f,
                "BPE merge {} refers to a token id that does not exist yet",
                index
            ),
            Self::InvalidInstruction { line, reason } => {
                write!(f, "instruction line {} is invalid: {}", line, reason)
            }
        }
    }
}

impl Error for DataError {}
