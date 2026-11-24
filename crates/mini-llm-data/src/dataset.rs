//! Next-token windows sampled from a token stream.

use mini_llm_random::Rng;

use crate::DataError;

/// One batch of next-token prediction examples, flattened row-major.
#[derive(Debug, Clone, PartialEq)]
pub struct Batch {
    /// `[batch * seq]` token ids fed to the model.
    pub inputs: Vec<usize>,
    /// `[batch * seq]` token ids shifted one to the right.
    pub targets: Vec<usize>,
    pub batch: usize,
    pub seq: usize,
}

impl Batch {
    pub fn len(&self) -> usize {
        self.inputs.len()
    }

    pub fn is_empty(&self) -> bool {
        self.inputs.is_empty()
    }
}

/// A token stream that can sample fixed-length training windows.
pub struct TextDataset {
    tokens: Vec<usize>,
    block_size: usize,
}

impl TextDataset {
    /// `block_size` is the context length; the corpus must be at least one
    /// token longer so every window can produce a shifted target.
    pub fn new(tokens: Vec<usize>, block_size: usize) -> Result<Self, DataError> {
        if tokens.is_empty() {
            return Err(DataError::EmptyCorpus);
        }
        if block_size == 0 || tokens.len() <= block_size {
            return Err(DataError::BlockTooLarge {
                block_size,
                tokens: tokens.len(),
            });
        }
        Ok(Self { tokens, block_size })
    }

    pub fn tokens(&self) -> &[usize] {
        &self.tokens
    }

    pub fn block_size(&self) -> usize {
        self.block_size
    }

    /// Number of distinct window start positions.
    pub fn num_windows(&self) -> usize {
        self.tokens.len() - self.block_size
    }

    /// Samples `batch_size` random windows using `rng`.
    pub fn sample_batch(&self, batch_size: usize, rng: &mut Rng) -> Result<Batch, DataError> {
        if batch_size == 0 {
            return Err(DataError::EmptyBatch);
        }
        let windows = self.num_windows();
        let starts: Vec<usize> = (0..batch_size)
            .map(|_| (rng.next_f32() * windows as f32) as usize)
            .collect();
        self.batch_at(&starts)
    }

    /// Builds a batch from explicit window starts (useful for tests).
    pub fn batch_at(&self, starts: &[usize]) -> Result<Batch, DataError> {
        if starts.is_empty() {
            return Err(DataError::EmptyBatch);
        }
        let windows = self.num_windows();
        let mut inputs = Vec::with_capacity(starts.len() * self.block_size);
        let mut targets = Vec::with_capacity(starts.len() * self.block_size);
        for &start in starts {
            if start >= windows {
                return Err(DataError::WindowOutOfRange { start, windows });
            }
            let end = start + self.block_size;
            inputs.extend_from_slice(&self.tokens[start..end]);
            targets.extend_from_slice(&self.tokens[start + 1..=end]);
        }
        Ok(Batch {
            inputs,
            targets,
            batch: starts.len(),
            seq: self.block_size,
        })
    }
}
