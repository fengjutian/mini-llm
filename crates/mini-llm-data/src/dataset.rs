//! Next-token datasets: plain pretraining windows and supervised (masked)
//! fine-tuning windows.

use mini_llm_random::Rng;

use crate::DataError;

/// One batch of next-token prediction examples, flattened row-major.
#[derive(Debug, Clone, PartialEq)]
pub struct Batch {
    /// `[batch * seq]` token ids fed to the model.
    pub inputs: Vec<usize>,
    /// `[batch * seq]` token ids shifted one to the right.
    pub targets: Vec<usize>,
    /// `[batch * seq]` per-target loss weights; `0.0` masks a position out.
    pub weights: Vec<f32>,
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

    /// Number of positions that actually contribute to the loss.
    pub fn active_positions(&self) -> usize {
        self.weights.iter().filter(|weight| **weight > 0.0).count()
    }
}

/// Something that can produce training batches of a fixed context length.
pub trait Dataset {
    fn block_size(&self) -> usize;

    fn sample_batch(&self, batch_size: usize, rng: &mut Rng) -> Result<Batch, DataError>;
}

/// Pretraining: every target position contributes equally.
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

    /// Number of distinct window start positions.
    pub fn num_windows(&self) -> usize {
        self.tokens.len() - self.block_size
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
        let weights = vec![1.0f32; targets.len()];
        Ok(Batch {
            inputs,
            targets,
            weights,
            batch: starts.len(),
            seq: self.block_size,
        })
    }
}

impl Dataset for TextDataset {
    fn block_size(&self) -> usize {
        self.block_size
    }

    fn sample_batch(&self, batch_size: usize, rng: &mut Rng) -> Result<Batch, DataError> {
        if batch_size == 0 {
            return Err(DataError::EmptyBatch);
        }
        let windows = self.num_windows();
        let starts: Vec<usize> = (0..batch_size)
            .map(|_| (rng.next_f32() * windows as f32) as usize)
            .collect();
        self.batch_at(&starts)
    }
}

/// Supervised fine-tuning: only response tokens contribute to the loss.
///
/// `weights[i]` is the weight of token `i` as a *target*, so the prompt prefix
/// must be zeroed by the caller (see
/// [`CharTokenizer::encode_supervised`](crate::CharTokenizer::encode_supervised)).
pub struct SftDataset {
    tokens: Vec<usize>,
    weights: Vec<f32>,
    block_size: usize,
    /// Window starts that contain at least one supervised target, precomputed
    /// so sampling never has to retry.
    supervised_starts: Vec<usize>,
}

impl SftDataset {
    pub fn new(
        tokens: Vec<usize>,
        weights: Vec<f32>,
        block_size: usize,
    ) -> Result<Self, DataError> {
        if tokens.is_empty() {
            return Err(DataError::EmptyCorpus);
        }
        if tokens.len() != weights.len() {
            return Err(DataError::MaskLengthMismatch {
                tokens: tokens.len(),
                weights: weights.len(),
            });
        }
        if block_size == 0 || tokens.len() <= block_size {
            return Err(DataError::BlockTooLarge {
                block_size,
                tokens: tokens.len(),
            });
        }

        let supervised_starts = supervised_window_starts(&weights, block_size);
        if supervised_starts.is_empty() {
            return Err(DataError::EmptyMaskWindow);
        }

        Ok(Self {
            tokens,
            weights,
            block_size,
            supervised_starts,
        })
    }

    pub fn num_windows(&self) -> usize {
        self.tokens.len() - self.block_size
    }

    /// Window starts that actually carry a learning signal.
    pub fn supervised_windows(&self) -> usize {
        self.supervised_starts.len()
    }

    /// Builds a batch from explicit window starts (useful for tests).
    ///
    /// A window whose targets are all masked out carries no learning signal and
    /// is rejected rather than silently producing a zero loss.
    pub fn batch_at(&self, starts: &[usize]) -> Result<Batch, DataError> {
        if starts.is_empty() {
            return Err(DataError::EmptyBatch);
        }
        let windows = self.num_windows();
        let mut inputs = Vec::with_capacity(starts.len() * self.block_size);
        let mut targets = Vec::with_capacity(starts.len() * self.block_size);
        let mut weights = Vec::with_capacity(starts.len() * self.block_size);

        for &start in starts {
            if start >= windows {
                return Err(DataError::WindowOutOfRange { start, windows });
            }
            let end = start + self.block_size;
            inputs.extend_from_slice(&self.tokens[start..end]);
            targets.extend_from_slice(&self.tokens[start + 1..=end]);
            weights.extend_from_slice(&self.weights[start + 1..=end]);
        }

        if weights.iter().all(|weight| *weight == 0.0) {
            return Err(DataError::EmptyMaskWindow);
        }

        Ok(Batch {
            inputs,
            targets,
            weights,
            batch: starts.len(),
            seq: self.block_size,
        })
    }
}

impl Dataset for SftDataset {
    fn block_size(&self) -> usize {
        self.block_size
    }

    fn sample_batch(&self, batch_size: usize, rng: &mut Rng) -> Result<Batch, DataError> {
        if batch_size == 0 {
            return Err(DataError::EmptyBatch);
        }
        let candidates = self.supervised_starts.len();
        let starts: Vec<usize> = (0..batch_size)
            .map(|_| self.supervised_starts[(rng.next_f32() * candidates as f32) as usize])
            .collect();
        self.batch_at(&starts)
    }
}

/// Prefix-sum scan for windows whose targets contain a positive weight.
fn supervised_window_starts(weights: &[f32], block_size: usize) -> Vec<usize> {
    let mut prefix = vec![0usize; weights.len() + 1];
    for (index, &weight) in weights.iter().enumerate() {
        prefix[index + 1] = prefix[index] + usize::from(weight > 0.0);
    }

    let windows = weights.len() - block_size;
    (0..windows)
        .filter(|&start| prefix[start + 1 + block_size] - prefix[start + 1] > 0)
        .collect()
}
