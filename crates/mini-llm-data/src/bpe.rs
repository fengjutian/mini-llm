//! Byte-level byte-pair encoding (BPE).
//!
//! The vocabulary starts as the 256 possible bytes and grows by repeatedly
//! merging the most frequent adjacent pair in the training text. Encoding
//! replays the merges in the order they were learned, so any UTF-8 input is
//! representable and unknown characters never fail.

use std::collections::HashMap;

use crate::DataError;

/// Ids below this are the raw bytes.
const BASE: u32 = 256;

/// A byte-level BPE vocabulary defined by its ordered merge list.
#[derive(Debug, Clone, PartialEq, Default)]
pub struct BpeTokenizer {
    merges: Vec<(u32, u32)>,
}

impl BpeTokenizer {
    /// Learns merges from `text` until the vocabulary reaches `vocab_size`.
    ///
    /// Stops early when no pair occurs at least twice, or when the vocabulary
    /// would exceed the number of bytes in the corpus.
    pub fn train(text: &str, vocab_size: usize) -> Self {
        let mut tokens: Vec<u32> = text.bytes().map(u32::from).collect();
        let target = vocab_size.max(BASE as usize + 1);
        let mut merges = Vec::new();

        while BASE as usize + merges.len() < target {
            let Some(pair) = most_frequent_pair(&tokens) else {
                break;
            };
            let new_id = BASE + merges.len() as u32;
            merges.push(pair);
            tokens = merge_pair(&tokens, pair, new_id);
        }

        Self { merges }
    }

    /// Rebuilds a tokenizer from merges, validating that each merge only refers
    /// to ids that already exist.
    pub fn from_merges(merges: Vec<(u32, u32)>) -> Result<Self, DataError> {
        for (index, &(left, right)) in merges.iter().enumerate() {
            let next = BASE + index as u32;
            if left >= next || right >= next {
                return Err(DataError::InvalidMerge { index });
            }
        }
        Ok(Self { merges })
    }

    pub fn merges(&self) -> &[(u32, u32)] {
        &self.merges
    }

    pub fn vocab_size(&self) -> usize {
        BASE as usize + self.merges.len()
    }

    fn ranks(&self) -> HashMap<(u32, u32), u32> {
        self.merges
            .iter()
            .enumerate()
            .map(|(rank, &pair)| (pair, rank as u32))
            .collect()
    }

    /// Encodes `text` by repeatedly applying the lowest-ranked available merge.
    pub fn encode(&self, text: &str) -> Vec<usize> {
        let ranks = self.ranks();
        let mut tokens: Vec<u32> = text.bytes().map(u32::from).collect();

        loop {
            let mut best: Option<(u32, (u32, u32))> = None;
            for window in tokens.windows(2) {
                if let Some(&rank) = ranks.get(&(window[0], window[1])) {
                    if best.is_none_or(|(best_rank, _)| rank < best_rank) {
                        best = Some((rank, (window[0], window[1])));
                    }
                }
            }
            let Some((rank, pair)) = best else {
                break;
            };
            tokens = merge_pair(&tokens, pair, BASE + rank);
        }

        tokens.into_iter().map(|token| token as usize).collect()
    }

    /// Decodes token ids back into UTF-8 text.
    pub fn decode(&self, tokens: &[usize]) -> Result<String, DataError> {
        let mut bytes = Vec::new();
        for &token in tokens {
            let id = token as u32;
            if id >= BASE + self.merges.len() as u32 {
                return Err(DataError::TokenOutOfRange {
                    token,
                    vocab_size: self.vocab_size(),
                });
            }
            self.expand(id, &mut bytes);
        }
        String::from_utf8(bytes).map_err(|_| DataError::InvalidUtf8)
    }

    fn expand(&self, id: u32, out: &mut Vec<u8>) {
        if id < BASE {
            out.push(id as u8);
            return;
        }
        let (left, right) = self.merges[(id - BASE) as usize];
        self.expand(left, out);
        self.expand(right, out);
    }
}

/// Replaces every non-overlapping occurrence of `pair` with `new_id`.
fn merge_pair(tokens: &[u32], pair: (u32, u32), new_id: u32) -> Vec<u32> {
    let mut merged = Vec::with_capacity(tokens.len());
    let mut index = 0;
    while index < tokens.len() {
        if index + 1 < tokens.len() && tokens[index] == pair.0 && tokens[index + 1] == pair.1 {
            merged.push(new_id);
            index += 2;
        } else {
            merged.push(tokens[index]);
            index += 1;
        }
    }
    merged
}

/// Most frequent pair occurring at least twice; ties break on the smaller pair
/// so training is deterministic.
fn most_frequent_pair(tokens: &[u32]) -> Option<(u32, u32)> {
    let mut counts: HashMap<(u32, u32), usize> = HashMap::new();
    for window in tokens.windows(2) {
        *counts.entry((window[0], window[1])).or_default() += 1;
    }
    counts
        .into_iter()
        .filter(|(_, count)| *count >= 2)
        .max_by(|(left_pair, left_count), (right_pair, right_count)| {
            left_count
                .cmp(right_count)
                .then_with(|| right_pair.cmp(left_pair))
        })
        .map(|(pair, _)| pair)
}
