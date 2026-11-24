//! Byte-level byte-pair encoding (BPE).
//!
//! The vocabulary starts as the 256 possible bytes and grows by repeatedly
//! merging the most frequent adjacent pair in the training text. Encoding
//! replays the merges in the order they were learned, so any UTF-8 input is
//! representable and unknown characters never fail.

use std::cmp::Reverse;
use std::collections::{BinaryHeap, HashMap};

use crate::DataError;

/// Ids below this are the raw bytes.
const BASE: u32 = 256;

/// A byte-level BPE vocabulary defined by its ordered merge list.
#[derive(Debug, Clone, PartialEq, Default)]
pub struct BpeTokenizer {
    merges: Vec<(u32, u32)>,
    /// Derived from `merges` once, so encoding does not rebuild it per call.
    ranks: HashMap<(u32, u32), u32>,
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

        Self::from_merges(merges).expect("training produces valid merges")
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
        let ranks = merges
            .iter()
            .enumerate()
            .map(|(rank, &pair)| (pair, rank as u32))
            .collect();
        Ok(Self { merges, ranks })
    }

    pub fn merges(&self) -> &[(u32, u32)] {
        &self.merges
    }

    pub fn vocab_size(&self) -> usize {
        BASE as usize + self.merges.len()
    }

    /// Encodes `text` by repeatedly applying the lowest-ranked available merge.
    ///
    /// The sequence is kept as a doubly linked list and candidate merges live in
    /// a min-heap keyed by merge rank, so this is O(n log n) rather than
    /// rescanning the whole sequence per merge.
    pub fn encode(&self, text: &str) -> Vec<usize> {
        let bytes: Vec<u32> = text.bytes().map(u32::from).collect();
        if bytes.len() < 2 || self.merges.is_empty() {
            return bytes.into_iter().map(|byte| byte as usize).collect();
        }

        let ranks = &self.ranks;
        let mut symbols: Vec<Symbol> = bytes
            .iter()
            .enumerate()
            .map(|(index, &id)| Symbol {
                id,
                prev: index.checked_sub(1),
                next: (index + 1 < bytes.len()).then_some(index + 1),
                alive: true,
            })
            .collect();

        let mut heap: BinaryHeap<Reverse<(u32, usize)>> = BinaryHeap::new();
        for left in 0..symbols.len() - 1 {
            push_candidate(&symbols, ranks, &mut heap, left, left + 1);
        }

        while let Some(Reverse((rank, left))) = heap.pop() {
            if !symbols[left].alive {
                continue;
            }
            let Some(right) = symbols[left].next else {
                continue;
            };
            if !symbols[right].alive
                || ranks.get(&(symbols[left].id, symbols[right].id)) != Some(&rank)
            {
                continue;
            }

            symbols[left].id = BASE + rank;
            symbols[right].alive = false;
            let after = symbols[right].next;
            symbols[left].next = after;
            if let Some(after) = after {
                symbols[after].prev = Some(left);
            }

            if let Some(before) = symbols[left].prev {
                push_candidate(&symbols, ranks, &mut heap, before, left);
            }
            if let Some(after) = symbols[left].next {
                push_candidate(&symbols, ranks, &mut heap, left, after);
            }
        }

        let mut output = Vec::with_capacity(symbols.len());
        let mut cursor = Some(0);
        while let Some(index) = cursor {
            output.push(symbols[index].id as usize);
            cursor = symbols[index].next;
        }
        output
    }

    /// Decodes token ids back into text.
    ///
    /// Byte-level tokens can split a multi-byte UTF-8 character, so incomplete
    /// sequences are replaced with U+FFFD instead of failing; encoding the same
    /// text back and forth is always exact.
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
        Ok(String::from_utf8_lossy(&bytes).into_owned())
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

/// One node of the linked list used during encoding.
struct Symbol {
    id: u32,
    prev: Option<usize>,
    next: Option<usize>,
    alive: bool,
}

/// Pushes `(left, right)` as a merge candidate when that pair has a rank.
fn push_candidate(
    symbols: &[Symbol],
    ranks: &HashMap<(u32, u32), u32>,
    heap: &mut BinaryHeap<Reverse<(u32, usize)>>,
    left: usize,
    right: usize,
) {
    if !symbols[left].alive || !symbols[right].alive {
        return;
    }
    if let Some(&rank) = ranks.get(&(symbols[left].id, symbols[right].id)) {
        heap.push(Reverse((rank, left)));
    }
}
