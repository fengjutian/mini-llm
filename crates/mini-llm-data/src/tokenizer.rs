//! A character-level tokenizer.

use std::collections::{BTreeSet, HashMap};

use crate::{BpeTokenizer, DataError};

/// Maps characters to contiguous ids and back.
#[derive(Debug, Clone, PartialEq)]
pub struct CharTokenizer {
    characters: Vec<char>,
    /// Character -> id, so encoding is O(1) per character instead of a scan.
    index: HashMap<char, usize>,
}

impl CharTokenizer {
    /// Builds a vocabulary from every distinct character in `text`.
    ///
    /// Characters are sorted by code point so the mapping is deterministic.
    pub fn from_text(text: &str) -> Self {
        let characters: Vec<char> = text.chars().collect::<BTreeSet<_>>().into_iter().collect();
        Self::build(characters)
    }

    /// Builds a vocabulary from an explicit character list (e.g. a checkpoint).
    pub fn from_characters(characters: Vec<char>) -> Result<Self, DataError> {
        if characters.is_empty() {
            return Err(DataError::EmptyCorpus);
        }
        let unique: BTreeSet<char> = characters.iter().copied().collect();
        if unique.len() != characters.len() {
            return Err(DataError::EmptyCorpus);
        }
        Ok(Self::build(characters))
    }

    fn build(characters: Vec<char>) -> Self {
        let index = characters
            .iter()
            .enumerate()
            .map(|(id, &character)| (character, id))
            .collect();
        Self { characters, index }
    }

    pub fn vocab_size(&self) -> usize {
        self.characters.len()
    }

    pub fn characters(&self) -> &[char] {
        &self.characters
    }

    /// Encodes `text`, failing on characters that are not in the vocabulary.
    pub fn encode(&self, text: &str) -> Result<Vec<usize>, DataError> {
        text.chars()
            .map(|character| {
                self.index
                    .get(&character)
                    .copied()
                    .ok_or(DataError::UnknownCharacter { character })
            })
            .collect()
    }

    /// Encodes token ids back into text.
    pub fn decode(&self, tokens: &[usize]) -> Result<String, DataError> {
        let mut text = String::new();
        for &token in tokens {
            let character = self
                .characters
                .get(token)
                .ok_or(DataError::TokenOutOfRange {
                    token,
                    vocab_size: self.characters.len(),
                })?;
            text.push(*character);
        }
        Ok(text)
    }

    /// Encodes `prompt` followed by `response` and returns a loss mask.
    pub fn encode_supervised(
        &self,
        prompt: &str,
        response: &str,
    ) -> Result<(Vec<usize>, Vec<f32>), DataError> {
        build_supervised(self.encode(prompt)?, self.encode(response)?)
    }
}

/// Concatenates prompt and response tokens and masks the prompt out of the loss.
///
/// Predicting the first response token is supervised by the last prompt
/// position, so masking starts one token before the response; the final token
/// has no successor and is also masked.
fn build_supervised(
    prompt_tokens: Vec<usize>,
    response_tokens: Vec<usize>,
) -> Result<(Vec<usize>, Vec<f32>), DataError> {
    if response_tokens.is_empty() {
        return Err(DataError::EmptyResponse);
    }

    let prompt_len = prompt_tokens.len();
    let mut tokens = prompt_tokens;
    tokens.extend(response_tokens);

    let mut weights = vec![0.0f32; tokens.len()];
    let first_supervised = prompt_len.saturating_sub(1);
    for weight in weights.iter_mut().skip(first_supervised) {
        *weight = 1.0;
    }
    if let Some(last) = weights.last_mut() {
        *last = 0.0;
    }
    Ok((tokens, weights))
}

/// A vocabulary that is either character-level or byte-level BPE.
#[derive(Debug, Clone, PartialEq)]
pub enum Tokenizer {
    Char(CharTokenizer),
    Bpe(BpeTokenizer),
}

impl Tokenizer {
    /// Encodes `text` with the active vocabulary.
    pub fn encode(&self, text: &str) -> Result<Vec<usize>, DataError> {
        match self {
            Self::Char(tokenizer) => tokenizer.encode(text),
            Self::Bpe(tokenizer) => Ok(tokenizer.encode(text)),
        }
    }

    /// Decodes token ids back into text.
    pub fn decode(&self, tokens: &[usize]) -> Result<String, DataError> {
        match self {
            Self::Char(tokenizer) => tokenizer.decode(tokens),
            Self::Bpe(tokenizer) => tokenizer.decode(tokens),
        }
    }

    pub fn vocab_size(&self) -> usize {
        match self {
            Self::Char(tokenizer) => tokenizer.vocab_size(),
            Self::Bpe(tokenizer) => tokenizer.vocab_size(),
        }
    }

    /// Encodes a prompt/response pair and returns the loss mask.
    ///
    /// Works for both vocabularies, so SFT can use a BPE checkpoint.
    pub fn encode_supervised(
        &self,
        prompt: &str,
        response: &str,
    ) -> Result<(Vec<usize>, Vec<f32>), DataError> {
        build_supervised(self.encode(prompt)?, self.encode(response)?)
    }

    pub fn as_bpe(&self) -> Option<&BpeTokenizer> {
        match self {
            Self::Bpe(tokenizer) => Some(tokenizer),
            Self::Char(_) => None,
        }
    }

    pub fn as_char(&self) -> Option<&CharTokenizer> {
        match self {
            Self::Char(tokenizer) => Some(tokenizer),
            Self::Bpe(_) => None,
        }
    }
}

impl From<CharTokenizer> for Tokenizer {
    fn from(value: CharTokenizer) -> Self {
        Self::Char(value)
    }
}

impl From<BpeTokenizer> for Tokenizer {
    fn from(value: BpeTokenizer) -> Self {
        Self::Bpe(value)
    }
}
