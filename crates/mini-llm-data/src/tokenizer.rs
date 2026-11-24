//! A character-level tokenizer.

use std::collections::BTreeSet;

use crate::DataError;

/// Maps characters to contiguous ids and back.
#[derive(Debug, Clone, PartialEq)]
pub struct CharTokenizer {
    characters: Vec<char>,
}

impl CharTokenizer {
    /// Builds a vocabulary from every distinct character in `text`.
    ///
    /// Characters are sorted by code point so the mapping is deterministic.
    pub fn from_text(text: &str) -> Self {
        let characters: Vec<char> = text.chars().collect::<BTreeSet<_>>().into_iter().collect();
        Self { characters }
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
        Ok(Self { characters })
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
                self.characters
                    .iter()
                    .position(|&candidate| candidate == character)
                    .ok_or(DataError::UnknownCharacter { character })
            })
            .collect()
    }

    /// Decodes token ids back into text.
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
    ///
    /// `weights[i]` is `1.0` when the target following token `i` belongs to the
    /// response, and `0.0` for prompt tokens (and for the final token, which has
    /// no successor). This is exactly the mask
    /// [`SftDataset`](crate::SftDataset) expects.
    pub fn encode_supervised(
        &self,
        prompt: &str,
        response: &str,
    ) -> Result<(Vec<usize>, Vec<f32>), DataError> {
        let prompt_tokens = self.encode(prompt)?;
        let response_tokens = self.encode(response)?;
        if response_tokens.is_empty() {
            return Err(DataError::EmptyResponse);
        }

        let prompt_len = prompt_tokens.len();
        let mut tokens = prompt_tokens;
        tokens.extend(response_tokens);

        let mut weights = vec![0.0f32; tokens.len()];
        // Predicting the first response token is supervised by the last prompt
        // position, so masking starts one token before the response.
        let first_supervised = prompt_len.saturating_sub(1);
        for weight in weights.iter_mut().skip(first_supervised) {
            *weight = 1.0;
        }
        if let Some(last) = weights.last_mut() {
            *last = 0.0;
        }
        Ok((tokens, weights))
    }
}
