//! Binary checkpoint serialization: config + vocabulary + parameters, plus an
//! optional training state (step, RNG and Adam moments) for exact resumption.

use std::fs;
use std::path::Path;

use mini_llm_data::{BpeTokenizer, CharTokenizer, Tokenizer};
use mini_llm_model::{FeedForwardKind, ModelConfig, NormKind, PositionKind, TinyGpt};
use mini_llm_nn::Rng;
use mini_llm_tensor::Tensor;

use crate::{TrainError, TrainingState};

const MAGIC: &[u8; 8] = b"MINILLM1";
/// Version 1 stored only the five base dimensions. Version 2 adds the modern
/// architecture switches, version 3 the training state, and version 4 tags the
/// tokenizer (character-level or BPE). Older checkpoints still load.
const VERSION: u32 = 4;

const TOKENIZER_CHAR: u32 = 0;
const TOKENIZER_BPE: u32 = 1;

/// Writes `model` and its vocabulary to `path` (no training state).
pub fn save_checkpoint(
    path: impl AsRef<Path>,
    model: &TinyGpt,
    tokenizer: &Tokenizer,
) -> Result<(), TrainError> {
    write_checkpoint(path, model, tokenizer, None)
}

/// Writes `model`, its vocabulary and the run state needed to resume exactly.
pub fn save_training_checkpoint(
    path: impl AsRef<Path>,
    model: &TinyGpt,
    tokenizer: &Tokenizer,
    state: &TrainingState,
) -> Result<(), TrainError> {
    write_checkpoint(path, model, tokenizer, Some(state))
}

fn write_checkpoint(
    path: impl AsRef<Path>,
    model: &TinyGpt,
    tokenizer: &Tokenizer,
    state: Option<&TrainingState>,
) -> Result<(), TrainError> {
    if tokenizer.vocab_size() != model.config().vocab_size {
        return Err(TrainError::InvalidVocabulary);
    }

    let mut bytes = Vec::new();
    bytes.extend_from_slice(MAGIC);
    push_u32(&mut bytes, VERSION);

    let config = model.config();
    for value in [
        config.vocab_size,
        config.hidden_size,
        config.num_layers,
        config.num_heads,
        config.max_seq_len,
        config.num_kv_heads,
        config.norm.code() as usize,
        config.position.code() as usize,
        config.feed_forward.code() as usize,
        config.num_experts,
    ] {
        push_u32(&mut bytes, value as u32);
    }

    match tokenizer {
        Tokenizer::Char(tokenizer) => {
            push_u32(&mut bytes, TOKENIZER_CHAR);
            push_u32(&mut bytes, tokenizer.vocab_size() as u32);
            for &character in tokenizer.characters() {
                push_u32(&mut bytes, character as u32);
            }
        }
        Tokenizer::Bpe(tokenizer) => {
            push_u32(&mut bytes, TOKENIZER_BPE);
            let merges = tokenizer.merges();
            push_u32(&mut bytes, merges.len() as u32);
            for &(left, right) in merges {
                push_u32(&mut bytes, left);
                push_u32(&mut bytes, right);
            }
        }
    }

    let parameters = model.parameters();
    push_u32(&mut bytes, parameters.len() as u32);
    for parameter in &parameters {
        let value = parameter.value();
        push_u32(&mut bytes, value.rank() as u32);
        for &dimension in value.shape() {
            push_u32(&mut bytes, dimension as u32);
        }
        for entry in value.data() {
            bytes.extend_from_slice(&entry.to_le_bytes());
        }
    }

    // Version 4 marks whether a training state follows, so a model-only save
    // and a resumable save share one format.
    push_u32(&mut bytes, u32::from(state.is_some()));
    if let Some(state) = state {
        push_u32(&mut bytes, state.step as u32);
        push_u64(&mut bytes, state.rng_state);
        push_u32(&mut bytes, state.optimizer_step);
        for moments in [&state.first_moments, &state.second_moments] {
            push_u32(&mut bytes, moments.len() as u32);
            for values in moments {
                push_u32(&mut bytes, values.len() as u32);
                for entry in values {
                    bytes.extend_from_slice(&entry.to_le_bytes());
                }
            }
        }
    }

    fs::write(path, bytes)?;
    Ok(())
}

/// Reads a checkpoint written by [`save_checkpoint`] or
/// [`save_training_checkpoint`].
pub fn load_checkpoint(path: impl AsRef<Path>) -> Result<(TinyGpt, Tokenizer), TrainError> {
    let (model, tokenizer, _) = load_components(path)?;
    Ok((model, tokenizer))
}

/// Reads a checkpoint and the training state saved alongside it.
pub fn load_training_checkpoint(
    path: impl AsRef<Path>,
) -> Result<(TinyGpt, Tokenizer, TrainingState), TrainError> {
    let (model, tokenizer, state) = load_components(path)?;
    match state {
        Some(state) => Ok((model, tokenizer, state)),
        None => Err(TrainError::MissingTrainingState),
    }
}

fn load_components(
    path: impl AsRef<Path>,
) -> Result<(TinyGpt, Tokenizer, Option<TrainingState>), TrainError> {
    let bytes = fs::read(path)?;
    let mut reader = Reader::new(&bytes);

    if reader.take(MAGIC.len())? != MAGIC {
        return Err(TrainError::BadMagic);
    }
    let version = reader.u32()?;
    if version == 0 || version > VERSION {
        return Err(TrainError::UnsupportedVersion(version));
    }

    let vocab_size = reader.u32()? as usize;
    let hidden_size = reader.u32()? as usize;
    let num_layers = reader.u32()? as usize;
    let num_heads = reader.u32()? as usize;
    let max_seq_len = reader.u32()? as usize;

    let config = if version == 1 {
        ModelConfig {
            vocab_size,
            hidden_size,
            num_layers,
            num_heads,
            max_seq_len,
            num_kv_heads: num_heads,
            norm: NormKind::LayerNorm,
            position: PositionKind::Learned,
            feed_forward: FeedForwardKind::Relu,
            num_experts: 1,
        }
    } else {
        let num_kv_heads = reader.u32()? as usize;
        let norm = decode_architecture("norm", NormKind::from_code, reader.u32()?)?;
        let position = decode_architecture("position", PositionKind::from_code, reader.u32()?)?;
        let feed_forward =
            decode_architecture("feed_forward", FeedForwardKind::from_code, reader.u32()?)?;
        let num_experts = reader.u32()? as usize;
        ModelConfig {
            vocab_size,
            hidden_size,
            num_layers,
            num_heads,
            max_seq_len,
            num_kv_heads,
            norm,
            position,
            feed_forward,
            num_experts,
        }
    };

    let kind = if version >= 4 {
        reader.u32()?
    } else {
        TOKENIZER_CHAR
    };
    let tokenizer = match kind {
        TOKENIZER_CHAR => {
            let character_count = reader.u32()? as usize;
            if character_count != vocab_size {
                return Err(TrainError::InvalidVocabulary);
            }
            let mut characters = Vec::with_capacity(character_count);
            for _ in 0..character_count {
                let raw = reader.u32()?;
                characters.push(char::from_u32(raw).ok_or(TrainError::InvalidVocabulary)?);
            }
            Tokenizer::from(CharTokenizer::from_characters(characters)?)
        }
        TOKENIZER_BPE => {
            let merge_count = reader.u32()? as usize;
            let mut merges = Vec::with_capacity(merge_count);
            for _ in 0..merge_count {
                let left = reader.u32()?;
                let right = reader.u32()?;
                merges.push((left, right));
            }
            let tokenizer = BpeTokenizer::from_merges(merges)?;
            if tokenizer.vocab_size() != vocab_size {
                return Err(TrainError::InvalidVocabulary);
            }
            Tokenizer::from(tokenizer)
        }
        other => {
            return Err(TrainError::InvalidArchitecture {
                field: "tokenizer",
                value: other,
            });
        }
    };

    let model = TinyGpt::new(config, &mut Rng::new(0))?;

    let parameter_count = reader.u32()? as usize;
    let parameters = model.parameters();
    if parameter_count != parameters.len() {
        return Err(TrainError::ShapeMismatch {
            expected: vec![parameters.len()],
            actual: vec![parameter_count],
        });
    }

    for parameter in &parameters {
        let rank = reader.u32()? as usize;
        let mut shape = Vec::with_capacity(rank);
        for _ in 0..rank {
            shape.push(reader.u32()? as usize);
        }
        let expected = parameter.value().shape().to_vec();
        if shape != expected {
            return Err(TrainError::ShapeMismatch {
                expected,
                actual: shape,
            });
        }
        let length: usize = shape.iter().product();
        let mut data = Vec::with_capacity(length);
        for _ in 0..length {
            data.push(reader.f32()?);
        }
        parameter.set_value(Tensor::new(data, shape)?)?;
    }

    let has_state = if version >= 4 {
        reader.u32()? != 0
    } else {
        // Version 3 always wrote the training state.
        version >= 3
    };

    let state = if has_state {
        let step = reader.u32()? as usize;
        let rng_state = reader.u64()?;
        let optimizer_step = reader.u32()?;
        let mut moments = Vec::with_capacity(2);
        for _ in 0..2 {
            let count = reader.u32()? as usize;
            if count != parameters.len() {
                return Err(TrainError::ShapeMismatch {
                    expected: vec![parameters.len()],
                    actual: vec![count],
                });
            }
            let mut values = Vec::with_capacity(count);
            for _ in 0..count {
                let length = reader.u32()? as usize;
                let mut data = Vec::with_capacity(length);
                for _ in 0..length {
                    data.push(reader.f32()?);
                }
                values.push(data);
            }
            moments.push(values);
        }
        let second_moments = moments.pop().expect("two moment vectors");
        let first_moments = moments.pop().expect("two moment vectors");
        Some(TrainingState {
            step,
            rng_state,
            optimizer_step,
            first_moments,
            second_moments,
        })
    } else {
        None
    };

    Ok((model, tokenizer, state))
}

fn push_u32(bytes: &mut Vec<u8>, value: u32) {
    bytes.extend_from_slice(&value.to_le_bytes());
}

fn push_u64(bytes: &mut Vec<u8>, value: u64) {
    bytes.extend_from_slice(&value.to_le_bytes());
}

/// Decodes an architecture flag, reporting which field was invalid.
fn decode_architecture<T>(
    field: &'static str,
    decode: fn(u32) -> Option<T>,
    code: u32,
) -> Result<T, TrainError> {
    decode(code).ok_or(TrainError::InvalidArchitecture { field, value: code })
}

struct Reader<'a> {
    data: &'a [u8],
    position: usize,
}

impl<'a> Reader<'a> {
    fn new(data: &'a [u8]) -> Self {
        Self { data, position: 0 }
    }

    fn take(&mut self, length: usize) -> Result<&'a [u8], TrainError> {
        if self.position + length > self.data.len() {
            return Err(TrainError::Truncated);
        }
        let slice = &self.data[self.position..self.position + length];
        self.position += length;
        Ok(slice)
    }

    fn u32(&mut self) -> Result<u32, TrainError> {
        let bytes = self.take(4)?;
        let mut buffer = [0u8; 4];
        buffer.copy_from_slice(bytes);
        Ok(u32::from_le_bytes(buffer))
    }

    fn u64(&mut self) -> Result<u64, TrainError> {
        let bytes = self.take(8)?;
        let mut buffer = [0u8; 8];
        buffer.copy_from_slice(bytes);
        Ok(u64::from_le_bytes(buffer))
    }

    fn f32(&mut self) -> Result<f32, TrainError> {
        let bytes = self.take(4)?;
        let mut buffer = [0u8; 4];
        buffer.copy_from_slice(bytes);
        Ok(f32::from_le_bytes(buffer))
    }
}
