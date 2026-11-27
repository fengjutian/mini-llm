//! SafeTensors weight exchange for the educational `TinyGpt`.
//!
//! The native checkpoint remains the reproducible training format because it
//! also stores tokenizer and optimizer state. SafeTensors is deliberately a
//! weights-only interchange format understood by Hugging Face tooling.

use std::collections::HashMap;
use std::fs;
use std::path::Path;

use mini_llm_data::{BpeTokenizer, CharTokenizer, Tokenizer};
use mini_llm_model::{FeedForwardKind, ModelConfig, NormKind, PositionKind, TinyGpt};
use mini_llm_nn::Rng;
use mini_llm_tensor::Tensor;
use safetensors::tensor::{serialize_to_file, Dtype, SafeTensors, TensorView};
use serde::{Deserialize, Serialize};

use crate::TrainError;

const FORMAT: &str = "mini-llm-tiny-gpt-v1";
const MAX_FILE_BYTES: u64 = 1 << 30;
const MAX_SIDECAR_BYTES: u64 = 1 << 20;

#[derive(Debug, Serialize, Deserialize)]
struct BundleConfig {
    architectures: Vec<String>,
    model_type: String,
    vocab_size: usize,
    hidden_size: usize,
    num_layers: usize,
    num_heads: usize,
    num_kv_heads: usize,
    max_seq_len: usize,
    norm: String,
    position: String,
    feed_forward: String,
    num_experts: usize,
    dropout: f32,
}

#[derive(Debug, Serialize, Deserialize)]
#[serde(tag = "kind", rename_all = "snake_case")]
enum BundleTokenizer {
    Char {
        vocab_size: usize,
        characters: String,
    },
    Bpe {
        vocab_size: usize,
        merges: Vec<(u32, u32)>,
    },
}

/// Writes model weights to the standard SafeTensors container.
///
/// Tensor names follow the Hugging Face convention
/// (`model.embed_tokens.weight`, `model.layers.0.self_attn.q_proj.weight`, ...)
/// so the file is readable by external tooling; the order still matches
/// [`TinyGpt::parameters`] and the count is recorded in the metadata.
pub fn save_safetensors(path: impl AsRef<Path>, model: &TinyGpt) -> Result<(), TrainError> {
    let values: Vec<Tensor> = model
        .parameters()
        .iter()
        .map(|parameter| parameter.value())
        .collect();
    let buffers: Vec<Vec<u8>> = values
        .iter()
        .map(|tensor| {
            let mut bytes = Vec::with_capacity(tensor.data().len() * 4);
            for value in tensor.data() {
                bytes.extend_from_slice(&value.to_le_bytes());
            }
            bytes
        })
        .collect();
    let mut views = Vec::with_capacity(values.len());
    for ((name, tensor), bytes) in model
        .parameter_names()
        .into_iter()
        .zip(&values)
        .zip(&buffers)
    {
        let view = TensorView::new(Dtype::F32, tensor.shape().to_vec(), bytes)
            .map_err(|error| TrainError::SafeTensors(error.to_string()))?;
        views.push((name, view));
    }
    let metadata = HashMap::from([
        ("format".to_string(), FORMAT.to_string()),
        ("parameter_count".to_string(), values.len().to_string()),
        (
            "vocab_size".to_string(),
            model.config().vocab_size.to_string(),
        ),
        (
            "num_layers".to_string(),
            model.config().num_layers.to_string(),
        ),
        (
            "hidden_size".to_string(),
            model.config().hidden_size.to_string(),
        ),
        (
            "max_seq_len".to_string(),
            model.config().max_seq_len.to_string(),
        ),
    ]);
    serialize_to_file(views, Some(metadata), path.as_ref())
        .map_err(|error| TrainError::SafeTensors(error.to_string()))
}

/// Writes the architecture sidecar (`config.json`-style) for the exported weights.
///
/// This is intentionally a small, documented subset of the Hugging Face config
/// schema: enough for a human or tool to rebuild [`mini_llm_model::ModelConfig`].
pub fn save_config_json(path: impl AsRef<Path>, model: &TinyGpt) -> Result<(), TrainError> {
    let config = model.config();
    write_json(path.as_ref(), &BundleConfig::from(config))
}

/// Writes the tokenizer sidecar: characters for a char vocabulary, merges for BPE.
pub fn save_tokenizer_json(
    path: impl AsRef<Path>,
    tokenizer: &Tokenizer,
) -> Result<(), TrainError> {
    write_json(path.as_ref(), &BundleTokenizer::from(tokenizer))
}

/// Saves a self-contained, Hugging Face-style directory.
///
/// The directory contains `config.json`, `tokenizer.json`, and
/// `model.safetensors`. It preserves this project's complete inference state;
/// optimizer state remains exclusive to native training checkpoints.
pub fn save_hf_bundle(
    directory: impl AsRef<Path>,
    model: &TinyGpt,
    tokenizer: &Tokenizer,
) -> Result<(), TrainError> {
    if model.config().vocab_size != tokenizer.vocab_size() {
        return Err(TrainError::SafeTensors(format!(
            "model vocabulary {} does not match tokenizer vocabulary {}",
            model.config().vocab_size,
            tokenizer.vocab_size()
        )));
    }
    let directory = directory.as_ref();
    fs::create_dir_all(directory)?;
    save_config_json(directory.join("config.json"), model)?;
    save_tokenizer_json(directory.join("tokenizer.json"), tokenizer)?;
    save_safetensors(directory.join("model.safetensors"), model)
}

/// Reconstructs a model and tokenizer from a directory written by
/// [`save_hf_bundle`].
pub fn load_hf_bundle(directory: impl AsRef<Path>) -> Result<(TinyGpt, Tokenizer), TrainError> {
    let directory = directory.as_ref();
    let stored: BundleConfig = read_json(&directory.join("config.json"))?;
    if stored.model_type != "mini-llm-tiny-gpt" {
        return Err(TrainError::SafeTensors(format!(
            "unsupported model_type {:?}",
            stored.model_type
        )));
    }
    let config = ModelConfig::try_from(stored)?;
    let stored_tokenizer: BundleTokenizer = read_json(&directory.join("tokenizer.json"))?;
    let tokenizer = Tokenizer::try_from(stored_tokenizer)?;
    if config.vocab_size != tokenizer.vocab_size() {
        return Err(TrainError::SafeTensors(format!(
            "config vocabulary {} does not match tokenizer vocabulary {}",
            config.vocab_size,
            tokenizer.vocab_size()
        )));
    }
    let model = TinyGpt::new(config, &mut Rng::new(0))?;
    load_safetensors(directory.join("model.safetensors"), &model)?;
    Ok((model, tokenizer))
}

impl From<&ModelConfig> for BundleConfig {
    fn from(config: &ModelConfig) -> Self {
        Self {
            architectures: vec!["MiniLlmForCausalLM".to_string()],
            model_type: "mini-llm-tiny-gpt".to_string(),
            vocab_size: config.vocab_size,
            hidden_size: config.hidden_size,
            num_layers: config.num_layers,
            num_heads: config.num_heads,
            num_kv_heads: config.num_kv_heads,
            max_seq_len: config.max_seq_len,
            norm: format!("{:?}", config.norm),
            position: format!("{:?}", config.position),
            feed_forward: format!("{:?}", config.feed_forward),
            num_experts: config.num_experts,
            dropout: config.dropout,
        }
    }
}

impl TryFrom<BundleConfig> for ModelConfig {
    type Error = TrainError;

    fn try_from(config: BundleConfig) -> Result<Self, Self::Error> {
        Ok(Self {
            vocab_size: config.vocab_size,
            hidden_size: config.hidden_size,
            num_layers: config.num_layers,
            num_heads: config.num_heads,
            num_kv_heads: config.num_kv_heads,
            max_seq_len: config.max_seq_len,
            norm: parse_choice(
                &config.norm,
                "norm",
                &[
                    ("LayerNorm", NormKind::LayerNorm),
                    ("RmsNorm", NormKind::RmsNorm),
                ],
            )?,
            position: parse_choice(
                &config.position,
                "position",
                &[
                    ("Learned", PositionKind::Learned),
                    ("Rope", PositionKind::Rope),
                ],
            )?,
            feed_forward: parse_choice(
                &config.feed_forward,
                "feed_forward",
                &[
                    ("Relu", FeedForwardKind::Relu),
                    ("SwiGlu", FeedForwardKind::SwiGlu),
                ],
            )?,
            num_experts: config.num_experts,
            dropout: config.dropout,
        })
    }
}

impl From<&Tokenizer> for BundleTokenizer {
    fn from(tokenizer: &Tokenizer) -> Self {
        match tokenizer {
            Tokenizer::Char(value) => Self::Char {
                vocab_size: value.vocab_size(),
                characters: value.characters().iter().collect(),
            },
            Tokenizer::Bpe(value) => Self::Bpe {
                vocab_size: value.vocab_size(),
                merges: value.merges().to_vec(),
            },
        }
    }
}

impl TryFrom<BundleTokenizer> for Tokenizer {
    type Error = TrainError;

    fn try_from(tokenizer: BundleTokenizer) -> Result<Self, Self::Error> {
        let (declared_vocab, tokenizer) = match tokenizer {
            BundleTokenizer::Char {
                vocab_size,
                characters,
            } => {
                let value = CharTokenizer::from_characters(characters.chars().collect())?;
                (vocab_size, Tokenizer::from(value))
            }
            BundleTokenizer::Bpe { vocab_size, merges } => {
                let value = BpeTokenizer::from_merges(merges)?;
                (vocab_size, Tokenizer::from(value))
            }
        };
        if declared_vocab != tokenizer.vocab_size() {
            return Err(TrainError::SafeTensors(format!(
                "tokenizer declares vocabulary {declared_vocab}, but defines {} entries",
                tokenizer.vocab_size()
            )));
        }
        Ok(tokenizer)
    }
}

fn parse_choice<T: Copy>(value: &str, field: &str, choices: &[(&str, T)]) -> Result<T, TrainError> {
    choices
        .iter()
        .find_map(|(name, choice)| (*name == value).then_some(*choice))
        .ok_or_else(|| TrainError::SafeTensors(format!("invalid {field} value {value:?}")))
}

fn write_json(path: &Path, value: &impl Serialize) -> Result<(), TrainError> {
    let mut bytes = serde_json::to_vec_pretty(value)
        .map_err(|error| TrainError::SafeTensors(format!("cannot encode JSON: {error}")))?;
    bytes.push(b'\n');
    fs::write(path, bytes)?;
    Ok(())
}

fn read_json<T: for<'de> Deserialize<'de>>(path: &Path) -> Result<T, TrainError> {
    let size = fs::metadata(path)?.len();
    if size > MAX_SIDECAR_BYTES {
        return Err(TrainError::CheckpointLimit {
            field: "bundle sidecar bytes",
            value: size,
            limit: MAX_SIDECAR_BYTES,
        });
    }
    let bytes = fs::read(path)?;
    serde_json::from_slice(&bytes).map_err(|error| {
        TrainError::SafeTensors(format!("cannot parse {}: {error}", path.display()))
    })
}

/// Loads SafeTensors weights into an already configured model.
///
/// Supplying the model separately makes architecture mismatches explicit and
/// avoids pretending that a weights-only format contains a tokenizer.
pub fn load_safetensors(path: impl AsRef<Path>, model: &TinyGpt) -> Result<(), TrainError> {
    let size = fs::metadata(path.as_ref())?.len();
    if size > MAX_FILE_BYTES {
        return Err(TrainError::CheckpointLimit {
            field: "SafeTensors file bytes",
            value: size,
            limit: MAX_FILE_BYTES,
        });
    }
    let bytes = fs::read(path)?;
    let tensors = SafeTensors::deserialize(&bytes)
        .map_err(|error| TrainError::SafeTensors(error.to_string()))?;
    let parameters = model.parameters();
    for (index, (name, parameter)) in model
        .parameter_names()
        .into_iter()
        .zip(&parameters)
        .enumerate()
    {
        let legacy = format!("model.parameter.{index:05}");
        let view = tensors
            .tensor(&name)
            .or_else(|_| tensors.tensor(&legacy))
            .map_err(|error| TrainError::SafeTensors(error.to_string()))?;
        if view.dtype() != Dtype::F32 {
            return Err(TrainError::SafeTensors(format!(
                "{name} has dtype {:?}, expected F32",
                view.dtype()
            )));
        }
        let expected = parameter.value().shape().to_vec();
        if view.shape() != expected {
            return Err(TrainError::ShapeMismatch {
                expected,
                actual: view.shape().to_vec(),
            });
        }
        let mut data = Vec::with_capacity(view.data().len() / 4);
        let mut chunks = view.data().chunks_exact(4);
        for chunk in &mut chunks {
            data.push(f32::from_le_bytes(
                chunk.try_into().expect("four-byte chunk"),
            ));
        }
        if !chunks.remainder().is_empty() {
            return Err(TrainError::SafeTensors(format!(
                "{name} has a non-F32 byte length"
            )));
        }
        parameter.set_value(Tensor::new(data, view.shape().to_vec())?)?;
    }
    if tensors.len() != parameters.len() {
        return Err(TrainError::SafeTensors(format!(
            "file contains {} tensors, model expects {}",
            tensors.len(),
            parameters.len()
        )));
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use mini_llm_model::ModelConfig;
    use mini_llm_nn::Rng;

    #[test]
    fn export_records_architecture_metadata() {
        let model = TinyGpt::new(ModelConfig::modern(32), &mut Rng::new(2)).expect("model");
        let path = std::env::temp_dir().join("mini_llm_safetensors_meta.safetensors");
        save_safetensors(&path, &model).expect("save");
        let bytes = fs::read(&path).expect("read");
        let header_len = u64::from_le_bytes(bytes[..8].try_into().expect("header length")) as usize;
        let header = std::str::from_utf8(&bytes[8..8 + header_len]).expect("utf-8 header");
        assert!(header.contains("mini-llm-tiny-gpt-v1"), "{header}");
        assert!(header.contains("\"vocab_size\":\"32\""), "{header}");
        assert!(header.contains("\"num_layers\":\"2\""), "{header}");
        assert!(
            header.contains(&format!(
                "\"parameter_count\":\"{}\"",
                model.parameters().len()
            )),
            "{header}"
        );
        let _ = fs::remove_file(path);
    }

    #[test]
    fn config_sidecar_records_architecture() {
        let model = TinyGpt::new(ModelConfig::modern(32), &mut Rng::new(5)).expect("model");
        let path = std::env::temp_dir().join("mini_llm_config_sidecar.json");
        save_config_json(&path, &model).expect("write config");
        let json = fs::read_to_string(&path).expect("read config");
        assert!(json.contains("\"vocab_size\": 32"), "{json}");
        assert!(json.contains("\"num_layers\": 2"), "{json}");
        assert!(json.contains("\"norm\": \"RmsNorm\""), "{json}");
        let _ = fs::remove_file(path);
    }

    #[test]
    fn char_tokenizer_sidecar_escapes_special_characters() {
        let tokenizer = Tokenizer::Char(
            mini_llm_data::CharTokenizer::from_characters(vec!['"', '\\', 'a', ' '])
                .expect("char tokenizer"),
        );
        let path = std::env::temp_dir().join("mini_llm_tokenizer_sidecar.json");
        save_tokenizer_json(&path, &tokenizer).expect("write tokenizer");
        let json = fs::read_to_string(&path).expect("read tokenizer");
        let decoded: BundleTokenizer = serde_json::from_str(&json).expect("valid JSON");
        assert!(matches!(
            decoded,
            BundleTokenizer::Char { vocab_size: 4, .. }
        ));
        assert!(json.contains("\\\""), "escaped quote missing: {json}");
        assert!(json.contains("\\\\"), "escaped backslash missing: {json}");
        let _ = fs::remove_file(path);
    }

    #[test]
    fn bpe_tokenizer_sidecar_lists_merges() {
        let bpe = mini_llm_data::BpeTokenizer::train("abababababababababab", 260);
        let tokenizer = Tokenizer::Bpe(bpe);
        let path = std::env::temp_dir().join("mini_llm_bpe_sidecar.json");
        save_tokenizer_json(&path, &tokenizer).expect("write tokenizer");
        let json = fs::read_to_string(&path).expect("read tokenizer");
        let decoded: BundleTokenizer = serde_json::from_str(&json).expect("valid JSON");
        assert!(matches!(decoded, BundleTokenizer::Bpe { merges, .. } if !merges.is_empty()));
        let _ = fs::remove_file(path);
    }
}
