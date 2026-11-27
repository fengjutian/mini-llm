//! SafeTensors weight exchange for the educational `TinyGpt`.
//!
//! The native checkpoint remains the reproducible training format because it
//! also stores tokenizer and optimizer state. SafeTensors is deliberately a
//! weights-only interchange format understood by Hugging Face tooling.

use std::collections::HashMap;
use std::fs;
use std::path::Path;

use mini_llm_model::TinyGpt;
use mini_llm_tensor::Tensor;
use safetensors::tensor::{serialize_to_file, Dtype, SafeTensors, TensorView};

use crate::TrainError;

const FORMAT: &str = "mini-llm-tiny-gpt-v1";
const MAX_FILE_BYTES: u64 = 1 << 30;

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
}
