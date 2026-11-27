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
/// Names are stable by parameter order (`model.parameter.00000`, ...). The
/// order is part of mini-llm's checkpoint contract and is verified on import.
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
    for (index, (tensor, bytes)) in values.iter().zip(&buffers).enumerate() {
        let view = TensorView::new(Dtype::F32, tensor.shape().to_vec(), bytes)
            .map_err(|error| TrainError::SafeTensors(error.to_string()))?;
        views.push((format!("model.parameter.{index:05}"), view));
    }
    let metadata = HashMap::from([
        ("format".to_string(), FORMAT.to_string()),
        ("parameter_count".to_string(), values.len().to_string()),
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
    for (index, parameter) in parameters.iter().enumerate() {
        let name = format!("model.parameter.{index:05}");
        let view = tensors
            .tensor(&name)
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
