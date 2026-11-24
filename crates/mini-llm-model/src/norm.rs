//! Uniform wrapper over the two normalization layers.

use mini_llm_autograd::TensorValue;
use mini_llm_nn::{LayerNorm, Module, RmsNorm};

use crate::{ModelError, NormKind};

pub(crate) enum NormLayer {
    Layer(LayerNorm),
    Rms(RmsNorm),
}

impl NormLayer {
    pub(crate) fn new(kind: NormKind, dim: usize) -> Result<Self, ModelError> {
        Ok(match kind {
            NormKind::LayerNorm => Self::Layer(LayerNorm::new(dim)?),
            NormKind::RmsNorm => Self::Rms(RmsNorm::new(dim)?),
        })
    }

    pub(crate) fn forward(&self, input: &TensorValue) -> Result<TensorValue, ModelError> {
        Ok(match self {
            Self::Layer(layer) => layer.forward(input)?,
            Self::Rms(layer) => layer.forward(input)?,
        })
    }

    pub(crate) fn parameters(&self) -> Vec<TensorValue> {
        match self {
            Self::Layer(layer) => layer.parameters(),
            Self::Rms(layer) => layer.parameters(),
        }
    }
}
