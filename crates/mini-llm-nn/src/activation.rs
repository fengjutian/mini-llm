//! Stateless activation modules.

use mini_llm_autograd::TensorValue;

use crate::{Module, NnError};

/// Element-wise rectified linear unit.
#[derive(Debug, Clone, Copy, Default)]
pub struct Relu;

impl Module for Relu {
    fn forward(&self, input: &TensorValue) -> Result<TensorValue, NnError> {
        Ok(input.relu()?)
    }

    fn parameters(&self) -> Vec<TensorValue> {
        Vec::new()
    }

    fn name(&self) -> &'static str {
        "relu"
    }
}

/// Element-wise hyperbolic tangent.
#[derive(Debug, Clone, Copy, Default)]
pub struct Tanh;

impl Module for Tanh {
    fn forward(&self, input: &TensorValue) -> Result<TensorValue, NnError> {
        Ok(input.tanh()?)
    }

    fn parameters(&self) -> Vec<TensorValue> {
        Vec::new()
    }

    fn name(&self) -> &'static str {
        "tanh"
    }
}

/// Element-wise logistic sigmoid.
#[derive(Debug, Clone, Copy, Default)]
pub struct Sigmoid;

impl Module for Sigmoid {
    fn forward(&self, input: &TensorValue) -> Result<TensorValue, NnError> {
        Ok(input.sigmoid()?)
    }

    fn parameters(&self) -> Vec<TensorValue> {
        Vec::new()
    }

    fn name(&self) -> &'static str {
        "sigmoid"
    }
}
