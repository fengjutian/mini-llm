//! Root-mean-square normalization (no mean subtraction, no bias).

use mini_llm_autograd::TensorValue;
use mini_llm_tensor::Tensor;

use crate::{expect_matrix, Module, NnError};

const DEFAULT_EPS: f32 = 1e-5;

/// `y = gamma * x / sqrt(mean(x^2) + eps)`, normalized over the last axis.
///
/// Like [`crate::LayerNorm`] this is composed from autograd primitives, so the
/// backward pass is derived by the engine rather than written by hand.
pub struct RmsNorm {
    gamma: TensorValue,
    dim: usize,
    eps: f32,
}

impl RmsNorm {
    pub fn new(dim: usize) -> Result<Self, NnError> {
        Self::with_eps(dim, DEFAULT_EPS)
    }

    pub fn with_eps(dim: usize, eps: f32) -> Result<Self, NnError> {
        if dim == 0 {
            return Err(NnError::InvalidDimension {
                name: "normalized_shape",
                value: dim,
            });
        }
        Ok(Self {
            gamma: TensorValue::param(Tensor::full(vec![dim], 1.0)?),
            dim,
            eps,
        })
    }

    pub fn gamma(&self) -> &TensorValue {
        &self.gamma
    }

    pub fn eps(&self) -> f32 {
        self.eps
    }
}

impl Module for RmsNorm {
    fn forward(&self, input: &TensorValue) -> Result<TensorValue, NnError> {
        expect_matrix(input, self.dim)?;
        let mean_square = input.powf(2.0)?.mean_rows_keepdim()?;
        let inverted_rms = mean_square.add_scalar(self.eps)?.powf(-0.5)?;
        Ok(input.mul(&inverted_rms)?.mul(&self.gamma)?)
    }

    fn parameters(&self) -> Vec<TensorValue> {
        vec![self.gamma.clone()]
    }

    fn name(&self) -> &'static str {
        "rmsnorm"
    }

    fn summary(&self) -> String {
        format!("rmsnorm({}) ({} params)", self.dim, self.num_parameters())
    }
}
