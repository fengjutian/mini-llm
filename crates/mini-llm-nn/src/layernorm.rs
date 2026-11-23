//! Layer normalization over the last dimension.
//!
//! Unlike a fused kernel, this layer is *composed* from autograd primitives, so
//! its backward pass is derived by the engine itself:
//!
//! ```text
//! mu      = mean(x, dim=-1, keepdim)
//! var     = mean((x - mu)^2, dim=-1, keepdim)
//! inv_std = (var + eps)^(-1/2)
//! y       = (x - mu) * inv_std * gamma + beta
//! ```

use mini_llm_autograd::TensorValue;
use mini_llm_tensor::Tensor;

use crate::{expect_matrix, Module, NnError};

const DEFAULT_EPS: f32 = 1e-5;

pub struct LayerNorm {
    gamma: TensorValue,
    beta: TensorValue,
    dim: usize,
    eps: f32,
}

impl LayerNorm {
    /// Creates a layer with `gamma = 1` and `beta = 0`.
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
            beta: TensorValue::param(Tensor::zeros(vec![dim])?),
            dim,
            eps,
        })
    }

    pub fn gamma(&self) -> &TensorValue {
        &self.gamma
    }

    pub fn beta(&self) -> &TensorValue {
        &self.beta
    }

    pub fn eps(&self) -> f32 {
        self.eps
    }
}

impl Module for LayerNorm {
    fn forward(&self, input: &TensorValue) -> Result<TensorValue, NnError> {
        expect_matrix(input, self.dim)?;

        let mean = input.mean_rows_keepdim()?;
        let centered = input.sub(&mean)?;
        let variance = centered.powf(2.0)?.mean_rows_keepdim()?;
        let inv_std = variance.add_scalar(self.eps)?.powf(-0.5)?;
        let normalized = centered.mul(&inv_std)?;
        Ok(normalized.mul(&self.gamma)?.add(&self.beta)?)
    }

    fn parameters(&self) -> Vec<TensorValue> {
        vec![self.gamma.clone(), self.beta.clone()]
    }

    fn name(&self) -> &'static str {
        "layernorm"
    }

    fn summary(&self) -> String {
        format!("layernorm({}) ({} params)", self.dim, self.num_parameters())
    }
}
