//! A dense affine layer: `y = x W + b`.

use mini_llm_autograd::TensorValue;
use mini_llm_tensor::Tensor;

use crate::rng::{xavier_uniform, Rng};
use crate::{expect_matrix, Module, NnError};

/// `y = x W + b` with `W` of shape `[in_features, out_features]`.
pub struct Linear {
    weight: TensorValue,
    bias: Option<TensorValue>,
    in_features: usize,
    out_features: usize,
}

impl Linear {
    /// Creates a layer with Xavier-uniform weights and a zero bias.
    pub fn new(in_features: usize, out_features: usize, rng: &mut Rng) -> Result<Self, NnError> {
        Self::build(in_features, out_features, true, rng)
    }

    /// Creates a layer without a bias term.
    pub fn without_bias(
        in_features: usize,
        out_features: usize,
        rng: &mut Rng,
    ) -> Result<Self, NnError> {
        Self::build(in_features, out_features, false, rng)
    }

    fn build(
        in_features: usize,
        out_features: usize,
        with_bias: bool,
        rng: &mut Rng,
    ) -> Result<Self, NnError> {
        if in_features == 0 {
            return Err(NnError::InvalidDimension {
                name: "in_features",
                value: in_features,
            });
        }
        if out_features == 0 {
            return Err(NnError::InvalidDimension {
                name: "out_features",
                value: out_features,
            });
        }
        let weight = TensorValue::param(xavier_uniform(in_features, out_features, rng)?);
        let bias = if with_bias {
            Some(TensorValue::param(Tensor::zeros(vec![out_features])?))
        } else {
            None
        };
        Ok(Self {
            weight,
            bias,
            in_features,
            out_features,
        })
    }

    pub fn weight(&self) -> &TensorValue {
        &self.weight
    }

    pub fn bias(&self) -> Option<&TensorValue> {
        self.bias.as_ref()
    }

    pub fn in_features(&self) -> usize {
        self.in_features
    }

    pub fn out_features(&self) -> usize {
        self.out_features
    }
}

impl Module for Linear {
    fn forward(&self, input: &TensorValue) -> Result<TensorValue, NnError> {
        expect_matrix(input, self.in_features)?;
        let output = input.matmul(&self.weight)?;
        match &self.bias {
            Some(bias) => Ok(output.add(bias)?),
            None => Ok(output),
        }
    }

    fn parameters(&self) -> Vec<TensorValue> {
        match &self.bias {
            Some(bias) => vec![self.weight.clone(), bias.clone()],
            None => vec![self.weight.clone()],
        }
    }

    fn name(&self) -> &'static str {
        "linear"
    }

    fn summary(&self) -> String {
        format!(
            "linear({} -> {}{}) ({} params)",
            self.in_features,
            self.out_features,
            if self.bias.is_some() { ", bias" } else { "" },
            self.num_parameters()
        )
    }
}
