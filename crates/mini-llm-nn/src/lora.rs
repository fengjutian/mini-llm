//! Low-rank adaptation (LoRA) around a frozen [`Linear`] layer.
//!
//! ```text
//! y = W x + b + (alpha / rank) * (x A) B
//! ```
//!
//! `A` is `[in, rank]`, `B` is `[rank, out]` and starts at zero, so the adapted
//! layer begins exactly at the base function. Only `A` and `B` are reported by
//! [`Module::parameters`], which is what freezes the base weights as far as an
//! optimizer is concerned.

use mini_llm_autograd::TensorValue;
use mini_llm_random::Rng;
use mini_llm_tensor::Tensor;

use crate::{Linear, Module, NnError};

/// A frozen linear layer plus a trainable low-rank update.
pub struct LoraLinear {
    base: Linear,
    down: TensorValue,
    up: TensorValue,
    scale: f32,
    rank: usize,
}

impl LoraLinear {
    /// Wraps `base`, adding a rank-`rank` update scaled by `alpha / rank`.
    pub fn new(base: Linear, rank: usize, alpha: f32, rng: &mut Rng) -> Result<Self, NnError> {
        if rank == 0 {
            return Err(NnError::InvalidDimension {
                name: "lora_rank",
                value: rank,
            });
        }
        let in_features = base.in_features();
        let out_features = base.out_features();
        let down = TensorValue::param(rng.normal_tensor(vec![in_features, rank], 0.02)?);
        let up = TensorValue::param(Tensor::zeros(vec![rank, out_features])?);
        Ok(Self {
            base,
            down,
            up,
            scale: alpha / rank as f32,
            rank,
        })
    }

    pub fn base(&self) -> &Linear {
        &self.base
    }

    pub fn rank(&self) -> usize {
        self.rank
    }

    pub fn scale(&self) -> f32 {
        self.scale
    }

    /// All weights, base included; useful when saving a full checkpoint.
    pub fn all_parameters(&self) -> Vec<TensorValue> {
        let mut parameters = self.base.parameters();
        parameters.push(self.down.clone());
        parameters.push(self.up.clone());
        parameters
    }
}

impl Module for LoraLinear {
    fn forward(&self, input: &TensorValue) -> Result<TensorValue, NnError> {
        let base_output = self.base.forward(input)?;
        let update = input
            .matmul(&self.down)?
            .matmul(&self.up)?
            .scale(self.scale)?;
        Ok(base_output.add(&update)?)
    }

    /// Only the adapter matrices; the base layer is frozen.
    fn parameters(&self) -> Vec<TensorValue> {
        vec![self.down.clone(), self.up.clone()]
    }

    fn name(&self) -> &'static str {
        "lora"
    }

    fn summary(&self) -> String {
        format!(
            "lora({} -> {}, rank {}, scale {:.3}) ({} adapter params)",
            self.base.in_features(),
            self.base.out_features(),
            self.rank,
            self.scale,
            self.num_parameters()
        )
    }
}
