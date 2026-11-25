//! Dropout with inverted scaling.

use std::cell::RefCell;

use mini_llm_autograd::TensorValue;
use mini_llm_random::Rng;
use mini_llm_tensor::Tensor;

use crate::{Module, NnError};

/// Randomly zeroes elements with probability `p` and scales survivors by
/// `1 / (1 - p)` so the expected value is unchanged.
///
/// Active while the crate is in training mode; [`crate::mode::with_eval`]
/// (used by validation and generation) turns it into the identity. The mask RNG
/// is internal, so a resumed run does not replay the same dropout pattern.
pub struct Dropout {
    probability: f32,
    rng: RefCell<Rng>,
}

impl Dropout {
    pub fn new(probability: f32, seed: u64) -> Self {
        Self {
            probability: probability.clamp(0.0, 1.0),
            rng: RefCell::new(Rng::new(seed)),
        }
    }

    pub fn probability(&self) -> f32 {
        self.probability
    }
}

impl Module for Dropout {
    fn forward(&self, input: &TensorValue) -> Result<TensorValue, NnError> {
        if self.probability <= 0.0 || crate::mode::is_eval() {
            return Ok(input.clone());
        }

        let keep = 1.0 - self.probability;
        let scale = 1.0 / keep;
        let mut rng = self.rng.borrow_mut();
        let data: Vec<f32> = (0..input.data().len())
            .map(|_| if rng.next_f32() < keep { scale } else { 0.0 })
            .collect();
        let mask = TensorValue::new(Tensor::new(data, input.shape().to_vec())?);
        Ok(input.mul(&mask)?)
    }

    fn parameters(&self) -> Vec<TensorValue> {
        Vec::new()
    }

    fn name(&self) -> &'static str {
        "dropout"
    }

    fn summary(&self) -> String {
        format!("dropout(p={:.2})", self.probability)
    }
}
