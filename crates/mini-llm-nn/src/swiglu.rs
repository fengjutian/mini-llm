//! SwiGLU feed-forward layer: `W2 (SiLU(W1 x) ⊙ W3 x)`.

use mini_llm_autograd::TensorValue;
use mini_llm_random::Rng;

use crate::{Linear, Module, NnError};

/// The gated variant of the Transformer MLP.
///
/// Three projections instead of two, but the element-wise product with the
/// SiLU gate replaces the activation function and usually trains better.
pub struct SwiGlu {
    gate: Linear,
    up: Linear,
    down: Linear,
    hidden: usize,
    inner: usize,
}

impl SwiGlu {
    pub fn new(hidden: usize, inner: usize, rng: &mut Rng) -> Result<Self, NnError> {
        if hidden == 0 || inner == 0 {
            return Err(NnError::InvalidDimension {
                name: if hidden == 0 { "hidden" } else { "inner" },
                value: 0,
            });
        }
        Ok(Self {
            gate: Linear::new(hidden, inner, rng)?,
            up: Linear::new(hidden, inner, rng)?,
            down: Linear::new(inner, hidden, rng)?,
            hidden,
            inner,
        })
    }

    pub fn hidden(&self) -> usize {
        self.hidden
    }

    pub fn inner(&self) -> usize {
        self.inner
    }
}

impl Module for SwiGlu {
    fn forward(&self, input: &TensorValue) -> Result<TensorValue, NnError> {
        let gate = self.gate.forward(input)?;
        let activated = gate.mul(&gate.sigmoid()?)?;
        let up = self.up.forward(input)?;
        self.down.forward(&activated.mul(&up)?)
    }

    fn parameters(&self) -> Vec<TensorValue> {
        let mut parameters = self.gate.parameters();
        parameters.extend(self.up.parameters());
        parameters.extend(self.down.parameters());
        parameters
    }

    fn name(&self) -> &'static str {
        "swiglu"
    }

    fn summary(&self) -> String {
        format!(
            "swiglu({} -> {} -> {}) ({} params)",
            self.hidden,
            self.inner,
            self.hidden,
            self.num_parameters()
        )
    }
}
