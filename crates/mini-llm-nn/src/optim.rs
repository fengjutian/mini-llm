//! Optimizers: SGD (optionally with momentum) and AdamW.
//!
//! Parameters are updated in place through [`TensorValue::set_value`]. State is
//! keyed by the parameter's graph-node identity, so the same parameter is
//! recognized across steps even though forward passes rebuild the graph.

use std::collections::HashMap;

use mini_llm_autograd::TensorValue;
use mini_llm_tensor::Tensor;

use crate::NnError;

/// A parameter update rule.
pub trait Optimizer {
    /// Applies one update using the gradients currently stored on `parameters`.
    fn step(&mut self, parameters: &[TensorValue]) -> Result<(), NnError>;

    /// Clears gradients after an update.
    fn zero_grad(&self, parameters: &[TensorValue]) {
        for parameter in parameters {
            parameter.zero_grad();
        }
    }
}

/// Stochastic gradient descent.
///
/// With `momentum == 0` the update is `p <- p - lr * g`; otherwise a velocity
/// buffer `v <- momentum * v + g` is maintained and `p <- p - lr * v`.
pub struct Sgd {
    lr: f32,
    momentum: f32,
    velocity: HashMap<usize, Vec<f32>>,
}

impl Sgd {
    pub fn new(lr: f32) -> Self {
        Self {
            lr,
            momentum: 0.0,
            velocity: HashMap::new(),
        }
    }

    pub fn with_momentum(lr: f32, momentum: f32) -> Self {
        Self {
            lr,
            momentum,
            velocity: HashMap::new(),
        }
    }

    pub fn learning_rate(&self) -> f32 {
        self.lr
    }

    pub fn set_learning_rate(&mut self, lr: f32) {
        self.lr = lr;
    }
}

impl Optimizer for Sgd {
    fn step(&mut self, parameters: &[TensorValue]) -> Result<(), NnError> {
        for parameter in parameters {
            let Some(grad) = parameter.grad() else {
                continue;
            };
            let value = parameter.value();
            let shape = value.shape().to_vec();
            let mut data = value.data().to_vec();
            let grad_data = grad.data();

            if self.momentum > 0.0 {
                let velocity = self
                    .velocity
                    .entry(parameter.id())
                    .or_insert_with(|| vec![0.0; data.len()]);
                for index in 0..data.len() {
                    velocity[index] = self.momentum * velocity[index] + grad_data[index];
                    data[index] -= self.lr * velocity[index];
                }
            } else {
                for index in 0..data.len() {
                    data[index] -= self.lr * grad_data[index];
                }
            }
            parameter.set_value(Tensor::new(data, shape)?)?;
        }
        Ok(())
    }
}

#[derive(Default)]
struct AdamSlot {
    m: Vec<f32>,
    v: Vec<f32>,
}

/// AdamW: Adam with decoupled weight decay.
///
/// `m = b1 m + (1 - b1) g`, `v = b2 v + (1 - b2) g^2`, bias-corrected, then
/// `p <- p * (1 - lr * wd) - lr * m_hat / (sqrt(v_hat) + eps)`.
pub struct AdamW {
    lr: f32,
    beta1: f32,
    beta2: f32,
    eps: f32,
    weight_decay: f32,
    step_count: u32,
    slots: HashMap<usize, AdamSlot>,
}

impl AdamW {
    pub fn new(lr: f32) -> Self {
        Self {
            lr,
            beta1: 0.9,
            beta2: 0.999,
            eps: 1e-8,
            weight_decay: 0.01,
            step_count: 0,
            slots: HashMap::new(),
        }
    }

    pub fn with_weight_decay(mut self, weight_decay: f32) -> Self {
        self.weight_decay = weight_decay;
        self
    }

    pub fn with_betas(mut self, beta1: f32, beta2: f32) -> Self {
        self.beta1 = beta1;
        self.beta2 = beta2;
        self
    }

    pub fn learning_rate(&self) -> f32 {
        self.lr
    }

    pub fn set_learning_rate(&mut self, lr: f32) {
        self.lr = lr;
    }

    pub fn step_count(&self) -> u32 {
        self.step_count
    }
}

impl Optimizer for AdamW {
    fn step(&mut self, parameters: &[TensorValue]) -> Result<(), NnError> {
        if !parameters
            .iter()
            .any(|parameter| parameter.grad().is_some())
        {
            return Ok(());
        }
        self.step_count += 1;
        let step = self.step_count;
        let bias1 = 1.0 - self.beta1.powi(step as i32);
        let bias2 = 1.0 - self.beta2.powi(step as i32);

        for parameter in parameters {
            let Some(grad) = parameter.grad() else {
                continue;
            };
            let value = parameter.value();
            let shape = value.shape().to_vec();
            let mut data = value.data().to_vec();
            let grad_data = grad.data();
            let slot = self
                .slots
                .entry(parameter.id())
                .or_insert_with(|| AdamSlot {
                    m: vec![0.0; data.len()],
                    v: vec![0.0; data.len()],
                });

            for index in 0..data.len() {
                let g = grad_data[index];
                slot.m[index] = self.beta1 * slot.m[index] + (1.0 - self.beta1) * g;
                slot.v[index] = self.beta2 * slot.v[index] + (1.0 - self.beta2) * g * g;
                let m_hat = slot.m[index] / bias1;
                let v_hat = slot.v[index] / bias2;
                let decay = 1.0 - self.lr * self.weight_decay;
                data[index] = data[index] * decay - self.lr * m_hat / (v_hat.sqrt() + self.eps);
            }
            parameter.set_value(Tensor::new(data, shape)?)?;
        }
        Ok(())
    }
}
