//! Device-resident AdamW: moments, bias correction and the parameter update
//! all run through the device kernels.
//!
//! The update matches `mini_llm_nn::AdamW`:
//! `p <- p * (1 - lr * wd) - lr * m_hat / (sqrt(v_hat) + eps)`.

use crate::{BackendError, DeviceTensor};

/// AdamW optimizer whose state lives on the same device as the parameters.
pub struct DeviceAdamW {
    learning_rate: f32,
    beta1: f32,
    beta2: f32,
    eps: f32,
    weight_decay: f32,
    step_count: u32,
    moments: Vec<Option<(DeviceTensor, DeviceTensor)>>,
}

impl DeviceAdamW {
    pub fn new(learning_rate: f32) -> Self {
        Self {
            learning_rate,
            beta1: 0.9,
            beta2: 0.999,
            eps: 1e-8,
            weight_decay: 0.01,
            step_count: 0,
            moments: Vec::new(),
        }
    }

    pub fn with_betas(mut self, beta1: f32, beta2: f32) -> Self {
        self.beta1 = beta1;
        self.beta2 = beta2;
        self
    }

    pub fn with_weight_decay(mut self, weight_decay: f32) -> Self {
        self.weight_decay = weight_decay;
        self
    }

    pub fn with_eps(mut self, eps: f32) -> Self {
        self.eps = eps;
        self
    }

    pub fn step_count(&self) -> u32 {
        self.step_count
    }

    /// Updates the learning rate, e.g. from a warmup/cosine schedule.
    pub fn set_learning_rate(&mut self, learning_rate: f32) {
        self.learning_rate = learning_rate;
    }

    pub fn learning_rate(&self) -> f32 {
        self.learning_rate
    }

    /// Applies one update; `gradients` must match the parameters in count and shape.
    pub fn step(
        &mut self,
        parameters: &mut [DeviceTensor],
        gradients: &[DeviceTensor],
    ) -> Result<(), BackendError> {
        if parameters.len() != gradients.len() {
            return Err(BackendError::Backend(format!(
                "{} parameters but {} gradients",
                parameters.len(),
                gradients.len()
            )));
        }
        self.step_count += 1;
        let step = self.step_count;
        let bias1 = 1.0 - self.beta1.powi(step as i32);
        let bias2 = 1.0 - self.beta2.powi(step as i32);
        let decay = 1.0 - self.learning_rate * self.weight_decay;
        if self.moments.len() < parameters.len() {
            self.moments.resize_with(parameters.len(), || None);
        }

        for index in 0..parameters.len() {
            let gradient = &gradients[index];
            if gradient.shape() != parameters[index].shape() {
                return Err(BackendError::Backend(format!(
                    "gradient shape {:?} does not match parameter shape {:?}",
                    gradient.shape(),
                    parameters[index].shape()
                )));
            }
            let (first, second) = match &self.moments[index] {
                Some(pair) => pair.clone(),
                None => (
                    parameters[index].zeros_like()?,
                    parameters[index].zeros_like()?,
                ),
            };

            let first = first
                .scale(self.beta1)?
                .add(&gradient.scale(1.0 - self.beta1)?)?;
            let second = second
                .scale(self.beta2)?
                .add(&gradient.mul(gradient)?.scale(1.0 - self.beta2)?)?;
            let first_hat = first.scale(1.0 / bias1)?;
            let second_hat = second.scale(1.0 / bias2)?;
            let denominator = second_hat.sqrt()?.add_scalar(self.eps)?;
            let update = first_hat.div(&denominator)?;

            parameters[index] = parameters[index]
                .scale(decay)?
                .add(&update.scale(-self.learning_rate)?)?;
            self.moments[index] = Some((first, second));
        }
        Ok(())
    }
}
