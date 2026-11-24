//! Element-wise activations and softmax / cross-entropy.

use mini_llm_tensor::Tensor;

use crate::error::AutogradError;
use crate::value::TensorValue;

impl TensorValue {
    pub fn relu(&self) -> Result<TensorValue, AutogradError> {
        let x = self.value();
        let value = x.relu()?;
        self.unary("relu", value, move |g| {
            let mask = x.map(|v| if v > 0.0 { 1.0 } else { 0.0 })?;
            Ok(g.mul(&mask)?)
        })
    }

    pub fn tanh(&self) -> Result<TensorValue, AutogradError> {
        let value = self.value().tanh()?;
        let saved = value.clone();
        self.unary("tanh", value, move |g| {
            // d/dx tanh(x) = 1 - tanh(x)^2
            let slope = saved.map(|v| 1.0 - v * v)?;
            Ok(g.mul(&slope)?)
        })
    }

    pub fn sigmoid(&self) -> Result<TensorValue, AutogradError> {
        let value = self.value().sigmoid()?;
        let saved = value.clone();
        self.unary("sigmoid", value, move |g| {
            // d/dx sigmoid(x) = sigmoid(x) * (1 - sigmoid(x))
            let slope = saved.map(|v| v * (1.0 - v))?;
            Ok(g.mul(&slope)?)
        })
    }

    /// Row-wise softmax of a rank-2 value.
    pub fn softmax_rows(&self) -> Result<TensorValue, AutogradError> {
        let value = self.value().softmax_rows()?;
        let saved = value.clone();
        self.unary("softmax_rows", value, move |g| {
            let shape = saved.shape().to_vec();
            let (rows, cols) = (shape[0], shape[1]);
            let (soft, upstream) = (saved.data(), g.data());
            let mut out = vec![0.0; soft.len()];
            for row in 0..rows {
                let start = row * cols;
                let dot: f32 = (0..cols)
                    .map(|c| upstream[start + c] * soft[start + c])
                    .sum();
                for col in 0..cols {
                    out[start + col] = soft[start + col] * (upstream[start + col] - dot);
                }
            }
            Ok(Tensor::new(out, shape)?)
        })
    }

    /// Cross-entropy between rank-2 logits (`[batch, vocab]`) and token ids.
    ///
    /// Returns the mean negative log-likelihood as a scalar (shape `[1]`).
    pub fn cross_entropy(&self, targets: &[usize]) -> Result<TensorValue, AutogradError> {
        let logits = self.value();
        if logits.rank() != 2 {
            return Err(AutogradError::Tensor(
                mini_llm_tensor::TensorError::ExpectedRank {
                    expected: 2,
                    actual: logits.rank(),
                },
            ));
        }
        let (rows, cols) = (logits.shape()[0], logits.shape()[1]);
        if targets.len() != rows {
            return Err(AutogradError::TargetCountMismatch {
                logits_rows: rows,
                targets: targets.len(),
            });
        }
        if let Some(index) = targets.iter().copied().find(|&index| index >= cols) {
            return Err(AutogradError::TargetOutOfRange {
                index,
                vocab_size: cols,
            });
        }

        let probabilities = logits.softmax_rows()?;
        let probability_data = probabilities.data().to_vec();
        let loss = -targets
            .iter()
            .enumerate()
            .map(|(row, &target)| probability_data[row * cols + target].ln())
            .sum::<f32>()
            / rows as f32;

        let value = Tensor::new(vec![loss], vec![1])?;
        let saved_targets = targets.to_vec();
        self.unary("cross_entropy", value, move |g| {
            let scale = g.data().first().copied().unwrap_or(1.0) / rows as f32;
            let mut grad = probability_data.clone();
            for (row, &target) in saved_targets.iter().enumerate() {
                grad[row * cols + target] -= 1.0;
            }
            for entry in grad.iter_mut() {
                *entry *= scale;
            }
            Ok(Tensor::new(grad, vec![rows, cols])?)
        })
    }
}
