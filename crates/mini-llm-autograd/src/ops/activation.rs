//! Element-wise activations, log-softmax and cross-entropy.

use mini_llm_tensor::Tensor;

use crate::error::AutogradError;
use crate::graph::matrix_shape;
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

    /// `log(sigmoid(x))`, computed in a numerically stable way.
    ///
    /// This is the building block of preference losses such as DPO.
    pub fn log_sigmoid(&self) -> Result<TensorValue, AutogradError> {
        let x = self.value();
        let value = x.map(|v| {
            // log(sigmoid(v)) = min(v, 0) - log(1 + exp(-|v|))
            v.min(0.0) - (1.0 + (-v.abs()).exp()).ln()
        })?;
        self.unary("log_sigmoid", value, move |g| {
            let slope = x.map(|v| 1.0 / (1.0 + v.exp()))?;
            Ok(g.mul(&slope)?)
        })
    }

    /// Row-wise softmax of a rank-2 value.
    pub fn softmax_rows(&self) -> Result<TensorValue, AutogradError> {
        let input = self.value();
        let value = crate::backend::current()
            .softmax_rows(&input)
            .map_err(|error| AutogradError::Backend(error.to_string()))?;
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

    /// Row-wise log-softmax of a rank-2 value.
    ///
    /// Forward: `x - logsumexp(x)`; backward: `G - softmax(x) * sum(G)`.
    pub fn log_softmax_rows(&self) -> Result<TensorValue, AutogradError> {
        let logits = self.value();
        let (rows, cols) = matrix_shape(&logits)?;
        if cols == 0 {
            return Err(AutogradError::Tensor(
                mini_llm_tensor::TensorError::EmptySoftmaxDimension,
            ));
        }
        let input = logits.data();
        let mut out = vec![0.0f32; input.len()];
        for row in 0..rows {
            let start = row * cols;
            let slice = &input[start..start + cols];
            let max = slice.iter().copied().fold(f32::NEG_INFINITY, f32::max);
            let sum: f32 = slice.iter().map(|value| (value - max).exp()).sum();
            let logsumexp = max + sum.ln();
            for col in 0..cols {
                out[start + col] = slice[col] - logsumexp;
            }
        }
        let value = Tensor::new(out, logits.shape().to_vec())?;
        let saved_logits = logits.clone();
        self.unary("log_softmax_rows", value, move |g| {
            let probabilities = saved_logits.softmax_rows()?;
            let upstream = g.data();
            let probability_data = probabilities.data();
            let mut grad = vec![0.0f32; upstream.len()];
            for row in 0..rows {
                let start = row * cols;
                let total: f32 = upstream[start..start + cols].iter().sum();
                for col in 0..cols {
                    grad[start + col] =
                        upstream[start + col] - probability_data[start + col] * total;
                }
            }
            Ok(Tensor::new(grad, vec![rows, cols])?)
        })
    }

    /// Cross-entropy between rank-2 logits (`[batch, vocab]`) and token ids.
    ///
    /// Returns the mean negative log-likelihood as a scalar (shape `[1]`).
    pub fn cross_entropy(&self, targets: &[usize]) -> Result<TensorValue, AutogradError> {
        let weights = vec![1.0f32; targets.len()];
        self.cross_entropy_weighted(targets, &weights)
    }

    /// Cross-entropy with one weight per target position.
    ///
    /// A weight of `0.0` masks the position out entirely (no loss, no gradient),
    /// which is how supervised fine-tuning ignores prompt tokens. The loss is
    /// normalized by the sum of the weights.
    pub fn cross_entropy_weighted(
        &self,
        targets: &[usize],
        weights: &[f32],
    ) -> Result<TensorValue, AutogradError> {
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
        if weights.len() != targets.len() {
            return Err(AutogradError::WeightCountMismatch {
                targets: targets.len(),
                weights: weights.len(),
            });
        }
        if let Some(index) = targets.iter().copied().find(|&index| index >= cols) {
            return Err(AutogradError::TargetOutOfRange {
                index,
                vocab_size: cols,
            });
        }
        if let Some((index, value)) = weights
            .iter()
            .copied()
            .enumerate()
            .find(|(_, value)| !value.is_finite() || *value < 0.0)
        {
            return Err(AutogradError::InvalidWeight { index, value });
        }

        let total: f32 = weights.iter().sum();
        if total <= 0.0 {
            return Err(AutogradError::ZeroTotalWeight);
        }

        let probabilities = logits.softmax_rows()?;
        let probability_data = probabilities.data().to_vec();
        let loss = targets
            .iter()
            .enumerate()
            .map(|(row, &target)| -weights[row] * probability_data[row * cols + target].ln())
            .sum::<f32>()
            / total;

        let value = Tensor::new(vec![loss], vec![1])?;
        let saved_targets = targets.to_vec();
        let saved_weights = weights.to_vec();
        self.unary("cross_entropy", value, move |g| {
            let upstream = g.data().first().copied().unwrap_or(1.0);
            let mut grad = probability_data.clone();
            for (row, &target) in saved_targets.iter().enumerate() {
                grad[row * cols + target] -= 1.0;
                let scale = upstream * saved_weights[row] / total;
                for col in 0..cols {
                    grad[row * cols + col] *= scale;
                }
            }
            Ok(Tensor::new(grad, vec![rows, cols])?)
        })
    }
}
