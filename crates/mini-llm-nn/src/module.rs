//! The [`Module`] trait shared by every layer.

use mini_llm_autograd::TensorValue;

use crate::error::NnError;

/// A composable piece of a model: a forward function plus its parameters.
pub trait Module {
    /// Runs the forward pass on a rank-2 activation tensor.
    fn forward(&self, input: &TensorValue) -> Result<TensorValue, NnError>;

    /// Every learnable tensor owned by this module, in a stable order.
    fn parameters(&self) -> Vec<TensorValue>;

    /// Human-readable layer name, used in summaries.
    fn name(&self) -> &'static str {
        "module"
    }

    /// Total number of scalar parameters.
    fn num_parameters(&self) -> usize {
        self.parameters()
            .iter()
            .map(|parameter| parameter.value().len())
            .sum()
    }

    /// A one-line description of the module and its parameter count.
    fn summary(&self) -> String {
        format!("{} ({} params)", self.name(), self.num_parameters())
    }
}

/// Validates that `input` is rank-2 with `expected_width` columns.
pub(crate) fn expect_matrix(input: &TensorValue, expected_width: usize) -> Result<usize, NnError> {
    let shape = input.shape();
    if shape.len() != 2 {
        return Err(NnError::InputRank {
            expected: 2,
            actual: shape.len(),
        });
    }
    if shape[1] != expected_width {
        return Err(NnError::InputWidth {
            expected: expected_width,
            actual: shape[1],
        });
    }
    Ok(shape[0])
}
