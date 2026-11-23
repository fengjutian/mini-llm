//! A container that chains modules in order.

use mini_llm_autograd::TensorValue;

use crate::{Module, NnError};

/// Runs its children one after another: `x -> l0 -> l1 -> ...`.
pub struct Sequential {
    layers: Vec<Box<dyn Module>>,
}

impl Sequential {
    pub fn new(layers: Vec<Box<dyn Module>>) -> Self {
        Self { layers }
    }

    pub fn push(&mut self, layer: Box<dyn Module>) {
        self.layers.push(layer);
    }

    pub fn len(&self) -> usize {
        self.layers.len()
    }

    pub fn is_empty(&self) -> bool {
        self.layers.is_empty()
    }

    /// One summary line per child layer.
    pub fn describe(&self) -> Vec<String> {
        self.layers.iter().map(|layer| layer.summary()).collect()
    }
}

impl Module for Sequential {
    fn forward(&self, input: &TensorValue) -> Result<TensorValue, NnError> {
        let mut current = input.clone();
        for layer in &self.layers {
            current = layer.forward(&current)?;
        }
        Ok(current)
    }

    fn parameters(&self) -> Vec<TensorValue> {
        self.layers
            .iter()
            .flat_map(|layer| layer.parameters())
            .collect()
    }

    fn name(&self) -> &'static str {
        "sequential"
    }

    fn summary(&self) -> String {
        let inner: Vec<String> = self.describe();
        format!(
            "sequential[{}] ({} params)",
            inner.join(" -> "),
            self.num_parameters()
        )
    }
}
