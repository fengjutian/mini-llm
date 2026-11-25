//! A top-k gated mixture-of-experts feed-forward layer.
//!
//! Routing is explicit and easy to inspect:
//!
//! ```text
//! logits = router(x)                # [rows, experts]
//! probs  = softmax(logits, dim=-1)
//! topk   = top-k(probs)             # selected experts per row
//! y      = sum_e probs[:, e] * expert_e(x[rows assigned to e])
//! ```
//!
//! Dispatch is sparse: an expert is evaluated only on the rows that routed to
//! it, so `top_k / num_experts` of the expert compute is performed. The router
//! still receives gradients through the selected weights.

use mini_llm_autograd::TensorValue;
use mini_llm_random::Rng;

use crate::{expect_matrix, Linear, Module, NnError, SwiGlu};

/// A top-k gated mixture of [`SwiGlu`] experts.
pub struct Moe {
    router: Linear,
    experts: Vec<SwiGlu>,
    num_experts: usize,
    top_k: usize,
    hidden: usize,
}

impl Moe {
    pub fn new(
        hidden: usize,
        inner: usize,
        num_experts: usize,
        top_k: usize,
        rng: &mut Rng,
    ) -> Result<Self, NnError> {
        if num_experts == 0 {
            return Err(NnError::InvalidDimension {
                name: "num_experts",
                value: num_experts,
            });
        }
        if top_k == 0 || top_k > num_experts {
            return Err(NnError::InvalidTopK { top_k, num_experts });
        }
        let router = Linear::without_bias(hidden, num_experts, rng)?;
        let mut experts = Vec::with_capacity(num_experts);
        for _ in 0..num_experts {
            experts.push(SwiGlu::new(hidden, inner, rng)?);
        }
        Ok(Self {
            router,
            experts,
            num_experts,
            top_k,
            hidden,
        })
    }

    pub fn num_experts(&self) -> usize {
        self.num_experts
    }

    pub fn top_k(&self) -> usize {
        self.top_k
    }

    /// Router probabilities for `input`, useful for inspection.
    pub fn routing_probabilities(&self, input: &TensorValue) -> Result<TensorValue, NnError> {
        expect_matrix(input, self.hidden)?;
        Ok(self.router.forward(input)?.softmax_rows()?)
    }

    /// Dense reference path: evaluates every expert on every row.
    ///
    /// Kept for parity tests and for readers comparing dense and sparse
    /// execution; [`Module::forward`] uses sparse dispatch instead.
    pub fn forward_dense(&self, input: &TensorValue) -> Result<TensorValue, NnError> {
        use mini_llm_tensor::Tensor;

        let rows = expect_matrix(input, self.hidden)?;
        let probabilities = self.routing_probabilities(input)?;
        let mask = top_k_mask(&probabilities.data(), rows, self.num_experts, self.top_k);
        let mask_tensor = TensorValue::new(Tensor::from_2d(rows, self.num_experts, mask)?);
        let gate = probabilities.mul(&mask_tensor)?;

        let mut output: Option<TensorValue> = None;
        for (index, expert) in self.experts.iter().enumerate() {
            let weight = gate.slice_cols(index, 1)?;
            let transformed = expert.forward(input)?;
            let contribution = transformed.mul(&weight)?;
            output = Some(match output {
                Some(accumulated) => accumulated.add(&contribution)?,
                None => contribution,
            });
        }
        output.ok_or(NnError::InvalidDimension {
            name: "num_experts",
            value: 0,
        })
    }

    /// Number of rows whose top-1 choice is each expert.
    pub fn routing_counts(&self, input: &TensorValue) -> Result<Vec<usize>, NnError> {
        let rows = expect_matrix(input, self.hidden)?;
        let probabilities = self.routing_probabilities(input)?;
        let data = probabilities.data();
        let mut counts = vec![0usize; self.num_experts];
        for row in 0..rows {
            let slice = &data[row * self.num_experts..(row + 1) * self.num_experts];
            counts[argmax(slice)] += 1;
        }
        Ok(counts)
    }
}

impl Module for Moe {
    fn forward(&self, input: &TensorValue) -> Result<TensorValue, NnError> {
        let rows = expect_matrix(input, self.hidden)?;
        let probabilities = self.routing_probabilities(input)?;
        let assignments = top_k_indices(&probabilities.data(), rows, self.num_experts, self.top_k);

        let mut output: Option<TensorValue> = None;
        for (index, expert) in self.experts.iter().enumerate() {
            let token_rows: Vec<usize> = assignments
                .iter()
                .enumerate()
                .filter(|(_, selected)| selected.contains(&index))
                .map(|(row, _)| row)
                .collect();
            if token_rows.is_empty() {
                continue;
            }

            let hidden = input.gather_rows(&token_rows)?;
            let transformed = expert.forward(&hidden)?;
            let weight = probabilities
                .slice_cols(index, 1)?
                .gather_rows(&token_rows)?;
            let contribution = transformed
                .mul(&weight)?
                .scatter_add_rows(&token_rows, rows)?;

            output = Some(match output {
                Some(accumulated) => accumulated.add(&contribution)?,
                None => contribution,
            });
        }

        output.ok_or(NnError::InvalidDimension {
            name: "num_experts",
            value: 0,
        })
    }

    fn parameters(&self) -> Vec<TensorValue> {
        let mut parameters = self.router.parameters();
        for expert in &self.experts {
            parameters.extend(expert.parameters());
        }
        parameters
    }

    fn name(&self) -> &'static str {
        "moe"
    }

    fn summary(&self) -> String {
        format!(
            "moe({} experts, top-{}) ({} params)",
            self.num_experts,
            self.top_k,
            self.num_parameters()
        )
    }
}

/// Constant 0/1 mask of the top-k experts per row, used by the dense reference.
fn top_k_mask(probabilities: &[f32], rows: usize, experts: usize, top_k: usize) -> Vec<f32> {
    let mut mask = vec![0.0f32; rows * experts];
    for row in 0..rows {
        let slice = &probabilities[row * experts..(row + 1) * experts];
        let mut order: Vec<usize> = (0..experts).collect();
        order.sort_by(|&left, &right| slice[right].total_cmp(&slice[left]));
        for &expert in order.iter().take(top_k) {
            mask[row * experts + expert] = 1.0;
        }
    }
    mask
}

/// Selected expert ids per row, ordered by descending router probability.
fn top_k_indices(
    probabilities: &[f32],
    rows: usize,
    experts: usize,
    top_k: usize,
) -> Vec<Vec<usize>> {
    (0..rows)
        .map(|row| {
            let slice = &probabilities[row * experts..(row + 1) * experts];
            let mut order: Vec<usize> = (0..experts).collect();
            order.sort_by(|&left, &right| slice[right].total_cmp(&slice[left]));
            order.truncate(top_k);
            order
        })
        .collect()
}

fn argmax(values: &[f32]) -> usize {
    let mut best = 0usize;
    for (index, &value) in values.iter().enumerate() {
        if value > values[best] {
            best = index;
        }
    }
    best
}
