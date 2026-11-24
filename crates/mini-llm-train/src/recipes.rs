//! Preference-optimization recipes built on the autograd engine.
//!
//! - [`sequence_log_prob`]: log-probability of a response under a language model.
//! - [`dpo_loss`]: direct preference optimization (Rafailov et al., 2023).
//! - [`grpo_loss`]: group relative policy optimization (DeepSeekMath, 2024).

use mini_llm_autograd::TensorValue;
use mini_llm_tensor::Tensor;

use crate::TrainError;

/// Sum of per-token log-probabilities for `targets`, weighted by `weights`.
///
/// `logits` is `[rows, vocab]`; a weight of zero ignores that position, which is
/// how prompt tokens are excluded from a response's log-probability.
pub fn sequence_log_prob(
    logits: &TensorValue,
    targets: &[usize],
    weights: &[f32],
) -> Result<TensorValue, TrainError> {
    let shape = logits.shape();
    let (rows, vocab) = (shape[0], shape[1]);
    if targets.len() != rows || weights.len() != rows {
        return Err(TrainError::ShapeMismatch {
            expected: vec![rows, vocab],
            actual: vec![targets.len(), weights.len()],
        });
    }
    if let Some(&target) = targets.iter().find(|&&target| target >= vocab) {
        return Err(TrainError::Model(mini_llm_model::ModelError::Autograd(
            mini_llm_autograd::AutogradError::TargetOutOfRange {
                index: target,
                vocab_size: vocab,
            },
        )));
    }

    let mut one_hot = vec![0.0f32; rows * vocab];
    for (row, &target) in targets.iter().enumerate() {
        one_hot[row * vocab + target] = 1.0;
    }
    let mask = TensorValue::new(Tensor::new(one_hot, vec![rows, vocab])?);
    let weight_column = TensorValue::new(Tensor::new(weights.to_vec(), vec![rows, 1])?);

    let log_probabilities = logits.log_softmax_rows()?;
    Ok(log_probabilities.mul(&mask)?.mul(&weight_column)?.sum()?)
}

/// Direct preference optimization loss.
///
/// `loss = -logsigmoid(beta * ((logπ(y_w) - logπ(y_l)) - (logπ_ref(y_w) - logπ_ref(y_l))))`
///
/// Each argument is a scalar log-probability (shape `[1]`).
pub fn dpo_loss(
    policy_chosen: &TensorValue,
    policy_rejected: &TensorValue,
    reference_chosen: &TensorValue,
    reference_rejected: &TensorValue,
    beta: f32,
) -> Result<TensorValue, TrainError> {
    let policy_margin = policy_chosen.sub(policy_rejected)?;
    let reference_margin = reference_chosen.sub(reference_rejected)?;
    let advantage = policy_margin.sub(&reference_margin)?.scale(beta)?;
    Ok(advantage.log_sigmoid()?.mean()?.scale(-1.0)?)
}

/// Group relative policy optimization loss.
///
/// `token_log_probs[i]` is the log-probability of the action sampled for group
/// member `i`. Rewards are normalized within the group to form the advantage,
/// so no value network is needed.
pub fn grpo_loss(
    token_log_probs: &TensorValue,
    rewards: &[f32],
) -> Result<TensorValue, TrainError> {
    let group = token_log_probs.shape()[0];
    if group != rewards.len() || group == 0 {
        return Err(TrainError::ShapeMismatch {
            expected: vec![group, 1],
            actual: vec![rewards.len(), 1],
        });
    }
    if let Some((index, value)) = rewards
        .iter()
        .copied()
        .enumerate()
        .find(|(_, reward)| !reward.is_finite())
    {
        return Err(TrainError::InvalidReward { index, value });
    }

    let mean = rewards.iter().sum::<f32>() / group as f32;
    let variance = rewards
        .iter()
        .map(|reward| (reward - mean) * (reward - mean))
        .sum::<f32>()
        / group as f32;
    let std = (variance + 1e-8).sqrt();
    let advantages: Vec<f32> = rewards.iter().map(|reward| (reward - mean) / std).collect();
    let advantage_column = TensorValue::new(Tensor::new(advantages, vec![group, 1])?);
    Ok(token_log_probs
        .mul(&advantage_column)?
        .mean()?
        .scale(-1.0)?)
}
