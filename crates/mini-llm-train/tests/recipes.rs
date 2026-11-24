//! Tests for the preference-optimization recipes.

use mini_llm_autograd::TensorValue;
use mini_llm_train::{dpo_loss, grpo_loss, sequence_log_prob, TrainError};

fn log_softmax(value: f32, row: &[f32]) -> f32 {
    let max = row.iter().copied().fold(f32::NEG_INFINITY, f32::max);
    let sum: f32 = row.iter().map(|x| (x - max).exp()).sum();
    value - max - sum.ln()
}

#[test]
fn sequence_log_prob_sums_only_weighted_positions() {
    let logits = TensorValue::constant(vec![0.4, -0.2, 0.1, 0.9, -0.7, 0.3], vec![2, 3]).unwrap();
    let value = sequence_log_prob(&logits, &[1, 2], &[1.0, 0.0]).unwrap();
    let expected = log_softmax(-0.2, &[0.4, -0.2, 0.1]);
    assert!((value.data()[0] - expected).abs() < 1e-5);
}

#[test]
fn sequence_log_prob_rejects_bad_targets() {
    let logits = TensorValue::constant(vec![0.1, 0.2, 0.3], vec![1, 3]).unwrap();
    assert!(matches!(
        sequence_log_prob(&logits, &[3], &[1.0]),
        Err(TrainError::Model(_))
    ));
    assert!(matches!(
        sequence_log_prob(&logits, &[0, 1], &[1.0]),
        Err(TrainError::ShapeMismatch { .. })
    ));
}

#[test]
fn dpo_loss_prefers_the_chosen_response() {
    let favored = dpo_loss(
        &TensorValue::constant(vec![-1.0], vec![1]).unwrap(),
        &TensorValue::constant(vec![-2.0], vec![1]).unwrap(),
        &TensorValue::constant(vec![-1.5], vec![1]).unwrap(),
        &TensorValue::constant(vec![-1.5], vec![1]).unwrap(),
        0.1,
    )
    .unwrap();
    let disfavored = dpo_loss(
        &TensorValue::constant(vec![-2.0], vec![1]).unwrap(),
        &TensorValue::constant(vec![-1.0], vec![1]).unwrap(),
        &TensorValue::constant(vec![-1.5], vec![1]).unwrap(),
        &TensorValue::constant(vec![-1.5], vec![1]).unwrap(),
        0.1,
    )
    .unwrap();
    assert!(favored.data()[0] < disfavored.data()[0]);
}

#[test]
fn dpo_gradient_pushes_the_policy_towards_the_chosen_response() {
    let chosen = TensorValue::parameter(vec![-1.0], vec![1]).unwrap();
    let rejected = TensorValue::parameter(vec![-2.0], vec![1]).unwrap();
    let reference_chosen = TensorValue::constant(vec![-1.5], vec![1]).unwrap();
    let reference_rejected = TensorValue::constant(vec![-1.5], vec![1]).unwrap();

    let loss = dpo_loss(
        &chosen,
        &rejected,
        &reference_chosen,
        &reference_rejected,
        0.1,
    )
    .unwrap();
    loss.backward().unwrap();

    // Raising the chosen log-prob lowers the loss; raising the rejected one
    // raises it.
    assert!(chosen.grad().unwrap().data()[0] < 0.0);
    assert!(rejected.grad().unwrap().data()[0] > 0.0);
}

#[test]
fn grpo_advantages_are_zero_when_rewards_are_equal() {
    let log_probs = TensorValue::parameter(vec![-0.5, -1.5], vec![2, 1]).unwrap();
    let loss = grpo_loss(&log_probs, &[1.0, 1.0]).unwrap();
    assert!(loss.data()[0].abs() < 1e-6);
    loss.backward().unwrap();
    for grad in log_probs.grad().unwrap().data() {
        assert!(grad.abs() < 1e-6);
    }
}

#[test]
fn grpo_rewards_higher_scoring_actions() {
    let log_probs = TensorValue::parameter(vec![-0.5, -1.5], vec![2, 1]).unwrap();
    let loss = grpo_loss(&log_probs, &[2.0, 0.0]).unwrap();
    loss.backward().unwrap();
    let grad = log_probs.grad().unwrap();
    // dL/dlogp = -advantage / group, and advantages are +1 / -1.
    assert!(grad.data()[0] < 0.0);
    assert!(grad.data()[1] > 0.0);
}

#[test]
fn grpo_rejects_invalid_rewards() {
    let log_probs = TensorValue::constant(vec![-0.5], vec![1, 1]).unwrap();
    assert!(matches!(
        grpo_loss(&log_probs, &[f32::NAN]),
        Err(TrainError::InvalidReward { .. })
    ));
    assert!(matches!(
        grpo_loss(&log_probs, &[1.0, 2.0]),
        Err(TrainError::ShapeMismatch { .. })
    ));
}
