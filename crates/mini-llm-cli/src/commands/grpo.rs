//! GRPO on a toy bandit: the policy learns which arm pays out.

use mini_llm_autograd::TensorValue;
use mini_llm_nn::{AdamW, Linear, Module, Optimizer, Rng};
use mini_llm_tensor::Tensor;
use mini_llm_train::grpo_loss;

const GROUP: usize = 8;
const ARMS: usize = 5;
const TARGET_ARM: usize = 2;
const STEPS: usize = 200;

pub fn run() {
    let mut rng = Rng::new(31);
    let policy = Linear::new(1, ARMS, &mut rng).expect("policy");
    let input = TensorValue::constant(vec![1.0; GROUP], vec![GROUP, 1]).expect("input");
    let parameters = policy.parameters();
    let mut optimizer = AdamW::new(0.1);

    println!("GRPO bandit: {ARMS} arms, group size {GROUP}, arm {TARGET_ARM} pays 1.0");
    println!();

    for step in 0..=STEPS {
        let logits = policy.forward(&input).expect("forward");
        let probabilities = logits.softmax_rows().expect("softmax").data();

        let mut actions = Vec::with_capacity(GROUP);
        let mut rewards = Vec::with_capacity(GROUP);
        for row in 0..GROUP {
            let action = sample(&probabilities[row * ARMS..(row + 1) * ARMS], &mut rng);
            actions.push(action);
            rewards.push(if action == TARGET_ARM { 1.0 } else { 0.0 });
        }

        if step % 50 == 0 {
            let target_mass: f32 = probabilities
                .chunks(ARMS)
                .map(|row| row[TARGET_ARM])
                .sum::<f32>()
                / GROUP as f32;
            let hits = rewards.iter().filter(|reward| **reward > 0.0).count();
            println!("  step {step:>3}  mean P(target) {target_mass:.3}  hits {hits}/{GROUP}");
        }

        let one_hot = one_hot(&actions);
        let action_log_probs = logits
            .log_softmax_rows()
            .expect("log softmax")
            .mul(&one_hot)
            .expect("mask")
            .sum_rows_keepdim()
            .expect("row sums");
        let loss = grpo_loss(&action_log_probs, &rewards).expect("grpo loss");
        loss.backward().expect("backward");
        optimizer.step(&parameters).expect("optimizer step");
        loss.zero_grad();
    }

    let probabilities = policy
        .forward(&input)
        .expect("forward")
        .softmax_rows()
        .expect("softmax")
        .data();
    let target_mass: f32 = probabilities
        .chunks(ARMS)
        .map(|row| row[TARGET_ARM])
        .sum::<f32>()
        / GROUP as f32;
    println!();
    println!("  final mean P(target arm): {target_mass:.3}");
}

fn one_hot(actions: &[usize]) -> TensorValue {
    let mut data = vec![0.0f32; actions.len() * ARMS];
    for (row, &action) in actions.iter().enumerate() {
        data[row * ARMS + action] = 1.0;
    }
    TensorValue::new(Tensor::from_2d(actions.len(), ARMS, data).expect("one-hot"))
}

fn sample(probabilities: &[f32], rng: &mut Rng) -> usize {
    let mut threshold = rng.next_f32();
    for (index, &probability) in probabilities.iter().enumerate() {
        threshold -= probability;
        if threshold <= 0.0 {
            return index;
        }
    }
    probabilities.len().saturating_sub(1)
}
