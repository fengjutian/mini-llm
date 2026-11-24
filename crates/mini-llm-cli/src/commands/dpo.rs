//! DPO: teach a tiny policy to prefer one response over another.

use mini_llm_autograd::TensorValue;
use mini_llm_nn::{AdamW, Linear, Module, Optimizer, Rng};
use mini_llm_train::{dpo_loss, sequence_log_prob};

const CHOSEN: usize = 0;
const REJECTED: usize = 1;
const BETA: f32 = 0.5;
const STEPS: usize = 300;

pub fn run() {
    let mut rng = Rng::new(77);
    let policy = Linear::new(1, 3, &mut rng).expect("policy");
    let input = TensorValue::constant(vec![1.0], vec![1, 1]).expect("input");

    // The reference policy is a frozen copy of the initial policy.
    let reference_logits = TensorValue::new(
        input
            .value()
            .matmul(&policy.weight().value())
            .expect("matmul")
            .add(&policy.bias().expect("bias").value())
            .expect("bias add"),
    );

    let parameters = policy.parameters();
    let mut optimizer = AdamW::new(0.05);

    println!("DPO: three candidate responses, one chosen and one rejected");
    println!("  beta {BETA}, reference = frozen initial policy");
    println!();

    for step in 0..=STEPS {
        let logits = policy.forward(&input).expect("forward");
        let policy_chosen = sequence_log_prob(&logits, &[CHOSEN], &[1.0]).expect("logp");
        let policy_rejected = sequence_log_prob(&logits, &[REJECTED], &[1.0]).expect("logp");
        let reference_chosen =
            sequence_log_prob(&reference_logits, &[CHOSEN], &[1.0]).expect("logp");
        let reference_rejected =
            sequence_log_prob(&reference_logits, &[REJECTED], &[1.0]).expect("logp");

        if step % 100 == 0 {
            let (chosen, rejected) = (policy_chosen.data()[0], policy_rejected.data()[0]);
            println!(
                "  step {step:>3}  logp(chosen) {chosen:>7.3}  logp(rejected) {rejected:>7.3}  margin {:>7.3}",
                chosen - rejected
            );
        }

        let loss = dpo_loss(
            &policy_chosen,
            &policy_rejected,
            &reference_chosen,
            &reference_rejected,
            BETA,
        )
        .expect("dpo loss");
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
    println!();
    println!(
        "  final probabilities: chosen {:.3}, rejected {:.3}",
        probabilities[CHOSEN], probabilities[REJECTED]
    );
}
