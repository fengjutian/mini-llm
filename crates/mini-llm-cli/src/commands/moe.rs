//! Mixture-of-experts routing demo.

use mini_llm_autograd::TensorValue;
use mini_llm_nn::{AdamW, Module, Moe, Optimizer, Rng};

pub fn run() {
    let mut rng = Rng::new(7);
    let layer = Moe::new(8, 16, 4, 2, &mut rng).expect("moe");

    println!("mixture of experts (dense execution, top-2 gating)");
    println!("  {}", layer.summary());

    let mut features = Vec::with_capacity(8 * 8);
    for _ in 0..8 * 8 {
        features.push(rng.normal());
    }
    let input = TensorValue::constant(features, vec![8, 8]).expect("valid input");

    let counts = layer.routing_counts(&input).expect("routing counts");
    let probabilities = layer
        .routing_probabilities(&input)
        .expect("routing probabilities");
    println!("  top-1 routing counts per expert: {counts:?}");
    println!(
        "  router probabilities (first two rows): {:?}",
        &probabilities.data()[..8]
    );

    let output = layer.forward(&input).expect("forward");
    let loss = output.powf(2.0).expect("square").mean().expect("mean");
    loss.backward().expect("backward");
    println!(
        "  output shape {:?}, loss {:.4}",
        output.shape(),
        loss.data()[0]
    );

    let parameters = layer.parameters();
    let mut optimizer = AdamW::new(1e-2);
    optimizer.step(&parameters).expect("optimizer step");
    println!(
        "  one AdamW step applied to {} parameter tensors",
        parameters.len()
    );
}
