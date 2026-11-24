//! XOR: a two-layer MLP trained with AdamW and MSE.

use mini_llm_autograd::TensorValue;
use mini_llm_nn::{AdamW, Linear, Module, Optimizer, Rng, Sequential, Tanh};

const STEPS: usize = 1000;

pub fn run() {
    let mut rng = Rng::new(42);
    let model = Sequential::new(vec![
        Box::new(Linear::new(2, 8, &mut rng).expect("hidden layer")),
        Box::new(Tanh),
        Box::new(Linear::new(8, 1, &mut rng).expect("output layer")),
    ]);
    let parameters = model.parameters();
    let mut optimizer = AdamW::new(0.02);

    let features = TensorValue::constant(vec![0.0, 0.0, 0.0, 1.0, 1.0, 0.0, 1.0, 1.0], vec![4, 2])
        .expect("valid features");
    let targets =
        TensorValue::constant(vec![0.0, 1.0, 1.0, 0.0], vec![4, 1]).expect("valid targets");

    println!("XOR with a 2 -> 8 -> 1 MLP (tanh hidden layer, AdamW, MSE)");
    println!("  model: {}", model.summary());
    println!();

    for step in 0..STEPS {
        let loss = model
            .forward(&features)
            .expect("forward")
            .sub(&targets)
            .expect("residual")
            .powf(2.0)
            .expect("square")
            .mean()
            .expect("mean");

        if step % 300 == 0 || step == STEPS - 1 {
            println!("  step {step:>4}  loss {:>10.6}", loss.data()[0]);
        }

        loss.backward().expect("backward");
        optimizer.step(&parameters).expect("optimizer step");
        loss.zero_grad();
    }

    let predictions = model.forward(&features).expect("forward").data();
    println!();
    for (index, prediction) in predictions.iter().enumerate() {
        let input = [[0, 0], [0, 1], [1, 0], [1, 1]][index];
        let target = if input[0] == input[1] { 0 } else { 1 };
        let verdict = if (prediction - target as f32).abs() < 0.2 {
            "ok"
        } else {
            "wrong"
        };
        println!(
            "  {:?} -> {:.3}  (target {}, {})",
            input, prediction, target, verdict
        );
    }
}
