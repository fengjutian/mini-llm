//! A two-blob classifier trained with softmax + cross-entropy.

use mini_llm_autograd::TensorValue;
use mini_llm_nn::{AdamW, Linear, Module, Optimizer, Relu, Rng, Sequential};

const PER_CLASS: usize = 80;
const STEPS: usize = 400;
const HIDDEN: usize = 16;

pub fn run() {
    let mut rng = Rng::new(2024);
    let mut features = Vec::with_capacity(PER_CLASS * 4);
    let mut labels = Vec::with_capacity(PER_CLASS * 2);

    for index in 0..PER_CLASS * 2 {
        let (class, centre) = if index % 2 == 0 {
            (0usize, (-1.2f32, -1.0f32))
        } else {
            (1usize, (1.2f32, 1.0f32))
        };
        features.push(centre.0 + rng.normal() * 0.45);
        features.push(centre.1 + rng.normal() * 0.45);
        labels.push(class);
    }

    let batch = labels.len();
    let features = TensorValue::constant(features, vec![batch, 2]).expect("valid features");

    let model = Sequential::new(vec![
        Box::new(Linear::new(2, HIDDEN, &mut rng).expect("hidden layer")),
        Box::new(Relu),
        Box::new(Linear::new(HIDDEN, 2, &mut rng).expect("output layer")),
    ]);
    let parameters = model.parameters();
    let mut optimizer = AdamW::new(0.05);

    println!(
        "two-blob classification: {} samples, 2 -> {HIDDEN} -> 2",
        labels.len()
    );
    println!("  model: {}", model.summary());
    println!();

    for step in 0..=STEPS {
        let logits = model.forward(&features).expect("forward");
        let loss = logits.cross_entropy(&labels).expect("cross entropy");

        if step % 100 == 0 {
            let accuracy = accuracy(&logits.data(), &labels);
            println!(
                "  step {step:>3}  loss {:>8.4}  accuracy {:>5.1}%",
                loss.data()[0],
                accuracy * 100.0
            );
        }

        if step == STEPS {
            break;
        }
        loss.backward().expect("backward");
        optimizer.step(&parameters).expect("optimizer step");
        loss.zero_grad();
    }

    let logits = model.forward(&features).expect("forward");
    println!();
    println!(
        "  final accuracy: {:.1}%",
        accuracy(&logits.data(), &labels) * 100.0
    );
}

fn accuracy(logits: &[f32], labels: &[usize]) -> f32 {
    let classes = logits.len() / labels.len();
    let correct = labels
        .iter()
        .enumerate()
        .filter(|(row, &label)| {
            let start = row * classes;
            let row_logits = &logits[start..start + classes];
            let mut best = 0usize;
            for (index, &value) in row_logits.iter().enumerate() {
                if value > row_logits[best] {
                    best = index;
                }
            }
            best == label
        })
        .count();
    correct as f32 / labels.len() as f32
}
