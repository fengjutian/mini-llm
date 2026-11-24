//! LoRA: fine-tune a frozen classifier by training only low-rank adapters.

use mini_llm_autograd::TensorValue;
use mini_llm_nn::{AdamW, Linear, LoraLinear, Module, Optimizer, Relu, Rng};

const PER_CLASS: usize = 60;
const HIDDEN: usize = 16;
const RANK: usize = 4;

pub fn run() {
    let mut rng = Rng::new(2025);

    // 1. Pretrain a small classifier on task A.
    let hidden = Linear::new(2, HIDDEN, &mut rng).expect("hidden layer");
    let output = Linear::new(HIDDEN, 2, &mut rng).expect("output layer");
    let (features_a, labels_a) = blobs(&mut rng, (-1.0, -1.0), (1.0, 1.0));
    let (features_b, labels_b) = blobs(&mut rng, (-1.2, 1.2), (1.2, -1.2));

    let base_parameters: Vec<_> = hidden
        .parameters()
        .into_iter()
        .chain(output.parameters())
        .collect();
    let mut optimizer = AdamW::new(0.05);
    let input_a = input(&features_a);
    train_classifier(
        &input_a,
        &labels_a,
        &mut optimizer,
        &base_parameters,
        200,
        |x| logits(x, &hidden, &output),
    );

    println!("LoRA: frozen classifier adapted to a new decision boundary");
    println!(
        "  base: {} params, rank {RANK} adapters",
        base_parameters.len()
    );
    println!(
        "  task A accuracy: {:.0}%",
        accuracy(&logits(&input_a, &hidden, &output).data(), &labels_a) * 100.0
    );

    // 2. Freeze the base and wrap it in LoRA adapters.
    let hidden_before = hidden.weight().data();
    let lora_hidden = LoraLinear::new(hidden, RANK, 8.0, &mut rng).expect("hidden lora");
    let lora_output = LoraLinear::new(output, RANK, 8.0, &mut rng).expect("output lora");
    let adapter_parameters: Vec<_> = lora_hidden
        .parameters()
        .into_iter()
        .chain(lora_output.parameters())
        .collect();

    let input_b = input(&features_b);
    let before: f32 = accuracy(
        &logits(&input_b, &lora_hidden, &lora_output).data(),
        &labels_b,
    );

    // 3. Train only the adapters on task B.
    let mut adapter_optimizer = AdamW::new(0.05);
    train_classifier(
        &input_b,
        &labels_b,
        &mut adapter_optimizer,
        &adapter_parameters,
        300,
        |x| logits(x, &lora_hidden, &lora_output),
    );
    let after: f32 = accuracy(
        &logits(&input_b, &lora_hidden, &lora_output).data(),
        &labels_b,
    );

    println!("  task B accuracy before adapters: {:.0}%", before * 100.0);
    println!("  task B accuracy after adapters:  {:.0}%", after * 100.0);
    println!(
        "  base weights unchanged: {}",
        lora_hidden.base().weight().data() == hidden_before
    );
    println!(
        "  adapter tensors: {} ({} scalars)",
        adapter_parameters.len(),
        adapter_parameters
            .iter()
            .map(|parameter| parameter.value().len())
            .sum::<usize>()
    );
}

fn logits(x: &TensorValue, hidden: &impl Module, output: &impl Module) -> TensorValue {
    let h = Relu
        .forward(&hidden.forward(x).expect("hidden forward"))
        .expect("relu");
    output.forward(&h).expect("output forward")
}

fn train_classifier(
    features: &TensorValue,
    labels: &[usize],
    optimizer: &mut AdamW,
    parameters: &[TensorValue],
    steps: usize,
    forward: impl Fn(&TensorValue) -> TensorValue,
) {
    for _ in 0..steps {
        let logits = forward(features);
        let loss = logits.cross_entropy(labels).expect("cross entropy");
        loss.backward().expect("backward");
        optimizer.step(parameters).expect("optimizer step");
        loss.zero_grad();
    }
}

fn input(features: &[f32]) -> TensorValue {
    TensorValue::constant(features.to_vec(), vec![features.len() / 2, 2]).expect("valid input")
}

fn blobs(rng: &mut Rng, first: (f32, f32), second: (f32, f32)) -> (Vec<f32>, Vec<usize>) {
    let mut features = Vec::with_capacity(PER_CLASS * 4);
    let mut labels = Vec::with_capacity(PER_CLASS * 2);
    for index in 0..PER_CLASS * 2 {
        let (class, centre) = if index % 2 == 0 {
            (0usize, first)
        } else {
            (1usize, second)
        };
        features.push(centre.0 + rng.normal() * 0.35);
        features.push(centre.1 + rng.normal() * 0.35);
        labels.push(class);
    }
    (features, labels)
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
