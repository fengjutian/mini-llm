//! Trains a device-resident attention block on a tiny character corpus with a
//! held-out split, printing train and validation loss.
//!
//! The block is: embedding -> LayerNorm -> Q/K/V -> fused GQA attention ->
//! out-projection -> residual -> LayerNorm -> LM head -> cross-entropy.
//! Everything runs through `DeviceTensor` / `DeviceValue` (host storage) and
//! `DeviceAdamW`; only the corpus and the final printout touch plain Rust.
//!
//! Run with:
//! `cargo run -p mini-llm-backend --example device_attention_block`

use mini_llm_backend::{Device, DeviceAdamW, DeviceTensor, DeviceValue};
use mini_llm_tensor::Tensor;

const TEXT: &str = "the quick brown fox jumps over the lazy dog. the little language model learns to predict the next character. ";
const STEPS: usize = 150;
const PRINT_EVERY: usize = 10;
const LEARNING_RATE: f32 = 0.02;
const VALIDATION_FRACTION: f32 = 0.2;

fn sample(rows: usize, cols: usize, scale: f32, seed: usize) -> Tensor {
    let data = (0..rows * cols)
        .map(|index| {
            let mixed = (index * 37 + seed * 101 + 11) % 29;
            (mixed as f32 / 29.0 - 0.5) * 2.0 * scale
        })
        .collect();
    Tensor::from_2d(rows, cols, data).expect("valid tensor")
}

fn main() {
    let mut vocabulary: Vec<char> = TEXT.chars().collect();
    vocabulary.sort_unstable();
    vocabulary.dedup();
    let vocab = vocabulary.len();
    let encode = |character: char| {
        vocabulary
            .binary_search(&character)
            .expect("character is in the vocabulary")
    };
    let stream: Vec<usize> = TEXT.chars().map(encode).collect();

    let (hidden, heads, kv_heads, head_dim) = (16usize, 2usize, 1usize, 8usize);
    let (batch, seq) = (4usize, 4usize);

    // Windows of `seq` tokens plus the next-token target; hold out the tail.
    let windows = (stream.len() - 1) / seq;
    let train_windows = (((windows as f32) * (1.0 - VALIDATION_FRACTION)) as usize)
        .max(1)
        .min(windows - 1);
    let held_out = windows - train_windows;

    let device = Device::Cpu;
    let upload = |tensor: Tensor| DeviceTensor::from_host(tensor).to(&device).expect("upload");
    let mut parameters = vec![
        upload(sample(vocab, hidden, 0.5, 1)),
        upload(sample(1, hidden, 0.1, 2)),
        upload(sample(1, hidden, 0.1, 3)),
        upload(sample(hidden, heads * head_dim, 0.5, 4)),
        upload(sample(1, heads * head_dim, 0.1, 5)),
        upload(sample(hidden, kv_heads * head_dim, 0.5, 6)),
        upload(sample(1, kv_heads * head_dim, 0.1, 7)),
        upload(sample(hidden, kv_heads * head_dim, 0.5, 8)),
        upload(sample(1, kv_heads * head_dim, 0.1, 9)),
        upload(sample(heads * head_dim, hidden, 0.5, 10)),
        upload(sample(1, hidden, 0.1, 11)),
        upload(sample(1, hidden, 0.1, 12)),
        upload(sample(1, hidden, 0.1, 13)),
        upload(sample(hidden, vocab, 0.5, 14)),
        upload(sample(1, vocab, 0.1, 15)),
    ];

    println!(
        "device attention block: vocab {vocab}, hidden {hidden}, heads {heads}/{kv_heads}, seq {seq}, batch {batch}"
    );
    println!("vocabulary: {:?}", vocabulary.iter().collect::<String>());
    println!("windows: {windows} ({train_windows} train / {held_out} held out)");

    // Window `w` covers tokens `w*seq .. w*seq+seq` with next-token targets.
    let batch_tokens = |start: usize, count: usize| -> (Vec<usize>, Vec<usize>) {
        let mut inputs = Vec::with_capacity(count * seq);
        let mut targets = Vec::with_capacity(count * seq);
        for index in 0..count {
            let window = (start + index) % train_windows;
            for position in 0..seq {
                let token = window * seq + position;
                inputs.push(stream[token]);
                targets.push(stream[token + 1]);
            }
        }
        (inputs, targets)
    };

    let mut optimizer = DeviceAdamW::new(LEARNING_RATE);
    let mut train_loss = 0.0f32;
    let mut validation_loss: f32;
    let mut train_cursor = 0usize;

    for step in 1..=STEPS {
        let nodes: Vec<DeviceValue> = parameters
            .iter()
            .map(|parameter| DeviceValue::new(parameter.clone(), true))
            .collect();

        let (inputs, targets) = batch_tokens(train_cursor, batch);
        train_cursor = (train_cursor + batch) % train_windows;
        let loss = forward(&nodes, &inputs, &targets, batch, seq, heads, kv_heads);
        loss.backward().expect("backward");
        train_loss = loss.data().to_host().expect("loss").data()[0];

        let gradients: Vec<DeviceTensor> = nodes
            .iter()
            .map(|node| node.grad().expect("parameter gradient"))
            .collect();
        optimizer.step(&mut parameters, &gradients).expect("adamw");

        if step == 1 || step % PRINT_EVERY == 0 {
            validation_loss = evaluate(
                &parameters,
                &stream,
                train_windows,
                windows,
                batch,
                seq,
                heads,
                kv_heads,
            );
            println!("  step {step:>3}  train {train_loss:.4}  held-out {validation_loss:.4}");
        }
    }

    validation_loss = evaluate(
        &parameters,
        &stream,
        train_windows,
        windows,
        batch,
        seq,
        heads,
        kv_heads,
    );
    println!("final train loss: {train_loss:.4}");
    println!("final held-out loss: {validation_loss:.4}");
    println!(
        "parameters on {:?}; all device-resident: {}",
        parameters[0].device(),
        parameters
            .iter()
            .all(|parameter| parameter.device() == Device::Cpu)
    );
}

#[allow(clippy::too_many_arguments)]
fn forward(
    nodes: &[DeviceValue],
    inputs: &[usize],
    targets: &[usize],
    batch: usize,
    seq: usize,
    heads: usize,
    kv_heads: usize,
) -> DeviceValue {
    let embedded = nodes[0].gather_rows(inputs).expect("embedding");
    let normed = embedded.layer_norm(&nodes[1], &nodes[2], 1e-5).expect("ln1");
    let queries = normed.matmul(&nodes[3]).expect("q").add(&nodes[4]).expect("bq");
    let keys = normed.matmul(&nodes[5]).expect("k").add(&nodes[6]).expect("bk");
    let values = normed.matmul(&nodes[7]).expect("v").add(&nodes[8]).expect("bv");
    let attended = queries
        .attention(&keys, &values, batch, seq, heads, kv_heads, true)
        .expect("attention");
    let projected = attended
        .matmul(&nodes[9])
        .expect("out projection")
        .add(&nodes[10])
        .expect("out bias");
    let residual = embedded.add(&projected).expect("residual");
    let normed2 = residual.layer_norm(&nodes[11], &nodes[12], 1e-5).expect("ln2");
    let logits = normed2
        .matmul(&nodes[13])
        .expect("head")
        .add(&nodes[14])
        .expect("head bias");
    logits.cross_entropy(targets).expect("cross entropy")
}

/// Runs the held-out windows with gradients disabled and averages their loss.
#[allow(clippy::too_many_arguments)]
fn evaluate(
    parameters: &[DeviceTensor],
    stream: &[usize],
    train_windows: usize,
    windows: usize,
    batch: usize,
    seq: usize,
    heads: usize,
    kv_heads: usize,
) -> f32 {
    let nodes: Vec<DeviceValue> = parameters
        .iter()
        .map(|parameter| DeviceValue::new(parameter.clone(), false))
        .collect();
    let held_out = windows - train_windows;
    let batches = held_out.div_ceil(batch);
    let mut total = 0.0f32;
    for index in 0..batches {
        let mut inputs = Vec::with_capacity(batch * seq);
        let mut targets = Vec::with_capacity(batch * seq);
        for offset in 0..batch {
            let window = train_windows + (index * batch + offset) % held_out;
            for position in 0..seq {
                let token = window * seq + position;
                inputs.push(stream[token]);
                targets.push(stream[token + 1]);
            }
        }
        let loss = forward(&nodes, &inputs, &targets, batch, seq, heads, kv_heads);
        total += loss.data().to_host().expect("loss").data()[0] * batch as f32;
    }
    total / (batches * batch) as f32
}
