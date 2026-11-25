//! Trains a device-resident transformer block stack on a character corpus and
//! prints train (EMA) / held-out loss. Supports regularization sweeps and a
//! layer-count sweep.
//!
//! Each layer is: LayerNorm -> Q/K/V -> fused GQA attention -> out-projection ->
//! residual, then LayerNorm -> SwiGLU -> residual. A final LayerNorm + LM head
//! produces the logits. Everything runs through `DeviceTensor` / `DeviceValue`
//! (host storage) and `DeviceAdamW`.
//!
//! Run one configuration:
//! `cargo run -p mini-llm-backend --example device_attention_block -- --layers 2 --hidden 32`
//!
//! Layer sweep (1/2/3 blocks):
//! `cargo run -p mini-llm-backend --example device_attention_block -- --layer-sweep`
//!
//! Regularization sweep:
//! `cargo run -p mini-llm-backend --example device_attention_block -- --sweep`

use mini_llm_backend::{Device, DeviceAdamW, DeviceTensor, DeviceValue};
use mini_llm_tensor::Tensor;

const STEPS: usize = 1600;
const PRINT_EVERY: usize = 25;
const EVAL_WINDOWS: usize = 64;
const LEARNING_RATE: f32 = 0.02;
const VALIDATION_FRACTION: f32 = 0.2;
/// Parameters per layer: ln1(2) + qkv(6) + out(2) + ln2(2) + swiglu(6).
const LAYER_PARAMETERS: usize = 18;

fn sample(rows: usize, cols: usize, scale: f32, seed: usize) -> Tensor {
    let data = (0..rows * cols)
        .map(|index| {
            let mixed = (index * 37 + seed * 101 + 11) % 29;
            (mixed as f32 / 29.0 - 0.5) * 2.0 * scale
        })
        .collect();
    Tensor::from_2d(rows, cols, data).expect("valid tensor")
}

/// Small deterministic xorshift generator for dropout masks.
struct Rng(u64);

impl Rng {
    fn new(seed: u64) -> Self {
        Self(seed.max(1))
    }

    fn next_f32(&mut self) -> f32 {
        self.0 ^= self.0 << 13;
        self.0 ^= self.0 >> 7;
        self.0 ^= self.0 << 17;
        (self.0 >> 40) as f32 / (1u64 << 24) as f32
    }
}

struct Config {
    weight_decay: f32,
    dropout: f32,
    steps: usize,
    hidden: usize,
    layers: usize,
    seq: usize,
    batch: usize,
    learning_rate: f32,
    min_ratio: f32,
    warmup: usize,
    constant_lr: bool,
}

impl Config {
    fn label(&self) -> String {
        format!(
            "layers {} hidden {} wd {:.2} dropout {:.2}",
            self.layers, self.hidden, self.weight_decay, self.dropout
        )
    }
}

fn main() {
    let arguments: Vec<String> = std::env::args().skip(1).collect();
    if arguments.iter().any(|argument| argument == "--sweep") {
        sweep_regularization();
        return;
    }
    if arguments.iter().any(|argument| argument == "--layer-sweep") {
        sweep_layers(&arguments);
        return;
    }
    let config = Config {
        weight_decay: parse_flag(&arguments, "--weight-decay").unwrap_or(0.01),
        dropout: parse_flag(&arguments, "--dropout").unwrap_or(0.0),
        steps: parse_count(&arguments, "--steps").unwrap_or(STEPS),
        hidden: parse_count(&arguments, "--hidden").unwrap_or(32),
        layers: parse_count(&arguments, "--layers").unwrap_or(1).max(1),
        seq: parse_count(&arguments, "--seq").unwrap_or(8),
        batch: parse_count(&arguments, "--batch").unwrap_or(4),
        learning_rate: parse_flag(&arguments, "--lr").unwrap_or(LEARNING_RATE),
        min_ratio: parse_flag(&arguments, "--min-ratio").unwrap_or(0.1),
        warmup: parse_count(&arguments, "--warmup").unwrap_or(0),
        constant_lr: arguments.iter().any(|argument| argument == "--constant-lr"),
    };
    let (train_loss, held_out_loss, best_held_out) = train_with_best(&config, true);
    println!("final train loss: {train_loss:.4}");
    println!("final held-out loss: {held_out_loss:.4}");
    println!("best held-out loss: {best_held_out:.4}");
}

fn sweep_regularization() {
    println!("regularization sweep ({STEPS} steps, README + doc-comment corpus)");
    println!(
        "{:>44}  {:>10}  {:>12}  {:>14}",
        "configuration", "train", "held-out", "best held-out"
    );
    for weight_decay in [0.01f32, 0.1] {
        for dropout in [0.0f32, 0.3] {
            let config = Config {
                weight_decay,
                dropout,
                steps: STEPS,
                hidden: 32,
                layers: 1,
                seq: 8,
                batch: 4,
                learning_rate: LEARNING_RATE,
                min_ratio: 0.1,
                warmup: 0,
                constant_lr: false,
            };
            let (train_loss, held_out_loss, best_held_out) = train_with_best(&config, false);
            println!(
                "{:>44}  {:>10.4}  {:>12.4}  {:>14.4}",
                config.label(),
                train_loss,
                held_out_loss,
                best_held_out
            );
        }
    }
}

fn sweep_layers(arguments: &[String]) {
    let steps = parse_count(arguments, "--steps").unwrap_or(STEPS);
    let hidden = parse_count(arguments, "--hidden").unwrap_or(32);
    println!("layer sweep ({steps} steps, hidden {hidden}, README + doc-comment corpus)");
    println!(
        "{:>8}  {:>10}  {:>10}  {:>12}  {:>14}",
        "layers", "params", "train", "held-out", "best held-out"
    );
    for layers in 1..=3 {
        let config = Config {
            weight_decay: 0.01,
            dropout: 0.0,
            steps,
            hidden,
            layers,
            seq: parse_count(arguments, "--seq").unwrap_or(8),
            batch: parse_count(arguments, "--batch").unwrap_or(4),
            learning_rate: parse_flag(arguments, "--lr").unwrap_or(LEARNING_RATE),
            min_ratio: parse_flag(arguments, "--min-ratio").unwrap_or(0.1),
            warmup: parse_count(arguments, "--warmup").unwrap_or(0),
            constant_lr: arguments.iter().any(|argument| argument == "--constant-lr"),
        };
        let (train_loss, held_out_loss, best_held_out, parameters) =
            train_with_parameters(&config, false);
        println!(
            "{:>8}  {:>10}  {:>10.4}  {:>12.4}  {:>14.4}",
            layers,
            parameters.iter().map(parameter_len).sum::<usize>(),
            train_loss,
            held_out_loss,
            best_held_out
        );
    }
}

/// Warmup + cosine decay, or a constant rate with --constant-lr.
fn learning_rate(step: usize, config: &Config) -> f32 {
    if config.constant_lr {
        return config.learning_rate;
    }
    let warmup = if config.warmup > 0 {
        config.warmup
    } else {
        (config.steps / 10).clamp(1, 200)
    };
    if step <= warmup {
        return config.learning_rate * step as f32 / warmup as f32;
    }
    let progress = (step - warmup) as f32 / (config.steps.saturating_sub(warmup)).max(1) as f32;
    let minimum = config.learning_rate * config.min_ratio;
    minimum
        + 0.5 * (config.learning_rate - minimum) * (1.0 + (std::f32::consts::PI * progress).cos())
}

fn parse_flag(arguments: &[String], name: &str) -> Option<f32> {
    arguments
        .iter()
        .position(|argument| argument == name)
        .and_then(|index| arguments.get(index + 1))
        .and_then(|value| value.parse().ok())
}

fn parse_count(arguments: &[String], name: &str) -> Option<usize> {
    arguments
        .iter()
        .position(|argument| argument == name)
        .and_then(|index| arguments.get(index + 1))
        .and_then(|value| value.parse().ok())
}

fn parameter_len(parameter: &DeviceTensor) -> usize {
    parameter.shape().iter().product()
}

fn train_with_best(config: &Config, verbose: bool) -> (f32, f32, f32) {
    let (train_loss, held_out_loss, best_held_out, _) = train_with_parameters(config, verbose);
    (train_loss, held_out_loss, best_held_out)
}

fn train_with_parameters(config: &Config, verbose: bool) -> (f32, f32, f32, Vec<DeviceTensor>) {
    let mut vocabulary: Vec<char> = corpus().chars().collect();
    vocabulary.sort_unstable();
    vocabulary.dedup();
    let vocab = vocabulary.len();
    let encode = |character: char| {
        vocabulary
            .binary_search(&character)
            .expect("character is in the vocabulary")
    };
    let stream: Vec<usize> = corpus().chars().map(encode).collect();

    let heads = 4usize;
    let kv_heads = 2usize;
    let hidden = (config.hidden / heads).max(1) * heads;
    let head_dim = hidden / heads;
    let ffn = hidden * 2;
    let batch = config.batch.max(1);
    let seq = config.seq.max(2);

    let windows = (stream.len() - 1) / seq;
    let train_windows = (((windows as f32) * (1.0 - VALIDATION_FRACTION)) as usize)
        .max(1)
        .min(windows - 1);
    let held_out = windows - train_windows;

    let device = Device::Cpu;
    let mut parameters = build_parameters(
        config, vocab, hidden, ffn, heads, kv_heads, head_dim, &device,
    );

    if verbose {
        println!(
            "corpus: {} chars; vocab {vocab}; windows {windows} ({train_windows} train / {held_out} held out)",
            corpus().chars().count()
        );
        println!(
            "configuration: {} | heads {heads}/{kv_heads}, head_dim {head_dim}, seq {seq}, batch {batch}",
            config.label()
        );
        println!(
            "parameters: {} values in {} tensors",
            parameters.iter().map(parameter_len).sum::<usize>(),
            parameters.len()
        );
    }

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

    let mut optimizer = DeviceAdamW::new(LEARNING_RATE).with_weight_decay(config.weight_decay);
    let mut rng = Rng::new(7);
    let mut train_loss = 0.0f32;
    let mut train_ema: Option<f32> = None;
    let mut best_held_out: Option<f32> = None;
    let mut train_cursor = 0usize;

    for step in 1..=config.steps {
        let nodes: Vec<DeviceValue> = parameters
            .iter()
            .map(|parameter| DeviceValue::new(parameter.clone(), true))
            .collect();

        let (inputs, targets) = batch_tokens(train_cursor, batch);
        train_cursor = (train_cursor + batch) % train_windows;
        let loss = forward(
            &nodes, &inputs, &targets, batch, seq, heads, kv_heads, config, &mut rng,
        );
        loss.backward().expect("backward");
        train_loss = loss.data().to_host().expect("loss").data()[0];
        train_ema = Some(match train_ema {
            Some(previous) => 0.98 * previous + 0.02 * train_loss,
            None => train_loss,
        });

        let gradients: Vec<DeviceTensor> = nodes
            .iter()
            .map(|node| node.grad().expect("parameter gradient"))
            .collect();
        optimizer.set_learning_rate(learning_rate(step, config));
        optimizer.step(&mut parameters, &gradients).expect("adamw");

        if step == 1 || step % PRINT_EVERY == 0 {
            let measured = evaluate(
                &parameters,
                &stream,
                train_windows,
                windows,
                EVAL_WINDOWS,
                batch,
                seq,
                heads,
                kv_heads,
                config,
            );
            best_held_out = Some(match best_held_out {
                Some(best) => best.min(measured),
                None => measured,
            });
            if verbose {
                println!(
                    "  step {step:>5}  lr {:.5}  train(ema) {:.4}  held-out {measured:.4}",
                    optimizer.learning_rate(),
                    train_ema.unwrap_or(train_loss)
                );
            }
        }
    }

    let final_held_out = evaluate(
        &parameters,
        &stream,
        train_windows,
        windows,
        EVAL_WINDOWS,
        batch,
        seq,
        heads,
        kv_heads,
        config,
    );
    let best = best_held_out.unwrap_or(final_held_out).min(final_held_out);
    (
        train_ema.unwrap_or(train_loss),
        final_held_out,
        best,
        parameters,
    )
}

#[allow(clippy::too_many_arguments)]
fn build_parameters(
    config: &Config,
    vocab: usize,
    hidden: usize,
    ffn: usize,
    heads: usize,
    kv_heads: usize,
    head_dim: usize,
    device: &Device,
) -> Vec<DeviceTensor> {
    let upload = |tensor: Tensor| DeviceTensor::from_host(tensor).to(device).expect("upload");
    let mut seed = 1usize;
    let mut next = |rows: usize, cols: usize, scale: f32| {
        seed += 1;
        upload(sample(rows, cols, scale, seed))
    };

    let q_width = heads * head_dim;
    let kv_width = kv_heads * head_dim;
    let mut parameters = vec![next(vocab, hidden, 0.5)];
    for _ in 0..config.layers {
        parameters.push(next(1, hidden, 0.1));
        parameters.push(next(1, hidden, 0.1));
        parameters.push(next(hidden, q_width, 0.5));
        parameters.push(next(1, q_width, 0.1));
        parameters.push(next(hidden, kv_width, 0.5));
        parameters.push(next(1, kv_width, 0.1));
        parameters.push(next(hidden, kv_width, 0.5));
        parameters.push(next(1, kv_width, 0.1));
        parameters.push(next(q_width, hidden, 0.5));
        parameters.push(next(1, hidden, 0.1));
        parameters.push(next(1, hidden, 0.1));
        parameters.push(next(1, hidden, 0.1));
        parameters.push(next(hidden, ffn, 0.5));
        parameters.push(next(1, ffn, 0.1));
        parameters.push(next(hidden, ffn, 0.5));
        parameters.push(next(1, ffn, 0.1));
        parameters.push(next(ffn, hidden, 0.5));
        parameters.push(next(1, hidden, 0.1));
    }
    parameters.push(next(1, hidden, 0.1));
    parameters.push(next(1, hidden, 0.1));
    parameters.push(next(hidden, vocab, 0.5));
    parameters.push(next(1, vocab, 0.1));
    parameters
}

fn dropout(value: DeviceValue, probability: f32, rng: &mut Rng, rows: usize) -> DeviceValue {
    if probability <= 0.0 {
        return value;
    }
    let width = value.shape()[1];
    let scale = 1.0 / (1.0 - probability);
    let data: Vec<f32> = (0..rows * width)
        .map(|_| {
            if rng.next_f32() < probability {
                0.0
            } else {
                scale
            }
        })
        .collect();
    let mask = Tensor::from_2d(rows, width, data).expect("dropout mask");
    let mask = DeviceTensor::from_host(mask)
        .to(&value.data().device())
        .expect("upload mask");
    value.mul(&DeviceValue::new(mask, false)).expect("dropout")
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
    config: &Config,
    rng: &mut Rng,
) -> DeviceValue {
    let mut hidden = nodes[0].gather_rows(inputs).expect("embedding");
    for layer in 0..config.layers {
        let base = 1 + layer * LAYER_PARAMETERS;
        let normed = hidden
            .layer_norm(&nodes[base], &nodes[base + 1], 1e-5)
            .expect("ln1");
        let queries = normed
            .matmul(&nodes[base + 2])
            .expect("q")
            .add(&nodes[base + 3])
            .expect("bq");
        let keys = normed
            .matmul(&nodes[base + 4])
            .expect("k")
            .add(&nodes[base + 5])
            .expect("bk");
        let values = normed
            .matmul(&nodes[base + 6])
            .expect("v")
            .add(&nodes[base + 7])
            .expect("bv");
        let attended = queries
            .attention(&keys, &values, batch, seq, heads, kv_heads, true)
            .expect("attention");
        let projected = attended
            .matmul(&nodes[base + 8])
            .expect("out projection")
            .add(&nodes[base + 9])
            .expect("out bias");
        let projected = dropout(projected, config.dropout, rng, batch * seq);
        hidden = hidden.add(&projected).expect("attention residual");

        let normed2 = hidden
            .layer_norm(&nodes[base + 10], &nodes[base + 11], 1e-5)
            .expect("ln2");
        let gate = normed2
            .matmul(&nodes[base + 12])
            .expect("gate")
            .add(&nodes[base + 13])
            .expect("gate bias");
        let up = normed2
            .matmul(&nodes[base + 14])
            .expect("up")
            .add(&nodes[base + 15])
            .expect("up bias");
        let down = gate
            .silu()
            .expect("silu")
            .mul(&up)
            .expect("swiglu")
            .matmul(&nodes[base + 16])
            .expect("down")
            .add(&nodes[base + 17])
            .expect("down bias");
        let down = dropout(down, config.dropout, rng, batch * seq);
        hidden = hidden.add(&down).expect("ffn residual");
    }
    let base = 1 + config.layers * LAYER_PARAMETERS;
    let normed = hidden
        .layer_norm(&nodes[base], &nodes[base + 1], 1e-5)
        .expect("final ln");
    let logits = normed
        .matmul(&nodes[base + 2])
        .expect("head")
        .add(&nodes[base + 3])
        .expect("head bias");
    logits.cross_entropy(targets).expect("cross entropy")
}

#[allow(clippy::too_many_arguments)]
fn evaluate(
    parameters: &[DeviceTensor],
    stream: &[usize],
    train_windows: usize,
    windows: usize,
    window_limit: usize,
    batch: usize,
    seq: usize,
    heads: usize,
    kv_heads: usize,
    config: &Config,
) -> f32 {
    let nodes: Vec<DeviceValue> = parameters
        .iter()
        .map(|parameter| DeviceValue::new(parameter.clone(), false))
        .collect();
    let held_out = windows - train_windows;
    let count = held_out.min(window_limit);
    let batches = count.div_ceil(batch);
    let eval_config = Config {
        weight_decay: config.weight_decay,
        dropout: 0.0,
        steps: config.steps,
        hidden: config.hidden,
        layers: config.layers,
        seq: config.seq,
        batch: config.batch,
        learning_rate: config.learning_rate,
        min_ratio: config.min_ratio,
        warmup: config.warmup,
        constant_lr: config.constant_lr,
    };
    let mut total = 0.0f32;
    for index in 0..batches {
        let mut inputs = Vec::with_capacity(batch * seq);
        let mut targets = Vec::with_capacity(batch * seq);
        for offset in 0..batch {
            let window = train_windows + (index * batch + offset) % count;
            for position in 0..seq {
                let token = window * seq + position;
                inputs.push(stream[token]);
                targets.push(stream[token + 1]);
            }
        }
        let loss = forward(
            &nodes,
            &inputs,
            &targets,
            batch,
            seq,
            heads,
            kv_heads,
            &eval_config,
            &mut Rng::new(1),
        );
        total += loss.data().to_host().expect("loss").data()[0] * batch as f32;
    }
    total / (batches * batch) as f32
}

/// README plus every crate's `//!` / `///` doc comment (English prose).
/// Sorted traversal keeps the corpus deterministic across runs.
fn corpus() -> String {
    let mut text = String::from(include_str!("../../../README.md"));
    let root = std::path::Path::new(env!("CARGO_MANIFEST_DIR")).join("../..");
    let mut files = Vec::new();
    collect_rust_files(&root.join("crates"), &mut files);
    files.sort();
    for file in files {
        if let Ok(contents) = std::fs::read_to_string(&file) {
            // Test sources are prose-heavy and add real code tokens, so the
            // corpus includes them verbatim instead of only their comments.
            let is_test = file
                .components()
                .any(|component| component.as_os_str() == "tests");
            if is_test {
                text.push_str(&contents);
                text.push('\n');
                continue;
            }
            for line in contents.lines() {
                let trimmed = line.trim_start();
                if let Some(comment) = trimmed
                    .strip_prefix("//!")
                    .or_else(|| trimmed.strip_prefix("///"))
                {
                    text.push_str(comment.trim());
                    text.push('\n');
                }
            }
        }
    }
    let web_root = root.join("web/src");
    let mut web_files = Vec::new();
    collect_files_with_extensions(&web_root, &["ts", "tsx", "astro"], &mut web_files);
    web_files.sort();
    for file in web_files {
        if let Ok(contents) = std::fs::read_to_string(&file) {
            // Keep the corpus ASCII: the site mixes in CJK prose, which would
            // otherwise explode the character vocabulary.
            text.extend(contents.chars().filter(char::is_ascii));
            text.push('\n');
        }
    }
    if let Some((index, _)) = text.char_indices().nth(600_000) {
        text.truncate(index);
    }
    text
}

fn collect_files_with_extensions(
    directory: &std::path::Path,
    extensions: &[&str],
    files: &mut Vec<std::path::PathBuf>,
) {
    let Ok(entries) = std::fs::read_dir(directory) else {
        return;
    };
    for entry in entries.flatten() {
        let path = entry.path();
        if path.is_dir() {
            collect_files_with_extensions(&path, extensions, files);
        } else if let Some(extension) = path.extension().and_then(|value| value.to_str()) {
            if extensions.contains(&extension) {
                files.push(path);
            }
        }
    }
}

fn collect_rust_files(directory: &std::path::Path, files: &mut Vec<std::path::PathBuf>) {
    let Ok(entries) = std::fs::read_dir(directory) else {
        return;
    };
    for entry in entries.flatten() {
        let path = entry.path();
        if path.is_dir() {
            collect_rust_files(&path, files);
        } else if path.extension().and_then(|extension| extension.to_str()) == Some("rs") {
            files.push(path);
        }
    }
}
