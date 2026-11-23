use mini_llm_autograd::TensorValue;
use mini_llm_data::{CharTokenizer, TextDataset};
use mini_llm_model::{scaled_dot_product_attention, ModelConfig, TinyGpt};
use mini_llm_nn::{AdamW, Linear, Module, Optimizer, Relu, Rng, Sequential, Tanh};
use mini_llm_tensor::Tensor;
use mini_llm_train::{load_checkpoint, save_checkpoint, TrainConfig, Trainer};

const CHECKPOINT_PATH: &str = "checkpoints/tiny-gpt.bin";

/// A short public-domain-style paragraph used for the character-level demo.
const TINY_CORPUS: &str = "\
The little language model learns one character at a time. It reads a short story, \
predicts the next letter, and slowly discovers words, spaces and punctuation. \
Small models can still surprise you: they learn rhythm before meaning, and grammar \
before grace. Training is cheap here, so experiments are fast. The model does not \
know what a word is; it only knows which letter tends to follow which. That is \
enough to begin.";

fn main() {
    let arguments: Vec<String> = std::env::args().skip(1).collect();
    let command = arguments.first().map(String::as_str).unwrap_or("demo");
    let rest = arguments.get(1..).unwrap_or(&[]);

    match command {
        "demo" => demo(),
        "inspect-config" => inspect_config(),
        "autograd" => autograd_demo(),
        "train-line" => fit_line(),
        "xor" => xor_demo(),
        "classify" => classify_demo(),
        "train" => train_tiny_gpt(rest),
        "generate" => generate_text(rest),
        "help" | "--help" | "-h" => usage(),
        other => {
            eprintln!("Unknown command: {other}");
            usage();
            std::process::exit(2);
        }
    }
}

fn usage() {
    println!("Usage: cargo run -p mini-llm-cli -- <command> [args]");
    println!();
    println!("Commands:");
    println!("  demo                 scaled dot-product attention walkthrough");
    println!("  inspect-config       validate the default model configuration");
    println!("  autograd             computation graph + gradients for y = wx + b");
    println!("  train-line           fit y = 2x - 1 with manual SGD");
    println!("  xor                  train a small MLP on XOR (Linear + Tanh + AdamW)");
    println!("  classify             train a 2D two-blob classifier with softmax + cross-entropy");
    println!("  train [steps]        train Tiny GPT on the built-in corpus ({CHECKPOINT_PATH})");
    println!("  generate [prompt] [tokens] [temp]");
    println!("                       sample text from the saved checkpoint");
    println!("  help                 show this message");
}

fn demo() {
    let q = Tensor::from_2d(2, 2, vec![1.0, 0.0, 0.0, 1.0]).expect("valid query tensor");
    let k = Tensor::from_2d(2, 2, vec![1.0, 0.0, 0.0, 1.0]).expect("valid key tensor");
    let v = Tensor::from_2d(2, 2, vec![10.0, 0.0, 0.0, 10.0]).expect("valid value tensor");

    let (output, weights) = scaled_dot_product_attention(&q, &k, &v).expect("attention should run");

    println!("mini-llm CPU attention demo");
    println!("attention weights: {:?}", weights.data());
    println!("output shape: {:?}", output.shape());
    println!("output: {:?}", output.data());
}

fn inspect_config() {
    let config = ModelConfig::default();
    match config.validate() {
        Ok(()) => {
            println!("Model configuration is valid:");
            println!("  vocab_size:   {}", config.vocab_size);
            println!("  hidden_size:  {}", config.hidden_size);
            println!("  num_layers:   {}", config.num_layers);
            println!("  num_heads:    {}", config.num_heads);
            println!("  max_seq_len:  {}", config.max_seq_len);
        }
        Err(err) => {
            eprintln!("Invalid model configuration: {err}");
            std::process::exit(1);
        }
    }
}

/// Reproduces the introductory example from `docs/AUTOGRAD_ENGINE.md`:
/// `y = wx + b`, `L = (y - 10)^2`, with `x = 2`, `w = 3`, `b = 1`.
fn autograd_demo() {
    let x = TensorValue::constant(vec![2.0], vec![1, 1]).expect("valid tensor");
    let target = TensorValue::constant(vec![10.0], vec![1]).expect("valid tensor");
    let w = TensorValue::parameter(vec![3.0], vec![1, 1]).expect("valid tensor");
    let b = TensorValue::parameter(vec![1.0], vec![1]).expect("valid tensor");

    let y = x.matmul(&w).expect("matmul").add(&b).expect("bias add");
    let loss = y
        .sub(&target)
        .expect("subtract target")
        .powf(2.0)
        .expect("square")
        .mean()
        .expect("mean");

    println!("mini-llm autograd demo");
    println!("  x = 2, w = 3, b = 1  ->  y = {:.4}", y.data()[0]);
    println!("  loss = (y - 10)^2 = {:.4}", loss.data()[0]);

    loss.backward().expect("backward pass");

    println!();
    println!(
        "  dL/dw = {:.4}  (hand-derived: -12)",
        w.grad().unwrap().data()[0]
    );
    println!(
        "  dL/db = {:.4}  (hand-derived:  -6)",
        b.grad().unwrap().data()[0]
    );

    println!();
    println!("  computation graph (topological order):");
    for line in loss.graph_summary() {
        println!("    {line}");
    }

    println!();
    match gradient_check_error() {
        Some(error) => {
            println!("  finite-difference check: max |analytic - numeric| = {error:.2e}")
        }
        None => println!("  finite-difference check: skipped"),
    }
}

/// Compares the analytic gradient of `mean(matmul(x, w)^2)` with central
/// differences, mirroring the tests in `crates/mini-llm-autograd/tests`.
fn gradient_check_error() -> Option<f32> {
    let x_data = vec![0.5, -0.3, 0.8, 0.2, -0.6, 0.9];
    let initial = vec![0.7, -0.4, 0.1, 0.3, 0.9, -0.2];

    let loss_for = |weights: &[f32]| -> Option<f32> {
        let x = TensorValue::constant(x_data.clone(), vec![2, 3]).ok()?;
        let w = TensorValue::parameter(weights.to_vec(), vec![3, 2]).ok()?;
        x.matmul(&w)
            .ok()?
            .relu()
            .ok()?
            .mean()
            .ok()?
            .data()
            .first()
            .copied()
    };

    let x = TensorValue::constant(x_data.clone(), vec![2, 3]).ok()?;
    let w = TensorValue::parameter(initial.clone(), vec![3, 2]).ok()?;
    let loss = x.matmul(&w).ok()?.relu().ok()?.mean().ok()?;
    loss.backward().ok()?;
    let analytic = w.grad()?.data().to_vec();

    let step = 1e-2;
    let mut max_error = 0.0f32;
    for index in 0..initial.len() {
        let mut plus = initial.clone();
        plus[index] += step;
        let mut minus = initial.clone();
        minus[index] -= step;
        let numeric = (loss_for(&plus)? - loss_for(&minus)?) / (2.0 * step);
        max_error = max_error.max((analytic[index] - numeric).abs());
    }
    Some(max_error)
}

/// A tiny linear-regression loop: gradients come from autograd, the parameter
/// update is plain SGD written out by hand so the mechanics stay visible.
fn fit_line() {
    let xs: Vec<f32> = (0..16).map(|index| index as f32 / 4.0).collect();
    let ys: Vec<f32> = xs.iter().map(|x| 2.0 * x - 1.0).collect();
    let features = TensorValue::constant(xs.clone(), vec![16, 1]).expect("valid features");
    let targets = TensorValue::constant(ys, vec![16, 1]).expect("valid targets");

    let (mut weight, mut bias) = (vec![0.0f32], vec![0.0f32]);
    let learning_rate = 0.1;

    println!("fitting y = 2x - 1 with manual SGD (true values: w = 2, b = -1)");
    for step in 0..=200 {
        let w = TensorValue::parameter(weight.clone(), vec![1, 1]).expect("valid weight");
        let b = TensorValue::parameter(bias.clone(), vec![1]).expect("valid bias");

        let predictions = features
            .matmul(&w)
            .expect("matmul")
            .add(&b)
            .expect("bias broadcast");
        let loss = predictions
            .sub(&targets)
            .expect("residual")
            .powf(2.0)
            .expect("square")
            .mean()
            .expect("mean");

        if step % 50 == 0 {
            println!(
                "  step {step:>3}  loss {:>10.6}  w {:>8.4}  b {:>8.4}",
                loss.data()[0],
                weight[0],
                bias[0]
            );
        }

        loss.backward().expect("backward pass");
        weight[0] -= learning_rate * w.grad().expect("weight grad").data()[0];
        bias[0] -= learning_rate * b.grad().expect("bias grad").data()[0];
    }
    println!("  final: w = {:.4}, b = {:.4}", weight[0], bias[0]);
}

/// Trains a two-layer MLP on the XOR truth table with AdamW and MSE loss.
fn xor_demo() {
    const STEPS: usize = 1000;
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

/// Trains a two-layer classifier on two Gaussian blobs with softmax + cross-entropy.
fn classify_demo() {
    const PER_CLASS: usize = 80;
    const STEPS: usize = 400;
    const HIDDEN: usize = 16;

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
    let labels = labels;

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

/// Character-level Tiny GPT training on the built-in corpus.
fn train_tiny_gpt(arguments: &[String]) {
    const BLOCK_SIZE: usize = 16;
    const BATCH_SIZE: usize = 4;

    let steps = arguments
        .first()
        .and_then(|value| value.parse().ok())
        .unwrap_or(800);

    let tokenizer = CharTokenizer::from_text(TINY_CORPUS);
    let tokens = tokenizer.encode(TINY_CORPUS).expect("corpus encodes");
    let config = ModelConfig {
        vocab_size: tokenizer.vocab_size(),
        hidden_size: 48,
        num_layers: 2,
        num_heads: 4,
        max_seq_len: BLOCK_SIZE,
    };
    let mut rng = Rng::new(1337);
    let model = TinyGpt::new(config, &mut rng).expect("model");

    println!("Tiny GPT, character level");
    println!(
        "  corpus: {} chars, vocab {}, {} training windows",
        TINY_CORPUS.chars().count(),
        tokenizer.vocab_size(),
        tokens.len() - BLOCK_SIZE
    );
    println!(
        "  model: {} params | layers={} hidden={} heads={} block={}",
        model.num_parameters(),
        model.config().num_layers,
        model.config().hidden_size,
        model.config().num_heads,
        model.config().max_seq_len
    );
    println!("  optimizer: AdamW | batch={BATCH_SIZE} | steps={steps}");
    println!();

    let dataset = TextDataset::new(tokens, BLOCK_SIZE).expect("dataset");
    let mut trainer = Trainer::new(
        model,
        dataset,
        TrainConfig {
            steps,
            batch_size: BATCH_SIZE,
            learning_rate: 3e-3,
            weight_decay: 0.01,
            seed: 7,
        },
    );

    let mut last_loss = f32::NAN;
    trainer
        .train(steps, |report| {
            last_loss = report.loss;
            if report.step == 1 || report.step % 10 == 0 {
                println!(
                    "  step {:>4}  loss {:>7.4}  {:>6.1} ms/step",
                    report.step, report.loss, report.elapsed_ms
                );
            }
        })
        .expect("training");
    println!("  final sampled loss: {last_loss:.4}");

    let prompt = tokenizer.encode("The ").unwrap_or_else(|_| vec![0]);
    let mut sampler = Rng::new(99);
    let generated = trainer
        .model()
        .generate(&prompt, 160, 0.8, &mut sampler)
        .expect("generate");
    println!();
    println!("--- sample ---");
    println!("{}", tokenizer.decode(&generated).expect("decode"));

    std::fs::create_dir_all("checkpoints").ok();
    save_checkpoint(CHECKPOINT_PATH, trainer.model(), &tokenizer).expect("save checkpoint");
    println!();
    println!("checkpoint saved to {CHECKPOINT_PATH}");
}

/// Samples text from the saved checkpoint.
fn generate_text(arguments: &[String]) {
    let prompt = arguments.first().map(String::as_str).unwrap_or("The ");
    let count = arguments
        .get(1)
        .and_then(|value| value.parse().ok())
        .unwrap_or(120usize);
    let temperature = arguments
        .get(2)
        .and_then(|value| value.parse().ok())
        .unwrap_or(0.8f32);

    let (model, tokenizer) = match load_checkpoint(CHECKPOINT_PATH) {
        Ok(loaded) => loaded,
        Err(err) => {
            eprintln!("cannot load {CHECKPOINT_PATH}: {err}");
            eprintln!("run `cargo run -p mini-llm-cli -- train` first");
            std::process::exit(1);
        }
    };

    let filtered: String = prompt
        .chars()
        .filter(|character| tokenizer.characters().contains(character))
        .collect();
    let encoded = if filtered.is_empty() {
        vec![0]
    } else {
        tokenizer
            .encode(&filtered)
            .expect("filtered prompt encodes")
    };

    let mut rng = Rng::new(2024);
    let generated = model
        .generate(&encoded, count, temperature, &mut rng)
        .expect("generate");
    println!("{}", tokenizer.decode(&generated).expect("decode"));
}
