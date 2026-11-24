//! Micro-benchmarks for the kernels and the training loop.
//!
//! Run in release mode for meaningful numbers:
//! `cargo run --release -p mini-llm-cli -- bench`

use std::time::Instant;

use mini_llm_data::{CharTokenizer, TextDataset};
use mini_llm_model::{ModelConfig, TinyGpt};
use mini_llm_nn::Rng;
use mini_llm_tensor::Tensor;
use mini_llm_train::{TrainConfig, Trainer};

use super::TINY_CORPUS;

const ITERATIONS: usize = 10;

pub fn run() {
    println!("mini-llm benchmarks (release build recommended)");
    println!();

    for size in [64usize, 128, 256] {
        benchmark_matmul(size);
    }

    benchmark_forward();
    benchmark_train_step();
    benchmark_generation();
    benchmark_tensor_clone();
}

fn benchmark_matmul(size: usize) {
    let left = patterned(size, size, 1.0);
    let right = patterned(size, size, 0.5);
    // Warm up.
    let _ = left.matmul(&right).expect("valid matmul");

    let started = Instant::now();
    for _ in 0..ITERATIONS {
        let _ = left.matmul(&right).expect("valid matmul");
    }
    let elapsed = started.elapsed().as_secs_f64() / ITERATIONS as f64;
    let flops = 2.0 * (size * size * size) as f64;
    println!(
        "  matmul {:>3}x{:<3}x{:<3}: {:>8.2} ms  {:>7.1} MFLOP/s",
        size,
        size,
        size,
        elapsed * 1_000.0,
        flops / elapsed / 1e6
    );
}

fn patterned(rows: usize, cols: usize, scale: f32) -> Tensor {
    let data = (0..rows * cols)
        .map(|index| ((index % 17) as f32 - 8.0) * 0.03 * scale)
        .collect();
    Tensor::from_2d(rows, cols, data).expect("valid tensor")
}

fn tiny_model(block: usize) -> TinyGpt {
    let tokenizer = CharTokenizer::from_text(TINY_CORPUS);
    let mut config = ModelConfig::modern(tokenizer.vocab_size());
    config.hidden_size = 32;
    config.max_seq_len = block;
    TinyGpt::new(config, &mut Rng::new(1)).expect("model")
}

fn benchmark_forward() {
    let model = tiny_model(16);
    let tokens: Vec<usize> = (0..16 * 4).map(|index| index % 31).collect();
    let _ = model.forward(&tokens, 4).expect("forward");

    let started = Instant::now();
    for _ in 0..ITERATIONS {
        let _ = model.forward(&tokens, 4).expect("forward");
    }
    let elapsed = started.elapsed().as_secs_f64() / ITERATIONS as f64;
    println!(
        "  forward (2 layers, hidden 32, block 16, batch 4): {:>8.2} ms",
        elapsed * 1_000.0
    );
}

fn benchmark_train_step() {
    let tokenizer = CharTokenizer::from_text(TINY_CORPUS);
    let tokens = tokenizer.encode(TINY_CORPUS).expect("corpus encodes");
    let dataset = TextDataset::new(tokens, 16).expect("dataset");
    let model = tiny_model(16);
    let mut trainer = Trainer::new(
        model,
        dataset,
        TrainConfig {
            steps: 0,
            batch_size: 2,
            learning_rate: 3e-3,
            weight_decay: 0.0,
            seed: 3,
        },
    );

    for _ in 0..2 {
        trainer.step().expect("warmup step");
    }
    let started = Instant::now();
    for _ in 0..ITERATIONS {
        trainer.step().expect("step");
    }
    let elapsed = started.elapsed().as_secs_f64() / ITERATIONS as f64;
    println!(
        "  train step (batch 2, block 16):                  {:>8.2} ms",
        elapsed * 1_000.0
    );
}

fn benchmark_generation() {
    let model = tiny_model(16);
    let prompt = [1usize, 2, 3];

    let started = Instant::now();
    for _ in 0..ITERATIONS {
        let mut rng = Rng::new(5);
        let _ = model
            .generate(&prompt, 32, 0.8, &mut rng)
            .expect("generate");
    }
    let elapsed = started.elapsed().as_secs_f64() / ITERATIONS as f64;
    println!(
        "  generate 32 tokens (KV cache):                    {:>8.2} ms",
        elapsed * 1_000.0
    );
}

/// Tensor values share their buffer, so a clone is a refcount bump rather than
/// a 4 MB copy. The autograd engine relies on this on every node access.
fn benchmark_tensor_clone() {
    let tensor = Tensor::full(vec![1024, 1024], 1.0).expect("valid tensor");
    let iterations = 1_000;
    let started = Instant::now();
    let mut checksum = 0usize;
    for _ in 0..iterations {
        let cloned = tensor.clone();
        checksum += cloned.data().len();
    }
    let per_clone = started.elapsed().as_secs_f64() / iterations as f64;
    println!(
        "  clone 1024x1024 (4 MB shared buffer):             {:>8.3} us (checksum {checksum})",
        per_clone * 1e6
    );
}
