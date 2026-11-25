//! Micro-benchmarks for the kernels and the training loop.
//!
//! Run in release mode for meaningful numbers:
//! `cargo run --release -p mini-llm-cli -- bench`

use std::alloc::{GlobalAlloc, Layout, System};
use std::sync::atomic::{AtomicUsize, Ordering};
use std::time::Instant;

use mini_llm_data::{BpeTokenizer, CharTokenizer, TextDataset};
use mini_llm_model::{ModelConfig, TinyGpt};
use mini_llm_nn::Rng;
use mini_llm_tensor::{F16Matrix, Tensor};
use mini_llm_train::{TrainConfig, Trainer};

use super::TINY_CORPUS;

const ITERATIONS: usize = 10;
const ROUNDS: usize = 3;

/// Counts bytes handed out by the allocator, so the benchmark can report the
/// allocation cost the optimizer still pays per step.
struct CountingAllocator;

static ALLOCATED_BYTES: AtomicUsize = AtomicUsize::new(0);

unsafe impl GlobalAlloc for CountingAllocator {
    unsafe fn alloc(&self, layout: Layout) -> *mut u8 {
        ALLOCATED_BYTES.fetch_add(layout.size(), Ordering::Relaxed);
        unsafe { System.alloc(layout) }
    }

    unsafe fn dealloc(&self, ptr: *mut u8, layout: Layout) {
        unsafe { System.dealloc(ptr, layout) }
    }

    unsafe fn realloc(&self, ptr: *mut u8, layout: Layout, new_size: usize) -> *mut u8 {
        ALLOCATED_BYTES.fetch_add(new_size, Ordering::Relaxed);
        unsafe { System.realloc(ptr, layout, new_size) }
    }
}

#[global_allocator]
static GLOBAL: CountingAllocator = CountingAllocator;

fn allocated_since(start: usize) -> usize {
    ALLOCATED_BYTES.load(Ordering::Relaxed) - start
}

pub fn run() {
    println!("mini-llm benchmarks (release build recommended)");
    println!("  timing: best of {ROUNDS} rounds, {ITERATIONS} iterations each");
    println!();

    for size in [64usize, 128, 256] {
        benchmark_matmul(size);
    }
    benchmark_f16_matmul(256);

    for (hidden, block, batch) in [
        (32usize, 16usize, 4usize),
        (128, 32, 4),
        (256, 32, 4),
        (512, 32, 4),
    ] {
        benchmark_forward(hidden, block, batch);
    }
    for (hidden, block, batch) in [(32usize, 16usize, 2usize), (128, 32, 2), (256, 32, 2)] {
        benchmark_train_step(hidden, block, batch);
    }
    benchmark_generation();
    benchmark_batch_generation();
    benchmark_bpe_encode();
    benchmark_tensor_clone();
}

/// Runs `op` for several rounds and returns the best seconds-per-iteration plus
/// the bytes allocated per iteration (allocation volume is load-independent).
fn measure(iterations: usize, mut op: impl FnMut()) -> (f64, f64) {
    let mut best_seconds = f64::INFINITY;
    let mut bytes_per_iteration = 0.0;
    for _ in 0..ROUNDS {
        let started = Instant::now();
        let allocation_start = ALLOCATED_BYTES.load(Ordering::Relaxed);
        for _ in 0..iterations {
            op();
        }
        let elapsed = started.elapsed().as_secs_f64() / iterations as f64;
        bytes_per_iteration = allocated_since(allocation_start) as f64 / iterations as f64;
        best_seconds = best_seconds.min(elapsed);
    }
    (best_seconds, bytes_per_iteration)
}

fn benchmark_matmul(size: usize) {
    let left = patterned(size, size, 1.0);
    let right = patterned(size, size, 0.5);
    // Warm up.
    let _ = left.matmul(&right).expect("valid matmul");

    let (seconds, _) = measure(ITERATIONS, || {
        let _ = left.matmul(&right).expect("valid matmul");
    });
    let flops = 2.0 * (size * size * size) as f64;
    println!(
        "  matmul {:>3}x{:<3}x{:<3}: {:>8.2} ms  {:>7.1} MFLOP/s",
        size,
        size,
        size,
        seconds * 1_000.0,
        flops / seconds / 1e6
    );
}

fn patterned(rows: usize, cols: usize, scale: f32) -> Tensor {
    let data = (0..rows * cols)
        .map(|index| ((index % 17) as f32 - 8.0) * 0.03 * scale)
        .collect();
    Tensor::from_2d(rows, cols, data).expect("valid tensor")
}

fn tiny_model(block: usize, hidden: usize) -> TinyGpt {
    let tokenizer = CharTokenizer::from_text(TINY_CORPUS);
    let mut config = ModelConfig::modern(tokenizer.vocab_size());
    config.hidden_size = hidden;
    config.max_seq_len = block;
    TinyGpt::new(config, &mut Rng::new(1)).expect("model")
}

fn benchmark_forward(hidden: usize, block: usize, batch: usize) {
    let model = tiny_model(block, hidden);
    let tokens: Vec<usize> = (0..block * batch).map(|index| index % 31).collect();
    let _ = model.forward(&tokens, batch).expect("forward");

    let (seconds, allocated) = measure(ITERATIONS, || {
        let _ = model.forward(&tokens, batch).expect("forward");
    });
    println!(
        "  forward (hidden {hidden:>3}, block {block:>2}, batch {batch}): {:>8.2} ms  {:>7.1} MB alloc",
        seconds * 1_000.0,
        allocated / 1e6
    );
}

fn benchmark_train_step(hidden: usize, block: usize, batch: usize) {
    let tokenizer = CharTokenizer::from_text(TINY_CORPUS);
    let tokens = tokenizer.encode(TINY_CORPUS).expect("corpus encodes");
    let dataset = TextDataset::new(tokens, block).expect("dataset");
    let model = tiny_model(block, hidden);
    let mut trainer = Trainer::new(
        model,
        dataset,
        TrainConfig {
            steps: 0,
            batch_size: batch,
            learning_rate: 3e-3,
            weight_decay: 0.0,
            seed: 3,
        },
    );

    for _ in 0..2 {
        trainer.step().expect("warmup step");
    }
    let (seconds, allocated) = measure(ITERATIONS, || {
        trainer.step().expect("step");
    });
    println!(
        "  train step (hidden {hidden:>3}, block {block:>2}, batch {batch}): {:>8.2} ms  {:>7.1} MB alloc",
        seconds * 1_000.0,
        allocated / 1e6
    );
}

fn benchmark_generation() {
    let model = tiny_model(16, 32);
    let prompt = [1usize, 2, 3];

    let (seconds, allocated) = measure(ITERATIONS, || {
        let mut rng = Rng::new(5);
        let _ = model
            .generate(&prompt, 32, 0.8, &mut rng)
            .expect("generate");
    });
    println!(
        "  generate 32 tokens (KV cache):                    {:>8.2} ms  {:>7.2} MB alloc",
        seconds * 1_000.0,
        allocated / 1e6
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

/// Throughput of the linked-list + heap BPE encoder.
fn benchmark_bpe_encode() {
    let corpus: String = (0..500)
        .map(|index| format!("token{index} jumps over the lazy dog {index}. "))
        .collect();
    let tokenizer = BpeTokenizer::train(&corpus, 512);
    let text = corpus.repeat(4);
    let (seconds, _) = measure(3, || {
        let _ = tokenizer.encode(&text);
    });
    let megabytes = text.len() as f64 / 1e6;
    println!(
        "  bpe encode ({} KB, {} merges): {:>8.2} MB/s",
        text.len() / 1024,
        tokenizer.vocab_size() - 256,
        megabytes / seconds
    );
}

/// Batched generation: cached (one forward step per token) vs window recompute.
fn benchmark_batch_generation() {
    let model = tiny_model(32, 32);
    let prompts = vec![vec![1usize, 2, 3]; 8];

    let (cached, _) = measure(3, || {
        let mut rng = Rng::new(5);
        let _ = model
            .generate_batch_cached(
                &prompts,
                16,
                mini_llm_model::Sampling::with_temperature(0.8),
                &mut rng,
            )
            .expect("cached batch generate");
    });
    let (uncached, _) = measure(3, || {
        let mut rng = Rng::new(5);
        let _ = model
            .generate_batch(
                &prompts,
                16,
                mini_llm_model::Sampling::with_temperature(0.8),
                &mut rng,
            )
            .expect("uncached batch generate");
    });
    println!(
        "  batch generate 8x16 (cached):  {:>8.2} ms",
        cached * 1_000.0
    );
    println!(
        "  batch generate 8x16 (uncached):{:>8.2} ms",
        uncached * 1_000.0
    );
}

/// f16 storage with f32 accumulation, compared against the f32 kernel.
fn benchmark_f16_matmul(size: usize) {
    let left_f32 = patterned(size, size, 1.0);
    let right_f32 = patterned(size, size, 0.5);
    let left = F16Matrix::from_tensor(&left_f32).expect("valid f16 matrix");
    let right = F16Matrix::from_tensor(&right_f32).expect("valid f16 matrix");

    let (quantize_seconds, _) = measure(5, || {
        let _ = F16Matrix::from_tensor(&left_f32).expect("quantize");
    });
    let (seconds, _) = measure(ITERATIONS, || {
        let _ = left.matmul(&right).expect("f16 matmul");
    });
    let flops = 2.0 * (size * size * size) as f64;
    println!(
        "  f16 matmul {:>3}x{:<3}x{:<3}: {:>8.2} ms  {:>7.1} MFLOP/s  (quantize {:.2} ms)",
        size,
        size,
        size,
        seconds * 1_000.0,
        flops / seconds / 1e6,
        quantize_seconds * 1_000.0
    );

    let reference = left_f32.matmul(&right_f32).expect("f32 matmul");
    let quantized = left.matmul(&right).expect("f16 matmul");
    let max_error = reference
        .data()
        .iter()
        .zip(quantized.data())
        .map(|(expected, actual)| (expected - actual).abs())
        .fold(0.0f32, f32::max);
    println!("    max abs error vs f32: {max_error:.5}");
}