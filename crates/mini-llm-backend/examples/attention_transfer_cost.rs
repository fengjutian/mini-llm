//! Measures the cost of the Candle attention fallback.
//!
//! `DeviceTensor::attention_*` is native on host storage and delegates to the
//! same math (plus two transfers) on Candle storage. This example times both so
//! the fallback overhead is visible instead of guessed.
//!
//! Run with:
//! `cargo run --release -p mini-llm-backend --example attention_transfer_cost --features candle`

use std::hint::black_box;
use std::time::Instant;

#[cfg(feature = "candle")]
use mini_llm_backend::Device;
use mini_llm_backend::DeviceTensor;
use mini_llm_tensor::Tensor;

const ITERATIONS: usize = 20;

fn sample(rows: usize, cols: usize, scale: f32) -> Tensor {
    let data = (0..rows * cols)
        .map(|index| ((index % 11) as f32 - 5.0) * 0.1 * scale)
        .collect();
    Tensor::from_2d(rows, cols, data).expect("valid tensor")
}

fn main() {
    let (batch, seq, heads, kv_heads, head_dim) = (8usize, 16usize, 4usize, 2usize, 16usize);
    let rows = batch * seq;
    let q_host = sample(rows, heads * head_dim, 1.0);
    let k_host = sample(rows, kv_heads * head_dim, 0.5);
    let v_host = sample(rows, kv_heads * head_dim, 0.75);
    let output_width = heads * head_dim;
    let grad = Tensor::from_2d(
        rows,
        output_width,
        vec![1.0 / (rows * output_width) as f32; rows * output_width],
    )
    .expect("gradient");

    let q = DeviceTensor::from_host(q_host.clone());
    let k = DeviceTensor::from_host(k_host.clone());
    let v = DeviceTensor::from_host(v_host.clone());
    let g = DeviceTensor::from_host(grad.clone());

    let forward = || {
        q.attention_forward(&k, &v, batch, seq, heads, kv_heads, true)
            .expect("forward")
    };
    let backward = || {
        q.attention_backward(&k, &v, &g, batch, seq, heads, kv_heads, true)
            .expect("backward")
    };

    // Warm up both paths.
    for _ in 0..3 {
        black_box(forward());
        black_box(backward());
    }
    let host_forward = time(ITERATIONS, &forward);
    let host_backward = time(ITERATIONS, &backward);

    println!(
        "attention: batch {batch}, seq {seq}, heads {heads}/{kv_heads}, head_dim {head_dim} ({} rows)",
        rows
    );
    println!(
        "host  forward  {:>7.3} ms   backward  {:>7.3} ms",
        host_forward, host_backward
    );

    #[cfg(feature = "candle")]
    {
        let device = Device::candle_cpu();
        let q_device = q.to(&device).expect("upload q");
        let k_device = k.to(&device).expect("upload k");
        let v_device = v.to(&device).expect("upload v");
        let g_device = g.to(&device).expect("upload grad");

        let candle_forward = || {
            q_device
                .attention_forward(&k_device, &v_device, batch, seq, heads, kv_heads, true)
                .expect("candle forward")
        };
        let candle_backward = || {
            q_device
                .attention_backward(
                    &k_device, &v_device, &g_device, batch, seq, heads, kv_heads, true,
                )
                .expect("candle backward")
        };
        for _ in 0..3 {
            black_box(candle_forward());
            black_box(candle_backward());
        }
        let candle_forward_ms = time(ITERATIONS, &candle_forward);
        let candle_backward_ms = time(ITERATIONS, &candle_backward);

        // Pure transfer cost: host -> device -> host for the three inputs.
        let transfer = || {
            let q = DeviceTensor::from_host(q_host.clone())
                .to(&device)
                .expect("q");
            let k = DeviceTensor::from_host(k_host.clone())
                .to(&device)
                .expect("k");
            let v = DeviceTensor::from_host(v_host.clone())
                .to(&device)
                .expect("v");
            black_box((q, k, v));
        };
        let transfer_ms = time(ITERATIONS, &transfer);

        println!(
            "candle forward {:>7.3} ms   backward  {:>7.3} ms",
            candle_forward_ms, candle_backward_ms
        );
        println!("upload q/k/v  {:>7.3} ms", transfer_ms);
        println!(
            "candle overhead vs host: forward {:.1}x, backward {:.1}x",
            candle_forward_ms / host_forward,
            candle_backward_ms / host_backward
        );
    }
    #[cfg(not(feature = "candle"))]
    {
        println!("rebuild with `--features candle` to measure the Candle fallback");
    }
}

fn time<T>(iterations: usize, mut operation: impl FnMut() -> T) -> f64 {
    let started = Instant::now();
    for _ in 0..iterations {
        operation();
    }
    started.elapsed().as_secs_f64() * 1_000.0 / iterations as f64
}
