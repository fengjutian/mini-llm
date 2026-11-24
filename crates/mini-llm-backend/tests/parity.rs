//! CPU reference checks and, with `--features candle`, CPU/Candle parity.

use mini_llm_backend::{Backend, BackendError, CpuBackend};
use mini_llm_tensor::Tensor;

fn sample(rows: usize, cols: usize, seed: u32) -> Tensor {
    let mut state = seed.max(1);
    let mut data = Vec::with_capacity(rows * cols);
    for _ in 0..rows * cols {
        state ^= state << 13;
        state ^= state >> 17;
        state ^= state << 5;
        data.push((state as f32 / u32::MAX as f32) * 2.0 - 1.0);
    }
    Tensor::from_2d(rows, cols, data).unwrap()
}

/// Deliberately naive reference product, independent of the tensor kernel.
fn reference_matmul(left: &Tensor, right: &Tensor) -> Vec<f32> {
    let (m, k) = (left.shape()[0], left.shape()[1]);
    let n = right.shape()[1];
    let mut out = vec![0.0f32; m * n];
    for row in 0..m {
        for inner in 0..k {
            let a = left.data()[row * k + inner];
            for col in 0..n {
                out[row * n + col] += a * right.data()[inner * n + col];
            }
        }
    }
    out
}

fn reference_rms_norm(input: &Tensor, gamma: &Tensor, eps: f32) -> Vec<f32> {
    let (rows, cols) = (input.shape()[0], input.shape()[1]);
    let mut out = vec![0.0f32; rows * cols];
    for row in 0..rows {
        let start = row * cols;
        let mean_square: f32 = input.data()[start..start + cols]
            .iter()
            .map(|value| value * value)
            .sum::<f32>()
            / cols as f32;
        let denominator = (mean_square + eps).sqrt();
        for col in 0..cols {
            out[start + col] = input.data()[start + col] / denominator * gamma.data()[col];
        }
    }
    out
}

#[test]
fn cpu_backend_matches_the_reference_product() {
    let left = sample(4, 5, 1);
    let right = sample(5, 3, 2);
    let product = CpuBackend.matmul(&left, &right).unwrap();
    let expected = reference_matmul(&left, &right);
    assert_eq!(product.shape(), &[4, 3]);
    for (actual, expected) in product.data().iter().zip(expected.iter()) {
        assert!((actual - expected).abs() < 1e-5);
    }
}

#[test]
fn cpu_backend_rejects_bad_shapes() {
    let left = sample(2, 3, 3);
    let right = sample(2, 3, 4);
    assert!(matches!(
        CpuBackend.matmul(&left, &right),
        Err(BackendError::Tensor(_))
    ));
}

#[test]
fn cpu_softmax_matches_the_tensor_kernel() {
    let input = sample(3, 5, 7);
    let backend = CpuBackend.softmax_rows(&input).unwrap();
    let reference = input.softmax_rows().unwrap();
    for (actual, expected) in backend.data().iter().zip(reference.data()) {
        assert!((actual - expected).abs() < 1e-6);
    }
    for row in backend.data().chunks(5) {
        assert!((row.iter().sum::<f32>() - 1.0).abs() < 1e-5);
    }
}

#[test]
fn cpu_rms_norm_matches_the_manual_formula() {
    let input = sample(4, 6, 9);
    let mut gamma = Vec::with_capacity(6);
    for index in 0..6 {
        gamma.push(0.5 + index as f32 * 0.2);
    }
    let gamma = Tensor::from_1d(gamma).unwrap();

    let backend = CpuBackend.rms_norm(&input, &gamma, 1e-5).unwrap();
    let expected = reference_rms_norm(&input, &gamma, 1e-5);
    assert_eq!(backend.shape(), input.shape());
    for (actual, expected) in backend.data().iter().zip(expected.iter()) {
        assert!((actual - expected).abs() < 1e-5);
    }
}

#[cfg(feature = "candle")]
#[test]
fn candle_matches_the_cpu_backend() {
    use mini_llm_backend::CandleBackend;

    let backend = CandleBackend::cpu();

    for (rows, shared, cols, seed) in [(4, 5, 3, 11u32), (16, 16, 16, 12), (1, 7, 9, 13)] {
        let left = sample(rows, shared, seed);
        let right = sample(shared, cols, seed + 100);
        let cpu = CpuBackend.matmul(&left, &right).unwrap();
        let candle = backend.matmul(&left, &right).unwrap();
        assert_eq!(cpu.shape(), candle.shape());
        for (a, b) in cpu.data().iter().zip(candle.data()) {
            assert!((a - b).abs() < 1e-4, "candle mismatch: {a} vs {b}");
        }
    }

    let input = sample(4, 6, 17);
    let cpu = CpuBackend.softmax_rows(&input).unwrap();
    let candle = backend.softmax_rows(&input).unwrap();
    for (a, b) in cpu.data().iter().zip(candle.data()) {
        assert!((a - b).abs() < 1e-4);
    }

    let mut gamma = Vec::with_capacity(6);
    for index in 0..6 {
        gamma.push(0.5 + index as f32 * 0.2);
    }
    let gamma = Tensor::from_1d(gamma).unwrap();
    let cpu = CpuBackend.rms_norm(&input, &gamma, 1e-5).unwrap();
    let candle = backend.rms_norm(&input, &gamma, 1e-5).unwrap();
    for (a, b) in cpu.data().iter().zip(candle.data()) {
        assert!((a - b).abs() < 1e-4);
    }
}

#[cfg(not(feature = "candle"))]
#[test]
fn candle_parity_is_feature_gated() {
    // Run `cargo test -p mini-llm-backend --features candle` to exercise the
    // accelerated backend.
}
