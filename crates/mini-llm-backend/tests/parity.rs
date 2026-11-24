//! CPU reference checks and, with `--features candle`, CPU/Candle parity.

use mini_llm_backend::{BackendError, CpuBackend, MatMulBackend};
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
}

#[cfg(not(feature = "candle"))]
#[test]
fn candle_parity_is_feature_gated() {
    // Run `cargo test -p mini-llm-backend --features candle` to exercise the
    // accelerated backend.
}
