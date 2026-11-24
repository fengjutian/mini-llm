//! Gradient demo: the `y = wx + b` example from `docs/AUTOGRAD_ENGINE.md`.

use mini_llm_autograd::TensorValue;

/// Reproduces the introductory example: `y = wx + b`, `L = (y - 10)^2` with
/// `x = 2`, `w = 3`, `b = 1`.
pub fn run() {
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

/// Compares the analytic gradient of `mean(relu(matmul(x, w)))` with central
/// differences, mirroring `crates/mini-llm-autograd/tests/gradients.rs`.
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
