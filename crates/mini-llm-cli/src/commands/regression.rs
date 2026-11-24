//! Linear regression with gradients from autograd and a hand-written SGD step.

use mini_llm_autograd::TensorValue;

/// Fits `y = 2x - 1` so the update rule stays visible: `w <- w - lr * dL/dw`.
pub fn run() {
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
