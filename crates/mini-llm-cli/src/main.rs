use mini_llm_autograd::TensorValue;
use mini_llm_model::{scaled_dot_product_attention, ModelConfig};
use mini_llm_tensor::Tensor;

fn main() {
    let command = std::env::args()
        .nth(1)
        .unwrap_or_else(|| "demo".to_string());

    match command.as_str() {
        "demo" => demo(),
        "inspect-config" => inspect_config(),
        "autograd" => autograd_demo(),
        "train-line" => fit_line(),
        "help" | "--help" | "-h" => usage(),
        other => {
            eprintln!("Unknown command: {other}");
            usage();
            std::process::exit(2);
        }
    }
}

fn usage() {
    println!("Usage: cargo run -p mini-llm-cli -- <command>");
    println!();
    println!("Commands:");
    println!("  demo            scaled dot-product attention walkthrough");
    println!("  inspect-config  validate the default model configuration");
    println!("  autograd        computation graph + gradients for y = wx + b");
    println!("  train-line      fit y = 2x - 1 with manual SGD");
    println!("  help            show this message");
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
