//! Finite-difference gradient checks for the autograd engine.
//!
//! Analytic gradients produced by `backward()` are compared against central
//! differences of the same forward computation. Because the tensor engine is
//! `f32`, the step size is relatively large (`1e-2`) and comparisons use a
//! mixed absolute/relative tolerance.

use mini_llm_autograd::{AutogradError, TensorValue};

const STEP: f32 = 1e-2;

fn numeric_gradient(f: impl Fn(&[f32]) -> f32, params: &[f32]) -> Vec<f32> {
    let mut gradients = Vec::with_capacity(params.len());
    for index in 0..params.len() {
        let mut plus = params.to_vec();
        plus[index] += STEP;
        let mut minus = params.to_vec();
        minus[index] -= STEP;
        gradients.push((f(&plus) - f(&minus)) / (2.0 * STEP));
    }
    gradients
}

fn assert_close(actual: &[f32], expected: &[f32], context: &str) {
    assert_eq!(actual.len(), expected.len(), "{context}: length mismatch");
    for (index, (a, e)) in actual.iter().zip(expected.iter()).enumerate() {
        let tolerance = 1e-2 * (1.0 + e.abs());
        assert!(
            (a - e).abs() <= tolerance,
            "{context}: gradient[{index}] analytic={a} numeric={e} (tolerance {tolerance})"
        );
    }
}

/// Builds a one-parameter graph, differentiates it, and compares the result
/// with central differences evaluated by rebuilding the graph in `f32`.
fn check_gradient(
    param_data: &[f32],
    param_shape: &[usize],
    build: impl Fn(&TensorValue) -> Result<TensorValue, AutogradError>,
) {
    let param = TensorValue::parameter(param_data.to_vec(), param_shape.to_vec()).unwrap();
    let loss = build(&param).unwrap();
    loss.backward().unwrap();
    let analytic = param.grad().expect("parameter should receive a gradient");

    let numeric = numeric_gradient(
        |values| {
            let param = TensorValue::parameter(values.to_vec(), param_shape.to_vec()).unwrap();
            build(&param).unwrap().data()[0]
        },
        param_data,
    );

    assert_close(analytic.data(), &numeric, "gradient check");
}

#[test]
fn matmul_relu_mean_matches_finite_differences() {
    let x = TensorValue::constant(vec![0.5, -0.3, 0.8, 0.2, -0.6, 0.9], vec![2, 3]).unwrap();
    let weights = vec![0.7, -0.4, 0.1, 0.3, 0.9, -0.2];

    check_gradient(&weights, &[3, 2], |w| x.matmul(w)?.relu()?.mean());
}

#[test]
fn softmax_then_linear_matches_finite_differences() {
    let weights = vec![0.6, -0.2, 0.4, 0.1, 0.8, -0.5];

    check_gradient(&weights, &[2, 3], |logits| {
        let probabilities = logits.softmax_rows()?;
        let coefficients =
            TensorValue::constant(vec![2.0, -1.0, 0.5, 1.5, 0.25, -0.75], vec![2, 3]).unwrap();
        probabilities.mul(&coefficients)?.sum()
    });
}

#[test]
fn cross_entropy_matches_finite_differences() {
    let logits = vec![0.4, -0.2, 0.1, 0.9, -0.7, 0.3];
    check_gradient(&logits, &[2, 3], |values| values.cross_entropy(&[1, 2]));
}

#[test]
fn squared_error_matches_the_hand_written_example() {
    // From the autograd article: y = w*x + b, L = (y - 10)^2
    let x = TensorValue::constant(vec![2.0], vec![1, 1]).unwrap();
    let target = TensorValue::constant(vec![10.0], vec![1]).unwrap();
    let w = TensorValue::parameter(vec![3.0], vec![1, 1]).unwrap();
    let b = TensorValue::parameter(vec![1.0], vec![1]).unwrap();

    let y = x.matmul(&w).unwrap().add(&b).unwrap();
    assert!((y.data()[0] - 7.0).abs() < 1e-6);

    let loss = y.sub(&target).unwrap().powf(2.0).unwrap().mean().unwrap();
    assert!((loss.data()[0] - 9.0).abs() < 1e-5);

    loss.backward().unwrap();

    // dL/dw = 2(wx + b - 10) * x = 2 * (-3) * 2 = -12
    // dL/db = 2(wx + b - 10)     = -6
    assert!((w.grad().unwrap().data()[0] + 12.0).abs() < 1e-4);
    assert!((b.grad().unwrap().data()[0] + 6.0).abs() < 1e-4);
}

#[test]
fn shared_subexpressions_accumulate_gradients() {
    let x = TensorValue::parameter(vec![1.0, -2.0, 3.0], vec![3]).unwrap();
    // y = x * x + x  =>  dy/dx = 2x + 1
    let loss = x.mul(&x).unwrap().add(&x).unwrap().sum().unwrap();
    loss.backward().unwrap();

    assert_close(x.grad().unwrap().data(), &[3.0, -3.0, 7.0], "shared node");
}

#[test]
fn transpose_and_add_backward() {
    let a = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0];
    check_gradient(&a, &[2, 3], |value| value.transpose()?.sum());
}

#[test]
fn zero_grad_removes_stale_gradients() {
    let x = TensorValue::parameter(vec![1.0, 2.0], vec![2]).unwrap();
    let loss = x.mul(&x).unwrap().sum().unwrap();
    loss.backward().unwrap();
    assert!(x.grad().is_some());

    loss.zero_grad();
    assert!(x.grad().is_none());
}

#[test]
fn backward_without_tracking_is_rejected() {
    let constant = TensorValue::constant(vec![1.0], vec![1]).unwrap();
    assert_eq!(constant.backward(), Err(AutogradError::NoGradientTracked));
}

#[test]
fn cross_entropy_rejects_invalid_targets() {
    let logits = TensorValue::parameter(vec![0.1, 0.2, 0.3], vec![1, 3]).unwrap();
    assert!(matches!(
        logits.cross_entropy(&[3]),
        Err(AutogradError::TargetOutOfRange { .. })
    ));
    assert!(matches!(
        logits.cross_entropy(&[0, 1]),
        Err(AutogradError::TargetCountMismatch { .. })
    ));
}

#[test]
fn parameters_are_leaf_nodes_only() {
    let x = TensorValue::constant(vec![1.0, 2.0], vec![1, 2]).unwrap();
    let w = TensorValue::parameter(vec![0.5, -0.5], vec![2, 1]).unwrap();
    let b = TensorValue::parameter(vec![0.1], vec![1]).unwrap();
    let loss = x.matmul(&w).unwrap().add(&b).unwrap().sum().unwrap();

    let parameters = loss.parameters();
    assert_eq!(parameters.len(), 2);
    assert!(parameters.iter().all(|parameter| parameter.requires_grad()));
    assert!(parameters.iter().all(|parameter| parameter.is_leaf()));
    assert!(loss.graph_summary().len() >= 4);
}

#[test]
fn gather_rows_matches_finite_differences() {
    let embeddings = vec![0.1, -0.2, 0.3, 0.4, -0.5, 0.6];
    check_gradient(&embeddings, &[3, 2], |table| {
        let gathered = table.gather_rows(&[2, 0, 2])?;
        let coefficients =
            TensorValue::constant(vec![1.0, -1.0, 0.5, 2.0, -0.25, 0.75], vec![3, 2])?;
        gathered.mul(&coefficients)?.sum()
    });
}

#[test]
fn mean_rows_keepdim_matches_finite_differences() {
    let values = vec![1.0, 2.0, 3.0, 4.0, -1.0, 0.5];
    check_gradient(&values, &[2, 3], |x| {
        let centered = x.sub(&x.mean_rows_keepdim()?)?;
        centered.powf(2.0)?.mean_rows_keepdim()?.sum()
    });
}

#[test]
fn parameter_values_can_be_updated_in_place() {
    let mut weight = TensorValue::parameter(vec![1.0, 2.0], vec![2]).unwrap();
    let updated = weight.value().scale(0.5).unwrap();
    weight.set_value(updated).unwrap();
    assert_eq!(weight.data(), &[0.5, 1.0]);

    let constant = TensorValue::constant(vec![1.0], vec![1]).unwrap();
    assert_eq!(
        constant.set_value(constant.value()),
        Err(AutogradError::NotAParameter)
    );
    assert_ne!(weight.id(), constant.id());
}
