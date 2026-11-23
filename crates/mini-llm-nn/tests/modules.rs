//! Integration tests for the neural-network building blocks.

use mini_llm_autograd::TensorValue;
use mini_llm_nn::{
    AdamW, Embedding, LayerNorm, Linear, Module, Optimizer, Relu, Rng, Sequential, Sgd, Tanh,
};
use mini_llm_tensor::Tensor;

fn input(data: Vec<f32>, shape: Vec<usize>) -> TensorValue {
    TensorValue::constant(data, shape).unwrap()
}

#[test]
fn linear_maps_shapes_and_reports_parameters() {
    let mut rng = Rng::new(3);
    let layer = Linear::new(3, 4, &mut rng).unwrap();
    let x = input(vec![0.1; 6], vec![2, 3]);

    let y = layer.forward(&x).unwrap();
    assert_eq!(y.shape(), &[2, 4]);
    assert_eq!(layer.parameters().len(), 2);
    assert_eq!(layer.num_parameters(), 3 * 4 + 4);
}

#[test]
fn linear_without_bias_has_one_parameter() {
    let mut rng = Rng::new(3);
    let layer = Linear::without_bias(3, 4, &mut rng).unwrap();
    let x = input(vec![0.1; 6], vec![2, 3]);
    assert_eq!(layer.forward(&x).unwrap().shape(), &[2, 4]);
    assert_eq!(layer.num_parameters(), 12);
}

#[test]
fn linear_rejects_mismatched_input_width() {
    let mut rng = Rng::new(3);
    let layer = Linear::new(3, 2, &mut rng).unwrap();
    let x = input(vec![0.1, 0.2], vec![1, 2]);
    let error = layer.forward(&x).unwrap_err().to_string();
    assert!(error.contains("expected input width 3, got 2"), "{error}");
}

#[test]
fn linear_forward_matches_manual_product() {
    let mut rng = Rng::new(11);
    let layer = Linear::new(2, 2, &mut rng).unwrap();
    layer
        .weight()
        .set_value(Tensor::from_2d(2, 2, vec![1.0, 2.0, 3.0, 4.0]).unwrap())
        .unwrap();
    layer
        .bias()
        .unwrap()
        .set_value(Tensor::from_1d(vec![0.5, -0.5]).unwrap())
        .unwrap();

    let x = input(vec![1.0, 2.0], vec![1, 2]);
    let y = layer.forward(&x).unwrap();
    // [1, 2] @ [[1,2],[3,4]] = [7, 10]; + [0.5, -0.5] = [7.5, 9.5]
    assert_eq!(y.data(), &[7.5, 9.5]);
}

#[test]
fn layernorm_produces_zero_mean_unit_variance_rows() {
    let layer = LayerNorm::new(4).unwrap();
    let x = input(vec![1.0, 2.0, 3.0, 4.0, -2.0, 0.0, 2.0, 4.0], vec![2, 4]);
    let y = layer.forward(&x).unwrap();
    let data = y.data();
    for row in data.chunks(4) {
        let mean: f32 = row.iter().sum::<f32>() / 4.0;
        let variance: f32 = row.iter().map(|v| (v - mean).powi(2)).sum::<f32>() / 4.0;
        assert!(mean.abs() < 1e-5, "mean {mean}");
        assert!((variance - 1.0).abs() < 1e-3, "variance {variance}");
    }
}

#[test]
fn layernorm_backward_reaches_affine_parameters() {
    let layer = LayerNorm::new(3).unwrap();
    let x = input(vec![1.0, 2.0, 4.0, 0.5, -1.0, 2.0], vec![2, 3]);
    let loss = layer
        .forward(&x)
        .unwrap()
        .powf(2.0)
        .unwrap()
        .mean()
        .unwrap();
    loss.backward().unwrap();

    let gamma_grad = layer.gamma().grad().unwrap();
    let beta_grad = layer.beta().grad().unwrap();
    assert!(gamma_grad.data().iter().any(|v| v.abs() > 1e-6));
    assert_eq!(beta_grad.shape(), &[3]);
}

#[test]
fn embedding_lookup_scatters_gradients() {
    let mut rng = Rng::new(9);
    let table = Embedding::new(4, 3, 0.1, &mut rng).unwrap();
    let indices = [2, 0, 2];
    let gathered = table.lookup(&indices).unwrap();
    assert_eq!(gathered.shape(), &[3, 3]);

    gathered.sum().unwrap().backward().unwrap();
    let grad = table.weight().grad().unwrap();
    assert_eq!(grad.shape(), &[4, 3]);
    // Row 2 is used twice, rows 1 and 3 never.
    assert_eq!(&grad.data()[6..9], &[2.0, 2.0, 2.0]);
    assert_eq!(&grad.data()[0..3], &[1.0, 1.0, 1.0]);
    assert_eq!(&grad.data()[3..6], &[0.0, 0.0, 0.0]);
}

#[test]
fn embedding_rejects_unknown_tokens() {
    let mut rng = Rng::new(9);
    let table = Embedding::new(4, 3, 0.1, &mut rng).unwrap();
    assert!(table.lookup(&[4]).is_err());
}

#[test]
fn sequential_collects_parameters_in_order() {
    let mut rng = Rng::new(21);
    let model = Sequential::new(vec![
        Box::new(Linear::new(2, 4, &mut rng).unwrap()),
        Box::new(Relu),
        Box::new(Linear::new(4, 1, &mut rng).unwrap()),
    ]);
    assert_eq!(model.parameters().len(), 4);
    assert_eq!(model.len(), 3);
    assert_eq!(model.summary().matches("linear").count(), 2);
}

#[test]
fn sgd_minimizes_a_linear_regression_loss() {
    let mut rng = Rng::new(17);
    let layer = Linear::new(1, 1, &mut rng).unwrap();
    let parameters = layer.parameters();
    let mut optimizer = Sgd::new(0.05);

    let x = input(vec![1.0, 2.0, 3.0, 4.0], vec![4, 1]);
    let target = input(vec![3.0, 5.0, 7.0, 9.0], vec![4, 1]);

    let mut first_loss = 0.0;
    let mut last_loss = 0.0;
    for step in 0..400 {
        let loss = layer
            .forward(&x)
            .unwrap()
            .sub(&target)
            .unwrap()
            .powf(2.0)
            .unwrap()
            .mean()
            .unwrap();
        if step == 0 {
            first_loss = loss.data()[0];
        }
        last_loss = loss.data()[0];
        loss.backward().unwrap();
        optimizer.step(&parameters).unwrap();
        loss.zero_grad();
    }

    assert!(
        last_loss < first_loss * 0.01,
        "loss {first_loss} -> {last_loss}"
    );
    assert!(last_loss < 1e-3, "final loss {last_loss}");
}

#[test]
fn adamw_first_step_matches_the_reference_formula() {
    let parameter = TensorValue::parameter(vec![1.0], vec![1]).unwrap();
    let loss = parameter.powf(2.0).unwrap().sum().unwrap();
    loss.backward().unwrap();
    // d/dp p^2 = 2p = 2.0

    let mut optimizer = AdamW::new(0.1).with_weight_decay(0.01);
    optimizer.step(std::slice::from_ref(&parameter)).unwrap();

    // m_hat = g, v_hat = g^2 => p -= lr * g / (|g| + eps) = 0.1
    // decoupled decay: p = 1.0 * (1 - 0.1 * 0.01) - 0.1 = 0.899
    let value = parameter.value().data()[0];
    assert!((value - 0.899).abs() < 1e-5, "value {value}");
    assert_eq!(optimizer.step_count(), 1);
}

#[test]
fn identical_seeds_give_identical_parameters() {
    let mut first = Rng::new(123);
    let mut second = Rng::new(123);
    let a = Linear::new(3, 2, &mut first).unwrap();
    let b = Linear::new(3, 2, &mut second).unwrap();
    assert_eq!(a.weight().data(), b.weight().data());
}

#[test]
fn xor_is_learnable_end_to_end() {
    let mut rng = Rng::new(42);
    let model = Sequential::new(vec![
        Box::new(Linear::new(2, 8, &mut rng).unwrap()),
        Box::new(Tanh),
        Box::new(Linear::new(8, 1, &mut rng).unwrap()),
    ]);
    let parameters = model.parameters();
    let mut optimizer = AdamW::new(0.05);

    let features = input(vec![0.0, 0.0, 0.0, 1.0, 1.0, 0.0, 1.0, 1.0], vec![4, 2]);
    let targets = input(vec![0.0, 1.0, 1.0, 0.0], vec![4, 1]);

    for _ in 0..1500 {
        let loss = model
            .forward(&features)
            .unwrap()
            .sub(&targets)
            .unwrap()
            .powf(2.0)
            .unwrap()
            .mean()
            .unwrap();
        loss.backward().unwrap();
        optimizer.step(&parameters).unwrap();
        loss.zero_grad();
    }

    let predictions = model.forward(&features).unwrap().data();
    let expected = [0.0, 1.0, 1.0, 0.0];
    for (index, (&prediction, &target)) in predictions.iter().zip(expected.iter()).enumerate() {
        assert!(
            (prediction - target).abs() < 0.2,
            "sample {index}: predicted {prediction}, target {target}"
        );
    }
}
