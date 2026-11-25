//! Device-resident tensor transfers, device-side matmul and the `DeviceLinear`
//! forward/backward skeleton, checked against host autograd.

use mini_llm_autograd::TensorValue;
use mini_llm_backend::{Device, DeviceAdamW, DeviceLinear, DeviceMlp, DeviceTensor};
use mini_llm_tensor::Tensor;

fn sample(rows: usize, cols: usize, scale: f32) -> Tensor {
    let data = (0..rows * cols)
        .map(|index| ((index % 7) as f32 - 3.0) * 0.25 * scale)
        .collect();
    Tensor::from_2d(rows, cols, data).unwrap()
}

fn assert_close(left: &[f32], right: &[f32], tolerance: f32, label: &str) {
    assert_eq!(left.len(), right.len(), "{label}: length");
    for (a, b) in left.iter().zip(right) {
        assert!((a - b).abs() <= tolerance, "{label}: {a} vs {b}");
    }
}

#[test]
fn cpu_device_roundtrips_and_multiplies() {
    let left = sample(3, 4, 1.0);
    let right = sample(4, 2, 0.5);
    let expected = left.matmul(&right).unwrap();

    let device_left = DeviceTensor::from_host(left.clone())
        .to(&Device::Cpu)
        .unwrap();
    let device_right = DeviceTensor::from_host(right.clone())
        .to(&Device::Cpu)
        .unwrap();

    assert_eq!(device_left.device(), Device::Cpu);
    assert!(!device_left.is_on_device());

    let product = device_left.matmul(&device_right).unwrap();
    assert_eq!(product.shape(), expected.shape());
    assert_eq!(product.to_host().unwrap().data(), expected.data());
    assert_eq!(device_left.to_host().unwrap().data(), left.data());
}

#[cfg(feature = "candle")]
#[test]
fn candle_device_roundtrips_and_multiplies() {
    let device = Device::candle_cpu();
    let left = sample(4, 5, 1.0);
    let right = sample(5, 3, 0.5);
    let expected = left.matmul(&right).unwrap();

    let device_left = DeviceTensor::from_host(left.clone()).to(&device).unwrap();
    let device_right = DeviceTensor::from_host(right.clone()).to(&device).unwrap();
    assert!(device_left.is_on_device());
    assert_eq!(device_left.device(), device);

    let product = device_left.matmul(&device_right).unwrap();
    assert!(product.is_on_device());
    let host = product.to_host().unwrap();
    assert_eq!(host.shape(), expected.shape());
    for (expected, actual) in expected.data().iter().zip(host.data()) {
        assert!((expected - actual).abs() < 1e-4);
    }

    // A host operand cannot be multiplied with a device operand directly.
    let host_operand = DeviceTensor::from_host(left);
    assert!(device_right.matmul(&host_operand).is_err());

    // Moving back to the CPU device drops the device copy.
    let back = device_left.to(&Device::Cpu).unwrap();
    assert!(!back.is_on_device());
}

/// Host autograd reference for `y = x W + b`, `loss = mean(y)`.
fn host_reference(x: &Tensor, w: &Tensor, b: &Tensor) -> (f32, Vec<f32>, Vec<f32>) {
    let xv = TensorValue::constant(x.data().to_vec(), x.shape().to_vec()).unwrap();
    let wv = TensorValue::parameter(w.data().to_vec(), w.shape().to_vec()).unwrap();
    let bv = TensorValue::parameter(b.data().to_vec(), b.shape().to_vec()).unwrap();
    let output = xv.matmul(&wv).unwrap().add(&bv).unwrap();
    let loss = output.mean().unwrap();
    loss.backward().unwrap();
    (
        loss.data()[0],
        wv.grad().unwrap().data().to_vec(),
        bv.grad().unwrap().data().to_vec(),
    )
}

fn check_device_linear_backward(device: &Device, tolerance: f32) {
    let x = sample(3, 4, 1.0);
    let w = sample(4, 2, 0.5);
    let b = sample(1, 2, 0.25);
    let (expected_loss, expected_grad_w, expected_grad_b) = host_reference(&x, &w, &b);

    // d(mean)/dy = 1 / numel, so the upstream gradient is a scaled ones matrix.
    let numel = (3 * 2) as f32;
    let grad_output_host = Tensor::from_2d(3, 2, vec![1.0 / numel; 6]).unwrap();

    let x_device = DeviceTensor::from_host(x.clone()).to(device).unwrap();
    let w_device = DeviceTensor::from_host(w.clone()).to(device).unwrap();
    let b_device = DeviceTensor::from_host(b.clone()).to(device).unwrap();
    let grad_output = DeviceTensor::from_host(grad_output_host.clone())
        .to(device)
        .unwrap();

    let layer = DeviceLinear::new(w_device, Some(b_device));
    let output = layer.forward(&x_device).unwrap();
    let loss = output.mean_all_value().unwrap();
    assert!(
        (loss - expected_loss).abs() < tolerance,
        "loss {loss} vs {expected_loss}"
    );

    let (grad_x, grad_w, grad_b) = layer.backward(&x_device, &grad_output).unwrap();

    // dx = G Wᵀ
    let expected_grad_x = grad_output_host.matmul(&w.transpose_2d().unwrap()).unwrap();

    assert_close(
        grad_x.to_host().unwrap().data(),
        expected_grad_x.data(),
        tolerance,
        "dx",
    );
    assert_close(
        grad_w.to_host().unwrap().data(),
        &expected_grad_w,
        tolerance,
        "dW",
    );
    assert_close(
        grad_b.unwrap().to_host().unwrap().data(),
        &expected_grad_b,
        tolerance,
        "db",
    );
}

#[test]
fn device_linear_backward_matches_host_autograd_on_cpu() {
    check_device_linear_backward(&Device::Cpu, 1e-5);
}

#[cfg(feature = "candle")]
#[test]
fn device_linear_backward_matches_host_autograd_on_candle() {
    check_device_linear_backward(&Device::candle_cpu(), 1e-4);
}

/// Host autograd reference for a two-layer ReLU MLP: `mean(relu(xW1+b1)W2+b2)`.
#[allow(clippy::type_complexity)]
fn host_mlp(
    x: &Tensor,
    w1: &Tensor,
    b1: &Tensor,
    w2: &Tensor,
    b2: &Tensor,
) -> (f32, Vec<f32>, Vec<f32>, Vec<f32>, Vec<f32>, Vec<f32>) {
    let xv = TensorValue::parameter(x.data().to_vec(), x.shape().to_vec()).unwrap();
    let w1v = TensorValue::parameter(w1.data().to_vec(), w1.shape().to_vec()).unwrap();
    let b1v = TensorValue::parameter(b1.data().to_vec(), b1.shape().to_vec()).unwrap();
    let w2v = TensorValue::parameter(w2.data().to_vec(), w2.shape().to_vec()).unwrap();
    let b2v = TensorValue::parameter(b2.data().to_vec(), b2.shape().to_vec()).unwrap();
    let hidden = xv.matmul(&w1v).unwrap().add(&b1v).unwrap().relu().unwrap();
    let output = hidden.matmul(&w2v).unwrap().add(&b2v).unwrap();
    let loss = output.mean().unwrap();
    loss.backward().unwrap();
    (
        loss.data()[0],
        xv.grad().unwrap().data().to_vec(),
        w1v.grad().unwrap().data().to_vec(),
        b1v.grad().unwrap().data().to_vec(),
        w2v.grad().unwrap().data().to_vec(),
        b2v.grad().unwrap().data().to_vec(),
    )
}

fn check_device_mlp_backward(device: &Device, tolerance: f32) {
    let x = sample(3, 4, 1.0);
    let w1 = sample(4, 5, 0.5);
    let b1 = sample(1, 5, 0.25);
    let w2 = sample(5, 2, 0.5);
    let b2 = sample(1, 2, 0.25);

    let (expected_loss, expected_dx, expected_dw1, expected_db1, expected_dw2, expected_db2) =
        host_mlp(&x, &w1, &b1, &w2, &b2);

    let upload = |tensor: &Tensor| DeviceTensor::from_host(tensor.clone()).to(device).unwrap();
    let x_device = upload(&x);
    let layer1 = DeviceLinear::new(upload(&w1), Some(upload(&b1)));
    let layer2 = DeviceLinear::new(upload(&w2), Some(upload(&b2)));
    let mlp = DeviceMlp::new(vec![layer1, layer2]);

    let (output, inputs) = mlp.forward_with_cache(&x_device).unwrap();
    let loss = output.mean_all_value().unwrap();
    assert!(
        (loss - expected_loss).abs() < tolerance,
        "loss {loss} vs {expected_loss}"
    );

    let numel = (3 * 2) as f32;
    let grad_output = upload(&Tensor::from_2d(3, 2, vec![1.0 / numel; 6]).unwrap());
    let (grad_x, gradients) = mlp.backward(&inputs, &grad_output).unwrap();
    assert_eq!(gradients.len(), 2);

    assert_close(
        grad_x.to_host().unwrap().data(),
        &expected_dx,
        tolerance,
        "mlp dx",
    );
    assert_close(
        gradients[0].0.to_host().unwrap().data(),
        &expected_dw1,
        tolerance,
        "mlp dW1",
    );
    assert_close(
        gradients[0].1.as_ref().unwrap().to_host().unwrap().data(),
        &expected_db1,
        tolerance,
        "mlp db1",
    );
    assert_close(
        gradients[1].0.to_host().unwrap().data(),
        &expected_dw2,
        tolerance,
        "mlp dW2",
    );
    assert_close(
        gradients[1].1.as_ref().unwrap().to_host().unwrap().data(),
        &expected_db2,
        tolerance,
        "mlp db2",
    );
}

#[test]
fn device_mlp_backward_matches_host_autograd_on_cpu() {
    check_device_mlp_backward(&Device::Cpu, 1e-5);
}

#[cfg(feature = "candle")]
#[test]
fn device_mlp_backward_matches_host_autograd_on_candle() {
    check_device_mlp_backward(&Device::candle_cpu(), 1e-4);
}

use mini_llm_backend::DeviceValue;

fn check_device_tape(device: &Device, tolerance: f32) {
    let x = sample(3, 4, 1.0);
    let w1 = sample(4, 5, 0.5);
    let b1 = sample(1, 5, 0.25);
    let w2 = sample(5, 2, 0.5);
    let b2 = sample(1, 2, 0.25);
    let (expected_loss, expected_dx, expected_dw1, expected_db1, expected_dw2, expected_db2) =
        host_mlp(&x, &w1, &b1, &w2, &b2);

    let upload = |tensor: &Tensor| DeviceTensor::from_host(tensor.clone()).to(device).unwrap();
    let xv = DeviceValue::new(upload(&x), true);
    let w1v = DeviceValue::new(upload(&w1), true);
    let b1v = DeviceValue::new(upload(&b1), true);
    let w2v = DeviceValue::new(upload(&w2), true);
    let b2v = DeviceValue::new(upload(&b2), true);

    let hidden = xv.matmul(&w1v).unwrap().add(&b1v).unwrap().relu().unwrap();
    let output = hidden.matmul(&w2v).unwrap().add(&b2v).unwrap();
    let loss = output.mean().unwrap();

    let loss_value = loss.data().to_host().unwrap().data()[0];
    assert!(
        (loss_value - expected_loss).abs() < tolerance,
        "tape loss {loss_value} vs {expected_loss}"
    );

    loss.backward().unwrap();

    let grad = |value: &DeviceValue| value.grad().unwrap().to_host().unwrap().data().to_vec();
    assert_close(&grad(&xv), &expected_dx, tolerance, "tape dx");
    assert_close(&grad(&w1v), &expected_dw1, tolerance, "tape dW1");
    assert_close(&grad(&b1v), &expected_db1, tolerance, "tape db1");
    assert_close(&grad(&w2v), &expected_dw2, tolerance, "tape dW2");
    assert_close(&grad(&b2v), &expected_db2, tolerance, "tape db2");
}

#[test]
fn device_tape_matches_host_autograd_on_cpu() {
    check_device_tape(&Device::Cpu, 1e-5);
}

#[cfg(feature = "candle")]
#[test]
fn device_tape_matches_host_autograd_on_candle() {
    check_device_tape(&Device::candle_cpu(), 1e-4);
}

#[cfg(feature = "candle")]
#[test]
fn device_tape_matches_host_autograd_on_cuda_when_available() {
    let Ok(device) = Device::candle_cuda(0) else {
        eprintln!("no CUDA device available; skipping");
        return;
    };
    check_device_tape(&device, 2e-3);
}

#[test]
fn device_tape_accumulates_reused_gradients() {
    // loss = mean(y) + mean(y) must double dW and dx.
    let x = sample(3, 4, 1.0);
    let w = sample(4, 2, 0.5);

    let xv = TensorValue::parameter(x.data().to_vec(), x.shape().to_vec()).unwrap();
    let wv = TensorValue::parameter(w.data().to_vec(), w.shape().to_vec()).unwrap();
    let y = xv.matmul(&wv).unwrap();
    let loss = y.mean().unwrap().add(&y.mean().unwrap()).unwrap();
    loss.backward().unwrap();
    let expected_dx = xv.grad().unwrap().data().to_vec();
    let expected_dw = wv.grad().unwrap().data().to_vec();

    let upload = |tensor: &Tensor| {
        DeviceTensor::from_host(tensor.clone())
            .to(&Device::Cpu)
            .unwrap()
    };
    let device_x = DeviceValue::new(upload(&x), true);
    let device_w = DeviceValue::new(upload(&w), true);
    let device_y = device_x.matmul(&device_w).unwrap();
    let device_loss = device_y
        .mean()
        .unwrap()
        .add(&device_y.mean().unwrap())
        .unwrap();
    device_loss.backward().unwrap();

    assert_close(
        device_x.grad().unwrap().to_host().unwrap().data(),
        &expected_dx,
        1e-5,
        "accumulated dx",
    );
    assert_close(
        device_w.grad().unwrap().to_host().unwrap().data(),
        &expected_dw,
        1e-5,
        "accumulated dW",
    );
}

fn init_mlp_params() -> (Tensor, Tensor, Tensor, Tensor) {
    (
        sample(3, 4, 0.5),
        sample(1, 4, 0.25),
        sample(4, 2, 0.5),
        sample(1, 2, 0.25),
    )
}

fn regression_dataset() -> (Tensor, Tensor) {
    let x = Tensor::from_2d(
        8,
        3,
        (0..24)
            .map(|index| ((index as f32 % 5.0) - 2.0) * 0.3)
            .collect(),
    )
    .unwrap();
    let w = Tensor::from_2d(3, 2, vec![0.5, -0.25, 0.75, 0.1, -0.4, 0.6]).unwrap();
    let b = Tensor::from_2d(1, 2, vec![0.2, -0.1]).unwrap();
    let target = x.matmul(&w).unwrap().add(&b).unwrap();
    (x, target)
}

fn host_mlp_step(
    x: &TensorValue,
    target: &TensorValue,
    w1: &TensorValue,
    b1: &TensorValue,
    w2: &TensorValue,
    b2: &TensorValue,
) -> f32 {
    let hidden = x.matmul(w1).unwrap().add(b1).unwrap().relu().unwrap();
    let predicted = hidden.matmul(w2).unwrap().add(b2).unwrap();
    let diff = predicted.sub(target).unwrap();
    let loss = diff.mul(&diff).unwrap().mean().unwrap();
    loss.backward().unwrap();
    loss.data()[0]
}

fn device_mlp_step(
    x: &DeviceValue,
    target: &DeviceValue,
    w1: &DeviceValue,
    b1: &DeviceValue,
    w2: &DeviceValue,
    b2: &DeviceValue,
) -> f32 {
    let hidden = x.matmul(w1).unwrap().add(b1).unwrap().relu().unwrap();
    let predicted = hidden.matmul(w2).unwrap().add(b2).unwrap();
    let diff = predicted.sub(target).unwrap();
    let loss = diff.mul(&diff).unwrap().mean().unwrap();
    loss.backward().unwrap();
    loss.data().to_host().unwrap().data()[0]
}

fn sgd_update(param: &DeviceTensor, value: &DeviceValue, lr: f32) -> DeviceTensor {
    let grad = value.grad().expect("parameter gradient");
    param
        .add(&grad.scale(-lr).unwrap())
        .expect("device parameter update")
}

fn host_sgd(param: &mut [f32], value: &TensorValue, lr: f32) {
    let grad = value.grad().expect("parameter gradient");
    for (slot, gradient) in param.iter_mut().zip(grad.data()) {
        *slot -= lr * gradient;
    }
}

/// Runs the same 40-step MLP regression on host autograd and the device tape.
fn check_device_training(device: &Device, tolerance: f32) {
    let (x, target) = regression_dataset();
    let (w1_init, b1_init, w2_init, b2_init) = init_mlp_params();
    let (w1_data, b1_data, w2_data, b2_data) = (
        w1_init.data().to_vec(),
        b1_init.data().to_vec(),
        w2_init.data().to_vec(),
        b2_init.data().to_vec(),
    );

    // Host reference.
    let xv = TensorValue::constant(x.data().to_vec(), x.shape().to_vec()).unwrap();
    let tv = TensorValue::constant(target.data().to_vec(), target.shape().to_vec()).unwrap();
    let (mut host_w1, mut host_b1, mut host_w2, mut host_b2) = (
        w1_data.clone(),
        b1_data.clone(),
        w2_data.clone(),
        b2_data.clone(),
    );
    let mut host_losses = Vec::new();
    for _ in 0..40 {
        let w1 = TensorValue::parameter(host_w1.clone(), vec![3, 4]).unwrap();
        let b1 = TensorValue::parameter(host_b1.clone(), vec![1, 4]).unwrap();
        let w2 = TensorValue::parameter(host_w2.clone(), vec![4, 2]).unwrap();
        let b2 = TensorValue::parameter(host_b2.clone(), vec![1, 2]).unwrap();
        let loss = host_mlp_step(&xv, &tv, &w1, &b1, &w2, &b2);
        host_losses.push(loss);
        host_sgd(&mut host_w1, &w1, 0.05);
        host_sgd(&mut host_b1, &b1, 0.05);
        host_sgd(&mut host_w2, &w2, 0.05);
        host_sgd(&mut host_b2, &b2, 0.05);
    }

    // Device loop: parameters stay device-resident across every step.
    let upload = |tensor: &Tensor| DeviceTensor::from_host(tensor.clone()).to(device).unwrap();
    let device_x = DeviceValue::new(upload(&x), false);
    let device_target = DeviceValue::new(upload(&target), false);
    let mut w1 = upload(&w1_init);
    let mut b1 = upload(&b1_init);
    let mut w2 = upload(&w2_init);
    let mut b2 = upload(&b2_init);
    let mut device_losses = Vec::new();
    for _ in 0..40 {
        let w1_value = DeviceValue::new(w1.clone(), true);
        let b1_value = DeviceValue::new(b1.clone(), true);
        let w2_value = DeviceValue::new(w2.clone(), true);
        let b2_value = DeviceValue::new(b2.clone(), true);
        let loss = device_mlp_step(
            &device_x,
            &device_target,
            &w1_value,
            &b1_value,
            &w2_value,
            &b2_value,
        );
        device_losses.push(loss);
        w1 = sgd_update(&w1, &w1_value, 0.05);
        b1 = sgd_update(&b1, &b1_value, 0.05);
        w2 = sgd_update(&w2, &w2_value, 0.05);
        b2 = sgd_update(&b2, &b2_value, 0.05);
        assert_eq!(w1.is_on_device(), !matches!(device, Device::Cpu));
    }

    for (step, (host_loss, device_loss)) in host_losses.iter().zip(&device_losses).enumerate() {
        assert!(
            (host_loss - device_loss).abs() <= tolerance * host_loss.abs().max(1.0),
            "step {step}: host {host_loss} vs device {device_loss}"
        );
    }
    assert!(
        *device_losses.last().unwrap() < 0.75 * device_losses[0],
        "loss should fall: {:?} -> {:?}",
        device_losses[0],
        device_losses.last().unwrap()
    );

    assert_close(
        w1.to_host().unwrap().data(),
        &host_w1,
        tolerance,
        "trained W1",
    );
    assert_close(
        b1.to_host().unwrap().data(),
        &host_b1,
        tolerance,
        "trained b1",
    );
    assert_close(
        w2.to_host().unwrap().data(),
        &host_w2,
        tolerance,
        "trained W2",
    );
    assert_close(
        b2.to_host().unwrap().data(),
        &host_b2,
        tolerance,
        "trained b2",
    );
}

#[test]
fn device_training_loop_matches_host_autograd_on_cpu() {
    check_device_training(&Device::Cpu, 1e-4);
}

#[cfg(feature = "candle")]
#[test]
fn device_training_loop_matches_host_autograd_on_candle() {
    check_device_training(&Device::candle_cpu(), 1e-3);
}

fn host_adamw_reference(params: &mut [Vec<f32>], gradient_steps: &[Vec<Vec<f32>>], lr: f32) {
    let (beta1, beta2, eps, weight_decay) = (0.9f32, 0.999f32, 1e-8f32, 0.01f32);
    let mut first: Vec<Vec<f32>> = params.iter().map(|p| vec![0.0; p.len()]).collect();
    let mut second: Vec<Vec<f32>> = params.iter().map(|p| vec![0.0; p.len()]).collect();
    for (step, gradients) in gradient_steps.iter().enumerate() {
        let step = (step + 1) as i32;
        let bias1 = 1.0 - beta1.powi(step);
        let bias2 = 1.0 - beta2.powi(step);
        let decay = 1.0 - lr * weight_decay;
        for index in 0..params.len() {
            for position in 0..params[index].len() {
                let g = gradients[index][position];
                first[index][position] = beta1 * first[index][position] + (1.0 - beta1) * g;
                second[index][position] = beta2 * second[index][position] + (1.0 - beta2) * g * g;
                let m_hat = first[index][position] / bias1;
                let v_hat = second[index][position] / bias2;
                params[index][position] =
                    params[index][position] * decay - lr * m_hat / (v_hat.sqrt() + eps);
            }
        }
    }
}

fn check_device_adamw(device: &Device, tolerance: f32) {
    let first_param = sample(2, 3, 1.0);
    let second_param = sample(3, 1, 0.5);
    let shapes = [vec![2usize, 3], vec![3, 1]];

    let steps = 8usize;
    let mut gradient_steps = Vec::new();
    for step in 0..steps {
        let mut step_gradients = Vec::new();
        for (param_index, shape) in shapes.iter().enumerate() {
            let length: usize = shape.iter().product();
            let values = (0..length)
                .map(|index| {
                    ((index as f32 * 0.37) - 0.6)
                        * ((step + 1) as f32)
                        * 0.05
                        * (param_index as f32 + 1.0)
                })
                .collect();
            step_gradients.push(Tensor::new(values, shape.clone()).unwrap().data().to_vec());
        }
        gradient_steps.push(step_gradients);
    }

    let mut host_params = vec![first_param.data().to_vec(), second_param.data().to_vec()];
    host_adamw_reference(&mut host_params, &gradient_steps, 0.01);

    let upload = |tensor: &Tensor| DeviceTensor::from_host(tensor.clone()).to(device).unwrap();
    let mut device_params = vec![upload(&first_param), upload(&second_param)];
    let mut optimizer = DeviceAdamW::new(0.01);
    for step_gradients in &gradient_steps {
        let gradients: Vec<DeviceTensor> = step_gradients
            .iter()
            .zip(shapes.iter())
            .map(|(values, shape)| {
                DeviceTensor::from_host(Tensor::new(values.clone(), shape.clone()).unwrap())
                    .to(device)
                    .unwrap()
            })
            .collect();
        optimizer.step(&mut device_params, &gradients).unwrap();
    }

    assert_eq!(optimizer.step_count(), steps as u32);
    for (index, (device_param, host_param)) in device_params.iter().zip(&host_params).enumerate() {
        assert_close(
            device_param.to_host().unwrap().data(),
            host_param,
            tolerance,
            &format!("adamw param {index}"),
        );
        assert_eq!(
            device_param.is_on_device(),
            !matches!(device, Device::Cpu),
            "parameter {index} should stay on the device"
        );
    }
}

#[test]
fn device_adamw_matches_host_reference_on_cpu() {
    check_device_adamw(&Device::Cpu, 1e-5);
}

#[cfg(feature = "candle")]
#[test]
fn device_adamw_matches_host_reference_on_candle() {
    check_device_adamw(&Device::candle_cpu(), 1e-4);
}

#[test]
fn device_adamw_rejects_gradient_mismatch() {
    let parameter = DeviceTensor::from_host(sample(2, 2, 1.0));
    let wrong = DeviceTensor::from_host(sample(3, 3, 1.0));
    let mut parameters = vec![parameter];
    let mut optimizer = DeviceAdamW::new(0.01);
    assert!(optimizer.step(&mut parameters, &[wrong]).is_err());
}

fn check_softmax_parity(device: &Device, tolerance: f32) {
    let x = sample(3, 4, 1.5);
    let host_x = TensorValue::parameter(x.data().to_vec(), x.shape().to_vec()).unwrap();
    let host_out = host_x.softmax_rows().unwrap();
    let host_loss = host_out.mean().unwrap();
    host_loss.backward().unwrap();

    let device_x = DeviceValue::new(DeviceTensor::from_host(x.clone()).to(device).unwrap(), true);
    let out = device_x.softmax_rows().unwrap();
    let loss = out.mean().unwrap();
    loss.backward().unwrap();

    assert_close(
        out.data().to_host().unwrap().data(),
        &host_out.data(),
        tolerance,
        "softmax values",
    );
    assert_close(
        device_x.grad().unwrap().to_host().unwrap().data(),
        host_x.grad().unwrap().data(),
        tolerance,
        "softmax gradient",
    );
}

#[test]
fn device_softmax_matches_host_autograd_on_cpu() {
    check_softmax_parity(&Device::Cpu, 1e-5);
}

#[cfg(feature = "candle")]
#[test]
fn device_softmax_matches_host_autograd_on_candle() {
    check_softmax_parity(&Device::candle_cpu(), 1e-4);
}

fn check_embedding_parity(device: &Device, tolerance: f32) {
    let weight = sample(5, 3, 1.0);
    let indices = [0usize, 2, 4, 2];

    let host_weight =
        TensorValue::parameter(weight.data().to_vec(), weight.shape().to_vec()).unwrap();
    let host_rows = host_weight.gather_rows(&indices).unwrap();
    let host_loss = host_rows.mean().unwrap();
    host_loss.backward().unwrap();

    let device_weight = DeviceValue::new(
        DeviceTensor::from_host(weight.clone()).to(device).unwrap(),
        true,
    );
    let rows = device_weight.gather_rows(&indices).unwrap();
    let loss = rows.mean().unwrap();
    loss.backward().unwrap();

    assert_close(
        rows.data().to_host().unwrap().data(),
        &host_rows.data(),
        tolerance,
        "embedding values",
    );
    assert_close(
        device_weight.grad().unwrap().to_host().unwrap().data(),
        host_weight.grad().unwrap().data(),
        tolerance,
        "embedding gradient",
    );
}

#[test]
fn device_embedding_matches_host_autograd_on_cpu() {
    check_embedding_parity(&Device::Cpu, 1e-5);
}

#[cfg(feature = "candle")]
#[test]
fn device_embedding_matches_host_autograd_on_candle() {
    check_embedding_parity(&Device::candle_cpu(), 1e-4);
}

fn check_layer_norm_parity(device: &Device, tolerance: f32) {
    let x = sample(3, 4, 1.5);
    let gamma = sample(1, 4, 0.5);
    let beta = sample(1, 4, 0.25);
    let eps = 1e-5f32;

    let host_x = TensorValue::parameter(x.data().to_vec(), x.shape().to_vec()).unwrap();
    let host_gamma = TensorValue::parameter(gamma.data().to_vec(), gamma.shape().to_vec()).unwrap();
    let host_beta = TensorValue::parameter(beta.data().to_vec(), beta.shape().to_vec()).unwrap();
    let mean = host_x.mean_rows_keepdim().unwrap();
    let centered = host_x.sub(&mean).unwrap();
    let variance = centered.powf(2.0).unwrap().mean_rows_keepdim().unwrap();
    let inverse_std = variance.add_scalar(eps).unwrap().powf(-0.5).unwrap();
    let normalized = centered.mul(&inverse_std).unwrap();
    let host_out = normalized
        .mul(&host_gamma)
        .unwrap()
        .add(&host_beta)
        .unwrap();
    let host_loss = host_out.mean().unwrap();
    host_loss.backward().unwrap();

    let upload = |tensor: &Tensor| DeviceTensor::from_host(tensor.clone()).to(device).unwrap();
    let device_x = DeviceValue::new(upload(&x), true);
    let device_gamma = DeviceValue::new(upload(&gamma), true);
    let device_beta = DeviceValue::new(upload(&beta), true);
    let out = device_x
        .layer_norm(&device_gamma, &device_beta, eps)
        .unwrap();
    let loss = out.mean().unwrap();
    loss.backward().unwrap();

    assert_close(
        out.data().to_host().unwrap().data(),
        &host_out.data(),
        tolerance,
        "layernorm values",
    );
    assert_close(
        device_x.grad().unwrap().to_host().unwrap().data(),
        host_x.grad().unwrap().data(),
        tolerance,
        "layernorm dx",
    );
    assert_close(
        device_gamma.grad().unwrap().to_host().unwrap().data(),
        host_gamma.grad().unwrap().data(),
        tolerance,
        "layernorm dgamma",
    );
    assert_close(
        device_beta.grad().unwrap().to_host().unwrap().data(),
        host_beta.grad().unwrap().data(),
        tolerance,
        "layernorm dbeta",
    );
}

#[test]
fn device_layer_norm_matches_host_autograd_on_cpu() {
    check_layer_norm_parity(&Device::Cpu, 1e-5);
}

#[cfg(feature = "candle")]
#[test]
fn device_layer_norm_matches_host_autograd_on_candle() {
    check_layer_norm_parity(&Device::candle_cpu(), 1e-4);
}

fn check_cross_entropy_parity(device: &Device, tolerance: f32) {
    let logits = sample(4, 5, 1.5);
    let targets = [0usize, 2, 4, 1];

    let host_logits = TensorValue::parameter(logits.data().to_vec(), logits.shape().to_vec()).unwrap();
    let host_loss = host_logits.cross_entropy(&targets).unwrap();
    host_loss.backward().unwrap();

    let device_logits = DeviceValue::new(
        DeviceTensor::from_host(logits.clone()).to(device).unwrap(),
        true,
    );
    let loss = device_logits.cross_entropy(&targets).unwrap();
    loss.backward().unwrap();

    let host_value = host_loss.data()[0];
    let device_value = loss.data().to_host().unwrap().data()[0];
    assert!(
        (host_value - device_value).abs() <= tolerance,
        "cross entropy {device_value} vs {host_value}"
    );
    assert_close(
        device_logits.grad().unwrap().to_host().unwrap().data(),
        &host_logits.grad().unwrap().data(),
        tolerance,
        "cross entropy gradient",
    );
}

#[test]
fn device_cross_entropy_matches_host_autograd_on_cpu() {
    check_cross_entropy_parity(&Device::Cpu, 1e-5);
}

#[cfg(feature = "candle")]
#[test]
fn device_cross_entropy_matches_host_autograd_on_candle() {
    check_cross_entropy_parity(&Device::candle_cpu(), 1e-4);
}

fn check_rope_parity(device: &Device, tolerance: f32) {
    let x = sample(3, 4, 1.5);
    let positions = [0i64, 1, 3];
    let (head_dim, base) = (2usize, 10_000.0f32);

    let host_x = TensorValue::parameter(x.data().to_vec(), x.shape().to_vec()).unwrap();
    let host_out = host_x.rope(head_dim, &positions, base).unwrap();
    host_out.mean().unwrap().backward().unwrap();

    let device_x = DeviceValue::new(
        DeviceTensor::from_host(x.clone()).to(device).unwrap(),
        true,
    );
    let out = device_x.rope_rotate(head_dim, &positions, base).unwrap();
    out.mean().unwrap().backward().unwrap();

    assert_close(
        out.data().to_host().unwrap().data(),
        &host_out.data(),
        tolerance,
        "rope values",
    );
    assert_close(
        device_x.grad().unwrap().to_host().unwrap().data(),
        &host_x.grad().unwrap().data(),
        tolerance,
        "rope gradient",
    );
}

#[test]
fn device_rope_matches_host_autograd_on_cpu() {
    check_rope_parity(&Device::Cpu, 1e-5);
}

#[cfg(feature = "candle")]
#[test]
fn device_rope_matches_host_autograd_on_candle() {
    check_rope_parity(&Device::candle_cpu(), 1e-4);
}