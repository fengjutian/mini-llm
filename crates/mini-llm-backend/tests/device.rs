//! Device-resident tensor transfers and device-side matmul.

use mini_llm_backend::{Device, DeviceTensor};
use mini_llm_tensor::Tensor;

fn sample(rows: usize, cols: usize, scale: f32) -> Tensor {
    let data = (0..rows * cols)
        .map(|index| ((index % 7) as f32 - 3.0) * 0.25 * scale)
        .collect();
    Tensor::from_2d(rows, cols, data).unwrap()
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
