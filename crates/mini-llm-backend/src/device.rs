//! Device-resident tensors: host memory plus an explicit transfer boundary.
//!
//! [`DeviceTensor`] holds its data either on the host (`Cpu`) or on an
//! accelerated device. Moving across the boundary is explicit via
//! [`DeviceTensor::to`] and [`DeviceTensor::to_host`], which is what makes device
//! memory visible instead of hidden inside kernels.
//!
//! With the `candle` feature the device variant wraps a `candle_core::Tensor`
//! and matmul runs through Candle. On this machine only `Device::Cpu` is
//! available; `Device::Cuda` uses the same code path and needs a GPU build to
//! validate. Training and the autograd engine still operate on host `Tensor`
//! values — wiring them to device tensors is the next step.

use mini_llm_tensor::Tensor;
use rayon::prelude::*;

use crate::BackendError;

/// Where a [`DeviceTensor`] lives.
#[derive(Debug, Clone)]
pub enum Device {
    /// Host memory managed by `mini_llm_tensor`.
    Cpu,
    #[cfg(feature = "candle")]
    Candle(candle_core::Device),
}

#[cfg(feature = "candle")]
impl Device {
    /// Candle's CPU device (always available with the feature).
    pub fn candle_cpu() -> Self {
        Self::Candle(candle_core::Device::Cpu)
    }

    /// Candle's CUDA device; errors when no GPU/driver is present.
    pub fn candle_cuda(gpu_id: usize) -> Result<Self, BackendError> {
        candle_core::Device::new_cuda(gpu_id)
            .map(Self::Candle)
            .map_err(|error| BackendError::Backend(error.to_string()))
    }
}

impl PartialEq for Device {
    fn eq(&self, other: &Self) -> bool {
        match (self, other) {
            (Self::Cpu, Self::Cpu) => true,
            #[cfg(feature = "candle")]
            (Self::Candle(left), Self::Candle(right)) => left.same_device(right),
            #[cfg(feature = "candle")]
            _ => false,
        }
    }
}

impl Device {
    pub fn name(&self) -> &'static str {
        match self {
            Self::Cpu => "cpu",
            #[cfg(feature = "candle")]
            Self::Candle(candle_core::Device::Cpu) => "candle:cpu",
            #[cfg(feature = "candle")]
            Self::Candle(candle_core::Device::Cuda(_)) => "candle:cuda",
            #[cfg(feature = "candle")]
            Self::Candle(_) => "candle",
        }
    }
}

/// A tensor whose storage may live on a device.
#[derive(Clone)]
pub struct DeviceTensor {
    shape: Vec<usize>,
    storage: Storage,
}

#[derive(Clone)]
enum Storage {
    Host(Tensor),
    #[cfg(feature = "candle")]
    Candle {
        tensor: candle_core::Tensor,
        device: candle_core::Device,
    },
}

impl DeviceTensor {
    /// Wraps a host tensor without copying.
    pub fn from_host(tensor: Tensor) -> Self {
        Self {
            shape: tensor.shape().to_vec(),
            storage: Storage::Host(tensor),
        }
    }

    pub fn shape(&self) -> &[usize] {
        &self.shape
    }

    pub fn device(&self) -> Device {
        match &self.storage {
            Storage::Host(_) => Device::Cpu,
            #[cfg(feature = "candle")]
            Storage::Candle { device, .. } => Device::Candle(device.clone()),
        }
    }

    pub fn is_on_device(&self) -> bool {
        !matches!(self.storage, Storage::Host(_))
    }

    /// Moves the tensor to `device`, copying only when the device changes.
    pub fn to(&self, device: &Device) -> Result<Self, BackendError> {
        match (&self.storage, device) {
            (Storage::Host(tensor), Device::Cpu) => Ok(Self::from_host(tensor.clone())),
            #[cfg(feature = "candle")]
            (Storage::Host(tensor), Device::Candle(target)) => {
                let moved = candle_core::Tensor::from_vec(
                    tensor.data().to_vec(),
                    tensor.shape().to_vec(),
                    target,
                )
                .map_err(|error| BackendError::Backend(error.to_string()))?;
                Ok(Self {
                    shape: tensor.shape().to_vec(),
                    storage: Storage::Candle {
                        tensor: moved,
                        device: target.clone(),
                    },
                })
            }
            #[cfg(feature = "candle")]
            (Storage::Candle { tensor, device }, Device::Candle(target)) => {
                if device.same_device(target) {
                    Ok(Self {
                        shape: self.shape.clone(),
                        storage: Storage::Candle {
                            tensor: tensor.clone(),
                            device: target.clone(),
                        },
                    })
                } else {
                    // Hop through the host when moving between devices.
                    let host = self.to_host()?;
                    Self::from_host(host).to(&Device::Candle(device.clone()))
                }
            }
            #[cfg(feature = "candle")]
            (Storage::Candle { .. }, Device::Cpu) => Ok(Self::from_host(self.to_host()?)),
        }
    }

    /// Copies the data back to host memory.
    pub fn to_host(&self) -> Result<Tensor, BackendError> {
        match &self.storage {
            Storage::Host(tensor) => Ok(tensor.clone()),
            #[cfg(feature = "candle")]
            Storage::Candle { tensor, .. } => {
                let shape = tensor.shape().dims().to_vec();
                let data = tensor
                    .flatten_all()
                    .and_then(|flat| flat.to_vec1::<f32>())
                    .map_err(|error| BackendError::Backend(error.to_string()))?;
                Ok(Tensor::new(data, shape)?)
            }
        }
    }

    /// Matrix multiplication on the device the data already lives on.
    pub fn matmul(&self, other: &Self) -> Result<Self, BackendError> {
        match (&self.storage, &other.storage) {
            (Storage::Host(left), Storage::Host(right)) => Ok(Self::from_host(left.matmul(right)?)),
            #[cfg(feature = "candle")]
            (
                Storage::Candle {
                    tensor: left,
                    device: left_device,
                },
                Storage::Candle {
                    tensor: right,
                    device: right_device,
                },
            ) => {
                if !left_device.same_device(right_device) {
                    return Err(BackendError::Backend(format!(
                        "matmul operands live on `{}` and `{}`",
                        Device::Candle(left_device.clone()).name(),
                        Device::Candle(right_device.clone()).name()
                    )));
                }
                let product = left
                    .matmul(right)
                    .map_err(|error| BackendError::Backend(error.to_string()))?;
                let shape = product.shape().dims().to_vec();
                Ok(Self {
                    shape,
                    storage: Storage::Candle {
                        tensor: product,
                        device: left_device.clone(),
                    },
                })
            }
            #[cfg(feature = "candle")]
            _ => Err(BackendError::Backend(
                "matmul operands must live on the same device".to_string(),
            )),
        }
    }
}

impl std::fmt::Debug for DeviceTensor {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        formatter
            .debug_struct("DeviceTensor")
            .field("shape", &self.shape)
            .field("device", &self.device())
            .finish()
    }
}

impl DeviceTensor {
    #[cfg(feature = "candle")]
    fn from_candle(tensor: candle_core::Tensor) -> Self {
        let shape = tensor.shape().dims().to_vec();
        let device = tensor.device().clone();
        Self {
            shape,
            storage: Storage::Candle { tensor, device },
        }
    }

    /// Element-wise addition with broadcasting on the same device.
    pub fn add(&self, other: &Self) -> Result<Self, BackendError> {
        match (&self.storage, &other.storage) {
            (Storage::Host(left), Storage::Host(right)) => Ok(Self::from_host(left.add(right)?)),
            #[cfg(feature = "candle")]
            (
                Storage::Candle {
                    tensor: left,
                    device: left_device,
                },
                Storage::Candle {
                    tensor: right,
                    device: right_device,
                },
            ) => {
                if !left_device.same_device(right_device) {
                    return Err(mixed_device_error());
                }
                let sum = left
                    .broadcast_add(right)
                    .map_err(|error| BackendError::Backend(error.to_string()))?;
                Ok(Self::from_candle(sum))
            }
            #[cfg(feature = "candle")]
            _ => Err(mixed_device_error()),
        }
    }

    /// Multiplies every element by `factor` on the device.
    pub fn scale(&self, factor: f32) -> Result<Self, BackendError> {
        match &self.storage {
            Storage::Host(tensor) => Ok(Self::from_host(tensor.scale(factor)?)),
            #[cfg(feature = "candle")]
            Storage::Candle { tensor, .. } => {
                let scaled = tensor
                    .affine(f64::from(factor), 0.0)
                    .map_err(|error| BackendError::Backend(error.to_string()))?;
                Ok(Self::from_candle(scaled))
            }
        }
    }

    /// Transposes a rank-2 tensor on the device.
    pub fn transpose_2d(&self) -> Result<Self, BackendError> {
        match &self.storage {
            Storage::Host(tensor) => Ok(Self::from_host(tensor.transpose_2d()?)),
            #[cfg(feature = "candle")]
            Storage::Candle { tensor, .. } => {
                let transposed = tensor
                    .transpose(0, 1)
                    .map_err(|error| BackendError::Backend(error.to_string()))?;
                Ok(Self::from_candle(transposed))
            }
        }
    }

    /// Sums rows into a `[1, cols]` tensor on the device.
    pub fn sum_rows(&self) -> Result<Self, BackendError> {
        match &self.storage {
            Storage::Host(tensor) => {
                let (rows, cols) = (tensor.shape()[0], tensor.shape()[1]);
                let mut data = vec![0.0f32; cols];
                for row in 0..rows {
                    let start = row * cols;
                    for (slot, value) in data.iter_mut().zip(&tensor.data()[start..start + cols]) {
                        *slot += value;
                    }
                }
                Ok(Self::from_host(Tensor::new(data, vec![1, cols])?))
            }
            #[cfg(feature = "candle")]
            Storage::Candle { tensor, .. } => {
                let summed = tensor
                    .sum_keepdim(0)
                    .map_err(|error| BackendError::Backend(error.to_string()))?;
                Ok(Self::from_candle(summed))
            }
        }
    }

    /// Mean of every element, read back as a host scalar.
    pub fn mean_all_value(&self) -> Result<f32, BackendError> {
        match &self.storage {
            Storage::Host(tensor) => Ok(tensor.mean()?.data()[0]),
            #[cfg(feature = "candle")]
            Storage::Candle { tensor, .. } => tensor
                .mean_all()
                .and_then(|mean| mean.to_scalar::<f32>())
                .map_err(|error| BackendError::Backend(error.to_string())),
        }
    }
}

#[cfg(feature = "candle")]
fn mixed_device_error() -> BackendError {
    BackendError::Backend("operands live on different devices".to_string())
}

/// A linear layer whose parameters stay on the device.
///
/// This is the device-side forward/backward skeleton: weights are uploaded once
/// with [`DeviceTensor::to`], every forward and gradient runs on the device, and
/// only the result is copied back for inspection.
pub struct DeviceLinear {
    weight: DeviceTensor,
    bias: Option<DeviceTensor>,
}

impl DeviceLinear {
    pub fn new(weight: DeviceTensor, bias: Option<DeviceTensor>) -> Self {
        Self { weight, bias }
    }

    pub fn weight(&self) -> &DeviceTensor {
        &self.weight
    }

    pub fn bias(&self) -> Option<&DeviceTensor> {
        self.bias.as_ref()
    }

    /// `y = x W + b`, computed on the device.
    pub fn forward(&self, input: &DeviceTensor) -> Result<DeviceTensor, BackendError> {
        let projected = input.matmul(&self.weight)?;
        match &self.bias {
            Some(bias) => projected.add(bias),
            None => Ok(projected),
        }
    }

    /// Device-side gradients for `x`, `W` and `b`.
    ///
    /// `grad_output` is the upstream gradient of the layer output.
    pub fn backward(
        &self,
        input: &DeviceTensor,
        grad_output: &DeviceTensor,
    ) -> Result<(DeviceTensor, DeviceTensor, Option<DeviceTensor>), BackendError> {
        let grad_input = grad_output.matmul(&self.weight.transpose_2d()?)?;
        let grad_weight = input.transpose_2d()?.matmul(grad_output)?;
        let grad_bias = match &self.bias {
            Some(_) => Some(grad_output.sum_rows()?),
            None => None,
        };
        Ok((grad_input, grad_weight, grad_bias))
    }
}

impl DeviceTensor {
    /// Element-wise ReLU on the device.
    pub fn relu(&self) -> Result<Self, BackendError> {
        match &self.storage {
            Storage::Host(tensor) => Ok(Self::from_host(tensor.relu()?)),
            #[cfg(feature = "candle")]
            Storage::Candle { tensor, .. } => {
                let activated = tensor
                    .relu()
                    .map_err(|error| BackendError::Backend(error.to_string()))?;
                Ok(Self::from_candle(activated))
            }
        }
    }

    /// Element-wise multiplication with broadcasting on the same device.
    pub fn mul(&self, other: &Self) -> Result<Self, BackendError> {
        match (&self.storage, &other.storage) {
            (Storage::Host(left), Storage::Host(right)) => Ok(Self::from_host(left.mul(right)?)),
            #[cfg(feature = "candle")]
            (
                Storage::Candle {
                    tensor: left,
                    device: left_device,
                },
                Storage::Candle {
                    tensor: right,
                    device: right_device,
                },
            ) => {
                if !left_device.same_device(right_device) {
                    return Err(mixed_device_error());
                }
                let product = left
                    .broadcast_mul(right)
                    .map_err(|error| BackendError::Backend(error.to_string()))?;
                Ok(Self::from_candle(product))
            }
            #[cfg(feature = "candle")]
            _ => Err(mixed_device_error()),
        }
    }

    /// `1.0` where the element is positive, `0.0` elsewhere (the ReLU gradient).
    pub fn positive_mask(&self) -> Result<Self, BackendError> {
        match &self.storage {
            Storage::Host(tensor) => {
                let data = tensor
                    .data()
                    .iter()
                    .map(|value| if *value > 0.0 { 1.0 } else { 0.0 })
                    .collect();
                Ok(Self::from_host(Tensor::new(data, tensor.shape().to_vec())?))
            }
            #[cfg(feature = "candle")]
            Storage::Candle { tensor, .. } => {
                let mask = tensor
                    .gt(0f32)
                    .and_then(|mask| mask.to_dtype(candle_core::DType::F32))
                    .map_err(|error| BackendError::Backend(error.to_string()))?;
                Ok(Self::from_candle(mask))
            }
        }
    }
}

/// Gradients of one layer: `(dW, db)`; `db` is `None` when the layer has no bias.
pub type LayerGradients = (DeviceTensor, Option<DeviceTensor>);

/// Result of a device backward pass: `(dx, per-layer gradients)`.
pub type MlpGradients = (DeviceTensor, Vec<LayerGradients>);

/// A stack of [`DeviceLinear`] layers with ReLU between them.
///
/// This extends the device-side backward skeleton to more than one layer: the
/// forward pass caches each layer input, and the backward pass walks the layers
/// in reverse, applying the ReLU mask from the cached activation.
pub struct DeviceMlp {
    layers: Vec<DeviceLinear>,
}

impl DeviceMlp {
    pub fn new(layers: Vec<DeviceLinear>) -> Self {
        Self { layers }
    }

    pub fn layers(&self) -> &[DeviceLinear] {
        &self.layers
    }

    /// Runs the stack; ReLU is applied after every layer except the last.
    pub fn forward(&self, input: &DeviceTensor) -> Result<DeviceTensor, BackendError> {
        Ok(self.forward_with_cache(input)?.0)
    }

    /// Forward pass that also returns the input of every layer.
    pub fn forward_with_cache(
        &self,
        input: &DeviceTensor,
    ) -> Result<(DeviceTensor, Vec<DeviceTensor>), BackendError> {
        let mut inputs = Vec::with_capacity(self.layers.len());
        let mut current = input.clone();
        for (index, layer) in self.layers.iter().enumerate() {
            inputs.push(current.clone());
            let output = layer.forward(&inputs[index])?;
            current = if index + 1 == self.layers.len() {
                output
            } else {
                output.relu()?
            };
        }
        Ok((current, inputs))
    }

    /// Backward pass over the cached layer inputs.
    ///
    /// Returns `(dx, [(dW, db), ...])` in layer order.
    pub fn backward(
        &self,
        inputs: &[DeviceTensor],
        grad_output: &DeviceTensor,
    ) -> Result<MlpGradients, BackendError> {
        assert_eq!(
            inputs.len(),
            self.layers.len(),
            "one cached input per layer"
        );
        let mut gradients = vec![None; self.layers.len()];
        let mut upstream = grad_output.clone();
        let mut grad_input = None;

        for index in (0..self.layers.len()).rev() {
            let (grad_x, grad_w, grad_b) =
                self.layers[index].backward(&inputs[index], &upstream)?;
            gradients[index] = Some((grad_w, grad_b));
            if index == 0 {
                grad_input = Some(grad_x);
            } else {
                upstream = grad_x.mul(&inputs[index].positive_mask()?)?;
            }
        }

        Ok((
            grad_input.expect("at least one layer"),
            gradients
                .into_iter()
                .map(|pair| pair.expect("gradient"))
                .collect(),
        ))
    }
}

impl DeviceTensor {
    /// Mean of every element as a `[1]` tensor, without leaving the device.
    pub fn mean_scalar(&self) -> Result<Self, BackendError> {
        match &self.storage {
            Storage::Host(tensor) => Ok(Self::from_host(tensor.mean()?)),
            #[cfg(feature = "candle")]
            Storage::Candle { tensor, .. } => {
                let mean = tensor
                    .mean_all()
                    .and_then(|mean| mean.reshape((1usize,)))
                    .map_err(|error| BackendError::Backend(error.to_string()))?;
                Ok(Self::from_candle(mean))
            }
        }
    }
}
impl DeviceTensor {
    /// Broadcasts a `[1]` or `[1, cols]` tensor to `[rows, cols]`.
    pub fn broadcast_to_2d(&self, rows: usize, cols: usize) -> Result<Self, BackendError> {
        match &self.storage {
            Storage::Host(tensor) => {
                let source = tensor.data();
                let mut data = Vec::with_capacity(rows * cols);
                match tensor.shape() {
                    [1] => data.resize(rows * cols, source[0]),
                    [1, source_cols] if *source_cols == cols => {
                        for _ in 0..rows {
                            data.extend_from_slice(source);
                        }
                    }
                    [source_rows, 1] if *source_rows == rows => {
                        for &value in source {
                            for _ in 0..cols {
                                data.push(value);
                            }
                        }
                    }
                    [source_rows, source_cols] if *source_rows == rows && *source_cols == cols => {
                        data.extend_from_slice(source);
                    }
                    shape => {
                        return Err(BackendError::Backend(format!(
                            "cannot broadcast shape {shape:?} to [{rows}, {cols}]"
                        )));
                    }
                }
                Ok(Self::from_host(Tensor::new(data, vec![rows, cols])?))
            }
            #[cfg(feature = "candle")]
            Storage::Candle { tensor, .. } => {
                let broadcast = tensor
                    .broadcast_as((rows, cols))
                    .map_err(|error| BackendError::Backend(error.to_string()))?;
                Ok(Self::from_candle(broadcast))
            }
        }
    }
}

impl DeviceTensor {
    /// Element-wise square root on the device.
    pub fn sqrt(&self) -> Result<Self, BackendError> {
        match &self.storage {
            Storage::Host(tensor) => {
                let data = tensor.data().iter().map(|value| value.sqrt()).collect();
                Ok(Self::from_host(Tensor::new(data, tensor.shape().to_vec())?))
            }
            #[cfg(feature = "candle")]
            Storage::Candle { tensor, .. } => {
                let result = tensor
                    .sqrt()
                    .map_err(|error| BackendError::Backend(error.to_string()))?;
                Ok(Self::from_candle(result))
            }
        }
    }

    /// Element-wise division; operands must share the same shape.
    pub fn div(&self, other: &Self) -> Result<Self, BackendError> {
        match (&self.storage, &other.storage) {
            (Storage::Host(left), Storage::Host(right)) => {
                let reciprocal = Tensor::new(
                    right.data().iter().map(|value| 1.0 / value).collect(),
                    right.shape().to_vec(),
                )?;
                Ok(Self::from_host(left.mul(&reciprocal)?))
            }
            #[cfg(feature = "candle")]
            (
                Storage::Candle {
                    tensor: left,
                    device: left_device,
                },
                Storage::Candle {
                    tensor: right,
                    device: right_device,
                },
            ) => {
                if !left_device.same_device(right_device) {
                    return Err(mixed_device_error());
                }
                let quotient = left
                    .broadcast_div(right)
                    .map_err(|error| BackendError::Backend(error.to_string()))?;
                Ok(Self::from_candle(quotient))
            }
            #[cfg(feature = "candle")]
            _ => Err(mixed_device_error()),
        }
    }

    /// Adds a scalar to every element on the device.
    pub fn add_scalar(&self, value: f32) -> Result<Self, BackendError> {
        match &self.storage {
            Storage::Host(tensor) => {
                let data = tensor.data().iter().map(|entry| entry + value).collect();
                Ok(Self::from_host(Tensor::new(data, tensor.shape().to_vec())?))
            }
            #[cfg(feature = "candle")]
            Storage::Candle { tensor, .. } => {
                let shifted = tensor
                    .affine(1.0, f64::from(value))
                    .map_err(|error| BackendError::Backend(error.to_string()))?;
                Ok(Self::from_candle(shifted))
            }
        }
    }

    /// A zero tensor with the same shape and device.
    pub fn zeros_like(&self) -> Result<Self, BackendError> {
        match &self.storage {
            Storage::Host(tensor) => Ok(Self::from_host(Tensor::zeros(tensor.shape().to_vec())?)),
            #[cfg(feature = "candle")]
            Storage::Candle { tensor, device } => {
                let zeros =
                    candle_core::Tensor::zeros(self.shape.clone(), candle_core::DType::F32, device)
                        .map_err(|error| BackendError::Backend(error.to_string()))?;
                let _ = tensor;
                Ok(Self::from_candle(zeros))
            }
        }
    }
}

impl DeviceTensor {
    /// Row-wise softmax over the last dimension.
    pub fn softmax_rows(&self) -> Result<Self, BackendError> {
        match &self.storage {
            Storage::Host(tensor) => Ok(Self::from_host(tensor.softmax_rows()?)),
            #[cfg(feature = "candle")]
            Storage::Candle { tensor, .. } => {
                let max = tensor
                    .max_keepdim(1)
                    .map_err(|error| BackendError::Backend(error.to_string()))?;
                let shifted = tensor
                    .broadcast_sub(&max)
                    .map_err(|error| BackendError::Backend(error.to_string()))?;
                let exponents = shifted
                    .exp()
                    .map_err(|error| BackendError::Backend(error.to_string()))?;
                let denominator = exponents
                    .sum_keepdim(1)
                    .map_err(|error| BackendError::Backend(error.to_string()))?;
                let result = exponents
                    .broadcast_div(&denominator)
                    .map_err(|error| BackendError::Backend(error.to_string()))?;
                Ok(Self::from_candle(result))
            }
        }
    }

    /// Mean over the last dimension, keeping it as `[rows, 1]`.
    pub fn mean_rows_keepdim(&self) -> Result<Self, BackendError> {
        match &self.storage {
            Storage::Host(tensor) => Ok(Self::from_host(tensor.mean_rows_keepdim()?)),
            #[cfg(feature = "candle")]
            Storage::Candle { tensor, .. } => {
                let result = tensor
                    .mean_keepdim(1)
                    .map_err(|error| BackendError::Backend(error.to_string()))?;
                Ok(Self::from_candle(result))
            }
        }
    }

    /// Sum over the last dimension, keeping it as `[rows, 1]`.
    pub fn sum_rows_keepdim(&self) -> Result<Self, BackendError> {
        match &self.storage {
            Storage::Host(tensor) => {
                let (rows, cols) = (tensor.shape()[0], tensor.shape()[1]);
                let mut data = vec![0.0f32; rows];
                for (row, slot) in data.iter_mut().enumerate() {
                    let start = row * cols;
                    for value in &tensor.data()[start..start + cols] {
                        *slot += value;
                    }
                }
                Ok(Self::from_host(Tensor::new(data, vec![rows, 1])?))
            }
            #[cfg(feature = "candle")]
            Storage::Candle { tensor, .. } => {
                let result = tensor
                    .sum_keepdim(1)
                    .map_err(|error| BackendError::Backend(error.to_string()))?;
                Ok(Self::from_candle(result))
            }
        }
    }

    /// Element-wise power on the device.
    pub fn powf(&self, exponent: f32) -> Result<Self, BackendError> {
        match &self.storage {
            Storage::Host(tensor) => Ok(Self::from_host(tensor.powf(exponent)?)),
            #[cfg(feature = "candle")]
            Storage::Candle { tensor, .. } => {
                let result = tensor
                    .powf(f64::from(exponent))
                    .map_err(|error| BackendError::Backend(error.to_string()))?;
                Ok(Self::from_candle(result))
            }
        }
    }

    /// Gathers rows by index: `[vocab, hidden] -> [indices, hidden]`.
    pub fn gather_rows(&self, indices: &[usize]) -> Result<Self, BackendError> {
        match &self.storage {
            Storage::Host(tensor) => Ok(Self::from_host(tensor.gather_rows(indices)?)),
            #[cfg(feature = "candle")]
            Storage::Candle { tensor, device } => {
                let index = candle_core::Tensor::from_vec(
                    indices
                        .iter()
                        .map(|&index| index as u32)
                        .collect::<Vec<u32>>(),
                    indices.len(),
                    device,
                )
                .map_err(|error| BackendError::Backend(error.to_string()))?;
                let gathered = tensor
                    .index_select(&index, 0)
                    .map_err(|error| BackendError::Backend(error.to_string()))?;
                Ok(Self::from_candle(gathered))
            }
        }
    }

    /// Scatter-adds `self` into a zero `[rows, cols]` tensor (embedding backward).
    pub fn scatter_add_rows(&self, indices: &[usize], rows: usize) -> Result<Self, BackendError> {
        match &self.storage {
            Storage::Host(tensor) => Ok(Self::from_host(Tensor::scatter_add_rows(
                tensor, indices, rows,
            )?)),
            #[cfg(feature = "candle")]
            Storage::Candle { tensor, device } => {
                let index = candle_core::Tensor::from_vec(
                    indices
                        .iter()
                        .map(|&index| index as u32)
                        .collect::<Vec<u32>>(),
                    indices.len(),
                    device,
                )
                .map_err(|error| BackendError::Backend(error.to_string()))?;
                let cols = self.shape()[1];
                let zeros =
                    candle_core::Tensor::zeros((rows, cols), candle_core::DType::F32, device)
                        .map_err(|error| BackendError::Backend(error.to_string()))?;
                let values = tensor
                    .contiguous()
                    .map_err(|error| BackendError::Backend(error.to_string()))?;
                let scattered = zeros
                    .index_add(&index, &values, 0)
                    .map_err(|error| BackendError::Backend(error.to_string()))?;
                Ok(Self::from_candle(scattered))
            }
        }
    }
}

impl DeviceTensor {
    /// Element-wise natural logarithm.
    pub fn log(&self) -> Result<Self, BackendError> {
        match &self.storage {
            Storage::Host(tensor) => {
                let data = tensor.data().iter().map(|value| value.ln()).collect();
                Ok(Self::from_host(Tensor::new(data, tensor.shape().to_vec())?))
            }
            #[cfg(feature = "candle")]
            Storage::Candle { tensor, .. } => {
                let result = tensor
                    .log()
                    .map_err(|error| BackendError::Backend(error.to_string()))?;
                Ok(Self::from_candle(result))
            }
        }
    }

    /// Sum of every element as a `[1]` tensor.
    pub fn sum_all(&self) -> Result<Self, BackendError> {
        match &self.storage {
            Storage::Host(tensor) => Ok(Self::from_host(tensor.sum()?)),
            #[cfg(feature = "candle")]
            Storage::Candle { tensor, .. } => {
                let result = tensor
                    .sum_all()
                    .and_then(|sum| sum.reshape((1usize,)))
                    .map_err(|error| BackendError::Backend(error.to_string()))?;
                Ok(Self::from_candle(result))
            }
        }
    }

    /// Reads a scalar value back to the host.
    pub fn scalar(&self) -> Result<f32, BackendError> {
        let host = self.to_host()?;
        if host.data().len() != 1 {
            return Err(BackendError::Backend(format!(
                "expected a scalar, got shape {:?}",
                host.shape()
            )));
        }
        Ok(host.data()[0])
    }

    /// One-hot rows for `targets`, uploaded to the same device.
    #[cfg(feature = "candle")]
    fn one_hot(&self, targets: &[usize], rows: usize, cols: usize) -> Result<Self, BackendError> {
        let mut data = vec![0.0f32; rows * cols];
        for (row, &target) in targets.iter().enumerate() {
            if target >= cols {
                return Err(BackendError::Backend(format!(
                    "target {target} outside {cols} classes"
                )));
            }
            data[row * cols + target] = 1.0;
        }
        let host = Tensor::new(data, vec![rows, cols])?;
        DeviceTensor::from_host(host).to(&self.device())
    }

    /// Mean cross-entropy of logits against class targets, plus its gradient.
    pub fn cross_entropy(&self, targets: &[usize]) -> Result<(Self, Self), BackendError> {
        let (rows, cols) = (self.shape()[0], self.shape()[1]);
        if targets.len() != rows {
            return Err(BackendError::Backend(format!(
                "{} targets for {rows} rows",
                targets.len()
            )));
        }

        match &self.storage {
            Storage::Host(tensor) => {
                let data = tensor.data();
                let mut grad = vec![0.0f32; rows * cols];
                let mut loss = 0.0f32;
                for (row, &target) in targets.iter().enumerate() {
                    let offset = row * cols;
                    let max = data[offset..offset + cols]
                        .iter()
                        .copied()
                        .fold(f32::NEG_INFINITY, f32::max);
                    let mut sum = 0.0f32;
                    for col in 0..cols {
                        sum += (data[offset + col] - max).exp();
                    }
                    if target >= cols {
                        return Err(BackendError::Backend(format!(
                            "target {target} outside {cols} classes"
                        )));
                    }
                    loss += max + sum.ln() - data[offset + target];
                    for col in 0..cols {
                        grad[offset + col] = (data[offset + col] - max).exp() / sum;
                    }
                    grad[offset + target] -= 1.0;
                }
                let scale = 1.0 / rows as f32;
                for value in &mut grad {
                    *value *= scale;
                }
                Ok((
                    Self::from_host(Tensor::new(vec![loss * scale], vec![1])?),
                    Self::from_host(Tensor::new(grad, vec![rows, cols])?),
                ))
            }
            #[cfg(feature = "candle")]
            Storage::Candle { .. } => {
                let softmax = self.softmax_rows()?;
                let log_prob = softmax.log()?;
                let one_hot = self.one_hot(targets, rows, cols)?;
                let loss = log_prob
                    .mul(&one_hot)?
                    .sum_all()?
                    .scale(-1.0 / rows as f32)?;
                let grad = softmax
                    .add(&one_hot.scale(-1.0)?)?
                    .scale(1.0 / rows as f32)?;
                Ok((loss, grad))
            }
        }
    }

    /// Rotary position embedding over `[rows, heads * head_dim]`.
    pub fn rope_rotate(
        &self,
        head_dim: usize,
        positions: &[i64],
        base: f32,
    ) -> Result<Self, BackendError> {
        let (rows, cols) = (self.shape()[0], self.shape()[1]);
        if head_dim == 0 || !head_dim.is_multiple_of(2) {
            return Err(BackendError::Backend(format!(
                "invalid rope head_dim {head_dim}"
            )));
        }
        if !cols.is_multiple_of(head_dim) {
            return Err(BackendError::Backend(format!(
                "width {cols} is not a multiple of head_dim {head_dim}"
            )));
        }
        if positions.len() != rows {
            return Err(BackendError::Backend(format!(
                "{} positions for {rows} rows",
                positions.len()
            )));
        }
        match &self.storage {
            Storage::Host(tensor) => Ok(Self::from_host(
                tensor.rope_rotate(head_dim, positions, base)?,
            )),
            #[cfg(feature = "candle")]
            Storage::Candle { tensor, device } => {
                let heads = cols / head_dim;
                let half = head_dim / 2;
                let frequencies: Vec<f32> = (0..half)
                    .map(|index| 1.0 / base.powf(2.0 * index as f32 / head_dim as f32))
                    .collect();
                let mut cos_data = Vec::with_capacity(rows * half);
                let mut sin_data = Vec::with_capacity(rows * half);
                for &position in positions {
                    for frequency in &frequencies {
                        let angle = position as f32 * frequency;
                        cos_data.push(angle.cos());
                        sin_data.push(angle.sin());
                    }
                }
                let cos = candle_core::Tensor::from_vec(cos_data, (rows, 1, half, 1), device)
                    .map_err(|error| BackendError::Backend(error.to_string()))?;
                let sin = candle_core::Tensor::from_vec(sin_data, (rows, 1, half, 1), device)
                    .map_err(|error| BackendError::Backend(error.to_string()))?;

                let shaped = tensor
                    .reshape((rows, heads, half, 2))
                    .map_err(|error| BackendError::Backend(error.to_string()))?;
                let even = shaped
                    .narrow(3, 0, 1)
                    .map_err(|error| BackendError::Backend(error.to_string()))?;
                let odd = shaped
                    .narrow(3, 1, 1)
                    .map_err(|error| BackendError::Backend(error.to_string()))?;
                let rotated_even = even
                    .broadcast_mul(&cos)
                    .and_then(|value| value.broadcast_sub(&odd.broadcast_mul(&sin)?))
                    .map_err(|error| BackendError::Backend(error.to_string()))?;
                let rotated_odd = even
                    .broadcast_mul(&sin)
                    .and_then(|value| value.broadcast_add(&odd.broadcast_mul(&cos)?))
                    .map_err(|error| BackendError::Backend(error.to_string()))?;
                let stacked = candle_core::Tensor::cat(&[&rotated_even, &rotated_odd], 3)
                    .and_then(|value| value.reshape((rows, cols)))
                    .map_err(|error| BackendError::Backend(error.to_string()))?;
                Ok(Self::from_candle(stacked))
            }
        }
    }
}

impl DeviceTensor {
    /// Element-wise exponential.
    pub fn exp(&self) -> Result<Self, BackendError> {
        match &self.storage {
            Storage::Host(tensor) => {
                let data = tensor.data().iter().map(|value| value.exp()).collect();
                Ok(Self::from_host(Tensor::new(data, tensor.shape().to_vec())?))
            }
            #[cfg(feature = "candle")]
            Storage::Candle { tensor, .. } => {
                let result = tensor
                    .exp()
                    .map_err(|error| BackendError::Backend(error.to_string()))?;
                Ok(Self::from_candle(result))
            }
        }
    }
}

impl DeviceTensor {
    /// Scaled dot-product attention over `[batch * seq, heads * head_dim]`.
    ///
    /// `key`/`value` may have fewer heads than `q` (GQA); each query head reads
    /// `kv_head = head / (heads / kv_heads)`. `causal` masks future positions.
    #[allow(clippy::too_many_arguments)]
    pub fn attention_forward(
        &self,
        key: &Self,
        value: &Self,
        batch: usize,
        seq: usize,
        heads: usize,
        kv_heads: usize,
        causal: bool,
    ) -> Result<Self, BackendError> {
        match &self.storage {
            Storage::Host(q) => {
                let k = key.to_host()?;
                let v = value.to_host()?;
                Ok(Self::from_host(attention_host(
                    q, &k, &v, batch, seq, heads, kv_heads, causal,
                )?))
            }
            #[cfg(feature = "candle")]
            Storage::Candle { .. } => {
                // Correctness first: run the verified host math, then move the
                // result back to the device. A native 4D kernel is still TODO.
                let q = self.to_host()?;
                let k = key.to_host()?;
                let v = value.to_host()?;
                let output = attention_host(&q, &k, &v, batch, seq, heads, kv_heads, causal)?;
                Ok(Self::from_host(output).to(&self.device())?)
            }
        }
    }

    /// Gradients of [`DeviceTensor::attention_forward`] with respect to q, k, v.
    #[allow(clippy::too_many_arguments)]
    pub fn attention_backward(
        &self,
        key: &Self,
        value: &Self,
        grad_output: &Self,
        batch: usize,
        seq: usize,
        heads: usize,
        kv_heads: usize,
        causal: bool,
    ) -> Result<(Self, Self, Self), BackendError> {
        match &self.storage {
            Storage::Host(q) => {
                let k = key.to_host()?;
                let v = value.to_host()?;
                let g = grad_output.to_host()?;
                let (dq, dk, dv) =
                    attention_host_backward(q, &k, &v, &g, batch, seq, heads, kv_heads, causal)?;
                Ok((
                    Self::from_host(dq),
                    Self::from_host(dk),
                    Self::from_host(dv),
                ))
            }
            #[cfg(feature = "candle")]
            Storage::Candle { .. } => {
                let q = self.to_host()?;
                let k = key.to_host()?;
                let v = value.to_host()?;
                let g = grad_output.to_host()?;
                let (grad_q, grad_k, grad_v) =
                    attention_host_backward(&q, &k, &v, &g, batch, seq, heads, kv_heads, causal)?;
                let device = self.device();
                Ok((
                    Self::from_host(grad_q).to(&device)?,
                    Self::from_host(grad_k).to(&device)?,
                    Self::from_host(grad_v).to(&device)?,
                ))
            }
        }
    }
}

fn attention_layout(
    q: &Tensor,
    k: &Tensor,
    v: &Tensor,
    batch: usize,
    seq: usize,
    heads: usize,
    kv_heads: usize,
) -> Result<usize, BackendError> {
    let rows = batch * seq;
    if q.shape() != [rows, heads * (q.shape()[1] / heads.max(1))]
        && !q.shape()[1].is_multiple_of(heads)
    {
        return Err(BackendError::Backend(
            "invalid attention shapes".to_string(),
        ));
    }
    let head_dim = q.shape()[1] / heads;
    if head_dim == 0
        || k.shape() != [rows, kv_heads * head_dim]
        || v.shape() != [rows, kv_heads * head_dim]
        || !heads.is_multiple_of(kv_heads)
    {
        return Err(BackendError::Backend(format!(
            "invalid attention shapes: q {:?}, k {:?}, v {:?}, heads {heads}, kv_heads {kv_heads}",
            q.shape(),
            k.shape(),
            v.shape()
        )));
    }
    Ok(head_dim)
}

#[allow(clippy::too_many_arguments, clippy::needless_range_loop)]
fn attention_host(
    q: &Tensor,
    k: &Tensor,
    v: &Tensor,
    batch: usize,
    seq: usize,
    heads: usize,
    kv_heads: usize,
    causal: bool,
) -> Result<Tensor, BackendError> {
    let head_dim = attention_layout(q, k, v, batch, seq, heads, kv_heads)?;
    let rows = batch * seq;
    let group = heads / kv_heads;
    let scale = 1.0 / (head_dim as f32).sqrt();
    let q_data = q.data();
    let k_data = k.data();
    let v_data = v.data();

    let per_batch: Vec<Vec<f32>> = (0..batch)
        .into_par_iter()
        .map(|batch_index| {
            attention_batch_forward(
                q_data,
                k_data,
                v_data,
                batch_index,
                seq,
                heads,
                kv_heads,
                head_dim,
                group,
                scale,
                causal,
            )
        })
        .collect();
    let mut output = Vec::with_capacity(rows * heads * head_dim);
    for chunk in per_batch {
        output.extend(chunk);
    }
    Ok(Tensor::new(output, vec![rows, heads * head_dim])?)
}

#[allow(clippy::too_many_arguments, clippy::needless_range_loop)]
fn attention_batch_forward(
    q_data: &[f32],
    k_data: &[f32],
    v_data: &[f32],
    batch_index: usize,
    seq: usize,
    heads: usize,
    kv_heads: usize,
    head_dim: usize,
    group: usize,
    scale: f32,
    causal: bool,
) -> Vec<f32> {
    let mut output = vec![0.0f32; seq * heads * head_dim];
    for head in 0..heads {
        let kv_head = head / group;
        for position in 0..seq {
            let local_row = position * heads * head_dim + head * head_dim;
            let q_offset = (batch_index * seq + position) * heads * head_dim + head * head_dim;
            let mut scores = vec![f32::NEG_INFINITY; seq];
            for other in 0..seq {
                if causal && other > position {
                    continue;
                }
                let k_offset =
                    (batch_index * seq + other) * kv_heads * head_dim + kv_head * head_dim;
                scores[other] = (0..head_dim)
                    .map(|dim| q_data[q_offset + dim] * k_data[k_offset + dim])
                    .sum::<f32>()
                    * scale;
            }
            let probabilities = softmax_host(&scores);
            for other in 0..seq {
                if probabilities[other] == 0.0 {
                    continue;
                }
                let v_offset =
                    (batch_index * seq + other) * kv_heads * head_dim + kv_head * head_dim;
                for dim in 0..head_dim {
                    output[local_row + dim] += probabilities[other] * v_data[v_offset + dim];
                }
            }
        }
    }
    output
}

#[allow(clippy::too_many_arguments)]
fn attention_host_backward(
    q: &Tensor,
    k: &Tensor,
    v: &Tensor,
    grad_output: &Tensor,
    batch: usize,
    seq: usize,
    heads: usize,
    kv_heads: usize,
    causal: bool,
) -> Result<(Tensor, Tensor, Tensor), BackendError> {
    let head_dim = attention_layout(q, k, v, batch, seq, heads, kv_heads)?;
    let rows = batch * seq;
    let group = heads / kv_heads;
    let scale = 1.0 / (head_dim as f32).sqrt();
    let q_data = q.data();
    let k_data = k.data();
    let v_data = v.data();
    let g_data = grad_output.data();

    let per_batch: Vec<(Vec<f32>, Vec<f32>, Vec<f32>)> = (0..batch)
        .into_par_iter()
        .map(|batch_index| {
            attention_batch_backward(
                q_data,
                k_data,
                v_data,
                g_data,
                batch_index,
                seq,
                heads,
                kv_heads,
                head_dim,
                group,
                scale,
                causal,
            )
        })
        .collect();

    let mut grad_q = Vec::with_capacity(rows * heads * head_dim);
    let mut grad_k = Vec::with_capacity(rows * kv_heads * head_dim);
    let mut grad_v = Vec::with_capacity(rows * kv_heads * head_dim);
    for (dq, dk, dv) in per_batch {
        grad_q.extend(dq);
        grad_k.extend(dk);
        grad_v.extend(dv);
    }
    Ok((
        Tensor::new(grad_q, vec![rows, heads * head_dim])?,
        Tensor::new(grad_k, vec![rows, kv_heads * head_dim])?,
        Tensor::new(grad_v, vec![rows, kv_heads * head_dim])?,
    ))
}

#[allow(clippy::too_many_arguments, clippy::needless_range_loop)]
fn attention_batch_backward(
    q_data: &[f32],
    k_data: &[f32],
    v_data: &[f32],
    g_data: &[f32],
    batch_index: usize,
    seq: usize,
    heads: usize,
    kv_heads: usize,
    head_dim: usize,
    group: usize,
    scale: f32,
    causal: bool,
) -> (Vec<f32>, Vec<f32>, Vec<f32>) {
    let mut grad_q = vec![0.0f32; seq * heads * head_dim];
    let mut grad_k = vec![0.0f32; seq * kv_heads * head_dim];
    let mut grad_v = vec![0.0f32; seq * kv_heads * head_dim];

    for head in 0..heads {
        let kv_head = head / group;
        for position in 0..seq {
            let local_row = position * heads * head_dim + head * head_dim;
            let q_offset = (batch_index * seq + position) * heads * head_dim + head * head_dim;
            let mut scores = vec![f32::NEG_INFINITY; seq];
            for other in 0..seq {
                if causal && other > position {
                    continue;
                }
                let k_offset =
                    (batch_index * seq + other) * kv_heads * head_dim + kv_head * head_dim;
                scores[other] = (0..head_dim)
                    .map(|dim| q_data[q_offset + dim] * k_data[k_offset + dim])
                    .sum::<f32>()
                    * scale;
            }
            let probabilities = softmax_host(&scores);

            let mut grad_probabilities = vec![0.0f32; seq];
            for other in 0..seq {
                if causal && other > position {
                    continue;
                }
                let v_offset =
                    (batch_index * seq + other) * kv_heads * head_dim + kv_head * head_dim;
                grad_probabilities[other] = (0..head_dim)
                    .map(|dim| g_data[q_offset + dim] * v_data[v_offset + dim])
                    .sum::<f32>();
            }
            let weighted = (0..seq)
                .map(|other| probabilities[other] * grad_probabilities[other])
                .sum::<f32>();

            for other in 0..seq {
                if causal && other > position {
                    continue;
                }
                let key_row = other * kv_heads * head_dim + kv_head * head_dim;
                let grad_score = probabilities[other] * (grad_probabilities[other] - weighted);
                for dim in 0..head_dim {
                    grad_q[local_row + dim] += grad_score * k_data[key_row + dim] * scale;
                    grad_k[key_row + dim] += grad_score * q_data[q_offset + dim] * scale;
                    grad_v[key_row + dim] += probabilities[other] * g_data[q_offset + dim];
                }
            }
        }
    }
    (grad_q, grad_k, grad_v)
}

fn softmax_host(scores: &[f32]) -> Vec<f32> {
    let max = scores.iter().copied().fold(f32::NEG_INFINITY, f32::max);
    let mut result: Vec<f32> = scores
        .iter()
        .map(|score| {
            if score.is_finite() {
                (score - max).exp()
            } else {
                0.0
            }
        })
        .collect();
    let total: f32 = result.iter().sum();
    if total > 0.0 {
        for value in &mut result {
            *value /= total;
        }
    }
    result
}
