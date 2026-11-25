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
pub struct DeviceTensor {
    shape: Vec<usize>,
    storage: Storage,
}

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
