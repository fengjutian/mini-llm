//! # mini-llm-backend
//!
//! A thin seam between the educational tensor engine and accelerated kernels.
//!
//! [`CpuBackend`] is always available and simply delegates to
//! [`mini_llm_tensor::Tensor`]. [`CandleBackend`] is compiled with the `candle`
//! feature and implements the same trait with Hugging Face Candle, which makes
//! CPU/GPU parity tests a matter of comparing two [`MatMulBackend`]s.
//!
//! ```no_run
//! use mini_llm_backend::{CpuBackend, MatMulBackend};
//! use mini_llm_tensor::Tensor;
//!
//! let a = Tensor::from_2d(2, 2, vec![1.0, 2.0, 3.0, 4.0]).unwrap();
//! let b = Tensor::from_2d(2, 2, vec![5.0, 6.0, 7.0, 8.0]).unwrap();
//! let product = CpuBackend.matmul(&a, &b).unwrap();
//! assert_eq!(product.data(), &[19.0, 22.0, 43.0, 50.0]);
//! ```

use std::error::Error;
use std::fmt::{Display, Formatter};

use mini_llm_tensor::{Tensor, TensorError};

/// The one kernel the backends currently need to agree on.
pub trait MatMulBackend {
    fn matmul(&self, left: &Tensor, right: &Tensor) -> Result<Tensor, BackendError>;
}

/// Reference implementation backed by the transparent CPU tensor.
#[derive(Debug, Clone, Copy, Default)]
pub struct CpuBackend;

impl MatMulBackend for CpuBackend {
    fn matmul(&self, left: &Tensor, right: &Tensor) -> Result<Tensor, BackendError> {
        Ok(left.matmul(right)?)
    }
}

#[cfg(feature = "candle")]
pub use candle_backend::CandleBackend;

#[cfg(feature = "candle")]
mod candle_backend {
    use super::*;
    use candle_core::{Device, Tensor as CandleTensor};

    /// Candle-backed implementation, run on the CPU device by default.
    pub struct CandleBackend {
        device: Device,
    }

    impl CandleBackend {
        pub fn new(device: Device) -> Self {
            Self { device }
        }

        pub fn cpu() -> Self {
            Self::new(Device::Cpu)
        }

        pub fn device(&self) -> &Device {
            &self.device
        }
    }

    impl Default for CandleBackend {
        fn default() -> Self {
            Self::cpu()
        }
    }

    impl MatMulBackend for CandleBackend {
        fn matmul(&self, left: &Tensor, right: &Tensor) -> Result<Tensor, BackendError> {
            let left =
                CandleTensor::from_vec(left.data().to_vec(), left.shape().to_vec(), &self.device)
                    .map_err(|error| BackendError::Backend(error.to_string()))?;
            let right =
                CandleTensor::from_vec(right.data().to_vec(), right.shape().to_vec(), &self.device)
                    .map_err(|error| BackendError::Backend(error.to_string()))?;
            let product = left
                .matmul(&right)
                .map_err(|error| BackendError::Backend(error.to_string()))?;
            let shape = product.shape().dims().to_vec();
            let data = product
                .flatten_all()
                .and_then(|tensor| tensor.to_vec1::<f32>())
                .map_err(|error| BackendError::Backend(error.to_string()))?;
            Ok(Tensor::new(data, shape)?)
        }
    }
}

#[derive(Debug, Clone, PartialEq)]
pub enum BackendError {
    Tensor(TensorError),
    Backend(String),
}

impl Display for BackendError {
    fn fmt(&self, f: &mut Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Tensor(error) => Display::fmt(error, f),
            Self::Backend(message) => write!(f, "backend error: {message}"),
        }
    }
}

impl Error for BackendError {}

impl From<TensorError> for BackendError {
    fn from(value: TensorError) -> Self {
        Self::Tensor(value)
    }
}
