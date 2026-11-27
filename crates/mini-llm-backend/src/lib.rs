//! # mini-llm-backend
//!
//! A thin seam between the educational tensor engine and accelerated kernels.
//!
//! [`CpuBackend`] is always available and delegates to
//! [`mini_llm_tensor::Tensor`]. `CandleBackend` is compiled with the `candle`
//! feature and implements the same [`Backend`] trait, so CPU/GPU parity is a
//! matter of running both backends on the same inputs.
//!
//! The trait covers the kernels where acceleration matters: matrix
//! multiplication, row-wise softmax and RMS normalization. Kernels not
//! overridden by a backend fall back to the CPU reference implementation.
//!
//! ```no_run
//! use mini_llm_backend::{Backend, CpuBackend};
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

/// Kernels a backend must agree on.
pub trait Backend {
    /// `[m, k] x [k, n] -> [m, n]`.
    fn matmul(&self, left: &Tensor, right: &Tensor) -> Result<Tensor, BackendError>;

    /// Row-wise stable softmax.
    ///
    /// Backends that do not override this use the CPU reference implementation.
    fn softmax_rows(&self, input: &Tensor) -> Result<Tensor, BackendError> {
        Ok(input.softmax_rows()?)
    }

    /// `gamma * x / sqrt(mean(x^2) + eps)` over the last axis.
    ///
    /// The default implementation is composed from CPU tensor ops; accelerated
    /// backends override it to keep the whole normalisation on-device.
    fn rms_norm(&self, input: &Tensor, gamma: &Tensor, eps: f32) -> Result<Tensor, BackendError> {
        let mean_square = input.powf(2.0)?.mean_rows_keepdim()?;
        let denominator = mean_square.add_scalar(eps)?.powf(-0.5)?;
        Ok(input.mul(&denominator)?.mul(gamma)?)
    }
}

/// Reference implementation backed by the transparent CPU tensor.
#[derive(Debug, Clone, Copy, Default)]
pub struct CpuBackend;

impl Backend for CpuBackend {
    fn matmul(&self, left: &Tensor, right: &Tensor) -> Result<Tensor, BackendError> {
        Ok(left.matmul(right)?)
    }
}

#[cfg(feature = "candle")]
pub use candle_backend::CandleBackend;

#[cfg(feature = "candle")]
mod candle_backend {
    use super::*;
    use candle_core::{Device, Tensor as CandleTensor, D};

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

        fn to_candle(&self, tensor: &Tensor) -> Result<CandleTensor, BackendError> {
            CandleTensor::from_vec(
                tensor.data().to_vec(),
                tensor.shape().to_vec(),
                &self.device,
            )
            .map_err(|error| BackendError::Backend(error.to_string()))
        }

        fn from_candle(tensor: &CandleTensor) -> Result<Tensor, BackendError> {
            let shape = tensor.shape().dims().to_vec();
            let data = tensor
                .flatten_all()
                .and_then(|flat| flat.to_vec1::<f32>())
                .map_err(|error| BackendError::Backend(error.to_string()))?;
            Ok(Tensor::new(data, shape)?)
        }
    }

    impl Default for CandleBackend {
        fn default() -> Self {
            Self::cpu()
        }
    }

    impl Backend for CandleBackend {
        fn matmul(&self, left: &Tensor, right: &Tensor) -> Result<Tensor, BackendError> {
            let left = self.to_candle(left)?;
            let right = self.to_candle(right)?;
            let product = left
                .matmul(&right)
                .map_err(|error| BackendError::Backend(error.to_string()))?;
            Self::from_candle(&product)
        }

        fn softmax_rows(&self, input: &Tensor) -> Result<Tensor, BackendError> {
            // candle-core 0.11 keeps softmax in `candle-nn`; composing it here
            // avoids pulling that crate in just for one kernel.
            let input = self.to_candle(input)?;
            let max = input
                .max_keepdim(D::Minus1)
                .map_err(|error| BackendError::Backend(error.to_string()))?;
            let shifted = input
                .broadcast_sub(&max)
                .and_then(|value| value.exp())
                .map_err(|error| BackendError::Backend(error.to_string()))?;
            let sum = shifted
                .sum_keepdim(D::Minus1)
                .map_err(|error| BackendError::Backend(error.to_string()))?;
            let soft = shifted
                .broadcast_div(&sum)
                .map_err(|error| BackendError::Backend(error.to_string()))?;
            Self::from_candle(&soft)
        }

        fn rms_norm(
            &self,
            input: &Tensor,
            gamma: &Tensor,
            eps: f32,
        ) -> Result<Tensor, BackendError> {
            let input = self.to_candle(input)?;
            let gamma = self.to_candle(gamma)?;
            let eps = CandleTensor::new(eps, &self.device)
                .map_err(|error| BackendError::Backend(error.to_string()))?;

            let mean_square = input
                .sqr()
                .and_then(|squared| squared.mean_keepdim(D::Minus1))
                .map_err(|error| BackendError::Backend(error.to_string()))?;
            let denominator = mean_square
                .broadcast_add(&eps)
                .and_then(|sum| sum.sqrt())
                .map_err(|error| BackendError::Backend(error.to_string()))?;
            let normalised = input
                .broadcast_div(&denominator)
                .and_then(|scaled| scaled.broadcast_mul(&gamma))
                .map_err(|error| BackendError::Backend(error.to_string()))?;
            Self::from_candle(&normalised)
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
