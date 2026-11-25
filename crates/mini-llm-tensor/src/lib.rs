//! A deliberately small, transparent CPU tensor implementation.
//! This is the educational reference backend, not a high-performance library.

mod broadcast;
mod error;
mod f16_matrix;
mod half;
mod matrix;
mod tensor;

pub use error::TensorError;
pub use f16_matrix::F16Matrix;
pub use half::{f16_to_f32, f32_to_f16};
pub use tensor::Tensor;
