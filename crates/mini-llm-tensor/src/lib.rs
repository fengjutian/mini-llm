//! A deliberately small, transparent CPU tensor implementation.
//! This is the educational reference backend, not a high-performance library.

mod broadcast;
mod error;
mod matrix;
mod tensor;

pub use error::TensorError;
pub use tensor::Tensor;
