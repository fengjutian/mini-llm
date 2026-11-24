//! Tensor error types.

use std::error::Error;
use std::fmt::{Display, Formatter};

#[derive(Debug, Clone, PartialEq)]
pub enum TensorError {
    ShapeDataMismatch { shape: Vec<usize>, data_len: usize },
    ExpectedRank { expected: usize, actual: usize },
    MatMulShapeMismatch { left: Vec<usize>, right: Vec<usize> },
    ShapeMismatch { left: Vec<usize>, right: Vec<usize> },
    EmptySoftmaxDimension,
    EmptyTensor,
    RowIndexOutOfBounds { index: usize, rows: usize },
    InvalidRopeDimension { head_dim: usize },
    RopePositionMismatch { rows: usize, positions: usize },
}

impl Display for TensorError {
    fn fmt(&self, f: &mut Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::ShapeDataMismatch { shape, data_len } => write!(
                f,
                "shape {:?} does not match data length {}",
                shape, data_len
            ),
            Self::ExpectedRank { expected, actual } => {
                write!(f, "expected rank {}, got rank {}", expected, actual)
            }
            Self::MatMulShapeMismatch { left, right } => write!(
                f,
                "cannot multiply matrices with shapes {:?} and {:?}",
                left, right
            ),
            Self::ShapeMismatch { left, right } => {
                write!(f, "shape mismatch between {:?} and {:?}", left, right)
            }
            Self::EmptySoftmaxDimension => write!(f, "softmax dimension cannot be empty"),
            Self::EmptyTensor => write!(f, "operation requires at least one element"),
            Self::RowIndexOutOfBounds { index, rows } => write!(
                f,
                "row index {} is out of bounds for a tensor with {} rows",
                index, rows
            ),
            Self::InvalidRopeDimension { head_dim } => write!(
                f,
                "RoPE head dimension must be even and non-zero, got {}",
                head_dim
            ),
            Self::RopePositionMismatch { rows, positions } => {
                write!(f, "RoPE got {} positions for {} rows", positions, rows)
            }
        }
    }
}

impl Error for TensorError {}
