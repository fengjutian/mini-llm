//! A deliberately small, transparent CPU tensor implementation.
//! This is the educational reference backend, not a high-performance library.

use std::error::Error;
use std::fmt::{Display, Formatter};

#[derive(Debug, Clone, PartialEq)]
pub enum TensorError {
    ShapeDataMismatch { shape: Vec<usize>, data_len: usize },
    ExpectedRank { expected: usize, actual: usize },
    MatMulShapeMismatch { left: Vec<usize>, right: Vec<usize> },
    EmptySoftmaxDimension,
}

impl Display for TensorError {
    fn fmt(&self, f: &mut Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::ShapeDataMismatch { shape, data_len } =>
                write!(f, "shape {:?} does not match data length {}", shape, data_len),
            Self::ExpectedRank { expected, actual } =>
                write!(f, "expected rank {}, got rank {}", expected, actual),
            Self::MatMulShapeMismatch { left, right } =>
                write!(f, "cannot multiply matrices with shapes {:?} and {:?}", left, right),
            Self::EmptySoftmaxDimension => write!(f, "softmax dimension cannot be empty"),
        }
    }
}

impl Error for TensorError {}

#[derive(Debug, Clone, PartialEq)]
pub struct Tensor {
    data: Vec<f32>,
    shape: Vec<usize>,
}

impl Tensor {
    pub fn new(data: Vec<f32>, shape: Vec<usize>) -> Result<Self, TensorError> {
        let expected = shape.iter().try_fold(1usize, |acc, &dim| acc.checked_mul(dim));
        if expected != Some(data.len()) {
            return Err(TensorError::ShapeDataMismatch {
                shape,
                data_len: data.len(),
            });
        }
        Ok(Self { data, shape })
    }

    pub fn zeros(shape: Vec<usize>) -> Result<Self, TensorError> {
        let len = shape.iter().try_fold(1usize, |acc, &dim| acc.checked_mul(dim))
            .ok_or_else(|| TensorError::ShapeDataMismatch {
                shape: shape.clone(),
                data_len: 0,
            })?;
        Self::new(vec![0.0; len], shape)
    }

    pub fn from_2d(rows: usize, cols: usize, data: Vec<f32>) -> Result<Self, TensorError> {
        Self::new(data, vec![rows, cols])
    }

    pub fn shape(&self) -> &[usize] {
        &self.shape
    }

    pub fn data(&self) -> &[f32] {
        &self.data
    }

    pub fn into_data(self) -> Vec<f32> {
        self.data
    }

    pub fn rank(&self) -> usize {
        self.shape.len()
    }

    pub fn add(&self, other: &Self) -> Result<Self, TensorError> {
        if self.shape != other.shape {
            return Err(TensorError::ShapeDataMismatch {
                shape: other.shape.clone(),
                data_len: other.data.len(),
            });
        }
        let data = self.data.iter().zip(&other.data)
            .map(|(a, b)| a + b)
            .collect();
        Self::new(data, self.shape.clone())
    }

    pub fn relu(&self) -> Result<Self, TensorError> {
        Self::new(self.data.iter().map(|x| x.max(0.0)).collect(), self.shape.clone())
    }

    /// Row-wise stable softmax for a rank-2 tensor.
    pub fn softmax_rows(&self) -> Result<Self, TensorError> {
        if self.rank() != 2 {
            return Err(TensorError::ExpectedRank { expected: 2, actual: self.rank() });
        }
        let rows = self.shape[0];
        let cols = self.shape[1];
        if cols == 0 {
            return Err(TensorError::EmptySoftmaxDimension);
        }

        let mut output = vec![0.0; self.data.len()];
        for row in 0..rows {
            let start = row * cols;
            let slice = &self.data[start..start + cols];
            let max_value = slice.iter().copied().fold(f32::NEG_INFINITY, f32::max);
            let exps: Vec<f32> = slice.iter().map(|x| (x - max_value).exp()).collect();
            let sum: f32 = exps.iter().sum();
            for col in 0..cols {
                output[start + col] = exps[col] / sum;
            }
        }
        Self::new(output, self.shape.clone())
    }

    pub fn transpose_2d(&self) -> Result<Self, TensorError> {
        if self.rank() != 2 {
            return Err(TensorError::ExpectedRank { expected: 2, actual: self.rank() });
        }
        let (rows, cols) = (self.shape[0], self.shape[1]);
        let mut output = vec![0.0; self.data.len()];
        for r in 0..rows {
            for c in 0..cols {
                output[c * rows + r] = self.data[r * cols + c];
            }
        }
        Self::new(output, vec![cols, rows])
    }

    pub fn matmul(&self, other: &Self) -> Result<Self, TensorError> {
        if self.rank() != 2 {
            return Err(TensorError::ExpectedRank { expected: 2, actual: self.rank() });
        }
        if other.rank() != 2 {
            return Err(TensorError::ExpectedRank { expected: 2, actual: other.rank() });
        }

        let (m, k_left) = (self.shape[0], self.shape[1]);
        let (k_right, n) = (other.shape[0], other.shape[1]);
        if k_left != k_right {
            return Err(TensorError::MatMulShapeMismatch {
                left: self.shape.clone(),
                right: other.shape.clone(),
            });
        }

        let mut output = vec![0.0; m * n];
        for i in 0..m {
            for k in 0..k_left {
                let a = self.data[i * k_left + k];
                for j in 0..n {
                    output[i * n + j] += a * other.data[k * n + j];
                }
            }
        }
        Self::new(output, vec![m, n])
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn rejects_invalid_shape() {
        let result = Tensor::new(vec![1.0, 2.0], vec![3]);
        assert!(matches!(result, Err(TensorError::ShapeDataMismatch { .. })));
    }

    #[test]
    fn matrix_multiplication_works() {
        let a = Tensor::from_2d(2, 3, vec![1., 2., 3., 4., 5., 6.]).unwrap();
        let b = Tensor::from_2d(3, 2, vec![7., 8., 9., 10., 11., 12.]).unwrap();
        let c = a.matmul(&b).unwrap();
        assert_eq!(c.shape(), &[2, 2]);
        assert_eq!(c.data(), &[58., 64., 139., 154.]);
    }

    #[test]
    fn softmax_rows_sum_to_one() {
        let x = Tensor::from_2d(2, 3, vec![1., 2., 3., 2., 2., 2.]).unwrap();
        let y = x.softmax_rows().unwrap();
        for row in y.data().chunks(3) {
            let sum: f32 = row.iter().sum();
            assert!((sum - 1.0).abs() < 1e-5);
        }
    }

    #[test]
    fn relu_clamps_negative_values() {
        let x = Tensor::from_2d(1, 3, vec![-1., 0., 2.]).unwrap();
        assert_eq!(x.relu().unwrap().data(), &[0., 0., 2.]);
    }
}
