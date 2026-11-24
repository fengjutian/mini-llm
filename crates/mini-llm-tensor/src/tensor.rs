//! A deliberately small, transparent CPU tensor.
//!
//! Storage, constructors, element-wise arithmetic and reductions live here;
//! matrix kernels, gathers and RoPE live in `crate::matrix`.

use std::sync::Arc;

use crate::broadcast::{advance_index, broadcast_offsets, broadcast_shape, contiguous_strides};
use crate::error::TensorError;

/// A dense, row-major tensor with `f32` storage.
///
/// The buffer is shared behind an `Arc`, so cloning a `Tensor` is O(1). This
/// matters because the autograd engine clones node values on every access; with
/// an owned `Vec` that copy was proportional to the tensor size.
#[derive(Debug, Clone, PartialEq)]
pub struct Tensor {
    pub(crate) data: Arc<Vec<f32>>,
    pub(crate) shape: Vec<usize>,
}

impl Tensor {
    pub fn new(data: Vec<f32>, shape: Vec<usize>) -> Result<Self, TensorError> {
        let expected = shape
            .iter()
            .try_fold(1usize, |acc, &dim| acc.checked_mul(dim));
        if expected != Some(data.len()) {
            return Err(TensorError::ShapeDataMismatch {
                shape,
                data_len: data.len(),
            });
        }
        Ok(Self {
            data: Arc::new(data),
            shape,
        })
    }

    pub fn zeros(shape: Vec<usize>) -> Result<Self, TensorError> {
        let len = shape
            .iter()
            .try_fold(1usize, |acc, &dim| acc.checked_mul(dim))
            .ok_or_else(|| TensorError::ShapeDataMismatch {
                shape: shape.clone(),
                data_len: 0,
            })?;
        Self::new(vec![0.0; len], shape)
    }

    pub fn from_2d(rows: usize, cols: usize, data: Vec<f32>) -> Result<Self, TensorError> {
        Self::new(data, vec![rows, cols])
    }

    pub fn from_1d(data: Vec<f32>) -> Result<Self, TensorError> {
        let len = data.len();
        Self::new(data, vec![len])
    }

    pub fn full(shape: Vec<usize>, value: f32) -> Result<Self, TensorError> {
        let len = shape
            .iter()
            .try_fold(1usize, |acc, &dim| acc.checked_mul(dim))
            .ok_or_else(|| TensorError::ShapeDataMismatch {
                shape: shape.clone(),
                data_len: 0,
            })?;
        Self::new(vec![value; len], shape)
    }

    pub fn get(&self, index: usize) -> Option<f32> {
        self.data.get(index).copied()
    }

    pub fn shape(&self) -> &[usize] {
        &self.shape
    }

    pub fn data(&self) -> &[f32] {
        self.data.as_slice()
    }

    pub fn into_data(self) -> Vec<f32> {
        Arc::try_unwrap(self.data).unwrap_or_else(|shared| (*shared).clone())
    }

    pub fn rank(&self) -> usize {
        self.shape.len()
    }

    pub fn add(&self, other: &Self) -> Result<Self, TensorError> {
        self.zip_map(other, |a, b| a + b)
    }

    pub fn sub(&self, other: &Self) -> Result<Self, TensorError> {
        self.zip_map(other, |a, b| a - b)
    }

    /// Element-wise multiplication (Hadamard product).
    pub fn mul(&self, other: &Self) -> Result<Self, TensorError> {
        self.zip_map(other, |a, b| a * b)
    }

    fn zip_map(&self, other: &Self, f: impl Fn(f32, f32) -> f32) -> Result<Self, TensorError> {
        let out_shape = broadcast_shape(&self.shape, &other.shape)?;
        if self.shape == other.shape {
            let data = self
                .data
                .iter()
                .zip(other.data.iter())
                .map(|(a, b)| f(*a, *b))
                .collect();
            return Self::new(data, out_shape);
        }

        // NumPy-style right-aligned broadcasting over contiguous tensors.
        let out_len: usize = out_shape.iter().product();
        let out_rank = out_shape.len();
        let left_offsets = broadcast_offsets(&self.shape, out_rank);
        let right_offsets = broadcast_offsets(&other.shape, out_rank);

        let mut data = Vec::with_capacity(out_len);
        let mut index = vec![0usize; out_rank];
        for _ in 0..out_len {
            let mut left = 0usize;
            let mut right = 0usize;
            for dim in 0..out_rank {
                left += index[dim] * left_offsets[dim];
                right += index[dim] * right_offsets[dim];
            }
            data.push(f(self.data[left], other.data[right]));
            advance_index(&mut index, &out_shape);
        }
        Self::new(data, out_shape)
    }

    pub fn neg(&self) -> Result<Self, TensorError> {
        Self::new(self.data.iter().map(|x| -x).collect(), self.shape.clone())
    }

    pub fn scale(&self, factor: f32) -> Result<Self, TensorError> {
        Self::new(
            self.data.iter().map(|x| x * factor).collect(),
            self.shape.clone(),
        )
    }

    pub fn add_scalar(&self, value: f32) -> Result<Self, TensorError> {
        Self::new(
            self.data.iter().map(|x| x + value).collect(),
            self.shape.clone(),
        )
    }

    pub fn powf(&self, exponent: f32) -> Result<Self, TensorError> {
        Self::new(
            self.data.iter().map(|x| x.powf(exponent)).collect(),
            self.shape.clone(),
        )
    }

    /// Applies `f` element-wise, keeping the shape unchanged.
    pub fn map(&self, f: impl Fn(f32) -> f32) -> Result<Self, TensorError> {
        Self::new(
            self.data.iter().copied().map(f).collect(),
            self.shape.clone(),
        )
    }

    /// Collapses every element into a rank-1 tensor of length one.
    pub fn sum(&self) -> Result<Self, TensorError> {
        Self::new(vec![self.data.iter().sum()], vec![1])
    }

    /// Collapses every element into the mean, as a rank-1 tensor of length one.
    pub fn mean(&self) -> Result<Self, TensorError> {
        if self.data.is_empty() {
            return Err(TensorError::EmptyTensor);
        }
        let mean = self.data.iter().sum::<f32>() / self.data.len() as f32;
        Self::new(vec![mean], vec![1])
    }

    pub fn len(&self) -> usize {
        self.data.len()
    }

    pub fn is_empty(&self) -> bool {
        self.data.is_empty()
    }

    /// Sums this tensor down to `shape`, undoing a broadcast.
    ///
    /// Every dimension of `shape` must either match the corresponding
    /// right-aligned dimension of `self` or be one (or absent).
    pub fn sum_to(&self, shape: &[usize]) -> Result<Self, TensorError> {
        if self.shape == shape {
            return Ok(self.clone());
        }
        if broadcast_shape(&self.shape, shape)? != self.shape {
            return Err(TensorError::ShapeMismatch {
                left: self.shape.clone(),
                right: shape.to_vec(),
            });
        }

        let target_rank = shape.len();
        let source_rank = self.rank();
        let strides = contiguous_strides(shape);
        let target_len: usize = shape.iter().product();

        let mut data = vec![0.0; target_len];
        let mut index = vec![0usize; source_rank];
        for &value in self.data.iter() {
            let mut offset = 0usize;
            for dim in 0..target_rank {
                let source_dim = source_rank - target_rank + dim;
                let coordinate = if shape[dim] == 1 {
                    0
                } else {
                    index[source_dim]
                };
                offset += coordinate * strides[dim];
            }
            data[offset] += value;
            advance_index(&mut index, &self.shape);
        }
        Self::new(data, shape.to_vec())
    }

    pub fn relu(&self) -> Result<Self, TensorError> {
        Self::new(
            self.data.iter().map(|x| x.max(0.0)).collect(),
            self.shape.clone(),
        )
    }

    pub fn tanh(&self) -> Result<Self, TensorError> {
        self.map(f32::tanh)
    }

    pub fn sigmoid(&self) -> Result<Self, TensorError> {
        self.map(|x| 1.0 / (1.0 + (-x).exp()))
    }

    /// Row-wise stable softmax for a rank-2 tensor.
    pub fn softmax_rows(&self) -> Result<Self, TensorError> {
        if self.rank() != 2 {
            return Err(TensorError::ExpectedRank {
                expected: 2,
                actual: self.rank(),
            });
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
        // Accumulation order may differ from the naive kernel; compare with a
        // small tolerance instead of bit equality.
        let expected = [58., 64., 139., 154.];
        for (actual, expected) in c.data().iter().zip(expected.iter()) {
            assert!((actual - expected).abs() < 1e-5, "{actual} != {expected}");
        }
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

    #[test]
    fn elementwise_arithmetic_works() {
        let a = Tensor::from_1d(vec![1., 2., 3.]).unwrap();
        let b = Tensor::from_1d(vec![4., 5., 6.]).unwrap();
        assert_eq!(a.sub(&b).unwrap().data(), &[-3., -3., -3.]);
        assert_eq!(a.mul(&b).unwrap().data(), &[4., 10., 18.]);
        assert_eq!(a.scale(2.0).unwrap().data(), &[2., 4., 6.]);
    }

    #[test]
    fn mismatched_shapes_are_rejected() {
        let a = Tensor::from_1d(vec![1., 2.]).unwrap();
        let b = Tensor::from_1d(vec![1., 2., 3.]).unwrap();
        assert!(matches!(a.add(&b), Err(TensorError::ShapeMismatch { .. })));
    }

    #[test]
    fn reductions_collapse_to_a_single_value() {
        let x = Tensor::from_2d(2, 2, vec![1., 2., 3., 4.]).unwrap();
        assert_eq!(x.sum().unwrap().data(), &[10.]);
        assert_eq!(x.mean().unwrap().data(), &[2.5]);
        assert_eq!(x.sum().unwrap().shape(), &[1]);
    }

    #[test]
    fn broadcasting_expands_rank_one_biases() {
        let x = Tensor::from_2d(2, 3, vec![1., 2., 3., 4., 5., 6.]).unwrap();
        let bias = Tensor::from_1d(vec![10., 20., 30.]).unwrap();
        let y = x.add(&bias).unwrap();
        assert_eq!(y.shape(), &[2, 3]);
        assert_eq!(y.data(), &[11., 22., 33., 14., 25., 36.]);
    }

    #[test]
    fn gather_selects_rows_and_rejects_bad_indices() {
        let x = Tensor::from_2d(3, 2, vec![1., 2., 3., 4., 5., 6.]).unwrap();
        let picked = x.gather_rows(&[2, 0, 2]).unwrap();
        assert_eq!(picked.shape(), &[3, 2]);
        assert_eq!(picked.data(), &[5., 6., 1., 2., 5., 6.]);
        assert!(matches!(
            x.gather_rows(&[3]),
            Err(TensorError::RowIndexOutOfBounds { .. })
        ));
    }

    #[test]
    fn sum_to_reduces_broadcast_gradients() {
        let grad = Tensor::from_2d(2, 3, vec![1., 1., 1., 1., 1., 1.]).unwrap();
        assert_eq!(grad.sum_to(&[1, 3]).unwrap().data(), &[2., 2., 2.]);
        assert_eq!(grad.sum_to(&[1]).unwrap().data(), &[6.]);
        assert_eq!(grad.sum_to(&[2, 3]).unwrap().data(), grad.data());
    }

    #[test]
    fn rope_at_position_zero_is_the_identity() {
        let x = Tensor::from_2d(2, 4, vec![0.5, -1.0, 0.25, 0.75, 1.5, 0.0, -0.5, 2.0]).unwrap();
        let rotated = x.rope_rotate(4, &[0, 0], 10_000.0).unwrap();
        assert_eq!(rotated.data(), x.data());
    }

    #[test]
    fn rope_preserves_the_norm_of_each_head() {
        let x = Tensor::from_2d(1, 4, vec![0.5, -1.0, 0.25, 0.75]).unwrap();
        let rotated = x.rope_rotate(4, &[5], 10_000.0).unwrap();
        let norm = |values: &[f32]| values.iter().map(|v| v * v).sum::<f32>().sqrt();
        assert!((norm(rotated.data()) - norm(x.data())).abs() < 1e-6);
    }

    #[test]
    fn rope_scores_depend_only_on_relative_position() {
        let x = Tensor::from_2d(1, 4, vec![0.5, 1.0, -0.25, 0.75]).unwrap();
        let dot = |a: &Tensor, b: &Tensor| -> f32 {
            a.data().iter().zip(b.data()).map(|(l, r)| l * r).sum()
        };
        let q3 = x.rope_rotate(4, &[3], 10_000.0).unwrap();
        let k7 = x.rope_rotate(4, &[7], 10_000.0).unwrap();
        let q0 = x.rope_rotate(4, &[0], 10_000.0).unwrap();
        let k4 = x.rope_rotate(4, &[4], 10_000.0).unwrap();
        assert!((dot(&q3, &k7) - dot(&q0, &k4)).abs() < 1e-5);
    }

    #[test]
    fn rope_rejects_bad_dimensions() {
        let x = Tensor::from_2d(1, 4, vec![0.; 4]).unwrap();
        assert!(matches!(
            x.rope_rotate(3, &[0], 10_000.0),
            Err(TensorError::InvalidRopeDimension { .. })
        ));
        assert!(matches!(
            x.rope_rotate(4, &[0, 1], 10_000.0),
            Err(TensorError::RopePositionMismatch { .. })
        ));
    }
}
