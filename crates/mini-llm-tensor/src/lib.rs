//! A deliberately small, transparent CPU tensor implementation.
//! This is the educational reference backend, not a high-performance library.

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
            Self::RopePositionMismatch { rows, positions } => write!(
                f,
                "RoPE got {} positions for {} rows",
                positions, rows
            ),
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
        let expected = shape
            .iter()
            .try_fold(1usize, |acc, &dim| acc.checked_mul(dim));
        if expected != Some(data.len()) {
            return Err(TensorError::ShapeDataMismatch {
                shape,
                data_len: data.len(),
            });
        }
        Ok(Self { data, shape })
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
        &self.data
    }

    pub fn into_data(self) -> Vec<f32> {
        self.data
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
                .zip(&other.data)
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
        for &value in &self.data {
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

    pub fn transpose_2d(&self) -> Result<Self, TensorError> {
        if self.rank() != 2 {
            return Err(TensorError::ExpectedRank {
                expected: 2,
                actual: self.rank(),
            });
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

    /// Applies rotary position embedding (RoPE) to `[rows, heads * head_dim]`.
    ///
    /// Inside every head, the pair `(2i, 2i+1)` is rotated by
    /// `position * base^(-2i/head_dim)`. Negative positions apply the inverse
    /// rotation, which is exactly what the backward pass needs since the
    /// rotation matrix is orthogonal.
    pub fn rope_rotate(
        &self,
        head_dim: usize,
        positions: &[i64],
        base: f32,
    ) -> Result<Self, TensorError> {
        if head_dim == 0 || !head_dim.is_multiple_of(2) {
            return Err(TensorError::InvalidRopeDimension { head_dim });
        }
        if self.rank() != 2 {
            return Err(TensorError::ExpectedRank {
                expected: 2,
                actual: self.rank(),
            });
        }
        let (rows, cols) = (self.shape[0], self.shape[1]);
        if !cols.is_multiple_of(head_dim) {
            return Err(TensorError::ShapeMismatch {
                left: self.shape.clone(),
                right: vec![rows, head_dim],
            });
        }
        if positions.len() != rows {
            return Err(TensorError::RopePositionMismatch {
                rows,
                positions: positions.len(),
            });
        }

        let heads = cols / head_dim;
        let half = head_dim / 2;
        let frequencies: Vec<f32> = (0..half)
            .map(|index| 1.0 / base.powf(2.0 * index as f32 / head_dim as f32))
            .collect();

        let mut output = self.data.clone();
        for row in 0..rows {
            let position = positions[row] as f32;
            let row_offset = row * cols;
            for head in 0..heads {
                let head_offset = row_offset + head * head_dim;
                for index in 0..half {
                    let angle = position * frequencies[index];
                    let (sin, cos) = angle.sin_cos();
                    let first = head_offset + 2 * index;
                    let a = self.data[first];
                    let b = self.data[first + 1];
                    output[first] = a * cos - b * sin;
                    output[first + 1] = a * sin + b * cos;
                }
            }
        }
        Self::new(output, self.shape.clone())
    }

    /// Selects rows by index, producing `[indices.len(), cols]`.
    ///
    /// Repeated indices are allowed; the matching gradient accumulates.
    pub fn gather_rows(&self, indices: &[usize]) -> Result<Self, TensorError> {
        if self.rank() != 2 {
            return Err(TensorError::ExpectedRank {
                expected: 2,
                actual: self.rank(),
            });
        }
        let (rows, cols) = (self.shape[0], self.shape[1]);
        let mut data = Vec::with_capacity(indices.len() * cols);
        for &index in indices {
            if index >= rows {
                return Err(TensorError::RowIndexOutOfBounds { index, rows });
            }
            let start = index * cols;
            data.extend_from_slice(&self.data[start..start + cols]);
        }
        Self::new(data, vec![indices.len(), cols])
    }

    pub fn matmul(&self, other: &Self) -> Result<Self, TensorError> {
        if self.rank() != 2 {
            return Err(TensorError::ExpectedRank {
                expected: 2,
                actual: self.rank(),
            });
        }
        if other.rank() != 2 {
            return Err(TensorError::ExpectedRank {
                expected: 2,
                actual: other.rank(),
            });
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

/// Row-major strides of a contiguous tensor with `shape`.
fn contiguous_strides(shape: &[usize]) -> Vec<usize> {
    let mut strides = vec![1usize; shape.len()];
    for dim in (0..shape.len().saturating_sub(1)).rev() {
        strides[dim] = strides[dim + 1] * shape[dim + 1];
    }
    strides
}

/// NumPy-style broadcast shape: dimensions are right-aligned and may be equal
/// or one.
fn broadcast_shape(left: &[usize], right: &[usize]) -> Result<Vec<usize>, TensorError> {
    let rank = left.len().max(right.len());
    let mut shape = Vec::with_capacity(rank);
    for dim in 0..rank {
        let left_dim = aligned_dim(left, rank, dim);
        let right_dim = aligned_dim(right, rank, dim);
        shape.push(if left_dim == right_dim {
            left_dim
        } else if left_dim == 1 {
            right_dim
        } else if right_dim == 1 {
            left_dim
        } else {
            return Err(TensorError::ShapeMismatch {
                left: left.to_vec(),
                right: right.to_vec(),
            });
        });
    }
    Ok(shape)
}

fn aligned_dim(shape: &[usize], rank: usize, dim: usize) -> usize {
    let offset = rank - shape.len();
    if dim < offset {
        1
    } else {
        shape[dim - offset]
    }
}

/// Offsets (in elements of a contiguous tensor) contributed by each broadcast
/// dimension. A dimension of size one contributes nothing; missing leading
/// dimensions contribute nothing either.
fn broadcast_offsets(shape: &[usize], out_rank: usize) -> Vec<usize> {
    let strides = contiguous_strides(shape);
    let offset = out_rank - shape.len();
    (0..out_rank)
        .map(|dim| {
            if dim < offset {
                0
            } else {
                let source = dim - offset;
                if shape[source] == 1 {
                    0
                } else {
                    strides[source]
                }
            }
        })
        .collect()
}

/// Advances a multi-index odometer in row-major order.
fn advance_index(index: &mut [usize], shape: &[usize]) {
    for dim in (0..shape.len()).rev() {
        index[dim] += 1;
        if index[dim] < shape[dim] {
            return;
        }
        index[dim] = 0;
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
        let x = Tensor::from_2d(2, 4, vec![0.5, -1.0, 0.25, 0.75, 1.5, 0.0, -0.5, 2.0])
            .unwrap();
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
