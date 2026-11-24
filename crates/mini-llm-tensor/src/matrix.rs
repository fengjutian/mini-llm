//! Matrix kernels: transpose, gather, matmul and rotary positions.

use crate::error::TensorError;
use crate::tensor::Tensor;

impl Tensor {
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
        for (row, &position) in positions.iter().enumerate() {
            let position = position as f32;
            let row_offset = row * cols;
            for head in 0..heads {
                let head_offset = row_offset + head * head_dim;
                for (index, &frequency) in frequencies.iter().enumerate() {
                    let angle = position * frequency;
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
