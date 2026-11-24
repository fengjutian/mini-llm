//! Shape manipulation: transpose, slicing, row gathers, RoPE and row means.

use mini_llm_tensor::Tensor;

use crate::error::AutogradError;
use crate::graph::matrix_shape;
use crate::value::TensorValue;

impl TensorValue {
    /// Transposes a rank-2 value.
    pub fn transpose(&self) -> Result<TensorValue, AutogradError> {
        let value = self.value().transpose_2d()?;
        self.unary("transpose", value, |g| Ok(g.transpose_2d()?))
    }

    /// Selects `len` consecutive rows starting at `start` (rank-2 only).
    ///
    /// This is the primitive used to slice per-head projection weights.
    pub fn slice_rows(&self, start: usize, len: usize) -> Result<TensorValue, AutogradError> {
        let source = self.value();
        let (rows, cols) = matrix_shape(&source)?;
        if start + len > rows {
            return Err(AutogradError::SliceOutOfBounds {
                axis: "row",
                start,
                len,
                size: rows,
            });
        }
        let value = Tensor::new(
            source.data()[start * cols..(start + len) * cols].to_vec(),
            vec![len, cols],
        )?;
        self.unary("slice_rows", value, move |g| {
            let mut out = vec![0.0f32; rows * cols];
            let begin = start * cols;
            out[begin..begin + len * cols].copy_from_slice(g.data());
            Ok(Tensor::new(out, vec![rows, cols])?)
        })
    }

    /// Selects `len` consecutive columns starting at `start` (rank-2 only).
    pub fn slice_cols(&self, start: usize, len: usize) -> Result<TensorValue, AutogradError> {
        let source = self.value();
        let (rows, cols) = matrix_shape(&source)?;
        if start + len > cols {
            return Err(AutogradError::SliceOutOfBounds {
                axis: "column",
                start,
                len,
                size: cols,
            });
        }
        let input = source.data();
        let mut data = Vec::with_capacity(rows * len);
        for row in 0..rows {
            let base = row * cols + start;
            data.extend_from_slice(&input[base..base + len]);
        }
        let value = Tensor::new(data, vec![rows, len])?;
        self.unary("slice_cols", value, move |g| {
            let upstream = g.data();
            let mut out = vec![0.0f32; rows * cols];
            for row in 0..rows {
                let source = row * len;
                let target = row * cols + start;
                out[target..target + len].copy_from_slice(&upstream[source..source + len]);
            }
            Ok(Tensor::new(out, vec![rows, cols])?)
        })
    }

    /// Selects rows of a rank-2 value, e.g. embedding lookups.
    ///
    /// The gradient scatters back into the source rows and accumulates when an
    /// index appears more than once.
    pub fn gather_rows(&self, indices: &[usize]) -> Result<TensorValue, AutogradError> {
        let source = self.value();
        let (rows, cols) = (source.shape()[0], source.shape()[1]);
        let value = source.gather_rows(indices)?;
        let saved_indices = indices.to_vec();
        self.unary("gather_rows", value, move |g| {
            let upstream = g.data();
            let mut data = vec![0.0f32; rows * cols];
            for (position, &index) in saved_indices.iter().enumerate() {
                let target = index * cols;
                let source = position * cols;
                for col in 0..cols {
                    data[target + col] += upstream[source + col];
                }
            }
            Ok(Tensor::new(data, vec![rows, cols])?)
        })
    }

    /// Applies rotary position embedding to `[rows, heads * head_dim]`.
    ///
    /// `positions[r]` is the absolute position of row `r`. The backward pass
    /// rotates by the negated position, which is the exact inverse because
    /// rotation matrices are orthogonal.
    pub fn rope(
        &self,
        head_dim: usize,
        positions: &[usize],
        base: f32,
    ) -> Result<TensorValue, AutogradError> {
        let signed: Vec<i64> = positions.iter().map(|&position| position as i64).collect();
        let value = self.value().rope_rotate(head_dim, &signed, base)?;
        let inverse: Vec<i64> = signed.iter().map(|position| -position).collect();
        self.unary("rope", value, move |g| {
            Ok(g.rope_rotate(head_dim, &inverse, base)?)
        })
    }

    /// Row-wise sum of a rank-2 value, kept as `[rows, 1]` for broadcasting.
    pub fn sum_rows_keepdim(&self) -> Result<TensorValue, AutogradError> {
        let source = self.value();
        let (rows, cols) = matrix_shape(&source)?;
        let data = source.data();
        let mut sums = Vec::with_capacity(rows);
        for row in 0..rows {
            let start = row * cols;
            sums.push(data[start..start + cols].iter().sum::<f32>());
        }
        let value = Tensor::new(sums, vec![rows, 1])?;
        self.unary("sum_rows", value, move |g| {
            let upstream = g.data();
            let mut out = vec![0.0f32; rows * cols];
            for row in 0..rows {
                for col in 0..cols {
                    out[row * cols + col] = upstream[row];
                }
            }
            Ok(Tensor::new(out, vec![rows, cols])?)
        })
    }

    /// Row-wise mean of a rank-2 value, kept as `[rows, 1]` for broadcasting.
    pub fn mean_rows_keepdim(&self) -> Result<TensorValue, AutogradError> {
        let source = self.value();
        let (rows, cols) = matrix_shape(&source)?;
        if cols == 0 {
            return Err(AutogradError::Tensor(
                mini_llm_tensor::TensorError::EmptyTensor,
            ));
        }
        let data = source.data();
        let mut means = Vec::with_capacity(rows);
        for row in 0..rows {
            let start = row * cols;
            means.push(data[start..start + cols].iter().sum::<f32>() / cols as f32);
        }
        let value = Tensor::new(means, vec![rows, 1])?;
        self.unary("mean_rows", value, move |g| {
            let upstream = g.data();
            let mut out = vec![0.0f32; rows * cols];
            for row in 0..rows {
                let mean_grad = upstream[row] / cols as f32;
                for col in 0..cols {
                    out[row * cols + col] = mean_grad;
                }
            }
            Ok(Tensor::new(out, vec![rows, cols])?)
        })
    }
}
