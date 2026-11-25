//! Matrix kernels: transpose, gather, matmul and rotary positions.

use std::sync::Arc;

use rayon::prelude::*;

use crate::error::TensorError;
use crate::tensor::Tensor;

impl Tensor {
    /// Appends one row to a rank-2 tensor in place.
    ///
    /// The buffer is shared behind an `Arc`, so this is O(1) amortized when the
    /// tensor is uniquely owned and only falls back to a full copy while another
    /// handle still shares it. KV caches rely on the cheap case.
    pub fn append_row(&mut self, row: &[f32]) -> Result<(), TensorError> {
        if self.rank() != 2 {
            return Err(TensorError::ExpectedRank {
                expected: 2,
                actual: self.rank(),
            });
        }
        let cols = self.shape[1];
        if row.len() != cols {
            return Err(TensorError::ShapeMismatch {
                left: vec![1, cols],
                right: vec![1, row.len()],
            });
        }
        Arc::make_mut(&mut self.data).extend_from_slice(row);
        self.shape[0] += 1;
        Ok(())
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

    /// Scatters each row of `values` into a zero `[rows, cols]` tensor at
    /// `indices`, accumulating when an index appears more than once.
    ///
    /// This is the forward counterpart of [`Tensor::gather_rows`] and the
    /// building block of sparse mixture-of-experts dispatch.
    pub fn scatter_add_rows(
        values: &Tensor,
        indices: &[usize],
        rows: usize,
    ) -> Result<Tensor, TensorError> {
        if values.rank() != 2 {
            return Err(TensorError::ExpectedRank {
                expected: 2,
                actual: values.rank(),
            });
        }
        let (count, cols) = (values.shape()[0], values.shape()[1]);
        if indices.len() != count {
            return Err(TensorError::ShapeMismatch {
                left: vec![count],
                right: vec![indices.len()],
            });
        }

        let mut data = vec![0.0f32; rows * cols];
        for (position, &index) in indices.iter().enumerate() {
            if index >= rows {
                return Err(TensorError::RowIndexOutOfBounds { index, rows });
            }
            let target = index * cols;
            let source = position * cols;
            for col in 0..cols {
                data[target + col] += values.data()[source + col];
            }
        }
        Tensor::new(data, vec![rows, cols])
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

        let mut output = (*self.data).clone();
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

        // i-k-j ordering: the right operand's row stays hot in cache while the
        // same row of `self` is broadcast across it. This vectorizes well and
        // measured ~4x faster than a transposed dot-product version at the
        // sizes this project trains on (see `cargo run --release -- bench`).
        let left = self.data.as_slice();
        let right = other.data.as_slice();
        let mut output = vec![0.0; m * n];

        // Large products split the rows across the rayon pool. Each output
        // element is still accumulated in the same order, so results are
        // bit-identical to the sequential kernel.
        let work = m.saturating_mul(n).saturating_mul(k_left);
        let threads = if work >= PARALLEL_MATMUL_WORK {
            let cores = std::thread::available_parallelism()
                .map(|value| value.get())
                .unwrap_or(1);
            // Give each thread a meaningful chunk instead of always using one
            // per core: medium matrices lose more to scheduling than they gain
            // from the extra parallelism.
            let useful = work / WORK_PER_THREAD;
            cores.min(m).min(useful.max(1))
        } else {
            1
        };

        if threads <= 1 {
            multiply_rows(left, right, &mut output, 0, k_left, n);
        } else {
            // More chunks than threads keeps the pool balanced when some rows
            // finish early; rayon's persistent workers avoid a spawn per call.
            let chunk_rows = m.div_ceil(threads * 4).max(1);
            output
                .par_chunks_mut(chunk_rows * n)
                .enumerate()
                .for_each(|(chunk_index, chunk)| {
                    multiply_rows(left, right, chunk, chunk_index * chunk_rows, k_left, n);
                });
        }
        Self::new(output, vec![m, n])
    }
}

/// Multiply-adds below this count stay on the calling thread.
const PARALLEL_MATMUL_WORK: usize = 1_000_000;

/// Minimum multiply-adds handed to one thread once parallelism kicks in.
const WORK_PER_THREAD: usize = 2_000_000;

/// Computes `left[first_row.., :] x right` into `output` (`[rows, n]`).
///
/// Four output rows are accumulated at a time so each row of `right` is loaded
/// once and used four times. Row-wise accumulation order is unchanged, so the
/// result is bit-identical to the simple per-row loop.
///
/// An eight-row variant was measured against this one and showed no consistent
/// gain (the extra accumulators spill), so the simpler four-row block is kept.
fn multiply_rows(
    left: &[f32],
    right: &[f32],
    output: &mut [f32],
    first_row: usize,
    k: usize,
    n: usize,
) {
    let rows = output.len() / n;
    let mut row = 0;

    while row + 4 <= rows {
        let (row0, rest) = output[row * n..].split_at_mut(n);
        let (row1, rest) = rest.split_at_mut(n);
        let (row2, rest) = rest.split_at_mut(n);
        let (row3, _) = rest.split_at_mut(n);
        let base = (first_row + row) * k;

        for inner in 0..k {
            let a0 = left[base + inner];
            let a1 = left[base + k + inner];
            let a2 = left[base + 2 * k + inner];
            let a3 = left[base + 3 * k + inner];
            let right_row = &right[inner * n..(inner + 1) * n];
            for (col, &b) in right_row.iter().enumerate() {
                row0[col] += a0 * b;
                row1[col] += a1 * b;
                row2[col] += a2 * b;
                row3[col] += a3 * b;
            }
        }
        row += 4;
    }

    for (offset, out_row) in output[row * n..].chunks_mut(n).enumerate() {
        let source_row = first_row + row + offset;
        let left_row = &left[source_row * k..(source_row + 1) * k];
        for (inner, &a) in left_row.iter().enumerate() {
            let right_row = &right[inner * n..(inner + 1) * n];
            for (out, &b) in out_row.iter_mut().zip(right_row) {
                *out += a * b;
            }
        }
    }
}
