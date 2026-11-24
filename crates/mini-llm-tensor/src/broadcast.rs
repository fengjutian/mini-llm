//! NumPy-style broadcasting helpers shared by the tensor kernels.

use crate::error::TensorError;

/// Row-major strides of a contiguous tensor with `shape`.
pub(crate) fn contiguous_strides(shape: &[usize]) -> Vec<usize> {
    let mut strides = vec![1usize; shape.len()];
    for dim in (0..shape.len().saturating_sub(1)).rev() {
        strides[dim] = strides[dim + 1] * shape[dim + 1];
    }
    strides
}

/// NumPy-style broadcast shape: dimensions are right-aligned and may be equal
/// or one.
pub(crate) fn broadcast_shape(left: &[usize], right: &[usize]) -> Result<Vec<usize>, TensorError> {
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
pub(crate) fn broadcast_offsets(shape: &[usize], out_rank: usize) -> Vec<usize> {
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
pub(crate) fn advance_index(index: &mut [usize], shape: &[usize]) {
    for dim in (0..shape.len()).rev() {
        index[dim] += 1;
        if index[dim] < shape[dim] {
            return;
        }
        index[dim] = 0;
    }
}
