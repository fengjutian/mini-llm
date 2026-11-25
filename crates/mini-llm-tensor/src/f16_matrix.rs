//! Prototype f16 matrix storage with f32 accumulation.
//!
//! Weights are kept as `u16` half-precision bit patterns, halving their memory
//! footprint, while every multiply-add happens in `f32`. This is a storage
//! prototype: training and the reference kernels still run in f32.
//!
//! The matmul is `k`-major so each row of the right operand is dequantized once
//! and reused across all output rows:
//!
//! ```text
//! for k:  b_row = dequantize(right[k, :])   // once
//!         for i: out[i, :] += left[i, k] * b_row
//! ```

use rayon::prelude::*;

use crate::{f16_to_f32, f32_to_f16, Tensor, TensorError};

/// A rank-2 matrix stored as half-precision bit patterns.
#[derive(Debug, Clone, PartialEq)]
pub struct F16Matrix {
    rows: usize,
    cols: usize,
    data: Vec<u16>,
}

impl F16Matrix {
    /// Quantizes a rank-2 tensor to f16 storage.
    pub fn from_tensor(tensor: &Tensor) -> Result<Self, TensorError> {
        if tensor.rank() != 2 {
            return Err(TensorError::ExpectedRank {
                expected: 2,
                actual: tensor.rank(),
            });
        }
        Ok(Self {
            rows: tensor.shape()[0],
            cols: tensor.shape()[1],
            data: tensor
                .data()
                .iter()
                .map(|value| f32_to_f16(*value))
                .collect(),
        })
    }

    /// Rebuilds an f32 tensor from the stored halves.
    pub fn to_tensor(&self) -> Result<Tensor, TensorError> {
        Tensor::new(
            self.data.iter().map(|bits| f16_to_f32(*bits)).collect(),
            vec![self.rows, self.cols],
        )
    }

    pub fn rows(&self) -> usize {
        self.rows
    }

    pub fn cols(&self) -> usize {
        self.cols
    }

    pub fn bits(&self) -> &[u16] {
        &self.data
    }

    /// `[m, k] x [k, n] -> [m, n]` with f32 accumulation.
    pub fn matmul(&self, other: &Self) -> Result<Tensor, TensorError> {
        if self.cols != other.rows {
            return Err(TensorError::MatMulShapeMismatch {
                left: vec![self.rows, self.cols],
                right: vec![other.rows, other.cols],
            });
        }
        let (m, k, n) = (self.rows, self.cols, other.cols);
        let mut output = vec![0.0f32; m * n];
        let mut dequantized_row = vec![0.0f32; n];

        for inner in 0..k {
            let right_row = &other.data[inner * n..(inner + 1) * n];
            for (slot, &bits) in dequantized_row.iter_mut().zip(right_row) {
                *slot = f16_to_f32(bits);
            }
            for row in 0..m {
                let a = f16_to_f32(self.data[row * k + inner]);
                let out_row = &mut output[row * n..(row + 1) * n];
                for (out, &b) in out_row.iter_mut().zip(&dequantized_row) {
                    *out += a * b;
                }
            }
        }

        Tensor::new(output, vec![m, n])
    }

    /// Row-parallel variant of [F16Matrix::matmul].
    ///
    /// Rows are independent, so each output row converts its own right-hand
    /// values. That repeats the conversions but lets the kernel use every core.
    pub fn matmul_parallel(&self, other: &Self) -> Result<Tensor, TensorError> {
        if self.cols != other.rows {
            return Err(TensorError::MatMulShapeMismatch {
                left: vec![self.rows, self.cols],
                right: vec![other.rows, other.cols],
            });
        }
        let (m, k, n) = (self.rows, self.cols, other.cols);
        let mut output = vec![0.0f32; m * n];
        output
            .par_chunks_mut(n)
            .enumerate()
            .for_each(|(row, out_row)| {
                let left_row = &self.data[row * k..(row + 1) * k];
                for (inner, &left_bits) in left_row.iter().enumerate() {
                    let a = f16_to_f32(left_bits);
                    let right_row = &other.data[inner * n..(inner + 1) * n];
                    for (out, &bits) in out_row.iter_mut().zip(right_row) {
                        *out += a * f16_to_f32(bits);
                    }
                }
            });
        Tensor::new(output, vec![m, n])
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn quantized_roundtrip_is_close() {
        let values = vec![0.1, -2.5, 3.75, 100.0, -0.0001];
        let tensor = Tensor::from_2d(1, 5, values.clone()).unwrap();
        let quantized = F16Matrix::from_tensor(&tensor).unwrap();
        let restored = quantized.to_tensor().unwrap();
        for (original, roundtrip) in values.iter().zip(restored.data()) {
            let tolerance = original.abs() * 1e-3 + 1e-6;
            assert!((original - roundtrip).abs() <= tolerance);
        }
    }

    #[test]
    fn f16_matmul_matches_f32_within_half_precision() {
        let left_values: Vec<f32> = (0..24).map(|index| (index as f32 - 12.0) * 0.25).collect();
        let right_values: Vec<f32> = (0..30).map(|index| (index as f32 - 15.0) * 0.1).collect();
        let left = Tensor::from_2d(4, 6, left_values).unwrap();
        let right = Tensor::from_2d(6, 5, right_values).unwrap();

        let reference = left.matmul(&right).unwrap();
        let quantized = F16Matrix::from_tensor(&left)
            .unwrap()
            .matmul(&F16Matrix::from_tensor(&right).unwrap())
            .unwrap();

        assert_eq!(quantized.shape(), reference.shape());
        for (expected, actual) in reference.data().iter().zip(quantized.data()) {
            let tolerance = expected.abs() * 2e-3 + 1e-2;
            assert!(
                (expected - actual).abs() <= tolerance,
                "{expected} vs {actual}"
            );
        }
    }

    #[test]
    fn f16_matmul_rejects_bad_shapes() {
        let left = F16Matrix::from_tensor(&Tensor::from_2d(2, 3, vec![0.0; 6]).unwrap()).unwrap();
        let right = F16Matrix::from_tensor(&Tensor::from_2d(2, 3, vec![0.0; 6]).unwrap()).unwrap();
        assert!(matches!(
            left.matmul(&right),
            Err(TensorError::MatMulShapeMismatch { .. })
        ));
    }
}
