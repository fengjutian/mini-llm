//! Deterministic random number generation for parameter initialization.
//!
//! The generator is `xorshift64*`: tiny, dependency-free and good enough for
//! seeding small experiments reproducibly. It is **not** suitable for anything
//! cryptographic.

use mini_llm_tensor::{Tensor, TensorError};

/// A seeded, deterministic pseudo-random generator.
#[derive(Debug, Clone)]
pub struct Rng {
    state: u64,
}

impl Rng {
    pub fn new(seed: u64) -> Self {
        Self {
            // Zero is a fixed point of xorshift, so map it to the golden ratio.
            state: if seed == 0 {
                0x9E37_79B9_7F4A_7C15
            } else {
                seed
            },
        }
    }

    pub fn next_u64(&mut self) -> u64 {
        let mut x = self.state;
        x ^= x >> 12;
        x ^= x << 25;
        x ^= x >> 27;
        self.state = x;
        x.wrapping_mul(0x2545_F491_4F6C_DD1D)
    }

    /// Uniform sample in `[0, 1)` with 24 bits of precision.
    pub fn next_f32(&mut self) -> f32 {
        ((self.next_u64() >> 40) as f32) * (1.0 / 16_777_216.0)
    }

    /// Uniform sample in `[low, high)`.
    pub fn uniform(&mut self, low: f32, high: f32) -> f32 {
        low + (high - low) * self.next_f32()
    }

    /// Standard normal sample via the Box-Muller transform.
    pub fn normal(&mut self) -> f32 {
        let u1 = self.next_f32().max(f32::MIN_POSITIVE);
        let u2 = self.next_f32();
        (-2.0 * u1.ln()).sqrt() * (std::f32::consts::TAU * u2).cos()
    }

    /// Normal tensor with the given shape and standard deviation.
    pub fn normal_tensor(&mut self, shape: Vec<usize>, std: f32) -> Result<Tensor, TensorError> {
        let len = shape
            .iter()
            .try_fold(1usize, |acc, &dim| acc.checked_mul(dim))
            .ok_or_else(|| TensorError::ShapeDataMismatch {
                shape: shape.clone(),
                data_len: 0,
            })?;
        let data = (0..len).map(|_| self.normal() * std).collect();
        Tensor::new(data, shape)
    }

    /// Uniform tensor with the given shape over `[low, high)`.
    pub fn uniform_tensor(
        &mut self,
        shape: Vec<usize>,
        low: f32,
        high: f32,
    ) -> Result<Tensor, TensorError> {
        let len = shape
            .iter()
            .try_fold(1usize, |acc, &dim| acc.checked_mul(dim))
            .ok_or_else(|| TensorError::ShapeDataMismatch {
                shape: shape.clone(),
                data_len: 0,
            })?;
        let data = (0..len).map(|_| self.uniform(low, high)).collect();
        Tensor::new(data, shape)
    }
}

/// Xavier/Glorot uniform initialization for a `[fan_in, fan_out]` weight.
///
/// The bound is `sqrt(6 / (fan_in + fan_out))`, which keeps the variance of the
/// activations roughly constant across layers.
pub fn xavier_uniform(fan_in: usize, fan_out: usize, rng: &mut Rng) -> Result<Tensor, TensorError> {
    let limit = (6.0 / (fan_in + fan_out) as f32).sqrt();
    rng.uniform_tensor(vec![fan_in, fan_out], -limit, limit)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn same_seed_produces_same_sequence() {
        let mut a = Rng::new(42);
        let mut b = Rng::new(42);
        for _ in 0..8 {
            assert_eq!(a.next_u64(), b.next_u64());
        }
    }

    #[test]
    fn samples_stay_in_range() {
        let mut rng = Rng::new(7);
        for _ in 0..1_000 {
            let value = rng.next_f32();
            assert!((0.0..1.0).contains(&value));
        }
    }

    #[test]
    fn xavier_tensor_has_the_requested_shape() {
        let mut rng = Rng::new(1);
        let tensor = xavier_uniform(4, 8, &mut rng).unwrap();
        assert_eq!(tensor.shape(), &[4, 8]);
        let limit = (6.0f32 / 12.0).sqrt();
        assert!(tensor.data().iter().all(|v| v.abs() <= limit));
    }
}
