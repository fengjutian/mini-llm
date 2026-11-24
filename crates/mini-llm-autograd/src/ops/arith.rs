//! Arithmetic, reductions and losses.

use mini_llm_tensor::Tensor;

use crate::error::AutogradError;
use crate::value::TensorValue;

impl TensorValue {
    pub fn add(&self, other: &TensorValue) -> Result<TensorValue, AutogradError> {
        let value = self.value().add(&other.value())?;
        let (left_shape, right_shape) = (self.shape(), other.shape());
        self.binary(other, "add", value, move |g| {
            Ok((g.sum_to(&left_shape)?, g.sum_to(&right_shape)?))
        })
    }

    pub fn sub(&self, other: &TensorValue) -> Result<TensorValue, AutogradError> {
        let value = self.value().sub(&other.value())?;
        let (left_shape, right_shape) = (self.shape(), other.shape());
        self.binary(other, "sub", value, move |g| {
            Ok((g.sum_to(&left_shape)?, g.neg()?.sum_to(&right_shape)?))
        })
    }

    pub fn mul(&self, other: &TensorValue) -> Result<TensorValue, AutogradError> {
        let (a, b) = (self.value(), other.value());
        let (left_shape, right_shape) = (self.shape(), other.shape());
        let value = a.mul(&b)?;
        self.binary(other, "mul", value, move |g| {
            Ok((
                g.mul(&b)?.sum_to(&left_shape)?,
                g.mul(&a)?.sum_to(&right_shape)?,
            ))
        })
    }

    pub fn matmul(&self, other: &TensorValue) -> Result<TensorValue, AutogradError> {
        let (a, b) = (self.value(), other.value());
        let value = a.matmul(&b)?;
        self.binary(other, "matmul", value, move |g| {
            // dA = G @ B^T, dB = A^T @ G
            Ok((g.matmul(&b.transpose_2d()?)?, a.transpose_2d()?.matmul(g)?))
        })
    }

    /// Multiplies every element by a constant.
    pub fn scale(&self, factor: f32) -> Result<TensorValue, AutogradError> {
        let value = self.value().scale(factor)?;
        self.unary("scale", value, move |g| Ok(g.scale(factor)?))
    }

    /// Adds a constant to every element.
    pub fn add_scalar(&self, offset: f32) -> Result<TensorValue, AutogradError> {
        let value = self.value().add_scalar(offset)?;
        self.unary("add_scalar", value, |g| Ok(g.clone()))
    }

    /// Element-wise `x^exponent`.
    pub fn powf(&self, exponent: f32) -> Result<TensorValue, AutogradError> {
        let x = self.value();
        let value = x.powf(exponent)?;
        self.unary("powf", value, move |g| {
            let derivative = x.powf(exponent - 1.0)?.scale(exponent)?;
            Ok(g.mul(&derivative)?)
        })
    }

    /// Collapses every element into a scalar (shape `[1]`).
    pub fn sum(&self) -> Result<TensorValue, AutogradError> {
        let shape = self.shape();
        let value = self.value().sum()?;
        self.unary("sum", value, move |g| {
            let factor = g.data().first().copied().unwrap_or(1.0);
            Ok(Tensor::full(shape.clone(), factor)?)
        })
    }

    /// Mean of every element as a scalar (shape `[1]`).
    pub fn mean(&self) -> Result<TensorValue, AutogradError> {
        let shape = self.shape();
        let count = self.value().len() as f32;
        let value = self.value().mean()?;
        self.unary("mean", value, move |g| {
            let factor = g.data().first().copied().unwrap_or(1.0) / count;
            Ok(Tensor::full(shape.clone(), factor)?)
        })
    }

    /// Mean squared error against a constant target.
    pub fn mse(&self, target: &TensorValue) -> Result<TensorValue, AutogradError> {
        let diff = self.sub(target)?;
        diff.powf(2.0)?.mean()
    }
}
