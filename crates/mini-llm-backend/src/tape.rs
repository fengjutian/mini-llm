//! Reverse-mode automatic differentiation for device-resident tensors.
//!
//! This is the device counterpart of `mini_llm_autograd::TensorValue`: the same
//! topological-sort backward pass, but every value and gradient lives on a
//! [`DeviceTensor`] and every backward op runs through the device kernels.
//!
//! Supported ops: `matmul`, `add` (with `[1, cols]` bias broadcast), `relu`,
//! `mean` (to a scalar) and `scale`. Gradients accumulate, and nodes whose
//! `requires_grad` is false are skipped like in the host engine.

use std::cell::RefCell;
use std::collections::HashSet;
use std::rc::Rc;

use mini_llm_tensor::Tensor;

use crate::{BackendError, DeviceTensor};

type Backward = Box<dyn Fn(&DeviceTensor) -> Result<Vec<DeviceTensor>, BackendError>>;

struct Node {
    data: DeviceTensor,
    requires_grad: bool,
    grad: RefCell<Option<DeviceTensor>>,
    parents: Vec<DeviceValue>,
    backward: Option<Backward>,
}

/// A device value in a reverse-mode graph.
#[derive(Clone)]
pub struct DeviceValue {
    node: Rc<Node>,
}

impl DeviceValue {
    pub fn new(data: DeviceTensor, requires_grad: bool) -> Self {
        Self {
            node: Rc::new(Node {
                data,
                requires_grad,
                grad: RefCell::new(None),
                parents: Vec::new(),
                backward: None,
            }),
        }
    }

    pub fn data(&self) -> &DeviceTensor {
        &self.node.data
    }

    pub fn shape(&self) -> &[usize] {
        self.node.data.shape()
    }

    pub fn requires_grad(&self) -> bool {
        self.node.requires_grad
    }

    pub fn grad(&self) -> Option<DeviceTensor> {
        self.node.grad.borrow().clone()
    }

    pub fn zero_grad(&self) {
        *self.node.grad.borrow_mut() = None;
    }

    fn with_backward(data: DeviceTensor, parents: Vec<DeviceValue>, backward: Backward) -> Self {
        let requires_grad = parents.iter().any(DeviceValue::requires_grad);
        Self {
            node: Rc::new(Node {
                data,
                requires_grad,
                grad: RefCell::new(None),
                parents,
                backward: Some(backward),
            }),
        }
    }

    fn accumulate(&self, grad: DeviceTensor) -> Result<(), BackendError> {
        let mut slot = self.node.grad.borrow_mut();
        *slot = Some(match slot.take() {
            Some(existing) => existing.add(&grad)?,
            None => grad,
        });
        Ok(())
    }

    /// `self @ other`, with both gradients computed on the device.
    pub fn matmul(&self, other: &DeviceValue) -> Result<DeviceValue, BackendError> {
        let data = self.data().matmul(other.data())?;
        if !self.requires_grad() && !other.requires_grad() {
            return Ok(DeviceValue::new(data, false));
        }
        let left = self.data().clone();
        let right = other.data().clone();
        Ok(DeviceValue::with_backward(
            data,
            vec![self.clone(), other.clone()],
            Box::new(move |grad| {
                Ok(vec![
                    grad.matmul(&right.transpose_2d()?)?,
                    left.transpose_2d()?.matmul(grad)?,
                ])
            }),
        ))
    }

    /// Element-wise addition; a `[1, cols]` operand broadcasts over rows.
    pub fn add(&self, other: &DeviceValue) -> Result<DeviceValue, BackendError> {
        let data = self.data().add(other.data())?;
        if !self.requires_grad() && !other.requires_grad() {
            return Ok(DeviceValue::new(data, false));
        }
        let left_shape = self.shape().to_vec();
        let right_shape = other.shape().to_vec();
        Ok(DeviceValue::with_backward(
            data,
            vec![self.clone(), other.clone()],
            Box::new(move |grad| {
                Ok(vec![
                    reduce_to(grad, &left_shape)?,
                    reduce_to(grad, &right_shape)?,
                ])
            }),
        ))
    }

    /// Element-wise ReLU.
    pub fn relu(&self) -> Result<DeviceValue, BackendError> {
        let data = self.data().relu()?;
        if !self.requires_grad() {
            return Ok(DeviceValue::new(data, false));
        }
        let mask = data.clone();
        Ok(DeviceValue::with_backward(
            data,
            vec![self.clone()],
            Box::new(move |grad| Ok(vec![grad.mul(&mask.positive_mask()?)?])),
        ))
    }

    /// Mean of every element, as a `[1]` tensor.
    pub fn mean(&self) -> Result<DeviceValue, BackendError> {
        let data = self.data().mean_scalar()?;
        if !self.requires_grad() {
            return Ok(DeviceValue::new(data, false));
        }
        let (rows, cols) = (self.shape()[0], self.shape()[1]);
        let numel = (rows * cols) as f32;
        Ok(DeviceValue::with_backward(
            data,
            vec![self.clone()],
            Box::new(move |grad| {
                Ok(vec![grad
                    .scale(1.0 / numel)?
                    .broadcast_to_2d(rows, cols)?])
            }),
        ))
    }

    /// Multiplies every element by `factor`.
    pub fn scale(&self, factor: f32) -> Result<DeviceValue, BackendError> {
        let data = self.data().scale(factor)?;
        if !self.requires_grad() {
            return Ok(DeviceValue::new(data, false));
        }
        Ok(DeviceValue::with_backward(
            data,
            vec![self.clone()],
            Box::new(move |grad| Ok(vec![grad.scale(factor)?])),
        ))
    }

    /// Runs the reverse pass, seeding the root gradient with ones.
    pub fn backward(&self) -> Result<(), BackendError> {
        let mut visited = HashSet::new();
        let mut order = Vec::new();
        self.build_topo(&mut visited, &mut order);

        if self.grad().is_none() {
            let seed = Tensor::full(self.shape().to_vec(), 1.0)?;
            let seed = DeviceTensor::from_host(seed).to(&self.data().device())?;
            self.accumulate(seed)?;
        }

        for value in order.iter().rev() {
            let Some(grad) = value.grad() else {
                continue;
            };
            let Some(backward) = &value.node.backward else {
                continue;
            };
            let gradients = backward(&grad)?;
            for (parent, gradient) in value.node.parents.iter().zip(gradients) {
                if parent.requires_grad() {
                    parent.accumulate(gradient)?;
                }
            }
        }
        Ok(())
    }

    fn build_topo(&self, visited: &mut HashSet<usize>, order: &mut Vec<DeviceValue>) {
        let key = Rc::as_ptr(&self.node) as usize;
        if !visited.insert(key) {
            return;
        }
        for parent in &self.node.parents {
            parent.build_topo(visited, order);
        }
        order.push(self.clone());
    }
}

/// Reduces a gradient to `shape`, handling the `[1, cols]` bias broadcast.
fn reduce_to(grad: &DeviceTensor, shape: &[usize]) -> Result<DeviceTensor, BackendError> {
    if grad.shape() == shape {
        return Ok(grad.clone());
    }
    if shape.len() == 2 && shape[0] == 1 && grad.shape().len() == 2 && grad.shape()[1] == shape[1] {
        return grad.sum_rows();
    }
    Err(BackendError::Backend(format!(
        "cannot reduce gradient of shape {:?} to {:?}",
        grad.shape(),
        shape
    )))
}
