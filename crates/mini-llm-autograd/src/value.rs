//! Reverse-mode automatic differentiation over [`mini_llm_tensor::Tensor`].
//!
//! Every [`TensorValue`] wraps a node of a dynamically built computation graph.
//! Nodes that were created by an operation keep a `backward` closure that maps
//! the upstream gradient to the gradients of its parents. `backward()` walks the
//! graph in reverse topological order and accumulates gradients into the nodes
//! that require them.
//!
//! The implementation is deliberately small and readable: it favours
//! transparency over throughput, exactly like the tensor engine underneath it.

use std::cell::RefCell;
use std::collections::HashSet;
use std::fmt;
use std::rc::Rc;

use mini_llm_tensor::Tensor;

use crate::error::AutogradError;

/// The gradient of a node's output with respect to the final scalar loss.
type BackwardFn = dyn Fn(&Tensor) -> Result<Vec<Tensor>, AutogradError>;

pub(crate) struct Node {
    pub(crate) value: Tensor,
    pub(crate) requires_grad: bool,
    pub(crate) grad: Option<Tensor>,
    pub(crate) op: &'static str,
    pub(crate) parents: Vec<TensorValue>,
    pub(crate) backward: Option<Rc<BackwardFn>>,
}

/// A tensor participating in the computation graph.
///
/// Cloning a `TensorValue` is cheap: clones share the same graph node, which is
/// what makes gradient accumulation across shared sub-expressions work.
#[derive(Clone)]
pub struct TensorValue {
    pub(crate) node: Rc<RefCell<Node>>,
}

impl TensorValue {
    fn from_parts(
        value: Tensor,
        requires_grad: bool,
        op: &'static str,
        parents: Vec<TensorValue>,
        backward: Option<Rc<BackwardFn>>,
    ) -> Self {
        Self {
            node: Rc::new(RefCell::new(Node {
                value,
                requires_grad,
                grad: None,
                op,
                parents,
                backward,
            })),
        }
    }

    /// Wraps a constant tensor. No gradient is tracked through it.
    pub fn new(value: Tensor) -> Self {
        Self::from_parts(value, false, "constant", Vec::new(), None)
    }

    /// Wraps a learnable leaf tensor such as a weight matrix or a bias.
    pub fn param(value: Tensor) -> Self {
        Self::from_parts(value, true, "param", Vec::new(), None)
    }

    /// Wraps a constant tensor built from raw data and a shape.
    pub fn constant(data: Vec<f32>, shape: Vec<usize>) -> Result<Self, AutogradError> {
        Ok(Self::new(Tensor::new(data, shape)?))
    }

    /// Wraps a learnable tensor built from raw data and a shape.
    pub fn parameter(data: Vec<f32>, shape: Vec<usize>) -> Result<Self, AutogradError> {
        Ok(Self::param(Tensor::new(data, shape)?))
    }

    pub fn value(&self) -> Tensor {
        self.node.borrow().value.clone()
    }

    pub fn data(&self) -> Vec<f32> {
        self.node.borrow().value.data().to_vec()
    }

    pub fn shape(&self) -> Vec<usize> {
        self.node.borrow().value.shape().to_vec()
    }

    pub fn rank(&self) -> usize {
        self.node.borrow().value.rank()
    }

    pub fn requires_grad(&self) -> bool {
        self.node.borrow().requires_grad
    }

    /// The operation that produced this node, e.g. `matmul` or `relu`.
    pub fn op(&self) -> &'static str {
        self.node.borrow().op
    }

    pub fn grad(&self) -> Option<Tensor> {
        self.node.borrow().grad.clone()
    }

    pub fn is_leaf(&self) -> bool {
        self.node.borrow().backward.is_none()
    }

    /// Stable identity of the underlying graph node, useful as a map key for
    /// optimizer state.
    pub fn id(&self) -> usize {
        Rc::as_ptr(&self.node) as usize
    }

    /// Replaces the value of a learnable parameter in place.
    ///
    /// The shape must stay unchanged. The gradient is left untouched so an
    /// optimizer can read it before calling [`TensorValue::zero_grad`].
    pub fn set_value(&self, value: Tensor) -> Result<(), AutogradError> {
        let mut node = self.node.borrow_mut();
        if !node.requires_grad {
            return Err(AutogradError::NotAParameter);
        }
        if node.value.shape() != value.shape() {
            return Err(AutogradError::Tensor(
                mini_llm_tensor::TensorError::ShapeMismatch {
                    left: node.value.shape().to_vec(),
                    right: value.shape().to_vec(),
                },
            ));
        }
        node.value = value;
        Ok(())
    }

    // ---------------------------------------------------------------------
    // Graph plumbing
    // ---------------------------------------------------------------------

    fn accumulate_grad(&self, grad: &Tensor) -> Result<(), AutogradError> {
        let mut node = self.node.borrow_mut();
        if !node.requires_grad {
            return Ok(());
        }
        node.grad = Some(match node.grad.take() {
            Some(existing) => existing.add(grad)?,
            None => grad.clone(),
        });
        Ok(())
    }

    /// Runs reverse-mode differentiation from this (scalar) loss node.
    ///
    /// The seed gradient is one, so `loss` should be a collapsed scalar such as
    /// the output of [`TensorValue::mean`] or [`TensorValue::cross_entropy`].
    pub fn backward(&self) -> Result<(), AutogradError> {
        if !self.requires_grad() {
            return Err(AutogradError::NoGradientTracked);
        }

        let mut order = Vec::new();
        let mut visited = HashSet::new();
        collect_topological(&self.node, &mut order, &mut visited);

        // dLoss/dLoss = 1
        self.accumulate_grad(&Tensor::full(self.shape(), 1.0)?)?;

        for node in order.iter().rev() {
            let (grad, backward, parents) = {
                let node = node.borrow();
                (
                    node.grad.clone(),
                    node.backward.clone(),
                    node.parents.clone(),
                )
            };
            let (Some(grad), Some(backward)) = (grad, backward) else {
                continue;
            };
            let parent_grads = backward(&grad)?;
            for (parent, parent_grad) in parents.iter().zip(parent_grads.iter()) {
                parent.accumulate_grad(parent_grad)?;
            }
        }
        Ok(())
    }

    /// Clears every gradient in the graph rooted at this value.
    pub fn zero_grad(&self) {
        let mut order = Vec::new();
        let mut visited = HashSet::new();
        collect_topological(&self.node, &mut order, &mut visited);
        for node in order {
            node.borrow_mut().grad = None;
        }
    }

    /// Every learnable leaf reachable from this value, in topological order.
    pub fn parameters(&self) -> Vec<TensorValue> {
        let mut order = Vec::new();
        let mut visited = HashSet::new();
        collect_topological(&self.node, &mut order, &mut visited);
        order
            .into_iter()
            .filter(|node| {
                let node = node.borrow();
                node.requires_grad && node.backward.is_none()
            })
            .map(|node| TensorValue { node })
            .collect()
    }

    /// A human-readable one-line-per-node description of the graph.
    pub fn graph_summary(&self) -> Vec<String> {
        let mut order = Vec::new();
        let mut visited = HashSet::new();
        collect_topological(&self.node, &mut order, &mut visited);
        order
            .iter()
            .enumerate()
            .map(|(index, node)| {
                let node = node.borrow();
                format!(
                    "{index:>2}: {:<14} shape={:?} requires_grad={}",
                    node.op,
                    node.value.shape(),
                    node.requires_grad
                )
            })
            .collect()
    }

    // ---------------------------------------------------------------------
    // Operations
    // ---------------------------------------------------------------------

    fn binary(
        &self,
        other: &TensorValue,
        op: &'static str,
        value: Tensor,
        grads: impl Fn(&Tensor) -> Result<(Tensor, Tensor), AutogradError> + 'static,
    ) -> Result<TensorValue, AutogradError> {
        let requires_grad = self.requires_grad() || other.requires_grad();
        let backward = requires_grad.then(|| {
            Rc::new(move |grad: &Tensor| -> Result<Vec<Tensor>, AutogradError> {
                let (left, right) = grads(grad)?;
                Ok(vec![left, right])
            }) as Rc<BackwardFn>
        });
        Ok(TensorValue::from_parts(
            value,
            requires_grad,
            op,
            vec![self.clone(), other.clone()],
            backward,
        ))
    }

    fn unary(
        &self,
        op: &'static str,
        value: Tensor,
        grad_fn: impl Fn(&Tensor) -> Result<Tensor, AutogradError> + 'static,
    ) -> Result<TensorValue, AutogradError> {
        let requires_grad = self.requires_grad();
        let backward = requires_grad.then(|| {
            Rc::new(move |grad: &Tensor| -> Result<Vec<Tensor>, AutogradError> {
                Ok(vec![grad_fn(grad)?])
            }) as Rc<BackwardFn>
        });
        Ok(TensorValue::from_parts(
            value,
            requires_grad,
            op,
            vec![self.clone()],
            backward,
        ))
    }

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

    /// Transposes a rank-2 value.
    pub fn transpose(&self) -> Result<TensorValue, AutogradError> {
        let value = self.value().transpose_2d()?;
        self.unary("transpose", value, |g| Ok(g.transpose_2d()?))
    }

    pub fn relu(&self) -> Result<TensorValue, AutogradError> {
        let x = self.value();
        let value = x.relu()?;
        self.unary("relu", value, move |g| {
            let mask = x.map(|v| if v > 0.0 { 1.0 } else { 0.0 })?;
            Ok(g.mul(&mask)?)
        })
    }

    pub fn tanh(&self) -> Result<TensorValue, AutogradError> {
        let value = self.value().tanh()?;
        let saved = value.clone();
        self.unary("tanh", value, move |g| {
            // d/dx tanh(x) = 1 - tanh(x)^2
            let slope = saved.map(|v| 1.0 - v * v)?;
            Ok(g.mul(&slope)?)
        })
    }

    pub fn sigmoid(&self) -> Result<TensorValue, AutogradError> {
        let value = self.value().sigmoid()?;
        let saved = value.clone();
        self.unary("sigmoid", value, move |g| {
            // d/dx sigmoid(x) = sigmoid(x) * (1 - sigmoid(x))
            let slope = saved.map(|v| v * (1.0 - v))?;
            Ok(g.mul(&slope)?)
        })
    }

    /// Row-wise softmax of a rank-2 value.
    pub fn softmax_rows(&self) -> Result<TensorValue, AutogradError> {
        let value = self.value().softmax_rows()?;
        let saved = value.clone();
        self.unary("softmax_rows", value, move |g| {
            let shape = saved.shape().to_vec();
            let (rows, cols) = (shape[0], shape[1]);
            let (soft, upstream) = (saved.data(), g.data());
            let mut out = vec![0.0; soft.len()];
            for row in 0..rows {
                let start = row * cols;
                let dot: f32 = (0..cols)
                    .map(|c| upstream[start + c] * soft[start + c])
                    .sum();
                for col in 0..cols {
                    out[start + col] = soft[start + col] * (upstream[start + col] - dot);
                }
            }
            Ok(Tensor::new(out, shape)?)
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

    /// Cross-entropy between rank-2 logits (`[batch, vocab]`) and token ids.
    ///
    /// Returns the mean negative log-likelihood as a scalar (shape `[1]`).
    pub fn cross_entropy(&self, targets: &[usize]) -> Result<TensorValue, AutogradError> {
        let logits = self.value();
        if logits.rank() != 2 {
            return Err(AutogradError::Tensor(
                mini_llm_tensor::TensorError::ExpectedRank {
                    expected: 2,
                    actual: logits.rank(),
                },
            ));
        }
        let (rows, cols) = (logits.shape()[0], logits.shape()[1]);
        if targets.len() != rows {
            return Err(AutogradError::TargetCountMismatch {
                logits_rows: rows,
                targets: targets.len(),
            });
        }
        if let Some(index) = targets.iter().copied().find(|&index| index >= cols) {
            return Err(AutogradError::TargetOutOfRange {
                index,
                vocab_size: cols,
            });
        }

        let probabilities = logits.softmax_rows()?;
        let probability_data = probabilities.data().to_vec();
        let loss = -targets
            .iter()
            .enumerate()
            .map(|(row, &target)| probability_data[row * cols + target].ln())
            .sum::<f32>()
            / rows as f32;

        let value = Tensor::new(vec![loss], vec![1])?;
        let saved_targets = targets.to_vec();
        self.unary("cross_entropy", value, move |g| {
            let scale = g.data().first().copied().unwrap_or(1.0) / rows as f32;
            let mut grad = probability_data.to_vec();
            for (row, &target) in saved_targets.iter().enumerate() {
                grad[row * cols + target] -= 1.0;
            }
            for entry in grad.iter_mut() {
                *entry *= scale;
            }
            Ok(Tensor::new(grad, vec![rows, cols])?)
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

    /// Row-wise mean of a rank-2 value, kept as `[rows, 1]` for broadcasting.
    pub fn mean_rows_keepdim(&self) -> Result<TensorValue, AutogradError> {
        let source = self.value();
        if source.rank() != 2 {
            return Err(AutogradError::Tensor(
                mini_llm_tensor::TensorError::ExpectedRank {
                    expected: 2,
                    actual: source.rank(),
                },
            ));
        }
        let (rows, cols) = (source.shape()[0], source.shape()[1]);
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

fn collect_topological(
    node: &Rc<RefCell<Node>>,
    order: &mut Vec<Rc<RefCell<Node>>>,
    visited: &mut HashSet<usize>,
) {
    let key = Rc::as_ptr(node) as usize;
    if !visited.insert(key) {
        return;
    }
    let parents = node.borrow().parents.clone();
    for parent in &parents {
        collect_topological(&parent.node, order, visited);
    }
    order.push(Rc::clone(node));
}

impl fmt::Debug for TensorValue {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let node = self.node.borrow();
        f.debug_struct("TensorValue")
            .field("op", &node.op)
            .field("shape", &node.value.shape())
            .field("data", &node.value.data())
            .field("requires_grad", &node.requires_grad)
            .field("grad", &node.grad.as_ref().map(|g| g.data()))
            .finish()
    }
}

impl fmt::Display for TensorValue {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let node = self.node.borrow();
        write!(f, "{} {:?}", node.op, node.value.shape())
    }
}
