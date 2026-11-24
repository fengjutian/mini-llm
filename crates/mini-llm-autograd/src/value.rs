//! The [`TensorValue`] handle and the graph plumbing around it.
//!
//! Operations live in [`crate::ops`]; this module owns node creation, gradient
//! access and the reverse pass.

use std::cell::RefCell;
use std::collections::HashSet;
use std::fmt;
use std::rc::Rc;

use mini_llm_tensor::Tensor;

use crate::error::AutogradError;
use crate::graph::{collect_topological, BackwardFn, Node};

/// A tensor participating in the computation graph.
///
/// Cloning a `TensorValue` is cheap: clones share the same graph node, which is
/// what makes gradient accumulation across shared sub-expressions work.
#[derive(Clone)]
pub struct TensorValue {
    pub(crate) node: Rc<RefCell<Node>>,
}

impl TensorValue {
    pub(crate) fn from_parts(
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
    // Node builders used by the operations
    // ---------------------------------------------------------------------

    pub(crate) fn binary(
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

    pub(crate) fn unary(
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
