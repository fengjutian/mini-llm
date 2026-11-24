//! Computation-graph internals: nodes, backward closures and traversal.

use std::cell::RefCell;
use std::rc::Rc;

use mini_llm_tensor::Tensor;

use crate::error::AutogradError;
use crate::value::TensorValue;

/// Maps an upstream gradient to one gradient per parent node.
pub(crate) type BackwardFn = dyn Fn(&Tensor) -> Result<Vec<Tensor>, AutogradError>;

pub(crate) struct Node {
    pub(crate) value: Tensor,
    pub(crate) requires_grad: bool,
    pub(crate) grad: Option<Tensor>,
    pub(crate) op: &'static str,
    pub(crate) parents: Vec<TensorValue>,
    pub(crate) backward: Option<Rc<BackwardFn>>,
}

/// Depth-first post-order traversal: parents before children.
pub(crate) fn collect_topological(
    node: &Rc<RefCell<Node>>,
    order: &mut Vec<Rc<RefCell<Node>>>,
    visited: &mut std::collections::HashSet<usize>,
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

/// Returns `(rows, cols)` for a rank-2 tensor, or an `ExpectedRank` error.
pub(crate) fn matrix_shape(tensor: &Tensor) -> Result<(usize, usize), AutogradError> {
    if tensor.rank() != 2 {
        return Err(AutogradError::Tensor(
            mini_llm_tensor::TensorError::ExpectedRank {
                expected: 2,
                actual: tensor.rank(),
            },
        ));
    }
    Ok((tensor.shape()[0], tensor.shape()[1]))
}
