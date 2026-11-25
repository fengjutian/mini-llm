//! Scoped inference mode: operations run without recording a graph.
//!
//! ```
//! use mini_llm_autograd::{no_grad, TensorValue};
//!
//! let x = TensorValue::constant(vec![1.0, 2.0], vec![1, 2]).unwrap();
//! let y = no_grad::with_no_grad(|| x.scale(2.0)).unwrap();
//! assert!(!y.requires_grad());
//! assert_eq!(y.data(), &[2.0, 4.0]);
//! ```
//!
//! The flag is thread-local and restored when the closure returns (even on
//! panic), so a training loop can mix inference and training safely.

use std::cell::Cell;

thread_local! {
    static ENABLED: Cell<bool> = const { Cell::new(false) };
}

/// True inside a [`with_no_grad`] region on this thread.
pub fn is_enabled() -> bool {
    ENABLED.with(Cell::get)
}

/// Runs `body` without recording gradients.
pub fn with_no_grad<R>(body: impl FnOnce() -> R) -> R {
    let _guard = guard();
    body()
}

/// Enables no-grad mode until the returned guard is dropped.
///
/// Useful at the top of an inference function that has several exits.
pub fn guard() -> NoGradGuard {
    NoGradGuard(ENABLED.with(|flag| flag.replace(true)))
}

/// Restores the previous no-grad state on drop.
pub struct NoGradGuard(bool);

impl Drop for NoGradGuard {
    fn drop(&mut self) {
        ENABLED.with(|flag| flag.set(self.0));
    }
}
