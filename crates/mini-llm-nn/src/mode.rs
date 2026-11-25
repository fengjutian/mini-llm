//! Scoped evaluation mode.
//!
//! Training is the default; [`with_eval`] disables stochastic layers such as
//! dropout for validation, evaluation and generation. The flag is thread-local
//! and restored on drop, so training and evaluation can be interleaved.

use std::cell::Cell;

thread_local! {
    static EVAL: Cell<bool> = const { Cell::new(false) };
}

/// True while stochastic layers should behave deterministically.
pub fn is_eval() -> bool {
    EVAL.with(Cell::get)
}

/// Runs `body` in evaluation mode.
pub fn with_eval<R>(body: impl FnOnce() -> R) -> R {
    let _guard = guard();
    body()
}

/// Enables evaluation mode until the guard is dropped.
pub fn guard() -> EvalGuard {
    EvalGuard(EVAL.with(|flag| flag.replace(true)))
}

/// Restores the previous mode on drop.
pub struct EvalGuard(bool);

impl Drop for EvalGuard {
    fn drop(&mut self) {
        EVAL.with(|flag| flag.set(self.0));
    }
}
