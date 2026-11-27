//! Scoped backend selection for the differentiable ops.
//!
//! The autograd engine is pure CPU by default. A caller that wants accelerated
//! kernels wraps a region in [`with_backend`], and matrix multiplications and
//! softmax inside that region go through the chosen backend instead of the
//! reference tensor implementation. The selection is thread-local and scoped,
//! so nested calls restore the previous backend and no global state leaks
//! between runs.
//!
//! ```no_run
//! use std::sync::Arc;
//! use mini_llm_backend::CpuBackend;
//! use mini_llm_autograd::backend::with_backend;
//!
//! with_backend(Arc::new(CpuBackend), || {
//!     // ops here use the explicit backend
//! });
//! ```

use std::cell::RefCell;
use std::sync::{Arc, OnceLock};

use mini_llm_backend::{Backend, CpuBackend};
use mini_llm_tensor::Tensor;

use crate::error::AutogradError;

type SharedBackend = Arc<dyn Backend + Send + Sync>;

thread_local! {
    static ACTIVE: RefCell<Option<SharedBackend>> = const { RefCell::new(None) };
}

fn cpu_backend() -> SharedBackend {
    static CPU: OnceLock<SharedBackend> = OnceLock::new();
    CPU.get_or_init(|| Arc::new(CpuBackend)).clone()
}

/// The backend used by ops on the current thread.
pub fn current() -> SharedBackend {
    ACTIVE
        .with(|slot| slot.borrow().clone())
        .unwrap_or_else(cpu_backend)
}

/// Runs `body` with `backend` active on the current thread.
///
/// The previous selection is restored even if `body` panics.
pub fn with_backend<R>(backend: SharedBackend, body: impl FnOnce() -> R) -> R {
    struct Restore(Option<SharedBackend>);

    impl Drop for Restore {
        fn drop(&mut self) {
            ACTIVE.with(|slot| *slot.borrow_mut() = self.0.take());
        }
    }

    let previous = ACTIVE.with(|slot| slot.borrow_mut().replace(backend));
    let _restore = Restore(previous);
    body()
}

/// True when a backend other than the default CPU one is active.
pub fn is_accelerated() -> bool {
    ACTIVE.with(|slot| slot.borrow().is_some())
}

/// Multiplies through the active backend, mapping its error type.
///
/// Both the forward pass and the backward closures call this, so a scope that
/// covers `forward + backward` accelerates both directions.
pub fn matmul(left: &Tensor, right: &Tensor) -> Result<Tensor, AutogradError> {
    current()
        .matmul(left, right)
        .map_err(|error| AutogradError::Backend(error.to_string()))
}

pub fn matmul_add_bias(
    left: &Tensor,
    right: &Tensor,
    bias: &Tensor,
) -> Result<Tensor, AutogradError> {
    current()
        .matmul_add_bias(left, right, bias)
        .map_err(|error| AutogradError::Backend(error.to_string()))
}
