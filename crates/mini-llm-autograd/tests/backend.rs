//! Scoped backend selection: default is CPU, `with_backend` routes kernels and
//! restores the previous selection afterwards.

use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::Arc;

use mini_llm_autograd::backend::{is_accelerated, with_backend};
use mini_llm_autograd::TensorValue;
use mini_llm_backend::{Backend, BackendError, CpuBackend};
use mini_llm_tensor::Tensor;

#[derive(Default)]
struct CountingBackend {
    matmuls: AtomicUsize,
    softmaxes: AtomicUsize,
}

impl Backend for CountingBackend {
    fn matmul(&self, left: &Tensor, right: &Tensor) -> Result<Tensor, BackendError> {
        self.matmuls.fetch_add(1, Ordering::Relaxed);
        Ok(left.matmul(right)?)
    }

    fn softmax_rows(&self, input: &Tensor) -> Result<Tensor, BackendError> {
        self.softmaxes.fetch_add(1, Ordering::Relaxed);
        Ok(input.softmax_rows()?)
    }
}

#[test]
fn scoped_backend_receives_kernels_and_is_restored() {
    assert!(!is_accelerated());

    let counter = Arc::new(CountingBackend::default());
    let left = TensorValue::constant(vec![1.0, 2.0, 3.0, 4.0], vec![2, 2]).unwrap();
    let right = TensorValue::parameter(vec![0.5, 0.0, 0.0, 0.5], vec![2, 2]).unwrap();

    let product = with_backend(counter.clone(), || {
        assert!(is_accelerated());
        left.matmul(&right).unwrap()
    });

    assert!(!is_accelerated(), "scope must restore the CPU default");
    assert_eq!(counter.matmuls.load(Ordering::Relaxed), 1);
    assert_eq!(product.data(), &[0.5, 1.0, 1.5, 2.0]);
}

#[test]
fn nested_scopes_restore_the_outer_backend() {
    let counter = Arc::new(CountingBackend::default());
    let logits = TensorValue::constant(vec![1.0, 2.0, 3.0, 4.0], vec![2, 2]).unwrap();

    with_backend(Arc::new(CpuBackend), || {
        with_backend(counter.clone(), || {
            let _ = logits.softmax_rows().unwrap();
        });
        assert!(is_accelerated(), "outer backend is active again");
    });

    assert!(!is_accelerated());
    assert_eq!(counter.softmaxes.load(Ordering::Relaxed), 1);
}

#[test]
fn backend_errors_surface_as_autograd_errors() {
    struct FailingBackend;

    impl Backend for FailingBackend {
        fn matmul(&self, _left: &Tensor, _right: &Tensor) -> Result<Tensor, BackendError> {
            Err(BackendError::Backend("boom".into()))
        }
    }

    let left = TensorValue::constant(vec![1.0], vec![1, 1]).unwrap();
    let right = TensorValue::constant(vec![2.0], vec![1, 1]).unwrap();
    let error = with_backend(Arc::new(FailingBackend), || {
        left.matmul(&right).unwrap_err()
    });
    assert!(error.to_string().contains("boom"));
}
