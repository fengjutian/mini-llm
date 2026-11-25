//! Scoped inference mode: ops inside `with_no_grad` are not recorded.

use mini_llm_autograd::no_grad::{is_enabled, with_no_grad};
use mini_llm_autograd::{AutogradError, TensorValue};

#[test]
fn no_grad_region_skips_graph_recording() {
    assert!(!is_enabled());

    let weight = TensorValue::parameter(vec![1.0, 2.0], vec![1, 2]).unwrap();
    let input = TensorValue::constant(vec![3.0, 4.0], vec![1, 2]).unwrap();

    let product = with_no_grad(|| {
        assert!(is_enabled());
        input.mul(&weight).unwrap()
    });
    assert!(!is_enabled());
    assert!(!product.requires_grad());
    assert_eq!(product.data(), &[3.0, 8.0]);
    assert_eq!(
        product.sum().unwrap().backward(),
        Err(AutogradError::NoGradientTracked)
    );

    // Outside the region graph recording resumes.
    let tracked = input.mul(&weight).unwrap();
    assert!(tracked.requires_grad());
}

#[test]
fn nested_no_grad_regions_restore_the_outer_state() {
    with_no_grad(|| {
        with_no_grad(|| assert!(is_enabled()));
        assert!(is_enabled(), "outer region stays active");
    });
    assert!(!is_enabled());

    let guard = mini_llm_autograd::no_grad::guard();
    assert!(is_enabled());
    drop(guard);
    assert!(!is_enabled());
}
