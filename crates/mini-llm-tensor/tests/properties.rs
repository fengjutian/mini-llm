use mini_llm_tensor::Tensor;
use proptest::prelude::*;

proptest! {
    #[test]
    fn adding_a_broadcast_bias_matches_scalar_definition(
        rows in 1usize..16,
        cols in 1usize..16,
        values in prop::collection::vec(-100.0f32..100.0, 1..256),
        bias in prop::collection::vec(-100.0f32..100.0, 1..16),
    ) {
        prop_assume!(values.len() >= rows * cols);
        prop_assume!(bias.len() >= cols);
        let values = values[..rows * cols].to_vec();
        let bias = bias[..cols].to_vec();
        let input = Tensor::new(values.clone(), vec![rows, cols]).unwrap();
        let bias_tensor = Tensor::new(bias.clone(), vec![cols]).unwrap();
        let output = input.add(&bias_tensor).unwrap();
        for row in 0..rows {
            for col in 0..cols {
                prop_assert_eq!(output.data()[row * cols + col], values[row * cols + col] + bias[col]);
            }
        }
    }

    #[test]
    fn softmax_rows_are_finite_and_normalized(
        rows in 1usize..12,
        cols in 1usize..32,
        values in prop::collection::vec(-1_000.0f32..1_000.0, 1..384),
    ) {
        prop_assume!(values.len() >= rows * cols);
        let input = Tensor::new(values[..rows * cols].to_vec(), vec![rows, cols]).unwrap();
        let output = input.softmax_rows().unwrap();
        for row in output.data().chunks(cols) {
            prop_assert!(row.iter().all(|value| value.is_finite() && *value >= 0.0));
            prop_assert!((row.iter().sum::<f32>() - 1.0).abs() < 1e-4);
        }
    }
}
