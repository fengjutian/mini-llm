//! End-to-end tests for the Tiny GPT model.

use mini_llm_model::{
    FeedForwardKind, ModelConfig, ModelError, NormKind, PositionKind, Sampling, TinyGpt,
};
use mini_llm_nn::Rng;

fn tiny_config() -> ModelConfig {
    ModelConfig {
        vocab_size: 8,
        hidden_size: 16,
        num_layers: 1,
        num_heads: 2,
        max_seq_len: 8,
        num_kv_heads: 2,
        ..ModelConfig::default()
    }
}

/// Modern stack: RMSNorm + RoPE + SwiGLU + grouped-query attention.
fn modern_config() -> ModelConfig {
    ModelConfig {
        vocab_size: 8,
        hidden_size: 16,
        num_layers: 2,
        num_heads: 4,
        max_seq_len: 8,
        num_kv_heads: 2,
        norm: NormKind::RmsNorm,
        position: PositionKind::Rope,
        feed_forward: FeedForwardKind::SwiGlu,
        num_experts: 1,
    }
}

fn model() -> TinyGpt {
    let mut rng = Rng::new(11);
    TinyGpt::new(tiny_config(), &mut rng).unwrap()
}

fn modern_model() -> TinyGpt {
    let mut rng = Rng::new(13);
    TinyGpt::new(modern_config(), &mut rng).unwrap()
}

#[test]
fn forward_produces_logits_for_every_position() {
    let model = model();
    let tokens = [1, 2, 3, 4, 5, 6, 7, 0, 1, 2, 3, 4];
    let logits = model.forward(&tokens, 2).unwrap();
    assert_eq!(logits.shape(), &[12, 8]);
    assert!(logits.data().iter().all(|value| value.is_finite()));
}

#[test]
fn backward_reaches_every_parameter() {
    let model = model();
    let tokens = [1, 2, 3, 4, 1, 2, 3, 4];
    let targets = [2usize, 3, 4, 1, 2, 3, 4, 1];
    let loss = model
        .forward(&tokens, 1)
        .unwrap()
        .cross_entropy(&targets)
        .unwrap();
    loss.backward().unwrap();

    let parameters = model.parameters();
    assert!(!parameters.is_empty());
    for (index, parameter) in parameters.iter().enumerate() {
        let grad = parameter
            .grad()
            .unwrap_or_else(|| panic!("parameter {index} has no gradient"));
        assert!(
            grad.data().iter().all(|value| value.is_finite()),
            "parameter {index} has a non-finite gradient"
        );
    }
}

#[test]
fn causal_mask_hides_future_tokens() {
    let model = model();
    let baseline = [1, 2, 3, 4, 5, 6];
    let mut altered = baseline;
    altered[5] = 7;

    let first = model.forward(&baseline, 1).unwrap();
    let second = model.forward(&altered, 1).unwrap();

    // Logits at earlier positions must not depend on later tokens.
    let vocab = tiny_config().vocab_size;
    for position in 0..4 {
        let left = &first.data()[position * vocab..(position + 1) * vocab];
        let right = &second.data()[position * vocab..(position + 1) * vocab];
        for (a, b) in left.iter().zip(right.iter()) {
            assert!(
                (a - b).abs() < 1e-5,
                "position {position} leaked future tokens"
            );
        }
    }
}

#[test]
fn samples_do_not_cross_batch_boundaries() {
    let model = model();
    let sequence = [1, 2, 3];
    // Same sample twice in one batch must match a standalone forward pass.
    let tokens = [1, 2, 3, 1, 2, 3];
    let single = model.forward(&sequence, 1).unwrap();
    let batched = model.forward(&tokens, 2).unwrap();
    let vocab = tiny_config().vocab_size;
    for row in 0..6 {
        for column in 0..vocab {
            let a = single.data()[(row % 3) * vocab + column];
            let b = batched.data()[row * vocab + column];
            assert!((a - b).abs() < 1e-5, "batch boundary leaked at row {row}");
        }
    }
}

#[test]
fn generation_appends_the_requested_number_of_tokens() {
    let model = model();
    let mut rng = Rng::new(5);
    let prompt = [1usize, 2, 3];
    let generated = model.generate(&prompt, 5, 0.8, &mut rng).unwrap();
    assert_eq!(generated.len(), prompt.len() + 5);
    assert_eq!(&generated[..prompt.len()], &prompt);
    assert!(generated.iter().all(|&token| token < 8));
}

#[test]
fn greedy_generation_is_deterministic() {
    let model = model();
    let mut first = Rng::new(1);
    let mut second = Rng::new(1);
    let a = model.generate(&[1], 4, 0.0, &mut first).unwrap();
    let b = model.generate(&[1], 4, 0.0, &mut second).unwrap();
    assert_eq!(a, b);
}

#[test]
fn rejects_invalid_inputs() {
    let model = model();
    assert!(matches!(
        model.forward(&[1, 2], 3),
        Err(ModelError::BatchMismatch { .. })
    ));
    assert!(matches!(
        model.forward(&[0; 9], 1),
        Err(ModelError::SequenceTooLong { .. })
    ));
    assert!(matches!(
        model.forward(&[9, 1], 1),
        Err(ModelError::TokenOutOfRange { .. })
    ));
    let mut rng = Rng::new(3);
    assert!(matches!(
        model.generate(&[], 1, 0.0, &mut rng),
        Err(ModelError::EmptyPrompt)
    ));
}

#[test]
fn parameter_count_matches_the_configuration() {
    let model = model();
    assert!(model.num_parameters() > 1_000);
    assert_eq!(model.config().hidden_size, 16);
}

#[test]
fn modern_stack_trains_end_to_end() {
    let model = modern_model();
    let tokens = [1, 2, 3, 4, 5, 6, 7, 0];
    let targets = [2usize, 3, 4, 5, 6, 7, 0, 1];
    let loss = model
        .forward(&tokens, 1)
        .unwrap()
        .cross_entropy(&targets)
        .unwrap();
    loss.backward().unwrap();

    for parameter in model.parameters() {
        let grad = parameter.grad().expect("every parameter gets a gradient");
        assert!(grad.data().iter().all(|value| value.is_finite()));
    }
}

#[test]
fn cached_generation_matches_uncached_generation() {
    for model in [model(), modern_model()] {
        let prompt = [1usize, 2, 3];
        let mut cached_rng = Rng::new(2);
        let mut plain_rng = Rng::new(2);
        let cached = model.generate(&prompt, 4, 0.0, &mut cached_rng).unwrap();
        let plain = model
            .generate_uncached(&prompt, 4, 0.0, &mut plain_rng)
            .unwrap();
        assert_eq!(cached, plain);
    }
}

#[test]
fn grouped_query_attention_uses_fewer_parameters() {
    let mut dense = modern_config();
    dense.num_kv_heads = dense.num_heads;
    let mut grouped = modern_config();
    grouped.num_kv_heads = 2;

    let mut rng = Rng::new(1);
    let dense = TinyGpt::new(dense, &mut rng).unwrap();
    let mut rng = Rng::new(1);
    let grouped = TinyGpt::new(grouped, &mut rng).unwrap();
    assert!(grouped.num_parameters() < dense.num_parameters());
}

#[test]
fn mixture_of_experts_configuration_trains() {
    let mut config = modern_config();
    config.hidden_size = 8;
    config.num_heads = 2;
    config.num_kv_heads = 1;
    config.num_experts = 2;
    let mut rng = Rng::new(5);
    let model = TinyGpt::new(config, &mut rng).unwrap();

    let tokens = [1, 2, 3, 4];
    let targets = [2usize, 3, 4, 5];
    let loss = model
        .forward(&tokens, 1)
        .unwrap()
        .cross_entropy(&targets)
        .unwrap();
    loss.backward().unwrap();

    for parameter in model.parameters() {
        assert!(parameter.grad().is_some());
    }
}

#[test]
fn rope_generation_extends_past_the_training_context() {
    let model = modern_model();
    let mut rng = Rng::new(7);
    // Context is 8 tokens; RoPE keeps the position index growing.
    let generated = model.generate(&[1, 2], 20, 0.0, &mut rng).unwrap();
    assert_eq!(generated.len(), 22);
}

#[test]
fn learned_positions_stop_at_the_context_limit() {
    let model = model();
    let mut rng = Rng::new(7);
    let generated = model.generate(&[1, 2], 20, 0.0, &mut rng).unwrap();
    assert_eq!(generated.len(), 8);
}

#[test]
fn invalid_configurations_are_rejected() {
    let mut config = modern_config();
    config.num_kv_heads = 3;
    let mut rng = Rng::new(1);
    assert!(TinyGpt::new(config, &mut rng).is_err());

    let mut config = modern_config();
    config.num_kv_heads = 0;
    let mut rng = Rng::new(1);
    assert!(TinyGpt::new(config, &mut rng).is_err());
}

#[test]
fn traced_forward_records_one_matrix_per_layer_and_head() {
    let model = modern_model();
    let tokens = [1, 2, 3, 4, 5, 6, 7, 0];
    let (logits, records) = model.forward_traced(&tokens, 1).unwrap();

    assert_eq!(logits.shape(), &[8, 8]);
    let expected = model.config().num_layers * model.config().num_heads;
    assert_eq!(records.len(), expected);

    for record in &records {
        assert_eq!(record.weights.shape(), &[8, 8]);
        for row in record.weights.data().chunks(8) {
            let sum: f32 = row.iter().sum();
            assert!((sum - 1.0).abs() < 1e-4, "row does not sum to one: {sum}");
        }
    }
    // Layer-major order: layer 0 heads come before layer 1 heads.
    let heads_per_layer = model.config().num_heads;
    assert!(records[..heads_per_layer]
        .iter()
        .all(|record| record.layer == 0));
    assert!(records[heads_per_layer..]
        .iter()
        .all(|record| record.layer == 1));
}

#[test]
fn kv_cache_stores_one_row_per_token_with_grouped_queries() {
    let model = modern_model();
    // 4 query heads share 2 key/value heads; the cache must still grow by one
    // row per token, not by one row per query head.
    let mut caches = model.empty_caches();
    for (position, &token) in [1usize, 2, 3, 4].iter().enumerate() {
        model.step_token(token, position, &mut caches).unwrap();
    }
    for cache in &caches {
        assert_eq!(cache.attention.len(), 4);
    }
}

#[test]
fn cached_mask_stays_correct_across_shapes() {
    let model = model();
    let first = model.forward(&[1, 2, 3, 4], 1).unwrap().data();
    let second = model.forward(&[1, 2, 3, 4], 1).unwrap().data();
    assert_eq!(first, second);

    // A different (batch, seq) must rebuild the mask and still run.
    let batched = model.forward(&[1, 2, 3, 4, 5, 6], 2).unwrap();
    assert_eq!(batched.shape(), &[6, 8]);
}

#[test]
fn top_k_one_and_top_p_zero_match_greedy() {
    let model = modern_model();
    let prompt = [1usize, 2, 3];

    let greedy = model.generate(&prompt, 10, 0.0, &mut Rng::new(5)).unwrap();
    let top_k = model
        .generate_with(
            &prompt,
            10,
            Sampling::with_temperature(1.0).top_k(1),
            &mut Rng::new(5),
        )
        .unwrap();
    let top_p = model
        .generate_with(
            &prompt,
            10,
            Sampling::with_temperature(1.0).top_p(0.0),
            &mut Rng::new(5),
        )
        .unwrap();

    assert_eq!(greedy, top_k);
    assert_eq!(greedy, top_p);
}

#[test]
fn plain_temperature_sampling_is_unchanged() {
    let model = modern_model();
    let prompt = [1usize, 2, 3];
    let legacy = model.generate(&prompt, 10, 0.8, &mut Rng::new(7)).unwrap();
    let explicit = model
        .generate_with(
            &prompt,
            10,
            Sampling::with_temperature(0.8),
            &mut Rng::new(7),
        )
        .unwrap();
    assert_eq!(legacy, explicit);
}

#[test]
fn top_p_keeps_only_the_probability_head() {
    let model = modern_model();
    let prompt = [1usize, 2, 3];
    // p = 0 means "keep the single most likely token".
    let filtered = model
        .generate_with(
            &prompt,
            6,
            Sampling::with_temperature(0.5).top_p(0.0),
            &mut Rng::new(3),
        )
        .unwrap();
    let greedy = model.generate(&prompt, 6, 0.0, &mut Rng::new(3)).unwrap();
    assert_eq!(filtered, greedy);
}

#[test]
fn batch_generation_matches_individual_greedy_runs() {
    let model = modern_model();
    let prompts = vec![vec![1usize, 2, 3], vec![1, 2, 3]];
    let singles: Vec<Vec<usize>> = prompts
        .iter()
        .map(|prompt| model.generate(prompt, 5, 0.0, &mut Rng::new(9)).unwrap())
        .collect();
    let batch = model
        .generate_batch(&prompts, 5, Sampling::greedy(), &mut Rng::new(9))
        .unwrap();
    assert_eq!(singles, batch);
}

#[test]
fn batch_generation_pads_shorter_prompts_without_changing_them() {
    let model = modern_model();
    let short = vec![1usize];
    let single = model.generate(&short, 4, 0.0, &mut Rng::new(2)).unwrap();
    let batch = model
        .generate_batch(
            &[short, vec![1, 2, 3, 4]],
            4,
            Sampling::greedy(),
            &mut Rng::new(2),
        )
        .unwrap();

    assert_eq!(batch[0], single, "padding must not change the short prompt");
    assert_eq!(batch[1].len(), 4 + 4);
    assert!(batch[1].starts_with(&[1, 2, 3, 4]));
}

#[test]
fn batch_generation_rejects_empty_input() {
    let model = modern_model();
    assert!(matches!(
        model.generate_batch(&[], 4, Sampling::greedy(), &mut Rng::new(1)),
        Err(ModelError::EmptyPrompt)
    ));
    assert!(matches!(
        model.generate_batch(&[vec![]], 4, Sampling::greedy(), &mut Rng::new(1)),
        Err(ModelError::EmptyPrompt)
    ));
}
", batch[1]);
}