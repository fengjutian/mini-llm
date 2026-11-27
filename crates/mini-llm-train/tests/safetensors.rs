use mini_llm_model::{ModelConfig, TinyGpt};
use mini_llm_nn::Rng;
use mini_llm_train::{load_safetensors, save_safetensors};
use std::collections::HashSet;

#[test]
fn safetensors_roundtrip_preserves_every_parameter() {
    let config = ModelConfig {
        vocab_size: 32,
        hidden_size: 8,
        num_layers: 1,
        num_heads: 2,
        num_kv_heads: 1,
        max_seq_len: 8,
        ..ModelConfig::modern(32)
    };
    let source = TinyGpt::new(config.clone(), &mut Rng::new(7)).unwrap();
    let restored = TinyGpt::new(config, &mut Rng::new(99)).unwrap();
    let directory = tempfile::tempdir().unwrap();
    let path = directory.path().join("model.safetensors");

    save_safetensors(&path, &source).unwrap();
    load_safetensors(&path, &restored).unwrap();

    for (expected, actual) in source.parameters().iter().zip(restored.parameters()) {
        assert_eq!(expected.value().shape(), actual.value().shape());
        assert_eq!(expected.value().data(), actual.value().data());
    }
}

#[test]
fn parameter_names_are_unique_stable_and_human_readable() {
    let model = TinyGpt::new(ModelConfig::modern(32), &mut Rng::new(1)).unwrap();
    let names = model.parameter_names();
    assert_eq!(names.len(), model.parameters().len());
    assert_eq!(names.len(), names.iter().collect::<HashSet<_>>().len());
    assert_eq!(names.first().unwrap(), "model.embed_tokens.weight");
    assert_eq!(names.last().unwrap(), "lm_head.weight");
    assert!(names
        .iter()
        .any(|name| name.ends_with("self_attn.q_proj.weight")));
}

#[test]
fn parameter_names_cover_tiny_and_moe_architectures() {
    let tiny = TinyGpt::new(ModelConfig::tiny(16), &mut Rng::new(3)).unwrap();
    let mut moe_config = ModelConfig::modern(32);
    moe_config.num_experts = 2;
    let moe = TinyGpt::new(moe_config, &mut Rng::new(4)).unwrap();
    for model in [tiny, moe] {
        let names = model.parameter_names();
        assert_eq!(names.len(), model.parameters().len());
        assert_eq!(names.len(), names.iter().collect::<HashSet<_>>().len());
    }
}
