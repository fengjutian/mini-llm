use mini_llm_model::{ModelConfig, TinyGpt};
use mini_llm_nn::Rng;
use mini_llm_train::{load_safetensors, save_safetensors};

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
