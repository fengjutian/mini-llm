//! Small demos: the raw attention primitive and the default configuration.

use mini_llm_model::{scaled_dot_product_attention, ModelConfig};
use mini_llm_tensor::Tensor;

/// Runs scaled dot-product attention on a hand-written example.
pub fn attention_demo() {
    let q = Tensor::from_2d(2, 2, vec![1.0, 0.0, 0.0, 1.0]).expect("valid query tensor");
    let k = Tensor::from_2d(2, 2, vec![1.0, 0.0, 0.0, 1.0]).expect("valid key tensor");
    let v = Tensor::from_2d(2, 2, vec![10.0, 0.0, 0.0, 10.0]).expect("valid value tensor");

    let (output, weights) = scaled_dot_product_attention(&q, &k, &v).expect("attention should run");

    println!("mini-llm CPU attention demo");
    println!("attention weights: {:?}", weights.data());
    println!("output shape: {:?}", output.shape());
    println!("output: {:?}", output.data());
}

/// Prints and validates the default [`ModelConfig`].
pub fn inspect_config() {
    let config = ModelConfig::default();
    match config.validate() {
        Ok(()) => {
            println!("Model configuration is valid:");
            println!("  vocab_size:    {}", config.vocab_size);
            println!("  hidden_size:   {}", config.hidden_size);
            println!("  num_layers:    {}", config.num_layers);
            println!("  num_heads:     {}", config.num_heads);
            println!("  num_kv_heads:  {}", config.num_kv_heads);
            println!("  max_seq_len:   {}", config.max_seq_len);
            println!("  norm:          {:?}", config.norm);
            println!("  position:      {:?}", config.position);
            println!("  feed_forward:  {:?}", config.feed_forward);
            println!("  num_experts:   {}", config.num_experts);
        }
        Err(err) => {
            eprintln!("Invalid model configuration: {err}");
            std::process::exit(1);
        }
    }
}
