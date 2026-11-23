use mini_llm_model::{scaled_dot_product_attention, ModelConfig};
use mini_llm_tensor::Tensor;

fn main() {
    let command = std::env::args().nth(1).unwrap_or_else(|| "demo".to_string());

    match command.as_str() {
        "demo" => demo(),
        "inspect-config" => inspect_config(),
        other => {
            eprintln!("Unknown command: {other}");
            eprintln!("Usage: cargo run -p mini-llm-cli -- [demo|inspect-config]");
            std::process::exit(2);
        }
    }
}

fn demo() {
    let q = Tensor::from_2d(2, 2, vec![1.0, 0.0, 0.0, 1.0])
        .expect("valid query tensor");
    let k = Tensor::from_2d(2, 2, vec![1.0, 0.0, 0.0, 1.0])
        .expect("valid key tensor");
    let v = Tensor::from_2d(2, 2, vec![10.0, 0.0, 0.0, 10.0])
        .expect("valid value tensor");

    let (output, weights) = scaled_dot_product_attention(&q, &k, &v)
        .expect("attention should run");

    println!("mini-llm CPU attention demo");
    println!("attention weights: {:?}", weights.data());
    println!("output shape: {:?}", output.shape());
    println!("output: {:?}", output.data());
}

fn inspect_config() {
    let config = ModelConfig::default();
    match config.validate() {
        Ok(()) => {
            println!("Model configuration is valid:");
            println!("  vocab_size:   {}", config.vocab_size);
            println!("  hidden_size:  {}", config.hidden_size);
            println!("  num_layers:   {}", config.num_layers);
            println!("  num_heads:    {}", config.num_heads);
            println!("  max_seq_len:  {}", config.max_seq_len);
        }
        Err(err) => {
            eprintln!("Invalid model configuration: {err}");
            std::process::exit(1);
        }
    }
}
