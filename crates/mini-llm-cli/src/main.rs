//! mini-llm command-line entry point.
//!
//! `main` only dispatches; every demo lives in [`commands`].

mod commands;

use commands::{
    autograd, basics, bench, classify, dpo, generate, grpo, lora, moe, regression, sft, trace, train, xor,
};

fn main() {
    let arguments: Vec<String> = std::env::args().skip(1).collect();
    let command = arguments.first().map(String::as_str).unwrap_or("demo");
    let rest = arguments.get(1..).unwrap_or(&[]);

    match command {
        "demo" => basics::attention_demo(),
        "bench" => bench::run(),
        "inspect-config" => basics::inspect_config(),
        "autograd" => autograd::run(),
        "train-line" => regression::run(),
        "xor" => xor::run(),
        "classify" => classify::run(),
        "train" => train::run(rest),
        "generate" => generate::run(rest),
        "moe" => moe::run(),
        "sft" => sft::run(rest),
        "dpo" => dpo::run(),
        "grpo" => grpo::run(),
        "trace" => trace::run(rest),
        "lora" => lora::run(),
        "help" | "--help" | "-h" => usage(),
        other => {
            eprintln!("Unknown command: {other}");
            usage();
            std::process::exit(2);
        }
    }
}

fn usage() {
    println!("Usage: cargo run -p mini-llm-cli -- <command> [args]");
    println!();
    println!("Commands:");
    println!("  demo                 scaled dot-product attention walkthrough");
    println!("  bench                kernel + training micro-benchmarks (use --release)");
    println!("  inspect-config       validate and print the default model configuration");
    println!("  autograd             computation graph + gradients for y = wx + b");
    println!("  train-line           fit y = 2x - 1 with manual SGD");
    println!("  xor                  train a small MLP on XOR (Linear + Tanh + AdamW)");
    println!("  classify             train a 2D two-blob classifier with softmax + cross-entropy");
    println!("  train [steps]        train Tiny GPT on the built-in corpus");
    println!("  generate [prompt] [tokens] [temp]");
    println!("                       sample text from the saved checkpoint (KV cache)");
    println!("  moe                  mixture-of-experts routing demo");
    println!("  sft [steps]          supervised fine-tuning with prompt loss masking");
    println!("  lora                  fine-tune a frozen classifier with LoRA adapters");
    println!("  dpo                   preference optimization on a toy policy");
    println!("  grpo                  group relative policy optimization on a bandit");
    println!("  trace                 emit training events + capture attention to traces/");
    println!("  help                 show this message");
    println!();
    println!("checkpoint: {}", commands::CHECKPOINT_PATH);
}
