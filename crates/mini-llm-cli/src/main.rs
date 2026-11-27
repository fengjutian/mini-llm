//! mini-llm command-line entry point.
//!
//! `main` only parses the optional `--backend` flag and dispatches; every demo
//! lives in [`commands`].

mod commands;

use commands::{
    autograd, basics, bench, classify, dpo, eval, generate, grpo, lora, moe, regression, sft,
    trace, train, xor,
};

/// Which kernel backend the process should use.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum BackendChoice {
    Cpu,
    Candle,
}

fn main() {
    let arguments: Vec<String> = std::env::args().skip(1).collect();
    let (backend, arguments) = extract_backend(arguments);

    match backend {
        BackendChoice::Cpu => dispatch(&arguments),
        BackendChoice::Candle => run_with_candle(&arguments),
    }
}

#[cfg(feature = "candle")]
fn run_with_candle(arguments: &[String]) {
    use std::sync::Arc;

    let backend = Arc::new(mini_llm_backend::CandleBackend::cpu());
    mini_llm_autograd::backend::with_backend(backend, || dispatch(arguments));
}

#[cfg(not(feature = "candle"))]
fn run_with_candle(_arguments: &[String]) {
    eprintln!("error: `--backend candle` requires a build with the candle feature:");
    eprintln!("  cargo run --features candle -p mini-llm-cli -- <command> --backend candle");
    std::process::exit(2);
}

/// Strips `--backend <cpu|candle>` / `--backend=<cpu|candle>` from the arguments.
fn extract_backend(arguments: Vec<String>) -> (BackendChoice, Vec<String>) {
    let mut choice = BackendChoice::Cpu;
    let mut rest = Vec::with_capacity(arguments.len());
    let mut arguments = arguments.into_iter();

    while let Some(argument) = arguments.next() {
        if argument == "--backend" {
            match arguments.next().as_deref() {
                Some("cpu") => choice = BackendChoice::Cpu,
                Some("candle") => choice = BackendChoice::Candle,
                Some(other) => unknown_backend(other),
                None => missing_backend(),
            }
        } else if let Some(value) = argument.strip_prefix("--backend=") {
            match value {
                "cpu" => choice = BackendChoice::Cpu,
                "candle" => choice = BackendChoice::Candle,
                other => unknown_backend(other),
            }
        } else {
            rest.push(argument);
        }
    }

    (choice, rest)
}

fn unknown_backend(value: &str) -> ! {
    eprintln!("error: unknown backend `{value}` (expected `cpu` or `candle`)");
    std::process::exit(2);
}

fn missing_backend() -> ! {
    eprintln!("error: `--backend` expects `cpu` or `candle`");
    std::process::exit(2);
}

fn dispatch(arguments: &[String]) {
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
        "eval" => eval::run(rest),
        "sft" => sft::run(rest),
        "dpo" => dpo::run(),
        "grpo" => grpo::run(),
        "trace" => trace::run(rest),
        "device-train" => mini_llm_backend::device_training::run(rest.to_vec()),
        "serve" => serve::run(rest),
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
    println!("Usage: cargo run -p mini-llm-cli -- <command> [args] [--backend cpu|candle]");
    println!();
    println!("Commands:");
    println!("  demo                 scaled dot-product attention walkthrough");
    println!("  bench                kernel + training micro-benchmarks (use --release)");
    println!("  inspect-config       validate and print the default model configuration");
    println!("  autograd             computation graph + gradients for y = wx + b");
    println!("  train-line           fit y = 2x - 1 with manual SGD");
    println!("  xor                  train a small MLP on XOR (Linear + Tanh + AdamW)");
    println!("  classify             train a 2D two-blob classifier with softmax + cross-entropy");
    println!("  train [steps]        train Tiny GPT; model: --hidden N --layers N --heads N");
    println!("                       --kv-heads N --block N; training: --seed N --dropout P");
    println!("                       --accum N --early-stop N --resume --corpus PATH[,PATH]");
    println!("                       `--corpus` accepts several paths or a directory");
    println!("  generate [prompt] [tokens] [temp] [--top-k N] [--top-p P] [--batch N]");
    println!("                       [--seed N] [--checkpoint PATH]");
    println!("                       sample text from the saved checkpoint (KV cache)");
    println!("  moe                  mixture-of-experts routing demo");
    println!("  eval [batches]       validation loss (flags: --corpus PATH, --checkpoint PATH)");
    println!(
        "  sft [steps]          supervised fine-tuning (--bpe, --accum N, --early-stop N, --jsonl PATH)"
    );
    println!("  lora                  fine-tune a frozen classifier with LoRA adapters");
    println!("  dpo                   preference optimization on a toy policy");
    println!("  grpo                  group relative policy optimization on a bandit");
    println!("  trace                 emit training events + capture attention to traces/");
    println!("  serve [--addr H:P]    HTTP inference server (GET /health, POST /generate)");
    println!(
        "  device-train [flags]  device-resident attention block (--layers/--seq/--batch/--steps)"
    );
    println!("                       [--checkpoint PATH] [--load PATH] [--resume PATH] [--seed N]");
    println!("                       [--patience N] [--sample N] [--temperature T] [--top-k K]");
    println!("                       [--prompt TEXT] [--device cpu|cuda] [--corpus PATH]...");
    println!("                       [--corpus-chars N] [--bpe] [--bpe-vocab N]");
    println!("                       [--checkpoint-format f32|f16|q8]");
    println!("                       [--sweep|--layer-sweep|--lr-sweep]");
    println!("  help                 show this message");
    println!();
    println!("Options:");
    println!("  --backend cpu|candle route matmul/softmax through a backend (candle needs");
    println!("                       `--features candle` at build time)");
    println!();
    println!("checkpoint: {}", commands::CHECKPOINT_PATH);
}
