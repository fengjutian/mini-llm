//! Samples text from the saved checkpoint with the KV-cached generator.

use mini_llm_model::Sampling;
use mini_llm_nn::Rng;
use mini_llm_train::load_checkpoint;

use super::checkpoint_path;

const DEFAULT_PROMPT: &str = "The little language model learns ";
const DEFAULT_TOKENS: usize = 14;
const DEFAULT_TEMPERATURE: f32 = 0.3;

pub fn run(arguments: &[String]) {
    let mut positional = Vec::new();
    let mut top_k = None;
    let mut top_p = None;
    let mut batch = 1usize;
    let mut seed = 2024u64;
    let mut index = 0;
    while index < arguments.len() {
        let argument = &arguments[index];
        if argument == "--top-k" {
            top_k = arguments
                .get(index + 1)
                .and_then(|value| value.parse().ok());
            index += 2;
        } else if let Some(value) = argument.strip_prefix("--top-k=") {
            top_k = value.parse().ok();
            index += 1;
        } else if argument == "--seed" {
            seed = arguments
                .get(index + 1)
                .and_then(|value| value.parse().ok())
                .unwrap_or(seed);
            index += 2;
        } else if let Some(value) = argument.strip_prefix("--seed=") {
            seed = value.parse().unwrap_or(seed);
            index += 1;
        } else if argument == "--batch" {
            batch = arguments
                .get(index + 1)
                .and_then(|value| value.parse().ok())
                .unwrap_or(1);
            index += 2;
        } else if let Some(value) = argument.strip_prefix("--batch=") {
            batch = value.parse().unwrap_or(1);
            index += 1;
        } else if argument == "--top-p" {
            top_p = arguments
                .get(index + 1)
                .and_then(|value| value.parse().ok());
            index += 2;
        } else if let Some(value) = argument.strip_prefix("--top-p=") {
            top_p = value.parse().ok();
            index += 1;
        } else if argument == "--checkpoint" {
            index += 2;
        } else if argument.starts_with("--checkpoint=") {
            index += 1;
        } else {
            positional.push(argument.clone());
            index += 1;
        }
    }

    let prompt = positional
        .first()
        .map(String::as_str)
        .unwrap_or(DEFAULT_PROMPT);
    let count = positional
        .get(1)
        .and_then(|value| value.parse().ok())
        .unwrap_or(DEFAULT_TOKENS);
    let temperature = positional
        .get(2)
        .and_then(|value| value.parse().ok())
        .unwrap_or(DEFAULT_TEMPERATURE);
    let sampling = Sampling {
        temperature,
        top_k,
        top_p,
    };

    let checkpoint = checkpoint_path(arguments);
    let (model, tokenizer) = match load_checkpoint(&checkpoint) {
        Ok(loaded) => loaded,
        Err(err) => {
            eprintln!("cannot load {checkpoint}: {err}");
            eprintln!("run `cargo run -p mini-llm-cli -- train` first, or pass --checkpoint PATH");
            std::process::exit(1);
        }
    };

    // Drop characters the checkpoint never saw instead of failing outright.
    // BPE covers every byte, so it needs no filtering.
    let filtered: String = match tokenizer.as_char() {
        Some(char_tokenizer) => prompt
            .chars()
            .filter(|character| char_tokenizer.characters().contains(character))
            .collect(),
        None => prompt.to_string(),
    };
    let encoded = if filtered.is_empty() {
        vec![0]
    } else {
        tokenizer
            .encode(&filtered)
            .expect("filtered prompt encodes")
    };

    let mut rng = Rng::new(seed);
    if batch > 1 {
        let prompts = vec![encoded.clone(); batch];
        let generated = model
            .generate_batch_cached(&prompts, count, sampling, &mut rng)
            .expect("generate");
        for (index, tokens) in generated.iter().enumerate() {
            println!("  [{index}] {}", tokenizer.decode(tokens).expect("decode"));
        }
    } else {
        let generated = model
            .generate_with(&encoded, count, sampling, &mut rng)
            .expect("generate");
        println!("{}", tokenizer.decode(&generated).expect("decode"));
    }
}
