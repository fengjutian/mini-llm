//! Samples text from the saved checkpoint with the KV-cached generator.

use mini_llm_nn::Rng;
use mini_llm_train::load_checkpoint;

use super::CHECKPOINT_PATH;

const DEFAULT_PROMPT: &str = "The little language model learns ";
const DEFAULT_TOKENS: usize = 14;
const DEFAULT_TEMPERATURE: f32 = 0.3;

pub fn run(arguments: &[String]) {
    let prompt = arguments
        .first()
        .map(String::as_str)
        .unwrap_or(DEFAULT_PROMPT);
    let count = arguments
        .get(1)
        .and_then(|value| value.parse().ok())
        .unwrap_or(DEFAULT_TOKENS);
    let temperature = arguments
        .get(2)
        .and_then(|value| value.parse().ok())
        .unwrap_or(DEFAULT_TEMPERATURE);

    let (model, tokenizer) = match load_checkpoint(CHECKPOINT_PATH) {
        Ok(loaded) => loaded,
        Err(err) => {
            eprintln!("cannot load {CHECKPOINT_PATH}: {err}");
            eprintln!("run `cargo run -p mini-llm-cli -- train` first");
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

    let mut rng = Rng::new(2024);
    let generated = model
        .generate(&encoded, count, temperature, &mut rng)
        .expect("generate");
    println!("{}", tokenizer.decode(&generated).expect("decode"));
}
