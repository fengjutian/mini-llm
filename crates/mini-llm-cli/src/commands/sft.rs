//! Supervised fine-tuning with loss masking: only answer tokens are trained.
//!
//! Data can come from the built-in dialogue or from a JSONL file of
//! `{"prompt": ..., "response": ...}` objects (`--jsonl PATH`).

use mini_llm_data::{parse_instructions, BpeTokenizer, CharTokenizer, SftDataset, Tokenizer};
use mini_llm_model::{ModelConfig, TinyGpt};
use mini_llm_nn::Rng;
use mini_llm_train::{TrainConfig, Trainer};

use super::parse_accumulation;

const REPEATS: usize = 6;
const BLOCK_SIZE: usize = 32;
const BATCH_SIZE: usize = 2;

pub fn run(arguments: &[String]) {
    let steps = arguments
        .first()
        .and_then(|value| value.parse().ok())
        .unwrap_or(400);

    let accumulation = parse_accumulation(arguments);
    let use_bpe = arguments.iter().any(|argument| argument == "--bpe");
    let (pairs, source) = load_pairs(arguments);

    let corpus: String = pairs
        .iter()
        .map(|(prompt, response)| format!("{prompt}{response}"))
        .collect();
    let tokenizer: Tokenizer = if use_bpe {
        Tokenizer::from(BpeTokenizer::train(&corpus, 300))
    } else {
        Tokenizer::from(CharTokenizer::from_text(&corpus))
    };

    let (tokens, weights, repeats) = build_supervised_data(&tokenizer, &pairs);
    let supervised = weights.iter().filter(|weight| **weight > 0.0).count();
    let token_count = tokens.len();
    let dataset = SftDataset::new(tokens, weights, BLOCK_SIZE).expect("sft dataset");
    let mut config = ModelConfig::modern(tokenizer.vocab_size());
    config.hidden_size = 32;
    config.max_seq_len = BLOCK_SIZE;
    let model = TinyGpt::new(config, &mut Rng::new(11)).expect("model");

    println!(
        "SFT with loss masking ({source}, {kind} tokenizer)",
        kind = if use_bpe { "BPE" } else { "char" }
    );
    println!(
        "  examples: {} pairs x {repeats} | {} tokens, {} supervised | vocab {}",
        pairs.len(),
        token_count,
        supervised,
        tokenizer.vocab_size()
    );
    println!(
        "  model: {} params | block {BLOCK_SIZE} | batch {} x {accumulation}",
        model.num_parameters(),
        BATCH_SIZE
    );
    println!();

    let mut trainer = Trainer::new(
        model,
        dataset,
        TrainConfig {
            steps,
            batch_size: BATCH_SIZE,
            learning_rate: 3e-3,
            weight_decay: 0.01,
            seed: 5,
        },
    )
    .with_accumulation_steps(accumulation);

    let mut last = f32::NAN;
    trainer
        .train(steps, |report| {
            last = report.loss;
            if report.step == 1 || report.step % 50 == 0 {
                println!(
                    "  step {:>3}  masked loss {:>7.4}  {:>5.1} ms/step",
                    report.step, report.loss, report.elapsed_ms
                );
            }
        })
        .expect("training");
    println!("  final masked loss: {last:.4}");

    let prompt_text = &pairs[0].0;
    let prompt = tokenizer.encode(prompt_text).expect("prompt encodes");
    let mut sampler = Rng::new(3);
    let generated = trainer
        .model()
        .generate(&prompt, 8, 0.0, &mut sampler)
        .expect("generate");
    println!();
    println!("  prompt:   {prompt_text:?}");
    println!("  response: {:?}", tokenizer.decode(&generated).unwrap());
}

/// Loads instruction pairs from `--jsonl PATH`, or the built-in dialogue.
fn load_pairs(arguments: &[String]) -> (Vec<(String, String)>, String) {
    let path = arguments
        .iter()
        .position(|argument| argument == "--jsonl")
        .and_then(|index| arguments.get(index + 1).cloned())
        .or_else(|| {
            arguments
                .iter()
                .find_map(|argument| argument.strip_prefix("--jsonl=").map(str::to_string))
        });

    match path {
        Some(path) => {
            let text = match std::fs::read_to_string(&path) {
                Ok(text) => text,
                Err(error) => {
                    eprintln!("cannot read instructions `{path}`: {error}");
                    std::process::exit(2);
                }
            };
            match parse_instructions(&text) {
                Ok(pairs) => (
                    pairs
                        .into_iter()
                        .map(|pair| (pair.prompt, pair.response))
                        .collect(),
                    path,
                ),
                Err(error) => {
                    eprintln!("cannot parse `{path}`: {error}");
                    std::process::exit(2);
                }
            }
        }
        None => (
            vec![
                ("User: hello\nBot:".to_string(), " hi there\n".to_string()),
                ("User: bye\nBot:".to_string(), " see you\n".to_string()),
            ],
            "built-in dialogue".to_string(),
        ),
    }
}

/// Builds one masked token stream, repeating the pairs until there is enough
/// data to fill a few context windows.
fn build_supervised_data(
    tokenizer: &Tokenizer,
    pairs: &[(String, String)],
) -> (Vec<usize>, Vec<f32>, usize) {
    let mut tokens = Vec::new();
    let mut weights = Vec::new();
    let mut repeats = 0usize;
    while repeats < REPEATS || tokens.len() <= BLOCK_SIZE * 2 {
        if repeats >= 256 {
            break;
        }
        for (prompt, response) in pairs {
            let (mut example_tokens, mut example_weights) = tokenizer
                .encode_supervised(prompt, response)
                .expect("instruction text is in the vocabulary");
            tokens.append(&mut example_tokens);
            weights.append(&mut example_weights);
        }
        repeats += 1;
    }
    (tokens, weights, repeats)
}
