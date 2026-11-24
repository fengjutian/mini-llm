//! Supervised fine-tuning with loss masking: only answer tokens are trained.

use mini_llm_data::{CharTokenizer, SftDataset};
use mini_llm_model::{ModelConfig, TinyGpt};
use mini_llm_nn::Rng;
use mini_llm_train::{TrainConfig, Trainer};

const DIALOGUE: &str = "User: hello\nBot: hi there\nUser: bye\nBot: see you\n";
const SAMPLE_PROMPT: &str = "User: hello\nBot:";
const REPEATS: usize = 6;
const BLOCK_SIZE: usize = 32;

pub fn run(arguments: &[String]) {
    let steps = arguments
        .first()
        .and_then(|value| value.parse().ok())
        .unwrap_or(400);

    let tokenizer = CharTokenizer::from_text(DIALOGUE);
    let (tokens, weights) = build_supervised_data(&tokenizer);
    let supervised = weights.iter().filter(|weight| **weight > 0.0).count();

    let dataset = SftDataset::new(tokens, weights, BLOCK_SIZE).expect("sft dataset");
    let mut config = ModelConfig::modern(tokenizer.vocab_size());
    config.hidden_size = 32;
    config.max_seq_len = BLOCK_SIZE;
    let model = TinyGpt::new(config, &mut Rng::new(11)).expect("model");

    println!("SFT with loss masking (prompt tokens are ignored)");
    println!(
        "  examples: {REPEATS} x 2 | {} tokens, {} supervised | vocab {}",
        REPEATS * 2 * DIALOGUE.chars().count(),
        supervised,
        tokenizer.vocab_size()
    );
    println!(
        "  model: {} params | block {BLOCK_SIZE}",
        model.num_parameters()
    );
    println!();

    let mut trainer = Trainer::new(
        model,
        dataset,
        TrainConfig {
            steps,
            batch_size: 2,
            learning_rate: 3e-3,
            weight_decay: 0.01,
            seed: 5,
        },
    );

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

    let prompt = tokenizer.encode(SAMPLE_PROMPT).expect("prompt encodes");
    let mut sampler = Rng::new(3);
    let generated = trainer
        .model()
        .generate(&prompt, 8, 0.0, &mut sampler)
        .expect("generate");
    println!();
    println!("  prompt:   {SAMPLE_PROMPT:?}");
    println!("  response: {:?}", tokenizer.decode(&generated).unwrap());
}

/// Builds one token stream from repeated (prompt, response) pairs, masking the
/// prompt positions out of the loss.
fn build_supervised_data(tokenizer: &CharTokenizer) -> (Vec<usize>, Vec<f32>) {
    let pairs = [
        ("User: hello\nBot:", " hi there\n"),
        ("User: bye\nBot:", " see you\n"),
    ];
    let mut tokens = Vec::new();
    let mut weights = Vec::new();
    for _ in 0..REPEATS {
        for (prompt, response) in pairs {
            let (mut example_tokens, mut example_weights) = tokenizer
                .encode_supervised(prompt, response)
                .expect("dialogue characters are in the vocabulary");
            tokens.append(&mut example_tokens);
            weights.append(&mut example_weights);
        }
    }
    (tokens, weights)
}
