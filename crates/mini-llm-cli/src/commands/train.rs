//! Character-level Tiny GPT training on the built-in corpus.

use mini_llm_data::{CharTokenizer, TextDataset};
use mini_llm_model::{ModelConfig, TinyGpt};
use mini_llm_nn::Rng;
use mini_llm_train::{save_checkpoint, TrainConfig, Trainer};

use super::{CHECKPOINT_PATH, TINY_CORPUS};

const BLOCK_SIZE: usize = 48;
const BATCH_SIZE: usize = 2;
const DEFAULT_STEPS: usize = 800;
const SAMPLE_PROMPT: &str = "The little language model learns ";

pub fn run(arguments: &[String]) {
    let steps = arguments
        .first()
        .and_then(|value| value.parse().ok())
        .unwrap_or(DEFAULT_STEPS);

    let tokenizer = CharTokenizer::from_text(TINY_CORPUS);
    let tokens = tokenizer.encode(TINY_CORPUS).expect("corpus encodes");
    let mut config = ModelConfig::modern(tokenizer.vocab_size());
    config.hidden_size = 32;
    config.max_seq_len = BLOCK_SIZE;

    let mut rng = Rng::new(1337);
    let model = TinyGpt::new(config, &mut rng).expect("model");

    print_model_summary(&model, &tokenizer, &tokens, steps);

    let dataset = TextDataset::new(tokens, BLOCK_SIZE).expect("dataset");
    let mut trainer = Trainer::new(
        model,
        dataset,
        TrainConfig {
            steps,
            batch_size: BATCH_SIZE,
            learning_rate: 3e-3,
            weight_decay: 0.01,
            seed: 7,
        },
    );

    let mut last_loss = f32::NAN;
    trainer
        .train(steps, |report| {
            last_loss = report.loss;
            if report.step == 1 || report.step % 10 == 0 {
                println!(
                    "  step {:>4}  loss {:>7.4}  {:>6.1} ms/step",
                    report.step, report.loss, report.elapsed_ms
                );
            }
        })
        .expect("training");
    println!("  final sampled loss: {last_loss:.4}");

    print_sample(trainer.model(), &tokenizer);
    persist(trainer.model(), &tokenizer);
}

fn print_model_summary(model: &TinyGpt, tokenizer: &CharTokenizer, tokens: &[usize], steps: usize) {
    let config = model.config();
    println!("Tiny GPT, character level (modern stack)");
    println!(
        "  corpus: {} chars, vocab {}, {} training windows",
        TINY_CORPUS.chars().count(),
        tokenizer.vocab_size(),
        tokens.len() - BLOCK_SIZE
    );
    println!(
        "  model: {} params | layers={} hidden={} heads={} kv_heads={} block={}",
        model.num_parameters(),
        config.num_layers,
        config.hidden_size,
        config.num_heads,
        config.num_kv_heads,
        config.max_seq_len
    );
    println!(
        "  architecture: norm={:?} position={:?} ffn={:?} experts={}",
        config.norm, config.position, config.feed_forward, config.num_experts
    );
    println!("  optimizer: AdamW | batch={BATCH_SIZE} | steps={steps}");
    println!();
}

fn print_sample(model: &TinyGpt, tokenizer: &CharTokenizer) {
    let prompt = tokenizer.encode(SAMPLE_PROMPT).unwrap_or_else(|_| vec![0]);
    let mut sampler = Rng::new(99);
    let generated = model
        .generate(&prompt, 14, 0.3, &mut sampler)
        .expect("generate");
    println!();
    println!("--- sample (KV-cached, within the trained context) ---");
    println!("{}", tokenizer.decode(&generated).expect("decode"));
}

fn persist(model: &TinyGpt, tokenizer: &CharTokenizer) {
    std::fs::create_dir_all("checkpoints").ok();
    save_checkpoint(CHECKPOINT_PATH, model, tokenizer).expect("save checkpoint");
    println!();
    println!("checkpoint saved to {CHECKPOINT_PATH}");
}
