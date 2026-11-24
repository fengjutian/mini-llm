//! Character-level Tiny GPT training on the built-in corpus.
//!
//! Splits the corpus into train/validation windows, uses warmup+cosine learning
//! rate and gradient clipping, and saves a checkpoint that can be resumed
//! exactly (optimizer moments, RNG and step counter included).

use std::time::Instant;

use mini_llm_data::{CharTokenizer, TextDataset};
use mini_llm_model::{ModelConfig, TinyGpt};
use mini_llm_nn::Rng;
use mini_llm_train::{
    load_training_checkpoint, save_training_checkpoint, LrSchedule, TrainConfig, Trainer,
};

use super::{CHECKPOINT_PATH, TINY_CORPUS};

const BLOCK_SIZE: usize = 48;
const BATCH_SIZE: usize = 2;
const DEFAULT_STEPS: usize = 800;
const WARMUP_STEPS: usize = 40;
const GRAD_CLIP: f32 = 1.0;
const VALIDATION_FRACTION: f32 = 0.1;
const SAMPLE_PROMPT: &str = "The little language model learns ";

pub fn run(arguments: &[String]) {
    let resume = arguments.iter().any(|argument| argument == "--resume");
    let steps = arguments
        .iter()
        .find_map(|value| value.parse::<usize>().ok())
        .unwrap_or(DEFAULT_STEPS);

    let corpus_tokenizer = CharTokenizer::from_text(TINY_CORPUS);
    let tokens = corpus_tokenizer
        .encode(TINY_CORPUS)
        .expect("corpus encodes");
    let (train_tokens, validation_tokens) = split_tokens(&tokens, VALIDATION_FRACTION);
    let train_dataset = TextDataset::new(train_tokens, BLOCK_SIZE).expect("train dataset");
    let validation_dataset =
        TextDataset::new(validation_tokens, BLOCK_SIZE).expect("validation dataset");

    let (tokenizer, model, restored) = if resume {
        match load_training_checkpoint(CHECKPOINT_PATH) {
            Ok((model, tokenizer, state)) => (tokenizer, model, Some(state)),
            Err(error) => {
                println!("  cannot resume ({error}); starting a fresh run");
                (corpus_tokenizer, build_model(), None)
            }
        }
    } else {
        (corpus_tokenizer, build_model(), None)
    };

    print_model_summary(&model, &tokenizer, &tokens, steps);

    let config = TrainConfig {
        steps,
        batch_size: BATCH_SIZE,
        learning_rate: 3e-3,
        weight_decay: 0.01,
        seed: 7,
    };
    let mut trainer = Trainer::new(model, train_dataset, config)
        .with_schedule(LrSchedule::WarmupCosine {
            warmup_steps: WARMUP_STEPS,
            min_ratio: 0.1,
        })
        .with_grad_clip(GRAD_CLIP);

    if let Some(state) = restored {
        trainer
            .restore_state(&state)
            .expect("restore training state");
        println!(
            "  resumed from step {} (optimizer step {})",
            state.step, state.optimizer_step
        );
    }

    let mut last_loss = f32::NAN;
    for step in 1..=steps {
        let started = Instant::now();
        last_loss = trainer.step().expect("training step");
        if step == 1 || step % 10 == 0 {
            println!(
                "  step {:>4}  loss {:>7.4}  lr {:>8.6}  |g| {:>6.3}  {:>6.1} ms/step",
                step,
                last_loss,
                trainer.learning_rate(),
                trainer.last_grad_norm(),
                started.elapsed().as_secs_f32() * 1_000.0
            );
        }
        if step % 50 == 0 {
            let validation = trainer
                .evaluate_on(&validation_dataset, 10)
                .expect("validation");
            println!("    validation loss after step {step}: {validation:.4}");
        }
    }
    let validation = trainer
        .evaluate_on(&validation_dataset, 20)
        .expect("validation");
    println!("  final sampled loss: {last_loss:.4}");
    println!("  final validation loss: {validation:.4}");

    print_sample(trainer.model(), &tokenizer);
    persist(trainer.model(), &tokenizer, &trainer.state());
}

fn split_tokens(tokens: &[usize], fraction: f32) -> (Vec<usize>, Vec<usize>) {
    let split = ((tokens.len() as f32) * (1.0 - fraction)) as usize;
    let split = split.clamp(BLOCK_SIZE + 1, tokens.len().saturating_sub(BLOCK_SIZE + 1));
    (tokens[..split].to_vec(), tokens[split..].to_vec())
}

fn build_model() -> TinyGpt {
    let tokenizer = CharTokenizer::from_text(TINY_CORPUS);
    let mut config = ModelConfig::modern(tokenizer.vocab_size());
    config.hidden_size = 32;
    config.max_seq_len = BLOCK_SIZE;
    let mut rng = Rng::new(1337);
    TinyGpt::new(config, &mut rng).expect("model")
}

fn print_model_summary(model: &TinyGpt, tokenizer: &CharTokenizer, tokens: &[usize], steps: usize) {
    let config = model.config();
    println!("Tiny GPT, character level (modern stack)");
    println!(
        "  corpus: {} chars, vocab {}, {} training windows ({} held out)",
        TINY_CORPUS.chars().count(),
        tokenizer.vocab_size(),
        tokens.len() - BLOCK_SIZE,
        (tokens.len() as f32 * VALIDATION_FRACTION) as usize
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
    println!(
        "  optimizer: AdamW | batch={BATCH_SIZE} | steps={steps} | warmup={WARMUP_STEPS} | clip={GRAD_CLIP}"
    );
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

fn persist(model: &TinyGpt, tokenizer: &CharTokenizer, state: &mini_llm_train::TrainingState) {
    std::fs::create_dir_all("checkpoints").ok();
    save_training_checkpoint(CHECKPOINT_PATH, model, tokenizer, state).expect("save checkpoint");
    println!();
    println!("checkpoint saved to {CHECKPOINT_PATH}");
}
