//! Character-level Tiny GPT training on the built-in corpus.
//!
//! Splits the corpus into train/validation windows, uses warmup+cosine learning
//! rate and gradient clipping, and saves a checkpoint that can be resumed
//! exactly (optimizer moments, RNG and step counter included).

use std::time::Instant;

use mini_llm_data::{BpeTokenizer, CharTokenizer, TextDataset, Tokenizer};
use mini_llm_model::{ModelConfig, TinyGpt};
use mini_llm_nn::Rng;
use mini_llm_train::{
    load_training_checkpoint, save_training_checkpoint, EarlyStopping, LrSchedule, TrainConfig,
    Trainer,
};

use super::{corpus_text, parse_accumulation, CHECKPOINT_PATH};

const BLOCK_SIZE: usize = 48;
const BATCH_SIZE: usize = 2;
const DEFAULT_STEPS: usize = 800;
const WARMUP_STEPS: usize = 40;
const GRAD_CLIP: f32 = 1.0;
const VALIDATION_FRACTION: f32 = 0.1;

pub fn run(arguments: &[String]) {
    let resume = arguments.iter().any(|argument| argument == "--resume");
    let use_bpe = arguments.iter().any(|argument| argument == "--bpe");
    let vocab_size = parse_vocab_size(arguments);
    let dropout = parse_dropout(arguments);
    let accumulation = parse_accumulation(arguments);
    let steps = arguments
        .first()
        .and_then(|value| value.parse::<usize>().ok())
        .unwrap_or(DEFAULT_STEPS);

    let corpus = corpus_text(arguments);
    let (text, corpus_label) = corpus;

    let resumed = if resume {
        match load_training_checkpoint(CHECKPOINT_PATH) {
            Ok((model, tokenizer, state)) => {
                println!("  resumed checkpoints/tiny-gpt.bin at step {}", state.step);
                Some((model, tokenizer, state))
            }
            Err(error) => {
                println!("  cannot resume ({error}); starting a fresh run");
                None
            }
        }
    } else {
        None
    };

    let (tokenizer, model, restored_state) = match resumed {
        Some((model, tokenizer, state)) => (tokenizer, model, Some(state)),
        None => {
            let tokenizer: Tokenizer = if use_bpe {
                Tokenizer::from(BpeTokenizer::train(&text, vocab_size))
            } else {
                Tokenizer::from(CharTokenizer::from_text(&text))
            };
            let model = build_model(tokenizer.vocab_size(), dropout);
            (tokenizer, model, None)
        }
    };
    let tokens = match tokenizer.encode(&text) {
        Ok(tokens) => tokens,
        Err(error) => {
            eprintln!("the checkpoint vocabulary cannot encode `{corpus_label}`: {error}");
            eprintln!("start a fresh run (without --resume) or pass a compatible --corpus");
            std::process::exit(2);
        }
    };
    if tokens.len() <= BLOCK_SIZE + 1 {
        eprintln!("`{corpus_label}` is too short: need more than {BLOCK_SIZE} tokens");
        std::process::exit(2);
    }

    let (train_tokens, validation_tokens) = split_tokens(&tokens, VALIDATION_FRACTION);
    let train_dataset = TextDataset::new(train_tokens, BLOCK_SIZE).expect("train dataset");
    let validation_dataset =
        TextDataset::new(validation_tokens, BLOCK_SIZE).expect("validation dataset");

    print_model_summary(
        &model,
        &tokenizer,
        &text,
        &corpus_label,
        tokens.len(),
        steps,
    );

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
        .with_grad_clip(GRAD_CLIP)
        .with_accumulation_steps(accumulation);
    println!(
        "  effective batch: {} windows ({} x {})",
        BATCH_SIZE * accumulation,
        BATCH_SIZE,
        accumulation
    );

    if let Some(state) = restored_state {
        trainer
            .restore_state(&state)
            .expect("restore training state");
        println!(
            "  resumed from step {} (optimizer step {})",
            state.step, state.optimizer_step
        );
    }

    let mut last_loss = f32::NAN;
    let early_stop = parse_early_stop(arguments);
    let mut stopper = EarlyStopping::new(early_stop);
    let mut best_validation = f32::INFINITY;
    let mut best_step = 0usize;

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

            if stopper.observe(validation) {
                best_validation = validation;
                best_step = step;
                save_training_checkpoint(
                    CHECKPOINT_PATH,
                    trainer.model(),
                    &tokenizer,
                    &trainer.state(),
                )
                .expect("save checkpoint");
                if early_stop > 0 {
                    println!("    new best validation loss; checkpoint updated");
                }
            } else if stopper.should_stop() {
                println!(
                    "    early stopping after {} evaluations without improvement",
                    early_stop
                );
                break;
            }
        }
    }
    let validation = trainer
        .evaluate_on(&validation_dataset, 20)
        .expect("validation");
    println!("  final sampled loss: {last_loss:.4}");
    println!("  final validation loss: {validation:.4}");
    if early_stop > 0 {
        println!(
            "  best validation loss: {best_validation:.4} at step {best_step} (patience {early_stop})"
        );
    }

    print_sample(trainer.model(), &tokenizer, &text);
    if early_stop == 0 || validation < best_validation {
        persist(trainer.model(), &tokenizer, &trainer.state());
    } else {
        println!();
        println!(
            "best checkpoint kept: step {best_step}, validation {best_validation:.4}"
        );
    }
}

fn split_tokens(tokens: &[usize], fraction: f32) -> (Vec<usize>, Vec<usize>) {
    let split = ((tokens.len() as f32) * (1.0 - fraction)) as usize;
    let split = split.clamp(BLOCK_SIZE + 1, tokens.len().saturating_sub(BLOCK_SIZE + 1));
    (tokens[..split].to_vec(), tokens[split..].to_vec())
}

/// Reads `--early-stop N` (evaluations without improvement); 0 disables it.
fn parse_early_stop(arguments: &[String]) -> usize {
    let mut value = None;
    for (index, argument) in arguments.iter().enumerate() {
        if argument == "--early-stop" {
            value = arguments.get(index + 1).and_then(|next| next.parse().ok());
        } else if let Some(rest) = argument.strip_prefix("--early-stop=") {
            value = rest.parse().ok();
        }
    }
    value.unwrap_or(0)
}

/// Reads `--dropout P`; defaults to 0.0 (disabled).
fn parse_dropout(arguments: &[String]) -> f32 {
    let mut value: Option<f32> = None;
    for (index, argument) in arguments.iter().enumerate() {
        if argument == "--dropout" {
            value = arguments.get(index + 1).and_then(|next| next.parse().ok());
        } else if let Some(rest) = argument.strip_prefix("--dropout=") {
            value = rest.parse().ok();
        }
    }
    value.unwrap_or(0.0).clamp(0.0, 1.0)
}

/// Reads `--vocab N` for BPE training; defaults to 512 tokens.
fn parse_vocab_size(arguments: &[String]) -> usize {
    let mut value = None;
    for (index, argument) in arguments.iter().enumerate() {
        if argument == "--vocab" {
            value = arguments.get(index + 1).and_then(|next| next.parse().ok());
        } else if let Some(rest) = argument.strip_prefix("--vocab=") {
            value = rest.parse().ok();
        }
    }
    value.unwrap_or(512).max(257)
}

fn build_model(vocab_size: usize, dropout: f32) -> TinyGpt {
    let mut config = ModelConfig::modern(vocab_size);
    config.dropout = dropout;
    config.hidden_size = 32;
    config.max_seq_len = BLOCK_SIZE;
    let mut rng = Rng::new(1337);
    TinyGpt::new(config, &mut rng).expect("model")
}

fn print_model_summary(
    model: &TinyGpt,
    tokenizer: &Tokenizer,
    text: &str,
    corpus_label: &str,
    tokens: usize,
    steps: usize,
) {
    let config = model.config();
    let tokenizer_kind = if tokenizer.as_bpe().is_some() {
        "byte-level BPE"
    } else {
        "character"
    };
    println!("Tiny GPT, {tokenizer_kind} level (modern stack)");
    println!(
        "  corpus: {corpus_label} ({} chars), vocab {}, {} training windows ({} held out)",
        text.chars().count(),
        tokenizer.vocab_size(),
        tokens - BLOCK_SIZE,
        (tokens as f32 * VALIDATION_FRACTION) as usize
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
        "  optimizer: AdamW | batch={BATCH_SIZE} | steps={steps} | warmup={WARMUP_STEPS} | clip={GRAD_CLIP} | dropout={:.2}",
        config.dropout
    );
    println!();
}

fn print_sample(model: &TinyGpt, tokenizer: &Tokenizer, text: &str) {
    // Use the corpus prefix so the prompt is always in the vocabulary.
    let prompt_text: String = text.chars().take(24).collect();
    let prompt = match tokenizer.encode(&prompt_text) {
        Ok(prompt) if !prompt.is_empty() => prompt,
        _ => vec![0],
    };
    let mut sampler = Rng::new(99);
    let generated = model
        .generate(&prompt, 14, 0.3, &mut sampler)
        .expect("generate");
    println!();
    println!("--- sample (KV-cached, within the trained context) ---");
    println!("{}", tokenizer.decode(&generated).expect("decode"));
}

fn persist(model: &TinyGpt, tokenizer: &Tokenizer, state: &mini_llm_train::TrainingState) {
    std::fs::create_dir_all("checkpoints").ok();
    save_training_checkpoint(CHECKPOINT_PATH, model, tokenizer, state).expect("save checkpoint");
    println!();
    println!("checkpoint saved to {CHECKPOINT_PATH}");
}
