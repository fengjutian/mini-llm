//! Supervised fine-tuning with loss masking: only answer tokens are trained.
//!
//! Data can come from the built-in dialogue or from a JSONL file of
//! `{"prompt": ..., "response": ...}` objects (`--jsonl PATH`).

use mini_llm_data::{parse_instructions, BpeTokenizer, CharTokenizer, SftDataset, Tokenizer};
use mini_llm_model::{ModelConfig, TinyGpt};
use mini_llm_nn::Rng;
use mini_llm_train::{save_training_checkpoint, EarlyStopping, LrSchedule, TrainConfig, Trainer};

use super::{parse_accumulation, parse_early_stop, SFT_CHECKPOINT_PATH};

const REPEATS: usize = 6;
const BLOCK_SIZE: usize = 32;
const BATCH_SIZE: usize = 2;
const EVAL_EVERY: usize = 50;
const GRAD_CLIP: f32 = 1.0;
const DROPOUT: f32 = 0.05;

/// One prompt/response training pair.
type InstructionData = (String, String);

pub fn run(arguments: &[String]) {
    let steps = arguments
        .first()
        .and_then(|value| value.parse().ok())
        .unwrap_or(400);

    let accumulation = parse_accumulation(arguments);
    let use_bpe = arguments.iter().any(|argument| argument == "--bpe");
    let (pairs, source) = load_pairs(arguments);
    let (train_pairs, validation_pairs) = split_pairs(&pairs);

    let corpus: String = pairs
        .iter()
        .map(|(prompt, response)| format!("{prompt}{response}"))
        .collect();
    let tokenizer: Tokenizer = if use_bpe {
        Tokenizer::from(BpeTokenizer::train(&corpus, 300))
    } else {
        Tokenizer::from(CharTokenizer::from_text(&corpus))
    };

    let (tokens, weights, repeats) = build_supervised_data(&tokenizer, &train_pairs);
    let supervised = weights.iter().filter(|weight| **weight > 0.0).count();
    let token_count = tokens.len();
    let dataset = SftDataset::new(tokens, weights, BLOCK_SIZE).expect("sft dataset");
    let validation = build_validation_dataset(&tokenizer, &validation_pairs);
    let mut config = ModelConfig::modern(tokenizer.vocab_size());
    config.hidden_size = 32;
    config.max_seq_len = BLOCK_SIZE;
    config.dropout = DROPOUT;
    let model = TinyGpt::new(config, &mut Rng::new(11)).expect("model");

    println!(
        "SFT with loss masking ({source}, {kind} tokenizer)",
        kind = if use_bpe { "BPE" } else { "char" }
    );
    println!(
        "  train: {} pairs x {repeats} | {} tokens, {} supervised | vocab {}",
        train_pairs.len(),
        token_count,
        supervised,
        tokenizer.vocab_size()
    );
    let held_note = if validation_pairs.is_empty() {
        " (too few pairs to hold one out)"
    } else if validation.is_none() {
        " (too short to evaluate)"
    } else {
        ""
    };
    println!(
        "  held out: {} pairs{} | model {} params | block {BLOCK_SIZE} | batch {} x {accumulation}",
        validation_pairs.len(),
        held_note,
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
    .with_schedule(LrSchedule::WarmupCosine {
        warmup_steps: (steps / 10).max(1),
        min_ratio: 0.1,
    })
    .with_grad_clip(GRAD_CLIP)
    .with_accumulation_steps(accumulation);

    let mut last = f32::NAN;
    let early_stop = parse_early_stop(arguments);
    let mut stopper = EarlyStopping::new(early_stop);
    let mut best_loss = f32::INFINITY;
    let mut best_step = 0usize;
    let mut stopped_early = false;

    if early_stop > 0 && validation.is_none() {
        println!("  note: --early-stop is ignored without a usable held-out split");
    }

    for step in 1..=steps {
        let started = std::time::Instant::now();
        last = trainer.step().expect("training step");
        if step == 1 || step % 10 == 0 {
            println!(
                "  step {:>3}  masked loss {:>7.4}  lr {:>8.6}  {:>5.1} ms/step",
                step,
                last,
                trainer.learning_rate(),
                started.elapsed().as_secs_f32() * 1_000.0
            );
        }
        if step % EVAL_EVERY == 0 {
            if let Some(validation) = &validation {
                let loss = trainer.evaluate_on(validation, 10).expect("validation");
                println!("    held-out masked loss after step {step}: {loss:.4}");

                if stopper.observe(loss) {
                    best_loss = loss;
                    best_step = step;
                    save_training_checkpoint(
                        SFT_CHECKPOINT_PATH,
                        trainer.model(),
                        &tokenizer,
                        &trainer.state(),
                    )
                    .expect("save sft checkpoint");
                    if early_stop > 0 {
                        println!("    new best; checkpoint updated -> {SFT_CHECKPOINT_PATH}");
                    }
                } else if stopper.should_stop() {
                    println!(
                        "    early stopping after {} evaluations without improvement",
                        early_stop
                    );
                    stopped_early = true;
                    break;
                }
            }
        }
    }
    println!("  final masked loss: {last:.4}");
    if let Some(validation) = &validation {
        let loss = trainer.evaluate_on(validation, 20).expect("validation");
        println!("  final held-out masked loss: {loss:.4}");

        let keep_best = stopped_early || (early_stop > 0 && loss >= best_loss);
        if keep_best {
            println!(
                "  best checkpoint kept: {SFT_CHECKPOINT_PATH} (step {best_step}, loss {best_loss:.4})"
            );
        } else {
            save_training_checkpoint(
                SFT_CHECKPOINT_PATH,
                trainer.model(),
                &tokenizer,
                &trainer.state(),
            )
            .expect("save sft checkpoint");
            println!("  checkpoint saved to {SFT_CHECKPOINT_PATH}");
        }
    }

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
fn load_pairs(arguments: &[String]) -> (Vec<InstructionData>, String) {
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

/// Holds out about 10% of the pairs (at least one when possible).
fn split_pairs(pairs: &[(String, String)]) -> (Vec<InstructionData>, Vec<InstructionData>) {
    if pairs.len() < 4 {
        return (pairs.to_vec(), Vec::new());
    }
    let held = (pairs.len() / 10).max(1).min(pairs.len() - 1);
    let split = pairs.len() - held;
    (pairs[..split].to_vec(), pairs[split..].to_vec())
}

/// Builds a one-pass validation stream, or `None` when it is too short.
fn build_validation_dataset(
    tokenizer: &Tokenizer,
    pairs: &[(String, String)],
) -> Option<SftDataset> {
    if pairs.is_empty() {
        return None;
    }
    let mut tokens = Vec::new();
    let mut weights = Vec::new();
    for (prompt, response) in pairs {
        let (mut example_tokens, mut example_weights) =
            tokenizer.encode_supervised(prompt, response).ok()?;
        tokens.append(&mut example_tokens);
        weights.append(&mut example_weights);
    }
    if tokens.len() <= BLOCK_SIZE {
        return None;
    }
    SftDataset::new(tokens, weights, BLOCK_SIZE).ok()
}
