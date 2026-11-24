//! Evaluates a saved checkpoint on the held-out validation split.

use mini_llm_data::TextDataset;
use mini_llm_train::{load_checkpoint, load_training_checkpoint, TrainConfig, Trainer};

use super::{corpus_text, CHECKPOINT_PATH};

const BLOCK_SIZE: usize = 48;
const VALIDATION_FRACTION: f32 = 0.1;
const EVAL_BATCHES: usize = 20;

pub fn run(arguments: &[String]) {
    let batches = arguments
        .first()
        .and_then(|value| value.parse::<usize>().ok())
        .unwrap_or(EVAL_BATCHES);

    let (text, corpus_label) = corpus_text(arguments);

    let (model, tokenizer) = match load_training_checkpoint(CHECKPOINT_PATH) {
        Ok((model, tokenizer, state)) => {
            println!("  checkpoint step: {}", state.step);
            (model, tokenizer)
        }
        Err(_) => match load_checkpoint(CHECKPOINT_PATH) {
            Ok((model, tokenizer)) => (model, tokenizer),
            Err(error) => {
                eprintln!("cannot load {CHECKPOINT_PATH}: {error}");
                eprintln!("run `train` first to produce a checkpoint");
                return;
            }
        },
    };

    let tokens = match tokenizer.encode(&text) {
        Ok(tokens) => tokens,
        Err(error) => {
            eprintln!("checkpoint vocabulary cannot encode `{corpus_label}`: {error}");
            eprintln!("pass the same `--corpus` used for training");
            return;
        }
    };
    let (_, validation_tokens) = split_tokens(&tokens, VALIDATION_FRACTION);
    let dataset = match TextDataset::new(validation_tokens, BLOCK_SIZE) {
        Ok(dataset) => dataset,
        Err(error) => {
            eprintln!("cannot build the validation dataset: {error}");
            return;
        }
    };

    let config = TrainConfig {
        steps: 0,
        batch_size: 2,
        learning_rate: 3e-3,
        weight_decay: 0.01,
        seed: 7,
    };
    let mut trainer = Trainer::new(model, dataset, config);
    match trainer.evaluate(batches) {
        Ok(loss) => println!(
            "validation loss on `{corpus_label}` ({batches} batches of 2 windows): {loss:.4}"
        ),
        Err(error) => eprintln!("evaluation failed: {error}"),
    }
}

fn split_tokens(tokens: &[usize], fraction: f32) -> (Vec<usize>, Vec<usize>) {
    let split = ((tokens.len() as f32) * (1.0 - fraction)) as usize;
    let split = split.clamp(BLOCK_SIZE + 1, tokens.len().saturating_sub(BLOCK_SIZE + 1));
    (tokens[..split].to_vec(), tokens[split..].to_vec())
}
