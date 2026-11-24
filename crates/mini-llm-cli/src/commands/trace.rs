//! Training event protocol + tensor capture, written as JSONL.

use std::time::{SystemTime, UNIX_EPOCH};

use mini_llm_data::{CharTokenizer, Dataset, TextDataset};
use mini_llm_model::{ModelConfig, TinyGpt};
use mini_llm_nn::Rng;
use mini_llm_trace::{Event, EventSink, JsonlSink};
use mini_llm_train::{save_checkpoint, TrainConfig, Trainer};

use super::{CHECKPOINT_PATH, TINY_CORPUS};

const STEPS: usize = 30;
const BLOCK_SIZE: usize = 16;
const TRACE_DIR: &str = "traces";

pub fn run(_arguments: &[String]) {
    std::fs::create_dir_all(TRACE_DIR).ok();
    std::fs::create_dir_all("checkpoints").ok();

    let run_id = format!(
        "run-{}",
        SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .map(|duration| duration.as_millis())
            .unwrap_or(0)
    );
    let path = format!("{TRACE_DIR}/{run_id}.jsonl");
    let file = std::fs::File::create(&path).expect("create trace file");
    let mut sink = JsonlSink::with_run_id(std::io::BufWriter::new(file), run_id);

    let tokenizer = CharTokenizer::from_text(TINY_CORPUS);
    let tokens = tokenizer.encode(TINY_CORPUS).expect("corpus encodes");
    let dataset = TextDataset::new(tokens, BLOCK_SIZE).expect("dataset");

    let mut config = ModelConfig::modern(tokenizer.vocab_size());
    config.hidden_size = 24;
    config.max_seq_len = BLOCK_SIZE;
    let model = TinyGpt::new(config, &mut Rng::new(21)).expect("model");

    let mut trainer = Trainer::new(
        model,
        dataset,
        TrainConfig {
            steps: STEPS,
            batch_size: 2,
            learning_rate: 3e-3,
            weight_decay: 0.0,
            seed: 7,
        },
    );

    println!("training event protocol demo");
    println!(
        "  model: {} params | {} steps | sink: {path}",
        trainer.model().num_parameters(),
        STEPS
    );
    trainer
        .train_observed(STEPS, &mut sink)
        .expect("observed training");

    // Capture one batch: attention matrices + logits.
    let mut rng = Rng::new(99);
    let batch = trainer
        .dataset()
        .sample_batch(1, &mut rng)
        .expect("capture batch");
    let (logits, records) = trainer
        .model()
        .forward_traced(&batch.inputs, batch.batch)
        .expect("traced forward");

    for record in &records {
        sink.record(Event::Tensor {
            name: format!("attention.layer{}.head{}", record.layer, record.head),
            step: STEPS,
            shape: record.weights.shape().to_vec(),
            values: record.weights.data().to_vec(),
        });
    }
    sink.record(Event::Tensor {
        name: "logits".into(),
        step: STEPS,
        shape: logits.shape(),
        values: logits.data(),
    });

    save_checkpoint(CHECKPOINT_PATH, trainer.model(), &Tokenizer::from(tokenizer)).expect("save checkpoint");
    sink.record(Event::Checkpoint {
        path: CHECKPOINT_PATH.into(),
    });

    let seq = batch.seq;
    println!(
        "  captured {} attention matrices (seq {seq}) and logits {}x{}",
        records.len(),
        logits.shape()[0],
        logits.shape()[1]
    );
    for record in records.iter().take(2) {
        let rows = record.weights.shape()[0];
        let focus: f32 = record
            .weights
            .data()
            .chunks(seq)
            .map(|row| row.iter().copied().fold(0.0, f32::max))
            .sum::<f32>()
            / rows as f32;
        println!(
            "  layer {} head {}: mean max attention {:.3}",
            record.layer, record.head, focus
        );
    }
    sink.flush();
    println!("  wrote {} events to {path}", sink.events_written());
}
