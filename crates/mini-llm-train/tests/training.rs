//! Training-loop and checkpoint tests.

use std::path::PathBuf;

use mini_llm_data::{Batch, CharTokenizer, Dataset, TextDataset};
use mini_llm_model::{ModelConfig, TinyGpt};
use mini_llm_nn::Rng;
use mini_llm_trace::{Event, MemorySink};
use mini_llm_train::{
    load_checkpoint, load_training_checkpoint, save_checkpoint, save_training_checkpoint,
    LrSchedule, TrainConfig, TrainError, Trainer,
};

fn tokenizer() -> CharTokenizer {
    CharTokenizer::from_text("abcde")
}

fn config(vocab_size: usize) -> ModelConfig {
    ModelConfig {
        vocab_size,
        hidden_size: 16,
        num_layers: 1,
        num_heads: 2,
        max_seq_len: 8,
        num_kv_heads: 2,
        ..ModelConfig::default()
    }
}

fn dataset() -> TextDataset {
    let text = "abcdeabcdeabcdeabcdeabcdeabcde";
    let tokens = tokenizer().encode(text).unwrap();
    TextDataset::new(tokens, 8).unwrap()
}

fn train_config() -> TrainConfig {
    TrainConfig {
        steps: 0,
        batch_size: 2,
        learning_rate: 0.01,
        weight_decay: 0.0,
        seed: 3,
    }
}

fn trainer(seed: u64) -> Trainer<TextDataset> {
    let model = TinyGpt::new(config(5), &mut Rng::new(seed)).unwrap();
    Trainer::new(model, dataset(), train_config())
}

fn temp_path(name: &str) -> PathBuf {
    std::env::temp_dir().join(format!("mini-llm-{}-{}.bin", std::process::id(), name))
}

#[test]
fn training_reduces_the_loss() {
    let mut trainer = trainer(1);
    let mut losses = Vec::new();
    trainer
        .train(20, |report| losses.push(report.loss))
        .unwrap();

    let first: f32 = losses.iter().take(5).sum::<f32>() / 5.0;
    let last: f32 = losses.iter().rev().take(5).sum::<f32>() / 5.0;
    assert!(last < first, "loss did not decrease: {first} -> {last}");
}

#[test]
fn training_is_reproducible_for_a_seed() {
    let mut first = trainer(7);
    let mut second = trainer(7);
    first.train(5, |_| {}).unwrap();
    second.train(5, |_| {}).unwrap();

    let a = first.model().parameters();
    let b = second.model().parameters();
    assert_eq!(a.len(), b.len());
    for (left, right) in a.iter().zip(b.iter()) {
        assert_eq!(left.data(), right.data());
    }
}

#[test]
fn checkpoint_roundtrip_preserves_parameters() {
    let mut trainer = trainer(11);
    trainer.train(3, |_| {}).unwrap();
    let path = temp_path("roundtrip");

    save_checkpoint(&path, trainer.model(), &tokenizer()).unwrap();
    let (loaded, loaded_tokenizer) = load_checkpoint(&path).unwrap();
    std::fs::remove_file(&path).ok();

    assert_eq!(loaded_tokenizer.characters(), tokenizer().characters());
    assert_eq!(loaded.config().hidden_size, 16);
    let original = trainer.model().parameters();
    let restored = loaded.parameters();
    assert_eq!(original.len(), restored.len());
    for (left, right) in original.iter().zip(restored.iter()) {
        assert_eq!(left.data(), right.data());
    }
}

#[test]
fn checkpoint_rejects_foreign_files() {
    let path = temp_path("garbage");
    std::fs::write(&path, b"not a checkpoint").unwrap();
    let result = load_checkpoint(&path);
    std::fs::remove_file(&path).ok();
    assert!(matches!(result, Err(TrainError::BadMagic)));
}

#[test]
fn checkpoint_rejects_truncated_files() {
    let mut trainer = trainer(13);
    trainer.train(1, |_| {}).unwrap();
    let path = temp_path("truncated");
    save_checkpoint(&path, trainer.model(), &tokenizer()).unwrap();
    let bytes = std::fs::read(&path).unwrap();
    std::fs::write(&path, &bytes[..bytes.len() / 2]).unwrap();
    let result = load_checkpoint(&path);
    std::fs::remove_file(&path).ok();
    assert!(
        matches!(result, Err(TrainError::Truncated)),
        "unexpected: {:?}",
        result.err()
    );
}

#[test]
fn evaluate_returns_a_finite_loss() {
    let mut trainer = trainer(17);
    let loss = trainer.evaluate(2).unwrap();
    assert!(loss.is_finite());
    assert!(loss > 0.0);
}

#[test]
fn checkpoint_preserves_the_modern_architecture() {
    use mini_llm_model::{FeedForwardKind, NormKind, PositionKind};

    let config = ModelConfig {
        vocab_size: 5,
        hidden_size: 16,
        num_layers: 1,
        num_heads: 2,
        max_seq_len: 8,
        num_kv_heads: 1,
        norm: NormKind::RmsNorm,
        position: PositionKind::Rope,
        feed_forward: FeedForwardKind::SwiGlu,
        num_experts: 1,
    };
    let mut rng = Rng::new(23);
    let model = TinyGpt::new(config, &mut rng).unwrap();
    let path = temp_path("modern");

    save_checkpoint(&path, &model, &tokenizer()).unwrap();
    let (loaded, _) = load_checkpoint(&path).unwrap();
    std::fs::remove_file(&path).ok();

    assert_eq!(loaded.config().norm, NormKind::RmsNorm);
    assert_eq!(loaded.config().position, PositionKind::Rope);
    assert_eq!(loaded.config().feed_forward, FeedForwardKind::SwiGlu);
    assert_eq!(loaded.config().num_kv_heads, 1);
    assert_eq!(loaded.num_parameters(), model.num_parameters());
}

#[test]
fn observed_training_emits_structured_events() {
    let mut trainer = trainer(19);
    let mut sink = MemorySink::default();
    trainer.train_observed(3, &mut sink).unwrap();

    assert!(matches!(
        sink.events.first(),
        Some(Event::RunStarted { .. })
    ));
    let steps = sink
        .events
        .iter()
        .filter(|event| matches!(event, Event::Step { .. }))
        .count();
    assert_eq!(steps, 3);
    assert!(matches!(
        sink.events.last(),
        Some(Event::RunCompleted { .. })
    ));
}

#[test]
fn warmup_cosine_schedule_has_the_expected_shape() {
    let schedule = LrSchedule::WarmupCosine {
        warmup_steps: 4,
        min_ratio: 0.1,
    };
    // Warmup ramps linearly to the base rate.
    assert!((schedule.learning_rate(1.0, 1, 100) - 0.25).abs() < 1e-6);
    assert!((schedule.learning_rate(1.0, 4, 100) - 1.0).abs() < 1e-6);
    // Cosine decay never leaves the [min_ratio, 1] band and reaches the floor.
    let middle = schedule.learning_rate(1.0, 52, 100);
    assert!((0.1..=1.0).contains(&middle), "middle rate {middle}");
    assert!((schedule.learning_rate(1.0, 100, 100) - 0.1).abs() < 1e-5);
}

#[test]
fn gradient_clipping_bounds_the_reported_norm() {
    let mut trainer = trainer(5).with_grad_clip(0.01);
    let loss = trainer.step().unwrap();
    assert!(loss.is_finite());
    assert!(
        trainer.last_grad_norm() <= 0.0101,
        "clip did not bound the norm: {}",
        trainer.last_grad_norm()
    );
}

#[test]
fn resumed_training_matches_the_uninterrupted_run() {
    let mut first = trainer(1);
    first.train(3, |_| {}).unwrap();
    let path = temp_path("resume");
    save_training_checkpoint(&path, first.model(), &tokenizer(), &first.state()).unwrap();
    let next_loss = first.step().unwrap();

    let (model, tokenizer, state) = load_training_checkpoint(&path).unwrap();
    assert_eq!(tokenizer.vocab_size(), 5);
    let mut resumed = Trainer::new(model, dataset(), train_config());
    resumed.restore_state(&state).unwrap();
    assert_eq!(resumed.step_index(), 3);
    let resumed_loss = resumed.step().unwrap();
    assert_eq!(
        next_loss, resumed_loss,
        "resumed step diverged from the uninterrupted run"
    );

    let _ = std::fs::remove_file(path);
}

struct FixedDataset {
    batch: Batch,
}

impl Dataset for FixedDataset {
    fn block_size(&self) -> usize {
        self.batch.seq
    }

    fn sample_batch(
        &self,
        _batch_size: usize,
        _rng: &mut Rng,
    ) -> Result<Batch, mini_llm_data::DataError> {
        Ok(self.batch.clone())
    }
}

fn fixed_dataset() -> FixedDataset {
    FixedDataset {
        batch: Batch {
            inputs: vec![0, 1, 2, 3],
            targets: vec![1, 2, 3, 4],
            weights: vec![1.0; 4],
            batch: 1,
            seq: 4,
        },
    }
}

#[test]
fn gradient_accumulation_matches_a_single_step() {
    // Two identical micro-batches average back to the single-batch gradient,
    // so the accumulated run must match the plain one exactly.
    let accumulated_model = TinyGpt::new(config(5), &mut Rng::new(1)).unwrap();
    let single_model = TinyGpt::new(config(5), &mut Rng::new(1)).unwrap();
    let mut accumulated =
        Trainer::new(accumulated_model, fixed_dataset(), train_config()).with_accumulation_steps(2);
    let mut single = Trainer::new(single_model, fixed_dataset(), train_config());

    let accumulated_loss = accumulated.step().unwrap();
    let single_loss = single.step().unwrap();
    assert!((accumulated_loss - single_loss).abs() < 1e-6);
    assert_eq!(accumulated.step_index(), 1);
    assert_eq!(accumulated.accumulation_steps(), 2);

    for (left, right) in accumulated
        .model()
        .parameters()
        .iter()
        .zip(single.model().parameters().iter())
    {
        for (a, b) in left.data().iter().zip(right.data()) {
            assert!((a - b).abs() < 1e-6, "parameter diverged: {a} vs {b}");
        }
    }
}
