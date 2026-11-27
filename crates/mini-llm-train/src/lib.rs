//! # mini-llm-train
//!
//! A minimal training loop and a self-describing binary checkpoint format.
//!
//! [`Trainer`] is generic over [`mini_llm_model::LanguageModel`] and [`Dataset`],
//! so the loop only depends on next-token logits and parameter handles; `TinyGpt`
//! is the reference model. Checkpointing is specific to `TinyGpt` today.
//!
//! ```no_run
//! use mini_llm_data::{CharTokenizer, TextDataset};
//! use mini_llm_model::{ModelConfig, TinyGpt};
//! use mini_llm_nn::Rng;
//! use mini_llm_train::{TrainConfig, Trainer};
//!
//! let corpus = "hello mini-llm, hello world";
//! let tokenizer = CharTokenizer::from_text(corpus);
//! let tokens = tokenizer.encode(corpus).unwrap();
//! let dataset = TextDataset::new(tokens, 8).unwrap();
//!
//! let config = ModelConfig {
//!     vocab_size: tokenizer.vocab_size(),
//!     hidden_size: 32,
//!     num_layers: 1,
//!     num_heads: 2,
//!     max_seq_len: 8,
//!     num_kv_heads: 2,
//!     ..ModelConfig::default()
//! };
//! let model = TinyGpt::new(config, &mut Rng::new(1)).unwrap();
//! let mut trainer = Trainer::new(model, dataset, TrainConfig::default());
//! trainer.train(10, |_report| {}).unwrap();
//! ```
//!
//! ## Checkpoint format
//!
//! Little-endian, no external dependencies:
//!
//! ```text
//! magic "MINILLM1" | version u32 | 10 config u32s
//! vocab_size u32 | vocab_size * (char as u32)
//! parameter_count u32
//! parameter_count * { rank u32 | dims u32*rank | f32 data }
//! ```

mod checkpoint;
mod recipes;
mod safetensor;

pub use checkpoint::{
    load_checkpoint, load_training_checkpoint, save_checkpoint, save_checkpoint_f16,
    save_training_checkpoint, save_training_checkpoint_f16,
};
pub use recipes::{dpo_loss, grpo_loss, sequence_log_prob};
pub use safetensor::{load_safetensors, save_safetensors};

use std::error::Error;
use std::fmt::{Display, Formatter};
use std::time::Instant;

use mini_llm_autograd::TensorValue;
use mini_llm_data::{Batch, DataError, Dataset};
use mini_llm_model::{LanguageModel, ModelError};
use mini_llm_nn::{AdamW, NnError, Optimizer, Rng};
use mini_llm_trace::{Event, EventSink};

/// Hyperparameters for [`Trainer`].
#[derive(Debug, Clone)]
pub struct TrainConfig {
    pub steps: usize,
    pub batch_size: usize,
    pub learning_rate: f32,
    pub weight_decay: f32,
    pub seed: u64,
}

impl Default for TrainConfig {
    fn default() -> Self {
        Self {
            steps: 200,
            batch_size: 4,
            learning_rate: 3e-3,
            weight_decay: 0.01,
            seed: 1,
        }
    }
}

/// Tracks validation loss and decides when to stop.
#[derive(Debug, Clone)]
pub struct EarlyStopping {
    patience: usize,
    best: f32,
    stale: usize,
}

impl EarlyStopping {
    /// `patience == 0` disables stopping; improvements are still tracked.
    pub fn new(patience: usize) -> Self {
        Self {
            patience,
            best: f32::INFINITY,
            stale: 0,
        }
    }

    /// Records one validation loss; returns true when it is a new best.
    pub fn observe(&mut self, loss: f32) -> bool {
        if loss < self.best {
            self.best = loss;
            self.stale = 0;
            true
        } else {
            self.stale += 1;
            false
        }
    }

    /// True once `patience` consecutive observations failed to improve.
    pub fn should_stop(&self) -> bool {
        self.patience > 0 && self.stale >= self.patience
    }

    pub fn best(&self) -> f32 {
        self.best
    }

    pub fn stale_evaluations(&self) -> usize {
        self.stale
    }
}

/// One training step's report.
#[derive(Debug, Clone, Copy)]
pub struct StepReport {
    pub step: usize,
    pub loss: f32,
    pub elapsed_ms: f32,
    pub learning_rate: f32,
    pub grad_norm: f32,
}

/// How the learning rate changes over a run.
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum LrSchedule {
    /// Keep `TrainConfig::learning_rate` for every step.
    Constant,
    /// Linear warmup followed by a cosine decay to `min_ratio * base`.
    WarmupCosine { warmup_steps: usize, min_ratio: f32 },
}

impl LrSchedule {
    /// Learning rate for `step` (1-based) of a run with `total` steps.
    pub fn learning_rate(&self, base: f32, step: usize, total: usize) -> f32 {
        match *self {
            Self::Constant => base,
            Self::WarmupCosine {
                warmup_steps,
                min_ratio,
            } => {
                let min_ratio = min_ratio.clamp(0.0, 1.0);
                if warmup_steps > 0 && step <= warmup_steps {
                    return base * step as f32 / warmup_steps as f32;
                }
                let decay_steps = total.saturating_sub(warmup_steps).max(1);
                let progress =
                    (step.saturating_sub(warmup_steps) as f32 / decay_steps as f32).min(1.0);
                let cosine = 0.5 * (1.0 + (std::f32::consts::PI * progress).cos());
                base * (min_ratio + (1.0 - min_ratio) * cosine)
            }
        }
    }
}

/// Everything needed to resume a run exactly: step counter, RNG and Adam state.
#[derive(Debug, Clone, PartialEq)]
pub struct TrainingState {
    pub step: usize,
    pub rng_state: u64,
    pub optimizer_step: u32,
    pub first_moments: Vec<Vec<f32>>,
    pub second_moments: Vec<Vec<f32>>,
}

/// Owns a model, an optimizer and a dataset, and runs next-token training.
///
/// Generic over [`LanguageModel`]: `TinyGpt` is the reference implementation,
/// and any other architecture that produces next-token logits trains with the
/// same loop.
pub struct Trainer<M: LanguageModel, D: Dataset> {
    model: M,
    optimizer: AdamW,
    dataset: D,
    rng: Rng,
    config: TrainConfig,
    /// Parameter handles collected once; the optimizer mutates them in place.
    parameters: Vec<TensorValue>,
    schedule: LrSchedule,
    grad_clip: Option<f32>,
    accumulation_steps: usize,
    step_index: usize,
    last_grad_norm: f32,
}

impl<M: LanguageModel, D: Dataset> Trainer<M, D> {
    pub fn new(model: M, dataset: D, config: TrainConfig) -> Self {
        let optimizer = AdamW::new(config.learning_rate).with_weight_decay(config.weight_decay);
        let rng = Rng::new(config.seed);
        let parameters = model.parameters();
        Self {
            model,
            optimizer,
            dataset,
            rng,
            config,
            parameters,
            schedule: LrSchedule::Constant,
            grad_clip: None,
            accumulation_steps: 1,
            step_index: 0,
            last_grad_norm: 0.0,
        }
    }

    /// Averaged over this many micro-batches before one optimizer update.
    ///
    /// The effective batch size is `batch_size * accumulation_steps`.
    pub fn with_accumulation_steps(mut self, steps: usize) -> Self {
        self.accumulation_steps = steps.max(1);
        self
    }

    /// Uses a learning-rate schedule instead of the constant rate.
    pub fn with_schedule(mut self, schedule: LrSchedule) -> Self {
        self.schedule = schedule;
        self
    }

    /// Clips gradients to a global norm of `max_norm`.
    pub fn with_grad_clip(mut self, max_norm: f32) -> Self {
        self.grad_clip = Some(max_norm.max(0.0));
        self
    }

    pub fn model(&self) -> &M {
        &self.model
    }

    pub fn dataset(&self) -> &D {
        &self.dataset
    }

    pub fn config(&self) -> &TrainConfig {
        &self.config
    }

    /// The cached parameter handles, in a stable order.
    pub fn parameters(&self) -> &[TensorValue] {
        &self.parameters
    }

    /// Samples `accumulation_steps` micro-batches, accumulates their gradients,
    /// then performs one optimizer update.
    ///
    /// Returns the mean loss of the micro-batches *before* the update.
    pub fn step(&mut self) -> Result<f32, TrainError> {
        self.step_index += 1;
        let learning_rate = self.schedule.learning_rate(
            self.config.learning_rate,
            self.step_index,
            self.config.steps,
        );
        self.optimizer.set_learning_rate(learning_rate);

        let micro_batches = self.accumulation_steps;
        let mut total_loss = 0.0;
        for _ in 0..micro_batches {
            let batch = self
                .dataset
                .sample_batch(self.config.batch_size, &mut self.rng)?;
            let loss = self.loss(&batch)?;
            loss.backward()?;
            total_loss += loss.data()[0];
            // Free this graph before the next micro-batch: it is what keeps the
            // parameter buffers shared.
            drop(loss);
        }

        if micro_batches > 1 {
            let factor = 1.0 / micro_batches as f32;
            for parameter in &self.parameters {
                parameter.scale_grad(factor)?;
            }
        }

        self.last_grad_norm = self.grad_norm();
        if let Some(max_norm) = self.grad_clip {
            if self.last_grad_norm > max_norm && self.last_grad_norm > 0.0 {
                let factor = max_norm / self.last_grad_norm;
                for parameter in &self.parameters {
                    parameter.scale_grad(factor)?;
                }
                self.last_grad_norm = max_norm;
            }
        }

        let parameters = &self.parameters;
        self.optimizer.step(parameters)?;
        for parameter in parameters {
            parameter.zero_grad();
        }
        Ok(total_loss / micro_batches as f32)
    }

    /// Sum of squared gradients across all parameters, square-rooted.
    fn grad_norm(&self) -> f32 {
        self.parameters
            .iter()
            .filter_map(|parameter| parameter.grad())
            .map(|grad| grad.data().iter().map(|value| value * value).sum::<f32>())
            .sum::<f32>()
            .sqrt()
    }

    /// Global gradient norm measured during the most recent step.
    pub fn last_grad_norm(&self) -> f32 {
        self.last_grad_norm
    }

    /// Micro-batches accumulated per optimizer update.
    pub fn accumulation_steps(&self) -> usize {
        self.accumulation_steps
    }

    /// Number of optimizer updates performed so far (across restores).
    pub fn step_index(&self) -> usize {
        self.step_index
    }

    /// Learning rate used by the most recent step.
    pub fn learning_rate(&self) -> f32 {
        self.optimizer.learning_rate()
    }

    /// Snapshot of the run state for checkpointing.
    pub fn state(&self) -> TrainingState {
        let (first_moments, second_moments) = self.optimizer.export_moments(&self.parameters);
        TrainingState {
            step: self.step_index,
            rng_state: self.rng.state(),
            optimizer_step: self.optimizer.step_count(),
            first_moments,
            second_moments,
        }
    }

    /// Restores a snapshot produced by [`Trainer::state`].
    pub fn restore_state(&mut self, state: &TrainingState) -> Result<(), TrainError> {
        for (index, parameter) in self.parameters.iter().enumerate() {
            let length = parameter.value().data().len();
            let first = state.first_moments.get(index);
            let second = state.second_moments.get(index);
            match (first, second) {
                (Some(first), Some(second)) if first.len() == length && second.len() == length => {}
                (Some(first), Some(second)) => {
                    return Err(TrainError::StateShape {
                        index,
                        expected: length,
                        actual: first.len().max(second.len()),
                    });
                }
                _ => {
                    return Err(TrainError::StateShape {
                        index,
                        expected: self.parameters.len(),
                        actual: state.first_moments.len(),
                    });
                }
            }
        }
        self.optimizer.import_moments(
            &self.parameters,
            &state.first_moments,
            &state.second_moments,
        )?;
        self.optimizer.set_step_count(state.optimizer_step);
        self.step_index = state.step;
        self.rng = Rng::resume(state.rng_state);
        Ok(())
    }

    /// Evaluates the mean loss on a held-out dataset without updating anything.
    pub fn evaluate_on(&mut self, dataset: &D, batches: usize) -> Result<f32, TrainError> {
        if batches == 0 {
            return Ok(0.0);
        }
        let mut total = 0.0;
        let _eval = mini_llm_nn::mode::guard();
        for _ in 0..batches {
            let batch = dataset.sample_batch(self.config.batch_size, &mut self.rng)?;
            total += self.loss(&batch)?.data()[0];
        }
        Ok(total / batches as f32)
    }

    /// Forward-only loss for a given batch (no gradients are recorded for
    /// callers, but the graph itself is still built by the engine).
    pub fn loss(&self, batch: &Batch) -> Result<TensorValue, TrainError> {
        let logits = self.model.forward(&batch.inputs, batch.batch)?;
        Ok(logits.cross_entropy_weighted(&batch.targets, &batch.weights)?)
    }

    /// Evaluates the mean loss over randomly sampled batches.
    pub fn evaluate(&mut self, batches: usize) -> Result<f32, TrainError> {
        if batches == 0 {
            return Ok(0.0);
        }
        let mut total = 0.0;
        let _eval = mini_llm_nn::mode::guard();
        for _ in 0..batches {
            let batch = self
                .dataset
                .sample_batch(self.config.batch_size, &mut self.rng)?;
            total += self.loss(&batch)?.data()[0];
        }
        Ok(total / batches as f32)
    }

    /// Runs `steps` updates while emitting structured training events.
    ///
    /// The sink receives a `RunStarted`, one `Step` per update, and either
    /// `RunCompleted` or `RunFailed`.
    pub fn train_observed(
        &mut self,
        steps: usize,
        sink: &mut dyn EventSink,
    ) -> Result<(), TrainError> {
        sink.record(Event::RunStarted {
            steps,
            seed: self.config.seed,
        });
        let mut last_loss = 0.0;
        for step in 1..=steps {
            let started = Instant::now();
            match self.step() {
                Ok(loss) => {
                    last_loss = loss;
                    sink.record(Event::Step {
                        step,
                        loss,
                        elapsed_ms: started.elapsed().as_secs_f32() * 1_000.0,
                    });
                }
                Err(error) => {
                    sink.record(Event::RunFailed {
                        message: error.to_string(),
                    });
                    return Err(error);
                }
            }
        }
        sink.record(Event::RunCompleted {
            steps,
            loss: last_loss,
        });
        Ok(())
    }

    /// Runs `steps` updates, calling `report` after each one.
    pub fn train(
        &mut self,
        steps: usize,
        mut report: impl FnMut(StepReport),
    ) -> Result<(), TrainError> {
        for step in 1..=steps {
            let started = Instant::now();
            let loss = self.step()?;
            report(StepReport {
                step,
                loss,
                elapsed_ms: started.elapsed().as_secs_f32() * 1_000.0,
                learning_rate: self.learning_rate(),
                grad_norm: self.last_grad_norm(),
            });
        }
        Ok(())
    }
}

#[derive(Debug)]
pub enum TrainError {
    Model(ModelError),
    Data(DataError),
    Nn(NnError),
    Io(std::io::Error),
    BadMagic,
    UnsupportedVersion(u32),
    Truncated,
    /// A checkpoint asks for resources outside the educational runtime limits.
    CheckpointLimit {
        field: &'static str,
        value: u64,
        limit: u64,
    },
    SafeTensors(String),
    ShapeMismatch {
        expected: Vec<usize>,
        actual: Vec<usize>,
    },
    InvalidVocabulary,
    /// A checkpoint contains an unknown architecture flag.
    InvalidArchitecture {
        field: &'static str,
        value: u32,
    },
    /// A reward passed to a policy-gradient recipe is not finite.
    InvalidReward {
        index: usize,
        value: f32,
    },
    /// A checkpoint has no training state to resume from.
    MissingTrainingState,
    /// Restored optimizer state does not match the parameter list.
    StateShape {
        index: usize,
        expected: usize,
        actual: usize,
    },
}

impl Display for TrainError {
    fn fmt(&self, f: &mut Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Model(err) => Display::fmt(err, f),
            Self::Data(err) => Display::fmt(err, f),
            Self::Nn(err) => Display::fmt(err, f),
            Self::Io(err) => Display::fmt(err, f),
            Self::BadMagic => write!(f, "checkpoint magic mismatch"),
            Self::UnsupportedVersion(version) => {
                write!(f, "unsupported checkpoint version {}", version)
            }
            Self::Truncated => write!(f, "checkpoint ended unexpectedly"),
            Self::CheckpointLimit {
                field,
                value,
                limit,
            } => write!(
                f,
                "checkpoint {field} is {value}, exceeding the limit {limit}"
            ),
            Self::SafeTensors(message) => write!(f, "SafeTensors error: {message}"),
            Self::ShapeMismatch { expected, actual } => write!(
                f,
                "checkpoint parameter shape {:?} does not match model shape {:?}",
                actual, expected
            ),
            Self::InvalidVocabulary => write!(f, "checkpoint vocabulary is invalid"),
            Self::InvalidArchitecture { field, value } => {
                write!(f, "checkpoint has an unknown {} flag: {}", field, value)
            }
            Self::InvalidReward { index, value } => {
                write!(f, "reward {} is not finite: {}", index, value)
            }
            Self::MissingTrainingState => write!(f, "checkpoint has no training state"),
            Self::StateShape {
                index,
                expected,
                actual,
            } => write!(
                f,
                "training state for parameter {} has {} values, expected {}",
                index, actual, expected
            ),
        }
    }
}

impl Error for TrainError {}

impl From<ModelError> for TrainError {
    fn from(value: ModelError) -> Self {
        Self::Model(value)
    }
}

impl From<DataError> for TrainError {
    fn from(value: DataError) -> Self {
        Self::Data(value)
    }
}

impl From<NnError> for TrainError {
    fn from(value: NnError) -> Self {
        Self::Nn(value)
    }
}

impl From<std::io::Error> for TrainError {
    fn from(value: std::io::Error) -> Self {
        Self::Io(value)
    }
}

impl From<mini_llm_tensor::TensorError> for TrainError {
    fn from(value: mini_llm_tensor::TensorError) -> Self {
        Self::Model(ModelError::Autograd(
            mini_llm_autograd::AutogradError::Tensor(value),
        ))
    }
}

impl From<mini_llm_autograd::AutogradError> for TrainError {
    fn from(value: mini_llm_autograd::AutogradError) -> Self {
        Self::Model(ModelError::Autograd(value))
    }
}
