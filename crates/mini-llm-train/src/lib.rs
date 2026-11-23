//! # mini-llm-train
//!
//! A minimal training loop and a self-describing binary checkpoint format.
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
//! magic "MINILLM1" | version u32 | 5 config u32s
//! vocab_size u32 | vocab_size * (char as u32)
//! parameter_count u32
//! parameter_count * { rank u32 | dims u32*rank | f32 data }
//! ```

mod checkpoint;

pub use checkpoint::{load_checkpoint, save_checkpoint};

use std::error::Error;
use std::fmt::{Display, Formatter};
use std::time::Instant;

use mini_llm_autograd::TensorValue;
use mini_llm_data::{Batch, DataError, TextDataset};
use mini_llm_model::{ModelError, TinyGpt};
use mini_llm_nn::{AdamW, NnError, Optimizer, Rng};

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

/// One training step's report.
#[derive(Debug, Clone, Copy)]
pub struct StepReport {
    pub step: usize,
    pub loss: f32,
    pub elapsed_ms: f32,
}

/// Owns a model, an optimizer and a dataset, and runs next-token training.
pub struct Trainer {
    model: TinyGpt,
    optimizer: AdamW,
    dataset: TextDataset,
    rng: Rng,
    config: TrainConfig,
}

impl Trainer {
    pub fn new(model: TinyGpt, dataset: TextDataset, config: TrainConfig) -> Self {
        let optimizer = AdamW::new(config.learning_rate).with_weight_decay(config.weight_decay);
        let rng = Rng::new(config.seed);
        Self {
            model,
            optimizer,
            dataset,
            rng,
            config,
        }
    }

    pub fn model(&self) -> &TinyGpt {
        &self.model
    }

    pub fn dataset(&self) -> &TextDataset {
        &self.dataset
    }

    pub fn config(&self) -> &TrainConfig {
        &self.config
    }

    pub fn parameters(&self) -> Vec<TensorValue> {
        self.model.parameters()
    }

    /// Samples one batch, computes the loss, backpropagates and updates.
    ///
    /// Returns the loss *before* the update.
    pub fn step(&mut self) -> Result<f32, TrainError> {
        let batch = self
            .dataset
            .sample_batch(self.config.batch_size, &mut self.rng)?;
        let loss = self.loss(&batch)?;
        loss.backward()?;
        self.optimizer.step(&self.parameters())?;
        loss.zero_grad();
        Ok(loss.data()[0])
    }

    /// Forward-only loss for a given batch (no gradients are recorded for
    /// callers, but the graph itself is still built by the engine).
    pub fn loss(&self, batch: &Batch) -> Result<TensorValue, TrainError> {
        let logits = self.model.forward(&batch.inputs, batch.batch)?;
        Ok(logits.cross_entropy(&batch.targets)?)
    }

    /// Evaluates the mean loss over randomly sampled batches.
    pub fn evaluate(&mut self, batches: usize) -> Result<f32, TrainError> {
        if batches == 0 {
            return Ok(0.0);
        }
        let mut total = 0.0;
        for _ in 0..batches {
            let batch = self
                .dataset
                .sample_batch(self.config.batch_size, &mut self.rng)?;
            total += self.loss(&batch)?.data()[0];
        }
        Ok(total / batches as f32)
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
    ShapeMismatch {
        expected: Vec<usize>,
        actual: Vec<usize>,
    },
    InvalidVocabulary,
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
            Self::ShapeMismatch { expected, actual } => write!(
                f,
                "checkpoint parameter shape {:?} does not match model shape {:?}",
                actual, expected
            ),
            Self::InvalidVocabulary => write!(f, "checkpoint vocabulary is invalid"),
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
