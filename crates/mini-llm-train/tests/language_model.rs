//! The trainer must not depend on the Transformer: a different
//! [`LanguageModel`] trains with the same loop.

use mini_llm_autograd::TensorValue;
use mini_llm_data::TextDataset;
use mini_llm_model::{LanguageModel, ModelError};
use mini_llm_train::{TrainConfig, Trainer};

/// A bigram model: the logits for the next token are the row of `table` indexed
/// by the current token. Not a Transformer, but a valid next-token model whose
/// loss is minimizable, which makes it a good stand-in for "some other
/// architecture".
struct BigramModel {
    table: TensorValue,
    vocab: usize,
    context: usize,
}

impl BigramModel {
    fn new(vocab: usize, context: usize) -> Self {
        Self {
            table: TensorValue::parameter(vec![0.0; vocab * vocab], vec![vocab, vocab]).unwrap(),
            vocab,
            context,
        }
    }
}

impl LanguageModel for BigramModel {
    fn forward(&self, tokens: &[usize], _batch: usize) -> Result<TensorValue, ModelError> {
        self.table.gather_rows(tokens).map_err(ModelError::Autograd)
    }

    fn parameters(&self) -> Vec<TensorValue> {
        vec![self.table.clone()]
    }

    fn vocab_size(&self) -> usize {
        self.vocab
    }

    fn context_len(&self) -> usize {
        self.context
    }
}

fn train_config() -> TrainConfig {
    TrainConfig {
        steps: 0,
        batch_size: 8,
        learning_rate: 0.2,
        weight_decay: 0.0,
        seed: 5,
    }
}

#[test]
fn trainer_trains_a_model_that_is_not_tiny_gpt() {
    let vocab = 8;
    let context = 8;
    // A cyclic corpus: the next token is always `current + 1`, which a bigram
    // table can fit exactly.
    let tokens: Vec<usize> = (0..4000).map(|index| index % vocab).collect();
    let dataset = TextDataset::new(tokens, context).unwrap();

    let mut trainer = Trainer::new(BigramModel::new(vocab, context), dataset, train_config());
    assert_eq!(trainer.model().num_parameters(), vocab * vocab);
    assert_eq!(trainer.model().vocab_size(), vocab);
    assert_eq!(trainer.model().context_len(), context);

    let before = trainer.evaluate(8).unwrap();
    trainer.train(120, |_| {}).unwrap();
    let after = trainer.evaluate(8).unwrap();

    assert!(
        after < before * 0.5,
        "bigram loss should fall well below the start: {before} -> {after}"
    );
}

#[test]
fn trainer_updates_the_parameters_it_was_given() {
    let vocab = 4;
    let tokens: Vec<usize> = (0..512).map(|index| index % vocab).collect();
    let dataset = TextDataset::new(tokens, 4).unwrap();

    let mut trainer = Trainer::new(BigramModel::new(vocab, 4), dataset, train_config());
    let before = trainer.model().table.data();
    trainer.train(20, |_| {}).unwrap();
    let after = trainer.model().table.data();

    assert!(
        before.iter().zip(&after).any(|(old, new)| old != new),
        "training must mutate the model's parameter handles in place"
    );
}
