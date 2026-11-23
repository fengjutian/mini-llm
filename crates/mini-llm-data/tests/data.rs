//! Tests for tokenization and batching.

use mini_llm_data::{CharTokenizer, DataError, TextDataset};
use mini_llm_nn::Rng;

#[test]
fn tokenizer_roundtrips_text() {
    let text = "hello world";
    let tokenizer = CharTokenizer::from_text(text);
    let tokens = tokenizer.encode(text).unwrap();
    assert_eq!(tokens.len(), text.chars().count());
    assert_eq!(tokenizer.decode(&tokens).unwrap(), text);
}

#[test]
fn vocabulary_is_sorted_and_unique() {
    let tokenizer = CharTokenizer::from_text("cbabca");
    let characters: Vec<char> = tokenizer.characters().to_vec();
    assert_eq!(characters, vec!['a', 'b', 'c']);
    assert_eq!(tokenizer.vocab_size(), 3);
}

#[test]
fn unknown_characters_are_rejected() {
    let tokenizer = CharTokenizer::from_text("abc");
    assert!(matches!(
        tokenizer.encode("abz"),
        Err(DataError::UnknownCharacter { character: 'z' })
    ));
    assert!(matches!(
        tokenizer.decode(&[3]),
        Err(DataError::TokenOutOfRange { .. })
    ));
}

#[test]
fn batches_are_shifted_by_one_token() {
    let dataset = TextDataset::new(vec![0, 1, 2, 3, 4, 5], 3).unwrap();
    let batch = dataset.batch_at(&[0, 2]).unwrap();
    assert_eq!(batch.batch, 2);
    assert_eq!(batch.seq, 3);
    assert_eq!(batch.inputs, vec![0, 1, 2, 2, 3, 4]);
    assert_eq!(batch.targets, vec![1, 2, 3, 3, 4, 5]);
}

#[test]
fn sampling_is_deterministic_for_a_seed() {
    let dataset = TextDataset::new((0..64).collect(), 8).unwrap();
    let mut first = Rng::new(9);
    let mut second = Rng::new(9);
    let a = dataset.sample_batch(4, &mut first).unwrap();
    let b = dataset.sample_batch(4, &mut second).unwrap();
    assert_eq!(a, b);
    assert_eq!(a.inputs.len(), 32);
}

#[test]
fn rejects_invalid_datasets() {
    assert!(matches!(
        TextDataset::new(Vec::new(), 4),
        Err(DataError::EmptyCorpus)
    ));
    assert!(matches!(
        TextDataset::new(vec![0, 1, 2], 3),
        Err(DataError::BlockTooLarge { .. })
    ));
    let dataset = TextDataset::new(vec![0, 1, 2, 3], 2).unwrap();
    assert!(matches!(
        dataset.batch_at(&[5]),
        Err(DataError::WindowOutOfRange { .. })
    ));
    assert!(matches!(
        dataset.sample_batch(0, &mut Rng::new(1)),
        Err(DataError::EmptyBatch)
    ));
}
