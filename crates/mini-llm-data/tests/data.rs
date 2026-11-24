//! Tests for tokenization and batching.

use mini_llm_data::{
    Batch, BpeTokenizer, CharTokenizer, DataError, Dataset, SftDataset, TextDataset, Tokenizer,
};
use mini_llm_random::Rng;

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

#[test]
fn supervised_encoding_masks_the_prompt() {
    let tokenizer = CharTokenizer::from_text("abcde");
    let (tokens, weights) = tokenizer.encode_supervised("ab", "cde").unwrap();
    assert_eq!(tokens, tokenizer.encode("abcde").unwrap());
    // Predicting "c" (the first response token) is supervised by "b".
    assert_eq!(weights, vec![0.0, 1.0, 1.0, 1.0, 0.0]);
    assert!(matches!(
        tokenizer.encode_supervised("ab", ""),
        Err(DataError::EmptyResponse)
    ));
}

#[test]
fn sft_batches_align_weights_with_targets() {
    let tokenizer = CharTokenizer::from_text("abcdef");
    let (tokens, weights) = tokenizer.encode_supervised("ab", "cdef").unwrap();
    let dataset = SftDataset::new(tokens, weights, 2).unwrap();
    let batch = dataset.batch_at(&[0]).unwrap();
    assert_eq!(batch.seq, 2);
    assert_eq!(batch.inputs, vec![0, 1]); // a b
    assert_eq!(batch.targets, vec![1, 2]); // b c
                                           // Both targets are supervised: "b" is the last prompt token, "c" the first
                                           // response token.
    assert_eq!(batch.weights, vec![1.0, 1.0]);
    assert_eq!(batch.active_positions(), 2);
}

#[test]
fn sft_windows_without_supervision_are_rejected() {
    // Only the last two tokens are supervised, so a window at the start has no
    // signal at all.
    let dataset = SftDataset::new(vec![0, 1, 2, 3, 4, 5], vec![0., 0., 0., 0., 1., 0.], 2).unwrap();
    assert!(matches!(
        dataset.batch_at(&[0]),
        Err(DataError::EmptyMaskWindow)
    ));
    assert!(SftDataset::new(vec![0, 1], vec![1.0], 1).is_err());
}

#[test]
fn sft_sampling_only_draws_supervised_windows() {
    // Only the tail carries supervision, so most windows would be empty.
    let dataset = SftDataset::new(
        vec![0, 1, 2, 3, 4, 5, 6, 7],
        vec![0., 0., 0., 0., 1., 1., 1., 0.],
        2,
    )
    .unwrap();
    assert!(dataset.supervised_windows() < dataset.num_windows());

    let mut rng = Rng::new(11);
    for _ in 0..20 {
        let batch = dataset.sample_batch(2, &mut rng).unwrap();
        assert!(batch.weights.iter().any(|weight| *weight > 0.0));
    }
}

#[test]
fn sft_dataset_rejects_a_fully_masked_corpus() {
    assert!(matches!(
        SftDataset::new(vec![0, 1, 2, 3], vec![0.0; 4], 2),
        Err(DataError::EmptyMaskWindow)
    ));
}

#[test]
fn bpe_roundtrips_any_utf8_text() {
    let tokenizer = BpeTokenizer::train("the quick brown fox jumps over the lazy dog", 300);
    for text in ["the quick brown fox", "lazy dogs jump", "量子计算 and emoji 🚀"] {
        let tokens = tokenizer.encode(text);
        assert!(tokens.iter().all(|&token| token < tokenizer.vocab_size()));
        assert_eq!(tokenizer.decode(&tokens).unwrap(), text);
    }
}

#[test]
fn bpe_merges_compress_the_training_text() {
    let text = "the quick brown fox jumps over the lazy dog. the quick brown fox.";
    let tokenizer = BpeTokenizer::train(text, 300);
    assert!(tokenizer.vocab_size() > 256);
    assert!(tokenizer.encode(text).len() < text.len());
}

#[test]
fn bpe_training_is_deterministic_and_validates_merges() {
    let text = "abababab cdcdcdcd abab abab";
    let first = BpeTokenizer::train(text, 260);
    let second = BpeTokenizer::train(text, 260);
    assert_eq!(first, second);

    assert!(matches!(
        BpeTokenizer::from_merges(vec![(999, 0)]),
        Err(DataError::InvalidMerge { .. })
    ));
    let rebuilt = BpeTokenizer::from_merges(first.merges().to_vec()).unwrap();
    assert_eq!(rebuilt, first);
}

#[test]
fn tokenizer_enum_dispatches_to_both_vocabularies() {
    let chars = Tokenizer::from(CharTokenizer::from_text("abc"));
    assert_eq!(chars.encode("cab").unwrap().len(), 3);
    assert_eq!(chars.vocab_size(), 3);
    assert!(chars.as_char().is_some());

    let bpe = Tokenizer::from(BpeTokenizer::train("abcabc", 260));
    assert!(bpe.as_bpe().is_some());
    assert_eq!(bpe.decode(&bpe.encode("abcabc").unwrap()).unwrap(), "abcabc");
}