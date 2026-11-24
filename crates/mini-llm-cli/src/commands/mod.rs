//! One module per command, so each demo owns its imports and dependencies.

pub mod autograd;
pub mod basics;
pub mod bench;
pub mod classify;
pub mod dpo;
pub mod eval;
pub mod generate;
pub mod grpo;
pub mod lora;
pub mod moe;
pub mod regression;
pub mod sft;
pub mod trace;
pub mod train;
pub mod xor;

/// Where `train` writes and `generate` reads the Tiny GPT checkpoint.
pub(crate) const CHECKPOINT_PATH: &str = "checkpoints/tiny-gpt.bin";

/// Reads `--corpus <path>` / `--corpus=<path>`, falling back to the built-in text.
///
/// Returns the text and a human-readable label for logging.
pub(crate) fn corpus_text(arguments: &[String]) -> (String, String) {
    let path = arguments
        .iter()
        .position(|argument| argument == "--corpus")
        .and_then(|index| arguments.get(index + 1).cloned())
        .or_else(|| {
            arguments
                .iter()
                .find_map(|argument| argument.strip_prefix("--corpus=").map(str::to_string))
        });

    match path {
        Some(path) => match std::fs::read_to_string(&path) {
            Ok(text) => (text, path),
            Err(error) => {
                eprintln!("cannot read corpus `{path}`: {error}");
                std::process::exit(2);
            }
        },
        None => (TINY_CORPUS.to_string(), "built-in corpus".to_string()),
    }
}

/// A short public-domain-style paragraph used for the character-level demo.
pub(crate) const TINY_CORPUS: &str = "\
The little language model learns one character at a time. It reads a short story, \
predicts the next letter, and slowly discovers words, spaces and punctuation. \
Small models can still surprise you: they learn rhythm before meaning, and grammar \
before grace. Training is cheap here, so experiments are fast. The model does not \
know what a word is; it only knows which letter tends to follow which. That is \
enough to begin.";
