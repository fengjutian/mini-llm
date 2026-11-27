//! One module per command, so each demo owns its imports and dependencies.

pub mod autograd;
pub mod basics;
pub mod bench;
pub mod classify;
pub mod dpo;
pub mod eval;
pub mod export;
pub mod generate;
pub mod grpo;
pub mod lora;
pub mod moe;
pub mod regression;
pub mod serve;
pub mod sft;
pub mod trace;
pub mod train;
pub mod xor;

/// Where `train` writes and `generate` reads the Tiny GPT checkpoint.
pub(crate) const CHECKPOINT_PATH: &str = "checkpoints/tiny-gpt.bin";

/// Where `sft` writes its fine-tuned checkpoint.
pub(crate) const SFT_CHECKPOINT_PATH: &str = "checkpoints/sft.bin";

/// Reads every `--corpus <path>` / `--corpus=<path>`, falling back to the
/// built-in text. A directory contributes its `.txt` and `.md` files in sorted
/// order, so a corpus can be split across many files.
///
/// Returns the concatenated text and a human-readable label for logging.
pub(crate) fn corpus_text(arguments: &[String]) -> (String, String) {
    let mut paths = Vec::new();
    let mut index = 0;
    while index < arguments.len() {
        if arguments[index] == "--corpus" {
            if let Some(path) = arguments.get(index + 1) {
                paths.push(path.clone());
            }
            index += 2;
        } else if let Some(rest) = arguments[index].strip_prefix("--corpus=") {
            paths.push(rest.to_string());
            index += 1;
        } else {
            index += 1;
        }
    }

    if paths.is_empty() {
        return (TINY_CORPUS.to_string(), "built-in corpus".to_string());
    }

    let mut text = String::new();
    let mut labels = Vec::new();
    for path in paths {
        let path_ref = std::path::Path::new(&path);
        if path_ref.is_dir() {
            let mut entries: Vec<std::path::PathBuf> = match std::fs::read_dir(path_ref) {
                Ok(entries) => entries
                    .filter_map(|entry| entry.ok().map(|entry| entry.path()))
                    .filter(|entry| {
                        matches!(
                            entry.extension().and_then(|value| value.to_str()),
                            Some("txt" | "md")
                        )
                    })
                    .collect(),
                Err(error) => {
                    eprintln!("cannot read corpus directory `{path}`: {error}");
                    std::process::exit(2);
                }
            };
            entries.sort();
            for entry in entries {
                match std::fs::read_to_string(&entry) {
                    Ok(contents) => {
                        text.push_str(&contents);
                        text.push('\n');
                        labels.push(entry.display().to_string());
                    }
                    Err(error) => {
                        eprintln!("cannot read `{}`: {error}", entry.display());
                        std::process::exit(2);
                    }
                }
            }
        } else {
            match std::fs::read_to_string(&path) {
                Ok(contents) => {
                    text.push_str(&contents);
                    text.push('\n');
                    labels.push(path.clone());
                }
                Err(error) => {
                    eprintln!("cannot read corpus `{path}`: {error}");
                    std::process::exit(2);
                }
            }
        }
    }

    if text.trim().is_empty() {
        eprintln!("corpus is empty");
        std::process::exit(2);
    }
    (text, labels.join(", "))
}

/// Reads `--checkpoint PATH` / `--checkpoint=PATH`, defaulting to the trainer's.
pub(crate) fn value(arguments: &[String], name: &str) -> Option<String> {
    let mut result = None;
    for (index, argument) in arguments.iter().enumerate() {
        if argument == name {
            result = arguments.get(index + 1).cloned();
        } else if let Some(rest) = argument.strip_prefix(&format!("{name}=")) {
            result = Some(rest.to_string());
        }
    }
    result
}
pub(crate) fn checkpoint_path(arguments: &[String]) -> String {
    let mut value = None;
    for (index, argument) in arguments.iter().enumerate() {
        if argument == "--checkpoint" {
            value = arguments.get(index + 1).cloned();
        } else if let Some(rest) = argument.strip_prefix("--checkpoint=") {
            value = Some(rest.to_string());
        }
    }
    value.unwrap_or_else(|| CHECKPOINT_PATH.to_string())
}

/// Reads `--accum N` / `--accum=N`; defaults to one micro-batch.
pub(crate) fn parse_accumulation(arguments: &[String]) -> usize {
    let mut value = None;
    for (index, argument) in arguments.iter().enumerate() {
        if argument == "--accum" {
            value = arguments.get(index + 1).and_then(|next| next.parse().ok());
        } else if let Some(rest) = argument.strip_prefix("--accum=") {
            value = rest.parse().ok();
        }
    }
    value.unwrap_or(1).max(1)
}

/// Reads `--early-stop N` (evaluations without improvement); 0 disables it.
pub(crate) fn parse_early_stop(arguments: &[String]) -> usize {
    let mut value = None;
    for (index, argument) in arguments.iter().enumerate() {
        if argument == "--early-stop" {
            value = arguments.get(index + 1).and_then(|next| next.parse().ok());
        } else if let Some(rest) = argument.strip_prefix("--early-stop=") {
            value = rest.parse().ok();
        }
    }
    value.unwrap_or(0)
}

/// A short public-domain-style paragraph used for the character-level demo.
pub(crate) const TINY_CORPUS: &str = "\
The little language model learns one character at a time. It reads a short story, \
predicts the next letter, and slowly discovers words, spaces and punctuation. \
Small models can still surprise you: they learn rhythm before meaning, and grammar \
before grace. Training is cheap here, so experiments are fast. The model does not \
know what a word is; it only knows which letter tends to follow which. That is \
enough to begin.";
