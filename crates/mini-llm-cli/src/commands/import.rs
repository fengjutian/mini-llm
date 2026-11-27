//! Imports SafeTensors into a native checkpoint.
//!
//! Two modes:
//! - `--bundle DIR`: reconstruct model and tokenizer from `config.json` /
//!   `tokenizer.json` / `model.safetensors` (`load_hf_bundle`);
//! - default: load weights into the architecture and tokenizer of an existing
//!   native checkpoint (`--checkpoint` acts as the template).

use std::path::PathBuf;

pub fn run(arguments: &[String]) {
    let output = super::value(arguments, "--output")
        .map(PathBuf::from)
        .unwrap_or_else(|| PathBuf::from("checkpoints/tiny-gpt.bin"));

    let loaded = if let Some(bundle) = super::value(arguments, "--bundle").map(PathBuf::from) {
        match mini_llm_train::load_hf_bundle(&bundle) {
            Ok(loaded) => {
                println!("loaded bundle {}", bundle.display());
                loaded
            }
            Err(error) => {
                eprintln!("cannot read bundle {}: {error}", bundle.display());
                std::process::exit(1);
            }
        }
    } else {
        let input = super::value(arguments, "--input").map(PathBuf::from);
        let Some(input) = input else {
            eprintln!("usage: import-safetensors --input PATH [--checkpoint TEMPLATE] [--output PATH]");
            eprintln!("       import-safetensors --bundle DIR [--output PATH]");
            std::process::exit(2);
        };
        let template = super::checkpoint_path(arguments);
        let (model, tokenizer) = match mini_llm_train::load_checkpoint(&template) {
            Ok(loaded) => loaded,
            Err(error) => {
                eprintln!("cannot load template {template}: {error}");
                std::process::exit(1);
            }
        };
        if let Err(error) = mini_llm_train::load_safetensors(&input, &model) {
            eprintln!("cannot read {}: {error}", input.display());
            std::process::exit(1);
        }
        (model, tokenizer)
    };

    let (model, tokenizer) = loaded;
    if let Some(parent) = output.parent() {
        if let Err(error) = std::fs::create_dir_all(parent) {
            eprintln!("cannot create {}: {error}", parent.display());
            std::process::exit(1);
        }
    }
    if let Err(error) = mini_llm_train::save_checkpoint(&output, &model, &tokenizer) {
        eprintln!("cannot write {}: {error}", output.display());
        std::process::exit(1);
    }
    println!("imported -> {} (native checkpoint)", output.display());
}
