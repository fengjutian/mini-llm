//! Imports weights-only SafeTensors into a native checkpoint.
//!
//! SafeTensors carries no architecture or tokenizer, so the command needs an
//! existing native checkpoint as the template: it supplies the model config and
//! tokenizer, then only the weights are replaced.

use std::path::PathBuf;

pub fn run(arguments: &[String]) {
    let input = super::value(arguments, "--input").map(PathBuf::from);
    let Some(input) = input else {
        eprintln!("usage: import-safetensors --input PATH [--checkpoint TEMPLATE] [--output PATH]");
        std::process::exit(2);
    };
    let template = super::checkpoint_path(arguments);
    let output = super::value(arguments, "--output")
        .map(PathBuf::from)
        .unwrap_or_else(|| PathBuf::from(&template));

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
    println!(
        "imported {} -> {} (template {template})",
        input.display(),
        output.display()
    );
}
