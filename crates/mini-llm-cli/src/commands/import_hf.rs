//! Imports a Hugging Face-style directory written by `export-hf` back into a
//! native training checkpoint (no template checkpoint required).

use std::path::PathBuf;

pub fn run(arguments: &[String]) {
    let input = super::value(arguments, "--input").map(PathBuf::from);
    let Some(input) = input else {
        eprintln!("usage: import-hf --input DIR [--output PATH]");
        std::process::exit(2);
    };
    let output = super::value(arguments, "--output")
        .map(PathBuf::from)
        .unwrap_or_else(|| PathBuf::from("checkpoints/tiny-gpt.bin"));
    let (model, tokenizer) = match mini_llm_train::load_hf_bundle(&input) {
        Ok(loaded) => loaded,
        Err(error) => {
            eprintln!("cannot read bundle {}: {error}", input.display());
            std::process::exit(1);
        }
    };
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
        "imported {} -> {} (native checkpoint)",
        input.display(),
        output.display()
    );
}
