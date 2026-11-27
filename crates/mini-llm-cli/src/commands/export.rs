//! Converts a native checkpoint into weights-only SafeTensors.

use std::path::PathBuf;

pub fn run(arguments: &[String]) {
    let input = super::checkpoint_path(arguments);
    let output = super::value(arguments, "--output")
        .map(PathBuf::from)
        .unwrap_or_else(|| PathBuf::from("checkpoints/tiny-gpt.safetensors"));
    let (model, _) = match mini_llm_train::load_checkpoint(&input) {
        Ok(loaded) => loaded,
        Err(error) => {
            eprintln!("cannot load {input}: {error}");
            std::process::exit(1);
        }
    };
    if let Some(parent) = output.parent() {
        if let Err(error) = std::fs::create_dir_all(parent) {
            eprintln!("cannot create {}: {error}", parent.display());
            std::process::exit(1);
        }
    }
    if let Err(error) = mini_llm_train::save_safetensors(&output, &model) {
        eprintln!("cannot write {}: {error}", output.display());
        std::process::exit(1);
    }
    println!("exported {input} -> {}", output.display());
}
