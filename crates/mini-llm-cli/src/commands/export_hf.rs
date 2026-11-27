//! Exports a self-contained Hugging Face-style directory:
//! `config.json` + `tokenizer.json` + `model.safetensors`.

use std::path::PathBuf;

pub fn run(arguments: &[String]) {
    let input = super::checkpoint_path(arguments);
    let output = super::value(arguments, "--output")
        .map(PathBuf::from)
        .unwrap_or_else(|| PathBuf::from("checkpoints/tiny-gpt-hf"));
    let (model, tokenizer) = match mini_llm_train::load_checkpoint(&input) {
        Ok(loaded) => loaded,
        Err(error) => {
            eprintln!("cannot load {input}: {error}");
            std::process::exit(1);
        }
    };
    if let Err(error) = mini_llm_train::save_hf_bundle(&output, &model, &tokenizer) {
        eprintln!("cannot write bundle {}: {error}", output.display());
        std::process::exit(1);
    }
    println!(
        "exported {input} -> {} (config.json, tokenizer.json, model.safetensors)",
        output.display()
    );
}
