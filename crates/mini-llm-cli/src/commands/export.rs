//! Converts a native checkpoint into a weights-only SafeTensors package:
//! weights plus `config.json` and `tokenizer.json` sidecars.

use std::path::PathBuf;

pub fn run(arguments: &[String]) {
    let input = super::checkpoint_path(arguments);
    let output = super::value(arguments, "--output")
        .map(PathBuf::from)
        .unwrap_or_else(|| PathBuf::from("checkpoints/tiny-gpt.safetensors"));
    let (model, tokenizer) = match mini_llm_train::load_checkpoint(&input) {
        Ok(loaded) => loaded,
        Err(error) => {
            eprintln!("cannot load {input}: {error}");
            std::process::exit(1);
        }
    };
    if let Some(directory) = super::value(arguments, "--bundle").map(PathBuf::from) {
        if let Err(error) = mini_llm_train::save_hf_bundle(&directory, &model, &tokenizer) {
            eprintln!("cannot write bundle {}: {error}", directory.display());
            std::process::exit(1);
        }
        println!("exported {input} -> {}", directory.display());
        return;
    }
    if let Some(parent) = output.parent() {
        if let Err(error) = std::fs::create_dir_all(parent) {
            eprintln!("cannot create {}: {error}", parent.display());
            std::process::exit(1);
        }
    }
    let config_path = output.with_extension("config.json");
    let tokenizer_path = output.with_extension("tokenizer.json");
    for (path, result, label) in [
        (
            &output,
            mini_llm_train::save_safetensors(&output, &model),
            "weights",
        ),
        (
            &config_path,
            mini_llm_train::save_config_json(&config_path, &model),
            "config",
        ),
        (
            &tokenizer_path,
            mini_llm_train::save_tokenizer_json(&tokenizer_path, &tokenizer),
            "tokenizer",
        ),
    ] {
        if let Err(error) = result {
            eprintln!("cannot write {} ({label}): {error}", path.display());
            std::process::exit(1);
        }
    }
    println!(
        "exported {input} -> {} (+ {}, {})",
        output.display(),
        config_path.display(),
        tokenizer_path.display()
    );
}
