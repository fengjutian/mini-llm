//! Thin wrapper around [`mini_llm_backend::device_training::run`].
//!
//! The implementation lives in the library so the CLI can reuse it:
//! `mini-llm-cli device-train --layers 2 --seq 64 --batch 16 --steps 3000`.

fn main() {
    let arguments: Vec<String> = std::env::args().skip(1).collect();
    mini_llm_backend::device_training::run(arguments);
}
