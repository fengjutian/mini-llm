#![no_main]

use libfuzzer_sys::fuzz_target;
use mini_llm_data::BpeTokenizer;

fuzz_target!(|input: &[u8]| {
    let corpus = String::from_utf8_lossy(&input[..input.len().min(256)]);
    {
        let tokenizer = BpeTokenizer::train(&corpus, 320);
        let encoded = tokenizer.encode(&corpus);
        let decoded = tokenizer.decode(&encoded).expect("encoded corpus decodes");
        assert_eq!(decoded, corpus);
    }
});
