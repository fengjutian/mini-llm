# Async inference server

`mini-llm-server` is the deployable counterpart to the dependency-free teaching
server in `mini-llm-cli`. It uses Axum/Tokio for HTTP while a dedicated bounded
worker owns the non-`Send` educational model.

```bash
cargo run -p mini-llm-server -- \
  --checkpoint checkpoints/tiny-gpt.bin \
  --addr 127.0.0.1:8080 \
  --queue 32
```

Readiness:

```bash
curl http://127.0.0.1:8080/health
```

SSE generation:

```bash
curl -N -X POST http://127.0.0.1:8080/generate \
  -H 'content-type: application/json' \
  -d '{"prompt":"The ","tokens":32,"temperature":0.8,"seed":7}'
```

The HTTP side accepts concurrent clients, the queue applies backpressure, and a
full queue returns HTTP 429. Generation remains serialized intentionally because
the teaching autograd graph uses `Rc<RefCell<_>>`.

## SafeTensors exchange

Export weights for inspection with Hugging Face-compatible SafeTensors tooling:

```bash
cargo run -p mini-llm-cli -- export-safetensors \
  --checkpoint checkpoints/tiny-gpt.bin \
  --bundle checkpoints/tiny-gpt-hf
```

The bundle contains `config.json`, `tokenizer.json`, and `model.safetensors`.
`mini_llm_train::load_hf_bundle` reconstructs both the model and tokenizer from
that directory. The lower-level `save_safetensors`/`load_safetensors` APIs remain
available for weights-only exchange. Optimizer state stays in native checkpoints.
