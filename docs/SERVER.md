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
  --output checkpoints/tiny-gpt.safetensors
```

The public `mini_llm_train::load_safetensors` API imports those weights into an
already configured `TinyGpt`. Tokenizer and optimizer state remain in the native
checkpoint and are intentionally not duplicated in the weights-only file.
