use std::convert::Infallible;
use std::net::SocketAddr;
use std::path::PathBuf;
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::Arc;

use axum::extract::State;
use axum::http::StatusCode;
use axum::response::sse::{Event, KeepAlive, Sse};
use axum::response::IntoResponse;
use axum::routing::{get, post};
use axum::{Json, Router};
use mini_llm_model::Sampling;
use mini_llm_nn::Rng;
use serde::{Deserialize, Serialize};
use tokio::sync::mpsc;
use tokio_stream::wrappers::ReceiverStream;

const DEFAULT_QUEUE: usize = 32;
const MAX_TOKENS: usize = 512;

#[derive(Clone)]
struct AppState {
    jobs: mpsc::Sender<Job>,
    ready: Arc<AtomicBool>,
}

struct Job {
    request: GenerateRequest,
    events: mpsc::Sender<Result<Event, Infallible>>,
}

#[derive(Debug, Deserialize)]
struct GenerateRequest {
    prompt: String,
    #[serde(default = "default_tokens")]
    tokens: usize,
    #[serde(default = "default_temperature")]
    temperature: f32,
    top_k: Option<usize>,
    top_p: Option<f32>,
    #[serde(default = "default_seed")]
    seed: u64,
}

#[derive(Serialize)]
struct Health<'a> {
    status: &'a str,
}

fn default_tokens() -> usize {
    32
}
fn default_temperature() -> f32 {
    0.8
}
fn default_seed() -> u64 {
    2024
}

#[tokio::main]
async fn main() {
    let (address, checkpoint, queue) = arguments();
    let ready = Arc::new(AtomicBool::new(false));
    let (jobs, receiver) = mpsc::channel(queue);
    start_model_worker(checkpoint, receiver, ready.clone());

    let state = AppState { jobs, ready };
    let app = Router::new()
        .route("/health", get(health))
        .route("/generate", post(generate))
        .with_state(state);
    let listener = tokio::net::TcpListener::bind(address)
        .await
        .expect("bind inference server");
    println!("mini-llm async server listening on http://{address}");
    axum::serve(listener, app)
        .with_graceful_shutdown(shutdown())
        .await
        .expect("serve requests");
}

async fn health(State(state): State<AppState>) -> (StatusCode, Json<Health<'static>>) {
    if state.ready.load(Ordering::Acquire) {
        (StatusCode::OK, Json(Health { status: "ready" }))
    } else {
        (
            StatusCode::SERVICE_UNAVAILABLE,
            Json(Health { status: "loading" }),
        )
    }
}

async fn generate(
    State(state): State<AppState>,
    Json(mut request): Json<GenerateRequest>,
) -> Result<impl IntoResponse, StatusCode> {
    if !state.ready.load(Ordering::Acquire) {
        return Err(StatusCode::SERVICE_UNAVAILABLE);
    }
    request.tokens = request.tokens.clamp(1, MAX_TOKENS);
    request.temperature = request.temperature.clamp(0.0, 2.0);
    request.top_p = request.top_p.map(|value| value.clamp(0.0, 1.0));
    request.top_k = request.top_k.map(|value| value.max(1));
    let (events, stream) = mpsc::channel(16);
    state
        .jobs
        .try_send(Job { request, events })
        .map_err(|_| StatusCode::TOO_MANY_REQUESTS)?;
    Ok(Sse::new(ReceiverStream::new(stream)).keep_alive(KeepAlive::default()))
}

fn start_model_worker(checkpoint: PathBuf, mut jobs: mpsc::Receiver<Job>, ready: Arc<AtomicBool>) {
    std::thread::Builder::new()
        .name("mini-llm-model".into())
        .spawn(move || {
            let (model, tokenizer) = match mini_llm_train::load_checkpoint(&checkpoint) {
                Ok(loaded) => loaded,
                Err(error) => {
                    eprintln!("cannot load {}: {error}", checkpoint.display());
                    return;
                }
            };
            ready.store(true, Ordering::Release);
            while let Some(job) = jobs.blocking_recv() {
                let encoded = match tokenizer.encode(&job.request.prompt) {
                    Ok(tokens) if !tokens.is_empty() => tokens,
                    Ok(_) => vec![0],
                    Err(error) => {
                        let _ = job.events.blocking_send(Ok(Event::default()
                            .event("error")
                            .data(error.to_string())));
                        continue;
                    }
                };
                let sampling = Sampling {
                    temperature: job.request.temperature,
                    top_k: job.request.top_k,
                    top_p: job.request.top_p,
                };
                let mut rng = Rng::new(job.request.seed);
                match model.generate_with(&encoded, job.request.tokens, sampling, &mut rng) {
                    Ok(tokens) => {
                        for token in &tokens[encoded.len()..] {
                            let text = tokenizer.decode(&[*token]).unwrap_or_default();
                            if job
                                .events
                                .blocking_send(Ok(Event::default().event("token").data(text)))
                                .is_err()
                            {
                                break;
                            }
                        }
                        let _ = job
                            .events
                            .blocking_send(Ok(Event::default().event("done").data("{}")));
                    }
                    Err(error) => {
                        let _ = job.events.blocking_send(Ok(Event::default()
                            .event("error")
                            .data(error.to_string())));
                    }
                }
            }
        })
        .expect("start model worker");
}

async fn shutdown() {
    let _ = tokio::signal::ctrl_c().await;
}

fn arguments() -> (SocketAddr, PathBuf, usize) {
    let mut address = "127.0.0.1:8080".parse().unwrap();
    let mut checkpoint = PathBuf::from("checkpoints/tiny-gpt.bin");
    let mut queue = DEFAULT_QUEUE;
    let args: Vec<String> = std::env::args().skip(1).collect();
    let mut index = 0;
    while index < args.len() {
        match args[index].as_str() {
            "--addr" => {
                address = args
                    .get(index + 1)
                    .expect("--addr value")
                    .parse()
                    .expect("valid address")
            }
            "--checkpoint" => checkpoint = args.get(index + 1).expect("--checkpoint value").into(),
            "--queue" => {
                queue = args
                    .get(index + 1)
                    .expect("--queue value")
                    .parse::<usize>()
                    .expect("valid queue")
                    .max(1)
            }
            other => panic!("unknown argument: {other}"),
        }
        index += 2;
    }
    (address, checkpoint, queue)
}
