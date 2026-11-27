use std::convert::Infallible;
use std::net::SocketAddr;
use std::path::PathBuf;
use std::sync::atomic::{AtomicBool, AtomicU64, Ordering};
use std::sync::Arc;
use std::time::Instant;

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
    metrics: Arc<Metrics>,
    queue_capacity: usize,
}

#[derive(Default)]
struct Metrics {
    started: Option<Instant>,
    generate_requests: AtomicU64,
    tokens_generated: AtomicU64,
    errors: AtomicU64,
    rejected: AtomicU64,
}

impl Metrics {
    fn new() -> Self {
        Self {
            started: Some(Instant::now()),
            ..Self::default()
        }
    }

    fn snapshot(&self, ready: bool, queue_capacity: usize) -> MetricsSnapshot {
        MetricsSnapshot {
            uptime_seconds: self
                .started
                .map(|started| started.elapsed().as_secs())
                .unwrap_or(0),
            ready,
            queue_capacity,
            generate_requests: self.generate_requests.load(Ordering::Relaxed),
            tokens_generated: self.tokens_generated.load(Ordering::Relaxed),
            errors: self.errors.load(Ordering::Relaxed),
            rejected: self.rejected.load(Ordering::Relaxed),
        }
    }
}

#[derive(Serialize)]
struct MetricsSnapshot {
    uptime_seconds: u64,
    ready: bool,
    queue_capacity: usize,
    generate_requests: u64,
    tokens_generated: u64,
    errors: u64,
    rejected: u64,
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
    #[serde(default = "default_repetition_penalty")]
    repetition_penalty: f32,
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
fn default_repetition_penalty() -> f32 {
    1.0
}

fn default_seed() -> u64 {
    2024
}

#[tokio::main]
async fn main() {
    let (address, checkpoint, queue) = arguments();
    let ready = Arc::new(AtomicBool::new(false));
    let metrics = Arc::new(Metrics::new());
    let (jobs, receiver) = mpsc::channel(queue);
    start_model_worker(checkpoint, receiver, ready.clone(), metrics.clone());

    let state = AppState {
        jobs,
        ready,
        metrics,
        queue_capacity: queue,
    };
    let app = Router::new()
        .route("/health", get(health))
        .route("/metrics", get(metrics_view))
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

async fn metrics_view(State(state): State<AppState>) -> Json<MetricsSnapshot> {
    Json(
        state
            .metrics
            .snapshot(state.ready.load(Ordering::Acquire), state.queue_capacity),
    )
}

async fn generate(
    State(state): State<AppState>,
    Json(mut request): Json<GenerateRequest>,
) -> Result<impl IntoResponse, StatusCode> {
    if !state.ready.load(Ordering::Acquire) {
        state.metrics.rejected.fetch_add(1, Ordering::Relaxed);
        return Err(StatusCode::SERVICE_UNAVAILABLE);
    }
    request.tokens = request.tokens.clamp(1, MAX_TOKENS);
    request.temperature = request.temperature.clamp(0.0, 2.0);
    request.top_p = request.top_p.map(|value| value.clamp(0.0, 1.0));
    request.top_k = request.top_k.map(|value| value.max(1));
    let (events, stream) = mpsc::channel(16);
    state
        .metrics
        .generate_requests
        .fetch_add(1, Ordering::Relaxed);
    state.jobs.try_send(Job { request, events }).map_err(|_| {
        state.metrics.rejected.fetch_add(1, Ordering::Relaxed);
        StatusCode::TOO_MANY_REQUESTS
    })?;
    Ok(Sse::new(ReceiverStream::new(stream)).keep_alive(KeepAlive::default()))
}

fn start_model_worker(
    checkpoint: PathBuf,
    mut jobs: mpsc::Receiver<Job>,
    ready: Arc<AtomicBool>,
    metrics: Arc<Metrics>,
) {
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
                        metrics.errors.fetch_add(1, Ordering::Relaxed);
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
                    repetition_penalty: job.request.repetition_penalty,
                };
                let mut rng = Rng::new(job.request.seed);
                match model.generate_with(&encoded, job.request.tokens, sampling, &mut rng) {
                    Ok(tokens) => {
                        metrics
                            .tokens_generated
                            .fetch_add((tokens.len() - encoded.len()) as u64, Ordering::Relaxed);
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
                        metrics.errors.fetch_add(1, Ordering::Relaxed);
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

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn snapshot_reports_counters_and_uptime() {
        let metrics = Metrics::new();
        metrics.generate_requests.fetch_add(3, Ordering::Relaxed);
        metrics.tokens_generated.fetch_add(42, Ordering::Relaxed);
        metrics.errors.fetch_add(1, Ordering::Relaxed);
        metrics.rejected.fetch_add(2, Ordering::Relaxed);
        let snapshot = metrics.snapshot(true, 8);
        assert_eq!(snapshot.generate_requests, 3);
        assert_eq!(snapshot.tokens_generated, 42);
        assert_eq!(snapshot.errors, 1);
        assert_eq!(snapshot.rejected, 2);
        assert!(snapshot.ready);
        assert_eq!(snapshot.queue_capacity, 8);
    }
}
