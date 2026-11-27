//! `watch-trace`: follow a training trace and stream it as Server-Sent Events.
//!
//! `trace` writes one JSON object per line to `traces/run-<id>.jsonl`. This
//! command does not train and does not parse the events: it replays the lines
//! already on disk, then polls for appended bytes and emits every complete line
//! as one SSE `data:` frame. A browser attached to `/studio` can therefore watch
//! a loss curve grow while `trace` is still running.
//!
//! No async runtime and no dependencies, in the same style as [`super::serve`]:
//! a blocking `TcpListener` with one thread per subscriber.
//!
//! Routes:
//! - `GET /trace`  -> `text/event-stream`, one `data: <jsonl line>` frame each
//! - `GET /health` -> `{"file":"...","frames":N,"subscribers":M}`
//! - `GET /`       -> plain-text help
//!
//! Every response carries `Access-Control-Allow-Origin: *`, so the static site
//! on another origin can subscribe without a proxy.

use std::fs::File;
use std::io::{self, Read, Seek, SeekFrom, Write};
use std::net::{TcpListener, TcpStream};
use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::Arc;
use std::thread;
use std::time::{Duration, Instant};

/// How often an idle watcher re-checks the file length.
const POLL_INTERVAL: Duration = Duration::from_millis(120);
/// Idle connections get an SSE comment so proxies do not drop them.
const KEEP_ALIVE: Duration = Duration::from_secs(15);
/// Upper bound on the request head we are willing to buffer.
const HEAD_LIMIT: usize = 16 * 1024;
/// Read granularity for appended bytes.
const READ_CHUNK: usize = 64 * 1024;
const DEFAULT_ADDRESS: &str = "127.0.0.1:8090";

struct State {
    path: PathBuf,
    frames: AtomicU64,
    subscribers: AtomicU64,
}

pub fn run(arguments: &[String]) {
    let address = super::value(arguments, "--addr").unwrap_or_else(|| DEFAULT_ADDRESS.into());
    let path = match super::value(arguments, "--file") {
        Some(path) => PathBuf::from(path),
        None => match newest_trace(Path::new("traces")) {
            Some(path) => path,
            None => {
                eprintln!(
                    "no .jsonl trace found - run `cargo run -p mini-llm-cli -- trace` first,"
                );
                eprintln!("or pass --file PATH");
                std::process::exit(1);
            }
        },
    };

    let listener = match TcpListener::bind(&address) {
        Ok(listener) => listener,
        Err(error) => {
            eprintln!("cannot bind {address}: {error}");
            std::process::exit(1);
        }
    };

    println!("watching {} on http://{address}/trace", path.display());
    println!("routes: GET /trace (SSE), GET /health, GET /");
    println!("subscribe from /studio with ?live=http://{address}/trace");

    let state = Arc::new(State {
        path,
        frames: AtomicU64::new(0),
        subscribers: AtomicU64::new(0),
    });

    for stream in listener.incoming() {
        match stream {
            Ok(stream) => {
                let state = Arc::clone(&state);
                thread::spawn(move || handle(stream, &state));
            }
            Err(error) => eprintln!("connection error: {error}"),
        }
    }
}

/// Newest `traces/*.jsonl` by file name; the names embed a unix timestamp.
fn newest_trace(directory: &Path) -> Option<PathBuf> {
    let mut newest: Option<PathBuf> = None;
    for entry in std::fs::read_dir(directory).ok()?.flatten() {
        let path = entry.path();
        if path.extension().and_then(|value| value.to_str()) != Some("jsonl") {
            continue;
        }
        let name = entry.file_name();
        let replace = newest
            .as_ref()
            .and_then(|current| current.file_name())
            .map(|current| name > current)
            .unwrap_or(true);
        if replace {
            newest = Some(path);
        }
    }
    newest
}

fn handle(mut stream: TcpStream, state: &Arc<State>) {
    let _ = stream.set_read_timeout(Some(Duration::from_secs(10)));
    let head = match read_head(&mut stream) {
        Ok(head) => head,
        Err(()) => return,
    };
    let (method, path) = request_line(&head);

    match (method.as_str(), path.as_str()) {
        ("OPTIONS", _) => {
            let _ = respond(
                &mut stream,
                "204 No Content",
                "text/plain; charset=utf-8",
                "",
            );
        }
        ("GET", "/trace") => {
            let _ = stream.set_read_timeout(None);
            let _ = stream_to_sse(&mut stream, state);
        }
        ("GET", "/health") => {
            let body = format!(
                "{{\"file\":\"{}\",\"frames\":{},\"subscribers\":{}}}",
                escape_json(&state.path.display().to_string()),
                state.frames.load(Ordering::Relaxed),
                state.subscribers.load(Ordering::Relaxed),
            );
            let _ = respond(&mut stream, "200 OK", "application/json", &body);
        }
        ("GET", "/") => {
            let body = format!(
                "mini-llm watch-trace\n\nfile: {}\n\n\
                 GET /trace   text/event-stream, one data frame per JSONL line\n\
                 GET /health  connection counters\n",
                state.path.display()
            );
            let _ = respond(&mut stream, "200 OK", "text/plain; charset=utf-8", &body);
        }
        _ => {
            let _ = respond(
                &mut stream,
                "404 Not Found",
                "application/json",
                "{\"error\":\"not found\"}",
            );
        }
    }
}

fn stream_to_sse(stream: &mut TcpStream, state: &Arc<State>) -> io::Result<()> {
    stream.write_all(
        b"HTTP/1.1 200 OK\r\n\
          Content-Type: text/event-stream; charset=utf-8\r\n\
          Cache-Control: no-cache, no-store\r\n\
          Connection: close\r\n\
          X-Accel-Buffering: no\r\n\
          Access-Control-Allow-Origin: *\r\n\r\n",
    )?;
    // Ask EventSource to reconnect quickly after a dropped connection.
    stream.write_all(b"retry: 1000\n\n")?;
    stream.flush()?;

    state.subscribers.fetch_add(1, Ordering::Relaxed);
    let result = follow(stream, state);
    state.subscribers.fetch_sub(1, Ordering::Relaxed);
    result
}

/// Replay the file from byte 0, then stream appended lines until the peer leaves.
fn follow(stream: &mut TcpStream, state: &State) -> io::Result<()> {
    let mut pending = LineBuffer::default();
    let mut offset = 0u64;
    let mut file: Option<File> = None;
    let mut last_write = Instant::now();
    let mut chunk = vec![0u8; READ_CHUNK];

    loop {
        if file.is_none() {
            match File::open(&state.path) {
                Ok(opened) => {
                    file = Some(opened);
                    offset = 0;
                    pending.clear();
                }
                Err(_) => {
                    // The run may not have created the file yet; wait for it.
                    if last_write.elapsed() >= KEEP_ALIVE {
                        stream.write_all(b": waiting for the trace file\n\n")?;
                        stream.flush()?;
                        last_write = Instant::now();
                    }
                    thread::sleep(Duration::from_millis(250));
                    continue;
                }
            }
        }

        let handle = file.as_mut().expect("opened above");
        let length = handle.metadata().map(|meta| meta.len()).unwrap_or(0);
        if length < offset {
            // Truncated or replaced: treat it as a new run and replay.
            offset = 0;
            pending.clear();
        }

        let mut read_any = false;
        loop {
            handle.seek(SeekFrom::Start(offset))?;
            let read = handle.read(&mut chunk)?;
            if read == 0 {
                break;
            }
            offset += read as u64;
            read_any = true;
            for line in pending.push(&chunk[..read]) {
                write_frame(stream, &line)?;
                state.frames.fetch_add(1, Ordering::Relaxed);
            }
        }

        if read_any {
            stream.flush()?;
            last_write = Instant::now();
            continue;
        }
        if last_write.elapsed() >= KEEP_ALIVE {
            stream.write_all(b": keep-alive\n\n")?;
            stream.flush()?;
            last_write = Instant::now();
        }
        thread::sleep(POLL_INTERVAL);
    }
}

fn write_frame(stream: &mut TcpStream, line: &str) -> io::Result<()> {
    stream.write_all(b"data: ")?;
    stream.write_all(line.as_bytes())?;
    stream.write_all(b"\n\n")
}

/// Buffers partial reads and yields only newline-terminated lines.
#[derive(Default)]
struct LineBuffer {
    pending: Vec<u8>,
}

impl LineBuffer {
    fn clear(&mut self) {
        self.pending.clear();
    }

    fn push(&mut self, bytes: &[u8]) -> Vec<String> {
        self.pending.extend_from_slice(bytes);
        let mut lines = Vec::new();
        let mut start = 0usize;
        for (index, byte) in self.pending.iter().enumerate() {
            if *byte != b'\n' {
                continue;
            }
            let end = if index > start && self.pending[index - 1] == b'\r' {
                index - 1
            } else {
                index
            };
            let line = String::from_utf8_lossy(&self.pending[start..end])
                .trim()
                .to_string();
            if !line.is_empty() {
                lines.push(line);
            }
            start = index + 1;
        }
        if start > 0 {
            self.pending.drain(..start);
        }
        lines
    }
}

fn respond(stream: &mut TcpStream, status: &str, content_type: &str, body: &str) -> io::Result<()> {
    let head = format!(
        "HTTP/1.1 {status}\r\n\
         Content-Type: {content_type}\r\n\
         Content-Length: {}\r\n\
         Cache-Control: no-store\r\n\
         Access-Control-Allow-Origin: *\r\n\
         Access-Control-Allow-Methods: GET, OPTIONS\r\n\
         Access-Control-Allow-Headers: *\r\n\
         Connection: close\r\n\r\n",
        body.len()
    );
    stream.write_all(head.as_bytes())?;
    stream.write_all(body.as_bytes())?;
    stream.flush()
}

fn read_head(stream: &mut TcpStream) -> Result<String, ()> {
    let mut buffer = Vec::new();
    let mut chunk = [0u8; 1024];
    loop {
        if let Some(index) = find_head_end(&buffer) {
            return Ok(String::from_utf8_lossy(&buffer[..index]).to_string());
        }
        if buffer.len() > HEAD_LIMIT {
            return Err(());
        }
        let read = stream.read(&mut chunk).map_err(|_| ())?;
        if read == 0 {
            return Err(());
        }
        buffer.extend_from_slice(&chunk[..read]);
    }
}

fn find_head_end(buffer: &[u8]) -> Option<usize> {
    if let Some(index) = buffer.windows(4).position(|window| window == b"\r\n\r\n") {
        return Some(index);
    }
    buffer.windows(2).position(|window| window == b"\n\n")
}

fn request_line(head: &str) -> (String, String) {
    let mut parts = head.lines().next().unwrap_or_default().split_whitespace();
    let method = parts.next().unwrap_or_default().to_string();
    let path = parts
        .next()
        .unwrap_or("/")
        .split(['?', '#'])
        .next()
        .unwrap_or("/")
        .to_string();
    (method, path)
}

fn escape_json(value: &str) -> String {
    let mut escaped = String::with_capacity(value.len());
    for character in value.chars() {
        match character {
            '"' => escaped.push_str("\\\""),
            '\\' => escaped.push_str("\\\\"),
            '\n' => escaped.push_str("\\n"),
            '\r' => escaped.push_str("\\r"),
            '\t' => escaped.push_str("\\t"),
            control if control < ' ' => escaped.push_str(&format!("\\u{:04x}", control as u32)),
            other => escaped.push(other),
        }
    }
    escaped
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn buffered_lines_survive_a_split_chunk() {
        let mut buffer = LineBuffer::default();
        assert!(buffer.push(b"{\"a\":1}\n{\"b\"").len() == 1);
        let lines = buffer.push(b":2}\n");
        assert_eq!(lines, vec!["{\"b\":2}".to_string()]);
    }

    #[test]
    fn blank_lines_and_carriage_returns_are_dropped() {
        let mut buffer = LineBuffer::default();
        let lines = buffer.push(b"{\"a\":1}\r\n\n   \n{\"b\":2}");
        assert_eq!(lines, vec!["{\"a\":1}".to_string()]);
    }

    #[test]
    fn request_line_strips_the_query_string() {
        let head = "GET /trace?live=1 HTTP/1.1\r\nHost: x\r\n";
        assert_eq!(request_line(head), ("GET".into(), "/trace".into()));
    }

    #[test]
    fn json_escape_handles_windows_paths() {
        assert_eq!(escape_json("traces\\a.jsonl"), "traces\\\\a.jsonl");
    }

    #[test]
    fn newest_trace_picks_the_last_name() {
        let directory = std::env::temp_dir().join(format!(
            "mini-llm-watch-{}-{:?}",
            std::process::id(),
            std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .unwrap()
                .as_nanos()
        ));
        std::fs::create_dir_all(&directory).expect("temp dir");
        std::fs::write(directory.join("run-1.jsonl"), b"{}\n").expect("write");
        std::fs::write(directory.join("run-2.jsonl"), b"{}\n").expect("write");
        std::fs::write(directory.join("notes.txt"), b"x").expect("write");

        let newest = newest_trace(&directory).expect("a trace");
        assert_eq!(newest.file_name().unwrap(), "run-2.jsonl");

        let _ = std::fs::remove_dir_all(&directory);
    }
}
