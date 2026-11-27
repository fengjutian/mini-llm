//! Minimal blocking HTTP inference server for the saved Tiny GPT checkpoint.
//!
//! No async runtime and no external dependencies: a `TcpListener` accepts one
//! connection at a time, which keeps the model (which owns `Rc`-based tensor
//! values and is therefore not `Send`) on the main thread.
//!
//! Routes:
//! - `GET /health` -> `{"status":"ok"}`
//! - `GET /` -> short plain-text help
//! - `POST /generate` with a JSON body
//!   `{"prompt":"The","tokens":32,"temperature":0.8,"top_k":20,"top_p":0.9}`
//!   -> `{"text":"..."}`

use std::io::{Read, Write};
use std::net::{TcpListener, TcpStream};
use std::time::Duration;

use mini_llm_data::Tokenizer;
use mini_llm_model::{sample_next, Sampling, TinyGpt};
use mini_llm_nn::Rng;
use mini_llm_train::load_checkpoint;

use super::checkpoint_path;

const MAX_BODY: usize = 1 << 20;
const MAX_TOKENS: usize = 512;
const READ_TIMEOUT: Duration = Duration::from_secs(10);

struct GenerateRequest {
    prompt: String,
    tokens: usize,
    temperature: f32,
    top_k: Option<usize>,
    top_p: Option<f32>,
    stream: bool,
}

pub fn run(arguments: &[String]) {
    let address = super::value(arguments, "--addr").unwrap_or_else(|| "127.0.0.1:8080".into());
    let checkpoint = checkpoint_path(arguments);

    let (model, tokenizer) = match load_checkpoint(&checkpoint) {
        Ok(loaded) => loaded,
        Err(error) => {
            eprintln!("cannot load {checkpoint}: {error}");
            eprintln!("run `cargo run -p mini-llm-cli -- train` first, or pass --checkpoint PATH");
            std::process::exit(1);
        }
    };

    let listener = match TcpListener::bind(&address) {
        Ok(listener) => listener,
        Err(error) => {
            eprintln!("cannot bind {address}: {error}");
            std::process::exit(1);
        }
    };
    println!("serving on http://{address} (checkpoint {checkpoint})");
    println!("routes: GET /health, POST /generate (one connection at a time)");

    let mut rng = Rng::new(2024);
    for stream in listener.incoming() {
        match stream {
            Ok(stream) => handle(stream, &model, &tokenizer, &mut rng),
            Err(error) => eprintln!("connection error: {error}"),
        }
    }
}

fn handle(mut stream: TcpStream, model: &TinyGpt, tokenizer: &Tokenizer, rng: &mut Rng) {
    let _ = stream.set_read_timeout(Some(READ_TIMEOUT));
    let (method, path, body) = match read_request(&mut stream) {
        Ok(request) => request,
        Err(()) => {
            respond(&mut stream, 400, &error_json("malformed request"));
            return;
        }
    };

    match (method.as_str(), path.as_str()) {
        ("GET", "/health") => respond(&mut stream, 200, "{\"status\":\"ok\"}"),
        ("GET", "/") => respond_text(
            &mut stream,
            "mini-llm serve\n\nGET /health\nPOST /generate {\"prompt\":\"...\"}\n",
        ),
        ("POST", "/generate") => {
            let request = match parse_generate(&body) {
                Ok(request) => request,
                Err(message) => {
                    respond(&mut stream, 400, &error_json(&message));
                    return;
                }
            };
            let sampling = Sampling {
                temperature: request.temperature,
                top_k: request.top_k,
                top_p: request.top_p,
            };
            let encoded = match tokenizer.encode(&request.prompt) {
                Ok(tokens) if !tokens.is_empty() => tokens,
                Ok(_) => vec![0],
                Err(error) => {
                    respond(&mut stream, 400, &error_json(&error.to_string()));
                    return;
                }
            };
            if request.stream {
                stream_generate(
                    &mut stream,
                    model,
                    tokenizer,
                    rng,
                    &encoded,
                    &request,
                    sampling,
                );
                return;
            }
            match model.generate_with(&encoded, request.tokens, sampling, rng) {
                Ok(tokens) => {
                    let text = tokenizer.decode(&tokens).unwrap_or_default();
                    respond(
                        &mut stream,
                        200,
                        &format!("{{\"text\":\"{}\"}}", escape_json(&text)),
                    );
                }
                Err(error) => respond(&mut stream, 500, &error_json(&error.to_string())),
            }
        }
        _ => respond(&mut stream, 404, &error_json("not found")),
    }
}

fn stream_generate(
    stream: &mut TcpStream,
    model: &TinyGpt,
    tokenizer: &Tokenizer,
    rng: &mut Rng,
    prompt: &[usize],
    request: &GenerateRequest,
    sampling: Sampling,
) {
    let _ = stream.write_all(
        b"HTTP/1.1 200 OK\r\nContent-Type: text/event-stream\r\nCache-Control: no-cache\r\nConnection: close\r\nAccess-Control-Allow-Origin: *\r\n\r\n",
    );
    let mut tokens = prompt.to_vec();
    let mut caches = model.empty_caches();
    let mut logits = None;
    for (offset, &token) in tokens.iter().enumerate() {
        match model.step_token(token, offset, &mut caches) {
            Ok(value) => logits = Some(value),
            Err(error) => {
                let frame = format!("data: {}\n\n", error_json(&error.to_string()));
                let _ = stream.write_all(frame.as_bytes());
                return;
            }
        }
    }
    let Some(mut logits) = logits else {
        return;
    };
    let mut generated = Vec::new();
    let mut emitted = 0usize;
    for _ in 0..request.tokens {
        let next = match sample_next(&logits, model.config().vocab_size, sampling, rng) {
            Ok(token) => token,
            Err(error) => {
                let frame = format!("data: {}\n\n", error_json(&error.to_string()));
                let _ = stream.write_all(frame.as_bytes());
                return;
            }
        };
        generated.push(next);
        let text = tokenizer.decode(&generated).unwrap_or_default();
        let chunk: String = text.chars().skip(emitted).collect();
        if !chunk.is_empty() {
            emitted = text.chars().count();
            let frame = format!("data: {{\"token\":\"{}\"}}\n\n", escape_json(&chunk));
            if stream.write_all(frame.as_bytes()).is_err() {
                return;
            }
            let _ = stream.flush();
        }
        tokens.push(next);
        if tokens.len() >= model.config().max_seq_len {
            break;
        }
        logits = match model.step_token(next, tokens.len() - 1, &mut caches) {
            Ok(value) => value,
            Err(_) => break,
        };
    }
    let _ = stream.write_all(b"data: [DONE]\n\n");
    let _ = stream.flush();
}
fn read_request(stream: &mut TcpStream) -> Result<(String, String, String), ()> {
    let mut buffer = Vec::new();
    let mut chunk = [0u8; 2048];
    let mut head: Option<(usize, usize, String, String)> = None;

    loop {
        if head.is_none() {
            if let Some((index, separator)) = header_end(&buffer) {
                let head_text = String::from_utf8_lossy(&buffer[..index]).to_string();
                let (method, path, content_length) = parse_head(&head_text).ok_or(())?;
                if content_length > MAX_BODY {
                    return Err(());
                }
                head = Some((index + separator, content_length, method, path));
            }
        }
        if let Some((body_start, content_length, ref method, ref path)) = head {
            if buffer.len() >= body_start + content_length {
                let body =
                    String::from_utf8_lossy(&buffer[body_start..body_start + content_length])
                        .to_string();
                return Ok((method.clone(), path.clone(), body));
            }
        }
        let read = stream.read(&mut chunk).map_err(|_| ())?;
        if read == 0 {
            // EOF: accept a bodyless request if headers were complete.
            if let Some((body_start, content_length, method, path)) = head {
                if content_length == 0 && buffer.len() >= body_start {
                    return Ok((method, path, String::new()));
                }
            }
            return Err(());
        }
        buffer.extend_from_slice(&chunk[..read]);
        if buffer.len() > MAX_BODY + 8192 {
            return Err(());
        }
    }
}

fn header_end(buffer: &[u8]) -> Option<(usize, usize)> {
    if let Some(index) = buffer.windows(4).position(|window| window == b"\r\n\r\n") {
        return Some((index, 4));
    }
    buffer
        .windows(2)
        .position(|window| window == b"\n\n")
        .map(|index| (index, 2))
}

fn parse_head(head: &str) -> Option<(String, String, usize)> {
    let mut lines = head.lines();
    let request_line = lines.next()?;
    let mut parts = request_line.split_whitespace();
    let method = parts.next()?.to_string();
    let path = parts.next()?.split('?').next().unwrap_or("/").to_string();
    let content_length = lines
        .filter_map(|line| line.split_once(':'))
        .filter(|(name, _)| name.eq_ignore_ascii_case("content-length"))
        .find_map(|(_, value)| value.trim().parse::<usize>().ok())
        .unwrap_or(0);
    Some((method, path, content_length))
}

fn parse_generate(body: &str) -> Result<GenerateRequest, String> {
    let prompt = json_string(body, "prompt").unwrap_or_default();
    let tokens = json_number::<usize>(body, "tokens")
        .unwrap_or(32)
        .clamp(1, MAX_TOKENS);
    let temperature = json_number::<f32>(body, "temperature")
        .unwrap_or(0.8)
        .clamp(0.0, 2.0);
    let top_k = json_number::<usize>(body, "top_k").map(|value| value.max(1));
    let top_p = json_number::<f32>(body, "top_p").map(|value| value.clamp(0.0, 1.0));
    let stream = json_bool(body, "stream").unwrap_or(false);
    Ok(GenerateRequest {
        prompt,
        tokens,
        temperature,
        top_k,
        top_p,
        stream,
    })
}

fn json_bool(body: &str, key: &str) -> Option<bool> {
    let rest = json_value(body, key)?;
    if rest.starts_with("true") {
        Some(true)
    } else if rest.starts_with("false") {
        Some(false)
    } else {
        None
    }
}

fn json_value<'a>(body: &'a str, key: &str) -> Option<&'a str> {
    let needle = format!("\"{key}\"");
    let start = body.find(&needle)? + needle.len();
    let rest = &body[start..];
    let colon = rest.find(':')?;
    Some(rest[colon + 1..].trim_start())
}

fn json_string(body: &str, key: &str) -> Option<String> {
    let rest = json_value(body, key)?;
    let mut characters = rest.strip_prefix('"')?.chars();
    let mut text = String::new();
    loop {
        match characters.next()? {
            '"' => return Some(text),
            '\\' => match characters.next()? {
                '"' => text.push('"'),
                '\\' => text.push('\\'),
                'n' => text.push('\n'),
                't' => text.push('\t'),
                'r' => text.push('\r'),
                other => text.push(other),
            },
            character => text.push(character),
        }
    }
}

fn json_number<T: std::str::FromStr>(body: &str, key: &str) -> Option<T> {
    let rest = json_value(body, key)?;
    let end = rest
        .find(|character: char| character == ',' || character == '}' || character.is_whitespace())
        .unwrap_or(rest.len());
    rest[..end].parse().ok()
}

fn escape_json(text: &str) -> String {
    let mut escaped = String::with_capacity(text.len());
    for character in text.chars() {
        match character {
            '"' => escaped.push_str("\\\""),
            '\\' => escaped.push_str("\\\\"),
            '\n' => escaped.push_str("\\n"),
            '\r' => escaped.push_str("\\r"),
            '\t' => escaped.push_str("\\t"),
            control if control < ' ' => escaped.push(' '),
            other => escaped.push(other),
        }
    }
    escaped
}

fn error_json(message: &str) -> String {
    format!("{{\"error\":\"{}\"}}", escape_json(message))
}

fn respond(stream: &mut TcpStream, status: u16, body: &str) {
    let reason = match status {
        200 => "OK",
        400 => "Bad Request",
        404 => "Not Found",
        500 => "Internal Server Error",
        _ => "Service Unavailable",
    };
    let response = format!(
        "HTTP/1.1 {status} {reason}\r\nContent-Type: application/json\r\nContent-Length: {}\r\nAccess-Control-Allow-Origin: *\r\nConnection: close\r\n\r\n{body}",
        body.len()
    );
    let _ = stream.write_all(response.as_bytes());
    let _ = stream.flush();
}

fn respond_text(stream: &mut TcpStream, body: &str) {
    let response = format!(
        "HTTP/1.1 200 OK\r\nContent-Type: text/plain; charset=utf-8\r\nContent-Length: {}\r\nConnection: close\r\n\r\n{body}",
        body.len()
    );
    let _ = stream.write_all(response.as_bytes());
    let _ = stream.flush();
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn parses_generate_fields() {
        let body =
            r#"{"prompt":"The little ","tokens":8,"temperature":0.5,"top_k":10,"top_p":0.9}"#;
        let request = parse_generate(body).unwrap();
        assert_eq!(request.prompt, "The little ");
        assert_eq!(request.tokens, 8);
        assert!((request.temperature - 0.5).abs() < 1e-6);
        assert_eq!(request.top_k, Some(10));
        assert_eq!(request.top_p, Some(0.9));
        assert!(!request.stream);
    }

    #[test]
    fn parses_stream_flag() {
        let request = parse_generate(r#"{"prompt":"The","stream":true}"#).unwrap();
        assert!(request.stream);
    }

    #[test]
    fn defaults_apply_when_fields_are_missing() {
        let request = parse_generate("{}").unwrap();
        assert_eq!(request.prompt, "");
        assert_eq!(request.tokens, 32);
        assert!((request.temperature - 0.8).abs() < 1e-6);
    }

    #[test]
    fn escapes_and_unescapes_json_strings() {
        let escaped = escape_json("a\"b\\c\nd");
        assert_eq!(escaped, "a\\\"b\\\\c\\nd");
        let body = format!("{{\"prompt\":\"{escaped}\"}}");
        assert_eq!(json_string(&body, "prompt").unwrap(), "a\"b\\c\nd");
    }

    #[test]
    fn parses_http_request_line_and_content_length() {
        let head = "POST /generate HTTP/1.1\r\nContent-Length: 12\r\nHost: x";
        let (method, path, length) = parse_head(head).unwrap();
        assert_eq!(method, "POST");
        assert_eq!(path, "/generate");
        assert_eq!(length, 12);
    }

    #[test]
    fn finds_both_header_terminators() {
        assert_eq!(header_end(b"GET / HTTP/1.1\r\n\r\n"), Some((14, 4)));
        assert_eq!(header_end(b"GET / HTTP/1.1\n\n"), Some((14, 2)));
    }
}
