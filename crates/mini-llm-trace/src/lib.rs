//! # mini-llm-trace
//!
//! A small, dependency-free training event protocol.
//!
//! The trainer reports structured events (`RunStarted`, `Step`, `Tensor`,
//! `Checkpoint`, `RunCompleted`, `RunFailed`) to an [`EventSink`]. Sinks can
//! print, keep everything in memory for assertions, or write newline-delimited
//! JSON that a UI can tail.
//!
//! ```no_run
//! use mini_llm_trace::{Event, EventSink, MemorySink};
//!
//! let mut sink = MemorySink::default();
//! sink.record(Event::RunStarted { steps: 10, seed: 1 });
//! sink.record(Event::Step { step: 1, loss: 2.5, elapsed_ms: 3.0 });
//! assert_eq!(sink.events.len(), 2);
//! ```

use std::fmt::Write as _;
use std::io::{self, Write};
use std::time::{Instant, SystemTime, UNIX_EPOCH};

/// One structured training event.
#[derive(Debug, Clone, PartialEq)]
pub enum Event {
    RunStarted {
        steps: usize,
        seed: u64,
    },
    Step {
        step: usize,
        loss: f32,
        elapsed_ms: f32,
    },
    /// A captured tensor such as attention weights or logits.
    Tensor {
        name: String,
        step: usize,
        shape: Vec<usize>,
        values: Vec<f32>,
    },
    Checkpoint {
        path: String,
    },
    RunCompleted {
        steps: usize,
        loss: f32,
    },
    RunFailed {
        message: String,
    },
}

impl Event {
    /// Renders the event as one JSON object (no trailing newline).
    pub fn to_json(&self, run_id: &str) -> String {
        let mut out = String::new();
        let _ = write!(out, "{{\"run\":\"{}\"", escape(run_id));
        match self {
            Self::RunStarted { steps, seed } => {
                let _ = write!(
                    out,
                    ",\"event\":\"run_started\",\"steps\":{steps},\"seed\":{seed}"
                );
            }
            Self::Step {
                step,
                loss,
                elapsed_ms,
            } => {
                let _ = write!(
                    out,
                    ",\"event\":\"step\",\"step\":{step},\"loss\":{loss:.6},\"elapsed_ms\":{elapsed_ms:.3}"
                );
            }
            Self::Tensor {
                name,
                step,
                shape,
                values,
            } => {
                let _ = write!(
                    out,
                    ",\"event\":\"tensor\",\"step\":{step},\"name\":\"{}\",\"shape\":[",
                    escape(name)
                );
                for (index, dimension) in shape.iter().enumerate() {
                    if index > 0 {
                        out.push(',');
                    }
                    let _ = write!(out, "{dimension}");
                }
                out.push_str("],\"values\":[");
                for (index, value) in values.iter().enumerate() {
                    if index > 0 {
                        out.push(',');
                    }
                    let _ = write!(out, "{value:.6}");
                }
                out.push(']');
            }
            Self::Checkpoint { path } => {
                let _ = write!(
                    out,
                    ",\"event\":\"checkpoint\",\"path\":\"{}\"",
                    escape(path)
                );
            }
            Self::RunCompleted { steps, loss } => {
                let _ = write!(
                    out,
                    ",\"event\":\"run_completed\",\"steps\":{steps},\"loss\":{loss:.6}"
                );
            }
            Self::RunFailed { message } => {
                let _ = write!(
                    out,
                    ",\"event\":\"run_failed\",\"message\":\"{}\"",
                    escape(message)
                );
            }
        }
        out.push('}');
        out
    }
}

/// Something that consumes training events.
pub trait EventSink {
    fn record(&mut self, event: Event);
}

/// Keeps every event in memory (useful for tests and the CLI summary).
#[derive(Debug, Default)]
pub struct MemorySink {
    pub events: Vec<Event>,
}

impl EventSink for MemorySink {
    fn record(&mut self, event: Event) {
        self.events.push(event);
    }
}

impl MemorySink {
    /// Number of captured tensor events.
    pub fn tensor_count(&self) -> usize {
        self.events
            .iter()
            .filter(|event| matches!(event, Event::Tensor { .. }))
            .count()
    }
}

/// Writes newline-delimited JSON to any [`Write`] sink.
///
/// Writes are batched: the underlying writer is flushed every
/// [`JsonlSink::flush_every`] events (and on [`JsonlSink::flush`] or drop), which
/// keeps a training loop from issuing one syscall per event while still making
/// the file tail-able.
pub struct JsonlSink<W: Write> {
    writer: W,
    run_id: String,
    started: Instant,
    written: usize,
    flush_every: usize,
    pending: usize,
}

impl<W: Write> JsonlSink<W> {
    /// Derives a run id from the current time.
    pub fn new(writer: W) -> Self {
        Self::with_run_id(writer, default_run_id())
    }

    pub fn with_run_id(writer: W, run_id: impl Into<String>) -> Self {
        Self {
            writer,
            run_id: run_id.into(),
            started: Instant::now(),
            written: 0,
            flush_every: 8,
            pending: 0,
        }
    }

    /// Sets how many events are buffered before flushing (`1` = every event).
    pub fn with_flush_every(mut self, flush_every: usize) -> Self {
        self.flush_every = flush_every.max(1);
        self
    }

    pub fn run_id(&self) -> &str {
        &self.run_id
    }

    pub fn events_written(&self) -> usize {
        self.written
    }

    /// Flushes buffered events immediately, e.g. before printing a file path.
    pub fn flush(&mut self) {
        let _ = self.writer.flush();
        self.pending = 0;
    }

    /// Milliseconds since the sink was created, attached to every line.
    fn elapsed_ms(&self) -> f32 {
        self.started.elapsed().as_secs_f32() * 1_000.0
    }
}

impl<W: Write> EventSink for JsonlSink<W> {
    fn record(&mut self, event: Event) {
        let mut line = event.to_json(&self.run_id);
        // Append a wall-clock-free relative timestamp before the closing brace.
        let timestamp = format!(",\"t_ms\":{:.3}}}", self.elapsed_ms());
        line.pop();
        line.push_str(&timestamp);
        let _ = writeln!(self.writer, "{line}");
        self.written += 1;
        self.pending += 1;
        if self.pending >= self.flush_every {
            self.flush();
        }
    }
}

impl<W: Write> Drop for JsonlSink<W> {
    fn drop(&mut self) {
        let _ = self.writer.flush();
    }
}

/// Convenience sink that writes to a buffered file.
pub fn jsonl_file_sink(
    path: impl AsRef<std::path::Path>,
) -> io::Result<JsonlSink<io::BufWriter<std::fs::File>>> {
    let file = std::fs::File::create(path)?;
    Ok(JsonlSink::new(io::BufWriter::new(file)))
}

fn default_run_id() -> String {
    let millis = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map(|duration| duration.as_millis())
        .unwrap_or(0);
    format!("run-{millis}")
}

fn escape(text: &str) -> String {
    text.replace('\\', "\\\\").replace('"', "\\\"")
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::sync::{Arc, Mutex};

    #[derive(Clone, Default)]
    struct SharedWriter(Arc<Mutex<Vec<u8>>>);

    impl Write for SharedWriter {
        fn write(&mut self, buffer: &[u8]) -> io::Result<usize> {
            self.0.lock().unwrap().extend_from_slice(buffer);
            Ok(buffer.len())
        }

        fn flush(&mut self) -> io::Result<()> {
            Ok(())
        }
    }

    #[test]
    fn events_serialize_to_json() {
        let event = Event::Step {
            step: 3,
            loss: 1.25,
            elapsed_ms: 4.5,
        };
        let json = event.to_json("run-1");
        assert!(json.starts_with("{\"run\":\"run-1\""));
        assert!(json.contains("\"event\":\"step\""));
        assert!(json.contains("\"step\":3"));
        assert!(json.ends_with('}'));
    }

    #[test]
    fn jsonl_sink_writes_one_line_per_event() {
        let writer = SharedWriter::default();
        let mut sink = JsonlSink::with_run_id(writer.clone(), "r");
        sink.record(Event::RunStarted { steps: 2, seed: 7 });
        sink.record(Event::RunCompleted {
            steps: 2,
            loss: 0.5,
        });
        assert_eq!(sink.events_written(), 2);

        let text = String::from_utf8(writer.0.lock().unwrap().clone()).unwrap();
        let lines: Vec<&str> = text.lines().collect();
        assert_eq!(lines.len(), 2);
        assert!(lines[0].contains("run_started"));
        assert!(lines[0].contains("\"t_ms\":"));
        assert!(lines[0].ends_with('}'));
    }

    #[derive(Clone, Default)]
    struct FlushCounter(Arc<Mutex<usize>>);

    impl Write for FlushCounter {
        fn write(&mut self, buffer: &[u8]) -> io::Result<usize> {
            Ok(buffer.len())
        }

        fn flush(&mut self) -> io::Result<()> {
            *self.0.lock().unwrap() += 1;
            Ok(())
        }
    }

    impl FlushCounter {
        fn count(&self) -> usize {
            *self.0.lock().unwrap()
        }
    }

    #[test]
    fn jsonl_sink_batches_flushes() {
        let writer = FlushCounter::default();
        let mut sink = JsonlSink::with_run_id(writer.clone(), "r").with_flush_every(4);
        for step in 0..3 {
            sink.record(Event::Step {
                step,
                loss: 1.0,
                elapsed_ms: 1.0,
            });
        }
        assert_eq!(writer.count(), 0, "should still be buffering");

        sink.record(Event::Step {
            step: 3,
            loss: 1.0,
            elapsed_ms: 1.0,
        });
        assert_eq!(writer.count(), 1, "fourth event flushes");

        sink.flush();
        assert_eq!(writer.count(), 2);
    }

    #[test]
    fn tensors_keep_shape_and_values() {
        let event = Event::Tensor {
            name: "logits".into(),
            step: 1,
            shape: vec![1, 2],
            values: vec![0.5, -0.5],
        };
        let json = event.to_json("r");
        assert!(json.contains("\"shape\":[1,2]"));
        assert!(json.contains("\"values\":[0.500000,-0.500000]"));
    }

    #[test]
    fn memory_sink_counts_tensors() {
        let mut sink = MemorySink::default();
        sink.record(Event::Tensor {
            name: "a".into(),
            step: 0,
            shape: vec![1],
            values: vec![1.0],
        });
        sink.record(Event::Step {
            step: 0,
            loss: 1.0,
            elapsed_ms: 1.0,
        });
        assert_eq!(sink.events.len(), 2);
        assert_eq!(sink.tensor_count(), 1);
    }
}
