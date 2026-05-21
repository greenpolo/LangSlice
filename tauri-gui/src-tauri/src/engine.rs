use std::path::{Path, PathBuf};
use std::process::Stdio;
use std::sync::atomic::{AtomicU64, Ordering};
use std::time::{SystemTime, UNIX_EPOCH};

use serde::Deserialize;
use serde_json::{json, Value};
use tauri::{AppHandle, Emitter};
use tokio::io::{AsyncBufReadExt, AsyncWriteExt, BufReader};
use tokio::process::Command;

static REQUEST_COUNTER: AtomicU64 = AtomicU64::new(1);

#[derive(Debug, Deserialize, PartialEq)]
#[serde(tag = "type")]
enum EngineEnvelope {
    #[serde(rename = "event")]
    Event { id: String, event: EngineEvent },
    #[serde(rename = "result")]
    Result { id: String, result: Value },
    #[serde(rename = "error")]
    Error {
        id: Option<String>,
        error: EngineErrorPayload,
    },
}

#[derive(Debug, Deserialize, PartialEq)]
#[serde(tag = "kind")]
enum EngineEvent {
    #[serde(rename = "progress")]
    Progress {
        message: String,
        #[serde(default)]
        stage: Option<String>,
    },
    #[serde(rename = "log")]
    Log { message: String },
}

#[derive(Debug, Deserialize, PartialEq)]
struct EngineErrorPayload {
    code: String,
    message: String,
}

#[derive(Debug, PartialEq)]
enum EnvelopeOutcome {
    Ignore,
    EmitLog(String),
    Done(Result<Value, String>),
}

pub async fn run_engine_method(
    app: &AppHandle,
    method: &str,
    params: Value,
    log_prefix: Option<&str>,
) -> Result<Value, String> {
    let request_id = next_request_id();
    let mut command = build_engine_command();

    let mut child = command
        .stdin(Stdio::piped())
        .stdout(Stdio::piped())
        .stderr(Stdio::piped())
        .spawn()
        .map_err(|e| format!("Failed to spawn engine service: {}", e))?;

    let mut stdin = child
        .stdin
        .take()
        .ok_or("Engine service stdin unavailable")?;
    let stdout = child
        .stdout
        .take()
        .ok_or("Engine service stdout unavailable")?;
    let stderr = child
        .stderr
        .take()
        .ok_or("Engine service stderr unavailable")?;

    let payload = json!({
        "id": request_id,
        "method": method,
        "params": params,
    });
    let payload_line =
        serde_json::to_string(&payload).map_err(|e| format!("Failed to encode request: {}", e))?;
    stdin
        .write_all(payload_line.as_bytes())
        .await
        .map_err(|e| format!("Failed to write request: {}", e))?;
    stdin
        .write_all(b"\n")
        .await
        .map_err(|e| format!("Failed to finish request write: {}", e))?;
    drop(stdin);

    let stderr_app = app.clone();
    let stderr_prefix = log_prefix.map(ToOwned::to_owned);
    let stderr_task = tokio::spawn(async move {
        let reader = BufReader::new(stderr);
        let mut lines = reader.lines();
        while let Ok(Some(line)) = lines.next_line().await {
            let formatted = format_log_line(stderr_prefix.as_deref(), &line);
            let _ = stderr_app.emit("pipeline-log", formatted);
        }
    });

    let mut stdout_lines = BufReader::new(stdout).lines();
    let mut final_result: Option<Result<Value, String>> = None;
    let mut protocol_error: Option<String> = None;
    let mut should_terminate_child = false;
    while let Ok(Some(line)) = stdout_lines.next_line().await {
        match parse_envelope_line(&line) {
            Ok(envelope) => match handle_envelope(&request_id, envelope) {
                EnvelopeOutcome::Ignore => {}
                EnvelopeOutcome::EmitLog(message) => {
                    let formatted = format_log_line(log_prefix, &message);
                    let _ = app.emit("pipeline-log", formatted);
                }
                EnvelopeOutcome::Done(done) => {
                    if final_result.is_none() {
                        final_result = Some(done);
                    }
                }
            },
            Err(parse_error) => {
                let message = format!("Protocol parse error: {}", parse_error);
                let _ = app.emit("pipeline-log", format_log_line(Some("protocol"), &message));
                if final_result.is_none() && protocol_error.is_none() {
                    protocol_error = Some(message);
                    should_terminate_child = true;
                }
            }
        }

        if should_terminate_child {
            break;
        }
    }

    if should_terminate_child {
        let _ = child.kill().await;
    }

    stderr_task.await.ok();

    let status = child
        .wait()
        .await
        .map_err(|e| format!("Engine process wait failed: {}", e))?;

    if let Some(done) = final_result {
        if status.success() {
            return done;
        }
        return Err(format!(
            "Engine process exited with code {:?}",
            status.code()
        ));
    }

    if let Some(err) = protocol_error {
        return Err(err);
    }

    if !status.success() {
        return Err(format!(
            "Engine process exited before responding (code {:?})",
            status.code()
        ));
    }
    Err("Engine process ended without result envelope".to_string())
}

fn parse_envelope_line(line: &str) -> Result<EngineEnvelope, String> {
    serde_json::from_str(line).map_err(|e| format!("Invalid engine response line: {}", e))
}

fn handle_envelope(request_id: &str, envelope: EngineEnvelope) -> EnvelopeOutcome {
    match envelope {
        EngineEnvelope::Event { id, event } => {
            if id != request_id {
                return EnvelopeOutcome::Ignore;
            }
            match event {
                EngineEvent::Progress { message, stage } => {
                    let log = if let Some(stage_name) = stage.filter(|s| !s.is_empty()) {
                        format!("[{}] {}", stage_name, message)
                    } else {
                        message
                    };
                    EnvelopeOutcome::EmitLog(log)
                }
                EngineEvent::Log { message } => EnvelopeOutcome::EmitLog(message),
            }
        }
        EngineEnvelope::Result { id, result } => {
            if id == request_id {
                EnvelopeOutcome::Done(Ok(result))
            } else {
                EnvelopeOutcome::Ignore
            }
        }
        EngineEnvelope::Error { id, error } => {
            if id.as_deref() == Some(request_id) || id.is_none() {
                EnvelopeOutcome::Done(Err(format!(
                    "Engine error [{}]: {}",
                    error.code, error.message
                )))
            } else {
                EnvelopeOutcome::Ignore
            }
        }
    }
}

fn format_log_line(prefix: Option<&str>, message: &str) -> String {
    let trimmed_prefix = prefix.unwrap_or("").trim();
    if trimmed_prefix.is_empty() {
        message.to_string()
    } else {
        format!("[{}] {}", trimmed_prefix, message)
    }
}

fn next_request_id() -> String {
    let millis = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map(|d| d.as_millis())
        .unwrap_or(0);
    let counter = REQUEST_COUNTER.fetch_add(1, Ordering::Relaxed);
    format!("tauri-{}-{}", millis, counter)
}

fn build_engine_command() -> Command {
    let mut command = Command::new("python");
    command.args(["-m", "langslice_harness", "serve", "--stdio"]);
    apply_local_python_dev_bootstrap(&mut command);
    command
}

pub(crate) fn apply_local_python_dev_bootstrap(command: &mut Command) {
    if let Ok(current_dir) = std::env::current_dir() {
        if let Some(root) = find_workspace_root_with_src(&current_dir) {
            command.current_dir(&root);
            command.env("PYTHONPATH", build_pythonpath_for_local_src());
        }
    }
}

fn find_workspace_root_with_src(start: &Path) -> Option<PathBuf> {
    let mut cursor: Option<&Path> = Some(start);
    while let Some(dir) = cursor {
        if dir.join("src").join("langslice_harness").exists() {
            return Some(dir.to_path_buf());
        }
        cursor = dir.parent();
    }
    None
}

fn build_pythonpath_for_local_src() -> std::ffi::OsString {
    let mut paths = vec![PathBuf::from("src")];
    if let Some(existing) = std::env::var_os("PYTHONPATH") {
        paths.extend(std::env::split_paths(&existing));
    }
    std::env::join_paths(paths).unwrap_or_else(|_| std::ffi::OsString::from("src"))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn parse_progress_event_envelope() {
        let envelope = parse_envelope_line(
            r#"{"id":"req-1","type":"event","event":{"kind":"progress","message":"loading","stage":"estimate"}}"#,
        )
        .expect("should parse");
        match envelope {
            EngineEnvelope::Event { id, event } => {
                assert_eq!(id, "req-1");
                assert_eq!(
                    event,
                    EngineEvent::Progress {
                        message: "loading".to_string(),
                        stage: Some("estimate".to_string()),
                    }
                );
            }
            _ => panic!("expected event envelope"),
        }
    }

    #[test]
    fn handle_result_for_matching_request() {
        let outcome = handle_envelope(
            "req-1",
            EngineEnvelope::Result {
                id: "req-1".to_string(),
                result: json!({ "position_mm": 1.25 }),
            },
        );
        assert_eq!(
            outcome,
            EnvelopeOutcome::Done(Ok(json!({ "position_mm": 1.25 })))
        );
    }

    #[test]
    fn handle_error_includes_code_and_message() {
        let outcome = handle_envelope(
            "req-1",
            EngineEnvelope::Error {
                id: Some("req-1".to_string()),
                error: EngineErrorPayload {
                    code: "validation_error".to_string(),
                    message: "Request validation failed".to_string(),
                },
            },
        );
        assert_eq!(
            outcome,
            EnvelopeOutcome::Done(Err(
                "Engine error [validation_error]: Request validation failed".to_string()
            ))
        );
    }

    #[test]
    fn ignore_event_for_different_request() {
        let outcome = handle_envelope(
            "req-2",
            EngineEnvelope::Event {
                id: "req-1".to_string(),
                event: EngineEvent::Log {
                    message: "hello".to_string(),
                },
            },
        );
        assert_eq!(outcome, EnvelopeOutcome::Ignore);
    }

    #[test]
    fn log_formatting_applies_prefix() {
        assert_eq!(
            format_log_line(Some("quick-affine"), "working"),
            "[quick-affine] working"
        );
        assert_eq!(format_log_line(None, "working"), "working");
    }

    #[test]
    fn malformed_envelope_line_returns_parse_error() {
        let err = parse_envelope_line("{not json}").expect_err("should fail");
        assert!(err.contains("Invalid engine response line"));
    }

    #[test]
    fn find_workspace_root_detects_src_layout() {
        let uniq = format!("langslice_engine_test_{}", next_request_id());
        let base = std::env::temp_dir().join(uniq);
        let nested = base.join("a").join("b");
        let src_dir = base.join("src").join("langslice_harness");
        std::fs::create_dir_all(&nested).expect("create nested dirs");
        std::fs::create_dir_all(&src_dir).expect("create src layout");

        let found = find_workspace_root_with_src(&nested);
        assert_eq!(found, Some(base.clone()));

        let _ = std::fs::remove_dir_all(base);
    }
}
