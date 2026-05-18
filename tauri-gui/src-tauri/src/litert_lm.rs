//! Manage the bundled LiteRT-LM sidecar for fully-offline Gemma 4 inference.
//!
//! Mirrors the lifecycle shape of `ollama.rs` but spawns a binary that Tauri
//! bundles alongside the GUI rather than a system-installed one. The `.litertlm`
//! model file (~3.6 GB for Gemma 4 E4B) is too large to ship in the installer,
//! so it streams from HuggingFace into `%LOCALAPPDATA%/langslice/litert-lm/models/`
//! on first use.
//!
//! Once started, the sidecar exposes an OpenAI-compatible HTTP server on
//! `127.0.0.1:8765/v1`. `local_engines.rs` lists it in its registry so the
//! existing Local Models modal and `--endpoint` plumbing pick it up unchanged.

use std::path::PathBuf;
use std::process::Stdio;
use std::sync::Mutex;
use std::time::Duration;

use futures_util::StreamExt;
use serde::Serialize;
use serde_json::{json, Map, Value};
use tauri::{AppHandle, Emitter, State};
use tokio::io::AsyncWriteExt;
use tokio::process::{Child, Command};
use tokio::time::{sleep, timeout};

/// Default model the demo pulls. Mirrors the upstream community bundle Google
/// publishes for desktop. Users with a converted fine-tune can drop their own
/// `.litertlm` into the same directory under a different filename and select it.
pub const DEFAULT_MODEL_FILENAME: &str = "gemma-4-E4B-it.litertlm";
const DEFAULT_MODEL_URL: &str = "https://huggingface.co/litert-community/gemma-4-E4B-it-litert-lm/resolve/main/gemma-4-E4B-it.litertlm";

/// Port chosen to avoid clashing with anything in `local_engines.rs::ENGINES`
/// (Ollama 11434, LM Studio 1234, llama-server 8080, vLLM 8000, Jan 1337).
pub const DEFAULT_PORT: u16 = 8765;

#[derive(Serialize, Clone, Debug)]
pub enum LiteRtLmStatus {
    NotInstalled,
    NoModel,
    Downloading,
    Starting,
    Ready,
    Error(String),
}

pub struct LiteRtLmState {
    pub status: LiteRtLmStatus,
    pub port: u16,
    pub process: Option<Child>,
    pub models_dir: PathBuf,
    pub client: reqwest::Client,
}

impl Default for LiteRtLmState {
    fn default() -> Self {
        let models_dir = dirs::data_local_dir()
            .unwrap_or_else(|| PathBuf::from("."))
            .join("langslice")
            .join("litert-lm")
            .join("models");

        Self {
            status: LiteRtLmStatus::NotInstalled,
            port: DEFAULT_PORT,
            process: None,
            models_dir,
            client: reqwest::Client::new(),
        }
    }
}

fn status_label(status: &LiteRtLmStatus) -> &'static str {
    match status {
        LiteRtLmStatus::NotInstalled => "not_installed",
        LiteRtLmStatus::NoModel => "no_model",
        LiteRtLmStatus::Downloading => "downloading",
        LiteRtLmStatus::Starting => "starting",
        LiteRtLmStatus::Ready => "ready",
        LiteRtLmStatus::Error(_) => "error",
    }
}

async fn is_running(client: &reqwest::Client, port: u16) -> bool {
    let url = format!("http://127.0.0.1:{port}/v1/models");
    matches!(
        client.get(url).timeout(Duration::from_millis(400)).send().await,
        Ok(r) if r.status().is_success()
    )
}

/// Locate the bundled sidecar next to the running executable. Tauri's
/// `externalBin` config drops the target-triple-named binary into the same
/// directory as the app binary in both `tauri dev` and the installed bundle.
fn sidecar_path() -> Result<PathBuf, String> {
    let exe = std::env::current_exe().map_err(|e| format!("current_exe: {e}"))?;
    let dir = exe
        .parent()
        .ok_or_else(|| "current_exe has no parent".to_string())?;
    // Tauri appends the target triple to externalBin names. On Windows MSVC
    // builds that is `-x86_64-pc-windows-msvc.exe`.
    let candidates = [
        "litert-lm-x86_64-pc-windows-msvc.exe",
        "litert-lm.exe",
        "litert-lm",
    ];
    for name in &candidates {
        let p = dir.join(name);
        if p.is_file() {
            return Ok(p);
        }
    }
    Err(format!(
        "litert-lm sidecar not found next to {} (looked for {:?})",
        dir.display(),
        candidates
    ))
}

fn default_model_path(state: &LiteRtLmState) -> PathBuf {
    state.models_dir.join(DEFAULT_MODEL_FILENAME)
}

#[tauri::command]
pub async fn litert_lm_status(state: State<'_, Mutex<LiteRtLmState>>) -> Result<Value, String> {
    let (port, models_dir, client, current_status) = {
        let guard = state.lock().map_err(|err| err.to_string())?;
        (
            guard.port,
            guard.models_dir.clone(),
            guard.client.clone(),
            guard.status.clone(),
        )
    };

    // Liveness probe first — if the server is up, that wins regardless of
    // whatever the cached status says.
    let effective = if is_running(&client, port).await {
        let mut guard = state.lock().map_err(|err| err.to_string())?;
        guard.status = LiteRtLmStatus::Ready;
        LiteRtLmStatus::Ready
    } else {
        // Refresh from disk in case the user dropped a model in or the
        // sidecar was added between launches.
        let sidecar_ok = sidecar_path().is_ok();
        let model_ok = models_dir.join(DEFAULT_MODEL_FILENAME).is_file();
        let next = match (sidecar_ok, model_ok, &current_status) {
            (false, _, _) => LiteRtLmStatus::NotInstalled,
            (true, false, _) => LiteRtLmStatus::NoModel,
            // Preserve transient states (Downloading, Starting, Error) so the
            // UI can keep showing progress without us stomping it here.
            (true, true, LiteRtLmStatus::Downloading) => LiteRtLmStatus::Downloading,
            (true, true, LiteRtLmStatus::Starting) => LiteRtLmStatus::Starting,
            (true, true, LiteRtLmStatus::Error(m)) => LiteRtLmStatus::Error(m.clone()),
            (true, true, _) => LiteRtLmStatus::NotInstalled, // sidecar+model present, not running
        };
        let mut guard = state.lock().map_err(|err| err.to_string())?;
        guard.status = next.clone();
        next
    };

    let mut payload = Map::new();
    payload.insert(
        "status".to_string(),
        Value::String(status_label(&effective).to_string()),
    );
    payload.insert("port".to_string(), json!(port));
    payload.insert(
        "models_dir".to_string(),
        Value::String(models_dir.display().to_string()),
    );
    payload.insert(
        "default_model".to_string(),
        Value::String(DEFAULT_MODEL_FILENAME.to_string()),
    );
    payload.insert(
        "endpoint".to_string(),
        Value::String(format!("http://127.0.0.1:{port}/v1")),
    );
    if let LiteRtLmStatus::Error(msg) = &effective {
        payload.insert("error".to_string(), Value::String(msg.clone()));
    }

    Ok(Value::Object(payload))
}

#[tauri::command]
pub async fn litert_lm_download_model(
    app: AppHandle,
    state: State<'_, Mutex<LiteRtLmState>>,
) -> Result<(), String> {
    let (models_dir, client) = {
        let mut guard = state.lock().map_err(|err| err.to_string())?;
        guard.status = LiteRtLmStatus::Downloading;
        (guard.models_dir.clone(), guard.client.clone())
    };

    std::fs::create_dir_all(&models_dir).map_err(|e| e.to_string())?;
    let dest = models_dir.join(DEFAULT_MODEL_FILENAME);
    let tmp = models_dir.join(format!("{DEFAULT_MODEL_FILENAME}.part"));

    let _ = app.emit(
        "litert-lm-download-progress",
        json!({ "phase": "starting", "url": DEFAULT_MODEL_URL }),
    );

    let resp = client
        .get(DEFAULT_MODEL_URL)
        .send()
        .await
        .map_err(|e| {
            let msg = format!("request failed: {e}");
            let _ = set_error(&state, &msg);
            msg
        })?;

    if !resp.status().is_success() {
        let msg = format!("HTTP {}", resp.status().as_u16());
        let _ = set_error(&state, &msg);
        return Err(msg);
    }

    let total = resp.content_length();
    let mut file = tokio::fs::File::create(&tmp).await.map_err(|e| e.to_string())?;
    let mut stream = resp.bytes_stream();
    let mut downloaded: u64 = 0;
    // Throttle progress emits to roughly twice per percent so we don't drown
    // the renderer in events. The .litertlm is ~3.6 GB so unthrottled chunks
    // (8-16 KB each) would emit ~450k events.
    let mut next_emit_at: u64 = 0;

    while let Some(chunk) = stream.next().await {
        let bytes = chunk.map_err(|e| {
            let msg = format!("stream error: {e}");
            let _ = set_error(&state, &msg);
            msg
        })?;
        file.write_all(&bytes).await.map_err(|e| e.to_string())?;
        downloaded += bytes.len() as u64;
        if downloaded >= next_emit_at {
            let _ = app.emit(
                "litert-lm-download-progress",
                json!({
                    "phase": "downloading",
                    "completed": downloaded,
                    "total": total,
                }),
            );
            // Emit every ~16 MB so the UI sees motion on the 3.6 GB pull
            // without flooding the event loop.
            next_emit_at = downloaded + 16 * 1024 * 1024;
        }
    }

    file.flush().await.map_err(|e| e.to_string())?;
    drop(file);
    tokio::fs::rename(&tmp, &dest)
        .await
        .map_err(|e| format!("rename {} -> {}: {}", tmp.display(), dest.display(), e))?;

    {
        let mut guard = state.lock().map_err(|err| err.to_string())?;
        guard.status = LiteRtLmStatus::NotInstalled; // sidecar present + model present = ready-to-start
    }
    let _ = app.emit(
        "litert-lm-download-progress",
        json!({ "phase": "complete", "completed": downloaded, "total": total }),
    );
    Ok(())
}

fn set_error(
    state: &State<'_, Mutex<LiteRtLmState>>,
    msg: &str,
) -> Result<(), String> {
    let mut guard = state.lock().map_err(|err| err.to_string())?;
    guard.status = LiteRtLmStatus::Error(msg.to_string());
    Ok(())
}

#[tauri::command]
pub async fn litert_lm_start(
    state: State<'_, Mutex<LiteRtLmState>>,
    _app: AppHandle,
) -> Result<(), String> {
    let (port, client, model_path) = {
        let guard = state.lock().map_err(|err| err.to_string())?;
        (guard.port, guard.client.clone(), default_model_path(&guard))
    };

    if is_running(&client, port).await {
        let mut guard = state.lock().map_err(|err| err.to_string())?;
        guard.status = LiteRtLmStatus::Ready;
        return Ok(());
    }

    if !model_path.is_file() {
        let mut guard = state.lock().map_err(|err| err.to_string())?;
        guard.status = LiteRtLmStatus::NoModel;
        return Err(format!(
            "model not found at {} — run download first",
            model_path.display()
        ));
    }

    let sidecar = sidecar_path()?;

    // `litert-lm serve` exposes OpenAI-compatible /v1/chat/completions with
    // SSE streaming. --backend gpu uses the WebGPU/Vulkan delegate on
    // Windows; falls back to CPU automatically if the GPU path can't init.
    let mut command = Command::new(&sidecar);
    command
        .args([
            "serve",
            "--port",
            &port.to_string(),
            "--backend",
            "gpu",
            "--model",
        ])
        .arg(&model_path)
        .stdout(Stdio::null())
        .stderr(Stdio::null());

    let child = match command.spawn() {
        Ok(child) => child,
        Err(err) => {
            let mut guard = state.lock().map_err(|lock_err| lock_err.to_string())?;
            guard.status = LiteRtLmStatus::Error(err.to_string());
            return Err(format!("spawn {}: {}", sidecar.display(), err));
        }
    };

    {
        let mut guard = state.lock().map_err(|err| err.to_string())?;
        guard.status = LiteRtLmStatus::Starting;
        guard.process = Some(child);
    }

    // First load of a 3.6 GB model with GPU init can easily take 30-60 s,
    // so the readiness timeout is generous relative to Ollama's 30 s.
    match timeout(Duration::from_secs(120), async {
        loop {
            if is_running(&client, port).await {
                break;
            }
            sleep(Duration::from_millis(500)).await;
        }
    })
    .await
    {
        Ok(()) => {
            let mut guard = state.lock().map_err(|err| err.to_string())?;
            guard.status = LiteRtLmStatus::Ready;
            Ok(())
        }
        Err(_) => {
            let mut guard = state.lock().map_err(|err| err.to_string())?;
            guard.status = LiteRtLmStatus::Error("timeout".to_string());
            Err("timeout waiting for litert-lm serve to become ready".to_string())
        }
    }
}

#[tauri::command]
pub async fn litert_lm_stop(state: State<'_, Mutex<LiteRtLmState>>) -> Result<(), String> {
    let mut child = {
        let mut guard = state.lock().map_err(|err| err.to_string())?;
        guard.status = LiteRtLmStatus::NotInstalled;
        guard.process.take()
    };

    if let Some(child) = child.as_mut() {
        let _ = child.kill().await;
        let _ = child.wait().await;
    }

    Ok(())
}
