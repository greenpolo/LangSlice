use std::path::PathBuf;
use std::process::Stdio;
use std::sync::Mutex;
use std::time::Duration;

use serde::Serialize;
use serde_json::{json, Map, Value};
use tauri::State;
use tokio::process::{Child, Command};
use tokio::time::{sleep, timeout};

#[derive(Serialize, Clone, Debug)]
pub enum OllamaStatus {
    NotInstalled,
    Starting,
    Ready,
    Error(String),
}

pub struct OllamaState {
    pub status: OllamaStatus,
    pub port: u16,
    pub process: Option<Child>,
    pub models_dir: PathBuf,
    pub client: reqwest::Client,
}

impl Default for OllamaState {
    fn default() -> Self {
        let models_dir = dirs::data_local_dir()
            .unwrap_or_else(|| PathBuf::from("."))
            .join("langslice")
            .join("ollama")
            .join("models");

        Self {
            status: OllamaStatus::NotInstalled,
            port: 11435,
            process: None,
            models_dir,
            client: reqwest::Client::new(),
        }
    }
}

fn status_label(status: &OllamaStatus) -> &'static str {
    match status {
        OllamaStatus::NotInstalled => "not_installed",
        OllamaStatus::Starting => "starting",
        OllamaStatus::Ready => "ready",
        OllamaStatus::Error(_) => "error",
    }
}

async fn is_ollama_running(client: &reqwest::Client, port: u16) -> bool {
    let url = format!("http://localhost:{port}/");

    match client.get(url).send().await {
        Ok(response) => response.status().is_success(),
        Err(_) => false,
    }
}

#[tauri::command]
pub async fn ollama_status(state: State<'_, Mutex<OllamaState>>) -> Result<Value, String> {
    let (port, models_dir, client, current_status) = {
        let guard = state.lock().map_err(|err| err.to_string())?;
        (
            guard.port,
            guard.models_dir.clone(),
            guard.client.clone(),
            guard.status.clone(),
        )
    };

    let effective_status = if is_ollama_running(&client, port).await {
        let mut guard = state.lock().map_err(|err| err.to_string())?;
        guard.status = OllamaStatus::Ready;
        OllamaStatus::Ready
    } else {
        current_status
    };

    let mut payload = Map::new();
    payload.insert(
        "status".to_string(),
        Value::String(status_label(&effective_status).to_string()),
    );

    if !matches!(effective_status, OllamaStatus::NotInstalled) {
        payload.insert("port".to_string(), json!(port));
        payload.insert(
            "models_dir".to_string(),
            Value::String(models_dir.display().to_string()),
        );
    }

    Ok(Value::Object(payload))
}

#[tauri::command]
pub async fn ollama_start(
    state: State<'_, Mutex<OllamaState>>,
    _app: tauri::AppHandle,
) -> Result<(), String> {
    let (port, models_dir, client) = {
        let guard = state.lock().map_err(|err| err.to_string())?;
        (guard.port, guard.models_dir.clone(), guard.client.clone())
    };

    if is_ollama_running(&client, port).await {
        let mut guard = state.lock().map_err(|err| err.to_string())?;
        guard.status = OllamaStatus::Ready;
        return Ok(());
    }

    std::fs::create_dir_all(&models_dir).map_err(|err| err.to_string())?;

    let host = format!("127.0.0.1:{port}");
    let mut command = Command::new("ollama");
    command
        .args(["serve"])
        .env("OLLAMA_HOST", &host)
        .env("OLLAMA_MODELS", &models_dir)
        .stdout(Stdio::null())
        .stderr(Stdio::null());

    let child = match command.spawn() {
        Ok(child) => child,
        Err(err) => {
            let mut guard = state.lock().map_err(|lock_err| lock_err.to_string())?;
            if err.kind() == std::io::ErrorKind::NotFound {
                guard.status = OllamaStatus::NotInstalled;
                return Err("ollama binary not found".to_string());
            }
            guard.status = OllamaStatus::Error(err.to_string());
            return Err(err.to_string());
        }
    };

    {
        let mut guard = state.lock().map_err(|err| err.to_string())?;
        guard.status = OllamaStatus::Starting;
        guard.process = Some(child);
    }

    match timeout(Duration::from_secs(30), async {
        loop {
            if is_ollama_running(&client, port).await {
                break;
            }
            sleep(Duration::from_millis(500)).await;
        }
    })
    .await
    {
        Ok(()) => {
            let mut guard = state.lock().map_err(|err| err.to_string())?;
            guard.status = OllamaStatus::Ready;
            Ok(())
        }
        Err(_) => {
            let mut guard = state.lock().map_err(|err| err.to_string())?;
            guard.status = OllamaStatus::Error("timeout".to_string());
            Err("timeout".to_string())
        }
    }
}

#[tauri::command]
pub async fn ollama_stop(state: State<'_, Mutex<OllamaState>>) -> Result<(), String> {
    let mut child = {
        let mut guard = state.lock().map_err(|err| err.to_string())?;
        guard.status = OllamaStatus::NotInstalled;
        guard.process.take()
    };

    if let Some(child) = child.as_mut() {
        let _ = child.kill().await;
        let _ = child.wait().await;
    }

    Ok(())
}
