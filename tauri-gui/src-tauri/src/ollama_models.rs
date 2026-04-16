use std::collections::HashSet;
use std::sync::Mutex;

use futures_util::StreamExt;
use serde::Serialize;
use serde_json::{json, Value};
use tauri::{Emitter, State};

use crate::ollama::OllamaState;

#[derive(Serialize, Clone)]
pub struct CuratedModel {
    pub name: String,
    pub display_name: String,
    pub description: String,
    pub size_gb: f32,
    pub min_vram_gb: u32,
    pub capabilities: Vec<String>,
    pub recommended: bool,
}

fn curated_models() -> Vec<CuratedModel> {
    vec![
        CuratedModel {
            name: "gemma4:31b".to_string(),
            display_name: "Gemma 4 31B".to_string(),
            description: "Best accuracy for AP estimation. Requires 16+ GB VRAM.".to_string(),
            size_gb: 20.0,
            min_vram_gb: 14,
            capabilities: vec![
                "vision".to_string(),
                "tool-use".to_string(),
                "thinking".to_string(),
            ],
            recommended: true,
        },
        CuratedModel {
            name: "gemma4:26b".to_string(),
            display_name: "Gemma 4 26B".to_string(),
            description: "MoE variant, good accuracy".to_string(),
            size_gb: 16.0,
            min_vram_gb: 12,
            capabilities: vec![
                "vision".to_string(),
                "tool-use".to_string(),
                "thinking".to_string(),
            ],
            recommended: false,
        },
        CuratedModel {
            name: "gemma4:e4b".to_string(),
            display_name: "Gemma 4 E4B".to_string(),
            description: "Lightweight, runs on most GPUs".to_string(),
            size_gb: 3.0,
            min_vram_gb: 4,
            capabilities: vec!["vision".to_string(), "tool-use".to_string()],
            recommended: false,
        },
        CuratedModel {
            name: "gemma4:e2b".to_string(),
            display_name: "Gemma 4 E2B".to_string(),
            description: "Minimal footprint, runs on CPU".to_string(),
            size_gb: 1.5,
            min_vram_gb: 2,
            capabilities: vec!["vision".to_string(), "tool-use".to_string()],
            recommended: false,
        },
    ]
}

fn extract_installed_names(payload: &Value) -> HashSet<String> {
    payload
        .get("models")
        .and_then(Value::as_array)
        .into_iter()
        .flatten()
        .filter_map(|model| model.get("name").and_then(Value::as_str))
        .map(ToOwned::to_owned)
        .collect()
}

async fn get_tags(client: &reqwest::Client, port: u16) -> Result<Value, String> {
    let url = format!("http://localhost:{port}/api/tags");
    let response = client
        .get(url)
        .send()
        .await
        .map_err(|err| err.to_string())?;
    response
        .error_for_status()
        .map_err(|err| err.to_string())?
        .json::<Value>()
        .await
        .map_err(|err| err.to_string())
}

#[tauri::command]
pub async fn ollama_list_models(state: State<'_, Mutex<OllamaState>>) -> Result<Value, String> {
    let (port, client) = {
        let guard = state.lock().map_err(|err| err.to_string())?;
        (guard.port, guard.client.clone())
    };

    get_tags(&client, port).await
}

#[tauri::command]
pub async fn ollama_model_info(
    name: String,
    state: State<'_, Mutex<OllamaState>>,
) -> Result<Value, String> {
    let (port, client) = {
        let guard = state.lock().map_err(|err| err.to_string())?;
        (guard.port, guard.client.clone())
    };

    let url = format!("http://localhost:{port}/api/show");
    let response = client
        .post(url)
        .json(&json!({ "model": name }))
        .send()
        .await
        .map_err(|err| err.to_string())?;

    response
        .error_for_status()
        .map_err(|err| err.to_string())?
        .json::<Value>()
        .await
        .map_err(|err| err.to_string())
}

#[tauri::command]
pub async fn ollama_pull_model(
    name: String,
    state: State<'_, Mutex<OllamaState>>,
    app: tauri::AppHandle,
) -> Result<(), String> {
    let (port, client) = {
        let guard = state.lock().map_err(|err| err.to_string())?;
        (guard.port, guard.client.clone())
    };

    let url = format!("http://localhost:{port}/api/pull");
    let response = client
        .post(url)
        .json(&json!({ "model": name, "stream": true }))
        .send()
        .await
        .map_err(|err| err.to_string())?;

    let response = response.error_for_status().map_err(|err| err.to_string())?;
    let mut stream = response.bytes_stream();
    let mut buffer = Vec::<u8>::new();

    while let Some(chunk) = stream.next().await {
        let chunk = chunk.map_err(|err| err.to_string())?;
        buffer.extend_from_slice(&chunk);

        while let Some(newline_pos) = buffer.iter().position(|byte| *byte == b'\n') {
            let line_bytes: Vec<u8> = buffer.drain(..=newline_pos).collect();
            let line = String::from_utf8(line_bytes).map_err(|err| err.to_string())?;
            let trimmed = line.trim();

            if trimmed.is_empty() {
                continue;
            }

            let payload: Value = serde_json::from_str(trimmed).map_err(|err| err.to_string())?;
            emit_pull_progress(&app, &name, &payload);
        }
    }

    if !buffer.is_empty() {
        let line = String::from_utf8(buffer).map_err(|err| err.to_string())?;
        let trimmed = line.trim();
        if !trimmed.is_empty() {
            let payload: Value = serde_json::from_str(trimmed).map_err(|err| err.to_string())?;
            emit_pull_progress(&app, &name, &payload);
        }
    }

    Ok(())
}

fn emit_pull_progress(app: &tauri::AppHandle, model: &str, payload: &Value) {
    let status = payload
        .get("status")
        .and_then(Value::as_str)
        .map(str::to_owned);
    let completed_bytes = payload.get("completed").and_then(Value::as_u64);
    let total_bytes = payload.get("total").and_then(Value::as_u64);
    let percent = match (completed_bytes, total_bytes) {
        (Some(completed), Some(total)) if total > 0 => {
            Some((completed as f64 / total as f64) * 100.0)
        }
        _ => None,
    };

    app.emit(
        "ollama-pull-progress",
        json!({
            "model": model,
            "status": status,
            "completed_bytes": completed_bytes,
            "total_bytes": total_bytes,
            "percent": percent,
        }),
    )
    .ok();
}

#[tauri::command]
pub async fn ollama_delete_model(
    name: String,
    state: State<'_, Mutex<OllamaState>>,
) -> Result<(), String> {
    let (port, client) = {
        let guard = state.lock().map_err(|err| err.to_string())?;
        (guard.port, guard.client.clone())
    };

    let url = format!("http://localhost:{port}/api/delete");
    client
        .request(reqwest::Method::DELETE, url)
        .json(&json!({ "model": name }))
        .send()
        .await
        .map_err(|err| err.to_string())?
        .error_for_status()
        .map_err(|err| err.to_string())?;

    Ok(())
}

#[tauri::command]
pub async fn ollama_running_models(state: State<'_, Mutex<OllamaState>>) -> Result<Value, String> {
    let (port, client) = {
        let guard = state.lock().map_err(|err| err.to_string())?;
        (guard.port, guard.client.clone())
    };

    let url = format!("http://localhost:{port}/api/ps");
    let response = client
        .get(url)
        .send()
        .await
        .map_err(|err| err.to_string())?;
    response
        .error_for_status()
        .map_err(|err| err.to_string())?
        .json::<Value>()
        .await
        .map_err(|err| err.to_string())
}

#[tauri::command]
pub async fn ollama_available_models(
    state: State<'_, Mutex<OllamaState>>,
) -> Result<Value, String> {
    let (port, client) = {
        let guard = state.lock().map_err(|err| err.to_string())?;
        (guard.port, guard.client.clone())
    };

    let installed = extract_installed_names(&get_tags(&client, port).await?);
    let payload = curated_models()
        .into_iter()
        .map(|model| {
            let is_installed = installed.contains(&model.name);
            json!({
                "name": model.name,
                "display_name": model.display_name,
                "description": model.description,
                "size_gb": model.size_gb,
                "min_vram_gb": model.min_vram_gb,
                "capabilities": model.capabilities,
                "recommended": model.recommended,
                "installed": is_installed,
            })
        })
        .collect::<Vec<_>>();

    Ok(Value::Array(payload))
}
