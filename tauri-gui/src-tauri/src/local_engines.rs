//! Detect locally-running OpenAI-compatible LLM servers.
//!
//! Concurrently probes a small registry of well-known localhost ports for
//! `/v1/models`. Engines that respond are reported with their model list;
//! everything else comes back as `reachable: false` after a short timeout.
//!
//! Each engine is tried first on 127.0.0.1 and then on `[::1]`; some servers
//! on Windows bind to one stack and not the other.
//!
//! The Tauri command exposed here (`probe_local_engines`) is called by the
//! Local Models modal when it opens and on user-triggered refresh.
use std::time::Duration;

use futures_util::future::join_all;
use serde::Serialize;
use serde_json::Value;

/// One row in the Local Models modal.
#[derive(Serialize, Clone, Debug)]
pub struct LocalEngineStatus {
    pub name: String,
    pub port: u16,
    pub endpoint: String,
    pub reachable: bool,
    pub models: Vec<LocalModel>,
    pub error: Option<String>,
}

/// One model returned by an engine's `/v1/models` listing.
#[derive(Serialize, Clone, Debug)]
pub struct LocalModel {
    pub id: String,
    pub engine: String,
    pub endpoint: String,
}

/// Engines probed by default. Order is preserved for UI rendering.
///
/// `LiteRT-LM` runs on the port chosen by `litert_lm::DEFAULT_PORT` (8765) so
/// the GUI-managed sidecar shows up here without colliding with the canonical
/// llama-server port. If the user runs a separate `litert-lm serve` instance
/// on a different port, they can add it via the custom-endpoint flow.
const ENGINES: &[(&str, u16)] = &[
    ("Ollama", 11434),
    ("LM Studio", 1234),
    ("llama-server", 8080),
    ("vLLM", 8000),
    ("Jan", 1337),
    ("LiteRT-LM", 8765),
];

const PROBE_TIMEOUT: Duration = Duration::from_millis(400);

async fn try_host(
    client: &reqwest::Client,
    host: &str,
    port: u16,
) -> Option<(String, Vec<String>)> {
    let endpoint = format!("http://{}:{}/v1", host, port);
    let url = format!("{}/models", endpoint);
    let resp = client.get(&url).send().await.ok()?;
    if !resp.status().is_success() {
        return None;
    }
    let body: Value = resp.json().await.ok()?;
    let ids: Vec<String> = body
        .get("data")
        .and_then(Value::as_array)
        .into_iter()
        .flatten()
        .filter_map(|m| m.get("id").and_then(Value::as_str).map(str::to_owned))
        .collect();
    Some((endpoint, ids))
}

async fn probe_engine(client: &reqwest::Client, name: &str, port: u16) -> LocalEngineStatus {
    // Try IPv4 first, then IPv6. Most engines bind to one or the other but
    // not both, so a sequential fallback is fine here. Total worst-case
    // wait per engine is bounded by the client-level timeout * 2.
    for host in ["127.0.0.1", "[::1]"] {
        if let Some((endpoint, ids)) = try_host(client, host, port).await {
            let models = ids
                .into_iter()
                .map(|id| LocalModel {
                    id,
                    engine: name.to_string(),
                    endpoint: endpoint.clone(),
                })
                .collect();
            return LocalEngineStatus {
                name: name.to_string(),
                port,
                endpoint,
                reachable: true,
                models,
                error: None,
            };
        }
    }
    LocalEngineStatus {
        name: name.to_string(),
        port,
        endpoint: format!("http://127.0.0.1:{}/v1", port),
        reachable: false,
        models: Vec::new(),
        error: Some("not running".to_string()),
    }
}

/// Probe every engine in the registry concurrently. Always returns one
/// entry per engine, in registry order.
pub async fn probe_all_engines() -> Vec<LocalEngineStatus> {
    let client = match reqwest::Client::builder().timeout(PROBE_TIMEOUT).build() {
        Ok(c) => c,
        Err(_) => reqwest::Client::new(),
    };
    let futures = ENGINES
        .iter()
        .map(|(name, port)| probe_engine(&client, name, *port));
    join_all(futures).await
}

/// Probe a single user-supplied custom endpoint (e.g. a remote server or
/// a non-standard port). Treats whatever the user typed as the canonical
/// `endpoint` and labels the engine after the supplied label.
pub async fn probe_custom_endpoint(label: &str, base_url: &str) -> LocalEngineStatus {
    let client = match reqwest::Client::builder().timeout(PROBE_TIMEOUT).build() {
        Ok(c) => c,
        Err(_) => reqwest::Client::new(),
    };
    let endpoint = base_url.trim_end_matches('/').to_string();
    let url = format!("{}/models", endpoint);
    let resp = client.get(&url).send().await;
    match resp {
        Ok(r) if r.status().is_success() => {
            let body: Value = match r.json().await {
                Ok(v) => v,
                Err(e) => {
                    return LocalEngineStatus {
                        name: label.to_string(),
                        port: 0,
                        endpoint,
                        reachable: false,
                        models: Vec::new(),
                        error: Some(format!("invalid response: {}", e)),
                    };
                }
            };
            let models: Vec<LocalModel> = body
                .get("data")
                .and_then(Value::as_array)
                .into_iter()
                .flatten()
                .filter_map(|m| m.get("id").and_then(Value::as_str))
                .map(|id| LocalModel {
                    id: id.to_string(),
                    engine: label.to_string(),
                    endpoint: endpoint.clone(),
                })
                .collect();
            LocalEngineStatus {
                name: label.to_string(),
                port: 0,
                endpoint,
                reachable: true,
                models,
                error: None,
            }
        }
        Ok(r) => LocalEngineStatus {
            name: label.to_string(),
            port: 0,
            endpoint,
            reachable: false,
            models: Vec::new(),
            error: Some(format!("HTTP {}", r.status().as_u16())),
        },
        Err(e) => LocalEngineStatus {
            name: label.to_string(),
            port: 0,
            endpoint,
            reachable: false,
            models: Vec::new(),
            error: Some(e.to_string()),
        },
    }
}

#[tauri::command]
pub async fn probe_local_engines() -> Result<Vec<LocalEngineStatus>, String> {
    Ok(probe_all_engines().await)
}

#[tauri::command]
pub async fn probe_custom_local_endpoint(
    label: String,
    url: String,
) -> Result<LocalEngineStatus, String> {
    Ok(probe_custom_endpoint(&label, &url).await)
}
