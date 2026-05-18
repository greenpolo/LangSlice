use std::num::NonZeroUsize;
use std::sync::Mutex;

use base64::Engine;
use lru::LruCache;
use tauri::State;

use std::path::Path;

use crate::atlas::loader;
use crate::atlas::slicer;
use crate::atlas::types::{AtlasMetadata, MeshData, SliceResult};

/// Cached thumbnail: pre-encoded JPEG base64 + dimensions.
struct CachedThumb {
    b64: String,
    original_w: u32,
    original_h: u32,
    display_w: u32,
    display_h: u32,
}

/// Holds the currently loaded atlas and image cache.
pub struct AppState {
    pub atlas: Option<crate::atlas::types::AtlasState>,
    /// LRU cache of pre-encoded JPEG thumbnails, keyed by file path.
    pub image_cache: LruCache<String, CachedThumb>,
}

impl AppState {
    pub fn new() -> Self {
        Self {
            atlas: None,
            image_cache: LruCache::new(NonZeroUsize::new(500).unwrap()),
        }
    }
}

/// List downloaded BrainGlobe atlases.
#[tauri::command]
pub fn list_atlases() -> Result<Vec<String>, String> {
    loader::list_downloaded_atlases()
}

/// Metadata for one BrainGlobe atlas — what the manager modal renders per row.
#[derive(serde::Serialize)]
pub struct AvailableAtlas {
    pub name: String,
    pub version: String,
    pub latest_version: String,
    pub downloaded: bool,
    pub updated: bool,
}

/// List every atlas the BrainGlobe API knows about, with `downloaded` set per
/// entry. Shells out to Python because the registry lives in
/// `brainglobe-atlasapi`.
///
/// Uses `get_all_atlases_lastversions()` (the full remote registry, ~200
/// atlases) as the baseline and overlays per-atlas local state from
/// `get_atlases_lastversions()` so installed atlases show their cached
/// version. The remote call falls back to local-only if the network is
/// unreachable, so the modal still works offline.
#[tauri::command]
pub async fn list_available_atlases() -> Result<Vec<AvailableAtlas>, String> {
    use tokio::process::Command;

    let script = r#"
import json
from brainglobe_atlasapi.list_atlases import get_atlases_lastversions
local = get_atlases_lastversions()
try:
    from brainglobe_atlasapi.list_atlases import get_all_atlases_lastversions
    remote = get_all_atlases_lastversions()
except Exception:
    remote = {name: info.get("latest_version", "") for name, info in local.items()}
out = {}
for name, latest in remote.items():
    info = local.get(name)
    if info is not None:
        out[name] = {
            "version": info.get("version", ""),
            "latest_version": info.get("latest_version", latest),
            "downloaded": bool(info.get("downloaded", False)),
            "updated": bool(info.get("updated", False)),
        }
    else:
        out[name] = {
            "version": "",
            "latest_version": str(latest),
            "downloaded": False,
            "updated": False,
        }
print(json.dumps(out))
"#;

    let output = Command::new("python")
        .args(["-c", script])
        .output()
        .await
        .map_err(|e| format!("Failed to spawn python: {}", e))?;

    if !output.status.success() {
        let stderr = String::from_utf8_lossy(&output.stderr);
        return Err(format!("brainglobe listing failed: {}", stderr));
    }

    let stdout = String::from_utf8_lossy(&output.stdout);
    let parsed: serde_json::Map<String, serde_json::Value> =
        serde_json::from_str(stdout.trim())
            .map_err(|e| format!("Failed to parse atlas listing JSON: {}", e))?;

    let mut atlases: Vec<AvailableAtlas> = parsed
        .into_iter()
        .map(|(name, val)| AvailableAtlas {
            name,
            version: val
                .get("version")
                .and_then(serde_json::Value::as_str)
                .unwrap_or("")
                .to_string(),
            latest_version: val
                .get("latest_version")
                .and_then(serde_json::Value::as_str)
                .unwrap_or("")
                .to_string(),
            downloaded: val
                .get("downloaded")
                .and_then(serde_json::Value::as_bool)
                .unwrap_or(false),
            updated: val
                .get("updated")
                .and_then(serde_json::Value::as_bool)
                .unwrap_or(false),
        })
        .collect();

    atlases.sort_by(|a, b| a.name.cmp(&b.name));
    Ok(atlases)
}

/// Download a BrainGlobe atlas, emitting `atlas-download-progress` events as
/// the download proceeds. The Python helper streams JSON-line progress to
/// stdout; each line has `phase` plus optional `completed`/`total` bytes.
#[tauri::command]
pub async fn download_atlas(name: String, app: tauri::AppHandle) -> Result<(), String> {
    use tauri::Emitter;
    use tokio::io::{AsyncBufReadExt, BufReader};
    use tokio::process::Command;

    let script = r#"
import sys, json
from brainglobe_atlasapi import BrainGlobeAtlas
name = sys.argv[1]
last = [0]
def cb(completed, total):
    last[0] = total
    print(json.dumps({"phase":"downloading","completed":completed,"total":total}), flush=True)
print(json.dumps({"phase":"resolving"}), flush=True)
BrainGlobeAtlas(name, fn_update=cb, check_latest=False)
print(json.dumps({"phase":"extracting"}), flush=True)
print(json.dumps({"phase":"complete","total":last[0]}), flush=True)
"#;

    let mut child = Command::new("python")
        .args(["-c", script, &name])
        .stdout(std::process::Stdio::piped())
        .stderr(std::process::Stdio::piped())
        .spawn()
        .map_err(|e| format!("Failed to spawn python: {}", e))?;

    let stdout = child.stdout.take().ok_or("No stdout")?;
    let stderr = child.stderr.take().ok_or("No stderr")?;

    let app_for_stderr = app.clone();
    let name_for_stderr = name.clone();
    let stderr_handle = tokio::spawn(async move {
        let reader = BufReader::new(stderr);
        let mut lines = reader.lines();
        while let Ok(Some(line)) = lines.next_line().await {
            let _ = app_for_stderr.emit(
                "atlas-download-log",
                serde_json::json!({ "atlas": name_for_stderr, "line": line }),
            );
        }
    });

    let reader = BufReader::new(stdout);
    let mut lines = reader.lines();
    while let Ok(Some(line)) = lines.next_line().await {
        let trimmed = line.trim();
        if trimmed.is_empty() {
            continue;
        }
        let mut payload: serde_json::Value = match serde_json::from_str(trimmed) {
            Ok(v) => v,
            Err(_) => continue,
        };
        if let Some(obj) = payload.as_object_mut() {
            obj.insert("atlas".to_string(), serde_json::Value::String(name.clone()));
        }
        let _ = app.emit("atlas-download-progress", &payload);
    }

    stderr_handle.await.ok();

    let status = child
        .wait()
        .await
        .map_err(|e| format!("Process error: {}", e))?;
    if !status.success() {
        return Err(format!(
            "Atlas download failed (exit code {:?})",
            status.code()
        ));
    }
    Ok(())
}

/// Load an atlas by name. Caches volumes in managed state.
#[tauri::command]
pub fn load_atlas(
    name: String,
    app: tauri::AppHandle,
    state: State<'_, Mutex<AppState>>,
) -> Result<AtlasMetadata, String> {
    use tauri::Emitter;

    // Emit phase-boundary progress events so the frontend can render a
    // determinate bar. Payload shape matches AtlasLoadProgress in lib/types.ts.
    let atlas_state = loader::load_atlas(&name, |phase, percent| {
        let _ = app.emit(
            "atlas-load-progress",
            serde_json::json!({ "phase": phase, "percent": percent }),
        );
    })?;
    let metadata = atlas_state.metadata.clone();

    let mut app_state = state
        .lock()
        .map_err(|e| format!("State lock error: {}", e))?;
    app_state.atlas = Some(atlas_state);

    Ok(metadata)
}

/// Get atlas metadata and AP range for the currently loaded atlas.
#[tauri::command]
pub fn get_atlas_info(state: State<'_, Mutex<AppState>>) -> Result<serde_json::Value, String> {
    let app_state = state
        .lock()
        .map_err(|e| format!("State lock error: {}", e))?;
    let atlas = app_state.atlas.as_ref().ok_or("No atlas loaded")?;

    let (min_mm, max_mm) = atlas.ap_range_mm();
    Ok(serde_json::json!({
        "name": atlas.metadata.name,
        "shape": atlas.metadata.shape,
        "resolution": atlas.metadata.resolution,
        "orientation": atlas.metadata.orientation,
        "ap_min_mm": min_mm,
        "ap_max_mm": max_mm,
        "ap_count": atlas.ap_count(),
    }))
}

/// Return the entire pre-computed border volume as bit-packed base64.
/// Each bit = 1 border pixel. Frontend unpacks per slider position — zero IPC per tick.
#[tauri::command]
pub fn get_border_volume(state: State<'_, Mutex<AppState>>) -> Result<serde_json::Value, String> {
    let app_state = state
        .lock()
        .map_err(|e| format!("State lock error: {}", e))?;
    let atlas = app_state.atlas.as_ref().ok_or("No atlas loaded")?;

    let volume = &atlas.border_volume;
    let depth = volume.shape()[0] as u32;
    let height = volume.shape()[1] as u32;
    let width = volume.shape()[2] as u32;

    // Bit-pack: 8 pixels per byte
    let raw: Vec<u8> = volume.iter().copied().collect();
    let packed: Vec<u8> = raw
        .chunks(8)
        .map(|chunk| {
            let mut byte = 0u8;
            for (i, &val) in chunk.iter().enumerate() {
                if val > 0 {
                    byte |= 1 << i;
                }
            }
            byte
        })
        .collect();

    let b64 = base64::engine::general_purpose::STANDARD.encode(&packed);

    Ok(serde_json::json!({
        "data": b64,
        "depth": depth,
        "height": height,
        "width": width,
        "format": "bitpacked",
    }))
}

/// Extract a coronal slice at a given AP position in mm.
#[tauri::command]
pub fn get_coronal_slice(
    ap_mm: f64,
    state: State<'_, Mutex<AppState>>,
) -> Result<SliceResult, String> {
    let app_state = state
        .lock()
        .map_err(|e| format!("State lock error: {}", e))?;
    let atlas = app_state.atlas.as_ref().ok_or("No atlas loaded")?;

    let idx = atlas.ap_mm_to_index(ap_mm);
    slicer::extract_coronal_slice(&atlas.reference_volume, &atlas.annotation_volume, idx)
}

/// Load the whole-brain outline mesh. The root structure ID varies by atlas
/// (Allen mouse = 997, ADMBA p14 = 15564, etc.), so we look it up from
/// structures.json via ``acronym == "root"`` rather than hardcoding.
#[tauri::command]
pub fn get_brain_mesh(state: State<'_, Mutex<AppState>>) -> Result<MeshData, String> {
    let app_state = state
        .lock()
        .map_err(|e| format!("State lock error: {}", e))?;
    let atlas = app_state.atlas.as_ref().ok_or("No atlas loaded")?;

    let root_id = atlas
        .root_structure_id()
        .ok_or("No 'root' structure (acronym=root) found in atlas")?;

    let mesh_path = atlas
        .atlas_dir
        .join("meshes")
        .join(format!("{}.obj", root_id));
    if !mesh_path.exists() {
        return Err(format!(
            "Brain mesh not found for root id {}: {}",
            root_id,
            mesh_path.display()
        ));
    }

    loader::load_obj_mesh(&mesh_path)
}

/// Normalize a u16 volume to u8 (per-slice normalization).
fn normalize_volume_to_u8(vol: &ndarray::Array3<u16>) -> Vec<u8> {
    let (depth, height, width) = (vol.shape()[0], vol.shape()[1], vol.shape()[2]);
    let pixels_per_slice = height * width;
    let mut normalized = vec![0u8; depth * pixels_per_slice];

    for z in 0..depth {
        let slice_start = z * pixels_per_slice;
        let mut max_val: u16 = 0;
        for y in 0..height {
            for x in 0..width {
                let v = vol[[z, y, x]];
                if v > max_val {
                    max_val = v;
                }
            }
        }
        if max_val > 0 {
            let scale = 255.0 / max_val as f64;
            for (i, byte) in normalized[slice_start..slice_start + pixels_per_slice]
                .iter_mut()
                .enumerate()
            {
                let y = i / width;
                let x = i % width;
                *byte = (vol[[z, y, x]] as f64 * scale).min(255.0) as u8;
            }
        }
    }
    normalized
}

/// Return all atlas volumes normalized to u8 as base64.
/// Includes reference + any additional volumes (e.g. Nissl).
#[tauri::command]
pub fn get_all_volumes(state: State<'_, Mutex<AppState>>) -> Result<serde_json::Value, String> {
    let app_state = state
        .lock()
        .map_err(|e| format!("State lock error: {}", e))?;
    let atlas = app_state.atlas.as_ref().ok_or("No atlas loaded")?;

    let depth = atlas.reference_volume.shape()[0] as u32;
    let height = atlas.reference_volume.shape()[1] as u32;
    let width = atlas.reference_volume.shape()[2] as u32;

    // Normalize and encode reference
    log::info!("Normalizing reference volume...");
    let ref_norm = normalize_volume_to_u8(&atlas.reference_volume);
    let ref_b64 = base64::engine::general_purpose::STANDARD.encode(&ref_norm);

    // Normalize and encode additional volumes
    let mut additional: serde_json::Map<String, serde_json::Value> = serde_json::Map::new();
    for (name, vol) in &atlas.additional_volumes {
        log::info!("Normalizing additional volume: {}...", name);
        let norm = normalize_volume_to_u8(vol);
        let b64 = base64::engine::general_purpose::STANDARD.encode(&norm);
        additional.insert(name.clone(), serde_json::json!(b64));
    }

    log::info!(
        "All volumes normalized: reference + {} additional",
        additional.len()
    );

    Ok(serde_json::json!({
        "reference": ref_b64,
        "additional": additional,
        "additionalNames": atlas.additional_volumes.keys().collect::<Vec<_>>(),
        "depth": depth,
        "height": height,
        "width": width,
    }))
}

/// Load an image file and return as base64 JPEG, downsampled for display.
/// Uses an LRU cache — second load of the same image is instant.
#[tauri::command]
pub fn load_slice_image(
    path: String,
    max_edge: Option<u32>,
    state: State<'_, Mutex<AppState>>,
) -> Result<serde_json::Value, String> {
    // Check cache first
    {
        let mut app = state.lock().map_err(|e| format!("Lock: {}", e))?;
        if let Some(cached) = app.image_cache.get(&path) {
            return Ok(serde_json::json!({
                "image": cached.b64,
                "originalWidth": cached.original_w,
                "originalHeight": cached.original_h,
                "displayWidth": cached.display_w,
                "displayHeight": cached.display_h,
            }));
        }
    }

    // Cache miss — load from disk
    let img = image::open(&path).map_err(|e| format!("Cannot open image {}: {}", path, e))?;

    let max = max_edge.unwrap_or(1024);
    let (w, h) = (img.width(), img.height());

    let resized = if w > max || h > max {
        let scale = max as f64 / w.max(h) as f64;
        let new_w = (w as f64 * scale).round() as u32;
        let new_h = (h as f64 * scale).round() as u32;
        img.resize(new_w, new_h, image::imageops::FilterType::Triangle)
    } else {
        img
    };

    let rgb = resized.to_rgb8();
    let (out_w, out_h) = (rgb.width(), rgb.height());

    let mut buf: Vec<u8> = Vec::new();
    let mut encoder =
        image::codecs::jpeg::JpegEncoder::new_with_quality(std::io::Cursor::new(&mut buf), 85);
    encoder
        .encode(rgb.as_raw(), out_w, out_h, image::ColorType::Rgb8.into())
        .map_err(|e| format!("JPEG encode error: {}", e))?;

    let b64 = base64::engine::general_purpose::STANDARD.encode(&buf);

    // Store in cache
    {
        let mut app = state.lock().map_err(|e| format!("Lock: {}", e))?;
        app.image_cache.put(
            path.clone(),
            CachedThumb {
                b64: b64.clone(),
                original_w: w,
                original_h: h,
                display_w: out_w,
                display_h: out_h,
            },
        );
    }

    Ok(serde_json::json!({
        "image": b64,
        "originalWidth": w,
        "originalHeight": h,
        "displayWidth": out_w,
        "displayHeight": out_h,
    }))
}

/// Run `langslice estimate` CLI and return the JSON result + stdout logs.
///
/// Argument set is a trimmed subset of `_add_estimate_parser` in
/// `src/langslice_harness/cli.py`. Flags the GUI no longer surfaces
/// (--workflow, --vlm-resolution, --borders, --grid) are left to the
/// harness defaults: auto-picks workflow (tool_use for text/vision models),
/// 2048px VLM resolution, no borders, individual atlas slices.
#[tauri::command]
pub async fn run_estimate(
    image_path: String,
    atlas: String,
    plane: String,
    model: String,
    thinking: String,
    temperature: f64,
    media_resolution: String,
    max_iterations: u32,
    preprocess: String,
    provider: String,
    endpoint: Option<String>,
    app: tauri::AppHandle,
) -> Result<serde_json::Value, String> {
    use tauri::Emitter;
    use tokio::io::{AsyncBufReadExt, BufReader};
    use tokio::process::Command;

    let mut args = vec![
        "-m".to_string(),
        "langslice_harness".to_string(),
        "estimate".to_string(),
        image_path,
        "--atlas".to_string(),
        atlas,
        "--plane".to_string(),
        plane,
        "--model".to_string(),
        model,
        "--thinking".to_string(),
        thinking,
        "--temperature".to_string(),
        temperature.to_string(),
        "--media-resolution".to_string(),
        media_resolution,
        "--max-iterations".to_string(),
        max_iterations.to_string(),
        "--preprocess".to_string(),
        preprocess,
        "--provider".to_string(),
        provider,
        "--json".to_string(),
    ];
    if let Some(ep) = endpoint {
        if !ep.is_empty() {
            args.push("--endpoint".to_string());
            args.push(ep);
        }
    }

    let mut child = Command::new("python")
        .args(&args)
        .stdout(std::process::Stdio::piped())
        .stderr(std::process::Stdio::piped())
        .spawn()
        .map_err(|e| format!("Failed to spawn python: {}", e))?;

    let stdout = child.stdout.take().ok_or("No stdout")?;
    let stderr = child.stderr.take().ok_or("No stderr")?;

    let app_clone = app.clone();

    // Stream stderr as log lines
    let stderr_handle = tokio::spawn(async move {
        let reader = BufReader::new(stderr);
        let mut lines = reader.lines();
        while let Ok(Some(line)) = lines.next_line().await {
            let _ = app_clone.emit("pipeline-log", &line);
        }
    });

    // Capture all stdout
    let reader = BufReader::new(stdout);
    let mut lines = reader.lines();
    let mut all_stdout = Vec::new();
    while let Ok(Some(line)) = lines.next_line().await {
        let _ = app.emit("pipeline-log", &line);
        all_stdout.push(line);
    }

    stderr_handle.await.ok();

    let status = child
        .wait()
        .await
        .map_err(|e| format!("Process error: {}", e))?;
    if !status.success() {
        return Err(format!("Estimation failed (exit code {:?})", status.code()));
    }

    // Extract JSON from the end of stdout
    let stdout_text = all_stdout.join("\n");
    extract_json(&stdout_text)
}

/// Run `langslice register` CLI and return the JSON result.
///
/// Argument set is a trimmed subset of `_add_register_parser` in
/// `src/langslice_harness/cli.py`. Flags the GUI no longer surfaces
/// (--registration-mode, --max-candidates, --openai-image-route,
/// --vlm-resolution, --review-model) fall back to harness defaults: direct
/// mode (agentic review is unverified), 3 candidates, /v1/images route,
/// 2048px VLM, review-model defaults to the image-model fallback chain.
/// The register subcommand does NOT accept --media-resolution.
#[tauri::command]
pub async fn run_register(
    image_path: String,
    position_mm: f64,
    atlas: String,
    plane: String,
    image_model: String,
    thinking: String,
    temperature: f64,
    provider: String,
    endpoint: Option<String>,
    app: tauri::AppHandle,
) -> Result<serde_json::Value, String> {
    use tauri::Emitter;
    use tokio::io::{AsyncBufReadExt, BufReader};
    use tokio::process::Command;

    let mut args = vec![
        "-m".to_string(),
        "langslice_harness".to_string(),
        "register".to_string(),
        image_path,
        "--atlas".to_string(),
        atlas,
        "--position".to_string(),
        position_mm.to_string(),
        "--plane".to_string(),
        plane,
        "--image-model".to_string(),
        image_model,
        "--thinking".to_string(),
        thinking,
        "--temperature".to_string(),
        temperature.to_string(),
        "--provider".to_string(),
        provider,
        "--json".to_string(),
    ];
    if let Some(ep) = endpoint {
        if !ep.is_empty() {
            args.push("--endpoint".to_string());
            args.push(ep);
        }
    }
    let mut child = Command::new("python")
        .args(&args)
        .stdout(std::process::Stdio::piped())
        .stderr(std::process::Stdio::piped())
        .spawn()
        .map_err(|e| format!("Failed to spawn python: {}", e))?;

    let stdout = child.stdout.take().ok_or("No stdout")?;
    let stderr = child.stderr.take().ok_or("No stderr")?;

    let app_clone = app.clone();

    let stderr_handle = tokio::spawn(async move {
        let reader = BufReader::new(stderr);
        let mut lines = reader.lines();
        while let Ok(Some(line)) = lines.next_line().await {
            let _ = app_clone.emit("pipeline-log", &line);
        }
    });

    let reader = BufReader::new(stdout);
    let mut lines = reader.lines();
    let mut all_stdout = Vec::new();
    while let Ok(Some(line)) = lines.next_line().await {
        let _ = app.emit("pipeline-log", &line);
        all_stdout.push(line);
    }

    stderr_handle.await.ok();

    let status = child
        .wait()
        .await
        .map_err(|e| format!("Process error: {}", e))?;
    if !status.success() {
        return Err(format!(
            "Registration failed (exit code {:?})",
            status.code()
        ));
    }

    let stdout_text = all_stdout.join("\n");
    extract_json(&stdout_text)
}

/// Run the silhouette-based affine preview pipeline. Used by the GUI to
/// drop a brain-shaped warped slice into the 3D viewer the moment the user
/// locks an AP position, before the full nano-banana pipeline runs.
///
/// Output path is derived server-side from a hash of (image_path, position,
/// atlas, plane) inside Tauri's app cache directory — never the source
/// folder, so the histology scanner can't accidentally pick the preview up
/// as another slice. Re-locking at the same coordinates overwrites in place;
/// different positions produce distinct cache files.
#[tauri::command]
pub async fn run_quick_affine(
    image_path: String,
    position_mm: f64,
    atlas: String,
    plane: String,
    app: tauri::AppHandle,
) -> Result<serde_json::Value, String> {
    use std::hash::{Hash, Hasher};
    use tauri::{Emitter, Manager};
    use tokio::io::{AsyncBufReadExt, BufReader};
    use tokio::process::Command;

    // Hash the inputs into a stable filename. Quantize position to integer
    // micrometres so micro-jitter in the slider doesn't generate thousands
    // of distinct cache files.
    let mut hasher = std::collections::hash_map::DefaultHasher::new();
    image_path.hash(&mut hasher);
    atlas.hash(&mut hasher);
    plane.hash(&mut hasher);
    ((position_mm * 1000.0).round() as i64).hash(&mut hasher);
    let filename = format!("qa_{:016x}.png", hasher.finish());

    let cache_dir = app
        .path()
        .app_cache_dir()
        .map_err(|e| format!("Failed to resolve app cache dir: {}", e))?
        .join("quick_affine");
    std::fs::create_dir_all(&cache_dir)
        .map_err(|e| format!("Failed to create cache dir: {}", e))?;
    let out_path = cache_dir.join(&filename);
    let out_path_str = out_path.to_string_lossy().to_string();

    let args = vec![
        "-m".to_string(),
        "langslice_harness".to_string(),
        "quick-affine".to_string(),
        image_path,
        "--atlas".to_string(),
        atlas,
        "--position".to_string(),
        position_mm.to_string(),
        "--plane".to_string(),
        plane,
        "--out".to_string(),
        out_path_str,
        "--json".to_string(),
    ];

    let mut child = Command::new("python")
        .args(&args)
        .stdout(std::process::Stdio::piped())
        .stderr(std::process::Stdio::piped())
        .spawn()
        .map_err(|e| format!("Failed to spawn python: {}", e))?;

    let stdout = child.stdout.take().ok_or("No stdout")?;
    let stderr = child.stderr.take().ok_or("No stderr")?;
    let app_clone = app.clone();

    let stderr_handle = tokio::spawn(async move {
        let reader = BufReader::new(stderr);
        let mut lines = reader.lines();
        while let Ok(Some(line)) = lines.next_line().await {
            let _ = app_clone.emit("pipeline-log", &format!("[quick-affine] {}", line));
        }
    });

    let reader = BufReader::new(stdout);
    let mut lines = reader.lines();
    let mut all_stdout = Vec::new();
    while let Ok(Some(line)) = lines.next_line().await {
        let _ = app.emit("pipeline-log", &format!("[quick-affine] {}", line));
        all_stdout.push(line);
    }
    stderr_handle.await.ok();

    let status = child
        .wait()
        .await
        .map_err(|e| format!("Process error: {}", e))?;
    if !status.success() {
        return Err(format!(
            "Quick-affine failed (exit code {:?})",
            status.code()
        ));
    }

    let stdout_text = all_stdout.join("\n");
    extract_json(&stdout_text)
}

/// Extract the last JSON object from mixed stdout output.
fn extract_json(stdout: &str) -> Result<serde_json::Value, String> {
    let mut depth = 0i32;
    let mut json_start = None;
    let mut json_end = None;

    for (i, ch) in stdout.char_indices().rev() {
        if json_end.is_none() && ch == '}' {
            json_end = Some(i + 1);
            depth = 1;
        } else if json_end.is_some() {
            if ch == '}' {
                depth += 1;
            }
            if ch == '{' {
                depth -= 1;
            }
            if depth == 0 {
                json_start = Some(i);
                break;
            }
        }
    }

    match (json_start, json_end) {
        (Some(start), Some(end)) => serde_json::from_str(&stdout[start..end])
            .map_err(|e| format!("JSON parse error: {}", e)),
        _ => Err("No JSON object found in output".into()),
    }
}

/// Run `langslice_harness register` with export output.
#[tauri::command]
pub async fn run_export(
    image_path: String,
    position_mm: f64,
    atlas: String,
    plane: String,
    output_dir: String,
) -> Result<String, String> {
    use tokio::process::Command;

    let output = Command::new("python")
        .args([
            "-m",
            "langslice_harness",
            "register",
            &image_path,
            "--atlas",
            &atlas,
            "--position",
            &position_mm.to_string(),
            "--plane",
            &plane,
            "--out",
            &output_dir,
            "--json",
        ])
        .output()
        .await
        .map_err(|e| format!("Failed to spawn python: {}", e))?;

    if !output.status.success() {
        let stderr = String::from_utf8_lossy(&output.stderr);
        return Err(format!("Export failed: {}", stderr));
    }

    Ok(output_dir)
}

/// Read the .env file and return all key-value pairs.
#[tauri::command]
pub fn read_env_file() -> Result<serde_json::Value, String> {
    let env_path = find_env_path();
    let mut vars: serde_json::Map<String, serde_json::Value> = serde_json::Map::new();

    if env_path.exists() {
        let content =
            std::fs::read_to_string(&env_path).map_err(|e| format!("Cannot read .env: {}", e))?;
        for line in content.lines() {
            let trimmed = line.trim();
            if trimmed.is_empty() || trimmed.starts_with('#') {
                continue;
            }
            if let Some((key, val)) = trimmed.split_once('=') {
                let val = val.trim().trim_matches('"').trim_matches('\'');
                vars.insert(key.trim().to_string(), serde_json::json!(val));
            }
        }
    }

    Ok(serde_json::json!({
        "path": env_path.to_string_lossy(),
        "vars": vars,
    }))
}

/// Write key-value pairs to the .env file.
#[tauri::command]
pub fn write_env_file(vars: serde_json::Value) -> Result<(), String> {
    let env_path = find_env_path();
    let obj = vars.as_object().ok_or("Expected object")?;

    // Read existing content to preserve comments and ordering
    let mut lines: Vec<String> = Vec::new();
    let mut existing_keys: std::collections::HashSet<String> = std::collections::HashSet::new();

    if env_path.exists() {
        let content =
            std::fs::read_to_string(&env_path).map_err(|e| format!("Cannot read .env: {}", e))?;
        for line in content.lines() {
            let trimmed = line.trim();
            if let Some((key, _)) = trimmed.split_once('=') {
                let key = key.trim().to_string();
                if !trimmed.starts_with('#') {
                    if let Some(new_val) = obj.get(&key) {
                        let val_str = new_val.as_str().unwrap_or("");
                        lines.push(format!("{}=\"{}\"", key, val_str));
                        existing_keys.insert(key);
                        continue;
                    }
                }
            }
            lines.push(line.to_string());
        }
    }

    // Add new keys not already in the file
    for (key, val) in obj {
        if !existing_keys.contains(key) {
            let val_str = val.as_str().unwrap_or("");
            if !val_str.is_empty() {
                lines.push(format!("{}=\"{}\"", key, val_str));
            }
        }
    }

    std::fs::write(&env_path, lines.join("\n") + "\n")
        .map_err(|e| format!("Cannot write .env: {}", e))?;

    log::info!("Saved .env at {}", env_path.display());
    Ok(())
}

/// Find the .env file path (project root).
fn find_env_path() -> std::path::PathBuf {
    // Look for .env relative to the local development checkout.
    let candidates = [
        std::path::PathBuf::from("C:/LabSoftware/LangSlice/.env"),
        dirs::home_dir().unwrap_or_default().join(".langslice.env"),
    ];
    for c in &candidates {
        if c.exists() {
            return c.clone();
        }
    }
    // Default to the first candidate
    candidates[0].clone()
}

/// Pre-cache all images in a folder as JPEG thumbnails using parallel processing.
/// Emits "cache-progress" events to the frontend as images are processed.
#[tauri::command]
pub async fn precache_images(
    paths: Vec<String>,
    max_edge: Option<u32>,
    state: State<'_, Mutex<AppState>>,
    app: tauri::AppHandle,
) -> Result<u32, String> {
    use rayon::prelude::*;
    use tauri::Emitter;

    let max = max_edge.unwrap_or(1024);
    let total = paths.len();

    // Filter out already-cached paths
    let uncached: Vec<String> = {
        let app_state = state.lock().map_err(|e| format!("Lock: {}", e))?;
        paths
            .into_iter()
            .filter(|p| !app_state.image_cache.contains(p))
            .collect()
    };

    if uncached.is_empty() {
        return Ok(total as u32);
    }

    let _ = app.emit(
        "cache-progress",
        serde_json::json!({
            "done": total - uncached.len(),
            "total": total,
        }),
    );

    // Process all images in parallel using rayon
    let results: Vec<(String, Option<CachedThumb>)> = uncached
        .par_iter()
        .map(|path| {
            let thumb = encode_thumbnail(path, max);
            (path.clone(), thumb)
        })
        .collect();

    // Insert all results into the cache (must hold lock)
    let mut cached_count = 0u32;
    {
        let mut app_state = state.lock().map_err(|e| format!("Lock: {}", e))?;
        for (path, thumb) in results {
            if let Some(t) = thumb {
                app_state.image_cache.put(path, t);
                cached_count += 1;
            }
        }
    }

    let _ = app.emit(
        "cache-progress",
        serde_json::json!({
            "done": total,
            "total": total,
        }),
    );

    log::info!("Pre-cached {}/{} images", cached_count, total);
    Ok(cached_count)
}

/// Encode a single image to a JPEG thumbnail. Pure function, no state.
fn encode_thumbnail(path: &str, max_edge: u32) -> Option<CachedThumb> {
    let img = image::open(path).ok()?;
    let (w, h) = (img.width(), img.height());

    let resized = if w > max_edge || h > max_edge {
        let scale = max_edge as f64 / w.max(h) as f64;
        let new_w = (w as f64 * scale).round() as u32;
        let new_h = (h as f64 * scale).round() as u32;
        img.resize(new_w, new_h, image::imageops::FilterType::Triangle)
    } else {
        img
    };

    let rgb = resized.to_rgb8();
    let (out_w, out_h) = (rgb.width(), rgb.height());

    let mut buf: Vec<u8> = Vec::new();
    let mut encoder =
        image::codecs::jpeg::JpegEncoder::new_with_quality(std::io::Cursor::new(&mut buf), 85);
    encoder
        .encode(rgb.as_raw(), out_w, out_h, image::ColorType::Rgb8.into())
        .ok()?;

    let b64 = base64::engine::general_purpose::STANDARD.encode(&buf);

    Some(CachedThumb {
        b64,
        original_w: w,
        original_h: h,
        display_w: out_w,
        display_h: out_h,
    })
}

const IMAGE_EXTENSIONS: &[&str] = &["png", "jpg", "jpeg", "tif", "tiff", "bmp"];

/// Scan a directory for image files and return their paths + names.
#[tauri::command]
pub fn scan_image_folder(folder: String) -> Result<Vec<serde_json::Value>, String> {
    let dir = Path::new(&folder);
    if !dir.is_dir() {
        return Err(format!("Not a directory: {}", folder));
    }

    let mut images: Vec<serde_json::Value> = Vec::new();
    let entries = std::fs::read_dir(dir).map_err(|e| format!("Cannot read directory: {}", e))?;

    for entry in entries.flatten() {
        let path = entry.path();
        if path.is_file() {
            if let Some(ext) = path.extension().and_then(|e| e.to_str()) {
                if IMAGE_EXTENSIONS.contains(&ext.to_lowercase().as_str()) {
                    let name = path
                        .file_name()
                        .unwrap_or_default()
                        .to_string_lossy()
                        .to_string();
                    images.push(serde_json::json!({
                        "path": path.to_string_lossy(),
                        "name": name,
                    }));
                }
            }
        }
    }

    images.sort_by(|a, b| {
        a["name"]
            .as_str()
            .unwrap_or("")
            .cmp(b["name"].as_str().unwrap_or(""))
    });

    Ok(images)
}
