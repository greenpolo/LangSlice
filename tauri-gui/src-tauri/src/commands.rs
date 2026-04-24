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

/// Load an atlas by name. Caches volumes in managed state.
#[tauri::command]
pub fn load_atlas(
    name: String,
    state: State<'_, Mutex<AppState>>,
) -> Result<AtlasMetadata, String> {
    let atlas_state = loader::load_atlas(&name)?;
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

/// Load the whole-brain outline mesh (structure 997).
#[tauri::command]
pub fn get_brain_mesh(state: State<'_, Mutex<AppState>>) -> Result<MeshData, String> {
    let app_state = state
        .lock()
        .map_err(|e| format!("State lock error: {}", e))?;
    let atlas = app_state.atlas.as_ref().ok_or("No atlas loaded")?;

    let mesh_path = atlas.atlas_dir.join("meshes").join("997.obj");
    if !mesh_path.exists() {
        return Err(format!("Brain mesh not found: {}", mesh_path.display()));
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
#[tauri::command]
pub async fn run_estimate(
    image_path: String,
    atlas: String,
    model: String,
    thinking: String,
    temperature: f64,
    vlm_resolution: u32,
    max_iterations: u32,
    workflow: String,
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
        "--model".to_string(),
        model,
        "--thinking".to_string(),
        thinking,
        "--temperature".to_string(),
        temperature.to_string(),
        "--vlm-resolution".to_string(),
        vlm_resolution.to_string(),
        "--max-iterations".to_string(),
        max_iterations.to_string(),
        "--json".to_string(),
    ];
    if workflow != "auto" {
        args.push("--workflow".to_string());
        args.push(workflow);
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
#[tauri::command]
pub async fn run_register(
    image_path: String,
    position_mm: f64,
    atlas: String,
    model: String,
    thinking: String,
    temperature: f64,
    vlm_resolution: u32,
    app: tauri::AppHandle,
) -> Result<serde_json::Value, String> {
    use tauri::Emitter;
    use tokio::io::{AsyncBufReadExt, BufReader};
    use tokio::process::Command;

    let args = vec![
        "-m".to_string(),
        "langslice_harness".to_string(),
        "register".to_string(),
        image_path,
        "--atlas".to_string(),
        atlas,
        "--position".to_string(),
        position_mm.to_string(),
        "--model".to_string(),
        model,
        "--thinking".to_string(),
        thinking,
        "--temperature".to_string(),
        temperature.to_string(),
        "--vlm-resolution".to_string(),
        vlm_resolution.to_string(),
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

/// Run `langslice register` with export output.
#[tauri::command]
pub async fn run_export(
    image_path: String,
    position_mm: f64,
    atlas: String,
    output_dir: String,
) -> Result<String, String> {
    use tokio::process::Command;

    let output = Command::new("python")
        .args([
            "-m",
            "langslice",
            "register",
            &image_path,
            "--atlas",
            &atlas,
            "--position",
            &position_mm.to_string(),
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
    // Look for .env relative to the langslice package
    // In development, this is C:\LabSoftware\LangSlice\.env
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
