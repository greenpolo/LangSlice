use std::sync::Mutex;

use base64::Engine;
use tauri::State;

use std::path::Path;

use crate::atlas::loader;
use crate::atlas::slicer;
use crate::atlas::types::{AtlasMetadata, MeshData, SliceResult};

/// Holds the currently loaded atlas (if any).
pub struct AppState {
    pub atlas: Option<crate::atlas::types::AtlasState>,
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
pub fn get_atlas_info(
    state: State<'_, Mutex<AppState>>,
) -> Result<serde_json::Value, String> {
    let app_state = state
        .lock()
        .map_err(|e| format!("State lock error: {}", e))?;
    let atlas = app_state
        .atlas
        .as_ref()
        .ok_or("No atlas loaded")?;

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
pub fn get_border_volume(
    state: State<'_, Mutex<AppState>>,
) -> Result<serde_json::Value, String> {
    let app_state = state
        .lock()
        .map_err(|e| format!("State lock error: {}", e))?;
    let atlas = app_state
        .atlas
        .as_ref()
        .ok_or("No atlas loaded")?;

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
/// Requires reference volume to be loaded (lazy-load triggers on first call).
#[tauri::command]
pub fn get_coronal_slice(
    ap_mm: f64,
    state: State<'_, Mutex<AppState>>,
) -> Result<SliceResult, String> {
    let mut app_state = state
        .lock()
        .map_err(|e| format!("State lock error: {}", e))?;
    let atlas = app_state
        .atlas
        .as_mut()
        .ok_or("No atlas loaded")?;

    // Lazy-load reference volume on first use
    if atlas.reference_volume.is_none() {
        log::info!("Lazy-loading reference.tiff...");
        let ref_vol = loader::load_tiff_u16(&atlas.atlas_dir.join("reference.tiff"))?;
        log::info!("Reference volume loaded: shape={:?}", ref_vol.shape());
        atlas.reference_volume = Some(ref_vol);
    }

    let ref_vol = atlas.reference_volume.as_ref().unwrap();
    let idx = atlas.ap_mm_to_index(ap_mm);
    slicer::extract_coronal_slice(ref_vol, &atlas.annotation_volume, idx)
}

/// Load the whole-brain outline mesh (structure 997).
#[tauri::command]
pub fn get_brain_mesh(
    state: State<'_, Mutex<AppState>>,
) -> Result<MeshData, String> {
    let app_state = state
        .lock()
        .map_err(|e| format!("State lock error: {}", e))?;
    let atlas = app_state
        .atlas
        .as_ref()
        .ok_or("No atlas loaded")?;

    let mesh_path = atlas.atlas_dir.join("meshes").join("997.obj");
    if !mesh_path.exists() {
        return Err(format!("Brain mesh not found: {}", mesh_path.display()));
    }

    loader::load_obj_mesh(&mesh_path)
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
    let entries = std::fs::read_dir(dir)
        .map_err(|e| format!("Cannot read directory: {}", e))?;

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
        a["name"].as_str().unwrap_or("").cmp(b["name"].as_str().unwrap_or(""))
    });

    Ok(images)
}
