use std::collections::HashMap;
use std::fs::File;
use std::io::BufReader;
use std::path::{Path, PathBuf};

use ndarray::Array3;
use tiff::decoder::{Decoder, DecodingResult};

use super::types::{AtlasMetadata, AtlasState, AtlasStructure, MeshData};

/// Find the BrainGlobe atlas root directory (~/.brainglobe/).
pub fn brainglobe_dir() -> Result<PathBuf, String> {
    let home = dirs::home_dir().ok_or("Cannot determine home directory")?;
    let bg_dir = home.join(".brainglobe");
    if bg_dir.is_dir() {
        Ok(bg_dir)
    } else {
        Err(format!(
            "BrainGlobe directory not found: {}",
            bg_dir.display()
        ))
    }
}

/// List downloaded atlas names by scanning ~/.brainglobe/.
pub fn list_downloaded_atlases() -> Result<Vec<String>, String> {
    let bg_dir = brainglobe_dir()?;
    let mut names: Vec<String> = Vec::new();

    let entries = std::fs::read_dir(&bg_dir)
        .map_err(|e| format!("Cannot read {}: {}", bg_dir.display(), e))?;

    for entry in entries.flatten() {
        let dir_name = entry.file_name().to_string_lossy().to_string();
        // Atlas directories follow the pattern: <name>_v<version>
        // e.g. "allen_mouse_25um_v1.2"
        if entry.path().is_dir() {
            if let Some(pos) = dir_name.rfind("_v") {
                let atlas_name = &dir_name[..pos];
                // Verify it has a metadata.json
                if entry.path().join("metadata.json").exists() {
                    names.push(atlas_name.to_string());
                }
            }
        }
    }
    names.sort();
    Ok(names)
}

/// Find the actual directory path for a named atlas.
/// Scans for <name>_v* directories and picks the latest version.
pub fn find_atlas_dir(name: &str) -> Result<PathBuf, String> {
    let bg_dir = brainglobe_dir()?;
    let prefix = format!("{}_v", name);

    let mut best: Option<PathBuf> = None;
    let entries = std::fs::read_dir(&bg_dir)
        .map_err(|e| format!("Cannot read {}: {}", bg_dir.display(), e))?;

    for entry in entries.flatten() {
        let dir_name = entry.file_name().to_string_lossy().to_string();
        if dir_name.starts_with(&prefix) && entry.path().is_dir() {
            // Pick the lexicographically largest version
            match &best {
                Some(prev) => {
                    let prev_name = prev
                        .file_name()
                        .unwrap_or_default()
                        .to_string_lossy()
                        .to_string();
                    if dir_name > prev_name {
                        best = Some(entry.path());
                    }
                }
                None => best = Some(entry.path()),
            }
        }
    }

    best.ok_or(format!(
        "Atlas '{}' not found in {}",
        name,
        bg_dir.display()
    ))
}

/// Load metadata.json from an atlas directory.
pub fn load_metadata(atlas_dir: &Path) -> Result<AtlasMetadata, String> {
    let path = atlas_dir.join("metadata.json");
    let file = File::open(&path).map_err(|e| format!("Cannot open {}: {}", path.display(), e))?;
    let reader = BufReader::new(file);
    serde_json::from_reader(reader).map_err(|e| format!("Cannot parse metadata.json: {}", e))
}

/// Load structures.json from an atlas directory.
pub fn load_structures(atlas_dir: &Path) -> Result<Vec<AtlasStructure>, String> {
    let path = atlas_dir.join("structures.json");
    let file = File::open(&path).map_err(|e| format!("Cannot open {}: {}", path.display(), e))?;
    let reader = BufReader::new(file);
    serde_json::from_reader(reader).map_err(|e| format!("Cannot parse structures.json: {}", e))
}

/// Stride at which TIFF loaders invoke their progress callback. Per-page
/// would flood the Tauri IPC channel (~800 pages over a few seconds);
/// every 8 pages still gives ~100 updates for a typical atlas, which is
/// well within what the webview can render smoothly.
const TIFF_PROGRESS_STRIDE: usize = 8;

/// Load a multi-page TIFF as a 3D u16 volume (shape: [depth, height, width]).
///
/// ``on_page`` is invoked roughly every ``TIFF_PROGRESS_STRIDE`` pages with
/// the count of pages read so far. Pass ``|_| {}`` if you don't care.
pub fn load_tiff_u16<F>(path: &Path, on_page: F) -> Result<Array3<u16>, String>
where
    F: Fn(usize),
{
    let file = File::open(path).map_err(|e| format!("Cannot open {}: {}", path.display(), e))?;
    let mut decoder =
        Decoder::new(BufReader::new(file)).map_err(|e| format!("TIFF decode error: {}", e))?;

    let (width, height) = decoder
        .dimensions()
        .map_err(|e| format!("TIFF dimensions: {}", e))?;

    let mut slices: Vec<Vec<u16>> = Vec::new();
    loop {
        let data = decoder
            .read_image()
            .map_err(|e| format!("TIFF read page {}: {}", slices.len(), e))?;
        let slice_data = match data {
            DecodingResult::U16(d) => d,
            DecodingResult::U8(d) => d.into_iter().map(|v| v as u16).collect(),
            _ => return Err("Unsupported TIFF data type for reference volume".into()),
        };
        slices.push(slice_data);

        if slices.len() % TIFF_PROGRESS_STRIDE == 0 {
            on_page(slices.len());
        }

        if !decoder.more_images() {
            break;
        }
        decoder
            .next_image()
            .map_err(|e| format!("TIFF next page: {}", e))?;
    }
    // Final emit so the bar fills the phase even if total % stride != 0.
    on_page(slices.len());

    let depth = slices.len();
    let flat: Vec<u16> = slices.into_iter().flatten().collect();
    Array3::from_shape_vec((depth, height as usize, width as usize), flat)
        .map_err(|e| format!("Array reshape error: {}", e))
}

/// Load a multi-page TIFF as a 3D u32 volume (for annotation/label volumes).
pub fn load_tiff_u32<F>(path: &Path, on_page: F) -> Result<Array3<u32>, String>
where
    F: Fn(usize),
{
    let file = File::open(path).map_err(|e| format!("Cannot open {}: {}", path.display(), e))?;
    let mut decoder =
        Decoder::new(BufReader::new(file)).map_err(|e| format!("TIFF decode error: {}", e))?;

    let (width, height) = decoder
        .dimensions()
        .map_err(|e| format!("TIFF dimensions: {}", e))?;

    let mut slices: Vec<Vec<u32>> = Vec::new();
    loop {
        let data = decoder
            .read_image()
            .map_err(|e| format!("TIFF read page {}: {}", slices.len(), e))?;
        let slice_data = match data {
            DecodingResult::U32(d) => d,
            DecodingResult::U16(d) => d.into_iter().map(|v| v as u32).collect(),
            DecodingResult::U8(d) => d.into_iter().map(|v| v as u32).collect(),
            _ => return Err("Unsupported TIFF data type for annotation volume".into()),
        };
        slices.push(slice_data);

        if slices.len() % TIFF_PROGRESS_STRIDE == 0 {
            on_page(slices.len());
        }

        if !decoder.more_images() {
            break;
        }
        decoder
            .next_image()
            .map_err(|e| format!("TIFF next page: {}", e))?;
    }
    on_page(slices.len());

    let depth = slices.len();
    let flat: Vec<u32> = slices.into_iter().flatten().collect();
    Array3::from_shape_vec((depth, height as usize, width as usize), flat)
        .map_err(|e| format!("Array reshape error: {}", e))
}

/// Load the full atlas state: metadata, structures, volumes.
///
/// ``on_progress`` is invoked at phase boundaries with a human-readable phase
/// label and a cumulative percent (0..=100). The percent is reported at the
/// *start* of each phase so the UI bar moves ahead of the slow op, giving the
/// user immediate visual feedback. Pass ``|_, _| {}`` if you don't care.
pub fn load_atlas<F>(name: &str, on_progress: F) -> Result<AtlasState, String>
where
    F: Fn(&str, u8),
{
    on_progress("Locating atlas...", 0);
    let atlas_dir = find_atlas_dir(name)?;

    log::info!("Loading atlas '{}' from {}", name, atlas_dir.display());

    on_progress("Loading metadata...", 3);
    let mut metadata = load_metadata(&atlas_dir)?;
    let structures = load_structures(&atlas_dir)?;
    let structure_map: HashMap<u32, AtlasStructure> =
        structures.iter().map(|s| (s.id, s.clone())).collect();

    log::info!(
        "Atlas metadata: shape={:?}, resolution={:?}um, orientation={}",
        metadata.shape,
        metadata.resolution,
        metadata.orientation
    );

    // Map per-page TIFF progress into the phase's percent range so the bar
    // creeps during the multi-second read rather than freezing at the phase
    // start. expected_pages = metadata.shape[0] (AP count); the cast to f32
    // is for the percent math.
    let expected_pages = metadata.shape[0].max(1) as f32;

    on_progress("Loading reference volume...", 10);
    let reference_volume = load_tiff_u16(&atlas_dir.join("reference.tiff"), |page| {
        let frac = (page as f32 / expected_pages).min(1.0);
        let pct = (10.0 + 35.0 * frac).round() as u8;
        on_progress("Loading reference volume...", pct);
    })?;
    log::info!(
        "Reference volume loaded: shape={:?}",
        reference_volume.shape()
    );

    on_progress("Loading annotation volume...", 45);
    let annotation_volume = load_tiff_u32(&atlas_dir.join("annotation.tiff"), |page| {
        let frac = (page as f32 / expected_pages).min(1.0);
        let pct = (45.0 + 33.0 * frac).round() as u8;
        on_progress("Loading annotation volume...", pct);
    })?;
    log::info!(
        "Annotation volume loaded: shape={:?}",
        annotation_volume.shape()
    );

    on_progress("Computing region borders...", 78);
    let border_volume = super::slicer::precompute_border_volume_parallel(&annotation_volume);
    log::info!("Border volume ready: shape={:?}", border_volume.shape());

    on_progress("Loading additional volumes...", 90);
    // Load additional reference volumes (e.g. Nissl)
    // BrainGlobe stores these either in additional_references/ subdir
    // or at the atlas root level as <name>.tiff
    let mut additional_volumes: HashMap<String, Array3<u16>> = HashMap::new();
    for name in &metadata.additional_references {
        let candidates = [
            atlas_dir
                .join("additional_references")
                .join(format!("{}.tiff", name)),
            atlas_dir.join(format!("{}.tiff", name)),
        ];
        let found = candidates.iter().find(|p| p.exists());
        if let Some(tiff_path) = found {
            log::info!(
                "Loading additional reference: {} from {}",
                name,
                tiff_path.display()
            );
            match load_tiff_u16(tiff_path, |_| {}) {
                Ok(vol) => {
                    log::info!("  {} loaded: shape={:?}", name, vol.shape());
                    additional_volumes.insert(name.clone(), vol);
                }
                Err(e) => {
                    log::warn!("  Failed to load {}: {}", name, e);
                }
            }
        } else {
            log::warn!("  Additional reference '{}' not found on disk", name);
        }
    }
    if !additional_volumes.is_empty() {
        log::info!(
            "Loaded {} additional reference(s)",
            additional_volumes.len()
        );
    }

    // For Allen CCFv3 atlases without Nissl: try to borrow it from
    // ccfv3augmented_mouse_25um if that atlas is downloaded.
    if additional_volumes.is_empty() && is_allen_ccfv3_compatible(name) {
        if let Some(nissl) = try_load_augmented_nissl(name, &reference_volume) {
            log::info!("Injected Nissl from ccfv3augmented atlas");
            additional_volumes.insert("nissl".to_string(), nissl);
            // Update metadata so frontend knows about it
            metadata.additional_references.push("nissl".to_string());
        }
    }

    on_progress("Finalizing...", 99);
    Ok(AtlasState {
        metadata,
        structures,
        structure_map,
        reference_volume,
        annotation_volume,
        additional_volumes,
        border_volume,
        atlas_dir,
    })
}

/// Allen CCFv3-compatible atlas names that can borrow Nissl from the augmented atlas.
const ALLEN_CCFV3_COMPATIBLE: &[&str] = &[
    "allen_mouse_10um",
    "allen_mouse_25um",
    "allen_mouse_50um",
    "allen_mouse_100um",
    "kim_mouse_10um",
    "kim_mouse_25um",
    "kim_mouse_50um",
    "kim_mouse_100um",
    "osten_mouse_10um",
    "osten_mouse_25um",
    "osten_mouse_50um",
    "osten_mouse_100um",
    "perens_lsfm_mouse_20um",
    "princeton_mouse_20um",
];

fn is_allen_ccfv3_compatible(name: &str) -> bool {
    ALLEN_CCFV3_COMPATIBLE.contains(&name)
}

/// Try to load the Nissl volume from ccfv3augmented_mouse_25um and
/// crop/resample it to match the target atlas.
fn try_load_augmented_nissl(
    target_name: &str,
    target_reference: &Array3<u16>,
) -> Option<Array3<u16>> {
    // Find the augmented atlas directory (try 25um first, then 10um)
    let aug_dir = find_atlas_dir("ccfv3augmented_mouse_25um").ok()?;

    // Load the Nissl TIFF from the augmented atlas
    let nissl_candidates = [
        aug_dir
            .join("additional_references")
            .join("single_animal_nissl.tiff"),
        aug_dir.join("single_animal_nissl.tiff"),
    ];
    let nissl_path = nissl_candidates.iter().find(|p| p.exists())?;

    log::info!(
        "Found augmented Nissl for '{}' at {}",
        target_name,
        nissl_path.display()
    );

    let nissl_vol = load_tiff_u16(nissl_path, |_| {}).ok()?;
    let aug_depth = nissl_vol.shape()[0];
    let target_depth = target_reference.shape()[0];
    let target_height = target_reference.shape()[1];
    let target_width = target_reference.shape()[2];

    // Check DV/ML dimensions match
    if nissl_vol.shape()[1] != target_height || nissl_vol.shape()[2] != target_width {
        log::warn!(
            "Nissl DV/ML dimensions don't match target atlas: {:?} vs ({}, {})",
            &nissl_vol.shape()[1..],
            target_height,
            target_width
        );
        // Would need resampling — skip for now
        return None;
    }

    if aug_depth == target_depth {
        // Same depth — direct copy
        return Some(nissl_vol);
    }

    if aug_depth > target_depth {
        // Augmented is larger — find the best AP offset by correlating annotation.
        // Use a simple heuristic: offset 14 is known for 25um CCFv3 augmented vs standard.
        // For other resolutions, try a small search.
        let offset = if target_depth == 528 && aug_depth == 566 {
            14 // Known offset for 25um
        } else {
            // Rough estimate: center the smaller volume in the larger one
            (aug_depth - target_depth) / 2
        };

        log::info!(
            "Cropping augmented Nissl: offset={}, {} -> {} slices",
            offset,
            aug_depth,
            target_depth
        );

        let cropped = nissl_vol
            .slice(ndarray::s![offset..offset + target_depth, .., ..])
            .to_owned();
        return Some(cropped);
    }

    // Augmented is smaller than target — can't crop
    log::warn!(
        "Augmented Nissl is smaller than target: {} < {}",
        aug_depth,
        target_depth
    );
    None
}

/// Load an OBJ mesh file and return vertex/index data.
pub fn load_obj_mesh(path: &Path) -> Result<MeshData, String> {
    let (models, _materials) = tobj::load_obj(
        path,
        &tobj::LoadOptions {
            triangulate: true,
            single_index: true,
            ..Default::default()
        },
    )
    .map_err(|e| format!("OBJ load error: {}", e))?;

    if models.is_empty() {
        return Err("OBJ file contains no models".into());
    }

    let mesh = &models[0].mesh;
    Ok(MeshData {
        positions: mesh.positions.clone(),
        indices: mesh.indices.clone(),
        normals: mesh.normals.clone(),
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_brainglobe_dir_exists() {
        // This test requires ~/.brainglobe/ to exist
        let dir = brainglobe_dir();
        assert!(dir.is_ok(), "Expected ~/.brainglobe/ to exist");
    }

    #[test]
    fn test_list_atlases() {
        let atlases = list_downloaded_atlases();
        assert!(atlases.is_ok());
        let names = atlases.unwrap();
        println!("Downloaded atlases: {:?}", names);
        assert!(!names.is_empty(), "Expected at least one downloaded atlas");
    }

    #[test]
    fn test_find_atlas_dir() {
        let dir = find_atlas_dir("allen_mouse_25um");
        assert!(dir.is_ok(), "Expected allen_mouse_25um to be downloaded");
        let path = dir.unwrap();
        assert!(path.join("metadata.json").exists());
        assert!(path.join("reference.tiff").exists());
        assert!(path.join("annotation.tiff").exists());
    }

    #[test]
    fn test_load_metadata() {
        let dir = find_atlas_dir("allen_mouse_25um").unwrap();
        let meta = load_metadata(&dir).unwrap();
        assert!(!meta.name.is_empty());
        assert_eq!(meta.shape.len(), 3);
        assert!(meta.resolution[0] > 0.0);
        println!(
            "Atlas: {} shape={:?} res={:?}",
            meta.name, meta.shape, meta.resolution
        );
    }

    #[test]
    fn test_load_structures() {
        let dir = find_atlas_dir("allen_mouse_25um").unwrap();
        let structs = load_structures(&dir).unwrap();
        assert!(!structs.is_empty());
        // Structure 997 should be "root" (whole brain)
        let root = structs.iter().find(|s| s.id == 997);
        assert!(root.is_some(), "Expected structure 997 (root)");
        println!(
            "Loaded {} structures, root: {:?}",
            structs.len(),
            root.unwrap().name
        );
    }

    #[test]
    fn test_load_reference_volume() {
        let dir = find_atlas_dir("allen_mouse_25um").unwrap();
        let meta = load_metadata(&dir).unwrap();
        let vol = load_tiff_u16(&dir.join("reference.tiff"), |_| {}).unwrap();
        assert_eq!(vol.shape()[0], meta.shape[0]);
        assert_eq!(vol.shape()[1], meta.shape[1]);
        assert_eq!(vol.shape()[2], meta.shape[2]);
        println!("Reference volume shape: {:?}", vol.shape());
    }

    #[test]
    fn test_load_annotation_volume() {
        let dir = find_atlas_dir("allen_mouse_25um").unwrap();
        let meta = load_metadata(&dir).unwrap();
        let vol = load_tiff_u32(&dir.join("annotation.tiff"), |_| {}).unwrap();
        assert_eq!(vol.shape()[0], meta.shape[0]);
        println!("Annotation volume shape: {:?}", vol.shape());
    }

    #[test]
    fn test_load_brain_mesh() {
        let dir = find_atlas_dir("allen_mouse_25um").unwrap();
        let mesh_path = dir.join("meshes").join("997.obj");
        if mesh_path.exists() {
            let mesh = load_obj_mesh(&mesh_path).unwrap();
            assert!(!mesh.positions.is_empty());
            assert!(!mesh.indices.is_empty());
            println!(
                "Brain mesh: {} vertices, {} triangles",
                mesh.positions.len() / 3,
                mesh.indices.len() / 3
            );
        } else {
            println!("Skipping mesh test: 997.obj not found");
        }
    }
}
