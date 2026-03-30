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

/// Load a multi-page TIFF as a 3D u16 volume (shape: [depth, height, width]).
pub fn load_tiff_u16(path: &Path) -> Result<Array3<u16>, String> {
    let file =
        File::open(path).map_err(|e| format!("Cannot open {}: {}", path.display(), e))?;
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

        if !decoder.more_images() {
            break;
        }
        decoder
            .next_image()
            .map_err(|e| format!("TIFF next page: {}", e))?;
    }

    let depth = slices.len();
    let flat: Vec<u16> = slices.into_iter().flatten().collect();
    Array3::from_shape_vec((depth, height as usize, width as usize), flat)
        .map_err(|e| format!("Array reshape error: {}", e))
}

/// Load a multi-page TIFF as a 3D u32 volume (for annotation/label volumes).
pub fn load_tiff_u32(path: &Path) -> Result<Array3<u32>, String> {
    let file =
        File::open(path).map_err(|e| format!("Cannot open {}: {}", path.display(), e))?;
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

        if !decoder.more_images() {
            break;
        }
        decoder
            .next_image()
            .map_err(|e| format!("TIFF next page: {}", e))?;
    }

    let depth = slices.len();
    let flat: Vec<u32> = slices.into_iter().flatten().collect();
    Array3::from_shape_vec((depth, height as usize, width as usize), flat)
        .map_err(|e| format!("Array reshape error: {}", e))
}

/// Load the full atlas state: metadata, structures, volumes.
pub fn load_atlas(name: &str) -> Result<AtlasState, String> {
    let atlas_dir = find_atlas_dir(name)?;

    log::info!("Loading atlas '{}' from {}", name, atlas_dir.display());

    let metadata = load_metadata(&atlas_dir)?;
    let structures = load_structures(&atlas_dir)?;
    let structure_map: HashMap<u32, AtlasStructure> =
        structures.iter().map(|s| (s.id, s.clone())).collect();

    log::info!(
        "Atlas metadata: shape={:?}, resolution={:?}um, orientation={}",
        metadata.shape,
        metadata.resolution,
        metadata.orientation
    );

    // Skip reference.tiff for now — lazy-loaded when 2D views need it
    log::info!("Skipping reference.tiff (deferred to 2D view)");

    log::info!("Loading annotation.tiff...");
    let annotation_volume = load_tiff_u32(&atlas_dir.join("annotation.tiff"))?;
    log::info!(
        "Annotation volume loaded: shape={:?}",
        annotation_volume.shape()
    );

    log::info!("Pre-computing border volume (parallel)...");
    let border_volume = super::slicer::precompute_border_volume_parallel(&annotation_volume);
    log::info!("Border volume ready: shape={:?}", border_volume.shape());

    Ok(AtlasState {
        metadata,
        structures,
        structure_map,
        reference_volume: None,
        annotation_volume,
        border_volume,
        atlas_dir,
    })
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
        let vol = load_tiff_u16(&dir.join("reference.tiff")).unwrap();
        assert_eq!(vol.shape()[0], meta.shape[0]);
        assert_eq!(vol.shape()[1], meta.shape[1]);
        assert_eq!(vol.shape()[2], meta.shape[2]);
        println!("Reference volume shape: {:?}", vol.shape());
    }

    #[test]
    fn test_load_annotation_volume() {
        let dir = find_atlas_dir("allen_mouse_25um").unwrap();
        let meta = load_metadata(&dir).unwrap();
        let vol = load_tiff_u32(&dir.join("annotation.tiff")).unwrap();
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
