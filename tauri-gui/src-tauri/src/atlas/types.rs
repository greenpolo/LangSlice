use ndarray::Array3;
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::path::PathBuf;

/// BrainGlobe atlas metadata from metadata.json.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AtlasMetadata {
    pub name: String,
    pub citation: String,
    pub atlas_link: String,
    pub species: String,
    pub symmetric: bool,
    pub resolution: [f64; 3],
    pub orientation: String,
    pub shape: [usize; 3],
    #[serde(default)]
    pub additional_references: Vec<String>,
}

/// Single structure entry from structures.json.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AtlasStructure {
    pub id: u32,
    pub name: String,
    pub acronym: String,
    pub rgb_triplet: [u8; 3],
    #[serde(default)]
    pub structure_id_path: Vec<u32>,
    #[serde(default)]
    pub mesh_filename: Option<String>,
}

/// Vertex/index data for a 3D mesh, sent to the frontend.
#[derive(Debug, Clone, Serialize)]
pub struct MeshData {
    /// Flat [x,y,z, x,y,z, ...] positions
    pub positions: Vec<f32>,
    /// Triangle indices into the positions array
    pub indices: Vec<u32>,
    /// Flat [nx,ny,nz, ...] normals (may be empty)
    pub normals: Vec<f32>,
}

/// Coronal slice result sent to the frontend.
#[derive(Debug, Clone, Serialize)]
pub struct SliceResult {
    /// Base64-encoded PNG of the reference (grayscale)
    pub reference: String,
    /// Base64-encoded PNG of region borders
    pub borders: String,
    /// Base64-encoded PNG of composite (reference + green borders)
    pub composite: String,
    /// Width in pixels
    pub width: u32,
    /// Height in pixels
    pub height: u32,
}

/// Cached atlas state held in Tauri managed state.
pub struct AtlasState {
    pub metadata: AtlasMetadata,
    pub structures: Vec<AtlasStructure>,
    pub structure_map: HashMap<u32, AtlasStructure>,
    pub reference_volume: Array3<u16>,
    pub annotation_volume: Array3<u32>,
    /// Additional reference volumes (e.g. Nissl). Key = name, value = volume.
    pub additional_volumes: HashMap<String, Array3<u16>>,
    /// Pre-computed border volume: 1 = border pixel, 0 = background.
    /// Computed once at load time from annotation_volume.
    pub border_volume: Array3<u8>,
    pub atlas_dir: PathBuf,
}

impl AtlasState {
    /// Number of slices along the AP axis.
    pub fn ap_count(&self) -> usize {
        self.metadata.shape[0]
    }

    /// Resolution along the AP axis in mm.
    pub fn ap_resolution_mm(&self) -> f64 {
        self.metadata.resolution[0] / 1000.0
    }

    /// AP range in millimeters: (0.0, max_mm).
    pub fn ap_range_mm(&self) -> (f64, f64) {
        (0.0, (self.ap_count() - 1) as f64 * self.ap_resolution_mm())
    }

    /// Convert AP position in mm to array index (clamped).
    pub fn ap_mm_to_index(&self, ap_mm: f64) -> usize {
        let idx = (ap_mm / self.ap_resolution_mm()).round() as isize;
        idx.max(0).min(self.ap_count() as isize - 1) as usize
    }

    /// Convert array index to AP position in mm.
    pub fn index_to_ap_mm(&self, idx: usize) -> f64 {
        idx as f64 * self.ap_resolution_mm()
    }
}
