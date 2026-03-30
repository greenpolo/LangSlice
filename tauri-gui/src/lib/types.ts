/** Matches Rust AtlasMetadata */
export interface AtlasMetadata {
  name: string;
  citation: string;
  atlas_link: string;
  species: string;
  symmetric: boolean;
  resolution: [number, number, number];
  orientation: string;
  shape: [number, number, number];
  additional_references: string[];
}

/** Matches Rust atlas info response */
export interface AtlasInfo {
  name: string;
  shape: [number, number, number];
  resolution: [number, number, number];
  orientation: string;
  ap_min_mm: number;
  ap_max_mm: number;
  ap_count: number;
}

/** Matches Rust SliceResult */
export interface SliceResult {
  reference: string; // base64 PNG
  borders: string; // base64 PNG
  composite: string; // base64 PNG
  width: number;
  height: number;
}

/** Matches Rust MeshData */
export interface MeshData {
  positions: number[]; // flat [x,y,z, ...]
  indices: number[]; // triangle indices
  normals: number[]; // flat [nx,ny,nz, ...]
}

export type ViewMode = "3d" | "split" | "overlay";

export type NavigationView = "dashboard" | "brain-detail";

export type BrainStatus = "queued" | "running" | "complete";

export interface SliceInfo {
  path: string;
  name: string;
  status: "pending" | "running" | "done";
  apMm?: number;
}

export interface Brain {
  id: string;
  name: string;
  folderPath: string;
  slices: SliceInfo[];
  status: BrainStatus;
  completedCount: number;
}

export type PipelineStatus =
  | "idle"
  | "loading_atlas"
  | "estimating"
  | "registering"
  | "complete"
  | "error";
