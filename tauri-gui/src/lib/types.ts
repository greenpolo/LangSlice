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

export type ViewMode = "3d" | "split";

export type NavigationView = "dashboard" | "brain-detail";

export type BrainStatus = "queued" | "running" | "complete";

/** Slicing plane (normal axis). Matches the Python CLI --plane choices. */
export type Plane = "coronal" | "sagittal" | "horizontal";

/** Registration artifacts emitted by `langslice_harness register --json`.
 * Fields are camelCase here; the Python JSON uses snake_case and is mapped
 * in `commands.runRegister`. */
export interface RegistrationResult {
  warpedAtlasPath: string;
  warpedBorderOverlayPath: string;
  generatedSegmentationPath: string;
  /** Borders extracted directly from the model's image-gen output (no Elastix
   * warp), overlaid on the slice. Lets the GUI display the model's raw
   * prediction independent of the deformation step. */
  generatedBorderOverlayPath: string | null;
  sliceWarpedToAtlasPath: string | null;
  sliceAtlasBorderOverlayPath: string | null;
  inverseWarpStatus: "ok" | string | null; // "ok" or "failed: ..." or null
}

export interface SliceInfo {
  path: string;
  name: string;
  status: "pending" | "running" | "done";
  apMm?: number;
  /** User has committed this AP position — gates Run Registration. Set true
   * by Estimate Position on success or the explicit Lock Position button;
   * cleared whenever the slider is dragged. */
  apLocked?: boolean;
  registrationResult?: RegistrationResult;
  /** Fast affine-only preview warp produced when the user locks an AP
   * position. Drops a brain-silhouette-masked PNG into the 3D viewer
   * before the slow image-gen pipeline runs. Superseded by
   * `registrationResult.sliceWarpedToAtlasPath` when the full pipeline
   * completes. */
  quickAffineWarpedPath?: string;
  /** True while the quick-affine subprocess is in flight. Drives the
   * "Aligning preview..." pill so users know what's happening during the
   * ~15-20s Python/itk cold-start. */
  quickAffineRunning?: boolean;
  /** Most recent error from a quick-affine run, if any. Doesn't block UI
   * (preview is best-effort) — surfaced as a small dim line in the
   * settings panel for debugging. */
  quickAffineError?: string | null;
  /** Whether this slice's warped plane is shown in the 3D viewer. Only
   * meaningful once `apLocked` is true. `undefined` is treated as `true`
   * so newly-locked slices auto-appear in the volume; the slice list's
   * per-row eye toggle flips this. */
  visibleIn3D?: boolean;
}

export interface Brain {
  id: string;
  name: string;
  folderPath: string;
  slices: SliceInfo[];
  status: BrainStatus;
  completedCount: number;
  plane: Plane;
}

/** Atlas-load progress event payload emitted by the Rust loader. Percent is
 * cumulative (0..=100) and reported at the *start* of each phase so the bar
 * moves ahead of the slow op. */
export interface AtlasLoadProgress {
  phase: string;
  percent: number;
}

export type PipelineStatus =
  | "idle"
  | "loading_atlas"
  | "estimating"
  | "registering"
  | "complete"
  | "error";

/** One row in the atlas-manager modal — Rust returns these from
 * `list_available_atlases`. `downloaded` is true when the atlas is already
 * present in `~/.brainglobe/`. */
export interface AvailableAtlas {
  name: string;
  version: string;
  latest_version: string;
  downloaded: boolean;
  updated: boolean;
}

/** Progress payload emitted on `atlas-download-progress`. `phase` is one of
 * `resolving | downloading | extracting | complete`; `completed`/`total` are
 * byte counts (present during `downloading` and at `complete`). */
export interface AtlasDownloadProgress {
  atlas: string;
  phase: "resolving" | "downloading" | "extracting" | "complete";
  completed?: number;
  total?: number;
}

/** One model discovered via `/v1/models` on a local engine. The `endpoint`
 * is captured so the Estimate dropdown can route requests back here. */
export interface LocalModel {
  id: string;
  engine: string;
  endpoint: string;
}

/** One row in the Local Models modal. `reachable: false` rows are still
 * rendered with a "Not running" pill so the user knows we looked. */
export interface LocalEngineStatus {
  name: string;
  port: number;
  endpoint: string;
  reachable: boolean;
  models: LocalModel[];
  error: string | null;
}

/** User-saved extra endpoint (non-default port or remote server). */
export interface CustomEndpoint {
  label: string;
  url: string;
  apiKey?: string;
}

/** Selected model for the Estimate dropdown — captures the endpoint so the
 * pipeline can call back to the right server. */
export interface EstimateModelChoice {
  value: string;          // unique id used by the <select>
  modelId: string;        // bare model name passed to the harness
  label: string;          // shown to the user (e.g. "gemma4:e4b · Ollama")
  endpoint: string | null; // null for cloud models (Gemini)
  source: "cloud" | "local";
}

/** One turn in the BYO-engine chat drawer. Images, when present, are passed as
 * base64 PNG data URLs so they round-trip cleanly to OpenAI-compat
 * `image_url` content parts without an extra fetch. */
export interface ChatMessage {
  id: string;
  role: "user" | "assistant";
  content: string;
  images?: string[]; // data URLs (data:image/png;base64,...)
}

/** Live telemetry for the most recent chat exchange. Rendered as a thin
 * footer strip; fields are populated incrementally as the stream progresses
 * so the user gets immediate ttft + running tok/s feedback. */
export interface ChatTelemetry {
  txTokens?: number;       // input token count (estimated client-side)
  rxTokens?: number;       // output token count (counted from stream chunks)
  ttftMs?: number;         // time-to-first-token, ms
  elapsedMs?: number;      // wall time of the most recent generation
  tokensPerSec?: number;   // running rx tokens / elapsed seconds
}
