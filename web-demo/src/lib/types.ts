// Barrel re-export so other modules can grab these from `lib/types`. The
// source of truth lives in `localEngines.ts` (probe impl owns the shape).
export type { LocalEngineStatus, LocalModel, CustomEndpoint } from "./localEngines";

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

/** Registration artifacts produced by the in-browser image-gen pipeline
 * (`imageGenRegistration.ts`). All three are GPU-ready bitmaps; nothing is
 * persisted to disk in the browser demo. */
export interface RegistrationResult {
  warpedAtlas: ImageBitmap;
  warpedBorders: ImageBitmap;
  generatedColoredRegions: ImageBitmap;
}

export interface SliceInfo {
  path: string;
  /** Optional pre-baked CLAHE variant living alongside `path`. The bundled
   *  demo brain ships these; uploaded slices don't. When the user enables
   *  the CLAHE toggle (preview or stage-specific) and this is set, viewers
   *  and API callers swap to this URL instead of `path`. */
  clahePath?: string;
  name: string;
  status: "pending" | "running" | "done";
  apMm?: number;
  /** User has committed this AP position — gates Run Registration. Set true
   * by Estimate Position on success or the explicit Lock Position button;
   * cleared whenever the slider is dragged. */
  apLocked?: boolean;
  registrationResult?: RegistrationResult;
  /** Fast affine-only preview warp produced when the user locks an AP
   * position. RGBA bitmap clipped to the atlas root silhouette, used as a
   * Three.js texture on the warped slice plane. Superseded by
   * `registrationResult.warpedAtlas` once the full pipeline completes. */
  quickAffineWarpedBitmap?: ImageBitmap;
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

/** One turn in the BYO-engine chat drawer. Images, when present, are passed
 * as base64 PNG data URLs so they round-trip cleanly to OpenAI-compat
 * `image_url` content parts without an extra fetch. */
export interface ChatMessage {
  id: string;
  role: "user" | "assistant";
  content: string;
  images?: string[]; // data URLs (data:image/png;base64,...)
}

/** Live telemetry for the most recent chat exchange. */
export interface ChatTelemetry {
  txTokens?: number;
  rxTokens?: number;
  ttftMs?: number;
  elapsedMs?: number;
  tokensPerSec?: number;
}

