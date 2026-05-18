import { invoke } from "@tauri-apps/api/core";
import type {
  AtlasInfo,
  AtlasMetadata,
  AvailableAtlas,
  LocalEngineStatus,
  MeshData,
  Plane,
  RegistrationResult,
  SliceResult,
} from "./types";

export async function listAtlases(): Promise<string[]> {
  return invoke<string[]>("list_atlases");
}

export async function listAvailableAtlases(): Promise<AvailableAtlas[]> {
  return invoke<AvailableAtlas[]>("list_available_atlases");
}

export async function downloadAtlas(name: string): Promise<void> {
  return invoke<void>("download_atlas", { name });
}

export async function probeLocalEngines(): Promise<LocalEngineStatus[]> {
  return invoke<LocalEngineStatus[]>("probe_local_engines");
}

export async function probeCustomLocalEndpoint(
  label: string,
  url: string,
): Promise<LocalEngineStatus> {
  return invoke<LocalEngineStatus>("probe_custom_local_endpoint", { label, url });
}

export async function loadAtlas(name: string): Promise<AtlasMetadata> {
  return invoke<AtlasMetadata>("load_atlas", { name });
}

export async function getAtlasInfo(): Promise<AtlasInfo> {
  return invoke<AtlasInfo>("get_atlas_info");
}

export async function getCoronalSlice(apMm: number): Promise<SliceResult> {
  return invoke<SliceResult>("get_coronal_slice", { apMm });
}

export async function getBrainMesh(): Promise<MeshData> {
  return invoke<MeshData>("get_brain_mesh");
}

export interface BorderVolumeResult {
  data: string; // base64 bit-packed volume
  depth: number;
  height: number;
  width: number;
  format: "bitpacked";
}

export async function getBorderVolume(): Promise<BorderVolumeResult> {
  return invoke<BorderVolumeResult>("get_border_volume");
}

export interface AllVolumesResult {
  reference: string; // base64 u8 volume
  additional: Record<string, string>; // name → base64 u8 volume
  additionalNames: string[];
  depth: number;
  height: number;
  width: number;
}

export async function getAllVolumes(): Promise<AllVolumesResult> {
  return invoke<AllVolumesResult>("get_all_volumes");
}

export async function runEstimate(params: {
  imagePath: string;
  atlas: string;
  plane: Plane;
  model: string;
  thinking: string;
  temperature: number;
  mediaResolution: string;
  maxIterations: number;
  preprocess: "auto" | "none";
  provider: "google" | "openai";
  endpoint?: string | null;
}): Promise<Record<string, unknown>> {
  return invoke<Record<string, unknown>>("run_estimate", params);
}

/** Helper: coerce a JSON-as-object value into a string-or-null. */
function asStringOrNull(value: unknown): string | null {
  return typeof value === "string" ? value : null;
}

export async function runRegister(params: {
  imagePath: string;
  positionMm: number;
  atlas: string;
  plane: Plane;
  imageModel: string;
  thinking: string;
  temperature: number;
  provider: "google" | "openai";
  endpoint?: string | null;
}): Promise<RegistrationResult> {
  const raw = await invoke<Record<string, unknown>>("run_register", params);
  // Map Python snake_case → TS camelCase. Required forward-warp paths
  // default to empty strings so the type is non-null (the Python pipeline
  // always emits them on success — they only go null if the CLI returns
  // an unexpected shape).
  return {
    warpedAtlasPath: asStringOrNull(raw.warped_atlas_path) ?? "",
    warpedBorderOverlayPath: asStringOrNull(raw.warped_border_overlay_path) ?? "",
    generatedSegmentationPath:
      asStringOrNull(raw.generated_segmentation_path) ?? "",
    generatedBorderOverlayPath: asStringOrNull(
      raw.generated_border_overlay_path,
    ),
    sliceWarpedToAtlasPath: asStringOrNull(raw.slice_warped_to_atlas_path),
    sliceAtlasBorderOverlayPath: asStringOrNull(
      raw.slice_atlas_border_overlay_path,
    ),
    inverseWarpStatus: asStringOrNull(raw.inverse_warp_status),
  };
}

/** Silhouette-based affine preview registration. The output path is derived
 * server-side inside Tauri's app cache dir, keyed by a hash of the input
 * parameters — never the source folder, so the histology scanner can't
 * mistake the preview for another slice. */
export async function runQuickAffine(params: {
  imagePath: string;
  positionMm: number;
  atlas: string;
  plane: Plane;
}): Promise<{ warpedSlicePath: string; elapsedS: number; silhouetteIou: number }> {
  const raw = await invoke<Record<string, unknown>>("run_quick_affine", params);
  return {
    warpedSlicePath: asStringOrNull(raw.warped_slice_path) ?? "",
    elapsedS: typeof raw.elapsed_s === "number" ? raw.elapsed_s : 0,
    silhouetteIou: typeof raw.silhouette_iou === "number" ? raw.silhouette_iou : 0,
  };
}

export async function runExport(params: {
  imagePath: string;
  positionMm: number;
  atlas: string;
  plane: Plane;
  outputDir: string;
}): Promise<string> {
  return invoke<string>("run_export", params);
}

export interface LoadedSliceImage {
  image: string; // base64 PNG
  originalWidth: number;
  originalHeight: number;
  displayWidth: number;
  displayHeight: number;
}

export async function loadSliceImage(
  path: string,
  maxEdge?: number,
): Promise<LoadedSliceImage> {
  return invoke<LoadedSliceImage>("load_slice_image", {
    path,
    maxEdge: maxEdge ?? 2048,
  });
}

export interface EnvFileResult {
  path: string;
  vars: Record<string, string>;
}

export async function readEnvFile(): Promise<EnvFileResult> {
  return invoke<EnvFileResult>("read_env_file");
}

export async function writeEnvFile(vars: Record<string, string>): Promise<void> {
  return invoke<void>("write_env_file", { vars });
}

export async function precacheImages(
  paths: string[],
  maxEdge?: number,
): Promise<number> {
  return invoke<number>("precache_images", { paths, maxEdge: maxEdge ?? 1024 });
}

export interface ImageFileInfo {
  path: string;
  name: string;
}

export async function scanImageFolder(folder: string): Promise<ImageFileInfo[]> {
  return invoke<ImageFileInfo[]>("scan_image_folder", { folder });
}

// --- Ollama management ---

export interface OllamaStatus {
  status: "ready" | "starting" | "not_installed" | "error";
  version?: string;
  port?: number;
  models_dir?: string;
  error?: string;
}

export interface OllamaModel {
  name: string;
  size: number; // bytes
  modified_at: string;
  digest: string;
}

export interface CuratedModel {
  name: string;
  display_name: string;
  description: string;
  size_gb: number;
  min_vram_gb: number;
  capabilities: string[];
  recommended: boolean;
  installed: boolean; // merged from local models
  local_size?: number; // actual size in bytes if installed
}

export async function ollamaStatus(): Promise<OllamaStatus> {
  return invoke<OllamaStatus>("ollama_status");
}

export async function ollamaStart(): Promise<void> {
  return invoke<void>("ollama_start");
}

export async function ollamaStop(): Promise<void> {
  return invoke<void>("ollama_stop");
}

export async function ollamaListModels(): Promise<{ models: OllamaModel[] }> {
  return invoke<{ models: OllamaModel[] }>("ollama_list_models");
}

export async function ollamaModelInfo(
  name: string,
): Promise<Record<string, unknown>> {
  return invoke<Record<string, unknown>>("ollama_model_info", { name });
}

export async function ollamaPullModel(name: string): Promise<void> {
  return invoke<void>("ollama_pull_model", { name });
}

export async function ollamaDeleteModel(name: string): Promise<void> {
  return invoke<void>("ollama_delete_model", { name });
}

export async function ollamaAvailableModels(): Promise<CuratedModel[]> {
  return invoke<CuratedModel[]>("ollama_available_models");
}
