import { invoke } from "@tauri-apps/api/core";
import type {
  AtlasInfo,
  AtlasMetadata,
  MeshData,
  SliceResult,
} from "./types";

export async function listAtlases(): Promise<string[]> {
  return invoke<string[]>("list_atlases");
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
  model: string;
  thinking: string;
  temperature: number;
  vlmResolution: number;
  maxIterations: number;
  workflow: string;
}): Promise<Record<string, unknown>> {
  return invoke<Record<string, unknown>>("run_estimate", params);
}

export async function runRegister(params: {
  imagePath: string;
  positionMm: number;
  atlas: string;
  model: string;
  thinking: string;
  temperature: number;
  landmarks: number;
  vlmResolution: number;
  workflow: string;
}): Promise<Record<string, unknown>> {
  return invoke<Record<string, unknown>>("run_register", params);
}

export async function runExport(params: {
  imagePath: string;
  positionMm: number;
  atlas: string;
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
