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

export interface ImageFileInfo {
  path: string;
  name: string;
}

export async function scanImageFolder(folder: string): Promise<ImageFileInfo[]> {
  return invoke<ImageFileInfo[]>("scan_image_folder", { folder });
}
