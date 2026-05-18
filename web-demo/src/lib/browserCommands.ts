/**
 * Browser-native replacement for `lib/commands.ts`.
 *
 * Routes every former Tauri `invoke()` to a static manifest fetch, a
 * canvas-based image op, or `localStorage`. Drops the desktop-only surface
 * (ollama_*, probeLocalEngines, downloadAtlas, run_register/quick_affine/
 * export — those are owned by `agentLoop.ts`, `quickAffine.ts`,
 * `imageGenRegistration.ts` and called directly from the store).
 */

import type {
  AtlasInfo,
  AtlasMetadata,
  MeshData,
  Plane,
} from "./types";

const ATLAS_NAME = "allen_mouse_25um";
const ATLAS_ROOT = `${import.meta.env.BASE_URL}atlas/${ATLAS_NAME}`;
const DEMO_BRAIN_ROOT = `${import.meta.env.BASE_URL}demo_brain`;

// --- Atlas bundle types (mirrors what build_atlas_bundle.py emits) ---

interface AtlasSection {
  idx: number;
  mm: number;
  section: string;   // e.g. "sections/000.png"
  colored: string;   // e.g. "colored/000.png"
  borders: string;   // e.g. "borders/000.png"
}

export interface AtlasBundleManifest {
  atlas: string;
  plane: Plane;
  orientation: string;
  shape: [number, number, number];
  resolution_um: [number, number, number];
  ap_axis_index: number;
  ap_resolution_mm: number;
  ap_min_mm: number;
  ap_max_mm: number;
  n_sections: number;
  root_structure_id: number;
  mesh_file: string;
  structures_file: string;
  sections: AtlasSection[];
}

interface DemoBrainSection {
  filename: string;
  /** Optional pre-baked CLAHE variant (LAB-L equalized) sitting next to
   *  `filename`. Present on the bundled demo brain; absent on uploads. */
  clahe_filename?: string;
  ground_truth_mm: number;
  original_index: number;
}

export interface DemoBrainManifest {
  brain_id: string;
  atlas: string;
  plane: Plane;
  ap_convention: string;
  preprocessing: Record<string, unknown>;
  sections: DemoBrainSection[];
  skipped: string[];
}

// --- Module-level manifest cache (load once per session) ---

let atlasManifest: AtlasBundleManifest | null = null;
let demoBrainManifest: DemoBrainManifest | null = null;

async function fetchJson<T>(url: string): Promise<T> {
  // Don't force-cache: GH Pages serves the JSON manifests without an explicit
  // Cache-Control, and a redeploy that rebakes the bundle (drops slices, adds
  // CLAHE filenames, etc.) needs to invalidate cached copies. `no-cache` still
  // round-trips a conditional GET so we get a cheap 304 when nothing changed.
  const r = await fetch(url, { cache: "no-cache" });
  if (!r.ok) throw new Error(`Fetch ${url} failed: HTTP ${r.status}`);
  return (await r.json()) as T;
}

export async function getAtlasManifest(): Promise<AtlasBundleManifest> {
  if (!atlasManifest) {
    atlasManifest = await fetchJson<AtlasBundleManifest>(
      `${ATLAS_ROOT}/manifest.json`,
    );
  }
  return atlasManifest;
}

export async function getDemoBrainManifest(): Promise<DemoBrainManifest> {
  if (!demoBrainManifest) {
    demoBrainManifest = await fetchJson<DemoBrainManifest>(
      `${DEMO_BRAIN_ROOT}/manifest.json`,
    );
  }
  return demoBrainManifest;
}

// --- Atlas catalog: just the one bundled atlas ---

export async function listAtlases(): Promise<string[]> {
  return [ATLAS_NAME];
}

export async function loadAtlas(name: string): Promise<AtlasMetadata> {
  if (name !== ATLAS_NAME) {
    throw new Error(
      `Demo bundle ships only ${ATLAS_NAME}; got "${name}"`,
    );
  }
  const m = await getAtlasManifest();
  return {
    name: ATLAS_NAME,
    citation: "Wang et al 2020 (Allen CCFv3)",
    atlas_link: "https://atlas.brain-map.org/",
    species: "Mus musculus",
    symmetric: true,
    resolution: m.resolution_um,
    orientation: m.orientation,
    shape: m.shape,
    additional_references: [],
  };
}

export async function getAtlasInfo(): Promise<AtlasInfo> {
  const m = await getAtlasManifest();
  return {
    name: ATLAS_NAME,
    shape: m.shape,
    resolution: m.resolution_um,
    orientation: m.orientation,
    ap_min_mm: m.ap_min_mm,
    ap_max_mm: m.ap_max_mm,
    ap_count: m.n_sections,
  };
}

// --- Coronal slice fetching (one PNG per layer, no base64 round-trip) ---

/** Find the bundled section whose AP mm is closest to `apMm` via binary search.
 *  Manifest sections are sorted ascending by mm at bake time. */
export async function nearestSection(apMm: number): Promise<AtlasSection> {
  const m = await getAtlasManifest();
  const arr = m.sections;
  let lo = 0;
  let hi = arr.length - 1;
  while (lo < hi) {
    const mid = (lo + hi) >> 1;
    if (arr[mid].mm < apMm) lo = mid + 1;
    else hi = mid;
  }
  if (lo > 0 && Math.abs(arr[lo - 1].mm - apMm) < Math.abs(arr[lo].mm - apMm)) {
    return arr[lo - 1];
  }
  return arr[lo];
}

/** Per-layer URLs for a coronal slice at `apMm`. Components fetch these
 *  directly with <img> or `fetch` + ImageBitmap. No base64. */
export interface CoronalSliceLayers {
  sectionUrl: string;
  coloredUrl: string;
  bordersUrl: string;
  apMm: number;
  idx: number;
}

export async function getCoronalSliceLayers(apMm: number): Promise<CoronalSliceLayers> {
  const sec = await nearestSection(apMm);
  return {
    sectionUrl: `${ATLAS_ROOT}/${sec.section}`,
    coloredUrl: `${ATLAS_ROOT}/${sec.colored}`,
    bordersUrl: `${ATLAS_ROOT}/${sec.borders}`,
    apMm: sec.mm,
    idx: sec.idx,
  };
}

/** Fire-and-forget HTTP-cache warm-up for every section's borders PNG.
 *  Called once on atlas load so dragging the AP slider in 3D doesn't pay
 *  the network round-trip per tick. Best-effort: failures are swallowed
 *  and will surface on the actual render fetch. Limits parallelism to
 *  CONCURRENCY to keep the network stack from getting blown out. */
const PREFETCH_CONCURRENCY = 16;

async function runWithConcurrency<T>(
  items: T[],
  limit: number,
  fn: (item: T) => Promise<void>,
): Promise<void> {
  let cursor = 0;
  const workers = Array.from(
    { length: Math.min(limit, items.length) },
    async () => {
      while (cursor < items.length) {
        const i = cursor++;
        await fn(items[i]);
      }
    },
  );
  await Promise.all(workers);
}

export async function prefetchAllBorders(): Promise<void> {
  const m = await getAtlasManifest();
  const urls = m.sections.map((s) => `${ATLAS_ROOT}/${s.borders}`);
  await runWithConcurrency(urls, PREFETCH_CONCURRENCY, async (url) => {
    try {
      await fetch(url, {
        // priority hint is a recent addition; older TS lib types may not
        // know about it. Cast to satisfy older type defs without losing
        // the hint at runtime.
        priority: "low",
        cache: "force-cache",
      } as RequestInit & { priority?: "low" });
    } catch {
      // Best-effort warm-up.
    }
  });
}

// --- Brain mesh (OBJ → MeshData) ---

/** Minimal OBJ parser: reads `v x y z` and `f a b c` (1-indexed). Ignores
 *  normals, texture coords, and groups. Triangulates fan-style for n-gons. */
function parseObjToMesh(text: string): MeshData {
  const positions: number[] = [];
  const indices: number[] = [];
  const lines = text.split("\n");
  for (const raw of lines) {
    const line = raw.trim();
    if (!line || line.startsWith("#")) continue;
    const parts = line.split(/\s+/);
    if (parts[0] === "v" && parts.length >= 4) {
      positions.push(+parts[1], +parts[2], +parts[3]);
    } else if (parts[0] === "f" && parts.length >= 4) {
      // Each token may be "i", "i/t", "i/t/n", or "i//n". Take the first int.
      const v = parts.slice(1).map((tok) => parseInt(tok.split("/")[0], 10) - 1);
      // Fan-triangulate any n-gon (n >= 3).
      for (let i = 1; i < v.length - 1; i++) {
        indices.push(v[0], v[i], v[i + 1]);
      }
    }
  }
  return { positions, indices, normals: [] };
}

export async function getBrainMesh(): Promise<MeshData> {
  const m = await getAtlasManifest();
  const r = await fetch(`${ATLAS_ROOT}/${m.mesh_file}`, { cache: "force-cache" });
  if (!r.ok) throw new Error(`Fetch mesh failed: HTTP ${r.status}`);
  return parseObjToMesh(await r.text());
}

// --- Demo brain image loading ---

export interface LoadedSliceImage {
  /** Base64 PNG (no data URL prefix), mirrors the Tauri shape so existing
   *  components that expect this can be left untouched. */
  image: string;
  originalWidth: number;
  originalHeight: number;
  displayWidth: number;
  displayHeight: number;
}

/** Fetch a demo brain slice, downsample to `maxEdge`, return base64 PNG. */
export async function loadSliceImage(
  path: string,
  maxEdge?: number,
): Promise<LoadedSliceImage> {
  const target = maxEdge ?? 2048;
  // Demo-brain paths in the manifest are bare filenames; if the caller
  // passes an absolute path or URL we honor it as-is.
  const url = path.includes("/") || path.includes("\\")
    ? path
    : `${DEMO_BRAIN_ROOT}/${path}`;

  const bmp = await createImageBitmap(await (await fetch(url)).blob());
  const origW = bmp.width;
  const origH = bmp.height;
  const longEdge = Math.max(origW, origH);
  const scale = longEdge > target ? target / longEdge : 1;
  const w = Math.round(origW * scale);
  const h = Math.round(origH * scale);

  const canvas = document.createElement("canvas");
  canvas.width = w;
  canvas.height = h;
  const ctx = canvas.getContext("2d")!;
  ctx.imageSmoothingEnabled = true;
  ctx.imageSmoothingQuality = "high";
  ctx.drawImage(bmp, 0, 0, w, h);
  bmp.close();

  const blob = await new Promise<Blob>((res, rej) => {
    canvas.toBlob((b) => (b ? res(b) : rej(new Error("toBlob failed"))), "image/png");
  });
  const b64 = await blobToBase64(blob);
  return {
    image: b64,
    originalWidth: origW,
    originalHeight: origH,
    displayWidth: w,
    displayHeight: h,
  };
}

async function blobToBase64(blob: Blob): Promise<string> {
  const buf = await blob.arrayBuffer();
  const bytes = new Uint8Array(buf);
  let bin = "";
  for (let i = 0; i < bytes.length; i++) bin += String.fromCharCode(bytes[i]);
  return btoa(bin);
}

// --- Scan "image folder" → enumerate baked demo-brain manifest ---

export interface ImageFileInfo {
  path: string;
  name: string;
}

export async function scanImageFolder(folder: string): Promise<ImageFileInfo[]> {
  // In the browser there's no real folder; ignore `folder` and return the
  // single bundled demo brain. The Load Demo path is the only caller.
  void folder;
  const m = await getDemoBrainManifest();
  return m.sections.map((s) => ({ path: s.filename, name: s.filename }));
}

export async function precacheImages(
  paths: string[],
  _maxEdge?: number,
): Promise<number> {
  // Browser cache + the Vite-served static assets handle warm-up implicitly.
  // Kick a HEAD request per path so anything not yet pulled lands in the
  // disk cache before the user clicks it.
  await Promise.all(
    paths.map((p) =>
      fetch(
        p.includes("/") || p.includes("\\") ? p : `${DEMO_BRAIN_ROOT}/${p}`,
        { method: "GET", cache: "force-cache" },
      ).catch(() => null),
    ),
  );
  return paths.length;
}

// --- "Env file" → localStorage ---

const ENV_STORAGE_KEY = "langslice.env";

export interface EnvFileResult {
  path: string;
  vars: Record<string, string>;
}

export async function readEnvFile(): Promise<EnvFileResult> {
  const raw = localStorage.getItem(ENV_STORAGE_KEY);
  const vars = raw ? (JSON.parse(raw) as Record<string, string>) : {};
  return { path: `localStorage:${ENV_STORAGE_KEY}`, vars };
}

export async function writeEnvFile(vars: Record<string, string>): Promise<void> {
  localStorage.setItem(ENV_STORAGE_KEY, JSON.stringify(vars));
}

// --- Surface compatibility shims (throw on use, keep store typecheck happy) ---

/** Unsupported in the browser demo — wired through `agentLoop.ts` directly. */
export async function runEstimate(_p: unknown): Promise<Record<string, unknown>> {
  throw new Error(
    "runEstimate is not exposed via browserCommands; call agentLoop.runEstimate directly.",
  );
}

/** Unsupported in the browser demo — wired through `imageGenRegistration.ts` directly. */
export async function runRegister(_p: unknown): Promise<never> {
  throw new Error(
    "runRegister is not exposed via browserCommands; call imageGenRegistration directly.",
  );
}

/** Unsupported in the browser demo — wired through `quickAffine.ts` directly. */
export async function runQuickAffine(_p: unknown): Promise<never> {
  throw new Error(
    "runQuickAffine is not exposed via browserCommands; call quickAffine directly.",
  );
}

export async function runExport(_p: unknown): Promise<never> {
  throw new Error("runExport is out of scope for the GH Pages demo.");
}

// Stubs for the desktop-only surface the store still references at typecheck
// time. They no-op so the dashboard can mount even before the store is
// rewired in Phase 2a.

export async function listAvailableAtlases(): Promise<never[]> {
  return [];
}

export async function downloadAtlas(_name: string): Promise<void> {
  // No-op: the demo bundle is the only atlas.
}

export async function probeLocalEngines(): Promise<never[]> {
  return [];
}

export async function probeCustomLocalEndpoint(
  _label: string,
  _url: string,
): Promise<never> {
  throw new Error("Local-engine probing is not part of the GH Pages demo.");
}

// Volume-shaped getters: the Tauri build streams bit-packed border + reference
// volumes for the 3D viewer. The browser bundle ships per-section PNGs
// instead, so these throw — the store will skip them after Phase 2a rewire.

export async function getBorderVolume(): Promise<never> {
  throw new Error(
    "getBorderVolume is unsupported in the browser demo; the store should " +
    "fetch per-section borders/NNN.png on AP change instead.",
  );
}

export async function getAllVolumes(): Promise<never> {
  throw new Error(
    "getAllVolumes is unsupported in the browser demo; reference is per-section.",
  );
}

// Ollama: not part of the demo at all.

export interface OllamaStatus {
  status: "ready" | "starting" | "not_installed" | "error";
}

export async function ollamaStatus(): Promise<OllamaStatus> {
  return { status: "not_installed" };
}

export async function ollamaStart(): Promise<void> { /* no-op */ }
export async function ollamaStop(): Promise<void> { /* no-op */ }
export async function ollamaListModels(): Promise<{ models: never[] }> {
  return { models: [] };
}
export async function ollamaModelInfo(_n: string): Promise<Record<string, unknown>> {
  return {};
}
export async function ollamaPullModel(_n: string): Promise<void> { /* no-op */ }
export async function ollamaDeleteModel(_n: string): Promise<void> { /* no-op */ }
export async function ollamaAvailableModels(): Promise<never[]> { return []; }
