/** Browser-demo app store.
 *
 * Drives the SPA off the bundled atlas + demo brain, the litert-lm sidecar
 * agent, and the in-browser quick-affine + image-gen registration pipelines.
 * No filesystem, no Tauri, no local-engine discovery. */

import { create } from "zustand";
import type {
  AtlasInfo,
  AtlasLoadProgress,
  Brain,
  ChatMessage,
  ChatTelemetry,
  MeshData,
  NavigationView,
  Plane,
  PipelineStatus,
  RegistrationResult,
  SliceInfo,
  ViewMode,
} from "../lib/types";
import * as commands from "../lib/browserCommands";
import { runEstimate, type AgentEvent } from "../lib/agentLoop";
import { quickAffineRegister } from "../lib/quickAffine";
import { imageGenRegister } from "../lib/imageGenRegistration";
import {
  probeAllEngines,
  probeCustomEndpoint,
  type CustomEndpoint,
  type LocalEngineStatus,
} from "../lib/localEngines";

/** Which warp direction the registration result viewer should display. */
export type RegistrationViewMode = "atlas-to-slice" | "slice-to-atlas";

/** Atlas-borders source for the atlas→slice overlay. */
export type BorderSource = "elastix" | "generated";

interface AppState {
  // Navigation
  currentView: NavigationView;
  selectedBrainId: string | null;

  // Brains
  brains: Brain[];
  hoveredBrainId: string | null;
  /** Per-slice ground-truth AP mm stashed out of the agent's reach. Keyed
   *  by the slice's manifest filename. UI may surface, agent never sees. */
  demoBrainGroundTruthMm: Record<string, number>;

  // Atlas
  atlasName: string;
  atlasInfo: AtlasInfo | null;
  atlasLoading: boolean;
  atlasLoadProgress: AtlasLoadProgress | null;
  brainMesh: MeshData | null;

  // Current AP slider position (drives 3D plane + slice write-back)
  currentApMm: number;

  // Selected slice
  selectedSliceIndex: number | null;

  // === Estimate Position settings ===
  estimateModel: string;
  estimateTemperature: number;
  estimateMaxIterations: number;
  /** Send the CLAHE-equalized variant of the slice when available. */
  estimateUseClahe: boolean;
  /** Gemini-only knobs (silently ignored when routed to local sidecar). */
  estimateThinkingPreset: "off" | "low" | "medium" | "high";
  estimateMediaResolution: "low" | "medium" | "high";

  // === Run Registration settings ===
  registerImageModel: string;
  registerTemperature: number;
  registerUseClahe: boolean;
  registerThinkingPreset: "off" | "low" | "medium" | "high";
  registerMediaResolution: "low" | "medium" | "high";

  /** Display-only CLAHE toggle wired to the 3D + split viewers. Does NOT
   *  affect what the agent or registration call sees — those have their
   *  own per-stage toggles. */
  clahePreview: boolean;

  apiKeysOpen: boolean;
  localModelsOpen: boolean;

  // Chat drawer (BYO-engine) ───────────────────────────────
  chatOpen: boolean;
  chatMessages: ChatMessage[];
  chatEndpoint: string | null;
  chatModelId: string | null;
  chatEngineName: string | null;
  chatStreaming: boolean;
  chatError: string | null;
  chatTelemetry: ChatTelemetry;
  chatAbort: AbortController | null;
  /** Browser-only stand-in for the desktop's eagerly-loaded base64 slice
   * preview. Read by the chat drawer's "attach slice" affordance to pass
   * the currently-selected slice as an inline image_url part. */
  selectedSliceImage: string | null;

  // Local-engine discovery (Local Models modal)
  localEngines: LocalEngineStatus[];
  customEndpoints: CustomEndpoint[];
  localEnginesProbing: boolean;
  /** null = use the bundled-sidecar default (litert-lm @ 8765).
   *  Set to a model's `endpoint` (e.g. "http://127.0.0.1:11434/v1") when
   *  the user picks a non-litert local engine via the Local Models modal.
   *  Threaded into runEstimate's baseUrl so the agent talks to the right
   *  server. */
  estimateEndpoint: string | null;

  // Pipeline
  pipelineStatus: PipelineStatus;
  pipelineError: string | null;
  pipelineRunning: boolean;
  logs: string[];

  // View
  viewMode: ViewMode;
  registrationViewMode: RegistrationViewMode;
  borderSource: BorderSource;

  // Actions — navigation
  navigateToDashboard: () => void;
  navigateToBrain: (brainId: string) => void;

  // Actions — brains
  setHoveredBrain: (id: string | null) => void;
  removeBrain: (brainId: string) => void;
  setBrainPlane: (brainId: string, plane: Plane) => void;
  loadDemoBrain: () => Promise<void>;
  /** Browser-only: ingest a folder of image files picked via <input type="file">. */
  addBrainFromFiles: (files: File[]) => Promise<void>;

  setStageSetting: (
    stage: "estimate" | "register",
    key: string,
    value: string | number | boolean,
  ) => void;

  setClahePreview: (value: boolean) => void;

  setRegistrationViewMode: (mode: RegistrationViewMode) => void;
  setBorderSource: (source: BorderSource) => void;

  // Actions — pipeline
  runEstimatePosition: () => Promise<void>;
  runRegistration: () => Promise<void>;

  // Actions — slices
  selectSlice: (index: number) => void;

  // Actions — atlas
  loadAtlas: (name: string) => Promise<void>;

  // Actions — API keys modal
  openApiKeys: () => void;
  closeApiKeys: () => void;

  // Actions — local models modal
  openLocalModels: () => void;
  closeLocalModels: () => void;

  // Actions — chat drawer
  openChat: () => void;
  closeChat: () => void;
  setChatModel: (endpoint: string, modelId: string, engineName: string) => void;
  sendChatMessage: (text: string, attachSlice: boolean) => Promise<void>;
  cancelChatStream: () => void;
  clearChat: () => void;

  // Actions — local-engine discovery
  probeLocalEngines: () => Promise<void>;
  addCustomEndpoint: (ep: CustomEndpoint) => Promise<void>;
  removeCustomEndpoint: (url: string) => void;
  setEstimateModelChoice: (modelId: string, endpoint: string | null) => void;

  setApPosition: (apMm: number) => void;
  /** Toggle the lock on the selected slice's AP position. */
  setApLocked: (locked: boolean) => void;
  runQuickAffine: () => Promise<void>;
  toggleSliceVisibleIn3D: (sliceIndex: number) => void;
  setViewMode: (mode: ViewMode) => void;
  addLog: (msg: string) => void;
}

// litert-lm-serve default; pinned so the agent and the sidecar probe agree.
const SIDECAR_BASE_URL = "http://127.0.0.1:8765";
const DEFAULT_LOCAL_MODEL = "gemma-4-E4B-it";

// Monotonic counter for uploaded brain ids. Module-scope: survives Zustand
// resets, never collides with the bundled demo brain (which uses brain_id
// from the manifest).
let uploadedBrainCounter = 0;

/** Resolve a slice's `path` field to a URL the browser can fetch.
 *
 * - Bundled demo-brain slices store bare filenames (e.g. "section_01.png");
 *   prefix with the BASE_URL'd demo_brain root.
 * - Uploaded slices store `blob:` URLs (or in principle `http(s):` / `data:`);
 *   use them as-is. */
function resolveSliceUrl(path: string): string {
  if (
    path.startsWith("blob:") ||
    path.startsWith("http://") ||
    path.startsWith("https://") ||
    path.startsWith("data:")
  ) {
    return path;
  }
  return `${import.meta.env.BASE_URL}demo_brain/${path}`;
}

/** Fetch a URL and convert the response body to a `data:` URL. Used both by
 * `selectSlice` (background-load the selected slice's PNG so the chat drawer
 * can attach it as an inline image) and by `sendChatMessage` if it ever needs
 * a synchronous fallback. The browser stand-in for the Tauri command that
 * eagerly hands us a base64 PNG string. */
async function urlToDataUrl(url: string): Promise<string> {
  const blob = await (await fetch(url)).blob();
  return await new Promise<string>((resolve, reject) => {
    const reader = new FileReader();
    reader.onload = () => resolve(reader.result as string);
    reader.onerror = () =>
      reject(reader.error ?? new Error("FileReader failed"));
    reader.readAsDataURL(blob);
  });
}

export const useAppStore = create<AppState>((set, get) => ({
  // Navigation
  currentView: "dashboard",
  selectedBrainId: null,

  // Brains
  brains: [],
  hoveredBrainId: null,
  demoBrainGroundTruthMm: {},

  // Atlas
  atlasName: "allen_mouse_25um",
  atlasInfo: null,
  atlasLoading: false,
  atlasLoadProgress: null,
  brainMesh: null,

  currentApMm: 5.0,

  selectedSliceIndex: null,

  estimateModel: DEFAULT_LOCAL_MODEL,
  estimateTemperature: 1.0,
  estimateMaxIterations: 20,
  estimateUseClahe: true,
  estimateThinkingPreset: "medium",
  estimateMediaResolution: "medium",

  registerImageModel: "gemini-3-pro-image-preview",
  registerTemperature: 1.0,
  registerUseClahe: true,
  registerThinkingPreset: "medium",
  registerMediaResolution: "medium",

  clahePreview: true,

  apiKeysOpen: false,
  localModelsOpen: false,

  chatOpen: false,
  chatMessages: [],
  chatEndpoint: null,
  chatModelId: null,
  chatEngineName: null,
  chatStreaming: false,
  chatError: null,
  chatTelemetry: {},
  chatAbort: null,
  selectedSliceImage: null,

  localEngines: [],
  customEndpoints: [],
  localEnginesProbing: false,
  estimateEndpoint: null,

  pipelineStatus: "idle",
  pipelineError: null,
  pipelineRunning: false,
  logs: [],

  viewMode: "3d",
  registrationViewMode: "atlas-to-slice",
  borderSource: "elastix",

  // ── Navigation ─────────────────────────────────────────────
  navigateToDashboard: () =>
    set({ currentView: "dashboard", selectedBrainId: null }),

  navigateToBrain: (brainId: string) =>
    set({ currentView: "brain-detail", selectedBrainId: brainId }),

  setHoveredBrain: (id: string | null) => set({ hoveredBrainId: id }),

  removeBrain: (brainId: string) =>
    set((s) => {
      const target = s.brains.find((b) => b.id === brainId);
      if (target) {
        // Release blob: URLs minted by addBrainFromFiles to avoid leaks.
        for (const sl of target.slices) {
          if (sl.path.startsWith("blob:")) URL.revokeObjectURL(sl.path);
        }
      }
      return { brains: s.brains.filter((b) => b.id !== brainId) };
    }),

  setBrainPlane: (brainId: string, plane: Plane) =>
    set((s) => ({
      brains: s.brains.map((b) => (b.id === brainId ? { ...b, plane } : b)),
    })),

  loadDemoBrain: async () => {
    try {
      const m = await commands.getDemoBrainManifest();
      const slices: SliceInfo[] = m.sections.map((s) => ({
        path: s.filename,
        clahePath: s.clahe_filename,
        name: s.filename,
        status: "pending" as const,
        // Intentionally not setting apMm — the agent must estimate from the
        // image alone. GT mm lives in `demoBrainGroundTruthMm`.
      }));
      const brain: Brain = {
        id: m.brain_id,
        name: m.brain_id,
        folderPath: "(bundled)",
        slices,
        status: "queued",
        completedCount: 0,
        plane: m.plane,
      };
      const gt = Object.fromEntries(
        m.sections.map((s) => [s.filename, s.ground_truth_mm]),
      );
      set({ brains: [brain], demoBrainGroundTruthMm: gt });
      get().addLog(
        `Loaded demo brain ${m.brain_id} (${m.sections.length} sections)`,
      );
    } catch (e) {
      const msg = e instanceof Error ? e.message : String(e);
      get().addLog(`Failed to load demo brain: ${msg}`);
    }
  },

  addBrainFromFiles: async (files: File[]) => {
    if (files.length === 0) return;
    // Filter to images; sort by name so the slices line up A-to-P.
    const images = files
      .filter((f) => /\.(png|jpe?g|tiff?|webp)$/i.test(f.name))
      .sort((a, b) => a.name.localeCompare(b.name));
    if (images.length === 0) {
      get().addLog("No image files in selection (png/jpg/tiff/webp).");
      return;
    }
    // Mint object URLs per file; the agent / quick-affine / image-gen
    // pipelines all consume URLs and treat blob: identically to http:.
    // URLs are revoked in `removeBrain`.
    const slices: SliceInfo[] = images.map((f) => ({
      path: URL.createObjectURL(f),
      name: f.name,
      status: "pending" as const,
    }));
    // webkitRelativePath is set when the file picker has `webkitdirectory`;
    // when it isn't (plain `multiple`), fall back to a generic label.
    const firstRel = (images[0] as File & { webkitRelativePath?: string })
      .webkitRelativePath;
    const folderName =
      firstRel && firstRel.includes("/")
        ? firstRel.split("/")[0]
        : `upload-${++uploadedBrainCounter}`;
    const brain: Brain = {
      id: `upload-${uploadedBrainCounter}-${Date.now()}`,
      name: folderName,
      folderPath: "(uploaded)",
      slices,
      status: "queued",
      completedCount: 0,
      plane: "coronal",
    };
    set((s) => ({ brains: [...s.brains, brain] }));
    get().addLog(
      `Loaded ${images.length} slice${images.length === 1 ? "" : "s"} from upload.`,
    );
  },

  setStageSetting: (stage, key, value) => {
    const prefix = stage;
    const camel = key.charAt(0).toUpperCase() + key.slice(1);
    set({ [`${prefix}${camel}`]: value } as Partial<AppState>);
  },

  setClahePreview: (value: boolean) => set({ clahePreview: value }),

  setRegistrationViewMode: (mode: RegistrationViewMode) =>
    set({ registrationViewMode: mode }),

  setBorderSource: (source: BorderSource) => set({ borderSource: source }),

  // ── Pipeline: estimate via the litert-lm agent ────────────
  runEstimatePosition: async () => {
    const {
      selectedBrainId, selectedSliceIndex, brains, atlasName, atlasInfo,
      estimateMaxIterations, estimateModel, estimateEndpoint,
      estimateUseClahe, estimateThinkingPreset, estimateMediaResolution,
    } = get();

    const brain = brains.find((b) => b.id === selectedBrainId);
    if (!brain || selectedSliceIndex === null) {
      get().addLog("Select a slice first");
      return;
    }
    const slice = brain.slices[selectedSliceIndex];
    if (!slice) return;
    if (!atlasInfo) {
      get().addLog("Atlas not loaded yet");
      return;
    }
    if (atlasName !== "allen_mouse_25um" || brain.plane !== "coronal") {
      get().addLog(
        `Browser demo only supports allen_mouse_25um coronal; got ${atlasName}/${brain.plane}`,
      );
      return;
    }

    set({ pipelineRunning: true, pipelineStatus: "estimating", pipelineError: null });
    get().addLog(`Estimating AP position on ${slice.name}...`);

    set((s) => ({
      brains: s.brains.map((b) =>
        b.id === selectedBrainId
          ? {
              ...b,
              slices: b.slices.map((sl, i) =>
                i === selectedSliceIndex ? { ...sl, status: "running" as const } : sl,
              ),
            }
          : b,
      ),
    }));

    // CLAHE toggle picks the equalized URL when both are available; falls
    // back to the base PNG for uploaded slices that have no clahePath.
    const sourcePath =
      estimateUseClahe && slice.clahePath ? slice.clahePath : slice.path;
    const imageUrl = resolveSliceUrl(sourcePath);

    // Fine-tuned LangSlice-Gemma models don't need the broad/narrow playbook;
    // strip it (and the submit_estimate gate) when the user picks one.
    const freeMode = /gemma/i.test(estimateModel);

    try {
      const generator = runEstimate({
        imageUrl,
        atlasName: "allen_mouse_25um",
        plane: "coronal",
        posLo: atlasInfo.ap_min_mm,
        posHi: atlasInfo.ap_max_mm,
        species: "Mus musculus",
        modelId: estimateModel,
        baseUrl: estimateEndpoint ?? SIDECAR_BASE_URL,
        maxIterations: estimateMaxIterations,
        freeMode,
        thinkingPreset: estimateThinkingPreset,
        mediaResolution: estimateMediaResolution,
      });

      let finalEstimate: { positionMm: number; iterations: number } | null = null;
      while (true) {
        const step = await generator.next();
        if (step.done) {
          finalEstimate = step.value;
          break;
        }
        const evt: AgentEvent = step.value;
        switch (evt.type) {
          case "model_text":
            if (evt.text) get().addLog(`model: ${evt.text.slice(0, 500)}`);
            break;
          case "tool_call": {
            const args = evt.toolArgs ? JSON.stringify(evt.toolArgs) : "{}";
            get().addLog(`tool_call ${evt.toolName ?? "?"}(${args})`);
            break;
          }
          case "tool_result": {
            const status =
              (evt.toolResult as { status?: unknown })?.status ?? "?";
            get().addLog(`tool_result ${evt.toolName ?? "?"} → ${String(status)}`);
            break;
          }
          case "error":
            get().addLog(`agent error: ${evt.error ?? "?"}`);
            break;
          case "done":
            break;
        }
      }
      if (!finalEstimate) throw new Error("agent loop exited without a result");

      const positionMm = finalEstimate.positionMm;
      get().addLog(
        `AP estimated: ${positionMm.toFixed(3)} mm (${finalEstimate.iterations} iter)`,
      );
      get().setApPosition(positionMm);

      // Auto-lock and kick the quick-affine preview.
      set((s) => ({
        pipelineStatus: "complete",
        brains: s.brains.map((b) =>
          b.id === selectedBrainId
            ? {
                ...b,
                slices: b.slices.map((sl, i) =>
                  i === selectedSliceIndex
                    ? { ...sl, status: "done" as const, apMm: positionMm, apLocked: true }
                    : sl,
                ),
              }
            : b,
        ),
      }));
      void get().runQuickAffine();
    } catch (e) {
      const msg = e instanceof Error ? e.message : String(e);
      get().addLog(`Estimation error: ${msg}`);
      set({ pipelineStatus: "error", pipelineError: msg });
      set((s) => ({
        brains: s.brains.map((b) =>
          b.id === selectedBrainId
            ? {
                ...b,
                slices: b.slices.map((sl, i) =>
                  i === selectedSliceIndex
                    ? { ...sl, status: "pending" as const }
                    : sl,
                ),
              }
            : b,
        ),
      }));
    } finally {
      set({ pipelineRunning: false });
    }
  },

  // ── Pipeline: image-gen registration via Gemini ───────────
  runRegistration: async () => {
    const {
      selectedBrainId, selectedSliceIndex, brains, registerImageModel,
      registerUseClahe, registerThinkingPreset, registerMediaResolution,
    } = get();

    const brain = brains.find((b) => b.id === selectedBrainId);
    if (!brain || selectedSliceIndex === null) {
      get().addLog("Select a slice first");
      return;
    }
    const slice = brain.slices[selectedSliceIndex];
    if (!slice) return;
    if (slice.apMm === undefined) {
      get().addLog("No AP position set -- estimate or pick one with the slider first");
      return;
    }

    let geminiApiKey: string;
    try {
      const env = await commands.readEnvFile();
      geminiApiKey = env.vars.GEMINI_API_KEY ?? "";
    } catch (e) {
      const msg = e instanceof Error ? e.message : String(e);
      get().addLog(`Could not read API keys: ${msg}`);
      return;
    }
    if (!geminiApiKey.trim()) {
      get().addLog("Set GEMINI_API_KEY in the API Keys modal before registering");
      return;
    }

    set({ pipelineRunning: true, pipelineStatus: "registering", pipelineError: null });
    get().addLog(`Running image-gen registration at ${slice.apMm.toFixed(3)} mm...`);

    try {
      const sourcePath =
        registerUseClahe && slice.clahePath ? slice.clahePath : slice.path;
      const url = resolveSliceUrl(sourcePath);
      const sliceBitmap = await createImageBitmap(
        await (await fetch(url)).blob(),
      );
      const result = await imageGenRegister({
        slice: sliceBitmap,
        atlasApMm: slice.apMm,
        geminiApiKey,
        imageModelId: registerImageModel,
        thinkingPreset: registerThinkingPreset,
        mediaResolution: registerMediaResolution,
        onProgress: (stage, note) => {
          get().addLog(`registration ${stage}${note ? ": " + note : ""}`);
        },
      });
      sliceBitmap.close();

      const registrationResult: RegistrationResult = {
        warpedAtlas: result.warpedAtlas,
        warpedBorders: result.warpedBorders,
        generatedColoredRegions: result.generatedColoredRegions,
      };

      get().addLog(
        `Image-gen registration complete (${(result.elapsedMs / 1000).toFixed(1)}s)`,
      );

      set((s) => ({
        pipelineStatus: "complete",
        brains: s.brains.map((b) =>
          b.id === selectedBrainId
            ? {
                ...b,
                completedCount: b.completedCount + 1,
                slices: b.slices.map((sl, i) =>
                  i === selectedSliceIndex
                    ? { ...sl, status: "done" as const, registrationResult }
                    : sl,
                ),
              }
            : b,
        ),
      }));
    } catch (e) {
      const msg = e instanceof Error ? e.message : String(e);
      get().addLog(`Registration error: ${msg}`);
      set({ pipelineStatus: "error", pipelineError: msg });
    } finally {
      set({ pipelineRunning: false });
    }
  },

  selectSlice: (index: number) => {
    const { selectedBrainId, brains } = get();
    const brain = brains.find((b) => b.id === selectedBrainId);
    if (!brain || index < 0 || index >= brain.slices.length) return;

    set({ selectedSliceIndex: index, selectedSliceImage: null });

    const slice = brain.slices[index];
    if (slice.apMm !== undefined) {
      get().setApPosition(slice.apMm);
    }

    // Background-load the slice as a base64 string so the chat drawer can
    // attach it as a vision input without re-fetching. Mirrors the Tauri
    // store shape: bare base64 (no "data:image/...;base64," prefix). The
    // chat action re-applies the prefix when building the OpenAI payload.
    void urlToDataUrl(resolveSliceUrl(slice.path))
      .then((dataUrl) => {
        // Drop the result if the user moved on before it landed.
        if (
          get().selectedSliceIndex === index &&
          get().selectedBrainId === selectedBrainId
        ) {
          const b64 = dataUrl.replace(/^data:[^,]*,/, "");
          set({ selectedSliceImage: b64 });
        }
      })
      .catch(() => {});
  },

  // ── Atlas: fetch manifest + mesh; per-section PNGs come on demand ─
  loadAtlas: async (name: string) => {
    set({
      atlasLoading: true,
      atlasLoadProgress: { phase: "Loading manifest...", percent: 10 },
      pipelineStatus: "loading_atlas",
      pipelineError: null,
    });
    try {
      await commands.loadAtlas(name);
      set({ atlasLoadProgress: { phase: "Reading atlas info...", percent: 35 } });
      const info = await commands.getAtlasInfo();
      const midAp = (info.ap_min_mm + info.ap_max_mm) / 2;
      set({ atlasName: name, atlasInfo: info, currentApMm: midAp });
      get().addLog(`Atlas loaded: ${info.name} (${info.ap_count} slices)`);

      set({ atlasLoadProgress: { phase: "Loading brain mesh...", percent: 70 } });
      try {
        const mesh = await commands.getBrainMesh();
        set({ brainMesh: mesh });
        get().addLog(`Brain mesh loaded: ${mesh.positions.length / 3} vertices`);
      } catch (e) {
        const msg = e instanceof Error ? e.message : String(e);
        get().addLog(`Brain mesh unavailable: ${msg}`);
      }

      // Fire-and-forget: warm the HTTP cache for every section's borders PNG
      // so the 3D viewer's AP slider feels instant. Doesn't block atlas load.
      void commands.prefetchAllBorders();

      set({ pipelineStatus: "idle" });
    } catch (e) {
      const msg = e instanceof Error ? e.message : String(e);
      set({ pipelineStatus: "error", pipelineError: msg });
      get().addLog(`Error loading atlas: ${msg}`);
    } finally {
      set({ atlasLoading: false, atlasLoadProgress: null });
    }
  },

  openApiKeys: () => set({ apiKeysOpen: true }),
  closeApiKeys: () => set({ apiKeysOpen: false }),

  openLocalModels: () => {
    set({ localModelsOpen: true });
    // Fire-and-forget probe so the modal mounts with a fresh list.
    void get().probeLocalEngines();
  },
  closeLocalModels: () => set({ localModelsOpen: false }),

  // ── Chat drawer (BYO-engine) ─────────────────────────────
  openChat: () => {
    set({ chatOpen: true });
    // Refresh the local-engine list on open so the connection picker is
    // never showing stale "not running" rows. Doesn't block the drawer
    // from rendering — the UI handles the in-flight state.
    if (get().localEngines.length === 0) {
      void get().probeLocalEngines();
    }
  },
  closeChat: () => {
    // Cancel any in-flight stream when the drawer closes so we don't burn
    // tokens for a window the user can't see.
    get().cancelChatStream();
    set({ chatOpen: false });
  },
  setChatModel: (endpoint, modelId, engineName) =>
    set({
      chatEndpoint: endpoint,
      chatModelId: modelId,
      chatEngineName: engineName,
      chatError: null,
    }),

  cancelChatStream: () => {
    const abort = get().chatAbort;
    if (abort) abort.abort();
    set({ chatAbort: null, chatStreaming: false });
  },

  clearChat: () =>
    set({ chatMessages: [], chatTelemetry: {}, chatError: null }),

  sendChatMessage: async (text, attachSlice) => {
    const state = get();
    if (!state.chatEndpoint || !state.chatModelId) {
      set({ chatError: "No engine selected. Patch into a local model first." });
      return;
    }
    if (state.chatStreaming) return; // ignore re-entrance

    const trimmed = text.trim();
    if (!trimmed) return;

    // Build the new user turn. If the user opted in, attach the currently
    // selected slice as a data URL so vision-capable models can see it. In
    // the browser, slices live at URLs (blob:/http(s):/bundled-asset paths);
    // we read the eagerly-loaded base64 stashed by `selectSlice` and fall
    // back to a live URL→data-URL fetch if the background load hasn't
    // landed yet. Either way we hand the chat completions endpoint a
    // self-contained `image_url` part, never a same-origin reference.
    const sliceB64 = state.selectedSliceImage;
    const images: string[] = [];
    if (attachSlice) {
      if (sliceB64) {
        images.push(`data:image/png;base64,${sliceB64}`);
      } else {
        const brain = state.brains.find((b) => b.id === state.selectedBrainId);
        const slice =
          brain && state.selectedSliceIndex !== null
            ? brain.slices[state.selectedSliceIndex]
            : null;
        if (slice) {
          try {
            const dataUrl = await urlToDataUrl(resolveSliceUrl(slice.path));
            images.push(dataUrl);
          } catch (e) {
            // Non-fatal — surface as error but send the text-only message.
            const msg = e instanceof Error ? e.message : String(e);
            set({ chatError: `Could not attach slice: ${msg}` });
          }
        }
      }
    }

    const userMessage: ChatMessage = {
      id: `u-${Date.now()}`,
      role: "user",
      content: trimmed,
      images: images.length > 0 ? images : undefined,
    };
    const assistantMessage: ChatMessage = {
      id: `a-${Date.now()}`,
      role: "assistant",
      content: "",
    };

    set({
      chatMessages: [...state.chatMessages, userMessage, assistantMessage],
      chatStreaming: true,
      chatError: null,
      chatTelemetry: {},
    });

    // OpenAI-compat `messages` shape. For multimodal turns, the user
    // content becomes an array of {type:"text"|"image_url",...} parts.
    const messagesForApi = [...state.chatMessages, userMessage].map((m) => {
      if (m.role === "user" && m.images && m.images.length > 0) {
        return {
          role: m.role,
          content: [
            { type: "text", text: m.content },
            ...m.images.map((url) => ({
              type: "image_url",
              image_url: { url },
            })),
          ],
        };
      }
      return { role: m.role, content: m.content };
    });

    const controller = new AbortController();
    set({ chatAbort: controller });
    const startedAt = performance.now();
    let firstTokenAt: number | null = null;
    let rxTokens = 0;
    let assistantText = "";

    try {
      const res = await fetch(`${state.chatEndpoint}/chat/completions`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          model: state.chatModelId,
          messages: messagesForApi,
          stream: true,
        }),
        signal: controller.signal,
      });

      if (!res.ok || !res.body) {
        const detail = await res.text().catch(() => "");
        throw new Error(`HTTP ${res.status}: ${detail.slice(0, 200) || res.statusText}`);
      }

      const reader = res.body.getReader();
      const decoder = new TextDecoder();
      let buffer = "";

      while (true) {
        const { done, value } = await reader.read();
        if (done) break;
        buffer += decoder.decode(value, { stream: true });

        // SSE frames are separated by blank lines. Split, keep the trailing
        // partial in the buffer for the next chunk.
        const frames = buffer.split("\n\n");
        buffer = frames.pop() ?? "";

        for (const frame of frames) {
          for (const line of frame.split("\n")) {
            if (!line.startsWith("data:")) continue;
            const payload = line.slice(5).trim();
            if (!payload || payload === "[DONE]") continue;
            let parsed: { choices?: { delta?: { content?: string } }[] };
            try {
              parsed = JSON.parse(payload);
            } catch {
              continue;
            }
            const delta = parsed.choices?.[0]?.delta?.content;
            if (!delta) continue;

            if (firstTokenAt === null) {
              firstTokenAt = performance.now();
            }
            assistantText += delta;
            rxTokens += 1; // chunks ≈ tokens for most engines; close enough for HUD
            const now = performance.now();
            const elapsedMs = now - startedAt;
            const tokPerSec = elapsedMs > 0 ? (rxTokens * 1000) / elapsedMs : 0;
            set((s) => ({
              chatMessages: s.chatMessages.map((m) =>
                m.id === assistantMessage.id ? { ...m, content: assistantText } : m,
              ),
              chatTelemetry: {
                rxTokens,
                ttftMs: firstTokenAt !== null ? Math.round(firstTokenAt - startedAt) : undefined,
                elapsedMs: Math.round(elapsedMs),
                tokensPerSec: Math.round(tokPerSec * 10) / 10,
              },
            }));
          }
        }
      }
    } catch (e: unknown) {
      const name = e instanceof Error ? e.name : "";
      if (name === "AbortError") {
        // Quiet cancel — leave whatever tokens we already streamed.
      } else {
        const msg = e instanceof Error ? e.message : String(e);
        set({ chatError: msg });
      }
    } finally {
      set({ chatStreaming: false, chatAbort: null });
    }
  },

  // ── Local-engine discovery ───────────────────────────────
  probeLocalEngines: async () => {
    set({ localEnginesProbing: true });
    try {
      const builtIns = await probeAllEngines();
      // Re-probe each saved custom endpoint too so they show up alongside.
      const customResults = await Promise.all(
        get().customEndpoints.map((ep) =>
          probeCustomEndpoint(ep.label, ep.url).catch(
            (e): LocalEngineStatus => ({
              name: ep.label,
              port: 0,
              endpoint: ep.url,
              reachable: false,
              models: [],
              error: e instanceof Error ? e.message : String(e),
            }),
          ),
        ),
      );
      set({ localEngines: [...builtIns, ...customResults] });
    } catch (e) {
      const msg = e instanceof Error ? e.message : String(e);
      get().addLog(`Local-engine probe failed: ${msg}`);
    } finally {
      set({ localEnginesProbing: false });
    }
  },

  addCustomEndpoint: async (ep: CustomEndpoint) => {
    // Reject duplicates by URL so the list stays clean.
    const existing = get().customEndpoints;
    if (existing.some((e) => e.url === ep.url)) return;
    set({ customEndpoints: [...existing, ep] });
    await get().probeLocalEngines();
  },

  removeCustomEndpoint: (url: string) => {
    set((s) => ({
      customEndpoints: s.customEndpoints.filter((e) => e.url !== url),
      localEngines: s.localEngines.filter((eng) => eng.endpoint !== url),
    }));
  },

  setEstimateModelChoice: (modelId: string, endpoint: string | null) => {
    // Engine probes report endpoints with the OpenAI-compat `/v1` suffix
    // (e.g. http://127.0.0.1:8765/v1). The agent loop hands the baseUrl to
    // @google/genai which appends its own `/v1beta/...`, so strip the
    // trailing `/v1` before storing.
    const stripped =
      endpoint && endpoint.endsWith("/v1")
        ? endpoint.slice(0, -3)
        : endpoint;
    set({ estimateModel: modelId, estimateEndpoint: stripped });
  },

  setApLocked: (locked: boolean) => {
    const { selectedBrainId, selectedSliceIndex } = get();
    if (!selectedBrainId || selectedSliceIndex === null) return;
    set((s) => ({
      brains: s.brains.map((b) =>
        b.id !== selectedBrainId
          ? b
          : {
              ...b,
              slices: b.slices.map((sl, i) =>
                i === selectedSliceIndex ? { ...sl, apLocked: locked } : sl,
              ),
            },
      ),
    }));
    if (locked) {
      void get().runQuickAffine();
    }
  },

  toggleSliceVisibleIn3D: (sliceIndex: number) => {
    const { selectedBrainId } = get();
    if (!selectedBrainId) return;
    set((s) => ({
      brains: s.brains.map((b) =>
        b.id !== selectedBrainId
          ? b
          : {
              ...b,
              slices: b.slices.map((sl, i) =>
                i === sliceIndex
                  ? { ...sl, visibleIn3D: sl.visibleIn3D === false ? true : false }
                  : sl,
              ),
            },
      ),
    }));
  },

  // ── In-browser quick affine preview ───────────────────────
  runQuickAffine: async () => {
    const { selectedBrainId, selectedSliceIndex, atlasName, currentApMm, brains, clahePreview } = get();
    if (!selectedBrainId || selectedSliceIndex === null || !atlasName) return;
    const brain = brains.find((b) => b.id === selectedBrainId);
    if (!brain) return;
    const slice = brain.slices[selectedSliceIndex];
    if (!slice) return;

    const sliceIndexSnapshot = selectedSliceIndex;
    const apSnapshot = currentApMm;
    const brainIdSnapshot = selectedBrainId;
    // CLAHE preview toggle decides which variant feeds the 3D warped plane.
    const path = clahePreview && slice.clahePath ? slice.clahePath : slice.path;
    set((s) => ({
      brains: s.brains.map((b) =>
        b.id !== brainIdSnapshot
          ? b
          : {
              ...b,
              slices: b.slices.map((sl, i) =>
                i === sliceIndexSnapshot
                  ? { ...sl, quickAffineRunning: true, quickAffineError: null }
                  : sl,
              ),
            },
      ),
    }));

    try {
      const url = resolveSliceUrl(path);
      const bmp = await createImageBitmap(await (await fetch(url)).blob());
      const result = await quickAffineRegister({
        slice: bmp,
        atlasApMm: apSnapshot,
      });
      bmp.close();

      // Stale-result guard: drop if the user moved on.
      const now = get();
      if (
        now.selectedSliceIndex !== sliceIndexSnapshot ||
        now.selectedBrainId !== brainIdSnapshot ||
        Math.abs(now.currentApMm - apSnapshot) > 0.01
      ) {
        result.warped.close();
        return;
      }

      set((s) => ({
        brains: s.brains.map((b) =>
          b.id !== brainIdSnapshot
            ? b
            : {
                ...b,
                slices: b.slices.map((sl, i) =>
                  i === sliceIndexSnapshot
                    ? {
                        ...sl,
                        quickAffineRunning: false,
                        quickAffineWarpedBitmap: result.warped,
                        quickAffineError: null,
                      }
                    : sl,
                ),
              },
        ),
      }));
      get().addLog(
        `Quick affine preview ready (IoU ${(result.iou * 100).toFixed(1)}%, ${(result.elapsedMs / 1000).toFixed(1)}s)`,
      );
    } catch (e) {
      const msg = e instanceof Error ? e.message : String(e);
      set((s) => ({
        brains: s.brains.map((b) =>
          b.id !== brainIdSnapshot
            ? b
            : {
                ...b,
                slices: b.slices.map((sl, i) =>
                  i === sliceIndexSnapshot
                    ? { ...sl, quickAffineRunning: false, quickAffineError: msg }
                    : sl,
                ),
              },
        ),
      }));
      get().addLog(`Quick affine failed: ${msg}`);
    }
  },

  // Pure local: write to `currentApMm` and to the selected slice. Releases
  // the lock on any drag so the user must re-commit before re-registering.
  setApPosition: (apMm: number) => {
    const { selectedBrainId, selectedSliceIndex } = get();
    set((s) => {
      const next: Partial<AppState> = { currentApMm: apMm };
      if (selectedBrainId && selectedSliceIndex !== null) {
        next.brains = s.brains.map((b) => {
          if (b.id !== selectedBrainId) return b;
          return {
            ...b,
            slices: b.slices.map((sl, i) =>
              i === selectedSliceIndex ? { ...sl, apMm, apLocked: false } : sl,
            ),
          };
        });
      }
      return next;
    });
  },

  setViewMode: (mode: ViewMode) => set({ viewMode: mode }),

  addLog: (msg: string) =>
    set((s) => {
      const next = [...s.logs, `[${new Date().toLocaleTimeString()}] ${msg}`];
      return { logs: next.length > 500 ? next.slice(next.length - 500) : next };
    }),
}));
