import { create } from "zustand";
import type {
  AtlasDownloadProgress,
  AtlasInfo,
  AtlasLoadProgress,
  AvailableAtlas,
  Brain,
  ChatMessage,
  ChatTelemetry,
  CustomEndpoint,
  LocalEngineStatus,
  MeshData,
  NavigationView,
  Plane,
  SliceInfo,
  ViewMode,
  PipelineStatus,
} from "../lib/types";
import * as commands from "../lib/commands";

/** View mode for the registration result overlay (forward vs inverse warp). */
export type RegistrationViewMode = "atlas-to-slice" | "slice-to-atlas";

/** Which atlas-borders rendering to show in the atlas→slice overlay:
 *  - "elastix": Elastix-warped atlas regions → borders extracted → overlaid (current default)
 *  - "generated": region borders extracted directly from the model's image-gen
 *    output, with no Elastix step. Shows the raw model prediction. */
export type BorderSource = "elastix" | "generated";

/** Decode base64 to Uint8Array */
function b64ToBytes(b64: string): Uint8Array {
  const bin = atob(b64);
  const bytes = new Uint8Array(bin.length);
  for (let i = 0; i < bin.length; i++) bytes[i] = bin.charCodeAt(i);
  return bytes;
}

/** Unpack a single slice from the bit-packed border volume */
function unpackBorderSlice(
  packed: Uint8Array,
  sliceIndex: number,
  width: number,
  height: number,
): Uint8Array {
  const pixelsPerSlice = width * height;
  const bitOffset = sliceIndex * pixelsPerSlice;
  const result = new Uint8Array(pixelsPerSlice);

  for (let i = 0; i < pixelsPerSlice; i++) {
    const globalBit = bitOffset + i;
    const byteIdx = globalBit >> 3;
    const bitIdx = globalBit & 7;
    result[i] = (packed[byteIdx] >> bitIdx) & 1 ? 255 : 0;
  }

  return result;
}

interface BorderVolume {
  packed: Uint8Array;
  depth: number;
  height: number;
  width: number;
}

interface AppState {
  // Navigation
  currentView: NavigationView;
  selectedBrainId: string | null;

  // Brains
  brains: Brain[];
  hoveredBrainId: string | null;

  // Atlas
  availableAtlases: string[];
  atlasName: string;
  atlasInfo: AtlasInfo | null;
  atlasLoading: boolean;
  atlasLoadProgress: AtlasLoadProgress | null;
  brainMesh: MeshData | null;

  // Atlas manager modal — full BrainGlobe catalog + per-atlas download
  // progress. `atlasManagerFocus` is set when we open the modal in
  // response to a missing-atlas prompt (e.g. Load Demo) so the modal can
  // scroll and highlight that row.
  availableForDownload: AvailableAtlas[];
  atlasManagerOpen: boolean;
  atlasManagerFocus: string | null;
  atlasDownloadProgress: Record<string, AtlasDownloadProgress>;

  // Border volume (held in frontend memory)
  borderVolume: BorderVolume | null;

  // Current slice (extracted locally from borderVolume)
  currentApMm: number;
  currentBorderPixels: Uint8Array | null;
  borderWidth: number;
  borderHeight: number;

  // Selected slice
  selectedSliceIndex: number | null;
  selectedSliceImage: string | null; // base64 PNG
  sliceImageLoading: boolean;

  // Atlas volumes (held in frontend memory)
  referenceVolume: Uint8Array | null;
  additionalVolumes: Record<string, Uint8Array>;
  additionalVolumeNames: string[];
  volumesLoaded: boolean;

  // Channel toggles for split/overlay views
  showBorders: boolean;
  activeChannel: string; // "reference" or an additional volume name

  // Current composite (built locally from cached volumes)
  currentCompositeDataUrl: string | null;
  atlasOpacity: number;

  // Per-stage CLI parameters. Estimate and Register have overlapping but
  // not identical flag sets in `langslice_harness/cli.py`, so they get
  // separate slices here. Defaults mirror the CLI's `default=...` values.

  // === Estimate Position (langslice estimate) ===
  estimateModel: string;
  /** When the Estimate model is a local-engine model, holds its OpenAI-compat
   * base URL so the harness can be told `--endpoint <url>`. null for cloud. */
  estimateEndpoint: string | null;
  estimateThinking: string;
  estimateTemperature: number;
  estimateMediaResolution: string;
  estimateMaxIterations: number;
  estimatePreprocess: "auto" | "none";

  // === Run Registration (langslice register) ===
  /** Image-generation model — e.g. gemini-3-pro-image-preview. */
  registerImageModel: string;
  registerThinking: string;
  registerTemperature: number;
  registerEndpoint: string | null;

  // Local-engine browser (Local Models modal).
  localEngines: LocalEngineStatus[];
  /** User-added endpoints (non-default ports, remote machines, etc.). */
  customEndpoints: CustomEndpoint[];
  localModelsOpen: boolean;
  apiKeysOpen: boolean;
  /** Whether a probe is currently running, so the modal can show a spinner. */
  localEnginesProbing: boolean;

  // BYO-engine chat drawer. Conversation lives in-memory only — closing the
  // app discards it (intentional, this is a debug/inspection tool, not a
  // session-keeping notebook). Engine + model are sticky so reopening the
  // drawer doesn't force a fresh pick every time.
  chatOpen: boolean;
  chatMessages: ChatMessage[];
  chatEndpoint: string | null;   // e.g. "http://127.0.0.1:11434/v1"
  chatModelId: string | null;    // e.g. "gemma3:e4b-it"
  chatEngineName: string | null; // human label for the LINK pill ("Ollama")
  chatStreaming: boolean;
  chatError: string | null;
  chatTelemetry: ChatTelemetry;
  /** Abort handle for the in-flight stream so the user can cancel mid-token. */
  chatAbort: AbortController | null;

  // Pipeline
  pipelineStatus: PipelineStatus;
  pipelineError: string | null;
  pipelineRunning: boolean;
  logs: string[];

  // Ollama
  ollamaStatus: "ready" | "starting" | "not_installed" | "error";

  // View
  viewMode: ViewMode;
  /** Which warp direction the registration result viewer should display. */
  registrationViewMode: RegistrationViewMode;
  /** In atlas→slice view, choose between Elastix-warped borders or borders
   * extracted directly from the image-gen output. Ignored in slice→atlas. */
  borderSource: BorderSource;

  // Actions — navigation
  navigateToDashboard: () => void;
  navigateToBrain: (brainId: string) => void;

  // Actions — brains
  setHoveredBrain: (id: string | null) => void;
  addEmptyBrains: (count: number) => void;
  addBrainFromFolder: (folderPath: string) => Promise<void>;
  loadBrainImages: (brainId: string, folderPath: string) => Promise<void>;
  removeBrain: (brainId: string) => void;
  setBrainPlane: (brainId: string, plane: Plane) => void;

  /** Per-stage settings updater. `stage` selects which slice, `key` is the
   * field name inside that slice (e.g. setStageSetting("estimate","thinking","HIGH")). */
  setStageSetting: (
    stage: "estimate" | "register",
    key: string,
    value: string | number | boolean,
  ) => void;

  // Actions — view
  setRegistrationViewMode: (mode: RegistrationViewMode) => void;
  setBorderSource: (source: BorderSource) => void;

  // Actions — pipeline. The two stages are separately invokable so the user
  // can re-run registration without re-estimating, or skip estimation when
  // they've manually set the AP via the slider.
  runEstimatePosition: () => Promise<void>;
  runRegistration: () => Promise<void>;
  exportResults: () => Promise<void>;

  // Actions — slices
  selectSlice: (index: number) => Promise<void>;

  // Actions — atlas
  fetchAtlasList: () => Promise<void>;
  fetchAvailableAtlases: () => Promise<void>;
  openAtlasManager: (focus?: string | null) => void;
  closeAtlasManager: () => void;
  downloadAtlas: (name: string) => Promise<void>;
  loadAtlas: (name: string) => Promise<void>;

  // Actions — chat drawer
  openChat: () => void;
  closeChat: () => void;
  setChatModel: (endpoint: string, modelId: string, engineName: string) => void;
  sendChatMessage: (text: string, attachSlice: boolean) => Promise<void>;
  cancelChatStream: () => void;
  clearChat: () => void;

  // Actions — local engines + top-bar modals
  openLocalModels: () => void;
  closeLocalModels: () => void;
  openApiKeys: () => void;
  closeApiKeys: () => void;
  probeLocalEngines: () => Promise<void>;
  addCustomEndpoint: (ep: CustomEndpoint) => Promise<void>;
  removeCustomEndpoint: (url: string) => void;
  setEstimateModelChoice: (modelId: string, endpoint: string | null) => void;
  setApPosition: (apMm: number) => void;
  /** Toggle the lock on the selected slice's AP position. Run Registration
   * is gated on this so users can either Estimate (auto-locks) or manually
   * dial in a position and click Lock to commit. */
  setApLocked: (locked: boolean) => void;
  /** Affine-only Elastix preview. Fired automatically by Lock Position so a
   * brain-shaped warped slice appears in the 3D viewer in ~15-20s instead
   * of waiting for the full nano-banana pipeline (~45-60s). */
  runQuickAffine: () => Promise<void>;
  /** Toggle whether a given slice's warped plane is shown in the 3D viewer.
   * Only meaningful once the slice is locked; the slice-list eye toggle
   * surfaces this. */
  toggleSliceVisibleIn3D: (sliceIndex: number) => void;
  fetchBrainMesh: () => Promise<void>;
  buildComposite: () => void;
  setViewMode: (mode: ViewMode) => void;
  setAtlasOpacity: (opacity: number) => void;
  setShowBorders: (show: boolean) => void;
  setActiveChannel: (channel: string) => void;
  addLog: (msg: string) => void;
}

let brainIdCounter = 0;

export const useAppStore = create<AppState>((set, get) => ({
  // Navigation
  currentView: "dashboard",
  selectedBrainId: null,

  // Brains
  brains: [],
  hoveredBrainId: null,

  // Atlas
  availableAtlases: [],
  atlasName: "allen_mouse_25um",
  atlasInfo: null,
  atlasLoading: false,
  atlasLoadProgress: null,
  brainMesh: null,

  availableForDownload: [],
  atlasManagerOpen: false,
  atlasManagerFocus: null,
  atlasDownloadProgress: {},

  borderVolume: null,
  currentApMm: 5.0,
  currentBorderPixels: null,
  borderWidth: 0,
  borderHeight: 0,

  selectedSliceIndex: null,
  selectedSliceImage: null,
  sliceImageLoading: false,

  referenceVolume: null,
  additionalVolumes: {},
  additionalVolumeNames: [],
  volumesLoaded: false,

  showBorders: true,
  activeChannel: "reference",

  currentCompositeDataUrl: null,
  atlasOpacity: 0.5,

  // Defaults mirror `_add_estimate_parser` / `_add_register_parser` in
  // `src/langslice_harness/cli.py`. Estimate: Gemini 3 Flash text/vision
  // tool loop. Register: Nano Banana Pro for the colored-region warp.
  estimateModel: "gemini-3-flash-preview",
  estimateEndpoint: null,
  estimateThinking: "MEDIUM",
  estimateTemperature: 1.0,
  estimateMediaResolution: "medium",
  estimateMaxIterations: 20,
  estimatePreprocess: "auto",

  registerImageModel: "gemini-3-pro-image-preview",
  registerThinking: "MEDIUM",
  registerTemperature: 1.0,
  registerEndpoint: null,

  localEngines: [],
  customEndpoints: [],
  localModelsOpen: false,
  apiKeysOpen: false,
  localEnginesProbing: false,

  chatOpen: false,
  chatMessages: [],
  chatEndpoint: null,
  chatModelId: null,
  chatEngineName: null,
  chatStreaming: false,
  chatError: null,
  chatTelemetry: {},
  chatAbort: null,

  ollamaStatus: "not_installed",

  pipelineStatus: "idle",
  pipelineError: null,
  pipelineRunning: false,
  logs: [],

  viewMode: "3d",
  registrationViewMode: "atlas-to-slice",
  borderSource: "elastix",

  // Navigation actions
  navigateToDashboard: () => set({ currentView: "dashboard", selectedBrainId: null }),

  navigateToBrain: (brainId: string) => set({ currentView: "brain-detail", selectedBrainId: brainId }),

  setHoveredBrain: (id: string | null) => set({ hoveredBrainId: id }),

  addEmptyBrains: (count: number) => {
    const newBrains: Brain[] = [];
    for (let i = 0; i < count; i++) {
      const id = `brain-${++brainIdCounter}`;
      newBrains.push({
        id,
        name: `Brain ${brainIdCounter}`,
        folderPath: "",
        slices: [],
        status: "queued",
        completedCount: 0,
        plane: "coronal",
      });
    }
    set((s) => ({ brains: [...s.brains, ...newBrains] }));
    get().addLog(`Added ${count} empty brain(s)`);
  },

  addBrainFromFolder: async (folderPath: string) => {
    try {
      const images = await commands.scanImageFolder(folderPath);
      const folderName = folderPath.split(/[\\/]/).pop() || "Brain";
      const id = `brain-${++brainIdCounter}`;
      const slices: SliceInfo[] = images.map((img) => ({
        path: img.path,
        name: img.name,
        status: "pending" as const,
      }));

      const brain: Brain = {
        id,
        name: folderName,
        folderPath,
        slices,
        status: "queued",
        completedCount: 0,
        plane: "coronal",
      };

      set((s) => ({ brains: [...s.brains, brain] }));
      get().addLog(`Added brain "${folderName}": ${slices.length} slices`);

      // Pre-cache all thumbnails in parallel (background)
      const paths = slices.map((s) => s.path);
      get().addLog(`Pre-caching ${paths.length} thumbnails...`);
      commands.precacheImages(paths).then((n) => {
        get().addLog(`Pre-cached ${n} thumbnails`);
      });
    } catch (e) {
      const msg = e instanceof Error ? e.message : String(e);
      get().addLog(`Error scanning folder: ${msg}`);
    }
  },

  loadBrainImages: async (brainId: string, folderPath: string) => {
    try {
      const images = await commands.scanImageFolder(folderPath);
      if (images.length === 0) {
        get().addLog(`No images found in folder`);
        return;
      }

      const folderName = folderPath.split(/[\\/]/).pop() || "";
      const slices: SliceInfo[] = images.map((img) => ({
        path: img.path,
        name: img.name,
        status: "pending" as const,
      }));

      set((s) => ({
        brains: s.brains.map((b) =>
          b.id === brainId
            ? { ...b, slices, folderPath, name: folderName || b.name }
            : b
        ),
      }));
      get().addLog(`Loaded ${images.length} slices into brain`);

      // Pre-cache all thumbnails in parallel (background)
      const paths = slices.map((s) => s.path);
      get().addLog(`Pre-caching ${paths.length} thumbnails...`);
      commands.precacheImages(paths).then((n) => {
        get().addLog(`Pre-cached ${n} thumbnails`);
      });
    } catch (e) {
      const msg = e instanceof Error ? e.message : String(e);
      get().addLog(`Error loading images: ${msg}`);
    }
  },

  removeBrain: (brainId: string) =>
    set((s) => ({ brains: s.brains.filter((b) => b.id !== brainId) })),

  setBrainPlane: (brainId: string, plane: Plane) =>
    set((s) => ({
      brains: s.brains.map((b) => (b.id === brainId ? { ...b, plane } : b)),
    })),

  setStageSetting: (stage, key, value) => {
    // Stage-prefixed keys (e.g. "estimate" + "thinking" -> "estimateThinking")
    // keep the dispatch single-line at the call sites while still landing in
    // typed top-level state slots.
    const prefix = stage;
    const camel = key.charAt(0).toUpperCase() + key.slice(1);
    set({ [`${prefix}${camel}`]: value } as Partial<AppState>);
  },

  setRegistrationViewMode: (mode: RegistrationViewMode) =>
    set({ registrationViewMode: mode }),

  setBorderSource: (source: BorderSource) => set({ borderSource: source }),

  runEstimatePosition: async () => {
    const {
      selectedBrainId, selectedSliceIndex, brains, atlasName,
      estimateModel, estimateEndpoint, estimateThinking,
      estimateTemperature, estimateMediaResolution,
      estimateMaxIterations, estimatePreprocess,
    } = get();

    const brain = brains.find((b) => b.id === selectedBrainId);
    if (!brain || selectedSliceIndex === null) {
      get().addLog("Select a slice first");
      return;
    }
    const slice = brain.slices[selectedSliceIndex];
    if (!slice) return;

    set({ pipelineRunning: true, pipelineStatus: "estimating", pipelineError: null });
    get().addLog(`Estimating AP position on ${slice.name}...`);

    set((s) => ({
      brains: s.brains.map((b) =>
        b.id === selectedBrainId
          ? { ...b, slices: b.slices.map((sl, i) => i === selectedSliceIndex ? { ...sl, status: "running" as const } : sl) }
          : b
      ),
    }));

    try {
      const apResult = await commands.runEstimate({
        imagePath: slice.path,
        atlas: atlasName,
        plane: brain.plane,
        model: estimateModel,
        thinking: estimateThinking,
        temperature: estimateTemperature,
        mediaResolution: estimateMediaResolution,
        maxIterations: estimateMaxIterations,
        preprocess: estimatePreprocess,
        provider: estimateEndpoint ? "openai" : "google",
        endpoint: estimateEndpoint,
      });

      const positionMm = apResult.position_mm as number;
      get().addLog(`AP estimated: ${positionMm.toFixed(3)} mm`);
      get().setApPosition(positionMm);

      // Estimation is an explicit commit — auto-lock so the user can proceed
      // straight to Run Registration without an extra click.
      set((s) => ({
        pipelineStatus: "complete",
        brains: s.brains.map((b) =>
          b.id === selectedBrainId
            ? {
                ...b,
                slices: b.slices.map((sl, i) =>
                  i === selectedSliceIndex
                    ? { ...sl, status: "done" as const, apMm: positionMm, apLocked: true }
                    : sl
                ),
              }
            : b
        ),
      }));
      // Kick off the affine preview alongside auto-lock so the 3D viewer
      // shows a warped slice without waiting on the full pipeline.
      void get().runQuickAffine();
    } catch (e) {
      const msg = e instanceof Error ? e.message : String(e);
      get().addLog(`Estimation error: ${msg}`);
      set({ pipelineStatus: "error", pipelineError: msg });
      set((s) => ({
        brains: s.brains.map((b) =>
          b.id === selectedBrainId
            ? { ...b, slices: b.slices.map((sl, i) => i === selectedSliceIndex ? { ...sl, status: "pending" as const } : sl) }
            : b
        ),
      }));
    } finally {
      set({ pipelineRunning: false });
    }
  },

  runRegistration: async () => {
    const {
      selectedBrainId, selectedSliceIndex, brains, atlasName,
      registerImageModel,
      registerThinking, registerTemperature, registerEndpoint,
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

    set({ pipelineRunning: true, pipelineStatus: "registering", pipelineError: null });
    get().addLog(`Running registration at ${slice.apMm.toFixed(3)} mm...`);

    try {
      const registrationResult = await commands.runRegister({
        imagePath: slice.path,
        positionMm: slice.apMm,
        atlas: atlasName,
        plane: brain.plane,
        imageModel: registerImageModel,
        thinking: registerThinking,
        temperature: registerTemperature,
        provider: registerEndpoint ? "openai" : "google",
        endpoint: registerEndpoint,
      });

      get().addLog("Image-gen registration complete");
      if (registrationResult.inverseWarpStatus !== null) {
        get().addLog(`Inverse warp: ${registrationResult.inverseWarpStatus}`);
      }

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
                    : sl
                ),
              }
            : b
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

  exportResults: async () => {
    const { selectedBrainId, selectedSliceIndex, brains, atlasName } = get();
    const brain = brains.find((b) => b.id === selectedBrainId);
    if (!brain || selectedSliceIndex === null) return;

    const slice = brain.slices[selectedSliceIndex];
    if (!slice || slice.apMm === undefined) {
      get().addLog("Run the agent first to get AP position");
      return;
    }

    try {
      const { open } = await import("@tauri-apps/plugin-dialog");
      const folder = await open({ directory: true, title: "Select export folder" });
      if (!folder) return;

      get().addLog(`Exporting to ${folder}...`);
      await commands.runExport({
        imagePath: slice.path,
        positionMm: slice.apMm,
        atlas: atlasName,
        plane: brain.plane,
        outputDir: folder as string,
      });
      get().addLog("Export complete");
    } catch (e) {
      const msg = e instanceof Error ? e.message : String(e);
      get().addLog(`Export error: ${msg}`);
    }
  },

  selectSlice: async (index: number) => {
    const { selectedBrainId, brains } = get();
    const brain = brains.find((b) => b.id === selectedBrainId);
    if (!brain || index < 0 || index >= brain.slices.length) return;

    set({ selectedSliceIndex: index, sliceImageLoading: true, selectedSliceImage: null });

    // Snap the AP slider to this slice's saved position so the slider and
    // the slice stay tied together. If the slice has no AP yet (pre-estimate),
    // leave the slider where it was — the next drag will write its position
    // back to the slice via setApPosition. The selectedSliceIndex is already
    // updated above so setApPosition's write-back targets this slice.
    const slice = brain.slices[index];
    if (slice.apMm !== undefined) {
      get().setApPosition(slice.apMm);
    }

    try {
      const result = await commands.loadSliceImage(brain.slices[index].path);
      // Only apply if still the same selection
      if (get().selectedSliceIndex === index) {
        set({ selectedSliceImage: result.image, sliceImageLoading: false });
      }
    } catch (e) {
      console.error("Failed to load slice image:", e);
      if (get().selectedSliceIndex === index) {
        set({ sliceImageLoading: false });
      }
    }

    // Also fetch composite for 2D views
    const { viewMode } = get();
    if (viewMode === "split") {
      get().buildComposite();
    }
  },

  fetchAtlasList: async () => {
    try {
      const atlases = await commands.listAtlases();
      set({ availableAtlases: atlases });
    } catch (e) {
      console.error("Failed to list atlases:", e);
    }
  },

  fetchAvailableAtlases: async () => {
    try {
      const atlases = await commands.listAvailableAtlases();
      set({ availableForDownload: atlases });
    } catch (e) {
      console.error("Failed to list available atlases:", e);
      get().addLog(
        `Failed to list available atlases: ${e instanceof Error ? e.message : String(e)}`,
      );
    }
  },

  openAtlasManager: (focus?: string | null) => {
    set({ atlasManagerOpen: true, atlasManagerFocus: focus ?? null });
    // Fire-and-forget refresh — modal renders with the current list and
    // updates once the new payload lands.
    get().fetchAvailableAtlases();
  },

  closeAtlasManager: () =>
    set({ atlasManagerOpen: false, atlasManagerFocus: null }),

  // Chat drawer ───────────────────────────────────────────────────────
  openChat: () => {
    set({ chatOpen: true });
    // Refresh the local-engine list on open so the connection picker is
    // never showing stale "not running" rows. Doesn't block the drawer
    // from rendering — the UI handles the in-flight state.
    if (get().localEngines.length === 0) {
      get().probeLocalEngines();
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
    // selected slice as a data URL so vision-capable models can see it.
    const sliceB64 = state.selectedSliceImage;
    const images: string[] = [];
    if (attachSlice && sliceB64) {
      images.push(`data:image/png;base64,${sliceB64}`);
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

  openLocalModels: () => {
    set({ localModelsOpen: true });
    // Fire-and-forget probe so the modal mounts with a fresh list.
    get().probeLocalEngines();
  },

  closeLocalModels: () => set({ localModelsOpen: false }),

  openApiKeys: () => set({ apiKeysOpen: true }),
  closeApiKeys: () => set({ apiKeysOpen: false }),

  probeLocalEngines: async () => {
    set({ localEnginesProbing: true });
    try {
      const builtIns = await commands.probeLocalEngines();
      // Re-probe each saved custom endpoint too so they show up alongside.
      const customResults = await Promise.all(
        get().customEndpoints.map((ep) =>
          commands.probeCustomLocalEndpoint(ep.label, ep.url).catch((e) => ({
            name: ep.label,
            port: 0,
            endpoint: ep.url,
            reachable: false,
            models: [],
            error: e instanceof Error ? e.message : String(e),
          } as LocalEngineStatus)),
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

  setEstimateModelChoice: (modelId: string, endpoint: string | null) =>
    set({ estimateModel: modelId, estimateEndpoint: endpoint }),

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
    // Locking commits the AP — kick off a fast affine preview so the slice
    // appears in the 3D viewer immediately, before the slow image-gen
    // pipeline runs. Fire-and-forget so the UI stays responsive.
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

  /** Affine-only Elastix preview. Triggered automatically by Lock Position
   * (and by Estimate auto-lock). Stale-result guard: each call captures the
   * current (sliceIndex, apMm) snapshot; the result is discarded if either
   * has changed when the subprocess returns. */
  runQuickAffine: async () => {
    const state = get();
    const { selectedBrainId, selectedSliceIndex, atlasName, currentApMm } = state;
    if (!selectedBrainId || selectedSliceIndex === null || !atlasName) return;
    const brain = state.brains.find((b) => b.id === selectedBrainId);
    if (!brain) return;
    const slice = brain.slices[selectedSliceIndex];
    if (!slice) return;

    const sliceIndexSnapshot = selectedSliceIndex;
    const apSnapshot = currentApMm;
    const brainIdSnapshot = selectedBrainId;

    // Mark running on the snapshotted slice. Using a setter so concurrent
    // mutations to other slices don't clobber this one.
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
      const result = await commands.runQuickAffine({
        imagePath: slice.path,
        positionMm: apSnapshot,
        atlas: atlasName,
        plane: brain.plane,
      });

      // Stale-result guard: bail if the user changed slice or AP while we
      // were in-flight. The output file still exists on disk; we just
      // don't wire it into state. Next lock will overwrite.
      const now = get();
      if (
        now.selectedSliceIndex !== sliceIndexSnapshot ||
        now.selectedBrainId !== brainIdSnapshot ||
        Math.abs(now.currentApMm - apSnapshot) > 0.01
      ) {
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
                        quickAffineWarpedPath: result.warpedSlicePath,
                        quickAffineError: null,
                      }
                    : sl,
                ),
              },
        ),
      }));
      get().addLog(`Quick affine preview ready (${result.elapsedS}s)`);
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

  downloadAtlas: async (name: string) => {
    // Subscribe to per-atlas progress events for the lifetime of this call.
    // The Rust side emits with `phase: complete` when extraction is done.
    const { listen } = await import("@tauri-apps/api/event");
    const unlisten = await listen<AtlasDownloadProgress>(
      "atlas-download-progress",
      (event) => {
        const payload = event.payload;
        if (payload.atlas !== name) return;
        set((s) => ({
          atlasDownloadProgress: {
            ...s.atlasDownloadProgress,
            [name]: payload,
          },
        }));
      },
    );

    try {
      set((s) => ({
        atlasDownloadProgress: {
          ...s.atlasDownloadProgress,
          [name]: { atlas: name, phase: "resolving" },
        },
      }));
      get().addLog(`Downloading atlas "${name}"...`);
      await commands.downloadAtlas(name);
      get().addLog(`Atlas "${name}" downloaded.`);

      // Refresh both lists — installed shortcut for the selector and the
      // full catalog for the manager modal — so the row flips to Installed.
      await Promise.all([
        get().fetchAtlasList(),
        get().fetchAvailableAtlases(),
      ]);
    } catch (e) {
      const msg = e instanceof Error ? e.message : String(e);
      get().addLog(`Atlas download failed: ${msg}`);
      throw e;
    } finally {
      unlisten();
      // Leave the final progress payload in state so the row can show
      // "Installed" without flicker; consumers can read phase==="complete".
    }
  },

  loadAtlas: async (name: string) => {
    set({
      atlasLoading: true,
      atlasLoadProgress: { phase: "Starting...", percent: 0 },
      pipelineStatus: "loading_atlas",
      pipelineError: null,
    });

    // Subscribe to phase-boundary progress events emitted by the Rust loader.
    // The event-listener import is dynamic so this module doesn't pay for it
    // outside an atlas-load cycle.
    const { listen } = await import("@tauri-apps/api/event");
    const unlisten = await listen<AtlasLoadProgress>(
      "atlas-load-progress",
      (event) => set({ atlasLoadProgress: event.payload }),
    );
    try {
      await commands.loadAtlas(name);
      // Rust-side phases done; show a near-complete bar while we decode the
      // border + reference + additional volumes on the frontend.
      set({ atlasLoadProgress: { phase: "Caching volumes...", percent: 95 } });
      const info = await commands.getAtlasInfo();
      const midAp = (info.ap_min_mm + info.ap_max_mm) / 2;
      set({ atlasName: name, atlasInfo: info, currentApMm: midAp });
      get().addLog(`Atlas loaded: ${info.name} (${info.ap_count} slices)`);

      // Fetch mesh, border volume, and reference volumes in parallel
      get().addLog("Loading mesh + volumes...");
      const [, borderResult, volResult] = await Promise.all([
        get().fetchBrainMesh(),
        commands.getBorderVolume(),
        commands.getAllVolumes(),
      ]);

      // Decode bit-packed border volume
      const packed = b64ToBytes(borderResult.data);
      const borderVolume: BorderVolume = {
        packed,
        depth: borderResult.depth,
        height: borderResult.height,
        width: borderResult.width,
      };

      // Decode reference volume
      const refVol = b64ToBytes(volResult.reference);

      // Decode additional volumes
      const additionalVolumes: Record<string, Uint8Array> = {};
      for (const [volName, b64] of Object.entries(volResult.additional)) {
        additionalVolumes[volName] = b64ToBytes(b64 as string);
      }

      set({
        borderVolume,
        referenceVolume: refVol,
        additionalVolumes,
        additionalVolumeNames: volResult.additionalNames,
        volumesLoaded: true,
      });

      get().addLog(
        `Volumes cached: reference + ${volResult.additionalNames.length} additional ` +
        `(${((packed.length + refVol.length) / 1024 / 1024).toFixed(1)} MB)`
      );

      // Extract initial border slice for 3D view
      const midIdx = Math.round(
        (midAp / (info.ap_max_mm - info.ap_min_mm)) * (borderResult.depth - 1)
      );
      const pixels = unpackBorderSlice(packed, midIdx, borderResult.width, borderResult.height);
      set({
        currentBorderPixels: pixels,
        borderWidth: borderResult.width,
        borderHeight: borderResult.height,
        pipelineStatus: "idle",
      });
    } catch (e) {
      const msg = e instanceof Error ? e.message : String(e);
      set({ pipelineStatus: "error", pipelineError: msg });
      get().addLog(`Error loading atlas: ${msg}`);
    } finally {
      unlisten();
      set({ atlasLoading: false, atlasLoadProgress: null });
    }
  },

  // Pure local operation — no IPC, no debounce needed.
  //
  // When a slice is selected, the slider acts as the slice's AP-position
  // editor: every drag writes back to that slice's `apMm` so the slider and
  // the slice stay in lock-step. Without a selected slice, this is a plain
  // atlas-navigation control.
  setApPosition: (apMm: number) => {
    const { borderVolume, atlasInfo, selectedBrainId, selectedSliceIndex } = get();
    if (!borderVolume || !atlasInfo) {
      set({ currentApMm: apMm });
      return;
    }

    const apRange = atlasInfo.ap_max_mm - atlasInfo.ap_min_mm;
    const fraction = (apMm - atlasInfo.ap_min_mm) / apRange;
    const idx = Math.round(fraction * (borderVolume.depth - 1));
    const clamped = Math.max(0, Math.min(borderVolume.depth - 1, idx));

    const pixels = unpackBorderSlice(
      borderVolume.packed,
      clamped,
      borderVolume.width,
      borderVolume.height,
    );

    set((s) => {
      const next: Partial<AppState> = {
        currentApMm: apMm,
        currentBorderPixels: pixels,
      };
      // Write the AP back to the selected slice so the list, badges, and
      // pipeline all see the user-edited position. Any movement releases
      // the lock — the user must re-commit before registration can run.
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

    // Rebuild composite locally if in 2D view
    get().buildComposite();
  },

  fetchBrainMesh: async () => {
    try {
      const mesh = await commands.getBrainMesh();
      set({ brainMesh: mesh });
      get().addLog(`Brain mesh loaded: ${mesh.positions.length / 3} vertices`);
    } catch (e) {
      const msg = e instanceof Error ? e.message : String(e);
      console.error("Failed to fetch brain mesh:", e);
      get().addLog(`Brain mesh unavailable: ${msg}`);
    }
  },

  // Pure local: build composite from cached volumes + channel/border settings
  buildComposite: () => {
    const { referenceVolume, additionalVolumes, borderVolume, atlasInfo,
            currentApMm, viewMode, activeChannel, showBorders } = get();
    if (!borderVolume || !atlasInfo) return;
    if (viewMode !== "split") return;

    // Pick the active volume
    const vol = activeChannel === "reference"
      ? referenceVolume
      : additionalVolumes[activeChannel] ?? null;

    const { width, height, depth } = borderVolume;
    const apRange = atlasInfo.ap_max_mm - atlasInfo.ap_min_mm;
    const fraction = (currentApMm - atlasInfo.ap_min_mm) / apRange;
    const idx = Math.max(0, Math.min(depth - 1, Math.round(fraction * (depth - 1))));
    const pixelsPerSlice = width * height;
    const volOffset = idx * pixelsPerSlice;

    const canvas = document.createElement("canvas");
    canvas.width = width;
    canvas.height = height;
    const ctx = canvas.getContext("2d")!;
    const imageData = ctx.createImageData(width, height);
    const rgba = imageData.data;

    for (let i = 0; i < pixelsPerSlice; i++) {
      const gray = vol ? vol[volOffset + i] : 0;

      // Check border
      let isBorder = false;
      if (showBorders) {
        const globalBit = idx * pixelsPerSlice + i;
        isBorder = !!((borderVolume.packed[globalBit >> 3] >> (globalBit & 7)) & 1);
      }

      const j = i * 4;
      if (isBorder) {
        rgba[j] = Math.round(gray * 0.6);
        rgba[j + 1] = Math.round(gray * 0.6 + 255 * 0.4);
        rgba[j + 2] = Math.round(gray * 0.6 + 100 * 0.4);
      } else if (vol) {
        rgba[j] = gray;
        rgba[j + 1] = gray;
        rgba[j + 2] = gray;
      }
      rgba[j + 3] = vol || isBorder ? 255 : 0;
    }

    ctx.putImageData(imageData, 0, 0);
    set({ currentCompositeDataUrl: canvas.toDataURL("image/png") });
  },

  setViewMode: (mode: ViewMode) => {
    set({ viewMode: mode });
    if (mode === "split") {
      get().buildComposite();
    }
  },

  setAtlasOpacity: (opacity: number) =>
    set({ atlasOpacity: Math.max(0, Math.min(1, opacity)) }),

  setShowBorders: (show: boolean) => {
    set({ showBorders: show });
    get().buildComposite();
  },

  setActiveChannel: (channel: string) => {
    set({ activeChannel: channel });
    get().buildComposite();
  },

  addLog: (msg: string) =>
    set((s) => {
      // Cap log at 500 entries — the panel re-renders a <div> per entry so
      // unbounded growth tanks scroll perf on long pipelines.
      const next = [...s.logs, `[${new Date().toLocaleTimeString()}] ${msg}`];
      return { logs: next.length > 500 ? next.slice(next.length - 500) : next };
    }),
}));
