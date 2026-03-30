import { create } from "zustand";
import type {
  AtlasInfo,
  Brain,
  MeshData,
  NavigationView,
  SliceInfo,
  SliceResult,
  ViewMode,
  PipelineStatus,
} from "../lib/types";
import * as commands from "../lib/commands";

let sliceDebounceTimer: ReturnType<typeof setTimeout> | null = null;
let sliceRequestId = 0;

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
  brainMesh: MeshData | null;

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

  // Agent settings (mirrors CLI params)
  agentModel: string;
  agentWorkflow: string;
  agentThinking: string;
  agentTemperature: number;
  agentLandmarks: number;
  agentVlmResolution: number;
  agentMaxIterations: number;

  // Pipeline
  pipelineStatus: PipelineStatus;
  pipelineError: string | null;
  pipelineRunning: boolean;
  logs: string[];

  // View
  viewMode: ViewMode;

  // Actions — navigation
  navigateToDashboard: () => void;
  navigateToBrain: (brainId: string) => void;

  // Actions — brains
  setHoveredBrain: (id: string | null) => void;
  addEmptyBrains: (count: number) => void;
  addBrainFromFolder: (folderPath: string) => Promise<void>;
  loadBrainImages: (brainId: string, folderPath: string) => Promise<void>;
  removeBrain: (brainId: string) => void;

  // Actions — agent settings
  setAgentSetting: (key: string, value: string | number) => void;

  // Actions — pipeline
  runPipeline: () => Promise<void>;
  exportResults: () => Promise<void>;

  // Actions — slices
  selectSlice: (index: number) => Promise<void>;

  // Actions — atlas
  fetchAtlasList: () => Promise<void>;
  loadAtlas: (name: string) => Promise<void>;
  setApPosition: (apMm: number) => void;
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
  brainMesh: null,

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

  // Agent settings defaults (match CLI defaults)
  agentModel: "gemini-2.5-flash-preview-04-17",
  agentWorkflow: "auto",
  agentThinking: "HIGH",
  agentTemperature: 1.0,
  agentLandmarks: 14,
  agentVlmResolution: 2048,
  agentMaxIterations: 20,

  pipelineStatus: "idle",
  pipelineError: null,
  pipelineRunning: false,
  logs: [],

  viewMode: "3d",

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
      };

      set((s) => ({ brains: [...s.brains, brain] }));
      get().addLog(`Added brain "${folderName}": ${slices.length} slices`);
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
    } catch (e) {
      const msg = e instanceof Error ? e.message : String(e);
      get().addLog(`Error loading images: ${msg}`);
    }
  },

  removeBrain: (brainId: string) =>
    set((s) => ({ brains: s.brains.filter((b) => b.id !== brainId) })),

  setAgentSetting: (key: string, value: string | number) => {
    set({ [key]: value } as Partial<AppState>);
  },

  runPipeline: async () => {
    const { selectedBrainId, selectedSliceIndex, brains, atlasName,
            agentModel, agentWorkflow, agentThinking, agentTemperature,
            agentLandmarks, agentVlmResolution, agentMaxIterations } = get();

    const brain = brains.find((b) => b.id === selectedBrainId);
    if (!brain || selectedSliceIndex === null) {
      get().addLog("Select a slice first");
      return;
    }
    const slice = brain.slices[selectedSliceIndex];
    if (!slice) return;

    set({ pipelineRunning: true, pipelineStatus: "estimating", pipelineError: null });
    get().addLog(`Running AP estimation on ${slice.name}...`);

    // Update slice status
    set((s) => ({
      brains: s.brains.map((b) =>
        b.id === selectedBrainId
          ? { ...b, slices: b.slices.map((sl, i) => i === selectedSliceIndex ? { ...sl, status: "running" as const } : sl) }
          : b
      ),
    }));

    try {
      // Step 1: AP estimation
      const apResult = await commands.runEstimate({
        imagePath: slice.path,
        atlas: atlasName,
        model: agentModel,
        thinking: agentThinking,
        temperature: agentTemperature,
        vlmResolution: agentVlmResolution,
        maxIterations: agentMaxIterations,
        workflow: agentWorkflow === "auto" ? "auto" : agentWorkflow,
      });

      const positionMm = apResult.position_mm as number;
      get().addLog(`AP estimated: ${positionMm.toFixed(3)} mm`);
      get().setApPosition(positionMm);
      set({ pipelineStatus: "registering" });

      // Step 2: Registration
      get().addLog("Running registration...");
      const regResult = await commands.runRegister({
        imagePath: slice.path,
        positionMm,
        atlas: atlasName,
        model: agentModel,
        thinking: agentThinking,
        temperature: agentTemperature,
        landmarks: agentLandmarks,
        vlmResolution: agentVlmResolution,
        workflow: agentWorkflow === "auto" ? "auto" : agentWorkflow,
      });

      get().addLog(`Registration complete: ${(regResult.accepted_correspondences as unknown[])?.length ?? 0} correspondences`);

      // Update slice status to done with AP position
      set((s) => ({
        pipelineStatus: "complete",
        brains: s.brains.map((b) =>
          b.id === selectedBrainId
            ? {
                ...b,
                completedCount: b.completedCount + 1,
                slices: b.slices.map((sl, i) =>
                  i === selectedSliceIndex ? { ...sl, status: "done" as const, apMm: positionMm } : sl
                ),
              }
            : b
        ),
      }));
    } catch (e) {
      const msg = e instanceof Error ? e.message : String(e);
      get().addLog(`Pipeline error: ${msg}`);
      set({ pipelineStatus: "error", pipelineError: msg });

      // Reset slice status
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

  exportResults: async () => {
    const { selectedBrainId, selectedSliceIndex, brains, atlasName, currentApMm } = get();
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
    const { viewMode, currentApMm } = get();
    if (viewMode === "split" || viewMode === "overlay") {
      get().fetchComposite(currentApMm);
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

  loadAtlas: async (name: string) => {
    set({ atlasLoading: true, pipelineStatus: "loading_atlas", pipelineError: null });
    try {
      await commands.loadAtlas(name);
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
      set({ atlasLoading: false });
    }
  },

  // Pure local operation — no IPC, no debounce needed
  setApPosition: (apMm: number) => {
    const { borderVolume, atlasInfo } = get();
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

    set({
      currentApMm: apMm,
      currentBorderPixels: pixels,
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
      console.error("Failed to fetch brain mesh:", e);
    }
  },

  // Pure local: build composite from cached volumes + channel/border settings
  buildComposite: () => {
    const { referenceVolume, additionalVolumes, borderVolume, atlasInfo,
            currentApMm, viewMode, activeChannel, showBorders } = get();
    if (!borderVolume || !atlasInfo) return;
    if (viewMode !== "split" && viewMode !== "overlay") return;

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
    if (mode === "split" || mode === "overlay") {
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
    set((s) => ({ logs: [...s.logs, `[${new Date().toLocaleTimeString()}] ${msg}`] })),
}));
