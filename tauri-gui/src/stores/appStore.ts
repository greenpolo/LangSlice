import { create } from "zustand";
import type {
  AtlasInfo,
  Brain,
  MeshData,
  NavigationView,
  SliceInfo,
  ViewMode,
  PipelineStatus,
} from "../lib/types";
import * as commands from "../lib/commands";

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

  // Pipeline
  pipelineStatus: PipelineStatus;
  pipelineError: string | null;
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

  // Actions — atlas
  fetchAtlasList: () => Promise<void>;
  loadAtlas: (name: string) => Promise<void>;
  setApPosition: (apMm: number) => void;
  fetchBrainMesh: () => Promise<void>;
  setViewMode: (mode: ViewMode) => void;
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

  pipelineStatus: "idle",
  pipelineError: null,
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

      // Fetch mesh and border volume in parallel
      get().addLog("Loading brain mesh + border volume...");
      const [, borderResult] = await Promise.all([
        get().fetchBrainMesh(),
        commands.getBorderVolume(),
      ]);

      // Decode bit-packed border volume into frontend memory
      const packed = b64ToBytes(borderResult.data);
      const borderVolume: BorderVolume = {
        packed,
        depth: borderResult.depth,
        height: borderResult.height,
        width: borderResult.width,
      };
      set({ borderVolume });
      get().addLog(
        `Border volume cached: ${borderResult.depth} slices, ` +
        `${(packed.length / 1024 / 1024).toFixed(1)} MB packed`
      );

      // Extract initial slice
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

  setViewMode: (mode: ViewMode) => set({ viewMode: mode }),

  addLog: (msg: string) =>
    set((s) => ({ logs: [...s.logs, `[${new Date().toLocaleTimeString()}] ${msg}`] })),
}));
