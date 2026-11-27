// Which way the viewer shows where the sections are, in both views: '2d' (the pictures with
// the cutting axis under them) or '3d' (the pictures in a column beside the atlas brain in
// three dimensions, which takes the cutting axis's place). Kept in `localStorage`.
import { useSyncExternalStore } from 'react';

export type StageMode = '2d' | '3d';
const KEY = 'langslice.stage';

function read(): StageMode {
  try { return localStorage.getItem(KEY) === '3d' ? '3d' : '2d'; } catch { return '2d'; }
}

let mode: StageMode = read();
let switchedAt = 0;
const SWITCH_MS = 400;
const listeners = new Set<() => void>();

export function setStageMode(next: StageMode) {
  if (next !== mode) switchedAt = Date.now();
  mode = next;
  try { localStorage.setItem(KEY, next); } catch { /* the choice lasts this window only */ }
  for (const l of listeners) l();
}

// Whether the person switched 2D | 3D a moment ago: then the stage that comes in fades in
// (it is not animated when the window opens on it).
export function justSwitched(): boolean {
  return Date.now() - switchedAt < SWITCH_MS;
}

// The entrance of a stage the person just switched to: a short crossfade with a little
// blur, so the swap reads as one change rather than two pictures; opacity only with reduced
// motion. ease-out, under 300 ms (it answers a person).
export const STAGE_IN = { opacity: 0, filter: 'blur(4px)' } as const;
export const STAGE_SHOWN = { opacity: 1, filter: 'blur(0px)' } as const;

export function useStageMode(): StageMode {
  return useSyncExternalStore(cb => { listeners.add(cb); return () => { listeners.delete(cb); }; }, () => mode);
}
