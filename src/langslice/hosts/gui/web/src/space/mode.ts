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
const listeners = new Set<() => void>();

export function setStageMode(next: StageMode) {
  mode = next;
  try { localStorage.setItem(KEY, next); } catch { /* the choice lasts this window only */ }
  for (const l of listeners) l();
}

export function useStageMode(): StageMode {
  return useSyncExternalStore(cb => { listeners.add(cb); return () => { listeners.delete(cb); }; }, () => mode);
}
