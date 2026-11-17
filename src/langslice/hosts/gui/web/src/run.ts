// The run as the window knows it, learned by polling the local server: the job list every
// 2 s, the shown job's snapshot every 300 ms (its new items after a cursor, its tables when
// the state moved), its strip (the axis) when the sections change. One store, read by the
// components through `useRun`.
import { useSyncExternalStore } from 'react';

export type Row = {
  id: string;
  order?: number;
  position_mm: number | null;
  render_key?: string | null;
  transform?: string | null;
  deformation?: string | null;
  damaged?: boolean;
};
export type Call = {
  key: string; name: string; title: string; sections: string[];
  running: boolean; ended: boolean; failed: boolean;
};
export type Words = { kind: 'agent' | 'reasoning'; text: string; at: string; n: number };
export type Axis = { low: string; high: string; min_mm: number; max_mm: number };
export type Job = { index: number; name: string; path: string; changed_at?: number; status?: string };
export type Comment = { id: string; section: string; box: number[]; text: string; revised: boolean };

export type RunState = {
  connected: boolean;
  jobs: Job[] | null;
  job: number | null;
  status: string;            // Running, Submitted, Waiting, Idle, Failed
  sentence: string;
  model: string;
  rows: Row[];
  loaded: boolean;
  focus: string[] | null;    // the agent's focus: its latest followed call's sections
  focusCall: string | null;
  calls: Call[];             // in order
  words: Words | null;       // the agent's latest words
  axis: Axis | null;
  comments: Comment[];
};

const SNAPSHOT_MS = 300, JOBS_MS = 2000, STRIP_MS = 1500;

type Item = {
  kind: string; key?: string; phase?: string; name?: string; title?: string;
  sections?: string[]; failed?: boolean; text?: string; at?: string;
};

let state: RunState = {
  connected: false, jobs: null, job: null, status: '', sentence: '', model: '',
  rows: [], loaded: false, focus: null, focusCall: null, calls: [], words: null, axis: null, comments: [],
};
const listeners = new Set<() => void>();
function set(patch: Partial<RunState>) {
  state = { ...state, ...patch };
  for (const l of listeners) l();
}
export function useRun(): RunState {
  return useSyncExternalStore(cb => { listeners.add(cb); return () => listeners.delete(cb); }, () => state);
}

let watch = { cursor: 0, gen: 0, sv: -1 };
let said = 0;
let named: number | null = (() => {
  const n = Number(new URLSearchParams(location.hash.slice(1)).get('job'));
  return Number.isInteger(n) && location.hash.includes('job=') ? n : null;
})();
let chosen = false;

export function chooseJob(index: number) {
  chosen = true;
  select(index);
}
function select(index: number) {
  if (state.job === index) return;
  watch = { cursor: 0, gen: 0, sv: -1 };
  set({
    job: index, status: '', sentence: '', model: '', rows: [], loaded: false, focus: null,
    focusCall: null, calls: [], words: null, axis: null, comments: [],
  });
  const params = new URLSearchParams(location.hash.slice(1));
  params.set('job', String(index));
  history.replaceState(null, '', '#' + params.toString());
  lastStrip = 0;
  fetchStrip();
}
function pickJob(jobs: Job[]) {
  if (!jobs.length) return;
  const known = (i: number) => jobs.some(j => j.index === i);
  if (state.job === null && named !== null && known(named)) { chosen = true; select(named); return; }
  named = null;
  if (state.job !== null && known(state.job) && chosen) return;
  const changed = (j?: Job) => Number(j?.changed_at) || 0;
  const newest = jobs.reduce((a, b) => (changed(b) > changed(a) ? b : a));
  const shown = jobs.find(j => j.index === state.job);
  if (!shown || (newest.index !== state.job && changed(newest) > changed(shown) + 5)) select(newest.index);
}

async function pollJobs() {
  try {
    const r = await fetch('jobs', { cache: 'no-store' });
    if (r.ok) {
      const answer = await r.json();
      const jobs: Job[] = answer.jobs || [];
      set({ jobs });
      pickJob(jobs);
    }
  } catch { /* the snapshot poll reports the connection */ }
  setTimeout(pollJobs, JOBS_MS);
}

async function pollSnapshot() {
  const job = state.job;
  try {
    if (job !== null) {
      const r = await fetch(`jobs/${job}/snapshot?since=${watch.cursor}&gen=${watch.gen}&sv=${watch.sv}`, { cache: 'no-store' });
      if (!r.ok) throw new Error(String(r.status));
      const snap = await r.json();
      if (job === state.job) take(snap);
    }
    if (!state.connected) set({ connected: true });
  } catch {
    if (state.connected) set({ connected: false });
  }
  setTimeout(pollSnapshot, SNAPSHOT_MS);
}

function take(snap: any) {
  const patch: Partial<RunState> = {};
  let calls = state.calls;
  if (snap.reset && watch.gen !== 0) { calls = []; patch.words = null; }
  watch.gen = snap.generation;
  watch.cursor = snap.cursor;
  const items: Item[] = snap.items || [];
  if (items.length) {
    const byKey = new Map(calls.map(c => [c.key, c]));
    let changed = false;
    for (const item of items) {
      if (item.kind === 'tool' && item.key) {
        const old = byKey.get(item.key);
        const call: Call = old ? { ...old } : {
          key: item.key, name: item.name || '', title: item.title || item.name || '',
          sections: item.sections || [], running: false, ended: false, failed: false,
        };
        // A call's end may come in the same round as its start, or before it.
        if (item.phase === 'start') call.running = !call.ended;
        else { call.running = false; call.ended = true; call.failed = Boolean(item.failed); }
        byKey.set(item.key, call);
        if (!old) calls = [...calls, call];
        else calls = calls.map(c => (c.key === call.key ? call : c));
        changed = true;
      } else if ((item.kind === 'agent' || item.kind === 'reasoning') && item.text) {
        said += 1;
        patch.words = { kind: item.kind, text: firstParagraph(item.text), at: item.at || '', n: said };
      }
    }
    if (changed) patch.calls = calls.slice(-400);
  }
  if (snap.sections !== undefined) {
    watch.sv = snap.state_version;
    patch.rows = Array.isArray(snap.sections) ? snap.sections : [];
    patch.loaded = true;
    fetchStrip();
  }
  const focus = snap.focus ? (snap.focus.sections || []) : null;
  if (JSON.stringify(focus) !== JSON.stringify(state.focus) || (snap.focus?.call ?? null) !== state.focusCall) {
    patch.focus = focus;
    patch.focusCall = snap.focus?.call ?? null;
  }
  const word = snap.status?.word || '';
  if (word !== state.status) patch.status = word;
  const sentence = snap.status?.sentence || '';
  if (sentence !== state.sentence) patch.sentence = sentence;
  const model = snap.header?.model || '';
  if (model !== state.model) patch.model = model;
  const comments = snap.comments || [];
  if (JSON.stringify(comments) !== JSON.stringify(state.comments)) patch.comments = comments;
  if (Object.keys(patch).length) set(patch);
}

// The words the bubble says: an entry's first paragraph, without markdown marks.
function firstParagraph(text: string): string {
  const block = text.split(/\n\s*\n/).find(b => b.trim()) || '';
  return block.split('\n').map(l => l.replace(/^#{1,6}\s+|^\s*([-*]|\d+\.)\s+/, '')).join(' ')
    .replace(/`([^`]+)`/g, '$1').replace(/\*\*([^*]+)\*\*/g, '$1').trim();
}

let lastStrip = 0, stripTimer = 0, stripBusy = false;
async function fetchStrip() {
  const job = state.job;
  if (job === null || stripBusy) return;
  const wait = lastStrip + STRIP_MS - Date.now();
  if (wait > 0) {
    if (!stripTimer) stripTimer = window.setTimeout(() => { stripTimer = 0; fetchStrip(); }, wait);
    return;
  }
  stripBusy = true;
  lastStrip = Date.now();
  try {
    const r = await fetch(`jobs/${job}/strip`, { cache: 'no-store' });
    if (r.ok && job === state.job) {
      const info = await r.json();
      const a = info?.axis;
      if (a && JSON.stringify(a) !== JSON.stringify(state.axis)) set({ axis: a });
    }
  } catch { /* asked again when the sections change */ }
  finally { stripBusy = false; }
}

let started = false;
export function start() {
  if (started) return;
  started = true;
  pollJobs();
  pollSnapshot();
}
