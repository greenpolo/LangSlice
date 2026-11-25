// The run as the window knows it, learned by polling the local server: the job list every
// 2 s, the shown job's snapshot every 300 ms (its new items after a cursor, its tables when
// the state moved), its strip (the axis) when the sections change. One store, read by the
// components through `useRun(select)`, each re-drawn only when what it selects changes.
// The run's log keeps every entry in order (the agent's words and reasoning, its calls,
// the pictures it looked at, notes) for the Extended view's journal.
import { useSyncExternalStore } from 'react';

export type Row = {
  id: string;
  order?: number;
  position_mm: number | null;
  // Who set the position: 'initial' (the job's start and spacing), 'input' or 'written'.
  position_source?: string | null;
  render_key?: string | null;
  transform?: string | null;
  deformation?: string | null;
  damaged?: boolean;
};
// A section's name as the window labels it: its file name without the extension.
export const short = (id: string) => id.replace(/\.[A-Za-z0-9]{2,5}$/, '');

export type Call = {
  key: string; name: string; title: string; sections: string[];
  running: boolean; ended: boolean; failed: boolean;
  args?: unknown; step?: number | null; door?: string | null; at?: string; status?: string | null;
  pictures: number[];        // the pictures the agent looked at in this call
  seen?: string[];           // the sections its pictures show, for a call that named none
  zooms?: Record<string, number[]>; // a zoom's window on each section, fractions x0 y0 x1 y1
  endedAt?: number;          // when the window saw it end (Date.now()); 0 for one already over when it opened
};
export type Words = { kind: 'agent' | 'reasoning'; text: string; at: string; n: number };
export type Axis = { low: string; high: string; min_mm: number; max_mm: number };
// The strip: the atlas thumbnails' places along the axis and each section's thumbnail key.
export type Strip = {
  step_mm: number;
  atlas: { k: number; position_mm: number }[];
  angles: string;
  sections: { id: string; order: number; position_mm: number; position_source: string; thumb_key: string }[];
  ground: [number, number, number];   // the slides' own glass, which the window takes for its ground
};
export type Job = { index: number; name: string; path: string; changed_at?: number; status?: string };
// A review comment: its box as [x0, y0, x1, y1] fractions of the section's picture as
// the window draws it (the overlay's content, oriented as the section is kept).
export type Comment = { id: string; section: string; box: number[]; text: string; revised: boolean };
export type Entry =
  | { kind: 'words'; n: number; words: Words; full: string }
  | { kind: 'call'; key: string }
  | { kind: 'picture'; number: number; call?: string; caption?: string; sections?: string[]; tool?: string; title?: string }
  | { kind: 'note'; text: string; error?: boolean };

export type RunState = {
  connected: boolean;
  jobs: Job[] | null;
  home: string;              // the home folder, which the window writes ~ in paths
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
  strip: Strip | null;
  comments: Comment[];
  log: Entry[];              // the run so far, in order (at most LOG_MAX entries)
  header: Record<string, unknown>;
};

const SNAPSHOT_MS = 300, JOBS_MS = 2000, STRIP_MS = 1500, LOG_MAX = 4000;

type Item = {
  kind: string; key?: string; phase?: string; name?: string; title?: string;
  sections?: string[]; failed?: boolean; text?: string; at?: string;
  number?: number; call?: string; caption?: string; tool?: string; error?: boolean;
  args?: unknown; step?: number | null; door?: string | null; status?: string | null;
  views?: string[]; path?: string; zoom?: number[] | null;
};

// What a picture adds to the call that drew it: its number, its sections when the call
// named none (a zoom names a picture, not sections), and the window a zoom shows of each.
function seen(call: Call, shown: { number: number; sections: string[]; zoom: number[] | null }): Partial<Call> {
  const sections = call.sections.length ? call.sections : [...new Set([...(call.seen ?? []), ...shown.sections])];
  const zooms = shown.zoom ? { ...call.zooms, ...Object.fromEntries(shown.sections.map(id => [id, shown.zoom!])) } : call.zooms;
  return { pictures: [...call.pictures, shown.number], seen: sections, zooms };
}

let state: RunState = {
  connected: false, jobs: null, home: '', job: null, status: '', sentence: '', model: '',
  rows: [], loaded: false, focus: null, focusCall: null, calls: [], words: null, axis: null, strip: null, comments: [],
  log: [], header: {},
};
const listeners = new Set<() => void>();
function set(patch: Partial<RunState>) {
  state = { ...state, ...patch };
  for (const l of listeners) l();
}
const subscribe = (cb: () => void) => { listeners.add(cb); return () => { listeners.delete(cb); }; };
// The run, or the part of it *select* picks (return a field of the state, or a value
// that is the same when nothing it reads changed).
export function useRun(): RunState;
export function useRun<T>(select: (s: RunState) => T): T;
export function useRun<T>(select?: (s: RunState) => T) {
  return useSyncExternalStore(subscribe, () => (select ? select(state) : state));
}

let watch = { cursor: 0, gen: 0, sv: -1 };
// A view folder a call's end names -> that call; a picture no call has named yet.
let claims = new Map<string, string>();
type Loose = { number: number; sections: string[]; zoom: number[] | null };
let loose = new Map<string, Loose>();
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
  claims = new Map(); loose = new Map();
  set({
    job: index, status: '', sentence: '', model: '', rows: [], loaded: false, focus: null,
    focusCall: null, calls: [], words: null, axis: null, strip: null, comments: [], log: [], header: {},
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
      const home = String(answer.home || '');
      set(home === state.home ? { jobs } : { jobs, home });
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
  let log = state.log;
  if (snap.reset && watch.gen !== 0) { calls = []; log = []; patch.words = null; }
  const added: Entry[] = [];
  const catchingUp = watch.cursor === 0;
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
          sections: item.sections || [], running: false, ended: false, failed: false, pictures: [],
        };
        if (item.args !== undefined && item.args !== null) call.args = item.args;
        if (item.step !== undefined) call.step = item.step;
        if (item.door) call.door = item.door;
        if (item.status) call.status = item.status;
        if (item.phase === 'start' && item.at) call.at = item.at;
        // The pictures a call's end names are its own, whether they came before or after.
        for (const path of item.views || []) {
          claims.set(path, call.key);
          const held = loose.get(path);
          if (held && !call.pictures.includes(held.number)) { Object.assign(call, seen(call, held)); loose.delete(path); }
        }
        // A call's end may come in the same round as its start, or before it.
        if (item.phase === 'start') call.running = !call.ended;
        else { call.running = false; call.ended = true; call.failed = Boolean(item.failed); call.endedAt ??= catchingUp ? 0 : Date.now(); }
        byKey.set(item.key, call);
        if (!old) { calls = [...calls, call]; added.push({ kind: 'call', key: call.key }); }
        else calls = calls.map(c => (c.key === call.key ? call : c));
        changed = true;
      } else if ((item.kind === 'agent' || item.kind === 'reasoning') && item.text) {
        said += 1;
        const words: Words = { kind: item.kind, text: firstParagraph(item.text), at: item.at || '', n: said };
        patch.words = words;
        added.push({ kind: 'words', n: said, words, full: item.text });
      } else if (item.kind === 'picture' && typeof item.number === 'number') {
        const key = item.path ? claims.get(item.path) : undefined;
        const owner = key ? calls.find(c => c.key === key) : undefined;
        const shown: Loose = { number: item.number, sections: item.sections || [], zoom: item.zoom ?? null };
        if (!owner && item.path) loose.set(item.path, shown);
        if (owner) {
          const updated = { ...owner, ...seen(owner, shown) };
          calls = calls.map(c => (c === owner ? updated : c));
          changed = true;
        }
        added.push({ kind: 'picture', number: item.number, call: item.call, caption: item.caption, sections: item.sections, tool: item.tool, title: item.title });
      } else if (item.kind === 'note' && item.text) {
        added.push({ kind: 'note', text: item.text, error: item.error });
      }
    }
    if (changed) patch.calls = calls.slice(-LOG_MAX);
  }
  if (added.length || log !== state.log) patch.log = [...log, ...added].slice(-LOG_MAX);
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
  if (snap.header && JSON.stringify(snap.header) !== JSON.stringify(state.header)) patch.header = snap.header;
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
      const patch: Partial<RunState> = {};
      if (a && JSON.stringify(a) !== JSON.stringify(state.axis)) patch.axis = a;
      if (info && Array.isArray(info.atlas)) {
        const strip: Strip = { step_mm: info.step_mm, atlas: info.atlas, angles: info.angles || '', sections: info.sections || [], ground: info.ground || [0, 0, 0] };
        if (JSON.stringify(strip) !== JSON.stringify(state.strip)) patch.strip = strip;
      }
      if (Object.keys(patch).length) set(patch);
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

// Keep or remove a review comment (`POST jobs/<n>/comments`); the server answers with the
// comments as they now stand. Returns the refusal's words, or '' when it was kept.
export async function sendComment(body: { section: string; box: number[]; text: string } | { remove: string }): Promise<string> {
  const job = state.job;
  if (job === null) return 'No job is shown.';
  try {
    const r = await fetch(`jobs/${job}/comments`, {
      method: 'POST', headers: { 'Content-Type': 'application/json' }, body: JSON.stringify(body), cache: 'no-store',
    });
    const answer = await r.json().catch(() => ({}));
    if (!r.ok) return String(answer.error || 'The comment was not kept.');
    if (Array.isArray(answer.comments) && job === state.job) set({ comments: answer.comments });
    return '';
  } catch {
    return 'LangSlice is not reachable.';
  }
}
