// What the agent is doing to each section, as the stage draws it. Every tool belongs to
// one family, and each family has one way of being shown (stage/Act.tsx): a fit breathes
// the atlas borders while it runs, a look settles a viewfinder on the picture, a made
// image is swept over the tissue, a damage mark is drawn in. An act lasts while its call
// runs and lingers after it ends (`LINGER`), so a call too quick to see still reads.
import { useEffect, useState } from 'react';
import { useRun, type Call } from './run';

export type Family = 'fit' | 'look' | 'prep' | 'damage' | 'landmarks' | 'place' | 'lookup' | 'rewind' | 'books';

type Kind = { family: Family; engine: string; what: string };

// Each tool's family, the engine or tool named on the tile's chip, and what it does.
const TOOLS: Record<string, Kind> = {
  elastix_affine: { family: 'fit', engine: 'elastix', what: 'affine fit' },
  interactive_transform: { family: 'fit', engine: 'by hand', what: 'in-plane transform' },
  fireants_syn: { family: 'fit', engine: 'FireANTs', what: 'SyN warp' },
  stalign_lddmm: { family: 'fit', engine: 'STalign', what: 'LDDMM warp' },
  spomialign: { family: 'fit', engine: 'SPOmiAlign', what: 'warp' },
  emlddmm_stack: { family: 'fit', engine: 'emlddmm', what: 'stack warp' },
  trace_borders: { family: 'fit', engine: 'image model', what: 'border trace' },
  landmark_triangulation: { family: 'landmarks', engine: 'landmarks', what: 'triangulation warp' },
  set_landmarks: { family: 'landmarks', engine: 'landmarks', what: 'pairs set' },
  look: { family: 'look', engine: 'look', what: 'looking' },
  zoom: { family: 'look', engine: 'zoom', what: 'looking closer' },
  make_image: { family: 'prep', engine: 'image', what: 'preprocessing' },
  make_mask: { family: 'prep', engine: 'mask', what: 'what a fit compares' },
  mark_damage: { family: 'damage', engine: 'damage', what: 'marked' },
  position_sections: { family: 'place', engine: 'position', what: 'placing' },
  stalign_plane_search: { family: 'place', engine: 'STalign', what: 'plane search' },
  emlddmm_plane: { family: 'place', engine: 'emlddmm', what: 'plane search' },
  deepslice_place: { family: 'place', engine: 'DeepSlice', what: 'placing' },
  brainways_ap: { family: 'place', engine: 'Brainways', what: 'placing' },
  grep_atlas: { family: 'lookup', engine: 'atlas', what: 'lookup' },
  grep_atlas_view: { family: 'lookup', engine: 'atlas', what: 'regions' },
  undo: { family: 'rewind', engine: 'undo', what: 'last step undone' },
  redo: { family: 'rewind', engine: 'redo', what: 'step redone' },
  restore_registration: { family: 'rewind', engine: 'restore', what: 'earlier registration' },
};

export const LINGER_MS = 1600;
// How long a family's act stays after its call: a made image needs time to be drawn and
// swept, a damage mark to be drawn in.
const LINGER: Partial<Record<Family, number>> = { prep: 3600, damage: 2600, fit: 1800, look: 1500 };
const ZOOM_LINGER_MS = 3200;
const lingerOf = (call: Call) => (call.name === 'zoom' ? ZOOM_LINGER_MS : LINGER[kindOf(call.name).family] ?? LINGER_MS);

// The few words a call's own arguments add to its title (regions marked, a recipe, the
// regions looked up), for the line under the bubble.
export function callDetail(call: Call): string {
  const args = (call.args && typeof call.args === 'object' ? call.args : {}) as Record<string, unknown>;
  return detail(call.name, args);
}

export type Act = {
  key: string;
  family: Family;
  tool: string;
  engine: string;
  what: string;
  running: boolean;
  failed: boolean;
  args: Record<string, unknown>;
  sections: string[];
  count: number;              // how many sections the call works on
  zooms: Record<string, number[]>; // a zoom's window on each section
  hold: boolean;              // a read of a batch still being read: its zoom stays pushed in
};

// The sections a call acts on: those it named, else those its pictures show.
const targets = (call: Call) => (call.sections.length ? call.sections : call.seen ?? []);

export function kindOf(tool: string): Kind {
  return TOOLS[tool] ?? { family: 'books', engine: tool.replace(/_/g, ' '), what: '' };
}

function actOf(call: Call, hold = false): Act {
  const kind = kindOf(call.name);
  const args = (call.args && typeof call.args === 'object' ? call.args : {}) as Record<string, unknown>;
  return {
    key: call.key, family: kind.family, tool: call.name, engine: kind.engine, what: detail(call.name, args) || kind.what,
    running: call.running, failed: call.failed, args, sections: targets(call), count: targets(call).length,
    zooms: call.zooms ?? {}, hold,
  };
}

// An agent often asks for several looks or zooms in one turn; the door runs them one after
// another, each starting as the one before ends, and the agent reads their pictures
// together. Such a batch stays on the stage as one: each read lingers until the last one
// has ended, and its zoom stays pushed in meanwhile (`hold`) and for HOLD_AFTER_MS after:
// long enough for the next read of the batch to come, and for the last zoom, whose window
// is known only when it ends, to push in beside the others before they all let go.
const BATCH_GAP_MS = 1000;
const HOLD_AFTER_MS = 2600;
const isRead = (call: Call) => kindOf(call.name).family === 'look';
function batches(calls: Call[]): Map<string, { running: boolean; endedAt: number }> {
  const out = new Map<string, { running: boolean; endedAt: number }>();
  let members: Call[] = [];
  const close = () => {
    const batch = { running: members.some(c => c.running), endedAt: Math.max(0, ...members.map(c => c.endedAt ?? 0)) };
    for (const c of members) out.set(c.key, batch);
    members = [];
  };
  for (const call of calls) {
    const before = members.at(-1);
    const joins = before && isRead(call) && Date.parse(call.at ?? '') - Date.parse(before.doneAt ?? '') <= BATCH_GAP_MS;
    if (!joins && members.length) close();
    if (isRead(call)) members.push(call);
  }
  if (members.length) close();
  return out;
}

// The few words a chip adds for a call, from its own arguments.
function detail(tool: string, args: Record<string, unknown>): string {
  const list = (v: unknown) => (Array.isArray(v) ? v.map(String) : []);
  if (tool === 'mark_damage') return list(args.regions).join(', ') || 'cleared';
  if (tool === 'make_image') return recipe(args);
  if (tool === 'grep_atlas_view') return list(args.regions).join(', ');
  if (tool === 'grep_atlas') return String(args.query ?? '');
  if (tool === 'set_landmarks') {
    const pairs = Array.isArray(args.pairs) ? args.pairs.length : 0;
    return pairs ? `${pairs} pair${pairs === 1 ? '' : 's'}` : 'pairs changed';
  }
  return '';
}

// A made image's recipe in a few words: its name, then what was done to the channels.
function recipe(args: Record<string, unknown>): string {
  const parts = [String(args.name ?? '')];
  if (args.invert) parts.push('inverted');
  if (args.clahe_clip != null) parts.push(`CLAHE ${Number(args.clahe_clip).toFixed(1)}`);
  if (args.n4) parts.push('N4');
  return parts.filter(Boolean).join(' · ');
}

// The act on each section now: a running call's, else one that ended within LINGER_MS;
// the newest call wins. Re-read when a call starts or ends and when a lingering one runs out.
export function useActs(): Map<string, Act> {
  const calls = useRun(s => s.calls);
  const [now, setNow] = useState(() => Date.now());
  const acts = new Map<string, Act>();
  let wake = Infinity;
  const recent = calls.slice(-40);
  const batchOf = batches(recent);
  for (let i = recent.length - 1; i >= 0; i--) {
    const call = recent[i];
    const batch = batchOf.get(call.key);
    const running = call.running || Boolean(batch?.running);
    const until = Math.max(call.endedAt ?? 0, batch?.endedAt ?? 0) + lingerOf(call);
    if (!running && until <= now) continue;
    if (!running) wake = Math.min(wake, until);
    const grace = (batch?.endedAt ?? 0) + HOLD_AFTER_MS;
    if (!running && batch && grace > now) wake = Math.min(wake, grace);
    const act = actOf(call, running || Boolean(batch && grace > now));
    for (const id of act.sections) if (!acts.has(id)) acts.set(id, act);
  }
  useEffect(() => {
    if (!Number.isFinite(wake)) return;
    const t = window.setTimeout(() => setNow(Date.now()), Math.max(16, wake - Date.now()));
    return () => clearTimeout(t);
  }, [wake]);
  useEffect(() => { setNow(Date.now()); }, [calls]);
  return acts;
}

// The newest act with no sections of its own (an atlas look, a lookup, bookkeeping),
// while it runs or lingers: the stage as a whole shows it.
export function useStageAct(): Act | null {
  const calls = useRun(s => s.calls);
  const [now, setNow] = useState(() => Date.now());
  const call = calls.at(-1);
  const until = (call?.endedAt ?? 0) + (call ? lingerOf(call) : 0);
  const live = call && !targets(call).length && (call.running || until > now) ? call : null;
  useEffect(() => {
    if (!call || call.running || until <= Date.now()) return;
    const t = window.setTimeout(() => setNow(Date.now()), until - Date.now());
    return () => clearTimeout(t);
  }, [call?.key, call?.running, until]);
  useEffect(() => { setNow(Date.now()); }, [calls]);
  return live ? actOf(live) : null;
}
