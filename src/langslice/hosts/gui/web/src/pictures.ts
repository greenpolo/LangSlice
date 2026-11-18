// The sections' pictures the server draws. A job draws one picture at a time and the
// browser keeps six connections to the server, so at most three load at once (the polls
// keep theirs), the lowest rank first. A loaded picture is decoded before it is handed
// out and kept (the newest `KEPT`), so a picture fetched ahead of its focus is shown the
// moment its focus is: the pictures are drawn on canvases from these images, never asked
// for again.

const MAX = 3;
const KEPT = 96;
const TRIES = 12;
const AGAIN_MS = 700;

type Waiting = { url: string; rank: number; tries: number; done: ((img: HTMLImageElement | null) => void)[] };

const ready = new Map<string, HTMLImageElement>();
// Each section's newest picture at any size: shown at once while the exact one comes.
const newest = new Map<string, HTMLImageElement>();
// The sizes pictures are drawn at, so a section shown a little smaller keeps its picture.
const EDGES = [256, 512, 768, 1024, 1280];

// The edge to draw a picture shown *width* CSS pixels wide.
export function snapEdge(width: number): number {
  const need = width * Math.min(2, window.devicePixelRatio || 1);
  return EDGES.find(e => e >= need * 0.9) ?? EDGES[EDGES.length - 1];
}

// A section's newest picture, whatever its size or registration; null before any.
export function newestOf(id: string): HTMLImageElement | null {
  return newest.get(id) ?? null;
}
const flying = new Map<string, Waiting>();
let queue: Waiting[] = [];
let running = 0;

export function cached(url: string): HTMLImageElement | null {
  const img = ready.get(url);
  if (img) { ready.delete(url); ready.set(url, img); }
  return img ?? null;
}

// The picture at *url*, decoded; null when it cannot be drawn (the server's refusal).
export function picture(url: string, rank = 1): Promise<HTMLImageElement | null> {
  const img = cached(url);
  if (img) return Promise.resolve(img);
  return new Promise(resolve => {
    const known = flying.get(url);
    if (known) { known.rank = Math.min(known.rank, rank); known.done.push(resolve); return; }
    const item: Waiting = { url, rank, tries: 0, done: [resolve] };
    flying.set(url, item);
    queue.push(item);
    pump();
  });
}

// Pictures wanted later (the finished run's grid): fetched one at a time, only while
// nothing else is wanted, so they never hold up the pictures being watched.
let later: string[] = [];
export function warm(urls: string[]) {
  later = urls.filter(u => !ready.has(u));
  pump();
}

function pump() {
  if (!queue.length && running === 0 && later.length) {
    const url = later.shift()!;
    if (!ready.has(url) && !flying.has(url)) {
      const item: Waiting = { url, rank: 1000, tries: 0, done: [] };
      flying.set(url, item);
      queue.push(item);
    }
  }
  while (running < MAX && queue.length) {
    queue.sort((a, b) => a.rank - b.rank);
    const item = queue.shift()!;
    running += 1;
    fetchOne(item).finally(() => { running -= 1; pump(); });
  }
}

async function fetchOne(item: Waiting) {
  const img = new Image();
  img.decoding = 'async';
  const ok = await new Promise<boolean>(resolve => {
    img.onload = () => resolve(true);
    img.onerror = () => resolve(false);
    img.src = item.url;
  });
  if (ok) {
    await img.decode().catch(() => {});
    ready.set(item.url, img);
    while (ready.size > KEPT) ready.delete(ready.keys().next().value!);
    const id = sectionOf(item.url);
    if (id) newest.set(id, img);
    finish(item, img);
    return;
  }
  // Still being drawn (503) or drawn meanwhile: ask again after a pause; a refusal ends it.
  let status = 0;
  try { status = (await fetch(item.url, { cache: 'no-store' })).status; } catch { status = 0; }
  item.tries += 1;
  if ((status === 503 || status === 200) && item.tries < TRIES) {
    setTimeout(() => { queue.push(item); pump(); }, status === 200 ? 0 : AGAIN_MS);
    return;
  }
  finish(item, null);
}

function finish(item: Waiting, img: HTMLImageElement | null) {
  flying.delete(item.url);
  for (const done of item.done) done(img);
}

function sectionOf(url: string): string | null {
  const m = url.match(/render\/overlay\/([^?]+)/);
  return m ? decodeURIComponent(m[1]) : null;
}

// How the window's look draws a section: quiet on the dark stage, or on paper in the
// light look. The viewer is mounted again when the look changes, so an address made
// from this is never stale.
export function pictureStyle(): 'calm' | 'paper' {
  return document.documentElement.dataset.theme === 'light' ? 'paper' : 'calm';
}

// The address of a section's picture under its current registration, in the window's look.
export function overlayUrl(job: number, id: string, edge: number, key: string | null | undefined) {
  return `jobs/${job}/render/overlay/${encodeURIComponent(id)}?edge=${edge}&key=${encodeURIComponent(String(key ?? ''))}&style=${pictureStyle()}`;
}
