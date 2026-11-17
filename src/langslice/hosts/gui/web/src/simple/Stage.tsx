// The sections the agent is working on, on one stage. Up to three are large, side by
// side; more are dealt out as a grid; a finished run shows every section. Every tile has
// its own place (computed here, not by the page's flow), so on a new focus a section
// still shown glides and resizes to its new place, a new one arrives where it will stay
// and a gone one fades where it was: nothing rises from below, and a complete swap is a
// crossfade, never an empty stage.
//
// `useDirector` decides what is shown and when: it holds each focus at least DWELL_MS (a
// grid DECK_DWELL_MS) and shows the next once its pictures are fetched and decoded (at
// most READY_WAIT_MS), so a burst of quick calls reads as steps. `geometry` decides where.
import { AnimatePresence, motion } from 'motion/react';
import { useEffect, useLayoutEffect, useRef, useState } from 'react';
import { DUR, EASE_OUT, LAYOUT_SPRING } from '../motion';
import { overlayUrl, picture, snapEdge } from '../pictures';
import type { Row } from '../run';
import { Picture } from './Picture';

const DWELL_MS = 2400;
const DECK_DWELL_MS = 4000;
const READY_WAIT_MS = 4000;
const LARGE_MAX = 3;
const STAGE_MAX_W = 1240;
const GAP = 12;
const NAME_H = 26;
const ASPECT = 1.35;
// From this many tiles an entrance is opacity and scale only (many blur filters cost).
const MANY = 12;

export type Box = { width: number; height: number };
type Place = { x: number; y: number };
type Geometry = { cols: number; width: number; height: number; edge: number; named: boolean; places: Place[] };

// Where n pictures go in the box: one row up to three, else the columns that make them
// largest; each row centred, the whole block centred.
export function geometry(n: number, box: Box): Geometry {
  const width = Math.min(box.width, STAGE_MAX_W), height = box.height;
  const named = n <= 6;
  let best = { cols: 1, w: 0 };
  const choices = n <= LARGE_MAX ? [Math.max(1, n)] : Array.from({ length: n }, (_, i) => i + 1);
  for (const cols of choices) {
    const lines = Math.ceil(n / cols);
    const w = Math.min(
      (width - GAP * (cols - 1)) / cols,
      ((height - GAP * (lines - 1)) / lines - (named ? NAME_H : 0)) * ASPECT,
    );
    if (w > best.w) best = { cols, w };
  }
  const w = Math.max(40, Math.floor(best.w));
  const h = Math.floor(w / ASPECT);
  const tileH = h + (named ? NAME_H : 0);
  const lines = Math.ceil(n / best.cols);
  const top = (box.height - (lines * tileH + (lines - 1) * GAP)) / 2;
  const places: Place[] = [];
  for (let i = 0; i < n; i++) {
    const line = Math.floor(i / best.cols);
    const inLine = Math.min(best.cols, n - line * best.cols);
    const left = (box.width - (inLine * w + (inLine - 1) * GAP)) / 2;
    places.push({ x: Math.round(left + (i % best.cols) * (w + GAP)), y: Math.round(top + line * (tileH + GAP)) });
  }
  return { cols: best.cols, width: w, height: h, edge: snapEdge(w), named, places };
}

export function Stage({ job, shown, rows, box, onPick, detailed = false }: {
  job: number; shown: string[]; rows: Row[]; box: Box | null; onPick?: (id: string) => void; detailed?: boolean;
}) {
  const byId = new Map(rows.map(r => [r.id, r]));
  if (!box) return null;
  const g = geometry(shown.length, box);
  const many = shown.length > MANY;
  return (
    <div className="stage-grid">
      <AnimatePresence initial={false}>
        {shown.map((id, i) => {
          const row = byId.get(id);
          const at = g.places[i];
          return (
            <motion.figure
              key={id}
              layout
              className={'tile' + (onPick ? ' pickable' : '')}
              style={{ left: at.x, top: at.y, width: g.width }}
              onClick={onPick && shown.length > 1 ? () => onPick(id) : undefined}
              initial={many ? { opacity: 0, scale: 0.96 } : { opacity: 0, scale: 0.97, filter: 'blur(6px)' }}
              animate={many ? { opacity: 1, scale: 1 } : { opacity: 1, scale: 1, filter: 'blur(0px)' }}
              exit={{ opacity: 0, transition: { duration: DUR.quick, ease: EASE_OUT } }}
              transition={{
                layout: LAYOUT_SPRING,
                default: {
                  duration: many ? 0.32 : DUR.enter,
                  ease: EASE_OUT,
                  delay: Math.min(0.24, 0.24 * Math.sqrt(i / Math.max(1, shown.length))),
                },
              }}
            >
              <div className="tile-picture" style={{ height: g.height }}>
                <Picture id={id} url={overlayUrl(job, id, g.edge, row?.render_key)} rank={i} label={'The section ' + short(id)} />
              </div>
              {g.named && (
                <motion.figcaption layout="position" className="tile-name">
                  {short(id)}
                  {typeof row?.position_mm === 'number' && <span className="tile-where">{row.position_mm.toFixed(2)} mm</span>}
                  {detailed && row && <Tags row={row} />}
                </motion.figcaption>
              )}
            </motion.figure>
          );
        })}
      </AnimatePresence>
    </div>
  );
}

// How a section stands, in plain words, for the Extended view's tiles.
const HOW: Record<string, string> = { elastix: 'aligned automatically', interactive: 'aligned by hand' };
function Tags({ row }: { row: Row }) {
  const tags = [
    row.transform ? HOW[row.transform] || row.transform : '',
    row.deformation ? 'warped' : '',
    typeof row.position_mm !== 'number' ? 'not placed' : '',
  ].filter(Boolean);
  return (
    <>
      {tags.map(t => <span key={t} className="tile-tag">{t}</span>)}
      {row.damaged && <span className="tile-tag damaged">damaged</span>}
    </>
  );
}

// What the stage shows: the wanted sections, once the last focus has had its time and
// their pictures are ready (asked for with the rows as they are when it is applied).
export function useDirector(job: number | null, wanted: string[], rows: Row[], box: Box | null): string[] {
  const [shown, setShown] = useState<string[]>([]);
  const shownAt = useRef(0);
  const token = useRef(0);
  const key = wanted.join('|');
  const latest = useRef({ wanted, rows, box });
  latest.current = { wanted, rows, box };
  useEffect(() => {
    if (key === shown.join('|')) return;
    const mine = ++token.current;
    let timer = 0;
    const go = () => {
      const { wanted: ids, box: b } = latest.current;
      if (!b || job === null || mine !== token.current) return;
      const edge = geometry(ids.length, b).edge;
      const urls = () => {
        const byId = new Map(latest.current.rows.map(r => [r.id, r]));
        return ids.map(id => overlayUrl(job, id, edge, byId.get(id)?.render_key));
      };
      const apply = () => {
        if (mine !== token.current) return;
        shownAt.current = Date.now();
        setShown(ids);
      };
      if (!shown.length) { apply(); return; }
      Promise.race([
        Promise.all(urls().map((url, i) => picture(url, i))),
        new Promise(done => setTimeout(done, READY_WAIT_MS)),
      ]).then(() => {
        // Pictures redrawn meanwhile are asked for again, briefly, before the change.
        Promise.race([
          Promise.all(urls().map((url, i) => picture(url, i))),
          new Promise(done => setTimeout(done, 600)),
        ]).then(apply);
      });
    };
    const dwell = shown.length > LARGE_MAX ? DECK_DWELL_MS : DWELL_MS;
    const wait = shownAt.current + dwell - Date.now();
    if (wait > 0) timer = window.setTimeout(go, wait); else go();
    return () => clearTimeout(timer);
  }, [key, job, box !== null]); // eslint-disable-line react-hooks/exhaustive-deps
  const known = new Set(rows.map(r => r.id));
  return shown.filter(id => known.has(id));
}

export function useBox<T extends HTMLElement>(): [React.RefObject<T | null>, Box | null] {
  const ref = useRef<T>(null);
  const [box, setBox] = useState<Box | null>(null);
  useLayoutEffect(() => {
    const el = ref.current;
    if (!el) return;
    const measure = () => {
      const r = el.getBoundingClientRect();
      if (r.width > 8 && r.height > 8) setBox(old => (old && old.width === r.width && old.height === r.height ? old : { width: r.width, height: r.height }));
    };
    measure();
    const o = new ResizeObserver(measure);
    o.observe(el);
    return () => o.disconnect();
  }, []);
  return [ref, box];
}

export const short = (id: string) => id.replace(/\.[A-Za-z0-9]{2,5}$/, '');
