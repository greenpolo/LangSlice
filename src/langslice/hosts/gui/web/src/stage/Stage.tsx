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
import { AnimatePresence, motion, useIsPresent } from 'motion/react';
import { useEffect, useLayoutEffect, useRef, useState, type ReactNode } from 'react';
import { DUR, EASE_IN_OUT, EASE_OUT, LAYOUT_SPRING } from '../ui/motion';
import { useActs } from '../data/acts';
import { partUrls, picture, regionUrl, snapEdge } from '../data/pictures';
import { short, useRun, type Row } from '../data/run';
import { Act, actClass } from './Act';
import { Picture } from './Picture';
import { Regions } from './Regions';

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
// The words' regions are outlined on this many tiles at most: on a grid they are too small.
const REGION_TILES = 6;

export type Box = { width: number; height: number };
type Place = { x: number; y: number };
type Geometry = { cols: number; width: number; height: number; edge: number; named: boolean; places: Place[] };

// Where n pictures go in the box: one row up to three (in a box wider than tall), else
// the columns that make them largest; each row centred, the whole block centred.
export function geometry(n: number, box: Box): Geometry {
  const width = Math.min(box.width, STAGE_MAX_W), height = box.height;
  const named = n <= 6;
  let best = { cols: 1, w: 0 };
  const choices = n <= LARGE_MAX && box.width >= box.height ? [Math.max(1, n)] : Array.from({ length: n }, (_, i) => i + 1);
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
  const acts = useActs();
  const pointed = useRun(s => s.words?.pointed);
  if (!box) return null;
  const g = geometry(shown.length, box);
  const many = shown.length > MANY;
  const named = new Set(pointed?.sections ?? []);
  const regions = shown.length <= REGION_TILES ? pointed?.regions ?? [] : [];
  return (
    <div className="stage-grid">
      <AnimatePresence initial={false}>
        {shown.map((id, i) => {
          const row = byId.get(id);
          const at = g.places[i];
          const act = acts.get(id);
          return (
            <motion.figure
              key={id}
              layout
              className={'tile' + (onPick ? ' pickable' : '') + actClass(act) + (named.has(id) ? ' pointed' : '') + (regions.length ? ' has-regions' : '')}
              style={{ left: at.x, top: at.y, width: g.width }}
              onClick={onPick && shown.length > 1 ? () => onPick(id) : undefined}
              initial={many ? { opacity: 0, scale: 0.96 } : { opacity: 0, scale: 0.97, filter: 'blur(2px)' }}
              animate={many ? { opacity: 1, scale: 1 } : { opacity: 1, scale: 1, filter: 'blur(0px)' }}
              exit={{ opacity: 0, transition: { duration: DUR.enter, ease: EASE_IN_OUT } }}
              transition={{
                layout: LAYOUT_SPRING,
                default: {
                  duration: many ? 0.32 : DUR.enter,
                  ease: EASE_OUT,
                  delay: many ? Math.min(0.24, 0.24 * Math.sqrt(i / shown.length)) : Math.min(0.06, 0.03 * i),
                },
              }}
            >
              <div className="tile-picture" style={{ height: g.height }}>
                <Picture id={id} urls={partUrls(job, id, g.edge, row?.render_key)} rank={i} label={'The section ' + short(id)}
                  reveal={act?.family === 'damage'} />
                <AnimatePresence>
                  {regions.length > 0 && (
                    <Regions key={regions.join(',')} url={regionUrl(job, id, g.edge, row?.render_key, regions)} />
                  )}
                </AnimatePresence>
                <Act act={act} job={job} id={id} edge={g.edge} renderKey={row?.render_key} tiles={shown.length}
                  pointed={named.has(id)} />
              </div>
              {g.named && (
                <Caption>
                  {short(id)}
                  {typeof row?.position_mm === 'number' && <span className="tile-where">{row.position_mm.toFixed(2)} mm</span>}
                  {detailed && row && <Tags row={row} />}
                </Caption>
              )}
            </motion.figure>
          );
        })}
      </AnimatePresence>
    </div>
  );
}

// A tile's name. While its tile leaves, the name goes at once, and a new tile's name
// comes only after that, so two names in the same place never cross: a picture can
// crossfade into another, words cannot.
function Caption({ children }: { children: ReactNode }) {
  const present = useIsPresent();
  return (
    <motion.figcaption
      className="tile-name"
      initial={{ opacity: 0 }}
      animate={{ opacity: present ? 1 : 0 }}
      transition={present ? { duration: DUR.quick, ease: EASE_OUT, delay: DUR.quick } : { duration: DUR.quick / 2 }}
    >
      {children}
    </motion.figcaption>
  );
}

// How a section stands, in plain words, for the Extended view's tiles.
const HOW: Record<string, string> = { elastix: 'aligned automatically', interactive: 'aligned by hand' };
function Tags({ row }: { row: Row }) {
  const tags = [
    row.transform ? HOW[row.transform] || row.transform : '',
    row.deformation ? 'warped' : '',
    row.position_source === 'initial' ? 'initial position' : '',
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
        return ids.flatMap(id => Object.values(partUrls(job, id, edge, byId.get(id)?.render_key)));
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
  }, [key, job, box !== null]);
  const known = new Set(rows.map(r => r.id));
  return shown.filter(id => known.has(id));
}

export function useBox<T extends HTMLElement>(): [(el: T | null) => void, Box | null] {
  const [el, setEl] = useState<T | null>(null);
  const [box, setBox] = useState<Box | null>(null);
  useLayoutEffect(() => {
    if (!el) return;
    const measure = () => {
      const r = el.getBoundingClientRect();
      if (r.width > 8 && r.height > 8) setBox(old => (old && old.width === r.width && old.height === r.height ? old : { width: r.width, height: r.height }));
    };
    measure();
    const o = new ResizeObserver(measure);
    o.observe(el);
    return () => o.disconnect();
  }, [el]);
  return [setEl, box];
}

