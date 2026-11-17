// The cutting axis, under the pictures: where every section sits in the brain. Along the
// top, the atlas's own picture at regular places of the axis; under it a millimetre
// ruler with its two ends named; below, every placed section as its thumbnail in stair
// rows (each in the first row where it clears its neighbour), with a key on the ruler at
// its position. When the agent moves a section its thumbnail glides to its new place, so
// a reshuffle is seen as it happens. The sections the agent is working on are in the
// accent, the ones on the stage ringed, with a leader to their place. The wheel zooms
// about the pointer, dragging pans, a double-click shows the whole axis; hovering names a
// section and clicking shows it. Sections not placed yet are counted at the right.
import { AnimatePresence, motion } from 'motion/react';
import { useEffect, useLayoutEffect, useMemo, useRef, useState } from 'react';
import { DUR, EASE_OUT } from '../motion';
import { picture } from '../pictures';
import type { Axis, Row, Strip } from '../run';
import { short } from './Stage';

const PAD = 14;               // side margins, px
const ATLAS_H = 44;           // the atlas row's pictures
const RULER_H = 22;           // the ruler band under the atlas row
const ROW_H = 40;             // a stair row's thumbnails
const ROW_STEP = 0.9;         // stair rows overlap a little
const ASPECT = 1.4;           // a coronal section's width to height
const MAX_ZOOM = 30;

type Domain = { lo: number; hi: number };

export function CuttingAxis({ job, rows, axis, strip, focus, shown, done, onPick, stairRows = 2 }: {
  job: number | null; rows: Row[]; axis: Axis | null; strip: Strip | null;
  focus: string[]; shown: string[]; done: boolean; onPick?: (id: string) => void; stairRows?: number;
}) {
  const ref = useRef<HTMLDivElement>(null);
  const [width, setWidth] = useState(0);
  useLayoutEffect(() => {
    const el = ref.current!;
    const measure = () => setWidth(el.clientWidth);
    measure();
    const o = new ResizeObserver(measure);
    o.observe(el);
    return () => o.disconnect();
  }, []);
  const whole: Domain | null = axis ? { lo: axis.min_mm, hi: axis.max_mm } : null;
  const [domain, setDomain] = useState<Domain | null>(null);
  const view = domain ?? whole;
  const toX = (mm: number) => (view && width ? PAD + ((mm - view.lo) / (view.hi - view.lo)) * (width - 2 * PAD) : 0);

  // Each placed section's place: x on the ruler, its stair row.
  const thumbW = ROW_H * ASPECT;
  const placed = useMemo(() => {
    const out = rows
      .filter(r => typeof r.position_mm === 'number')
      .map(r => ({ id: r.id, mm: r.position_mm as number, x: toX(r.position_mm as number), row: 0 }))
      .sort((a, b) => a.x - b.x);
    const ends = Array(stairRows).fill(-Infinity);
    for (const p of out) {
      let row = ends.findIndex(end => p.x - thumbW / 2 >= end + 2);
      if (row < 0) row = ends.indexOf(Math.min(...ends));
      p.row = row;
      ends[row] = p.x + thumbW / 2;
    }
    return out;
  }, [rows, width, view?.lo, view?.hi, stairRows]); // eslint-disable-line react-hooks/exhaustive-deps
  const waiting = rows.length - placed.length;
  const thumbKey = new Map((strip?.sections ?? []).map(s => [s.id, s.thumb_key]));
  const some = focus.length < rows.length;
  const axisY = ATLAS_H + RULER_H / 2;
  const stairTop = ATLAS_H + RULER_H + 6;
  const height = stairTop + ROW_H * (1 + ROW_STEP * (stairRows - 1)) + 4;

  // Wheel zoom about the pointer, drag to pan, double-click for the whole axis.
  const drag = useRef<{ x: number; dom: Domain } | null>(null);
  const onWheel = (e: React.WheelEvent) => {
    if (!view || !whole || !width) return;
    const at = view.lo + ((e.clientX - ref.current!.getBoundingClientRect().left - PAD) / (width - 2 * PAD)) * (view.hi - view.lo);
    const span = Math.min(whole.hi - whole.lo, Math.max((whole.hi - whole.lo) / MAX_ZOOM, (view.hi - view.lo) * Math.exp(e.deltaY * 0.0015)));
    const f = (at - view.lo) / (view.hi - view.lo);
    let lo = at - f * span;
    lo = Math.min(whole.hi - span, Math.max(whole.lo, lo));
    setDomain(span >= whole.hi - whole.lo - 1e-6 ? null : { lo, hi: lo + span });
  };
  const onDown = (e: React.PointerEvent) => {
    if (e.button !== 0 || !view || !domain) return;
    drag.current = { x: e.clientX, dom: domain };
    (e.currentTarget as HTMLElement).setPointerCapture(e.pointerId);
  };
  const onMove = (e: React.PointerEvent) => {
    const d = drag.current;
    if (!d || !whole || !width) return;
    const span = d.dom.hi - d.dom.lo;
    let lo = d.dom.lo - ((e.clientX - d.x) / (width - 2 * PAD)) * span;
    lo = Math.min(whole.hi - span, Math.max(whole.lo, lo));
    setDomain({ lo, hi: lo + span });
  };

  const [tip, setTip] = useState<{ id: string; x: number } | null>(null);
  const ticks = view ? tickMarks(view, width) : [];
  return (
    <div
      ref={ref}
      className={'cut' + (domain ? ' zoomed' : '') + (done ? ' done' : '')}
      style={{ height }}
      onWheel={onWheel}
      onPointerDown={onDown}
      onPointerMove={onMove}
      onPointerUp={() => { drag.current = null; }}
      onDoubleClick={() => setDomain(null)}
      aria-label="Where the sections sit in the brain"
    >
      {view && job !== null && strip?.atlas.map(a => {
        const x = toX(a.position_mm);
        const w = Math.max(16, Math.min(ATLAS_H * ASPECT, (toX(a.position_mm + strip.step_mm) - x) - 3));
        if (x < -w || x > width + w) return null;
        return (
          <div key={a.k} className="cut-atlas" style={{ transform: `translateX(${x - w / 2}px)`, width: w, height: ATLAS_H }}>
            <Thumb url={`jobs/${job}/render/atlas/${a.k}?edge=128`} rank={900} />
          </div>
        );
      })}
      <div className="cut-ruler" style={{ top: axisY }} />
      {ticks.map(t => (
        <span key={t.mm} className={'cut-tick' + (t.label ? ' major' : '')} style={{ transform: `translateX(${toX(t.mm)}px)`, top: axisY }}>
          {t.label && <span className="cut-tick-label">{t.mm}</span>}
        </span>
      ))}
      <span className="cut-end low" style={{ top: axisY }}>{axis?.low || 'anterior'}</span>
      <span className="cut-end high" style={{ top: axisY }}>{axis?.high || 'posterior'}</span>
      {placed.map(p => {
        const isTarget = some && focus.includes(p.id), isShown = shown.includes(p.id);
        const y = stairTop + p.row * ROW_H * ROW_STEP;
        return (
          <div key={'k' + p.id}>
            <span
              className={'cut-key' + (isTarget ? ' target' : '') + (isShown ? ' shown' : '')}
              style={{ transform: `translate(${p.x}px, ${axisY}px)` }}
            />
            {(isShown || isTarget) && (
              <span className={'cut-leader' + (isShown ? ' shown' : '')}
                style={{ transform: `translate(${p.x}px, ${axisY}px)`, height: y - axisY }} />
            )}
          </div>
        );
      })}
      <AnimatePresence initial={false}>
        {job !== null && placed.map(p => {
          const isTarget = some && focus.includes(p.id), isShown = shown.includes(p.id);
          const y = stairTop + p.row * ROW_H * ROW_STEP;
          if (p.x < -thumbW || p.x > width + thumbW) return null;
          return (
            <motion.button
              type="button"
              key={p.id}
              className={'cut-thumb' + (isTarget ? ' target' : '') + (isShown ? ' shown' : '')}
              style={{ width: thumbW, height: ROW_H, zIndex: isShown ? 3 : isTarget ? 2 : 1 }}
              initial={{ opacity: 0, x: p.x - thumbW / 2, y: y - 8 }}
              animate={{ opacity: 1, x: p.x - thumbW / 2, y }}
              exit={{ opacity: 0, transition: { duration: DUR.quick } }}
              transition={{ x: { duration: DUR.glide, ease: [0.77, 0, 0.175, 1] }, y: { duration: DUR.glide, ease: [0.77, 0, 0.175, 1] }, opacity: { duration: DUR.enter, ease: EASE_OUT } }}
              onClick={() => onPick?.(p.id)}
              onPointerEnter={() => setTip({ id: p.id, x: p.x })}
              onPointerLeave={() => setTip(t => (t?.id === p.id ? null : t))}
              aria-label={short(p.id)}
            >
              <Thumb url={`jobs/${job}/render/thumb/${encodeURIComponent(p.id)}?edge=128&key=${encodeURIComponent(thumbKey.get(p.id) ?? '')}&style=calm`} rank={600} />
            </motion.button>
          );
        })}
      </AnimatePresence>
      {tip && (
        <span className="cut-tip" style={{ transform: `translateX(${tip.x}px)` }}>
          <b>{short(tip.id)}</b> {(() => { const r = rows.find(x => x.id === tip.id); return typeof r?.position_mm === 'number' ? `${r.position_mm.toFixed(2)} mm` : ''; })()}
        </span>
      )}
      <span className="cut-waiting">{waiting ? (waiting === rows.length ? `${rows.length} sections, not placed yet` : `${waiting} not placed yet`) : ''}</span>
    </div>
  );
}

// A small picture drawn on a canvas from the picture queue (lowest priority), contained.
function Thumb({ url, rank }: { url: string; rank: number }) {
  const ref = useRef<HTMLCanvasElement>(null);
  const [img, setImg] = useState<HTMLImageElement | null>(null);
  useEffect(() => {
    let alive = true;
    picture(url, rank).then(i => { if (alive && i) setImg(i); });
    return () => { alive = false; };
  }, [url, rank]);
  useLayoutEffect(() => {
    const c = ref.current;
    if (!c || !img) return;
    const box = c.parentElement!.getBoundingClientRect();
    const dpr = Math.min(2, window.devicePixelRatio || 1);
    c.width = Math.max(1, Math.round(box.width * dpr)); c.height = Math.max(1, Math.round(box.height * dpr));
    const s = Math.min(c.width / img.naturalWidth, c.height / img.naturalHeight);
    const ctx = c.getContext('2d')!;
    ctx.drawImage(img, (c.width - img.naturalWidth * s) / 2, (c.height - img.naturalHeight * s) / 2, img.naturalWidth * s, img.naturalHeight * s);
  }, [img]);
  return <canvas ref={ref} className={'cut-canvas' + (img ? ' ready' : '')} />;
}

// Millimetre ticks: every mm, finer when zoomed in; a label every few.
function tickMarks(view: Domain, width: number) {
  const span = view.hi - view.lo;
  const pxPerMm = (width - 2 * PAD) / span;
  const step = pxPerMm > 120 ? 0.25 : pxPerMm > 50 ? 0.5 : 1;
  const every = pxPerMm > 50 ? 1 : pxPerMm > 22 ? 2 : 5;
  const out: { mm: number; label: boolean }[] = [];
  for (let mm = Math.ceil(view.lo / step) * step; mm <= view.hi + 1e-9; mm += step) {
    const r = Math.round(mm * 100) / 100;
    out.push({ mm: r, label: Math.abs(r / every - Math.round(r / every)) < 1e-6 });
  }
  return out;
}
