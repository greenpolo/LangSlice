// The cutting axis, under the pictures: the order the agent has put the sections in,
// front to back. A millimetre ruler over the part of the axis the placed sections span
// (with a little margin), its two ends named; in Extended, the atlas's own pictures along
// the top. Under the ruler every placed section's thumbnail stands in one row, in the
// order of its position, each joined to its key on the ruler by a thin line; when they
// do not fit one row at full size they alternate between two half-offset rows, so
// left to right is still the order. When the agent moves a section its thumbnail glides
// to its new place in the row, so a reshuffle is seen as it happens. The sections the
// agent is working on are in the accent, the ones on the stage ringed. The wheel zooms
// about the pointer, dragging pans, a double-click shows the whole span; hovering names
// a section and clicking shows it. Sections not placed yet are counted at the right.
import { AnimatePresence, motion } from 'motion/react';
import { useEffect, useLayoutEffect, useMemo, useRef, useState } from 'react';
import { DUR, EASE_IN_OUT, EASE_OUT } from '../motion';
import { picture } from '../pictures';
import type { Axis, Row, Strip } from '../run';
import { short } from './Stage';

const PAD = 72;               // side margins, px: the ends' names sit in them
const ATLAS_H = 40;           // the atlas row's pictures (Extended)
const TOP = 16;               // room for the millimetre labels above the ruler
const RULER_H = 22;           // the ruler band
const FAN = 20;               // the lines from the ruler down to the row
const ASPECT = 1.4;           // a coronal section's width to height
const MAX_W = 58;             // a thumbnail's widest
const ONE_ROW_MIN_W = 26;     // narrower than this, two half-offset rows
const GAP = 3;
const MAX_ZOOM = 30;

type Domain = { lo: number; hi: number };

export function CuttingAxis({ job, rows, axis, strip, focus, shown, done, onPick, atlas = false }: {
  job: number | null; rows: Row[]; axis: Axis | null; strip: Strip | null;
  focus: string[]; shown: string[]; done: boolean; onPick?: (id: string) => void; atlas?: boolean;
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

  const placedRows = rows.filter(r => typeof r.position_mm === 'number');
  const whole = span(axis, placedRows.map(r => r.position_mm as number));
  const [domain, setDomain] = useState<Domain | null>(null);
  // A zoom made over another span is dropped when the span moves.
  const wholeKey = whole ? `${whole.lo.toFixed(3)}:${whole.hi.toFixed(3)}` : '';
  useEffect(() => setDomain(null), [wholeKey]);
  const view = domain ?? whole;
  const inner = Math.max(1, width - 2 * PAD);
  const toX = (mm: number) => (view ? PAD + ((mm - view.lo) / (view.hi - view.lo)) * inner : 0);

  // The row: each placed section in view, in the order of its position, at its own place
  // where there is room and pushed aside just enough where there is not.
  const layout = useMemo(() => {
    const inView = placedRows
      .map(r => ({ id: r.id, mm: r.position_mm as number, order: r.order ?? 0 }))
      .filter(p => !view || (p.mm >= view.lo - 1e-9 && p.mm <= view.hi + 1e-9))
      .sort((a, b) => a.mm - b.mm || a.order - b.order);
    const n = inView.length;
    const oneRowW = Math.min(MAX_W, n ? inner / n - GAP : MAX_W);
    const two = oneRowW < ONE_ROW_MIN_W;
    const w = Math.max(16, two ? Math.min(MAX_W, (2 * inner) / Math.max(1, n) - GAP) : oneRowW);
    const step = two ? (w + GAP) / 2 : w + GAP;
    const xs = spread(inView.map(p => toX(p.mm)), step, 16 + w / 2, width - 16 - w / 2);
    return { w, h: w / ASPECT, two, items: inView.map((p, i) => ({ ...p, key: toX(p.mm), x: xs[i], row: two ? i % 2 : 0 })) };
  }, [rows, width, view?.lo, view?.hi]); // eslint-disable-line react-hooks/exhaustive-deps

  const waiting = rows.length - placedRows.length;
  const thumbKey = new Map((strip?.sections ?? []).map(s => [s.id, s.thumb_key]));
  const some = focus.length < rows.length;
  const axisY = (atlas ? ATLAS_H + TOP : TOP) + RULER_H / 2 - 6;
  const rowTop = axisY + RULER_H / 2 + FAN;
  const rowStep = layout.h + GAP + 2;   // rows never overlap
  const height = rowTop + layout.h + (layout.two ? rowStep : 0) + 18;

  // Wheel zoom about the pointer, drag to pan, double-click for the whole span.
  const drag = useRef<{ x: number; dom: Domain } | null>(null);
  const onWheel = (e: React.WheelEvent) => {
    if (!view || !whole || !width) return;
    const at = view.lo + ((e.clientX - ref.current!.getBoundingClientRect().left - PAD) / inner) * (view.hi - view.lo);
    const size = Math.min(whole.hi - whole.lo, Math.max((whole.hi - whole.lo) / MAX_ZOOM, (view.hi - view.lo) * Math.exp(e.deltaY * 0.0015)));
    const f = (at - view.lo) / (view.hi - view.lo);
    const lo = Math.min(whole.hi - size, Math.max(whole.lo, at - f * size));
    setDomain(size >= whole.hi - whole.lo - 1e-6 ? null : { lo, hi: lo + size });
  };
  const onDown = (e: React.PointerEvent) => {
    if (e.button !== 0 || !domain) return;
    drag.current = { x: e.clientX, dom: domain };
    (e.currentTarget as HTMLElement).setPointerCapture(e.pointerId);
  };
  const onMove = (e: React.PointerEvent) => {
    const d = drag.current;
    if (!d || !whole) return;
    const size = d.dom.hi - d.dom.lo;
    const lo = Math.min(whole.hi - size, Math.max(whole.lo, d.dom.lo - ((e.clientX - d.x) / inner) * size));
    setDomain({ lo, hi: lo + size });
  };

  const [tip, setTip] = useState<{ id: string; x: number } | null>(null);
  const ticks = view ? tickMarks(view, inner) : [];
  const glide = { duration: DUR.glide, ease: EASE_IN_OUT };
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
      aria-label="The order of the sections, front to back"
    >
      {atlas && view && job !== null && strip?.atlas.map(a => {
        const x = toX(a.position_mm);
        const w = Math.max(16, Math.min(ATLAS_H * ASPECT, (toX(a.position_mm + strip.step_mm) - x) - 3));
        if (x < PAD - w / 2 || x > width - PAD + w / 2) return null;
        return (
          <div key={a.k} className="cut-atlas" style={{ transform: `translateX(${x - w / 2}px)`, width: w, height: ATLAS_H }}>
            <Thumb url={`jobs/${job}/render/atlas/${a.k}?edge=128`} rank={900} cutout />
          </div>
        );
      })}
      <div className="cut-ruler" style={{ top: axisY, left: PAD, right: PAD }} />
      {ticks.map(t => (
        <span key={t.mm} className={'cut-tick' + (t.label ? ' major' : '')} style={{ transform: `translateX(${toX(t.mm)}px)`, top: axisY }}>
          {t.label && <span className="cut-tick-label">{fmt(t.mm)}</span>}
        </span>
      ))}
      <span className="cut-end low" style={{ top: axisY }}>{axis?.low || 'anterior'}</span>
      <span className="cut-end high" style={{ top: axisY }}>{axis?.high || 'posterior'}</span>
      <svg className="cut-fan" width={width} height={height} aria-hidden="true">
        {layout.items.map(p => {
          const isTarget = some && focus.includes(p.id), isShown = shown.includes(p.id);
          return (
            <motion.line
              key={p.id}
              className={'cut-link' + (isTarget ? ' target' : '') + (isShown ? ' shown' : '')}
              initial={false}
              animate={{ x1: p.key, y1: axisY + 3, x2: p.x, y2: rowTop + p.row * rowStep }}
              transition={glide}
            />
          );
        })}
      </svg>
      {layout.items.map(p => {
        const isTarget = some && focus.includes(p.id), isShown = shown.includes(p.id);
        return (
          <span
            key={'k' + p.id}
            className={'cut-key' + (isTarget ? ' target' : '') + (isShown ? ' shown' : '')}
            style={{ transform: `translate(${p.key}px, ${axisY}px)` }}
          />
        );
      })}
      <AnimatePresence initial={false}>
        {job !== null && layout.items.map(p => {
          const isTarget = some && focus.includes(p.id), isShown = shown.includes(p.id);
          const y = rowTop + p.row * rowStep;
          return (
            <motion.button
              type="button"
              key={p.id}
              className={'cut-thumb' + (isTarget ? ' target' : '') + (isShown ? ' shown' : '')}
              style={{ zIndex: isShown ? 3 : isTarget ? 2 : 1 }}
              initial={{ opacity: 0, x: p.x - layout.w / 2, y: y + 6, width: layout.w, height: layout.h }}
              animate={{ opacity: 1, x: p.x - layout.w / 2, y, width: layout.w, height: layout.h }}
              exit={{ opacity: 0, transition: { duration: DUR.quick } }}
              transition={{ x: glide, y: glide, width: glide, height: glide, opacity: { duration: DUR.enter, ease: EASE_OUT } }}
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
        <span className="cut-tip" style={{ transform: `translateX(${tip.x}px)`, top: axisY - RULER_H / 2 - 4 }}>
          <b>{short(tip.id)}</b> {(() => { const r = rows.find(x => x.id === tip.id); return typeof r?.position_mm === 'number' ? `${r.position_mm.toFixed(2)} mm` : ''; })()}
        </span>
      )}
      <span className="cut-waiting">{waiting ? (waiting === rows.length ? `${rows.length} sections, not placed yet` : `${waiting} not placed yet`) : ''}</span>
    </div>
  );
}

// The part of the axis to show: the placed sections' span with a margin, within the
// atlas; the whole atlas while nothing is placed.
function span(axis: Axis | null, mms: number[]): Domain | null {
  if (!axis) return null;
  if (!mms.length) return { lo: axis.min_mm, hi: axis.max_mm };
  let lo = Math.min(...mms), hi = Math.max(...mms);
  const margin = Math.max(0.25, (hi - lo) * 0.04);
  if (hi - lo < 1) { const mid = (lo + hi) / 2; lo = mid - 0.5; hi = mid + 0.5; }
  lo = Math.max(axis.min_mm, lo - margin);
  hi = Math.min(axis.max_mm, hi + margin);
  return hi > lo ? { lo, hi } : { lo: axis.min_mm, hi: axis.max_mm };
}

// Places sorted ideal centres at least `step` apart within [lo, hi], each as near its
// ideal as that allows: pushed right past its left neighbour, then back from the right end.
function spread(ideal: number[], step: number, lo: number, hi: number) {
  const n = ideal.length;
  if (!n) return [];
  const room = hi - lo;
  if ((n - 1) * step > room) step = n > 1 ? room / (n - 1) : step;
  const xs = ideal.map(x => Math.min(hi, Math.max(lo, x)));
  for (let i = 1; i < n; i++) xs[i] = Math.max(xs[i], xs[i - 1] + step);
  xs[n - 1] = Math.min(xs[n - 1], hi);
  for (let i = n - 2; i >= 0; i--) xs[i] = Math.min(xs[i], xs[i + 1] - step);
  xs[0] = Math.max(xs[0], lo);
  for (let i = 1; i < n; i++) xs[i] = Math.max(xs[i], xs[i - 1] + step);
  return xs;
}

// A small picture drawn on a canvas from the picture queue, contained, redrawn when its
// box changes size. A `cutout` (the atlas's pictures) loses its black ground: each pixel
// takes the canvas's text colour, its brightness as its opacity.
function Thumb({ url, rank, cutout = false }: { url: string; rank: number; cutout?: boolean }) {
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
    const draw = () => {
      const box = c.getBoundingClientRect();
      const dpr = Math.min(2, window.devicePixelRatio || 1);
      const w = Math.max(1, Math.round(box.width * dpr)), h = Math.max(1, Math.round(box.height * dpr));
      if (c.width === w && c.height === h && c.dataset.src === img.src) return;
      c.width = w; c.height = h; c.dataset.src = img.src;
      const s = Math.min(w / img.naturalWidth, h / img.naturalHeight);
      const ctx = c.getContext('2d', { willReadFrequently: cutout })!;
      ctx.clearRect(0, 0, w, h);
      ctx.drawImage(img, (w - img.naturalWidth * s) / 2, (h - img.naturalHeight * s) / 2, img.naturalWidth * s, img.naturalHeight * s);
      if (cutout) {
        const [r, g, b] = rgbOf(getComputedStyle(c).color);
        const data = ctx.getImageData(0, 0, w, h), px = data.data;
        for (let i = 0; i < px.length; i += 4) {
          const lum = Math.max(px[i], px[i + 1], px[i + 2]);
          px[i] = r; px[i + 1] = g; px[i + 2] = b;
          px[i + 3] = lum < 12 ? 0 : Math.min(255, (lum - 12) * 1.25);
        }
        ctx.putImageData(data, 0, 0);
      }
    };
    draw();
    const o = new ResizeObserver(draw);
    o.observe(c);
    return () => o.disconnect();
  }, [img]);
  return <canvas ref={ref} className={'cut-canvas' + (cutout ? ' cutout' : '') + (img ? ' ready' : '')} />;
}

function rgbOf(css: string): [number, number, number] {
  const m = css.match(/\d+(\.\d+)?/g);
  return m && m.length >= 3 ? [Number(m[0]), Number(m[1]), Number(m[2])] : [200, 200, 210];
}

// Millimetre ticks: finer when the span is short or zoomed in; a label every few.
function tickMarks(view: Domain, inner: number) {
  const pxPerMm = inner / (view.hi - view.lo);
  const step = pxPerMm > 240 ? 0.1 : pxPerMm > 120 ? 0.25 : pxPerMm > 50 ? 0.5 : 1;
  const every = pxPerMm > 240 ? 0.5 : pxPerMm > 50 ? 1 : pxPerMm > 22 ? 2 : 5;
  const out: { mm: number; label: boolean }[] = [];
  for (let mm = Math.ceil(view.lo / step - 1e-9) * step; mm <= view.hi + 1e-9; mm += step) {
    const r = Math.round(mm * 100) / 100;
    out.push({ mm: r, label: Math.abs(r / every - Math.round(r / every)) < 1e-6 });
  }
  return out;
}

const fmt = (mm: number) => (Number.isInteger(mm) ? String(mm) : mm.toFixed(1));
