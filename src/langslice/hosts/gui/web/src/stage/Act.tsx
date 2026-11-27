// What the agent is doing to one section, drawn over its picture (data/acts.ts names the
// families), within the rect the picture is drawn in. A chip in the corner names the
// engine or tool and what it does; a fit makes the atlas borders breathe while it runs
// (the tile's class, act.css) and a light travels along the picture's foot; a look settles a
// viewfinder on the picture; a made image is swept over the tissue by a scan line, so
// preprocessing is seen as what it does to the section. Everything here leaves with the
// act, quickly.
import { AnimatePresence, motion } from 'motion/react';
import { useEffect, useLayoutEffect, useRef, useState } from 'react';
import { DUR, EASE_IN_OUT, EASE_OUT } from '../ui/motion';
import type { Act as ActData } from '../data/acts';
import { cached, partUrls, picture, type PartUrls } from '../data/pictures';
import { Canvas } from './Picture';
import './act.css';

const SWEEP_S = 1.1;

export function actClass(act: ActData | undefined): string {
  if (!act) return '';
  return ` act act-${act.family}` + (act.running ? ' act-running' : '') + (act.failed ? ' act-failed' : '');
}

type Rect = { x: number; y: number; w: number; h: number };

// Where the section's picture is drawn in its tile: contained and centred, as the
// canvases draw it (stage/Picture.tsx), from the tissue's own size.
function useDrawnRect(tissue: string): [React.RefObject<HTMLDivElement | null>, Rect | null] {
  const ref = useRef<HTMLDivElement>(null);
  const [rect, setRect] = useState<Rect | null>(null);
  useLayoutEffect(() => {
    const layer = ref.current;
    if (!layer) return;
    const place = () => {
      const img = cached(tissue);
      const W = layer.clientWidth, H = layer.clientHeight;
      if (!img || !W || !H) { setRect({ x: 0, y: 0, w: W, h: H }); return; }
      const k = Math.min(W / img.naturalWidth, H / img.naturalHeight);
      const w = img.naturalWidth * k, h = img.naturalHeight * k;
      setRect(old => {
        const next = { x: (W - w) / 2, y: (H - h) / 2, w, h };
        return old && Math.abs(old.x - next.x) + Math.abs(old.y - next.y) + Math.abs(old.w - next.w) < 0.5 ? old : next;
      });
    };
    place();
    const o = new ResizeObserver(place);
    o.observe(layer);
    const t = window.setTimeout(place, 400);
    return () => { o.disconnect(); clearTimeout(t); };
  }, [tissue]);
  return [ref, rect];
}

// The few tiles (FEW at most) carry a chip and a viewfinder each; a grid shows its act
// on the pictures alone, and the line under the bubble names it once.
const FEW = 4;

export function Act({ act, job, id, edge, renderKey, tiles, pointed = false }: {
  act: ActData | undefined; job: number; id: string; edge: number; renderKey: string | null | undefined; tiles: number;
  pointed?: boolean;
}) {
  const tissue = partUrls(job, id, edge, renderKey).tissue;
  const [ref, rect] = useDrawnRect(tissue);
  const made = act && act.tool === 'make_image' && !act.failed && !act.args.delete ? String(act.args.name ?? '') : '';
  const few = tiles <= FEW;
  const zoom = act?.family === 'look' ? act.zooms[id] : undefined;
  return (
    <div className="act-layer" ref={ref}>
      {rect && (
        <div className="act-frame" style={{ left: rect.x, top: rect.y, width: rect.w, height: rect.h }}>
          <AnimatePresence>
            {made && <MadeImage key={act!.key + made} url={partUrls(job, id, edge, renderKey, made).tissue} />}
          </AnimatePresence>
          <AnimatePresence>
            {act?.family === 'look' && !zoom && few && <Viewfinder key={act.key} />}
            {zoom && <Zoom key={`${act!.key}:${Math.round(rect.w)}x${Math.round(rect.h)}`} held={`${act!.key}:${id}`} box={zoom} frame={rect} hold={act!.hold} sharp={partUrls(job, id, SHARP_EDGE, renderKey)} />}
          </AnimatePresence>
          <AnimatePresence>
            {act && act.family !== 'books' && few && <Chip key={act.key} act={act} />}
          </AnimatePresence>
          {act?.running && <i className="act-travel" />}
          <AnimatePresence>
            {pointed && (
              <motion.i
                key="ring"
                className="pointer-ring"
                aria-hidden="true"
                initial={{ opacity: 0, scale: 1.03 }}
                animate={{ opacity: 1, scale: 1 }}
                exit={{ opacity: 0, transition: { duration: DUR.quick, ease: EASE_OUT } }}
                transition={{ duration: DUR.enter, ease: EASE_OUT }}
              />
            )}
          </AnimatePresence>
        </div>
      )}
    </div>
  );
}

// The engine or tool, and what it does, in the picture's top-left corner; a pulse while
// the call runs, a tick once it has landed, a cross when it failed.
function Chip({ act }: { act: ActData }) {
  const state = act.failed ? 'failed' : act.running ? 'running' : 'done';
  return (
    <motion.div
      className={'act-chip act-chip-' + state}
      initial={{ opacity: 0, y: -4 }}
      animate={{ opacity: 1, y: 0 }}
      exit={{ opacity: 0, transition: { duration: DUR.quick, ease: EASE_OUT } }}
      transition={{ duration: DUR.quick, ease: EASE_OUT }}
    >
      <span className="act-glyph" aria-hidden="true">
        {state === 'done' ? <svg viewBox="0 0 12 12"><path d="M2.5 6.4 5 8.8 9.6 3.6" /></svg>
          : state === 'failed' ? <svg viewBox="0 0 12 12"><path d="M3.5 3.5 8.5 8.5M8.5 3.5 3.5 8.5" /></svg>
            : <i />}
      </span>
      <span className="act-engine">{act.engine}</span>
      {act.what && <span className="act-what">{act.what}</span>}
    </motion.div>
  );
}

// Four corner marks that close in on the picture: the agent is looking at it.
function Viewfinder() {
  return (
    <motion.div
      className="viewfinder"
      aria-hidden="true"
      initial={{ opacity: 0, scale: 1.06 }}
      animate={{ opacity: 1, scale: 1 }}
      exit={{ opacity: 0, scale: 1.02, transition: { duration: DUR.quick, ease: EASE_OUT } }}
      transition={{ duration: DUR.enter, ease: EASE_OUT }}
    >
      <i /><i /><i /><i />
    </motion.div>
  );
}

// The made image over the section, uncovered from the top by a scan line, held while the
// act lingers, then faded back to the section's own picture.
function MadeImage({ url }: { url: string }) {
  const [img, setImg] = useState<HTMLImageElement | null>(null);
  useEffect(() => {
    let alive = true;
    picture(url, 0).then(found => { if (alive) setImg(found); });
    return () => { alive = false; };
  }, [url]);
  if (!img) return null;
  return (
    <motion.div className="made" initial={{ opacity: 1 }} exit={{ opacity: 0, transition: { duration: DUR.morph, ease: EASE_IN_OUT } }}>
      <motion.div
        className="made-image"
        initial={{ clipPath: 'inset(0 0 100% 0)' }}
        animate={{ clipPath: 'inset(0 0 0% 0)' }}
        transition={{ duration: SWEEP_S, ease: EASE_IN_OUT }}
      >
        <Canvas img={img} className="picture-layer" />
      </motion.div>
      <motion.i
        className="scanline"
        initial={{ top: '0%', opacity: 1 }}
        animate={{ top: '100%', opacity: [1, 1, 0] }}
        transition={{ duration: SWEEP_S, ease: EASE_IN_OUT, opacity: { duration: SWEEP_S, times: [0, 0.85, 1] } }}
      />
    </motion.div>
  );
}

// A zoom, on one timeline (ZOOM_MS): the corner marks close in on the window the agent
// asked for (fractions of the section's picture), the picture is pushed in until that
// window fills the tile, held, and let back. While its batch is still being read (`hold`)
// the timeline waits at HOLD_AT, pushed in, so the batch's zooms are seen together. The
// picture always comes back by itself, and at once when the act is cut short.
const ZOOM_MS = 2800;
const HOLD_AT = 0.6;
const PUSHED_UNTIL = 0.78;
// The pushed-in picture is laid over with the window drawn again at this edge, so it is
// sharp, not the small picture scaled up.
const SHARP_EDGE = 2048;
const ZOOM_EASE = 'cubic-bezier(0.77, 0, 0.175, 1)';
// The zooms that reached their pushed-in hold, by act and section: when the stage lays
// its tiles out again while a batch is held (another section joins it), the zoom is drawn
// again at the new size, already pushed in, not played from the start.
const PUSHED = new Set<string>();
function Zoom({ box, frame, sharp, hold, held }: { box: number[]; frame: Rect; sharp: PartUrls; hold: boolean; held: string }) {
  const ref = useRef<HTMLDivElement>(null);
  const holding = useRef(hold);
  holding.current = hold;
  const timeline = useRef<Animation[]>([]);
  const lens = useRef<HTMLCanvasElement>(null);
  const r = { x: box[0] * frame.w, y: box[1] * frame.h, w: (box[2] - box[0]) * frame.w, h: (box[3] - box[1]) * frame.h };
  useEffect(() => {
    const marks = ref.current;
    const layer = marks?.closest<HTMLElement>('.act-layer');
    const pictureEl = layer?.parentElement?.querySelector<HTMLElement>('.picture');
    if (!marks || !layer || !pictureEl) return;
    const whole = { left: '0px', top: '0px', width: `${frame.w}px`, height: `${frame.h}px` };
    const asked = { left: `${r.x}px`, top: `${r.y}px`, width: `${r.w}px`, height: `${r.h}px` };
    const closing = marks.animate([
      { ...whole, opacity: 0, offset: 0 },
      { ...whole, opacity: 1, offset: 0.06 },
      { ...asked, opacity: 1, offset: 0.24, easing: ZOOM_EASE },
      { ...asked, opacity: 0, offset: 0.36 },
      { ...asked, opacity: 0, offset: 1 },
    ], { duration: ZOOM_MS, easing: 'linear' });
    let push: Animation | null = null;
    if (!matchMedia('(prefers-reduced-motion: reduce)').matches) {
      const W = layer.clientWidth, H = layer.clientHeight;
      const k = Math.min(W / r.w, H / r.h, 6);
      const cx = frame.x + r.x + r.w / 2, cy = frame.y + r.y + r.h / 2;
      const to = `translate(${W / 2 - cx * k}px, ${H / 2 - cy * k}px) scale(${k})`;
      pictureEl.style.transformOrigin = '0 0';
      push = pictureEl.animate([
        { transform: 'none', offset: 0 },
        { transform: 'none', offset: 0.22, easing: ZOOM_EASE },
        { transform: to, offset: 0.44 },
        { transform: to, offset: PUSHED_UNTIL, easing: ZOOM_EASE },
        { transform: 'none', offset: 1 },
      ], { duration: ZOOM_MS });
    }
    let alive = true, shown: Animation | null = null;
    const started = performance.now();
    Promise.all([picture(sharp.tissue, 0), picture(sharp.borders, 0)]).then(([tissue, borders]) => {
      const canvas = lens.current;
      const left = ZOOM_MS * 0.74 - (performance.now() - started);
      if (!alive || !tissue || !canvas || !push || (left < 300 && !holding.current)) return;
      const W = layer.clientWidth, H = layer.clientHeight;
      const dpr = Math.min(2, window.devicePixelRatio || 1);
      canvas.width = Math.round(W * dpr); canvas.height = Math.round(H * dpr);
      canvas.style.width = `${W}px`; canvas.style.height = `${H}px`;
      const k = Math.min(W / r.w, H / r.h, 6);
      const ctx = canvas.getContext('2d')!;
      const sx = tissue.naturalWidth / frame.w, sy = tissue.naturalHeight / frame.h;
      const cw = W / k, ch = H / k;
      const x0 = r.x + r.w / 2 - cw / 2, y0 = r.y + r.h / 2 - ch / 2;
      ctx.imageSmoothingQuality = 'high';
      for (const img of [tissue, borders]) {
        if (img) ctx.drawImage(img, x0 * sx, y0 * sy, cw * sx, ch * sy, 0, 0, canvas.width, canvas.height);
      }
      shown = canvas.animate([{ opacity: 0, offset: 0 }, { opacity: 0, offset: 0.44 }, { opacity: 1, offset: 0.53 },
        { opacity: 1, offset: 0.74 }, { opacity: 0, offset: 0.8 }, { opacity: 0, offset: 1 }],
        { duration: ZOOM_MS, easing: 'linear' });
      shown.currentTime = push.currentTime;
      if (push.playState === 'paused') shown.pause();
      timeline.current.push(shown);
    });
    timeline.current = [closing, ...(push ? [push] : [])];
    const holdNow = () => {
      if (!holding.current) return;
      PUSHED.add(held);
      for (const a of timeline.current) { a.currentTime = ZOOM_MS * HOLD_AT; a.pause(); }
    };
    if (PUSHED.has(held) && holding.current) holdNow();
    const pause = window.setTimeout(holdNow, PUSHED.has(held) ? 0 : ZOOM_MS * HOLD_AT);
    return () => { alive = false; clearTimeout(pause); closing.cancel(); push?.cancel(); shown?.cancel(); };
  }, []);
  useEffect(() => {
    for (const a of timeline.current) {
      const t = Number(a.currentTime ?? 0);
      if (!hold && a.playState === 'paused') a.play();
      else if (hold && a.playState === 'running' && t >= ZOOM_MS * HOLD_AT && t < ZOOM_MS * PUSHED_UNTIL) a.pause();
    }
  }, [hold]);
  return (
    <>
      <canvas ref={lens} className="zoom-lens" aria-hidden="true" style={{ left: -frame.x, top: -frame.y }} />
      <div ref={ref} className="viewfinder zoomed" aria-hidden="true"><i /><i /><i /><i /></div>
    </>
  );
}
