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
import { cached, partUrls, picture } from '../data/pictures';
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

export function Act({ act, job, id, edge, renderKey, tiles }: {
  act: ActData | undefined; job: number; id: string; edge: number; renderKey: string | null | undefined; tiles: number;
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
            {zoom && <Zoom key={act!.key} box={zoom} frame={rect} />}
          </AnimatePresence>
          <AnimatePresence>
            {act && act.family !== 'books' && few && <Chip key={act.key} act={act} />}
          </AnimatePresence>
          {act?.running && <i className="act-travel" />}
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

// A zoom: the corner marks close in on the window the agent asked for (fractions of the
// section's picture), then the picture is pushed in until that window fills the tile,
// held while the act lingers, and let back.
const PUSH_MS = 620;
function Zoom({ box, frame }: { box: number[]; frame: Rect }) {
  const ref = useRef<HTMLDivElement>(null);
  const r = { x: box[0] * frame.w, y: box[1] * frame.h, w: (box[2] - box[0]) * frame.w, h: (box[3] - box[1]) * frame.h };
  useEffect(() => {
    const layer = ref.current?.closest<HTMLElement>('.act-layer');
    const pictureEl = layer?.parentElement?.querySelector<HTMLElement>('.picture');
    if (!layer || !pictureEl || matchMedia('(prefers-reduced-motion: reduce)').matches) return;
    const W = layer.clientWidth, H = layer.clientHeight;
    const k = Math.min(W / r.w, H / r.h, 6);
    const cx = frame.x + r.x + r.w / 2, cy = frame.y + r.y + r.h / 2;
    const to = `translate(${W / 2 - cx * k}px, ${H / 2 - cy * k}px) scale(${k})`;
    pictureEl.style.transformOrigin = '0 0';
    const ease = 'cubic-bezier(0.77, 0, 0.175, 1)';
    const push = pictureEl.animate([{ transform: 'none' }, { transform: to }], { duration: PUSH_MS, delay: 420, easing: ease, fill: 'forwards' });
    return () => {
      push.cancel();
      pictureEl.animate([{ transform: to }, { transform: 'none' }], { duration: PUSH_MS * 0.8, easing: ease });
    };
  }, []);
  return (
    <motion.div
      ref={ref}
      className="viewfinder zoomed"
      aria-hidden="true"
      initial={{ left: 0, top: 0, width: frame.w, height: frame.h, opacity: 0 }}
      animate={{ left: r.x, top: r.y, width: r.w, height: r.h, opacity: [0, 1, 1, 0] }}
      exit={{ opacity: 0 }}
      transition={{ duration: 0.5, ease: EASE_IN_OUT, opacity: { duration: 1.0, times: [0, 0.2, 0.6, 1] } }}
    >
      <i /><i /><i /><i />
    </motion.div>
  );
}
