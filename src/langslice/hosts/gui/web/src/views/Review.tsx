// Review: one section at a time, every section in stack order. The arrow keys (or the
// pager) flip through them; the wheel zooms about the pointer and a double-click resets;
// dragging on the picture draws a box, and a short comment typed for it is kept in the job
// (`review.json`), where the agent's `review_comments` tool reads it. A comment's box is
// a fraction of the picture as drawn (its content, contained in the frame), so it lands
// on the same tissue at any size or zoom. A comment whose section the agent has changed
// since shows as revised, and the section's new picture fades in over the old one
// (`REDRAW_MS`), so what the agent changed is seen.
import { AnimatePresence, motion } from 'motion/react';
import { useCallback, useEffect, useLayoutEffect, useRef, useState } from 'react';
import { DUR, EASE_OUT } from '../ui/motion';
import { partUrls, picture, snapEdge } from '../data/pictures';
import { sendComment, useRun, type Comment } from '../data/run';
import { short } from '../data/run';

type View = { s: number; x: number; y: number };
type Rect = { left: number; top: number; width: number; height: number };
const MAX_ZOOM = 24;
const REDRAW_MS = 1100;

export function Review({ start, onClose, onShow }: { start: string | null; onClose: () => void; onShow?: (id: string) => void }) {
  const rows = useRun(s => s.rows);
  const job = useRun(s => s.job);
  const comments = useRun(s => s.comments);
  const [index, setIndex] = useState(() => Math.max(0, rows.findIndex(r => r.id === start)));
  const row = rows[Math.min(index, rows.length - 1)];
  const flip = useCallback((d: number) => setIndex(i => (i + d + rows.length) % Math.max(1, rows.length)), [rows.length]);
  useEffect(() => {
    const key = (e: KeyboardEvent) => {
      if ((e.target as HTMLElement)?.closest('input, textarea')) return;
      if (e.key === 'ArrowLeft') { flip(-1); e.preventDefault(); }
      else if (e.key === 'ArrowRight') { flip(1); e.preventDefault(); }
      else if (e.key === 'Escape') onClose();
    };
    window.addEventListener('keydown', key);
    return () => window.removeEventListener('keydown', key);
  }, [flip, onClose]);
  useEffect(() => { if (row) onShow?.(row.id); }, [row?.id]);
  if (!row || job === null) return null;
  const mine = comments.filter(c => c.section === row.id);
  return (
    <section className="review" aria-label="Review">
      <header className="review-bar">
        <button className="ghost-button" onClick={() => flip(-1)} aria-label="Previous section">‹</button>
        <span className="review-name">{short(row.id)}</span>
        {typeof row.position_mm === 'number' && <span className="review-where">{row.position_mm.toFixed(2)} mm</span>}
        <span className="review-count">{index + 1} of {rows.length}</span>
        <button className="ghost-button" onClick={() => flip(1)} aria-label="Next section">›</button>
        <span className="review-hint">Drag a box on the picture to comment · ← → to flip</span>
        <button className="ghost-button review-close" onClick={onClose}>Done</button>
      </header>
      <ReviewPicture key={row.id} job={job} id={row.id} renderKey={row.render_key} comments={mine} />
    </section>
  );
}

function ReviewPicture({ job, id, renderKey, comments }: { job: number; id: string; renderKey?: string | null; comments: Comment[] }) {
  const frame = useRef<HTMLDivElement>(null);
  const canvas = useRef<HTMLCanvasElement>(null);
  const ghost = useRef<HTMLCanvasElement>(null);
  const shown = useRef<HTMLImageElement | null>(null);
  const extra = useRef<HTMLImageElement[]>([]);
  const shownKey = useRef<string | null | undefined>(undefined);
  const [rect, setRect] = useState<Rect | null>(null);
  const [view, setView] = useState<View>({ s: 1, x: 0, y: 0 });
  const [draft, setDraft] = useState<number[] | null>(null);
  const [composing, setComposing] = useState(false);
  const [text, setText] = useState('');
  const [problem, setProblem] = useState('');
  const start = useRef<{ x: number; y: number } | null>(null);

  // The picture, at the frame's size, and where its content lies in the frame.
  useLayoutEffect(() => {
    const el = frame.current!;
    let alive = true;
    const draw = (image: HTMLImageElement) => {
      const box = el.getBoundingClientRect();
      const dpr = Math.min(2, window.devicePixelRatio || 1);
      const c = canvas.current!;
      const before = ghost.current;
      if (before && shown.current && shownKey.current !== renderKey && c.width > 0 && !window.matchMedia('(prefers-reduced-motion: reduce)').matches) {
        before.width = c.width; before.height = c.height;
        before.getContext('2d')!.drawImage(c, 0, 0);
        before.animate([{ opacity: 1 }, { opacity: 0 }], { duration: REDRAW_MS, easing: 'cubic-bezier(0.45, 0, 0.55, 1)' });
      }
      shown.current = image;
      shownKey.current = renderKey;
      c.width = Math.round(box.width * dpr); c.height = Math.round(box.height * dpr);
      const s = Math.min(box.width / image.naturalWidth, box.height / image.naturalHeight);
      const r = { width: image.naturalWidth * s, height: image.naturalHeight * s, left: 0, top: 0 };
      r.left = (box.width - r.width) / 2; r.top = (box.height - r.height) / 2;
      const ctx = c.getContext('2d')!;
      ctx.imageSmoothingQuality = 'high';
      ctx.clearRect(0, 0, c.width, c.height);
      for (const layer of [image, ...extra.current]) ctx.drawImage(layer, r.left * dpr, r.top * dpr, r.width * dpr, r.height * dpr);
      setRect(r);
    };
    const load = () => {
      const box = el.getBoundingClientRect();
      const urls = partUrls(job, id, snapEdge(Math.max(box.width, box.height)), renderKey);
      Promise.all([picture(urls.tissue, 0), picture(urls.borders, 0), picture(urls.marks, 0)]).then(([image, borders, marks]) => {
        if (!alive || !image) return;
        extra.current = [borders, marks].filter((x): x is HTMLImageElement => !!x);
        draw(image);
      });
    };
    load();
    const o = new ResizeObserver(() => { if (shown.current) draw(shown.current); load(); });
    o.observe(el);
    return () => { alive = false; o.disconnect(); };
  }, [job, id, renderKey]);

  // A pointer's place as fractions of the picture's content, through the zoom.
  const fraction = (e: React.PointerEvent | React.WheelEvent) => {
    const box = frame.current!.getBoundingClientRect();
    if (!rect) return null;
    const x = (e.clientX - box.left - view.x) / view.s, y = (e.clientY - box.top - view.y) / view.s;
    const clamp = (n: number) => Math.min(1, Math.max(0, n));
    return { x: clamp((x - rect.left) / rect.width), y: clamp((y - rect.top) / rect.height) };
  };
  const onWheel = (e: React.WheelEvent) => {
    const box = frame.current!.getBoundingClientRect();
    const px = e.clientX - box.left, py = e.clientY - box.top;
    setView(v => {
      const s = Math.min(MAX_ZOOM, Math.max(1, v.s * Math.exp(-e.deltaY * 0.0018)));
      let x = px - ((px - v.x) * s) / v.s, y = py - ((py - v.y) * s) / v.s;
      x = Math.min(0, Math.max(box.width - box.width * s, x));
      y = Math.min(0, Math.max(box.height - box.height * s, y));
      return { s, x, y };
    });
  };
  const onDown = (e: React.PointerEvent) => {
    if (e.button !== 0 || composing || (e.target as HTMLElement).closest('.mark-chip, .composer')) return;
    const at = fraction(e);
    if (!at) return;
    start.current = at;
    setDraft(null);
    (e.target as HTMLElement).setPointerCapture(e.pointerId);
  };
  const onMove = (e: React.PointerEvent) => {
    if (!start.current) return;
    const at = fraction(e), s = start.current;
    if (at) setDraft([Math.min(s.x, at.x), Math.min(s.y, at.y), Math.max(s.x, at.x), Math.max(s.y, at.y)]);
  };
  const onUp = () => {
    if (!start.current) return;
    start.current = null;
    if (draft && draft[2] - draft[0] > 0.01 && draft[3] - draft[1] > 0.01) { setComposing(true); setText(''); setProblem(''); }
    else setDraft(null);
  };
  const cancel = () => { setDraft(null); setComposing(false); setText(''); setProblem(''); };
  const keep = async () => {
    if (!draft || !text.trim()) return;
    const refused = await sendComment({ section: id, box: draft, text: text.trim() });
    if (refused) setProblem(refused); else cancel();
  };
  const place = (box: number[]): React.CSSProperties | undefined => rect ? {
    left: rect.left + box[0] * rect.width, top: rect.top + box[1] * rect.height,
    width: (box[2] - box[0]) * rect.width, height: (box[3] - box[1]) * rect.height,
  } : undefined;
  return (
    <div
      className="review-frame"
      ref={frame}
      onWheel={onWheel}
      onPointerDown={onDown}
      onPointerMove={onMove}
      onPointerUp={onUp}
      onPointerCancel={cancel}
      onDoubleClick={e => { if (!(e.target as HTMLElement).closest('.composer')) setView({ s: 1, x: 0, y: 0 }); }}
    >
      <div className="review-layer" style={{ transform: `translate(${view.x}px, ${view.y}px) scale(${view.s})` }}>
        <canvas ref={canvas} className="review-canvas" />
        <canvas ref={ghost} className="review-canvas review-ghost" aria-hidden="true" />
        <AnimatePresence>
          {comments.map(c => (
            <motion.div
              key={c.id}
              className={'mark' + (c.revised ? ' revised' : '')}
              style={place(c.box)}
              initial={{ opacity: 0 }}
              animate={{ opacity: 1 }}
              exit={{ opacity: 0, transition: { duration: DUR.quick } }}
              transition={{ duration: DUR.enter, ease: EASE_OUT }}
            >
              <span className="mark-chip">
                {c.revised && <span className="mark-state">✓ Revised</span>}
                <span className="mark-text">{c.text}</span>
                <button className="mark-remove" aria-label={'Remove the comment ' + c.text} onClick={() => sendComment({ remove: c.id })}>×</button>
              </span>
            </motion.div>
          ))}
        </AnimatePresence>
        {draft && <div className="mark draft" style={place(draft)} />}
      </div>
      <AnimatePresence>
        {composing && draft && rect && (
          <motion.form
            className="composer"
            style={{
              left: Math.min(Math.max(8, view.x + (rect.left + draft[0] * rect.width) * view.s), (frame.current?.clientWidth ?? 400) - 348),
              top: Math.max(8, view.y + (rect.top + draft[1] * rect.height) * view.s - 84),
            }}
            initial={{ opacity: 0, y: 4, scale: 0.98 }}
            animate={{ opacity: 1, y: 0, scale: 1 }}
            exit={{ opacity: 0, transition: { duration: 0.12 } }}
            transition={{ duration: 0.18, ease: EASE_OUT }}
            onSubmit={e => { e.preventDefault(); keep(); }}
          >
            <input
              autoFocus
              className="composer-input"
              maxLength={500}
              placeholder="What is off here?"
              aria-label="Your comment on the box"
              value={text}
              onChange={e => setText(e.target.value)}
              onKeyDown={e => { if (e.key === 'Escape') { e.preventDefault(); e.stopPropagation(); cancel(); } }}
            />
            <span className="composer-note">{problem || 'Enter to keep · Esc to cancel'}</span>
          </motion.form>
        )}
      </AnimatePresence>
    </div>
  );
}
