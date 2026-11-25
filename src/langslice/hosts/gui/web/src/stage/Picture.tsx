// One section's picture under its current registration, as three parts laid over one
// another (data/pictures.ts partUrls): the tissue, the atlas borders, and what the job
// marked on it. Each new drawing of it (the agent moved or marked the section) is laid
// over the one before and fades in while the old one stays, so the tissue is seen
// settling into its outline; then the old set goes. A damage mark the agent has just made
// is drawn in by a diagonal wipe. The parts are drawn on canvases, contained and centred,
// again whenever the box changes size; the borders are a canvas of their own so a fit can
// make them breathe over still tissue (stage/Act.tsx sets the class).
import { motion } from 'motion/react';
import { useEffect, useLayoutEffect, useRef, useState } from 'react';
import { DUR, EASE_IN_OUT } from '../ui/motion';
import { cached, newestOf, picture, type Part, type PartUrls } from '../data/pictures';

// A redraw is a symmetric crossfade, so the settling is seen.
const REDRAW = { duration: DUR.morph, ease: EASE_IN_OUT } as const;
const PARTS: Part[] = ['tissue', 'borders', 'marks'];
// A picture the server could not hand over yet (busy past the loader's tries) is asked
// for again after this long, so a large tile never stays on a small picture scaled up.
const RETRY_MS = 1500;

type Images = Partial<Record<Part, HTMLImageElement>>;
type Drawn = { key: string; images: Images; n: number };

function ready(urls: PartUrls): Images | null {
  const images: Images = {};
  for (const part of PARTS) {
    const img = cached(urls[part]);
    if (!img) return null;
    images[part] = img;
  }
  return images;
}

export function Picture({ id, urls, rank, label, reveal = false }: {
  id: string; urls: PartUrls; rank: number; label: string; reveal?: boolean;
}) {
  // The exact parts if they are ready, else the section's newest tissue until they are.
  const [sets, setSets] = useState<Drawn[]>(() => {
    const exact = ready(urls);
    if (exact) return [{ key: urls.tissue, images: exact, n: 0 }];
    const any = newestOf(id);
    return any ? [{ key: '', images: { tissue: any }, n: 0 }] : [];
  });
  const count = useRef(1);
  const [attempt, setAttempt] = useState(0);
  const key = urls.tissue;
  useEffect(() => {
    if (sets.at(-1)?.key === key) return;
    let alive = true;
    let again = 0;
    Promise.all(PARTS.map(part => picture(urls[part], rank))).then(([tissue, borders, marks]) => {
      if (!alive) return;
      if (!tissue) { again = window.setTimeout(() => setAttempt(n => n + 1), RETRY_MS); return; }
      const images: Images = { tissue, borders: borders ?? undefined, marks: marks ?? undefined };
      setSets(old => (old.at(-1)?.key === key ? old : [...old.slice(-1), { key, images, n: count.current++ }]));
    });
    return () => { alive = false; clearTimeout(again); };
  }, [key, rank, sets, attempt]);
  return (
    <div className="picture" role="img" aria-label={label}>
      {sets.map((set, i) => {
        const fresh = sets.length > 1 && i === sets.length - 1;
        return (
          <motion.div
            key={set.n}
            className="picture-set"
            data-src={set.key}
            initial={fresh ? { opacity: 0, filter: 'blur(2px)' } : false}
            animate={{ opacity: 1, filter: 'blur(0px)' }}
            transition={REDRAW}
            onAnimationComplete={() => { if (fresh) setSets(old => old.slice(-1)); }}
          >
            {PARTS.map(part => set.images[part] && (
              <Canvas key={part} img={set.images[part]!}
                className={`picture-layer picture-${part}` + (part === 'marks' && fresh && reveal ? ' reveal' : '')} />
            ))}
          </motion.div>
        );
      })}
    </div>
  );
}

export function Canvas({ img, className }: { img: HTMLImageElement; className: string }) {
  const ref = useRef<HTMLCanvasElement>(null);
  useLayoutEffect(() => {
    const canvas = ref.current!;
    const draw = () => {
      const box = canvas.parentElement!.getBoundingClientRect();
      const dpr = Math.min(2, window.devicePixelRatio || 1);
      const w = Math.max(1, Math.round(box.width * dpr)), h = Math.max(1, Math.round(box.height * dpr));
      if (canvas.width !== w || canvas.height !== h) { canvas.width = w; canvas.height = h; }
      const ctx = canvas.getContext('2d')!;
      ctx.clearRect(0, 0, w, h);
      const s = Math.min(w / img.naturalWidth, h / img.naturalHeight);
      const dw = img.naturalWidth * s, dh = img.naturalHeight * s;
      ctx.imageSmoothingQuality = 'high';
      ctx.drawImage(img, (w - dw) / 2, (h - dh) / 2, dw, dh);
    };
    draw();
    const observer = new ResizeObserver(draw);
    observer.observe(canvas.parentElement!);
    return () => observer.disconnect();
  }, [img]);
  return <canvas ref={ref} className={className} />;
}
