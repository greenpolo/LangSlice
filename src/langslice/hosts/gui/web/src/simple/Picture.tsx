// One section's picture under its current registration. Each new drawing of it (the agent
// moved the section) is laid over the one before and fades in while the old one stays,
// so the tissue is seen settling into its outline; then the old layer goes. The images
// are drawn on canvases, contained and centred, again whenever the box changes size.
import { motion } from 'motion/react';
import { useEffect, useLayoutEffect, useRef, useState } from 'react';
import { DUR, EASE_IN_OUT } from '../motion';
import { cached, newestOf, picture } from '../pictures';

// A redraw is a symmetric crossfade, so the settling is seen.
const REDRAW = { duration: DUR.morph, ease: EASE_IN_OUT } as const;

type Layer = { url: string; img: HTMLImageElement; n: number };

export function Picture({ id, url, rank, label }: { id: string; url: string; rank: number; label: string }) {
  // The exact picture if it is ready, else the section's newest one until it is.
  const [layers, setLayers] = useState<Layer[]>(() => {
    const exact = cached(url);
    if (exact) return [{ url, img: exact, n: 0 }];
    const any = newestOf(id);
    return any ? [{ url: '', img: any, n: 0 }] : [];
  });
  const count = useRef(1);
  useEffect(() => {
    if (layers.at(-1)?.url === url) return;
    let alive = true;
    picture(url, rank).then(img => {
      if (!alive || !img) return;
      setLayers(old => (old.at(-1)?.url === url ? old : [...old.slice(-1), { url, img, n: count.current++ }]));
    });
    return () => { alive = false; };
  }, [url, rank, layers]);
  return (
    <div className="picture" role="img" aria-label={label}>
      {layers.map((layer, i) => (
        <Canvas
          key={layer.n}
          img={layer.img}
          fresh={layers.length > 1 && i === layers.length - 1}
          onShown={() => setLayers(old => old.slice(-1))}
        />
      ))}
    </div>
  );
}

function Canvas({ img, fresh, onShown }: { img: HTMLImageElement; fresh: boolean; onShown: () => void }) {
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
  return (
    <motion.canvas
      ref={ref}
      className="picture-layer"
      initial={fresh ? { opacity: 0, filter: 'blur(2px)' } : false}
      animate={{ opacity: 1, filter: 'blur(0px)' }}
      transition={REDRAW}
      onAnimationComplete={() => { if (fresh) onShown(); }}
    />
  );
}
