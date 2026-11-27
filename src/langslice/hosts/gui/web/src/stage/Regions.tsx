// The atlas regions the agent's words name ("the hippocampus", "CA1"), outlined in the
// accent over a section's picture while those words are in the bubble.
import { motion } from 'motion/react';
import { useEffect, useState } from 'react';
import { DUR, EASE_OUT } from '../ui/motion';
import { picture } from '../data/pictures';
import { Canvas } from './Picture';

export function Regions({ url }: { url: string }) {
  const [img, setImg] = useState<HTMLImageElement | null>(null);
  useEffect(() => {
    let alive = true;
    picture(url, 0).then(found => { if (alive) setImg(found); });
    return () => { alive = false; };
  }, [url]);
  if (!img) return null;
  return (
    <motion.div className="regions" aria-hidden="true"
      initial={{ opacity: 0 }} animate={{ opacity: 1 }} exit={{ opacity: 0, transition: { duration: DUR.quick } }}
      transition={{ duration: DUR.enter, ease: EASE_OUT }}>
      <Canvas img={img} className="picture-layer" />
    </motion.div>
  );
}
