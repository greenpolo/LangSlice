// A picture the agent looked at, large, with its caption; the arrow keys go through the
// pictures it was opened with, Esc closes it.
import { motion } from 'motion/react';
import { useEffect, useState } from 'react';
import { EASE_OUT } from '../../ui/motion';
import { useRun } from '../../data/run';

export function Lightbox({ numbers, start, onClose }: { numbers: number[]; start: number; onClose: () => void }) {
  const job = useRun(s => s.job);
  const log = useRun(s => s.log);
  const [i, setI] = useState(Math.max(0, numbers.indexOf(start)));
  useEffect(() => {
    const key = (e: KeyboardEvent) => {
      if (e.key === 'Escape') onClose();
      else if (e.key === 'ArrowLeft') setI(n => (n - 1 + numbers.length) % numbers.length);
      else if (e.key === 'ArrowRight') setI(n => (n + 1) % numbers.length);
      else return;
      e.preventDefault();
      e.stopPropagation();
    };
    window.addEventListener('keydown', key, true);
    return () => window.removeEventListener('keydown', key, true);
  }, [numbers.length, onClose]);
  const number = numbers[i];
  const about = log.find(e => e.kind === 'picture' && e.number === number) as { title?: string; caption?: string } | undefined;
  if (job === null) return null;
  return (
    <motion.div
      className="lightbox"
      role="dialog"
      aria-label="Picture"
      initial={{ opacity: 0 }}
      animate={{ opacity: 1 }}
      exit={{ opacity: 0 }}
      transition={{ duration: 0.2, ease: EASE_OUT }}
      onClick={e => { if (e.target === e.currentTarget) onClose(); }}
    >
      <motion.figure className="lightbox-card" initial={{ scale: 0.97 }} animate={{ scale: 1 }} transition={{ duration: 0.24, ease: EASE_OUT }}>
        <header className="lightbox-head">
          <span>{about?.title || 'Picture'}</span>
          <span className="lightbox-count">{i + 1} of {numbers.length}</span>
          <button className="ghost-button" onClick={onClose} aria-label="Close">×</button>
        </header>
        <img className="lightbox-image" src={`jobs/${job}/picture/${number}`} alt={about?.title || 'A picture the agent looked at'} />
        {about?.caption && <figcaption className="lightbox-caption">{about.caption}</figcaption>}
      </motion.figure>
    </motion.div>
  );
}
