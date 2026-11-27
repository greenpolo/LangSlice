// The agent viewer: Simple or Extended, both following the agent until the person picks a
// section (a dot, a picture, a name in the journal), and Review over either. The keys:
// `v` switches the view, `r` opens Review, `f` follows the agent again.
import { AnimatePresence, motion } from 'motion/react';
import { useCallback, useEffect, useState } from 'react';
import { EASE_OUT } from '../ui/motion';
import { go, useNav } from '../data/nav';
import { useRun } from '../data/run';
import { setStageMode, useStageMode } from '../stage/space/mode';
import { ExtendedView } from './extended/ExtendedView';
import { Lightbox } from './extended/Lightbox';
import { Review } from './Review';
import type { Focus } from './focus';
import { SimpleView } from './SimpleView';

export function Viewer() {
  const nav = useNav();
  const stage = useStageMode();
  const rows = useRun(s => s.rows);
  const agentFocus = useRun(s => s.focus);
  const done = useRun(s => s.status === 'Submitted');
  const comments = useRun(s => s.comments);
  const [picked, setPicked] = useState<string | null>(null);
  const [review, setReview] = useState<string | null | false>(false);
  const [pictures, setPictures] = useState<{ numbers: number[]; start: number } | null>(null);
  const [reviewing, setReviewing] = useState<string | null>(null);
  const agentIds = rows.length
    ? done || !agentFocus?.length
      ? rows.map(r => r.id)
      : [...new Set(agentFocus.filter(id => rows.some(r => r.id === id)))]
    : [];
  const pick = useCallback((id: string) => setPicked(id), []);
  const follow = useCallback(() => setPicked(null), []);
  const focus: Focus = { ids: picked ? [picked] : agentIds, following: !picked, pick, follow };
  useEffect(() => {
    const key = (e: KeyboardEvent) => {
      if (e.ctrlKey || e.metaKey || e.altKey || (e.target as HTMLElement)?.closest('input, textarea, select')) return;
      if (review !== false || pictures) return;
      if (e.key === 'v' || e.key === 'V') go({ view: nav.view === 'simple' ? 'extended' : 'simple' });
      else if (e.key === 'r' || e.key === 'R') setReview(focus.ids[0] ?? null);
      else if (e.key === 'f' || e.key === 'F') follow();
    };
    window.addEventListener('keydown', key);
    return () => window.removeEventListener('keydown', key);
  });
  const open = comments.filter(c => !c.revised).length;
  return (
    <div className={'viewer ' + nav.view}>
      <div className="viewer-controls">
        <AnimatePresence>
          {!focus.following && review === false && (
            <motion.button
              className="ghost-button"
              initial={{ opacity: 0, x: 6 }}
              animate={{ opacity: 1, x: 0 }}
              exit={{ opacity: 0, transition: { duration: 0.12 } }}
              transition={{ duration: 0.2, ease: EASE_OUT }}
              onClick={follow}
            >
              Follow the agent
            </motion.button>
          )}
        </AnimatePresence>
        <button className={'ghost-button review-button' + (done ? ' ready' : '')} aria-pressed={review !== false} onClick={() => setReview(review === false ? focus.ids[0] ?? null : false)}>
          Review{open > 0 && <span className="badge">{open}</span>}
        </button>
        <div className="views-switch" role="group" aria-label="Where the sections are shown">
          <button className={stage === '2d' ? 'on' : ''} aria-pressed={stage === '2d'} onClick={() => setStageMode('2d')}>2D</button>
          <button className={stage === '3d' ? 'on' : ''} aria-pressed={stage === '3d'} onClick={() => setStageMode('3d')}>3D</button>
        </div>
        <div className="views-switch" role="group" aria-label="How much the viewer shows">
          <button className={nav.view === 'simple' ? 'on' : ''} aria-pressed={nav.view === 'simple'} onClick={() => go({ view: 'simple' })}>Simple</button>
          <button className={nav.view === 'extended' ? 'on' : ''} aria-pressed={nav.view === 'extended'} onClick={() => go({ view: 'extended' })}>Extended</button>
        </div>
      </div>
      {nav.view === 'simple'
        ? <SimpleView focus={focus} ring={review === false ? null : reviewing} review={review === false ? null : <Review start={review} onShow={setReviewing} onClose={() => setReview(false)} />} />
        : <ExtendedView focus={focus} ring={review === false ? null : reviewing} onPictures={(numbers, start) => setPictures({ numbers, start })}
            review={review === false ? null : <Review start={review} onShow={setReviewing} onClose={() => setReview(false)} />} />}
      <AnimatePresence>
        {pictures && <Lightbox numbers={pictures.numbers} start={pictures.start} onClose={() => setPictures(null)} />}
      </AnimatePresence>
    </div>
  );
}
