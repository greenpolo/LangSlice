// The 3D view's volume: the scene (`scene.ts`) in a box, fed from the server. The atlas
// brain's mask (`jobs/<n>/shell`) is asked for once per job and meshed here; the sections'
// corners (`jobs/<n>/space`) again whenever a section's registration changes, at most every
// SPACE_MS; each section's quiet thumbnail through the window's picture queue, after the
// stage's pictures. A slab the person clicks is picked like a picture.
import { AnimatePresence, motion } from 'motion/react';
import { useEffect, useRef, useState } from 'react';
import { EASE_OUT } from '../motion';
import { picture } from '../pictures';
import type { Row, Strip } from '../run';
import { short } from '../simple/Stage';
import { Space, type Placed } from './scene';
import { surface, type Shell } from './surface';

const SPACE_MS = 1500;
const THUMB_EDGE = 256;
const THUMB_RANK = 500;

type SpaceAnswer = { thickness_um: number; given_thickness: boolean; plane_axis: number; sections: Placed[] };

export function thumbUrl(job: number, id: string, key: string) {
  return `jobs/${job}/render/thumb/${encodeURIComponent(id)}?edge=${THUMB_EDGE}&key=${encodeURIComponent(key)}&style=calm`;
}

export function Volume({ job, rows, strip, focus, onPick }: {
  job: number; rows: Row[]; strip: Strip | null; focus: string[]; onPick: (id: string) => void;
}) {
  const host = useRef<HTMLDivElement>(null);
  const space = useRef<Space | null>(null);
  const [ready, setReady] = useState(false);
  const [touched, setTouched] = useState(false);
  const [hover, setHover] = useState<{ id: string; x: number; y: number } | null>(null);
  const [facts, setFacts] = useState<SpaceAnswer | null>(null);
  const pick = useRef(onPick);
  pick.current = onPick;

  useEffect(() => {
    const reduced = window.matchMedia('(prefers-reduced-motion: reduce)').matches;
    const s = new Space(host.current!, reduced, {
      pick: id => pick.current(id),
      hover: (id, x, y) => setHover(old => (id === null ? null : old?.id === id && Math.abs(old.x - x) + Math.abs(old.y - y) < 2 ? old : { id, x, y })),
      touched: () => setTouched(true),
    });
    space.current = s;
    return () => { s.dispose(); space.current = null; };
  }, []);

  useEffect(() => {
    let alive = true;
    setReady(false);
    fetch(`jobs/${job}/shell`, { cache: 'no-store' })
      .then(r => (r.ok ? r.json() : null))
      .then((shell: Shell | null) => {
        if (!alive || !shell || !space.current) return;
        space.current.setShell(shell, surface(shell));
        setReady(true);
      })
      .catch(() => {});
    return () => { alive = false; };
  }, [job]);

  const keys = rows.map(r => r.id + ':' + (r.render_key ?? '')).join('|');
  const asked = useRef(0);
  useEffect(() => {
    if (!ready) return;
    let alive = true;
    const go = () => {
      asked.current = Date.now();
      fetch(`jobs/${job}/space`, { cache: 'no-store' })
        .then(r => (r.ok ? r.json() : null))
        .then((answer: SpaceAnswer | null) => {
          if (!alive || !answer || !space.current) return;
          space.current.setSections(answer.sections, answer.thickness_um);
          setFacts(answer);
        })
        .catch(() => {});
    };
    const wait = asked.current + SPACE_MS - Date.now();
    const timer = window.setTimeout(go, Math.max(0, wait));
    return () => { alive = false; clearTimeout(timer); };
  }, [job, keys, ready]);

  const thumbs = strip?.sections.map(s => s.id + ':' + s.thumb_key).join('|') ?? '';
  useEffect(() => {
    if (!facts || !strip) return;
    let alive = true;
    strip.sections.forEach((s, i) => {
      const url = thumbUrl(job, s.id, s.thumb_key);
      picture(url, THUMB_RANK + i).then(img => { if (alive && img) space.current?.setImage(s.id, url, img); });
    });
    return () => { alive = false; };
  }, [job, thumbs, facts !== null]); // eslint-disable-line react-hooks/exhaustive-deps

  const focusKey = focus.join('|');
  useEffect(() => { space.current?.setFocus(focus); }, [focusKey]); // eslint-disable-line react-hooks/exhaustive-deps

  const placed = facts ? facts.sections.filter(s => s.corners_um).length : 0;
  const row = hover ? rows.find(r => r.id === hover.id) : undefined;
  return (
    <div className="volume" ref={host}>
      <AnimatePresence>
        {!ready && (
          <motion.p key="wait" className="volume-wait" initial={{ opacity: 0 }} animate={{ opacity: 1 }} exit={{ opacity: 0 }} transition={{ duration: 0.3, ease: EASE_OUT }}>
            Drawing the atlas brain…
          </motion.p>
        )}
      </AnimatePresence>
      {facts && (
        <p className="volume-facts">
          <b>{placed}</b> of {facts.sections.length} placed
          <span className="volume-thick">{facts.given_thickness ? `${fmt(facts.thickness_um)} µm sections` : `${fmt(facts.thickness_um)} µm sections (assumed)`}</span>
        </p>
      )}
      <AnimatePresence>
        {ready && !touched && (
          <motion.p key="hint" className="volume-hint" initial={{ opacity: 0 }} animate={{ opacity: 1, transition: { delay: 1.2, duration: 0.4 } }} exit={{ opacity: 0, transition: { duration: 0.2 } }}>
            Drag to turn, scroll to zoom
          </motion.p>
        )}
      </AnimatePresence>
      {hover && (
        <div className="volume-tip" style={{ transform: `translate(${Math.round(hover.x + 14)}px, ${Math.round(hover.y + 12)}px)` }}>
          {short(hover.id)}
          <span>{typeof row?.position_mm === 'number' ? `${row.position_mm.toFixed(2)} mm` : 'not placed'}</span>
        </div>
      )}
    </div>
  );
}

const fmt = (um: number) => (Number.isInteger(um) ? String(um) : um.toFixed(1));
