// The front-to-back line under the pictures: a dot per section. Before a section is
// placed its dot waits, dim, in stack order along the line; placed, it glides to its
// position over the placed sections' span (half a millimetre beyond them), so the agent
// placing and reshuffling the stack is seen as it happens. One to three moved dots each
// leave a fading mark where they were; a reshuffle of more sets off front to back as a
// wave. The sections the agent is working on are in the accent, the ones shown ringed.
// Hovering names a section.
import { useEffect, useRef, useState } from 'react';
import type { Axis, Row } from '../run';
import { short } from './Stage';

const MOVED_MM = 0.02;
const FEW = 3;

type Ghost = { id: number; x: number };

export function FrontToBack({ rows, axis, focus, shown, done }: {
  rows: Row[]; axis: Axis | null; focus: string[]; shown: string[]; done: boolean;
}) {
  const placed = rows.map(r => r.position_mm).filter((v): v is number => typeof v === 'number');
  const span = placed.length >= 2
    ? { lo: Math.min(...placed) - 0.5, hi: Math.max(Math.max(...placed), Math.min(...placed) + 1) + 0.5 }
    : axis ? { lo: axis.min_mm, hi: axis.max_mm } : null;
  const n = rows.length;
  const xs = new Map(rows.map((r, order) => [
    r.id,
    typeof r.position_mm === 'number' && span
      ? Math.min(100, Math.max(0, (100 * (r.position_mm - span.lo)) / (span.hi - span.lo)))
      : (100 * (order + 0.5)) / n,
  ]));

  // What moved since the rows were last seen: a few each leave a ghost; many are a wave.
  const before = useRef(new Map<string, { mm: number | null; x: number }>());
  const [ghosts, setGhosts] = useState<Ghost[]>([]);
  const [wave, setWave] = useState(false);
  const ghostId = useRef(0);
  useEffect(() => {
    const moved = rows.filter(r => {
      const b = before.current.get(r.id);
      return b && b.mm !== null && typeof r.position_mm === 'number' && Math.abs(b.mm - r.position_mm) > MOVED_MM;
    });
    setWave(moved.length > FEW);
    if (moved.length && moved.length <= FEW) {
      const fresh = moved.map(r => ({ id: ghostId.current++, x: before.current.get(r.id)!.x }));
      setGhosts(g => [...g, ...fresh]);
      setTimeout(() => setGhosts(g => g.filter(x => !fresh.includes(x))), 1400);
    }
    before.current = new Map(rows.map(r => [r.id, { mm: r.position_mm ?? null, x: xs.get(r.id)! }]));
  }, [rows]); // eslint-disable-line react-hooks/exhaustive-deps

  const [tip, setTip] = useState<{ id: string; x: number } | null>(null);
  if (!n) return null;
  const waiting = rows.filter(r => typeof r.position_mm !== 'number').length;
  const ticks = span ? Array.from({ length: Math.floor(span.hi) - Math.ceil(span.lo) + 1 }, (_, i) => Math.ceil(span.lo) + i) : [];
  const some = focus.length < n;
  return (
    <div className="ftb" aria-label="Where the sections sit, front to back">
      <span className="ftb-end">{axis?.low || 'anterior'}</span>
      <div className={'ftb-line' + (wave ? ' wave' : '') + (done ? ' done' : '')}>
        {span && ticks.map(mm => (
          <span key={mm} className="ftb-tick" style={{ left: `${(100 * (mm - span.lo)) / (span.hi - span.lo)}%` }} />
        ))}
        {ghosts.map(g => <span key={g.id} className="ftb-ghost" style={{ left: `${g.x}%` }} />)}
        {rows.map(r => {
          const x = xs.get(r.id)!;
          const cls = ['ftb-dot',
            typeof r.position_mm !== 'number' ? 'unplaced' : '',
            some && focus.includes(r.id) ? 'target' : '',
            shown.includes(r.id) ? 'shown' : ''].filter(Boolean).join(' ');
          return (
            <span key={r.id} className="ftb-slot" style={{ transform: `translateX(${x}%)`, '--x': Math.round(x) } as React.CSSProperties}>
              <span
                className={cls}
                onPointerEnter={() => setTip({ id: r.id, x })}
                onPointerLeave={() => setTip(t => (t?.id === r.id ? null : t))}
              />
            </span>
          );
        })}
        {tip && (
          <span className="ftb-tip" style={{ left: `${tip.x}%` }}>
            <b>{short(tip.id)}</b>{' '}
            {(() => { const r = rows.find(x => x.id === tip.id); return typeof r?.position_mm === 'number' ? `${r.position_mm.toFixed(2)} mm` : 'not placed yet'; })()}
          </span>
        )}
      </div>
      <span className="ftb-end">{axis?.high || 'posterior'}</span>
      <span className="ftb-waiting">{waiting ? (waiting === n ? `${n} sections, not placed yet` : `${waiting} not placed yet`) : ''}</span>
    </div>
  );
}
