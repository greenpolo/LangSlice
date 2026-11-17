// The Extended view, for a person who wants the whole run: the speaker band with how far
// the registration is (placed, aligned, warped); the sections in focus, four to a page,
// each with its name, position and how it stands, or the sections table; the
// front-to-back line under them; and on the right the journal.
import { motion } from 'motion/react';
import type { ReactNode } from 'react';
import { useState } from 'react';
import { EASE_OUT } from '../motion';
import { useRun, type Row } from '../run';
import { CuttingAxis } from '../simple/CuttingAxis';
import { Speaker } from '../simple/Speaker';
import { short, Stage, useBox, useDirector } from '../simple/Stage';
import type { Focus } from '../Viewer';
import { Journal } from './Journal';

const PAGE = 4;

export function ExtendedView({ focus, review, ring, onPictures }: {
  focus: Focus; review: ReactNode; ring: string | null; onPictures: (numbers: number[], start: number) => void;
}) {
  const run = useRun();
  const [stageRef, box] = useBox<HTMLDivElement>();
  const [mode, setMode] = useState<'pictures' | 'table'>('pictures');
  const [page, setPage] = useState(0);
  const pages = Math.max(1, Math.ceil(focus.ids.length / PAGE));
  const onPage = focus.ids.slice(Math.min(page, pages - 1) * PAGE, Math.min(page, pages - 1) * PAGE + PAGE);
  const shown = useDirector(run.job, onPage, run.rows, box);
  const done = run.status === 'Submitted';
  const running = [...run.calls].reverse().find(c => c.running);
  const call = running ?? run.calls.at(-1) ?? null;
  return (
    <main className="extended">
      <div className="extended-head">
        <Speaker model={run.model} status={run.status} words={run.words} call={call} focusCount={shown.length} />
        <Meters rows={run.rows} />
      </div>
      <div className="extended-body">
        <section className="extended-main">
          <div className="extended-tools">
            <div className="segmented" role="group" aria-label="What this area shows">
              <button className={mode === 'pictures' ? 'on' : ''} aria-pressed={mode === 'pictures'} onClick={() => setMode('pictures')}>Pictures</button>
              <button className={mode === 'table' ? 'on' : ''} aria-pressed={mode === 'table'} onClick={() => setMode('table')}>Table</button>
            </div>
            <span className="extended-count">
              {!focus.following ? 'You are looking at ' : done ? 'Registered: ' : 'The agent is working on '}
              <b>{focus.ids.length === run.rows.length ? `all ${run.rows.length} sections` : `${focus.ids.length} of ${run.rows.length} sections`}</b>
              {pages > 1 && `, page ${Math.min(page, pages - 1) + 1} of ${pages}`}
            </span>
            {pages > 1 && (
              <span className="pager">
                <button className="ghost-button" aria-label="Previous page" onClick={() => setPage(p => (p - 1 + pages) % pages)}>←</button>
                <button className="ghost-button" aria-label="Next page" onClick={() => setPage(p => (p + 1) % pages)}>→</button>
              </span>
            )}
          </div>
          <div className="stage" ref={stageRef}>
            {review ?? (mode === 'pictures'
              ? run.job !== null && <Stage job={run.job} shown={shown} rows={run.rows} box={box} onPick={focus.pick} detailed />
              : <SectionsTable rows={run.rows} focus={focus} />)}
          </div>
          <CuttingAxis job={run.job} rows={run.rows} axis={run.axis} strip={run.strip} focus={ring ? [ring] : focus.ids}
            shown={ring ? [ring] : shown} done={done} onPick={focus.pick} stairRows={3} />
        </section>
        <Journal onPictures={onPictures} onSection={focus.pick} />
      </div>
    </main>
  );
}

// How far the registration is: of the job's sections, how many are placed, aligned and
// warped.
function Meters({ rows }: { rows: Row[] }) {
  if (!rows.length) return <div className="meters" />;
  const meters = [
    ['Placed in the atlas', rows.filter(r => typeof r.position_mm === 'number').length],
    ['Aligned to the atlas', rows.filter(r => r.transform).length],
    ['Warped to fit', rows.filter(r => r.deformation).length],
  ] as const;
  return (
    <dl className="meters" aria-label="How far the registration is">
      {meters.map(([label, n]) => (
        <div key={label} className={'meter' + (n === rows.length ? ' full' : n === 0 ? ' none' : '')}>
          <dt>{label}</dt>
          <dd>
            <span className="meter-count">{n} of {rows.length}</span>
            <span className="meter-bar">
              <motion.span className="meter-fill" initial={false} animate={{ scaleX: n / rows.length }} transition={{ duration: 0.5, ease: EASE_OUT }} />
            </span>
          </dd>
        </div>
      ))}
    </dl>
  );
}

function SectionsTable({ rows, focus }: { rows: Row[]; focus: Focus }) {
  return (
    <div className="table-scroll">
      <table className="sections-table">
        <thead>
          <tr><th className="num">#</th><th>Section</th><th className="num">Position (mm)</th><th>Transform</th><th>Deformation</th><th>Damage</th></tr>
        </thead>
        <tbody>
          {rows.map((r, i) => (
            <tr key={r.id} className={focus.ids.includes(r.id) ? 'target' : ''} onClick={() => focus.pick(r.id)}>
              <td className="num">{r.order ?? i + 1}</td>
              <td className="name">{short(r.id)}</td>
              <td className={typeof r.position_mm === 'number' ? 'num' : 'none'}>{typeof r.position_mm === 'number' ? r.position_mm.toFixed(3) : 'not placed'}</td>
              <td>{r.transform || ''}</td>
              <td>{r.deformation || ''}</td>
              <td className={r.damaged ? 'damaged' : ''}>{r.damaged ? 'damaged' : ''}</td>
            </tr>
          ))}
        </tbody>
      </table>
    </div>
  );
}
