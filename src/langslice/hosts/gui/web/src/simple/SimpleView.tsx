// The Simple view: who is speaking and what it says, the sections it works on (or Review),
// and where they sit front to back.
import type { ReactNode } from 'react';
import { useEffect } from 'react';
import { overlayUrl, warm } from '../pictures';
import { useRun } from '../run';
import type { Focus } from '../Viewer';
import { FrontToBack } from './FrontToBack';
import { Speaker } from './Speaker';
import { geometry, Stage, useBox, useDirector } from './Stage';

export function SimpleView({ focus, review }: { focus: Focus; review: ReactNode }) {
  const run = useRun();
  const [stageRef, box] = useBox<HTMLDivElement>();
  const done = run.status === 'Submitted';
  const shown = useDirector(run.job, focus.ids, run.rows, box);
  const running = [...run.calls].reverse().find(c => c.running);
  const call = running ?? run.calls.at(-1) ?? null;
  // Every section's picture at the grid's size, fetched while nothing else is, so the
  // finished run's grid is ready when it comes.
  const job = run.job;
  const rowKeys = run.rows.map(r => r.id + ':' + (r.render_key ?? '')).join('|');
  useEffect(() => {
    if (job === null || !box || !run.rows.length) return;
    const edge = geometry(run.rows.length, box).edge;
    warm(run.rows.map(r => overlayUrl(job, r.id, edge, r.render_key)));
  }, [job, rowKeys, box]); // eslint-disable-line react-hooks/exhaustive-deps
  return (
    <main className="simple">
      <Speaker model={run.model} status={run.status} words={run.words} call={call} focusCount={focus.ids.length} />
      <div className="stage" ref={stageRef}>
        {review ?? (run.job !== null && <Stage job={run.job} shown={shown} rows={run.rows} box={box} onPick={focus.pick} />)}
        {run.jobs && !run.jobs.length && <p className="empty">No jobs yet: open a job folder with <code>langslice gui FOLDER</code>.</p>}
      </div>
      <FrontToBack rows={run.rows} axis={run.axis} focus={shown} shown={shown.length <= 3 ? shown : []} done={done} onPick={focus.pick} />
    </main>
  );
}
