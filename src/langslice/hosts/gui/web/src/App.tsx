// The LangSlice window. This build holds the agent viewer's Simple view; the views not
// rebuilt yet (Extended, the Prompt generator, Settings) open in the window's older page.
import { useEffect } from 'react';
import { chooseJob, start, useRun } from './run';
import { FrontToBack } from './simple/FrontToBack';
import { Speaker } from './simple/Speaker';
import { MotionConfig } from 'motion/react';
import { geometry, Stage, useBox, useDirector } from './simple/Stage';
import { overlayUrl, warm } from './pictures';

export function App() {
  const run = useRun();
  const [stageRef, box] = useBox<HTMLDivElement>();
  const done = run.status === 'Submitted';
  const ids = run.rows.length
    ? done || !run.focus?.length
      ? run.rows.map(r => r.id)
      : [...new Set(run.focus.filter(id => run.rows.some(r => r.id === id)))]
    : [];
  const shown = useDirector(run.job, ids, run.rows, box);
  useEffect(() => { start(); }, []);
  // Every section's picture at the grid's size, fetched while nothing else is, so the
  // finished run's grid is ready when it comes.
  const job = run.job;
  const rowKeys = run.rows.map(r => r.id + ':' + (r.render_key ?? '')).join('|');
  useEffect(() => {
    if (job === null || !box || !run.rows.length) return;
    const edge = geometry(run.rows.length, box).edge;
    warm(run.rows.map(r => overlayUrl(job, r.id, edge, r.render_key)));
  }, [job, rowKeys, box]); // eslint-disable-line react-hooks/exhaustive-deps
  const running = [...run.calls].reverse().find(c => c.running);
  const call = running ?? run.calls.at(-1) ?? null;
  useEffect(() => { document.documentElement.classList.toggle('run-done', done); }, [done]);
  const older = (tab: string) => `./#tab=${tab}${run.job !== null ? `&job=${run.job}` : ''}`;
  return (
    <MotionConfig reducedMotion="user">
    <div className="window">
      <header className="bar">
        <img className="lockup" src="lockup.svg" alt="LangSlice" />
        <nav className="tabs">
          <span className="tab active">Agent viewer</span>
          <a className="tab" href={older('prompt')}>Prompt generator</a>
        </nav>
        <div className="bar-end">
          {run.jobs && run.jobs.length > 1 && (
            <select className="job" value={run.job ?? ''} onChange={e => chooseJob(Number(e.target.value))} aria-label="Job">
              {run.jobs.map(j => <option key={j.index} value={j.index}>{j.name}</option>)}
            </select>
          )}
          <a className="tab" href={older('settings')}>Settings</a>
        </div>
      </header>
      <main className="simple">
        <div className="views-switch" role="group" aria-label="How much the viewer shows">
          <span className="on">Simple</span>
          <a href={older('viewer')} onClick={() => { try { localStorage.setItem('langslice.detail', 'extended'); } catch { /* kept default */ } }}>Extended</a>
        </div>
        <Speaker model={run.model} status={run.status} words={run.words} call={call} focusCount={ids.length} />
        <div className="stage" ref={stageRef}>
          {run.job !== null && <Stage job={run.job} shown={shown} rows={run.rows} box={box} />}
          {run.jobs && !run.jobs.length && <p className="empty">No jobs yet: open a job folder with <code>langslice gui FOLDER</code>.</p>}
        </div>
        <FrontToBack rows={run.rows} axis={run.axis} focus={shown} shown={shown.length <= 3 ? shown : []} done={done} />
      </main>
      <footer className="foot">{run.connected ? '' : 'Waiting for LangSlice…'}</footer>
    </div>
    </MotionConfig>
  );
}
