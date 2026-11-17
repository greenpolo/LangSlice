// The LangSlice window: the top bar (the lockup, the tabs, the job, Settings) over the tab
// shown. The agent viewer has two views, Simple and Extended. Parts not rebuilt yet open
// the window's older page.
import { MotionConfig } from 'motion/react';
import { useEffect } from 'react';
import { go, useNav } from './nav';
import { chooseJob, start, useRun } from './run';
import { SimpleView } from './simple/SimpleView';

const older = (tab: string, job: number | null) => `./#tab=${tab}${job !== null ? `&job=${job}` : ''}`;

export function App() {
  const nav = useNav();
  const jobs = useRun(s => s.jobs);
  const job = useRun(s => s.job);
  const connected = useRun(s => s.connected);
  const done = useRun(s => s.status === 'Submitted');
  useEffect(() => { start(); }, []);
  useEffect(() => { document.documentElement.classList.toggle('run-done', done); }, [done]);
  return (
    <MotionConfig reducedMotion="user">
      <div className="window">
        <header className="bar">
          <img className="lockup" src="lockup.svg" alt="LangSlice" />
          <nav className="tabs">
            <button className={'tab' + (nav.tab === 'viewer' ? ' active' : '')} onClick={() => go({ tab: 'viewer' })}>Agent viewer</button>
            <a className="tab" href={older('prompt', job)}>Prompt generator</a>
          </nav>
          <div className="bar-end">
            {jobs && jobs.length > 1 && (
              <select className="job" value={job ?? ''} onChange={e => chooseJob(Number(e.target.value))} aria-label="Job">
                {jobs.map(j => <option key={j.index} value={j.index}>{j.name}</option>)}
              </select>
            )}
            <a className="tab" href={older('settings', job)}>Settings</a>
          </div>
        </header>
        {nav.tab === 'viewer' && (
          <div className="viewer">
            <div className="views-switch" role="group" aria-label="How much the viewer shows">
              <button className={nav.view === 'simple' ? 'on' : ''} aria-pressed={nav.view === 'simple'} onClick={() => go({ view: 'simple' })}>Simple</button>
              <a href={older('viewer', job)} onClick={() => go({ view: 'extended' })}>Extended</a>
            </div>
            <SimpleView />
          </div>
        )}
        <footer className="foot">{connected ? '' : 'Waiting for LangSlice…'}</footer>
      </div>
    </MotionConfig>
  );
}
