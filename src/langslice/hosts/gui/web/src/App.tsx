// The LangSlice window: the top bar (the lockup, the tabs, the job, Export, Settings) over
// the tab shown. The agent viewer has two views, Simple and Extended; Extended still opens
// the window's older page. The Prompt generator and Settings, once opened, stay mounted
// while hidden, so what was typed in them survives a visit to another tab.
import { MotionConfig, motion } from 'motion/react';
import { useEffect, useState, type ReactNode } from 'react';
import { Export } from './export/Export';
import { DUR, EASE_IN_OUT } from './motion';
import { go, useNav, type Tab } from './nav';
import { Prompt } from './prompt/Prompt';
import { chooseJob, start, useRun } from './run';
import { landOnSettingsIfUnset, Settings } from './settings/Settings';
import { SimpleView } from './simple/SimpleView';

const older = (tab: string, job: number | null) => `./#tab=${tab}${job !== null ? `&job=${job}` : ''}`;

export function App() {
  const nav = useNav();
  const jobs = useRun(s => s.jobs);
  const job = useRun(s => s.job);
  const connected = useRun(s => s.connected);
  const done = useRun(s => s.status === 'Submitted');
  const [visited, setVisited] = useState<Set<Tab>>(() => new Set([nav.tab]));
  useEffect(() => { start(); landOnSettingsIfUnset(); }, []);
  useEffect(() => { document.documentElement.classList.toggle('run-done', done); }, [done]);
  useEffect(() => { setVisited(v => (v.has(nav.tab) ? v : new Set([...v, nav.tab]))); }, [nav.tab]);
  return (
    <MotionConfig reducedMotion="user">
      <div className="window">
        <header className="bar">
          <img className="lockup" src={document.documentElement.dataset.theme === 'light' ? 'lockup-light.svg' : 'lockup.svg'} alt="LangSlice" />
          <nav className="tabs" aria-label="Tabs">
            <TabButton tab="viewer" current={nav.tab}>Agent viewer</TabButton>
            <TabButton tab="prompt" current={nav.tab}>Prompt generator</TabButton>
          </nav>
          <div className="bar-end">
            {jobs && jobs.length > 1 && (
              <select className="job" value={job ?? ''} onChange={e => chooseJob(Number(e.target.value))} aria-label="Job">
                {jobs.map(j => <option key={j.index} value={j.index}>{j.name}</option>)}
              </select>
            )}
            <Export />
            <TabButton tab="settings" current={nav.tab}>Settings</TabButton>
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
        {visited.has('prompt') && <div className="pane" hidden={nav.tab !== 'prompt'}><Prompt /></div>}
        {visited.has('settings') && <div className="pane" hidden={nav.tab !== 'settings'}><Settings shown={nav.tab === 'settings'} /></div>}
        <footer className="foot">{connected ? '' : 'Waiting for LangSlice…'}</footer>
      </div>
    </MotionConfig>
  );
}

// A tab of the top bar; the shown one is underlined in the accent, and the line slides
// from the tab left to the tab chosen.
function TabButton({ tab, current, children }: { tab: Tab; current: Tab; children: ReactNode }) {
  const on = tab === current;
  return (
    <button type="button" className={'tab' + (on ? ' active' : '')} aria-current={on ? 'page' : undefined} onClick={() => go({ tab })}>
      {children}
      {on && <motion.span layoutId="tab-line" className="tab-line" transition={{ duration: DUR.indicator, ease: EASE_IN_OUT }} />}
    </button>
  );
}
