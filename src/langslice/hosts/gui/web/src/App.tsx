// The LangSlice window: the top bar (the lockup, the tabs, the job, Export, Settings) over
// the tab shown. The agent viewer has two views, Simple and Extended. The Prompt generator and Settings, once opened, stay mounted
// while hidden, so what was typed in them survives a visit to another tab.
import { MotionConfig, motion } from 'motion/react';
import { useEffect, useState, type ReactNode } from 'react';
import { Export } from './panels/export/Export';
import { DUR, EASE_IN_OUT } from './ui/motion';
import { go, useNav, type Tab } from './data/nav';
import { Prompt } from './panels/prompt/Prompt';
import { chooseJob, start, useRun } from './data/run';
import { landOnSettingsIfUnset, Settings } from './panels/settings/Settings';
import { Viewer } from './views/Viewer';


export function App() {
  const nav = useNav();
  const jobs = useRun(s => s.jobs);
  const job = useRun(s => s.job);
  const connected = useRun(s => s.connected);
  const done = useRun(s => s.status === 'Submitted');
  const [visited, setVisited] = useState<Set<Tab>>(() => new Set([nav.tab]));
  const [theme, setTheme] = useState(() => (document.documentElement.dataset.theme === 'light' ? 'light' : 'dark'));
  const flipTheme = () => {
    const next = theme === 'dark' ? 'light' : 'dark';
    document.documentElement.dataset.theme = next;
    try { localStorage.setItem('langslice.theme', next); } catch { /* not kept */ }
    setTheme(next);
  };
  const shownJob = jobs?.find(j => j.index === job);
  useEffect(() => { start(); landOnSettingsIfUnset(); }, []);
  useEffect(() => { document.documentElement.classList.toggle('run-done', done); }, [done]);
  useEffect(() => { setVisited(v => (v.has(nav.tab) ? v : new Set([...v, nav.tab]))); }, [nav.tab]);
  return (
    <MotionConfig reducedMotion="user">
      <div className="window">
        <header className="bar">
          <img className="lockup" src={theme === 'light' ? 'lockup-light.svg' : 'lockup.svg'} alt="LangSlice" />
          <nav className="tabs" aria-label="Tabs">
            <TabButton tab="viewer" current={nav.tab}>Agent viewer</TabButton>
            <TabButton tab="prompt" current={nav.tab}>Prompt generator</TabButton>
          </nav>
          <div className="bar-end">
            <button type="button" className="bar-quiet" onClick={flipTheme} title="Switch between the dark and the light look">
              {theme === 'dark' ? 'Light' : 'Dark'}
            </button>
            {jobs && jobs.length > 1 ? (
              <select className="job" value={job ?? ''} onChange={e => chooseJob(Number(e.target.value))} aria-label="Job">
                {jobs.map(j => <option key={j.index} value={j.index}>{j.name}</option>)}
              </select>
            ) : shownJob && <span className="job-name" title={shownJob.path}>{shownJob.name}</span>}
            <Export />
            <TabButton tab="settings" current={nav.tab}>Settings</TabButton>
          </div>
        </header>
        {nav.tab === 'viewer' && <Viewer key={theme} />}
        {visited.has('prompt') && <div className="pane" hidden={nav.tab !== 'prompt'}><Prompt /></div>}
        {visited.has('settings') && <div className="pane" hidden={nav.tab !== 'settings'}><Settings shown={nav.tab === 'settings'} /></div>}
        <footer className="foot">
          <span>{connected ? 'Connected locally' : 'Waiting for LangSlice…'}</span>
          <span>Closing this window does not stop the run.</span>
        </footer>
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
