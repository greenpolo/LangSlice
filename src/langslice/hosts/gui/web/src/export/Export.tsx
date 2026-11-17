// The Export menu in the top bar: the shown job's registration as an ABBA project, a
// QuickNII file or a VisuAlign file, written by the server under the job's exports/
// folder (hosts/gui/export.py). It starts an export, polls its state while it runs, then
// says where the file is, with Open folder and Copy path.
//
// The menu is a WAI-ARIA menu button: Enter, Space or the down arrow open it on its first
// item, the up arrow on its last; the arrows, Home and End move between the items; Esc
// closes it and gives the focus back to the button; Tab or a click outside closes it.
import { useCallback, useEffect, useId, useRef, useState } from 'react';
import { useRun } from '../run';
import './export.css';

const FORMATS = [
  { kind: 'abba', name: 'ABBA project (.abba)',
    about: 'Open in ABBA with File > Load State: each section with its affine and its warp.' },
  { kind: 'quicknii', name: 'QuickNII (.json)',
    about: 'The linear placement of each section, with its image.' },
  { kind: 'visualign', name: 'VisuAlign (.json)',
    about: 'The placement and each warp as VisuAlign markers.' },
];
const NAMES: Record<string, string> = Object.fromEntries(FORMATS.map(f => [f.kind, f.name]));
const POLL_MS = 700;

type State = {
  state: 'idle' | 'running' | 'done' | 'failed'; format?: string; path?: string;
  message?: string; skipped?: number; seconds?: number;
};

async function post(resource: string, body?: object) {
  const r = await fetch(resource, {
    method: 'POST', headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify(body || {}), cache: 'no-store',
  });
  const answer = await r.json().catch(() => ({}));
  if (!r.ok) throw new Error(answer.error || `The server answered ${r.status}.`);
  return answer;
}

export function Export() {
  const job = useRun(s => s.job);
  const home = useRun(s => s.home);
  const [open, setOpen] = useState(false);
  const [note, setNote] = useState<State | null>(null);
  const [copied, setCopied] = useState(false);
  const asked = useRef(false);
  const timer = useRef(0);
  const root = useRef<HTMLDivElement>(null);
  const button = useRef<HTMLButtonElement>(null);
  const items = useRef<(HTMLButtonElement | null)[]>([]);
  const firstFocus = useRef(0);
  const menuId = useId();
  const currentJob = useRef<number | null>(job);

  const shown = (path: string) => (home && path.toLowerCase().startsWith(home.toLowerCase()) ? '~' + path.slice(home.length) : path);

  const poll = useCallback(async (index: number) => {
    clearTimeout(timer.current);
    try {
      const r = await fetch(`jobs/${index}/export`, { cache: 'no-store' });
      if (!r.ok || index !== currentJob.current) return;
      const state: State = await r.json();
      // An export finished before this page opened is not news.
      if (state.state !== 'running' && !asked.current) return;
      setNote(state);
      if (state.state === 'running') timer.current = window.setTimeout(() => poll(index), POLL_MS);
    } catch {
      timer.current = window.setTimeout(() => poll(index), POLL_MS * 3);
    }
  }, []);

  // A new job: the note goes, and an export running for it is shown.
  useEffect(() => {
    currentJob.current = job;
    asked.current = false;
    setNote(null);
    setOpen(false);
    if (job !== null) poll(job);
    return () => clearTimeout(timer.current);
  }, [job, poll]);

  async function start(kind: string) {
    close(true);
    if (job === null) return;
    const index = job;
    asked.current = true;
    setCopied(false);
    try {
      setNote(await post(`jobs/${index}/export`, { format: kind }));
      poll(index);
    } catch (error) {
      setNote({ state: 'failed', format: '', message: (error as Error).message });
    }
  }

  function openMenu(first: number) {
    firstFocus.current = first;
    setNote(null);
    setOpen(true);
  }
  function close(refocus: boolean) {
    setOpen(false);
    if (refocus) button.current?.focus();
  }

  // Focus the item asked for once the menu is drawn.
  useEffect(() => {
    if (!open) return;
    const at = firstFocus.current < 0 ? FORMATS.length - 1 : firstFocus.current;
    items.current[at]?.focus();
  }, [open]);

  // A click outside closes the menu (and leaves a note open: it is closed by its own button).
  useEffect(() => {
    if (!open) return;
    const away = (e: PointerEvent) => { if (!root.current?.contains(e.target as Node)) setOpen(false); };
    document.addEventListener('pointerdown', away);
    return () => document.removeEventListener('pointerdown', away);
  }, [open]);

  // Esc closes the note too.
  useEffect(() => {
    if (!note || note.state === 'idle') return;
    const esc = (e: KeyboardEvent) => { if (e.key === 'Escape' && !open) setNote(null); };
    document.addEventListener('keydown', esc);
    return () => document.removeEventListener('keydown', esc);
  }, [note, open]);

  function onButtonKey(e: React.KeyboardEvent) {
    if (e.key === 'ArrowDown') { e.preventDefault(); openMenu(0); }
    else if (e.key === 'ArrowUp') { e.preventDefault(); openMenu(-1); }
  }
  function onMenuKey(e: React.KeyboardEvent) {
    const at = items.current.indexOf(document.activeElement as HTMLButtonElement);
    const go = (i: number) => { e.preventDefault(); items.current[(i + FORMATS.length) % FORMATS.length]?.focus(); };
    if (e.key === 'ArrowDown') go(at + 1);
    else if (e.key === 'ArrowUp') go(at - 1);
    else if (e.key === 'Home') go(0);
    else if (e.key === 'End') go(FORMATS.length - 1);
    else if (e.key === 'Escape') { e.preventDefault(); close(true); }
    else if (e.key === 'Tab') setOpen(false);
  }

  const running = note?.state === 'running';
  const name = (note?.format && NAMES[note.format]) || 'export';
  return (
    <div className={'export' + (job === null ? ' off' : '')} ref={root}>
      <button
        ref={button}
        type="button"
        className={'bar-button' + (open ? ' open' : '')}
        aria-haspopup="menu"
        aria-expanded={open}
        aria-controls={open ? menuId : undefined}
        disabled={running || job === null}
        title="Write this job's registration as a file for another program"
        onClick={() => (open ? close(false) : openMenu(0))}
        onKeyDown={onButtonKey}
      >
        {running && <span className="busy-dot" aria-hidden="true" />}
        Export
        <span className="chevron" aria-hidden="true" />
      </button>
      {/* Always drawn, so it can leave as it came: CSS moves it (see export.css). */}
      <div id={menuId} className="pop menu" role="menu" aria-label="Export as" data-open={open || undefined}
        inert={!open} onKeyDown={onMenuKey}>
            {FORMATS.map((f, i) => (
              <button
                key={f.kind}
                ref={el => { items.current[i] = el; }}
                type="button"
                role="menuitem"
                tabIndex={-1}
                className="menu-item"
                onClick={() => start(f.kind)}
              >
                <span className="menu-item-name">{f.name}</span>
                <span className="menu-item-about">{f.about}</span>
              </button>
        ))}
      </div>
      {!open && note && note.state !== 'idle' && (
        <div className={'pop note ' + note.state} role="status">
            {note.state === 'running' && (
              <>
                <p className="note-title">Writing the {name}…</p>
                {note.format === 'abba' && <p className="note-message">Each warp becomes landmark pairs; this can take a few minutes.</p>}
                <span className="note-progress" aria-hidden="true"><span /></span>
              </>
            )}
            {note.state === 'done' && (
              <>
                <p className="note-title">{name} written</p>
                <p className="note-path">{shown(note.path || '')}</p>
                <p className="note-message">
                  {note.message}{note.skipped ? `; ${note.skipped} left out (abba_export.json says why)` : ''}, in {note.seconds} s.
                </p>
                <div className="note-actions">
                  <button type="button" className="button" onClick={() => {
                    if (job !== null) post(`jobs/${job}/export/reveal`).catch(e => setNote({ state: 'failed', format: '', message: e.message }));
                  }}>Open folder</button>
                  <button type="button" className="button" onClick={() => {
                    navigator.clipboard?.writeText(note.path || '').then(() => setCopied(true));
                  }}>
                    <span className="swap" data-on={copied || undefined}>
                      <span>Copy path</span><span aria-hidden={!copied}>Copied</span>
                    </span>
                  </button>
                  <button type="button" className="button quiet" onClick={() => setNote(null)}>Close</button>
                </div>
              </>
            )}
            {note.state === 'failed' && (
              <>
                <p className="note-title">The {name} was not written</p>
                <p className="note-message">{note.message}</p>
                <div className="note-actions">
                  <button type="button" className="button quiet" onClick={() => setNote(null)}>Close</button>
                </div>
              </>
            )}
        </div>
      )}
    </div>
  );
}
