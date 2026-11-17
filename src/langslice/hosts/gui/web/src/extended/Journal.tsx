// The journal: the run so far, read top to bottom. The agent's own words are its entries,
// each at its minute of the run on a thin spine (a summary of its reasoning a step
// quieter); its tool calls are quiet lines between them, in plain words with the tool's
// own name beside, opening to their inputs. Calls of one tool in a row fold into one line
// (`title ×n`). A call's pictures open in the lightbox. The journal follows its end until
// the person scrolls away.
import { AnimatePresence, motion } from 'motion/react';
import { useLayoutEffect, useMemo, useRef, useState } from 'react';
import { DUR, EASE_OUT } from '../motion';
import { useRun, type Call, type Entry } from '../run';
import { short } from '../simple/Stage';

type Line =
  | { kind: 'words'; entry: Extract<Entry, { kind: 'words' }> }
  | { kind: 'calls'; calls: Call[] }
  | { kind: 'note'; text: string; error?: boolean };

export function Journal({ onPictures, onSection }: { onPictures: (numbers: number[], start: number) => void; onSection: (id: string) => void }) {
  const log = useRun(s => s.log);
  const calls = useRun(s => s.calls);
  const byKey = useMemo(() => new Map(calls.map(c => [c.key, c])), [calls]);
  const lines = useMemo(() => {
    const out: Line[] = [];
    for (const entry of log) {
      if (entry.kind === 'words') out.push({ kind: 'words', entry });
      else if (entry.kind === 'note') out.push({ kind: 'note', text: entry.text, error: entry.error });
      else if (entry.kind === 'call') {
        const call = byKey.get(entry.key);
        if (!call) continue;
        const last = out.at(-1);
        if (last?.kind === 'calls' && last.calls[0].name === call.name) last.calls.push(call);
        else out.push({ kind: 'calls', calls: [call] });
      }
    }
    return out;
  }, [log, byKey]);
  const first = log.find(e => e.kind === 'words') as Extract<Entry, { kind: 'words' }> | undefined;
  const started = first ? Date.parse(first.words.at) : NaN;
  const scroller = useRef<HTMLDivElement>(null);
  const [following, setFollowing] = useState(true);
  useLayoutEffect(() => {
    const el = scroller.current;
    if (el && following) el.scrollTop = el.scrollHeight;
  }, [lines, following]);
  const latestWords = [...lines].reverse().find(l => l.kind === 'words');
  return (
    <aside className="journal" aria-label="The run so far">
      <h2 className="journal-title">The run so far</h2>
      <div
        className="journal-scroll"
        ref={scroller}
        onScroll={e => {
          const el = e.currentTarget;
          setFollowing(el.scrollHeight - el.scrollTop - el.clientHeight < 60);
        }}
      >
        {!lines.length && <p className="journal-empty">No activity yet.</p>}
        <div className="journal-items">
          {lines.map((line, i) => {
            if (line.kind === 'words') {
              return (
                <motion.article
                  key={'w' + line.entry.n}
                  className={'entry ' + line.entry.words.kind + (line === latestWords ? ' latest' : '')}
                  initial={{ opacity: 0, y: 6 }}
                  animate={{ opacity: 1, y: 0 }}
                  transition={{ duration: DUR.enter, ease: EASE_OUT }}
                >
                  <span className="entry-when">{minuteOf(line.entry.words.at, started)}</span>
                  <div className="entry-body">{paragraphs(line.entry.full)}</div>
                </motion.article>
              );
            }
            if (line.kind === 'note') return <p key={'n' + i} className={'journal-note' + (line.error ? ' error' : '')}>{line.text}</p>;
            return <CallLine key={'c' + line.calls[0].key} calls={line.calls} onPictures={onPictures} onSection={onSection} />;
          })}
        </div>
      </div>
      <AnimatePresence>
        {!following && (
          <motion.button
            className="journal-jump"
            initial={{ opacity: 0, y: 6 }}
            animate={{ opacity: 1, y: 0 }}
            exit={{ opacity: 0, y: 6, transition: { duration: DUR.quick } }}
            transition={{ duration: DUR.quick, ease: EASE_OUT }}
            onClick={() => setFollowing(true)}
          >
            ↓ Latest activity
          </motion.button>
        )}
      </AnimatePresence>
    </aside>
  );
}

function CallLine({ calls, onPictures, onSection }: { calls: Call[]; onPictures: (numbers: number[], start: number) => void; onSection: (id: string) => void }) {
  const [open, setOpen] = useState(false);
  const first = calls[0], many = calls.length > 1;
  const running = calls.some(c => c.running), failed = calls.some(c => c.failed);
  const pictures = calls.flatMap(c => c.pictures);
  return (
    <div className={'call' + (running ? ' running' : '') + (failed ? ' failed' : '') + (open ? ' open' : '')}>
      <button className="call-line" onClick={() => setOpen(o => !o)} aria-expanded={open}>
        <span className="call-title">{first.title}{many && <span className="call-times"> ×{calls.length}</span>}</span>
        {first.title !== first.name && <span className="call-name">{first.name}</span>}
        {!many && <Targets ids={first.sections} onSection={onSection} />}
        <span className="call-state">{running ? 'now' : failed ? 'failed' : ''}</span>
      </button>
      {pictures.length > 0 && (
        <button className="call-pictures" onClick={() => onPictures(pictures, pictures[0])}>
          {pictures.length === 1 ? '1 picture' : `${pictures.length} pictures`}
        </button>
      )}
      <AnimatePresence initial={false}>
        {open && (
          <motion.div
            className="call-detail"
            initial={{ height: 0, opacity: 0 }}
            animate={{ height: 'auto', opacity: 1 }}
            exit={{ height: 0, opacity: 0 }}
            transition={{ duration: DUR.quick, ease: EASE_OUT }}
          >
            {many
              ? calls.map(c => (
                <div key={c.key} className="call-sub">
                  <Targets ids={c.sections} onSection={onSection} />
                  <Inputs call={c} />
                </div>
              ))
              : <Inputs call={first} />}
          </motion.div>
        )}
      </AnimatePresence>
    </div>
  );
}

function Targets({ ids, onSection }: { ids: string[]; onSection: (id: string) => void }) {
  if (!ids.length) return null;
  if (ids.length === 1) {
    return (
      <span className="call-targets" role="link" tabIndex={0} onClick={e => { e.stopPropagation(); onSection(ids[0]); }}>
        {short(ids[0])}
      </span>
    );
  }
  return <span className="call-targets" title={ids.map(short).join(', ')}>{ids.length} sections</span>;
}

function Inputs({ call }: { call: Call }) {
  const text = call.args === undefined || call.args === null ? '' : typeof call.args === 'string' ? call.args : JSON.stringify(call.args, null, 2);
  const meta = [call.step != null ? `step ${call.step}` : '', call.status || '', call.door ? `via ${call.door}` : ''].filter(Boolean).join(' · ');
  return (
    <>
      {text && <pre className="call-args">{text}</pre>}
      {meta && <p className="call-meta">{meta}</p>}
    </>
  );
}

function minuteOf(at: string, started: number) {
  const t = Date.parse(at);
  if (!Number.isFinite(t) || !Number.isFinite(started)) return '';
  const m = Math.round((t - started) / 60000);
  return m < 1 ? 'Start' : `${m} min`;
}

// The agent's text as paragraphs and list lines, `code` kept; nothing read as markup.
function paragraphs(text: string) {
  return text.split(/\n\s*\n/).filter(b => b.trim()).map((block, i) => (
    <p key={i}>
      {block.split('\n').map((line, j) => (
        <span key={j}>{j > 0 && <br />}{inline(line.replace(/^#{1,6}\s+/, ''))}</span>
      ))}
    </p>
  ));
}
function inline(text: string) {
  return text.split(/(`[^`]+`|\*\*[^*]+\*\*)/).filter(Boolean).map((part, i) =>
    part.startsWith('`') && part.endsWith('`') && part.length > 2 ? <code key={i}>{part.slice(1, -1)}</code>
      : part.startsWith('**') && part.endsWith('**') && part.length > 4 ? <strong key={i}>{part.slice(2, -2)}</strong>
        : part);
}
