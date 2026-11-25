// Who is speaking, and what it says. The LangSlice mark, the model's name and its state
// under it (Working, Thinking, Done); beside them a
// speech bubble of a fixed height with the agent's latest words, said one by one; under
// the bubble, the call it is making in plain words.
import { AnimatePresence, motion } from 'motion/react';
import { useEffect, useLayoutEffect, useRef, useState } from 'react';
import { DUR, EASE_OUT } from '../ui/motion';
import type { Call, Words } from '../data/run';

const STATE_WORDS: Record<string, string> = { Running: 'Working', Submitted: 'Done', Waiting: 'Waiting', Idle: 'Idle', Failed: 'Stopped' };

export function Speaker({ model, status, words, call, focusCount }: {
  model: string; status: string; words: Words | null; call: Call | null; focusCount: number;
}) {
  const thinking = words?.kind === 'reasoning';
  const working = Boolean(call?.running);
  const speaking = useSpeaking(words);
  const done = status === 'Submitted';
  const state = status === 'Running' && thinking ? 'Thinking' : STATE_WORDS[status] ?? status;
  return (
    <header className={['speaker', working && 'working', thinking && 'thinking', speaking && 'speaking', done && 'done'].filter(Boolean).join(' ')}>
      <div className="who">
        <img className="logo-mark" src="logomark.svg" alt="" />
        <p className="who-name">{model || 'The agent'}</p>
        <State word={state} status={status} />
      </div>
      <div className="speaker-said">
        {/* The old line and the new share one cell: the old fades over as the new is said. */}
        <div className="bubble">
          <AnimatePresence initial={false}>
            {words ? (
              <Said
                key={words.n}
                className="words"
                title={words.text}
              >
                {spoken(words.text)}
              </Said>
            ) : (
              <motion.p key="waiting" className="words waiting" exit={{ opacity: 0, transition: { duration: DUR.quick } }}>
                {done ? 'The run is finished.' : call ? 'The agent is at work; its words will show here.' : 'Waiting for the agent to begin…'}
              </motion.p>
            )}
          </AnimatePresence>
        </div>
        <div className="doing-cell">
        <AnimatePresence initial={false}>
          <motion.p
            key={(call?.title ?? 'none') + (working ? ':now' : ':last')}
            className={'doing' + (working ? ' running' : '')}
            initial={{ opacity: 0, y: 3 }}
            animate={{ opacity: 1, y: 0 }}
            exit={{ opacity: 0, transition: { duration: DUR.quick, ease: EASE_OUT } }}
            transition={{ duration: DUR.quick, ease: EASE_OUT }}
          >
            {call && (
              <>
                <span className="doing-lead">{working ? 'Now:' : 'Last step:'}</span>{' '}
                <span className="doing-what">{call.title}</span>
                <span className="doing-on">{onWords(call.sections, focusCount)}</span>
              </>
            )}
          </motion.p>
        </AnimatePresence>
        </div>
      </div>
    </header>
  );
}

// The agent's state, as a small pill under its name: a glyph that moves the way the
// state does (dots that take turns while it thinks, a beat while it works, a tick when it
// is done) and the word.
function State({ word, status }: { word: string; status: string }) {
  const kind = word === 'Thinking' ? 'thinking' : status === 'Running' ? 'working' : status === 'Submitted' ? 'done'
    : status === 'Failed' ? 'failed' : 'idle';
  return (
    <p className={'state state-' + kind} aria-live="polite">
      <span className="state-glyph" aria-hidden="true">
        {kind === 'thinking' ? <><i /><i /><i /></>
          : kind === 'done' ? <svg viewBox="0 0 12 12"><path d="M2.5 6.4 5 8.8 9.6 3.6" /></svg>
            : <i />}
      </span>
      <span className="state-word">{word || 'Waiting'}</span>
    </p>
  );
}

// The agent's words in the bubble, whole: a long message is set a little smaller until it
// fits the bubble's three lines (never below MIN_VOICE px), so nothing is cut off.
const MIN_VOICE = 15;
function Said({ className, title, children }: { className: string; title: string; children: React.ReactNode }) {
  const ref = useRef<HTMLParagraphElement>(null);
  useLayoutEffect(() => {
    const el = ref.current;
    if (!el) return;
    const box = el.parentElement!;
    const fit = () => {
      el.style.fontSize = '';
      let size = parseFloat(getComputedStyle(el).fontSize);
      const room = box.clientHeight - parseFloat(getComputedStyle(box).paddingTop) - parseFloat(getComputedStyle(box).paddingBottom);
      while (el.scrollHeight > room + 1 && size > MIN_VOICE) {
        size -= 1;
        el.style.fontSize = `${size}px`;
      }
    };
    fit();
    const o = new ResizeObserver(fit);
    o.observe(box);
    return () => o.disconnect();
  }, []);
  return (
    <motion.p ref={ref} className={className} title={title}
      exit={{ opacity: 0, y: -4, transition: { duration: DUR.quick, ease: EASE_OUT } }}>
      {children}
    </motion.p>
  );
}

// The words, one span each; a word's delay follows its length, as speech does, and the
// pace quickens a little as the sentence goes on so a long one is never waited for.
function spoken(text: string) {
  let at = 0;
  return text.split(/(\s+)/).map((part, i) => {
    if (!part.trim()) return part;
    const delay = Math.round(at);
    at += Math.min(70, 18 + part.length * 4) * Math.max(0.45, 1 - at / 3200);
    return <span key={i} className="word" style={{ animationDelay: `${delay}ms` }}>{part}</span>;
  });
}

function useSpeaking(words: Words | null) {
  const [on, setOn] = useState(false);
  useEffect(() => {
    if (!words) return;
    setOn(true);
    const t = setTimeout(() => setOn(false), Math.min(4200, 600 + words.text.length * 14));
    return () => clearTimeout(t);
  }, [words?.n]);
  return on;
}

const onWords = (ids: string[], shown: number) => {
  const n = ids.length || shown;
  return !n ? '' : n === 1 ? ', on one section' : `, on ${n} sections`;
};

