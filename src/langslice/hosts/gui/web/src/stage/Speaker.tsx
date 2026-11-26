// The agent's margin: a column beside the pictures. At its head the model and its state
// (Working, Thinking, Done); under it the agent's latest words, said one by one in the
// serif face, in italic while it is only thinking; in the Extended view the whole log
// takes the words' place. At its foot, what is happening now and for how long (Doing), so
// a still screen is never a mystery.
import { AnimatePresence, motion } from 'motion/react';
import { useEffect, useLayoutEffect, useRef, useState } from 'react';
import type { ReactNode } from 'react';
import { DUR, EASE_OUT } from '../ui/motion';
import type { Call, Words } from '../data/run';
import { Doing } from './Doing';

const STATE_WORDS: Record<string, string> = { Running: 'Working', Submitted: 'Done', Waiting: 'Waiting', Idle: 'Idle', Failed: 'Stopped' };

export function Speaker({ model, status, words, calls, children }: {
  model: string; status: string; words: Words | null; calls: Call[]; children?: ReactNode;
}) {
  const thinking = words?.kind === 'reasoning';
  const working = calls.some(c => c.running);
  const speaking = useSpeaking(words);
  const done = status === 'Submitted';
  const state = status === 'Running' && !working ? 'Thinking' : STATE_WORDS[status] ?? status;
  return (
    <aside className={['speaker', children ? 'with-log' : '', working && 'working', thinking && 'thinking', speaking && 'speaking', done && 'done'].filter(Boolean).join(' ')}>
      <div className="who">
        <img className="logo-mark" src="logomark.svg" alt="" />
        <p className="who-name">{model || 'The agent'}</p>
        <State word={state} status={status} />
      </div>
      {children ?? (
        <div className="voice">
          <AnimatePresence initial={false}>
            {words ? (
              <Said key={words.n} className="words" title={words.text}>{spoken(words.text)}</Said>
            ) : (
              <motion.p key="waiting" className="words waiting" exit={{ opacity: 0, transition: { duration: DUR.quick } }}>
                {done ? 'The run is finished.' : calls.length ? 'The agent is at work; its words will show here.' : 'Waiting for the agent to begin…'}
              </motion.p>
            )}
          </AnimatePresence>
        </div>
      )}
      <Doing calls={calls} status={status} since={words?.seenAt ?? 0} />
    </aside>
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

// The agent's words, whole: a long message is set a little smaller until it fits the
// margin's room for words (never below MIN_VOICE px), so nothing is cut off.
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
