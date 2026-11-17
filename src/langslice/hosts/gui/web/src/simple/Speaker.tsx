// Who is speaking, and what it says. The LangSlice mark and the agent's figure side by
// side, its model's name and its state under them (Working, Thinking, Done); beside them a
// speech bubble of a fixed height with the agent's latest words, said one by one; under
// the bubble, the call it is making in plain words. The figure has one gesture per state:
// it walks while a tool runs, looks up while it thinks, talks while its words come in, and
// hops once when the run is done.
import { AnimatePresence, motion } from 'motion/react';
import { useEffect, useState } from 'react';
import { DUR, EASE_OUT } from '../motion';
import type { Call, Words } from '../run';

const STATE_WORDS: Record<string, string> = { Running: 'Working', Submitted: 'Done', Waiting: 'Waiting', Idle: 'Idle', Failed: 'Stopped' };

export function Speaker({ model, status, words, call, focusCount }: {
  model: string; status: string; words: Words | null; call: Call | null; focusCount: number;
}) {
  const thinking = words?.kind === 'reasoning';
  const working = Boolean(call?.running);
  const speaking = useSpeaking(words);
  const done = status === 'Submitted';
  const state = status === 'Running' && thinking ? 'Thinking' : STATE_WORDS[status] ?? status;
  const maker = makerOf(model);
  return (
    <header className={['speaker', working && 'working', thinking && 'thinking', speaking && 'speaking', done && 'done'].filter(Boolean).join(' ')}>
      <div className="who">
        <img className="mark" src="logomark.svg" alt="" />
        <Figure maker={maker} />
        <p className="who-name">{model || 'The agent'}</p>
        <p className="who-state"><span className={'dot ' + status} />{state}</p>
      </div>
      <div className="said">
        <div className="bubble">
          <AnimatePresence mode="popLayout" initial={false}>
            {words ? (
              <motion.p
                key={words.n}
                className="words"
                title={words.text}
                exit={{ opacity: 0, y: -4, transition: { duration: 0.16, ease: EASE_OUT } }}
              >
                {spoken(words.text)}
              </motion.p>
            ) : (
              <motion.p key="waiting" className="words waiting" exit={{ opacity: 0, transition: { duration: 0.16 } }}>
                {done ? 'The run is finished.' : call ? 'The agent is at work; its words will show here.' : 'Waiting for the agent to begin…'}
              </motion.p>
            )}
          </AnimatePresence>
        </div>
        <AnimatePresence mode="wait" initial={false}>
          <motion.p
            key={call?.key ?? 'none'}
            className={'doing' + (working ? ' running' : '')}
            initial={{ opacity: 0, y: 3 }}
            animate={{ opacity: 1, y: 0 }}
            exit={{ opacity: 0, transition: { duration: 0.12 } }}
            transition={{ duration: DUR.enter, ease: EASE_OUT }}
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
    </header>
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
  }, [words?.n]); // eslint-disable-line react-hooks/exhaustive-deps
  return on;
}

const onWords = (ids: string[], shown: number) => {
  const n = ids.length || shown;
  return !n ? '' : n === 1 ? ', on one section' : `, on ${n} sections`;
};

function makerOf(model: string) {
  const name = model.toLowerCase();
  if (/claude|fable|mythos|opus|sonnet|haiku|anthropic/.test(name)) return 'claude';
  if (/gemini|google/.test(name)) return 'gemini';
  if (/gpt|openai|codex|chatgpt|\bo[0-9]/.test(name)) return 'openai';
  return 'plain';
}

// The agent's figure: Clawd for a Claude model (drawn from the pixel map Claude Code
// carries), else a plain face. Other makers' marks come with the Extended view's port.
function Figure({ maker }: { maker: string }) {
  if (maker === 'claude') {
    return (
      <svg className="figure clawd" viewBox="0 0 19 10" role="img" aria-label="Clawd, for a Claude model" shapeRendering="crispEdges">
        <g className="clawd-body">
          <rect x="3" y="1" width="13" height="6" />
          <rect x="1" y="5" width="2" height="1" />
          <rect x="16" y="5" width="2" height="1" />
          <g className="legs-a"><rect x="3" y="7" width="1" height="2" /><rect x="13" y="7" width="1" height="2" /></g>
          <g className="legs-b"><rect x="5" y="7" width="1" height="2" /><rect x="15" y="7" width="1" height="2" /></g>
        </g>
        <rect className="eye" x="6" y="3" width="1" height="2" />
        <rect className="eye" x="12" y="3" width="1" height="2" />
      </svg>
    );
  }
  return (
    <svg className="figure plain" viewBox="0 0 48 48" role="img" aria-label="The agent">
      <path className="face" d="M24 7c10.5 0 17 6.2 17 16.5S34.5 41 24 41 7 33.8 7 23.5 13.5 7 24 7z" />
      <circle className="eye" cx="17.5" cy="22" r="2.3" />
      <circle className="eye" cx="30.5" cy="22" r="2.3" />
    </svg>
  );
}
