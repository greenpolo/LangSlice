// Why the screen is still: a card under the agent's words for each tool call that is
// running, with a clock that counts up and a bar. When this job has run the same tool
// before and it took at least MIN_USUAL_S, the bar fills towards the time it took then
// (its median), and turns to a slow sweep once the call runs longer than that; any other
// call sweeps from the start.
// Between calls, while the run is going, the card says the agent is choosing its next step
// and for how long; once the run stops, it names the last step and how long it took.
import { AnimatePresence, motion } from 'motion/react';
import { useEffect, useState } from 'react';
import { DUR, EASE_OUT } from '../ui/motion';
import type { Call } from '../data/run';
import { callDetail } from '../data/acts';

const SHOWN_AT_ONCE = 2;
const NEAR_FULL = 0.94;
const MIN_USUAL_S = 5;

export function Doing({ calls, status, since }: {
  calls: Call[]; status: string; since: number;
}) {
  const now = useNow();
  const running = calls.filter(c => c.running);
  const going = status === 'Running';
  const last = calls.at(-1) ?? null;
  return (
    <div className="doing-cards" aria-live="polite">
      <AnimatePresence initial={false} mode="popLayout">
        {running.slice(-SHOWN_AT_ONCE).map(c => (
          <Card key={c.key} lead="Running" title={c.title} name={c.name} detail={detailOf(c)}
            elapsed={secondsSince(c.seenAt, now)} usual={usualSeconds(calls, c.name)} live />
        ))}
        {running.length > SHOWN_AT_ONCE && (
          <motion.p key="more" className="doing-more" {...FADE}>and {running.length - SHOWN_AT_ONCE} more running</motion.p>
        )}
        {!running.length && going && (
          <Card key={'between:' + (last?.key ?? '')} lead="Between steps"
            title="Choosing its next step" detail={last ? `Last step: ${last.title}` : ''}
            elapsed={secondsSince(Math.max(last?.endedAt || Date.parse(last?.doneAt ?? '') || 0, since || 0) || undefined, now)} live />
        )}
        {!running.length && !going && last && (
          <Card key={'last:' + last.key} lead={last.failed ? 'Last step failed' : 'Last step'} title={last.title}
            name={last.name} detail={detailOf(last)} took={tookSeconds(last)} />
        )}
      </AnimatePresence>
    </div>
  );
}

const FADE = {
  initial: { opacity: 0, y: 6 },
  animate: { opacity: 1, y: 0, transition: { duration: DUR.enter, ease: EASE_OUT } },
  exit: { opacity: 0, transition: { duration: DUR.quick, ease: EASE_OUT } },
};

function Card({ lead, title, name, detail, elapsed, usual, took, live }: {
  lead: string; title: string; name?: string; detail: string;
  elapsed?: number | null; usual?: number | null; took?: number | null; live?: boolean;
}) {
  const over = usual != null && elapsed != null && elapsed > usual;
  const share = usual && elapsed != null && !over ? Math.min(NEAR_FULL, elapsed / usual) : null;
  return (
    <motion.section layout className={'doing-card' + (live ? ' live' : '')} {...FADE}>
      <header className="doing-head">
        <span className="doing-lead">{lead}</span>
        {elapsed != null && <span className="doing-clock">{clock(elapsed)}</span>}
        {took != null && <span className="doing-clock">took {clock(took)}</span>}
      </header>
      <p className="doing-title">{title}{name && name !== title && <code className="doing-name">{name}</code>}</p>
      {detail && <p className="doing-detail">{detail}</p>}
      {live && (
        <div className={'doing-bar' + (share == null ? ' sweep' : '')} role="progressbar"
          aria-valuenow={share == null ? undefined : Math.round(share * 100)}>
          {share != null && <span className="doing-fill" style={{ transform: `scaleX(${share})` }} />}
        </div>
      )}
      {live && usual != null && (
        <p className="doing-usual">{over ? `longer than its usual ${clock(usual)}` : `usually about ${clock(usual)} in this run`}</p>
      )}
    </motion.section>
  );
}

function detailOf(call: Call) {
  const detail = callDetail(call);
  const n = call.sections.length;
  const on = !n ? '' : n === 1 ? 'on one section' : `on ${n} sections`;
  return [detail, on].filter(Boolean).join(', ');
}

function usualSeconds(calls: Call[], name: string): number | null {
  const times = calls.filter(c => c.name === name && c.ended && !c.failed).map(tookSeconds)
    .filter((t): t is number => t != null).sort((a, b) => a - b);
  const median = times.length ? times[Math.floor(times.length / 2)] : null;
  return median != null && median >= MIN_USUAL_S ? median : null;
}

function tookSeconds(call: Call): number | null {
  const a = Date.parse(call.at ?? ''), b = Date.parse(call.doneAt ?? '');
  return Number.isFinite(a) && Number.isFinite(b) && b >= a ? (b - a) / 1000 : null;
}

function secondsSince(at: number | undefined, now: number): number | null {
  return at && Number.isFinite(at) ? Math.max(0, (now - at) / 1000) : null;
}

function clock(seconds: number) {
  const s = Math.round(seconds);
  if (s < 60) return `${s} s`;
  const m = Math.floor(s / 60), h = Math.floor(m / 60);
  return h ? `${h} h ${String(m % 60).padStart(2, '0')} min` : `${m}:${String(s % 60).padStart(2, '0')}`;
}

function useNow() {
  const [now, setNow] = useState(Date.now());
  useEffect(() => {
    const t = setInterval(() => setNow(Date.now()), 1000);
    return () => clearInterval(t);
  }, []);
  return now;
}
