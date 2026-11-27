// What the agent is waiting on: a quiet line under its words for each tool call that has
// run longer than SHOW_AFTER_S, the calls that actually pause a run (fits, plane searches,
// making images); quick calls never show. The line names the call in plain words, counts
// up and carries a hairline bar. When this job has run the same tool before and it took at
// least SHOW_AFTER_S, the bar fills towards the time it took then (its median) and turns to
// a slow sweep once the call runs longer; any other call sweeps from the start. A call
// still open after GIVEN_UP_S belongs to a run that stopped without closing it, and is
// not shown.
import { AnimatePresence, motion } from 'motion/react';
import { useEffect, useState } from 'react';
import { DUR, EASE_OUT } from '../ui/motion';
import type { Call } from '../data/run';
import { callDetail } from '../data/acts';

const SHOWN_AT_ONCE = 2;
const NEAR_FULL = 0.94;
const SHOW_AFTER_S = 4;
const GIVEN_UP_S = 6 * 3600;

export function Doing({ calls }: { calls: Call[] }) {
  const now = useNow();
  const slow = calls.filter(c => {
    const s = secondsSince(c.seenAt, now) ?? 0;
    return c.running && s >= SHOW_AFTER_S && s < GIVEN_UP_S;
  });
  return (
    <div className="doing" aria-live="polite">
      <AnimatePresence initial={false}>
        {slow.slice(-SHOWN_AT_ONCE).map(c => (
          <Wait key={c.key} call={c} elapsed={secondsSince(c.seenAt, now) ?? 0} usual={usualSeconds(calls, c.name)} />
        ))}
        {slow.length > SHOWN_AT_ONCE && (
          <motion.p key="more" className="doing-more" {...FADE}>and {slow.length - SHOWN_AT_ONCE} more running</motion.p>
        )}
      </AnimatePresence>
    </div>
  );
}

const FADE = {
  initial: { opacity: 0 },
  animate: { opacity: 1, transition: { duration: DUR.enter, ease: EASE_OUT } },
  exit: { opacity: 0, transition: { duration: DUR.quick, ease: EASE_OUT } },
};

function Wait({ call, elapsed, usual }: { call: Call; elapsed: number; usual: number | null }) {
  const over = usual != null && elapsed > usual;
  const share = usual != null && !over ? Math.min(NEAR_FULL, elapsed / usual) : null;
  const detail = detailOf(call);
  return (
    <motion.div className="wait" {...FADE}>
      <p className="wait-line">
        <span className="wait-what">{call.title}</span>
        <span className="wait-detail">{detail}</span>
        <span className="wait-clock">{clock(elapsed)}{usual != null && <span className="wait-usual">{over ? ` / usually ${clock(usual)}` : ` of about ${clock(usual)}`}</span>}</span>
      </p>
      <div className={'wait-bar' + (share == null ? ' sweep' : '')} role="progressbar"
        aria-valuenow={share == null ? undefined : Math.round(share * 100)}>
        {share != null && <span className="wait-fill" style={{ transform: `scaleX(${share})` }} />}
      </div>
    </motion.div>
  );
}

function detailOf(call: Call) {
  const detail = callDetail(call);
  const n = call.sections.length;
  const on = !n ? '' : n === 1 ? 'one section' : `${n} sections`;
  return [detail, on].filter(Boolean).join(', ');
}

function usualSeconds(calls: Call[], name: string): number | null {
  const times = calls.filter(c => c.name === name && c.ended && !c.failed).map(tookSeconds)
    .filter((t): t is number => t != null).sort((a, b) => a - b);
  const median = times.length ? times[Math.floor(times.length / 2)] : null;
  return median != null && median >= SHOW_AFTER_S ? median : null;
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
