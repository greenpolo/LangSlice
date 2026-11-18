// The Simple view: who is speaking and what it says, the sections it works on (or Review),
// and where they sit: front to back (2D), or in the atlas brain beside them (3D).
import { motion } from 'motion/react';
import type { ReactNode } from 'react';
import { useEffect } from 'react';
import { EASE_OUT } from '../motion';
import { overlayUrl, warm } from '../pictures';
import { useRun } from '../run';
import type { Focus } from '../Viewer';
import { justSwitched, STAGE_IN, STAGE_SHOWN, useStageMode } from '../space/mode';
import { SpaceStage } from '../space/SpaceStage';
import { CuttingAxis } from './CuttingAxis';
import { Speaker } from './Speaker';
import { geometry, Stage, useBox, useDirector } from './Stage';

export function SimpleView({ focus, review, ring }: { focus: Focus; review: ReactNode; ring: string | null }) {
  const run = useRun();
  const [stageRef, box] = useBox<HTMLDivElement>();
  const done = run.status === 'Submitted';
  const space = useStageMode() === '3d' && !review && run.job !== null;
  const shown = useDirector(run.job, focus.ids, run.rows, box);
  const running = [...run.calls].reverse().find(c => c.running);
  const call = running ?? run.calls.at(-1) ?? null;
  // Every section's picture at the grid's size, fetched while nothing else is, so the
  // finished run's grid is ready when it comes.
  const job = run.job;
  const rowKeys = run.rows.map(r => r.id + ':' + (r.render_key ?? '')).join('|');
  useEffect(() => {
    if (job === null || !box || !run.rows.length) return;
    const edge = geometry(run.rows.length, box).edge;
    warm(run.rows.map(r => overlayUrl(job, r.id, edge, r.render_key)));
  }, [job, rowKeys, box]); // eslint-disable-line react-hooks/exhaustive-deps
  return (
    <main className={'simple' + (space ? ' space' : '')}>
      <Speaker model={run.model} status={run.status} words={run.words} call={call} focusCount={shown.length} />
      {space && run.job !== null ? (
        <SpaceStage job={run.job} rows={run.rows} strip={run.strip} shown={shown} box={box} sideRef={stageRef}
          onPick={focus.pick} focus={ring ? [ring] : focus.ids.length === run.rows.length ? [] : shown} />
      ) : (
        <motion.div className="stage" ref={stageRef} initial={justSwitched() ? STAGE_IN : false} animate={STAGE_SHOWN}
          transition={{ duration: 0.24, ease: EASE_OUT }}>
          {review ?? (run.job !== null && <Stage job={run.job} shown={shown} rows={run.rows} box={box} onPick={focus.pick} />)}
          {run.jobs && !run.jobs.length && <p className="empty">No jobs yet: open a job folder with <code>langslice gui FOLDER</code>.</p>}
        </motion.div>
      )}
      {!space && (
        <CuttingAxis job={run.job} rows={run.rows} axis={run.axis} strip={run.strip} focus={ring ? [ring] : shown}
          shown={ring ? [ring] : shown.length <= 3 ? shown : []} done={done} onPick={focus.pick} />
      )}
    </main>
  );
}
