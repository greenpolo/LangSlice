// The Extended view, for a person who wants the whole run: the speaker band with how far
// the registration is (placed, aligned, warped); the sections in focus, four to a page,
// each with its name, position and how it stands, and the front-to-back line under them
// (2D), or the same sections in a column beside the atlas brain with every section where
// it is placed (3D, which takes the line's place); and on the right the journal.
import { motion } from 'motion/react';
import type { ReactNode } from 'react';
import { useState } from 'react';
import { EASE_OUT } from '../motion';
import { useRun, type Row } from '../run';
import { CuttingAxis } from '../simple/CuttingAxis';
import { Speaker } from '../simple/Speaker';
import { Stage, useBox, useDirector } from '../simple/Stage';
import { justSwitched, STAGE_IN, STAGE_SHOWN, useStageMode } from '../space/mode';
import { SpaceStage } from '../space/SpaceStage';
import type { Focus } from '../Viewer';
import { Journal } from './Journal';

const PAGE = 4;

export function ExtendedView({ focus, review, ring, onPictures }: {
  focus: Focus; review: ReactNode; ring: string | null; onPictures: (numbers: number[], start: number) => void;
}) {
  const run = useRun();
  const [stageRef, box] = useBox<HTMLDivElement>();
  const [page, setPage] = useState(0);
  const space = useStageMode() === '3d' && !review && run.job !== null;
  const pages = Math.max(1, Math.ceil(focus.ids.length / PAGE));
  const onPage = focus.ids.slice(Math.min(page, pages - 1) * PAGE, Math.min(page, pages - 1) * PAGE + PAGE);
  const shown = useDirector(run.job, onPage, run.rows, box);
  const done = run.status === 'Submitted';
  const running = [...run.calls].reverse().find(c => c.running);
  const call = running ?? run.calls.at(-1) ?? null;
  return (
    <main className={'extended' + (space ? ' space' : '')}>
      <div className="extended-head">
        <Speaker model={run.model} status={run.status} words={run.words} call={call} focusCount={shown.length} />
        <Progress rows={run.rows} tasks={run.header.tasks} done={done} />
      </div>
      <div className="extended-body">
        <section className="extended-main">
          <div className="extended-tools">
            <span className="extended-count">
              {!focus.following ? 'You are looking at ' : done ? 'Registered: ' : 'The agent is working on '}
              <b>{focus.ids.length === run.rows.length ? `all ${run.rows.length} sections` : `${focus.ids.length} of ${run.rows.length} sections`}</b>
              {pages > 1 && `, page ${Math.min(page, pages - 1) + 1} of ${pages}`}
            </span>
            {pages > 1 && (
              <span className="pager">
                <button className="ghost-button" aria-label="Previous page" onClick={() => setPage(p => (p - 1 + pages) % pages)}>←</button>
                <button className="ghost-button" aria-label="Next page" onClick={() => setPage(p => (p + 1) % pages)}>→</button>
              </span>
            )}
          </div>
          {space && run.job !== null ? (
            <SpaceStage job={run.job} rows={run.rows} strip={run.strip} shown={shown} box={box} sideRef={stageRef}
              onPick={focus.pick} focus={ring ? [ring] : focus.ids.length === run.rows.length ? [] : shown} />
          ) : (
            <motion.div className="stage" ref={stageRef} initial={justSwitched() ? STAGE_IN : false} animate={STAGE_SHOWN}
          transition={{ duration: 0.24, ease: EASE_OUT }}>
              {review ?? (run.job !== null && <Stage job={run.job} shown={shown} rows={run.rows} box={box} onPick={focus.pick} detailed />)}
            </motion.div>
          )}
          {!space && (
            <CuttingAxis job={run.job} rows={run.rows} axis={run.axis} strip={run.strip} focus={ring ? [ring] : focus.ids}
              shown={ring ? [ring] : shown} done={done} onPick={focus.pick} atlas />
          )}
        </section>
        <Journal onPictures={onPictures} onSection={focus.pick} />
      </div>
    </main>
  );
}

// How far the registration is, in the steps the job asked for (`tasks` in its settings):
// placed, aligned, warped, each counted over the job's sections. A job without a task list
// (an older one) shows all three. Done once the agent has submitted.
const STEPS = [
  { task: 'position', label: 'Placed in the atlas', has: (r: Row) => typeof r.position_mm === 'number' && r.position_source !== 'initial' },
  { task: 'transform', label: 'Aligned to the atlas', has: (r: Row) => Boolean(r.transform) },
  { task: 'nonlinear', label: 'Warped to fit', has: (r: Row) => Boolean(r.deformation) },
] as const;

function Progress({ rows, tasks, done }: { rows: Row[]; tasks: unknown; done: boolean }) {
  const asked = Array.isArray(tasks) ? (tasks as string[]) : null;
  const steps = STEPS.filter(s => !asked || asked.includes(s.task));
  if (!rows.length || !steps.length) return <div className="progress-slot" />;
  const counts = steps.map(s => rows.filter(s.has).length);
  const finished = counts.filter(n => n === rows.length).length;
  const left = asked ? STEPS.filter(s => !asked.includes(s.task)).map(s => s.label.toLowerCase()) : [];
  return (
    <section className="progress" aria-label="How far the registration is">
      <div className="progress-head">
        <b>{done ? 'Submitted' : 'Progress'}</b>
        <span>{finished} of {steps.length} {steps.length === 1 ? 'step' : 'steps'}</span>
      </div>
      <ol className="progress-steps">
        {steps.map((s, i) => {
          const n = counts[i];
          return (
            <li key={s.task} className={'meter' + (n === rows.length ? ' full' : n === 0 ? ' none' : '')}>
              <span className="meter-label">{s.label}</span>
              <span className="meter-count">{n} of {rows.length}</span>
              <span className="meter-bar">
                <motion.span className="meter-fill" initial={false} animate={{ scaleX: n / rows.length }} transition={{ duration: 0.5, ease: EASE_OUT }} />
              </span>
            </li>
          );
        })}
      </ol>
      {left.length > 0 && <p className="progress-note">Not asked for: {left.join(', ')}.</p>}
    </section>
  );
}
