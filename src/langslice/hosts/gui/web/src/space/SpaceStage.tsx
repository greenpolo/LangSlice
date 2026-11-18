// The 3D stage of both views: the sections in focus in a narrow column at the left (the
// same Stage, its columns chosen by the column's box) and the atlas brain filling the rest.
import { motion, useReducedMotion } from 'motion/react';
import { EASE_OUT } from '../motion';
import type { Row, Strip } from '../run';
import { Stage, type Box } from '../simple/Stage';
import { justSwitched, STAGE_IN, STAGE_SHOWN } from './mode';
import { Volume } from './Volume';

export function SpaceStage({ job, rows, strip, shown, focus, box, sideRef, onPick }: {
  job: number; rows: Row[]; strip: Strip | null; shown: string[]; focus: string[]; box: Box | null;
  sideRef: (el: HTMLDivElement | null) => void; onPick: (id: string) => void;
}) {
  const reduce = useReducedMotion();
  return (
    <motion.div className="stage space-split"
      initial={justSwitched() ? (reduce ? { opacity: 0 } : STAGE_IN) : false}
      animate={reduce ? { opacity: 1 } : STAGE_SHOWN} transition={{ duration: 0.24, ease: EASE_OUT }}>
      <div className="space-side" ref={sideRef}>
        <Stage job={job} shown={shown} rows={rows} box={box} onPick={onPick} />
      </div>
      <Volume job={job} rows={rows} strip={strip} onPick={onPick} focus={focus} />
    </motion.div>
  );
}
