// The 3D stage of both views: the sections in focus in a narrow column at the left (the
// same Stage, its columns chosen by the column's box) and the atlas brain filling the rest.
import type { Row, Strip } from '../run';
import { Stage, type Box } from '../simple/Stage';
import { Volume } from './Volume';

export function SpaceStage({ job, rows, strip, shown, focus, box, sideRef, onPick }: {
  job: number; rows: Row[]; strip: Strip | null; shown: string[]; focus: string[]; box: Box | null;
  sideRef: (el: HTMLDivElement | null) => void; onPick: (id: string) => void;
}) {
  return (
    <div className="stage space-split">
      <div className="space-side" ref={sideRef}>
        <Stage job={job} shown={shown} rows={rows} box={box} onPick={onPick} />
      </div>
      <Volume job={job} rows={rows} strip={strip} onPick={onPick} focus={focus} />
    </div>
  );
}
