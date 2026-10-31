"""Stand in for the user's agent app on a job prepared from ABBA.

``E2EFollowCheck`` runs this after the connector has prepared the job and
started the follower. Every call is a process of its own through the agent
CLI (``langslice-job FOLDER VERB``), exactly as a Codex or Claude Code run
makes them, so the follower sees the job's ``state.json`` change the way it
does in use. No model is called. The last line, ``DRIVER final: {...}``, is
the job's state as saved, for the Java side to compare with ABBA.

Usage: ``python driver.py PROJECT_FOLDER`` (the folder holding the snapshots).
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
import time
from datetime import UTC, datetime
from pathlib import Path

#: Long verbs (the FireANTs fit) run in the calling process, as one call.
ENVIRONMENT = dict(os.environ, LANGSLICE_DETACH="0")


def call(
    folder: Path, verb: str, *flags: str, args: dict[str, object] | None = None,
) -> dict[str, object]:
    """Run one agent CLI call; stop the driver when it fails."""
    argv = [sys.executable, "-m", "langslice.doors.agent_cli.main", str(folder), verb, *flags]
    if args is not None:
        argv += ["--args", json.dumps(args)]
    started = time.perf_counter()
    done = subprocess.run(argv, capture_output=True, text=True, env=ENVIRONMENT, check=False)
    seconds = time.perf_counter() - started
    try:
        reply = json.loads(done.stdout)
    except json.JSONDecodeError:
        sys.exit(f"DRIVER {verb}: no reply (exit {done.returncode}): {done.stderr[-2000:]}")
    result = reply.get("result")
    brief = ({key: result.get(key) for key in ("status", "message")}
             if isinstance(result, dict) else {})
    # When the call ended, to set against abba_applied.jsonl's "at": how long ABBA took.
    ended = datetime.now(UTC).isoformat(timespec="milliseconds")
    print(f"DRIVER {verb}: ok={reply.get('ok')} {seconds:.1f} s, ended {ended} "
          f"{json.dumps(brief)[:300]}", flush=True)
    if not reply.get("ok"):
        sys.exit(f"DRIVER {verb} failed: {json.dumps(reply.get('error'))[:2000]}")
    return reply


def final_state(folder: Path) -> dict[str, object]:
    """The job's state as last saved: what ABBA should show once the follower has caught up."""
    from langslice.doors.jobs import open_folder

    opened = open_folder(folder, persist=False)
    with opened.job.reading():
        state = opened.job.state
        sections = {
            section.id: {
                "position_mm": section.position_mm,
                "flip": section.flip,
                "rotation_deg": section.rotation_deg,
                "transform": section.transform is not None,
                "deformation": section.deformation is not None,
            }
            for section in state.in_order()
        }
        pitch, yaw = state.stack_angles
    return {"sections": sections, "pitch_deg": pitch, "yaw_deg": yaw, "submitted": state.submitted}


def main() -> None:
    folder = Path(sys.argv[1])
    ids = sorted(json.loads((folder / "abba_sections.json").read_text()))
    print("DRIVER sections:", ids, flush=True)
    # The prepared job carries ABBA's inputs; init continues it with the tasks only.
    call(folder, "init", "--tasks", "position,transform,nonlinear")
    call(folder, "position_sections", args={
        "sections": [{"id": ids[0], "position_mm": 6.0}, {"id": ids[1], "position_mm": 6.3}],
        "cutting_angles": {"pitch_deg": 2.0, "yaw_deg": -1.0}, "view": False})
    call(folder, "interactive_transform", args={
        "sections": [{"id": ids[0], "rotation_deg": 10.0, "scale_x": 1.1, "translate_x_mm": 0.2}],
        "view": False})
    call(folder, "interactive_transform", args={
        "sections": [{"id": ids[1], "flip": True, "rotate_quarter": 90, "scale_x": 1.0}],
        "view": False})
    # Both sections in one call: the follower converts their warps side by side.
    call(folder, "fireants_syn", args={"sections": ids, "view": False})
    call(folder, "interactive_transform", args={
        "sections": [{"id": ids[1], "translate_y_mm": -0.1}], "view": False})
    call(folder, "undo")
    call(folder, "interactive_transform", args={
        "sections": [{"id": ids[1], "rotation_deg": 3.0}], "view": False})
    call(folder, "submit", args={
        "summary": "End-to-end probe of the ABBA connector.", "notes": "", "interval_breaks": []})
    print("DRIVER final:", json.dumps(final_state(folder)), flush=True)


if __name__ == "__main__":
    main()
