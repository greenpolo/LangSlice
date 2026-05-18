"""Live RLVR training progress tracker for the QC inventory app.

This module defines the **on-disk contract** between the RLVR trainer and the
QC web app, and implements the QC-side reader.

CONTRACT
========

The trainer writes to ``_local/rlvr_progress/`` (created on first write):

1. ``consumed.jsonl`` -- **append-only** history. One line per training step.
   Schema::

       {
         "step":  int,                     # monotonically increasing step idx
         "ts":    float,                   # epoch seconds (time.time())
         "ids":   ["<dataset>/<section_id>", ...],   # 1+ ids; group samples
                                                     # bundle 2-8 ids
         "kind":  "single" | "group",
         "phase": "train" | "eval"         # which loop produced this step
       }

   The trainer must ``open(path, "a", encoding="utf-8")``, write the JSON-encoded
   line followed by ``"\n"``, and ``flush()``/``close()`` so partial reads on
   the QC side don't see torn lines. The file is read incrementally on the
   QC side (only the bytes appended since the last poll), so size is fine
   even for multi-day runs that grow into the hundreds of MB.

2. ``queue.json`` -- **atomically rewritten** every time the trainer knows its
   next-batch contents. Schema::

       {
         "step":   int,                                # the step this queue
                                                       # represents (i.e. the
                                                       # next step to run)
         "ts":     float,
         "queued": [
           {"ids": ["<dataset>/<section_id>", ...], "kind": "single" | "group"},
           ...
         ]
       }

   The trainer should write to ``queue.json.tmp`` then ``os.replace`` it to
   ``queue.json`` so the QC app never reads a half-written file. If the queue
   is empty (e.g. trainer paused), write ``{"step": N, "ts": ..., "queued": []}``
   rather than deleting the file.

3. ``run.json`` -- **optional** run metadata. Schema (all keys optional except
   ``started_at``)::

       {
         "config":         "<config name>",   # e.g. "grpo_phase_b"
         "started_at":     <epoch float>,
         "target_steps":   <int>,
         "target_samples": <int>,
         "model":          "<model name>",
         "notes":          "..."
       }

   Rewritten atomically (``run.json.tmp`` -> ``os.replace``). When present, the
   QC app surfaces these in the Training tab's header strip.

Defensive behavior: every file is optional. Missing files yield empty counts,
not 500s. Malformed JSON lines in ``consumed.jsonl`` are skipped with a warning.

QC-SIDE READER
==============

``TrainingProgress`` is a thread-safe lazy singleton-ish reader. It caches a
parsed ``id -> {step, ts, kind, phase}`` map for the consumed log and
recomputes the queue + run blob each time the underlying file mtime changes.
Hot path is ``status_payload()`` which is called from the polling endpoint
every ~2 seconds; the per-id consumed map is only re-tailed if the file grew.
"""

from __future__ import annotations

import json
import logging
import threading
import time
from collections import deque
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterable

logger = logging.getLogger("qc_app.training_progress")

# Default progress directory, sibling to data/ and other _local/* artifacts.
DEFAULT_PROGRESS_DIR_NAME = "_local/rlvr_progress"
CONSUMED_FILENAME = "consumed.jsonl"
QUEUE_FILENAME = "queue.json"
RUN_FILENAME = "run.json"

# Per-step records kept in memory for the "recent activity" feed. Bounded so
# multi-day runs don't OOM. We never need to scan more than this for the UI.
_RECENT_TAIL_MAX = 500


@dataclass
class _ConsumedRecord:
    step: int
    ts: float
    kind: str
    phase: str


@dataclass
class _StepEvent:
    """One row of consumed.jsonl, kept verbatim for the recent-activity feed."""

    step: int
    ts: float
    ids: list[str]
    kind: str
    phase: str


@dataclass
class _ConsumedState:
    """Cache state for consumed.jsonl. Not exposed; mutated under a lock."""

    last_size: int = 0
    last_mtime: float = 0.0
    # Latest consumed-event metadata per id. Re-trains overwrite earlier rows.
    by_id: dict[str, _ConsumedRecord] = field(default_factory=dict)
    # Tail of step events for the activity feed. Bounded.
    recent: deque[_StepEvent] = field(default_factory=lambda: deque(maxlen=_RECENT_TAIL_MAX))
    last_step: int = -1


class TrainingProgress:
    """Reads RLVR trainer telemetry from ``progress_dir``.

    Designed to be cheap on the hot path: ``status_payload()`` re-stats three
    files and only re-parses when something changed. ``consumed.jsonl`` reads
    are incremental -- we keep ``last_size`` and only seek to that offset on
    the next tick.
    """

    def __init__(self, progress_dir: Path) -> None:
        self.progress_dir = Path(progress_dir)
        self._lock = threading.Lock()
        self._consumed = _ConsumedState()
        # queue.json + run.json are tiny -- just cache the parsed value per
        # mtime. Defensive defaults so a never-written file == empty queue.
        self._queue_mtime: float = 0.0
        self._queue_payload: dict = {"step": -1, "ts": 0.0, "queued": []}
        self._run_mtime: float = 0.0
        self._run_payload: dict | None = None

    # -- file paths ----------------------------------------------------
    @property
    def consumed_path(self) -> Path:
        return self.progress_dir / CONSUMED_FILENAME

    @property
    def queue_path(self) -> Path:
        return self.progress_dir / QUEUE_FILENAME

    @property
    def run_path(self) -> Path:
        return self.progress_dir / RUN_FILENAME

    # -- public API used by the HTTP layer -----------------------------
    def refresh(self) -> None:
        """Re-read any progress files whose mtime changed.

        Call this before reading ``consumed_by_id`` / ``queued_ids`` / etc. so
        the data is fresh. Safe to call from many threads concurrently.
        """
        with self._lock:
            self._refresh_consumed_locked()
            self._refresh_queue_locked()
            self._refresh_run_locked()

    def consumed_by_id(self) -> dict[str, dict]:
        """Snapshot of id -> {step, ts, kind, phase} (the latest event per id)."""
        with self._lock:
            return {
                rid: {
                    "step": rec.step,
                    "ts": rec.ts,
                    "kind": rec.kind,
                    "phase": rec.phase,
                }
                for rid, rec in self._consumed.by_id.items()
            }

    def queued_ids(self) -> set[str]:
        """Set of ids appearing anywhere in the current queue.json contents."""
        with self._lock:
            return self._queued_ids_locked()

    def status_payload(self, *, recent_limit: int = 20) -> dict:
        """JSON-serializable snapshot for ``/api/training/status``.

        Includes everything the live UI needs in one round-trip:
          * current step
          * counts (consumed_ids / queued_ids / unseen requires manifest, so
            we only return what we have here -- the route layer crosses this
            with the rlvr split for unseen)
          * a tail of recent events
          * queue contents
          * run metadata (if present)
        """
        self.refresh()
        with self._lock:
            recent = [
                {
                    "step": ev.step,
                    "ts": ev.ts,
                    "ids": list(ev.ids),
                    "kind": ev.kind,
                    "phase": ev.phase,
                }
                for ev in list(self._consumed.recent)[-recent_limit:]
            ]
            recent.reverse()  # newest first
            queue_payload = dict(self._queue_payload)
            queued_ids = self._queued_ids_locked()
            return {
                "available": self._any_file_present_locked(),
                "current_step": self._consumed.last_step,
                "n_consumed_ids": len(self._consumed.by_id),
                "n_queued_ids": len(queued_ids),
                "queue": queue_payload,
                "recent": recent,
                "run": dict(self._run_payload) if self._run_payload else None,
                "last_consumed_ts": (
                    self._consumed.recent[-1].ts if self._consumed.recent else None
                ),
                "polled_at": time.time(),
            }

    def recent_payload(self, limit: int) -> list[dict]:
        """Last ``limit`` consumed events, newest-first."""
        self.refresh()
        with self._lock:
            recent = list(self._consumed.recent)[-limit:]
            recent.reverse()
            return [
                {
                    "step": ev.step,
                    "ts": ev.ts,
                    "ids": list(ev.ids),
                    "kind": ev.kind,
                    "phase": ev.phase,
                }
                for ev in recent
            ]

    # -- internal: consumed.jsonl --------------------------------------
    def _refresh_consumed_locked(self) -> None:
        path = self.consumed_path
        try:
            st = path.stat()
        except FileNotFoundError:
            # If the file disappeared (cleared between runs), reset state.
            if self._consumed.by_id or self._consumed.last_size:
                self._consumed = _ConsumedState()
            return

        # File got smaller -> truncated (new run probably). Wipe and rebuild.
        if st.st_size < self._consumed.last_size:
            self._consumed = _ConsumedState()

        if st.st_size == self._consumed.last_size and st.st_mtime == self._consumed.last_mtime:
            return  # unchanged

        # Tail-read from the previous offset to current end. The trainer is
        # supposed to flush full lines, but be defensive: if the last partial
        # line is incomplete, leave it unread until next tick.
        try:
            with path.open("rb") as fh:
                fh.seek(self._consumed.last_size)
                buf = fh.read()
        except OSError as exc:
            logger.warning("consumed.jsonl read failed: %s", exc)
            return

        text = buf.decode("utf-8", errors="replace")
        # If the buffer doesn't end on a newline, we have a partial trailing
        # line; trim back to the last complete line so we re-read it next tick.
        if text and not text.endswith("\n"):
            cut = text.rfind("\n")
            if cut == -1:
                # Whole buffer is a partial line; leave last_size unchanged.
                return
            consumed_bytes = (text[: cut + 1]).encode("utf-8")
            new_size = self._consumed.last_size + len(consumed_bytes)
            text = text[: cut + 1]
        else:
            new_size = st.st_size

        for raw in text.splitlines():
            line = raw.strip()
            if not line:
                continue
            try:
                rec = json.loads(line)
            except json.JSONDecodeError:
                logger.warning("skipping malformed consumed.jsonl line: %s", line[:120])
                continue
            self._apply_consumed_record_locked(rec)

        self._consumed.last_size = new_size
        self._consumed.last_mtime = st.st_mtime

    def _apply_consumed_record_locked(self, rec: dict) -> None:
        try:
            step = int(rec.get("step", -1))
        except (TypeError, ValueError):
            return
        ts = float(rec.get("ts", 0.0) or 0.0)
        ids_field = rec.get("ids") or []
        if not isinstance(ids_field, list):
            return
        ids = [str(x) for x in ids_field if isinstance(x, str)]
        kind = str(rec.get("kind", "single"))
        phase = str(rec.get("phase", "train"))
        ev = _StepEvent(step=step, ts=ts, ids=ids, kind=kind, phase=phase)
        self._consumed.recent.append(ev)
        for rid in ids:
            self._consumed.by_id[rid] = _ConsumedRecord(
                step=step, ts=ts, kind=kind, phase=phase
            )
        if step > self._consumed.last_step:
            self._consumed.last_step = step

    # -- internal: queue.json ------------------------------------------
    def _refresh_queue_locked(self) -> None:
        path = self.queue_path
        try:
            st = path.stat()
        except FileNotFoundError:
            if self._queue_mtime != 0.0:
                self._queue_mtime = 0.0
                self._queue_payload = {"step": -1, "ts": 0.0, "queued": []}
            return
        if st.st_mtime == self._queue_mtime:
            return
        try:
            text = path.read_text(encoding="utf-8")
            payload = json.loads(text)
        except (OSError, json.JSONDecodeError) as exc:
            logger.warning("queue.json unreadable: %s", exc)
            return
        if not isinstance(payload, dict):
            logger.warning("queue.json not a dict, ignoring")
            return
        # Light shape validation.
        queued = payload.get("queued", [])
        if not isinstance(queued, list):
            queued = []
        cleaned = []
        for item in queued:
            if not isinstance(item, dict):
                continue
            ids = item.get("ids") or []
            if not isinstance(ids, list):
                continue
            cleaned.append(
                {
                    "ids": [str(x) for x in ids if isinstance(x, str)],
                    "kind": str(item.get("kind", "single")),
                }
            )
        self._queue_payload = {
            "step": int(payload.get("step", -1) or -1),
            "ts": float(payload.get("ts", 0.0) or 0.0),
            "queued": cleaned,
        }
        self._queue_mtime = st.st_mtime

    def _queued_ids_locked(self) -> set[str]:
        out: set[str] = set()
        for item in self._queue_payload.get("queued", []):
            for rid in item.get("ids", []):
                if isinstance(rid, str):
                    out.add(rid)
        return out

    # -- internal: run.json --------------------------------------------
    def _refresh_run_locked(self) -> None:
        path = self.run_path
        try:
            st = path.stat()
        except FileNotFoundError:
            if self._run_mtime != 0.0:
                self._run_mtime = 0.0
                self._run_payload = None
            return
        if st.st_mtime == self._run_mtime:
            return
        try:
            text = path.read_text(encoding="utf-8")
            payload = json.loads(text)
        except (OSError, json.JSONDecodeError) as exc:
            logger.warning("run.json unreadable: %s", exc)
            return
        if not isinstance(payload, dict):
            return
        self._run_payload = payload
        self._run_mtime = st.st_mtime

    def _any_file_present_locked(self) -> bool:
        return (
            self.consumed_path.exists()
            or self.queue_path.exists()
            or self.run_path.exists()
        )


def status_per_id(
    progress: TrainingProgress, ids: Iterable[str]
) -> dict[str, dict]:
    """Convenience: project consumed/queued metadata for a set of ids.

    Output: id -> {"status": "consumed"|"queued"|"unseen",
                   "step": int|None, "ts": float|None,
                   "queued_again": bool}

    A row that is *both* consumed and queued is reported as ``status="consumed"``
    with ``queued_again=True`` so the UI can show the "consumed, queued for
    re-train" sub-marker.
    """
    progress.refresh()
    consumed = progress.consumed_by_id()
    queued = progress.queued_ids()
    out: dict[str, dict] = {}
    for rid in ids:
        c = consumed.get(rid)
        is_queued = rid in queued
        if c is not None:
            out[rid] = {
                "status": "consumed",
                "step": c["step"],
                "ts": c["ts"],
                "kind": c["kind"],
                "phase": c["phase"],
                "queued_again": is_queued,
            }
        elif is_queued:
            out[rid] = {
                "status": "queued",
                "step": None,
                "ts": None,
                "kind": None,
                "phase": None,
                "queued_again": False,
            }
        else:
            out[rid] = {
                "status": "unseen",
                "step": None,
                "ts": None,
                "kind": None,
                "phase": None,
                "queued_again": False,
            }
    return out
