"""Append-only JSONL writer for per-section curriculum events.

Each rollout / eval pass should append a row per section so Phase C can
read back per-bin abs-error distributions and update sampling weights
between expert-iteration rounds. The writer is intentionally minimal:
single-process, line-buffered append, no locking.

Schema
------
::

    {"round": int,
     "section_id": str,
     "bin_idx": int,
     "plane": str,
     "abs_err_mm": float,
     "accuracy_pct": float,
     "ts": iso8601}

``ts`` is filled in by the writer if the caller omits it, so callers can
just pass the four "what happened" fields plus ``round``.
"""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

REQUIRED_FIELDS: tuple[str, ...] = (
    "round",
    "section_id",
    "bin_idx",
    "plane",
    "abs_err_mm",
    "accuracy_pct",
)


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


class CurriculumLogger:
    """Append rows to ``path`` as JSONL.

    The file (and its parent directory) is created on first append. Calls
    to :meth:`append` are individually flushed so a crashed run still
    leaves a readable log.
    """

    def __init__(self, path: Path | str) -> None:
        self.path = Path(path)

    def append(self, row: dict[str, Any]) -> None:
        """Write ``row`` as a single JSONL line. Adds ``ts`` if missing.

        Raises ``ValueError`` if any of the required fields is missing —
        we'd rather fail loudly than silently log half-rows that Phase C
        can't aggregate.
        """
        missing = [f for f in REQUIRED_FIELDS if f not in row]
        if missing:
            raise ValueError(
                f"CurriculumLogger.append: missing required field(s): {missing}"
            )
        out = dict(row)
        if "ts" not in out:
            out["ts"] = _now_iso()
        self.path.parent.mkdir(parents=True, exist_ok=True)
        # ``a`` is fine for our single-process append-only use case. If we
        # ever spread logging across multiple processes we'd want fcntl /
        # msvcrt locking around the write, but that's not the situation now.
        with self.path.open("a", encoding="utf-8") as fh:
            fh.write(json.dumps(out, sort_keys=False) + "\n")
