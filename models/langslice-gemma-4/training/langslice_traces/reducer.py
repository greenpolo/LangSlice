"""Best-per-id reducer over a directory tree of raw trace JSON files."""

from __future__ import annotations

import json
from pathlib import Path

from .accuracy import (
    canonicalize_positions,
    plane_rescue_threshold_mm,
    plane_tolerance_mm,
)


def best_traces(
    trace_dir: Path, *, strict_only: bool = False,
) -> dict[str, dict]:
    """Walk every runs*/batch_*/runs/<id>/raw_trace.json and pick the lowest-error
    trace per id, scored canonical + plane-relative.

    By default keeps any trace with err <= rescue_threshold (the loose pool —
    strict accepts plus near-misses). Set strict_only=True to keep only the
    in-tolerance subset.

    Each entry's `tier` field is "strict" (err <= tolerance) or "rescued"
    (tolerance < err <= rescue_threshold) so callers can stratify.
    """
    best: dict[str, dict] = {}
    for runs_dir in sorted(trace_dir.glob("runs*")):
        if not runs_dir.is_dir():
            continue
        # Two on-disk layouts coexist:
        #   batched:  runs_dir / batch_NNN / runs / <id> / raw_trace.json
        #   flat:     runs_dir / runs / <id> / raw_trace.json
        # rglob covers both without false-positive matches.
        for raw in runs_dir.rglob("runs/*/raw_trace.json"):
            try:
                d = json.loads(raw.read_text(encoding="utf-8"))
            except Exception:
                continue
            m = d.get("manifest") or {}
            sid = m.get("id")
            if not sid or m.get("kind") != "single":
                continue
            plane = m.get("plane")
            atlas = m.get("atlas")
            sub = d.get("submitted_positions_mm") or []
            tru = m.get("truth_positions_mm") or []
            if not plane or not atlas or not sub or not tru:
                continue
            sub_c = canonicalize_positions(sub, atlas, plane) or sub
            tru_c = canonicalize_positions(tru, atlas, plane) or tru
            err = abs(float(sub_c[0]) - float(tru_c[0]))
            tol = plane_tolerance_mm(atlas, plane)
            rescue = plane_rescue_threshold_mm(atlas, plane)
            cap = tol if strict_only else rescue
            if err > cap:
                continue
            existing = best.get(sid)
            if existing is None or err < existing["err"]:
                best[sid] = {
                    "raw_path": raw,
                    "err": err,
                    "tol": tol,
                    "rescue": rescue,
                    "tier": "strict" if err <= tol else "rescued",
                    "sub_canonical_mm": float(sub_c[0]),
                    "manifest": m,
                    "events": d.get("events") or [],
                    "model": d.get("model"),
                }
    return best
