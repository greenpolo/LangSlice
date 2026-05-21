"""Pre-render every atlas slice referenced by the QC manifest to disk.

The QC app's `/api/atlas` endpoint cold-loads BrainGlobe NIFTI volumes on
first request — 30-60s per atlas. With ~30 atlases per session that's a
half-hour of UI stalls. This script walks the manifest, batches all
required (atlas, plane, position_mm) tuples by atlas (one cold-load per
atlas instead of one per first-navigation), renders each composite slice,
and writes PNGs to `_local/qc_app/atlas_cache/`. The QC app reads from
that cache before falling back to live render.

Usage:
    python _local/qc_app/precompute_atlas_slices.py
    python _local/qc_app/precompute_atlas_slices.py --include-estimates \
_local/trace_collection/runs/.../results.jsonl
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
import time
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(REPO_ROOT / "src"))

from langslice_data.qc_app.app import (  # noqa: E402
    ATLAS_DISK_CACHE,
    _disk_cache_path,
    _quantize_position,
    render_atlas_png,
)

DEFAULT_MANIFEST = REPO_ROOT / "_local/trace_collection/gemini31_position_trace_manifest_qc.jsonl"

logger = logging.getLogger("precompute_atlas_slices")


def collect_targets(
    manifest_path: Path, estimates_path: Path | None
) -> list[tuple[str, str, float]]:
    """Return a deduplicated list of (atlas, plane, position_mm) tuples."""
    seen: set[tuple[str, str, int]] = set()
    targets: list[tuple[str, str, float]] = []

    estimates: dict[str, dict] = {}
    if estimates_path and estimates_path.exists():
        for raw in estimates_path.read_text(encoding="utf-8").splitlines():
            s = raw.strip()
            if not s:
                continue
            d = json.loads(s)
            tid = d.get("id")
            if tid:
                estimates[tid] = d

    def _add(atlas: str, plane: str, pos: float) -> None:
        if not atlas or not plane:
            return
        key = (atlas, plane, _quantize_position(pos))
        if key in seen:
            return
        seen.add(key)
        targets.append((atlas, plane, pos))

    for raw in manifest_path.read_text(encoding="utf-8").splitlines():
        s = raw.strip()
        if not s:
            continue
        e = json.loads(s)
        if e.get("metadata", {}).get("atlas_unmappable"):
            continue
        atlas = e.get("atlas")
        plane = e.get("plane")
        if not atlas or not plane:
            continue
        if e.get("kind") == "group":
            for p in e.get("positions_mm", []):
                if isinstance(p, (int, float)):
                    _add(atlas, plane, float(p))
        else:
            p = e.get("position_mm")
            if isinstance(p, (int, float)):
                _add(atlas, plane, float(p))

        est = estimates.get(e.get("id"))
        if not est:
            continue
        if e.get("kind") == "group":
            for p in est.get("estimated_positions_mm") or []:
                if isinstance(p, (int, float)):
                    _add(atlas, plane, float(p))
        else:
            p = est.get("estimated_position_mm")
            if isinstance(p, (int, float)):
                _add(atlas, plane, float(p))

    return targets


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument(
        "--include-estimates",
        type=Path,
        default=None,
        help="optional Gemini results jsonl; if given, also render estimated positions",
    )
    parser.add_argument(
        "--mode",
        default="composite",
        choices=("composite", "reference"),
        help="atlas render mode to cache (UI uses composite)",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=0,
        help="render at most N targets (0 = no limit, useful for smoke tests)",
    )
    args = parser.parse_args(argv)

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )

    targets = collect_targets(args.manifest, args.include_estimates)
    targets.sort(key=lambda t: (t[0], t[1], t[2]))
    if args.limit:
        targets = targets[: args.limit]

    by_atlas: dict[str, int] = {}
    for atlas, _, _ in targets:
        by_atlas[atlas] = by_atlas.get(atlas, 0) + 1
    logger.info(
        "%d unique slices across %d atlases -> %s",
        len(targets),
        len(by_atlas),
        ATLAS_DISK_CACHE,
    )

    rendered = 0
    skipped_existing = 0
    failed = 0
    t_start = time.monotonic()
    last_atlas = None
    for atlas, plane, pos in targets:
        if atlas != last_atlas:
            logger.info("atlas %s (%d slices)", atlas, by_atlas[atlas])
            last_atlas = atlas
        cache_path = _disk_cache_path(atlas, plane, _quantize_position(pos), args.mode)
        if cache_path.exists():
            skipped_existing += 1
            continue
        try:
            render_atlas_png(atlas, plane, pos, args.mode)
            rendered += 1
        except Exception:
            logger.exception("render failed: %s / %s / %.4f", atlas, plane, pos)
            failed += 1

    elapsed = time.monotonic() - t_start
    logger.info(
        "done: rendered=%d skipped_existing=%d failed=%d in %.1fs",
        rendered,
        skipped_existing,
        failed,
        elapsed,
    )
    return 0 if failed == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
