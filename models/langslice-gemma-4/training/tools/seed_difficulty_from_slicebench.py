"""Seed Round-0 per-section difficulty from a slicebench summary JSON.

Round-0 of the curriculum-driven RLVR run needs an estimate of "how hard is
this section" for the *full* ~50k-section pool so the AdaRFT-style sampler
has something to bias toward before live rollouts have covered everything.
Live rollouts only score sections the policy actually visits; slicebench
can score the entire pool independently and writes those scores to a
``summary.json`` (see ``slicebench/score.py``).

This one-shot CLI converts that summary into the JSON shape that
:meth:`single_turn_rl.manifest_index.ManifestIndex.bulk_load_difficulty`
already consumes. The trainer (Task 6) calls ``bulk_load_difficulty`` at
startup; sections that don't appear in the seed file fall back to the
curriculum's uniform default.

Two seed paths
--------------

The summary JSON may carry a ``per_section`` block (one row per section
with its own ``mae_mm``) or only a ``per_coord_bin`` block (per-plane,
per-quintile MAE). The CLI prefers the former when present and otherwise
expands the bin MAE across every matching section in the manifest:

    1. ``per_section`` — one output row per per_section entry.
       ``difficulty_score = mae_mm / axis_extent_mm``.
    2. ``per_coord_bin`` — for every (plane, q_label) bin, walk the manifest
       index, compute each section's quintile via
       :func:`slicebench.score._bin_index`, and write
       ``difficulty_score = bin_mae / axis_extent_mm`` for sections in
       that bin.

If neither block is present the CLI exits with a clear error so a wrong
``--slicebench-summary`` doesn't silently produce an empty seed file.

Output shape
------------

Mirrors what ``bulk_load_difficulty`` reads::

    {
      "rows": [
        {
          "plane": "coronal",
          "dataset": "allen_connectivity",
          "section_id": "272414403:272414770",
          "difficulty_score": 0.42,
          "source": "slicebench-tiny-round0"
        },
        ...
      ]
    }

Atomic write
------------

Writes to ``<output>.tmp`` and ``os.replace``s into place — same pattern
as :meth:`ManifestIndex.persist_live_difficulty` so a crash mid-write
never leaves a partial seed file in the consumed location.

Invocation
----------

::

    python models/langslice-gemma-4/training/tools/seed_difficulty_from_slicebench.py \\
        --slicebench-summary slicebench/runs/tiny/<run>/summary.json \\
        --manifest-root data/manifest \\
        --output _local/rlvr_progress/seeded_difficulty.json \\
        [--source-label "slicebench-tiny-round0"]
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from collections.abc import Callable, Iterable
from pathlib import Path
from typing import Any


# Make the gemma single-turn RL package + slicebench importable when the tool
# is run directly from the checkout.
def _find_repo_root() -> Path:
    for parent in Path(__file__).resolve().parents:
        if (parent / "pyproject.toml").is_file():
            return parent
    msg = "Could not locate LangSlice repo root from seed-difficulty tool path."
    raise RuntimeError(msg)


_REPO_ROOT = _find_repo_root()
_GEMMA4_TRAINING = Path(__file__).resolve().parents[1]
for _p in (_REPO_ROOT, _GEMMA4_TRAINING):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

# Wildcard import would drag in the QC-app-loader side effect at module import
# time; explicit names keep the dependency surface visible.
from single_turn_rl.manifest_index import ManifestIndex, Section  # noqa: E402

from slicebench.score import _bin_index  # noqa: E402, PLC2701 — stable internal API

# Cap the number of orphan keys we list inline before truncating with
# "... (N more)". Mirrors ``manifest_index._ORPHAN_KEY_PREVIEW_LIMIT`` so
# operator-facing logs look the same.
_ORPHAN_KEY_PREVIEW_LIMIT: int = 5

# Number of quintile buckets slicebench writes for ``per_coord_bin``. The
# label maps q1..q5 → bin index 0..4 (see ``slicebench.score._coord_bin_label``).
_N_COORD_BINS: int = 5


# ---------------------------------------------------------------------------
# Atlas-extent helper (lazy-import, cached). Mirrors
# ``terminal_states._atlas_valid_range_mm``.
# ---------------------------------------------------------------------------


_ATLAS_EXTENT_CACHE: dict[tuple[str, str], float] = {}


def _axis_extent_mm(atlas_name: str, plane: str) -> float:
    """Return ``pos_hi - pos_lo`` for ``(atlas, plane)``, cached.

    Lazy-imports ``langslice_harness.atlas.core`` so unit tests can monkeypatch
    this function without needing a BrainGlobe atlas on disk.
    """
    key = (atlas_name, plane)
    cached = _ATLAS_EXTENT_CACHE.get(key)
    if cached is not None:
        return cached
    # Local import — keeps the module importable in unit tests that don't
    # have BrainGlobe atlases on disk (mirrors the pattern used by
    # ``single_turn_rl.terminal_states._atlas_valid_range_mm`` and
    # ``curriculum.bins._plane_extent_mm``).
    from langslice_harness.atlas.core import (  # noqa: PLC0415
        get_position_range_mm,
        load_atlas,
    )

    atlas = load_atlas(atlas_name)
    pos_lo, pos_hi = get_position_range_mm(atlas, plane=plane)  # type: ignore[arg-type]
    extent = float(pos_hi) - float(pos_lo)
    _ATLAS_EXTENT_CACHE[key] = extent
    return extent


def _clear_extent_cache() -> None:
    """Test helper — drop the in-process atlas-extent cache."""
    _ATLAS_EXTENT_CACHE.clear()


# ---------------------------------------------------------------------------
# Per-section parsing
# ---------------------------------------------------------------------------


def _parse_per_section_key(
    key: str, entry: dict[str, Any],
) -> tuple[str, str, str] | None:
    """Resolve ``(plane, dataset, section_id)`` from a per_section row.

    Slicebench's ``per_section`` block is keyed by section_id (see
    ``score._collapse_by_section``). The entry value typically carries
    ``plane`` and ``dataset`` fields, but historical / external producers
    may flatten the key as ``"<plane>:<dataset>:<section_id>"`` or
    ``"<dataset>:<section_id>"``. We tolerate all three by preferring the
    fields on the entry value when present and falling back to the key
    parts.

    Returns ``None`` when neither the entry nor the key carries enough
    information to produce a ``(plane, dataset, section_id)`` triple.
    """
    plane = entry.get("plane")
    dataset = entry.get("dataset")
    section_id_field = entry.get("section_id")

    # Decompose the key into colon-separated parts so we can use them as
    # fallbacks for any missing entry field.
    parts = key.split(":")
    plane_from_key: str | None = None
    dataset_from_key: str | None = None
    section_from_key: str

    if len(parts) >= 3 and parts[0] in {"coronal", "sagittal", "horizontal"}:
        # "<plane>:<dataset>:<section_id>" — section_id may itself contain ':'
        # so we re-join the tail.
        plane_from_key = parts[0]
        dataset_from_key = parts[1]
        section_from_key = ":".join(parts[2:])
    elif len(parts) >= 2:
        # "<dataset>:<section_id>" or "<subject>:<section>" — ambiguous, but
        # since the QC app's manifest section_id can also contain ':' (e.g.
        # ``"272414403:272414770"``), we treat the whole key as the
        # section_id and rely on the entry's plane+dataset fields.
        section_from_key = key
    else:
        section_from_key = key

    plane_str = str(plane) if plane else (plane_from_key or "")
    dataset_str = str(dataset) if dataset else (dataset_from_key or "")
    section_id_str = (
        str(section_id_field) if section_id_field else section_from_key
    )

    if not plane_str or not dataset_str or not section_id_str:
        return None
    return plane_str, dataset_str, section_id_str


# ---------------------------------------------------------------------------
# Row-emission paths
# ---------------------------------------------------------------------------


def _rows_from_per_section(
    per_section: dict[str, Any],
    *,
    source_label: str,
    extent_lookup: Callable[[str, str], float] | None = None,
) -> tuple[list[dict[str, Any]], int, list[tuple[str, str, str]]]:
    """Emit one row per ``per_section`` entry.

    Returns ``(rows, n_skipped, orphan_keys)`` where ``orphan_keys`` lists
    entries whose (plane, dataset, section_id) couldn't be parsed or whose
    atlas extent couldn't be resolved (logged separately for triage).
    """
    extent_fn = extent_lookup or _axis_extent_mm
    rows: list[dict[str, Any]] = []
    n_skipped = 0
    orphan_keys: list[tuple[str, str, str]] = []

    for key, entry in per_section.items():
        if not isinstance(entry, dict):
            n_skipped += 1
            continue
        try:
            mae = float(entry["mae_mm"])
        except (KeyError, TypeError, ValueError):
            n_skipped += 1
            continue
        atlas = entry.get("atlas")
        plane_dataset_section = _parse_per_section_key(key, entry)
        if plane_dataset_section is None or not atlas:
            n_skipped += 1
            continue
        plane, dataset, section_id = plane_dataset_section
        try:
            extent = float(extent_fn(str(atlas), plane))
        except (ImportError, OSError, RuntimeError, ValueError, KeyError) as exc:
            print(
                f"[seed_difficulty] could not resolve axis extent for "
                f"({atlas!r}, {plane!r}): {exc}",
                file=sys.stderr,
                flush=True,
            )
            n_skipped += 1
            orphan_keys.append((plane, dataset, section_id))
            continue
        if extent <= 0:
            n_skipped += 1
            continue
        rows.append(
            {
                "plane": plane,
                "dataset": dataset,
                "section_id": section_id,
                # Clamp into [0, 1] — bad slicebench rows can produce
                # mae > extent which would otherwise push the curriculum
                # sampler past its expected difficulty domain.
                "difficulty_score": max(0.0, min(1.0, mae / extent)),
                "source": source_label,
            }
        )

    return rows, n_skipped, orphan_keys


def _rows_from_per_coord_bin(
    per_coord_bin: dict[str, Any],
    sections: Iterable[Section],
    *,
    source_label: str,
    extent_lookup: Callable[[str, str], float] | None = None,
) -> tuple[list[dict[str, Any]], int]:
    """Expand per-bin MAE across every matching section in ``sections``.

    For each ``(plane, q_label)`` in ``per_coord_bin`` we look up that bin's
    ``mae_mm`` and walk every manifest section on that plane. Each section
    is binned via :func:`slicebench.score._bin_index` against its own
    ``(atlas, plane)`` extent; rows landing in a bin with MAE data get a
    ``difficulty_score = bin_mae / extent`` entry.

    Returns ``(rows, n_skipped)`` where ``n_skipped`` counts sections whose
    atlas extent couldn't be resolved or whose bin had no MAE data
    (latter is normal — a plane may have ``q3`` MAE but no ``q5`` MAE if
    no eval row landed there).
    """
    extent_fn = extent_lookup or _axis_extent_mm

    # Pre-compute {plane: {bin_idx: mae_mm}} so the inner loop is O(1) per
    # section. Bin labels q1..q5 map to bin_idx 0..4 (see
    # ``slicebench.score._coord_bin_label``).
    plane_bin_mae: dict[str, dict[int, float]] = {}
    for plane, bins in per_coord_bin.items():
        if not isinstance(bins, dict):
            continue
        per_bin: dict[int, float] = {}
        for label, stats in bins.items():
            if not isinstance(stats, dict):
                continue
            try:
                bin_mae = float(stats["mae_mm"])
            except (KeyError, TypeError, ValueError):
                continue
            # Label is "q<N>" where N is 1-indexed; convert to 0-indexed bin_idx.
            if not (isinstance(label, str) and label.startswith("q")):
                continue
            try:
                bin_idx = int(label[1:]) - 1
            except ValueError:
                continue
            if 0 <= bin_idx < _N_COORD_BINS:
                per_bin[bin_idx] = bin_mae
        if per_bin:
            plane_bin_mae[str(plane)] = per_bin

    rows: list[dict[str, Any]] = []
    n_skipped = 0
    for section in sections:
        bin_table = plane_bin_mae.get(section.plane)
        if bin_table is None:
            continue
        try:
            extent = float(extent_fn(section.atlas, section.plane))
        except (ImportError, OSError, RuntimeError, ValueError, KeyError) as exc:
            print(
                f"[seed_difficulty] could not resolve axis extent for "
                f"({section.atlas!r}, {section.plane!r}): {exc}",
                file=sys.stderr,
                flush=True,
            )
            n_skipped += 1
            continue
        if extent <= 0:
            n_skipped += 1
            continue
        bin_idx = _bin_index(section.position_mm, extent, _N_COORD_BINS)
        bin_mae = bin_table.get(bin_idx)
        if bin_mae is None:
            # Plane has no MAE for this bin (e.g. no eval row in that
            # quintile). Skip — the curriculum's uniform fallback handles
            # unobserved sections.
            continue
        rows.append(
            {
                "plane": section.plane,
                "dataset": section.dataset,
                "section_id": section.section_id,
                # Clamp into [0, 1] — same rationale as the per_section
                # path; a bin_mae greater than the plane's extent (degenerate
                # slicebench output) must not push the curriculum sampler
                # past its expected difficulty domain.
                "difficulty_score": max(0.0, min(1.0, bin_mae / extent)),
                "source": source_label,
            }
        )
    return rows, n_skipped


def _filter_orphan_rows(
    rows: list[dict[str, Any]],
    index: ManifestIndex,
) -> tuple[list[dict[str, Any]], list[tuple[str, str, str]]]:
    """Drop rows whose ``(plane, dataset, section_id)`` isn't in the index.

    Returns ``(kept_rows, orphan_keys)``. Used for the per_section path —
    slicebench may have been run on a different section pool (a smaller
    eval cut) than the manifest the trainer reads, so some per_section
    keys won't resolve. We log them once with stderr but don't fail.
    """
    orphan_keys: list[tuple[str, str, str]] = []
    kept: list[dict[str, Any]] = []
    for row in rows:
        key = (row["plane"], row["dataset"], row["section_id"])
        if index.section_by_key(*key) is None:
            orphan_keys.append(key)
            continue
        kept.append(row)
    return kept, orphan_keys


def _log_orphan_warning(
    *,
    source: str,
    n_orphan: int,
    n_invalid: int,
    orphan_keys: list[tuple[str, str, str]],
) -> None:
    """Emit a stderr warning matching ``manifest_index._log_orphan_warning``.

    Lists up to :data:`_ORPHAN_KEY_PREVIEW_LIMIT` offending orphan keys so
    the same grep-from-the-log pattern works against the seed CLI's
    output as it does against ``bulk_load_difficulty`` warnings.
    """
    if n_orphan == 0 and n_invalid == 0:
        return
    preview = orphan_keys[:_ORPHAN_KEY_PREVIEW_LIMIT]
    extra = len(orphan_keys) - len(preview)
    suffix = ""
    if preview:
        suffix = f"; first orphans: {preview!r}"
        if extra > 0:
            suffix += f" ({extra} more)"
    print(
        f"[seed_difficulty] skipped {n_orphan} orphan + "
        f"{n_invalid} invalid row(s) from {source}{suffix}",
        file=sys.stderr,
        flush=True,
    )


# ---------------------------------------------------------------------------
# Driver
# ---------------------------------------------------------------------------


def build_rows(
    summary: dict[str, Any],
    index: ManifestIndex,
    *,
    source_label: str,
    extent_lookup: Callable[[str, str], float] | None = None,
    summary_path_for_logs: str | None = None,
) -> list[dict[str, Any]]:
    """Build the seed rows. Prefers ``per_section`` over ``per_coord_bin``.

    Raises ``ValueError`` if neither block is present (early-fail so a
    wrong ``--slicebench-summary`` doesn't silently emit an empty seed).
    """
    per_section = summary.get("per_section")
    per_coord_bin = summary.get("per_coord_bin")

    log_source = summary_path_for_logs or "<summary>"

    # Inline isinstance + truthiness so the narrow type flows into the
    # _rows_from_* call (basedpyright doesn't propagate narrowing through
    # an intermediate bool).
    if isinstance(per_section, dict) and per_section:
        rows, n_invalid, parse_orphans = _rows_from_per_section(
            per_section, source_label=source_label, extent_lookup=extent_lookup,
        )
        rows, manifest_orphans = _filter_orphan_rows(rows, index)
        all_orphans = parse_orphans + manifest_orphans
        _log_orphan_warning(
            source=log_source,
            n_orphan=len(all_orphans),
            n_invalid=n_invalid,
            orphan_keys=all_orphans,
        )
        return rows

    if isinstance(per_coord_bin, dict) and per_coord_bin:
        rows, n_invalid = _rows_from_per_coord_bin(
            per_coord_bin,
            index.query(),
            source_label=source_label,
            extent_lookup=extent_lookup,
        )
        if n_invalid:
            _log_orphan_warning(
                source=log_source,
                n_orphan=0,
                n_invalid=n_invalid,
                orphan_keys=[],
            )
        return rows

    raise ValueError(
        f"Slicebench summary at {log_source} has neither a non-empty "
        f"'per_section' block nor a non-empty 'per_coord_bin' block — "
        f"can't seed any difficulty rows. Re-check the summary file."
    )


def write_seed_atomic(rows: list[dict[str, Any]], output_path: Path) -> None:
    """Atomically write ``{"rows": rows}`` to ``output_path``.

    Mirrors :meth:`ManifestIndex.persist_live_difficulty`: writes to a
    sibling ``<output>.tmp`` first and ``os.replace``-into-place so a
    crash mid-write never leaves a half-written seed file in the
    consumed location.
    """
    output_path.parent.mkdir(parents=True, exist_ok=True)
    payload = {"rows": rows}
    # Build the tmp path manually rather than using ``with_suffix`` so a
    # multi-part suffix (``foo.json.gz`` → ``foo.json.gz.tmp``) survives.
    tmp_path = output_path.with_name(output_path.name + ".tmp")
    with tmp_path.open("w", encoding="utf-8") as fh:
        json.dump(payload, fh, ensure_ascii=False, indent=2)
        fh.write("\n")
    os.replace(tmp_path, output_path)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(
        prog="seed_difficulty_from_slicebench",
        description=(
            "Convert a slicebench summary JSON into a seeded-difficulty "
            "JSON for ManifestIndex.bulk_load_difficulty."
        ),
    )
    p.add_argument(
        "--slicebench-summary",
        type=Path,
        required=True,
        help="Path to a slicebench summary.json (see slicebench/score.py).",
    )
    p.add_argument(
        "--manifest-root",
        type=Path,
        default=Path("data/manifest"),
        help="Root of the data manifest (canonical: data/manifest).",
    )
    p.add_argument(
        "--output",
        type=Path,
        required=True,
        help="Destination JSON path (e.g. _local/rlvr_progress/seeded_difficulty.json).",
    )
    p.add_argument(
        "--source-label",
        type=str,
        default="slicebench",
        help=(
            "String stamped into each row's 'source' field. Defaults to "
            "'slicebench'; pass something more descriptive for a "
            "specific run (e.g. 'slicebench-tiny-round0')."
        ),
    )
    return p.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = _parse_args(argv)
    summary_path = Path(args.slicebench_summary)
    if not summary_path.is_file():
        print(
            f"[seed_difficulty] slicebench summary not found: {summary_path}",
            file=sys.stderr,
            flush=True,
        )
        return 2
    with summary_path.open("r", encoding="utf-8") as fh:
        summary = json.load(fh)
    if not isinstance(summary, dict):
        print(
            f"[seed_difficulty] slicebench summary at {summary_path} is not a "
            f"JSON object (got {type(summary).__name__}).",
            file=sys.stderr,
            flush=True,
        )
        return 2

    index = ManifestIndex.from_manifest_root(Path(args.manifest_root))

    try:
        rows = build_rows(
            summary,
            index,
            source_label=args.source_label,
            summary_path_for_logs=str(summary_path),
        )
    except ValueError as exc:
        print(f"[seed_difficulty] {exc}", file=sys.stderr, flush=True)
        return 2

    write_seed_atomic(rows, Path(args.output))
    print(
        f"[seed_difficulty] wrote {len(rows)} row(s) to {args.output} "
        f"(source label: {args.source_label!r})",
        file=sys.stderr,
        flush=True,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
