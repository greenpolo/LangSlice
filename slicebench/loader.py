"""Bench loader — joins data/slicebench.json + per-shard rows + active eval allocations.

A bench (small or large) is a list of `EvalRow`s. Each row carries the truth
position, atlas info, and `plane_extent_mm` (sagittal halved for canonical
hemisphere) so scoring can compute plane-relative accuracy without re-loading
atlases per row.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Iterator

from langslice_harness.atlas.core import get_position_range_mm, load_atlas

REPO_ROOT = Path(__file__).resolve().parents[1]
SLICEBENCH_JSON = REPO_ROOT / "data" / "slicebench.json"
SHARDS_ROOT = REPO_ROOT / "data" / "manifest" / "shards"
ALLOCATIONS_ROOT = REPO_ROOT / "data" / "manifest" / "allocations"


@dataclass(frozen=True)
class EvalRow:
    section_id: str
    dataset: str
    subject_id: str
    plane: str          # "coronal" | "sagittal" | "horizontal"
    atlas: str
    image_path: str     # relative to repo root
    truth_mm: float
    slice_axis: str     # "ap" | "ml" | "dv"
    species: str
    staining: str | None
    imaging: str | None
    plane_extent_mm: float   # canonical (sagittal halved) extent

    def to_dict(self) -> dict:
        return asdict(self)


def _iter_jsonl(path: Path) -> Iterator[dict]:
    with path.open("r", encoding="utf-8") as fh:
        for raw in fh:
            stripped = raw.strip()
            if not stripped:
                continue
            yield json.loads(stripped)


def _load_active_allocation(plane: str, split: str = "eval") -> set[str]:
    """Return the set of active section_ids for a (plane, split). Tombstones honored."""
    path = ALLOCATIONS_ROOT / plane / f"{split}.jsonl"
    if not path.exists():
        return set()
    state: dict[str, dict] = {}
    for entry in _iter_jsonl(path):
        sid = entry["section_id"]
        if entry.get("action") == "remove":
            state.pop(sid, None)
        else:
            state[sid] = entry
    return set(state)


def _plane_extent_mm(atlas_name: str, plane: str) -> float:
    """Plane's mm extent along the slice-normal axis. Sagittal halved (canonical hemisphere)."""
    atlas = load_atlas(atlas_name)
    _, hi = get_position_range_mm(atlas, plane=plane)
    if plane == "sagittal":
        return hi / 2.0
    return hi


def _stratified_sample(rows: list[EvalRow], n: int) -> list[EvalRow]:
    """Pick `n` evenly-spaced rows after sorting by truth_mm. Deterministic."""
    if n <= 0 or len(rows) <= n:
        return rows
    sorted_rows = sorted(rows, key=lambda r: r.truth_mm)
    # Evenly-spaced indices from 0 to len-1 inclusive (no RNG)
    indices = [round(i * (len(sorted_rows) - 1) / (n - 1)) for i in range(n)]
    # Dedupe in case rounding collides (shouldn't unless len < n, handled above)
    seen: set[int] = set()
    picked: list[EvalRow] = []
    for idx in indices:
        if idx in seen:
            continue
        seen.add(idx)
        picked.append(sorted_rows[idx])
    return picked


def load_bench(name: str) -> list[EvalRow]:
    """Return the tiny, small, or large SliceBench row list.

    Joins three sources of truth:
      - data/slicebench.json — the canonical brain list (tiny=8 brains × N/brain,
        small=8 brains, large=24 brains)
      - data/manifest/shards/<plane>/<dataset>.jsonl — per-section ground-truth rows
      - data/manifest/allocations/<plane>/eval.jsonl — active eval assignments

    A row is dropped (with stderr warning) if its section_id isn't in the active
    eval allocation. That keeps the bench honest after manifest curation that
    moves rows between splits.

    For ``tiny`` (or any bench whose brain entries carry an ``n_sections`` field),
    the loader deterministically samples N evenly-spaced sections per brain after
    sorting by truth_mm. Reproducible across runs without an RNG seed.
    """
    if name not in {"tiny", "small", "large"}:
        raise ValueError(f"unknown bench {name!r}; expected 'tiny', 'small' or 'large'")

    with SLICEBENCH_JSON.open("r", encoding="utf-8") as fh:
        sb = json.load(fh)
    if name not in sb:
        raise ValueError(f"data/slicebench.json missing key {name!r}")
    brain_list = sb[name]

    # Group brains by plane to load each shard + allocation only once
    by_plane: dict[str, list[dict]] = {}
    for brain in brain_list:
        by_plane.setdefault(brain["plane"], []).append(brain)

    rows: list[EvalRow] = []
    for plane, brains in by_plane.items():
        active_eval = _load_active_allocation(plane, "eval")

        # Group by dataset within plane so we read each shard at most once
        by_dataset: dict[str, set[str]] = {}
        brain_meta: dict[tuple[str, str], dict] = {}
        for brain in brains:
            dataset = brain["dataset"]
            subject_id = brain["subject_id"]
            by_dataset.setdefault(dataset, set()).add(subject_id)
            brain_meta[(dataset, subject_id)] = brain

        # Collect rows per (dataset, subject) so per-brain caps can apply
        rows_by_brain: dict[tuple[str, str], list[EvalRow]] = {}
        for dataset, subjects in by_dataset.items():
            shard_path = SHARDS_ROOT / plane / f"{dataset}.jsonl"
            if not shard_path.exists():
                print(f"WARN: shard not found, skipping: {shard_path}")
                continue

            for shard_row in _iter_jsonl(shard_path):
                if shard_row.get("subject_id") not in subjects:
                    continue
                section_id = shard_row["section_id"]
                if section_id not in active_eval:
                    continue

                atlas_name = shard_row["atlas"]
                row = EvalRow(
                    section_id=section_id,
                    dataset=dataset,
                    subject_id=shard_row["subject_id"],
                    plane=plane,
                    atlas=atlas_name,
                    image_path=shard_row["image_path"],
                    truth_mm=float(shard_row["position_mm"]),
                    slice_axis=shard_row["slice_axis"],
                    species=shard_row.get("species", brain_meta[(dataset, shard_row["subject_id"])].get("species", "unknown")),
                    staining=shard_row.get("staining") or brain_meta[(dataset, shard_row["subject_id"])].get("staining"),
                    imaging=shard_row.get("imaging") or brain_meta[(dataset, shard_row["subject_id"])].get("imaging"),
                    plane_extent_mm=_plane_extent_mm(atlas_name, plane),
                )
                rows_by_brain.setdefault((dataset, shard_row["subject_id"]), []).append(row)

        # Apply per-brain stratified sampling if requested
        for (dataset, subject_id), brain_rows in rows_by_brain.items():
            n_cap = brain_meta.get((dataset, subject_id), {}).get("n_sections")
            if n_cap is not None:
                rows.extend(_stratified_sample(brain_rows, int(n_cap)))
            else:
                rows.extend(brain_rows)

    return rows
