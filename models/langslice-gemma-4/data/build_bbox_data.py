"""Orchestrator for the BBox training data pipeline.

Two stages, both via this script's CLI:
- `--stage sample` writes a draft manifest + bbox-overlay PNGs (no API calls).
- `--stage submit --verdicts <path>` filters by QC verdicts, packs batch JSONL,
   submits to AI Studio Batch API, polls, retrieves, mm-strips, writes final SFT.

Spec: docs/superpowers/specs/2026-05-04-gemma4-bbox-training-design.md
"""
from __future__ import annotations

import argparse
import logging
import sys
from collections.abc import Sequence
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO_ROOT / "src"))
sys.path.insert(0, str(REPO_ROOT / "models" / "langslice-gemma-4" / "data"))
sys.path.insert(0, str(REPO_ROOT / "_local" / "eval" / "lib"))

log = logging.getLogger("build_bbox_data")

_MIN_REAL_SECTIONS_FOR_ELIGIBILITY = 4
_MAX_PER_BRAIN_REGION = 3
_SECTION_COUNTS = (4, 5, 6, 7, 8)
_SPACING_MIN_MM = 0.2
_SPACING_MAX_MM = 0.8


def pick_source(
    *,
    atlas: str,
    orientation: str,
    landmark: str,
    coverage_index: dict,
    rng: np.random.Generator,
    source_counts: dict[tuple[str, str], int] | None = None,
) -> dict:
    """Decide source type for one example.

    Spec §4.4: real Tier A if coverage index reports any brain with >=4 covered
    sections for this (atlas, orientation, landmark) tuple; else split between
    augmented and reference atlas with equal probability.
    """
    eligible_brains: list[tuple[str, list[str]]] = []
    for brain_id, by_landmark in coverage_index.items():
        entry = by_landmark.get(landmark)
        if entry is None:
            continue
        if entry["atlas"] != atlas or entry["orientation"] != orientation:
            continue
        section_ids = entry.get("section_ids", [])
        if source_counts is not None:
            if source_counts.get((brain_id, landmark), 0) >= _MAX_PER_BRAIN_REGION:
                continue
        if len(section_ids) >= _MIN_REAL_SECTIONS_FOR_ELIGIBILITY:
            eligible_brains.append((brain_id, section_ids))

    if eligible_brains:
        weights = np.array([len(secs) for _, secs in eligible_brains], dtype=np.float64)
        weights /= weights.sum()
        idx = int(rng.choice(len(eligible_brains), p=weights))
        brain_id, section_ids = eligible_brains[idx]
        return {
            "source_type": "real_histology",
            "source_brain": brain_id,
            "available_section_ids": section_ids,
        }

    # Atlas fallback -- equal split between augmented and reference.
    if rng.random() < 0.5:
        return {"source_type": "augmented_atlas", "source_brain": None}
    return {"source_type": "reference_atlas", "source_brain": None}


def sample_section_count(rng: np.random.Generator) -> int:
    return int(rng.choice(np.array(_SECTION_COUNTS, dtype=np.int64)))


def sample_spacings_mm(rng: np.random.Generator, n_gaps: int) -> list[float]:
    """Sample one independent continuous spacing for each adjacent-section gap."""
    return [
        float(rng.uniform(_SPACING_MIN_MM, _SPACING_MAX_MM))
        for _ in range(n_gaps)
    ]


def sample_anchor_mm(
    *,
    rng: np.random.Generator,
    region_mm_min: float,
    region_mm_max: float,
    spacings_mm: Sequence[float],
) -> float | None:
    """Anchor for the first-section position. Returns None if span doesn't fit."""
    span = sum(spacings_mm)
    max_anchor = region_mm_max - span
    if max_anchor < region_mm_min:
        return None
    return float(rng.uniform(region_mm_min, max_anchor))


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description="BBox training data orchestrator")
    p.add_argument("--stage", required=True, choices=("sample", "submit"))
    p.add_argument("--target-total", type=int, default=300)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument(
        "--out-dir",
        type=Path,
        default=REPO_ROOT / "_local" / "bbox_data",
    )
    p.add_argument(
        "--coverage-index",
        type=Path,
        default=REPO_ROOT / "_local" / "eval" / "data" / "landmark_coverage.json",
    )
    p.add_argument("--verdicts", type=Path, default=None,
                   help="QC verdicts JSONL (required for --stage submit)")
    p.add_argument("--model", default="gemini-3.1-pro-preview")
    p.add_argument("--dry-run", action="store_true",
                   help="Build batch JSONL but don't submit")
    return p.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    logging.basicConfig(level=logging.INFO)
    if args.stage == "sample":
        from _stage_sample import run_stage_sample  # type: ignore
        return run_stage_sample(args)
    if args.stage == "submit":
        if args.verdicts is None:
            log.error("--verdicts is required for --stage submit")
            return 2
        from _stage_submit import run_stage_submit  # type: ignore
        return run_stage_submit(args)
    return 2


if __name__ == "__main__":
    sys.exit(main())
