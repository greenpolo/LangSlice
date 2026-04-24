"""Incremental JSON checkpoint for whole-brain estimation."""

from __future__ import annotations

import json
import os

from langslice_harness.whole_brain.types import BrainEstimationConfig, SlicePosition


def save_checkpoint(
    path: str,
    config: BrainEstimationConfig,
    slices: list[SlicePosition],
) -> None:
    """Write current state to a JSON file (overwrites)."""
    data = {
        "atlas": config.atlas_name,
        "thickness_um": config.thickness_um,
        "interval_um": config.interval_um,
        "z_axis": config.z_axis,
        "slices": [
            {
                "filename": s.filename,
                "index": s.index,
                "position_mm": round(s.position_mm, 4),
                "source": s.source,
                "locked": s.locked,
                "raw_position_mm": round(s.raw_position_mm, 4)
                if s.raw_position_mm is not None
                else None,
            }
            for s in slices
        ],
    }
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2)


def load_checkpoint(path: str) -> list[SlicePosition]:
    """Load slice positions from a checkpoint file.

    Returns an empty list if the file does not exist.
    """
    if not os.path.exists(path):
        return []
    with open(path, encoding="utf-8") as f:
        data = json.load(f)
    return [
        SlicePosition(
            filename=s["filename"],
            index=s["index"],
            position_mm=s["position_mm"],
            source=s["source"],
            locked=s["locked"],
            raw_position_mm=s.get("raw_position_mm"),
        )
        for s in data["slices"]
    ]
