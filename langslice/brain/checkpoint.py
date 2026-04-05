"""Incremental JSON checkpoint for whole-brain estimation."""

from __future__ import annotations

import json
import os

from langslice.brain.types import BrainEstimationConfig, SlicePosition


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
        "ordering_mode": config.ordering,
        "z_axis": config.z_axis,
        "slices": [
            {
                "filename": s.filename,
                "index": s.index,
                "position_mm": round(s.position_mm, 4),
                "source": s.source,
                "locked": s.locked,
            }
            for s in slices
        ],
    }
    with open(path, "w") as f:
        json.dump(data, f, indent=2)


def load_checkpoint(path: str) -> list[SlicePosition]:
    """Load slice positions from a checkpoint file.

    Returns an empty list if the file does not exist.
    """
    if not os.path.exists(path):
        return []
    with open(path) as f:
        data = json.load(f)
    return [
        SlicePosition(
            filename=s["filename"],
            index=s["index"],
            position_mm=s["position_mm"],
            source=s["source"],
            locked=s["locked"],
        )
        for s in data["slices"]
    ]
