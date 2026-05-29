"""Strict nucleus-size table — the SINGLE SOURCE OF TRUTH for nucleus diameters.

Reads ``celltype_data/size_priors.csv`` (compiled adult-mouse DAPI nucleus sizes).
The stampers MUST size nuclei from here; do not hand-pick radii. All values are
nucleus DIAMETER in microns.

    oligodendrocyte 6.5 (5.0-8.0)   astrocyte 7.0 (5.0-9.0)
    microglia       6.5 (5.0-8.0)   neuron_generic 5.2 (4.7-5.8)
"""
from __future__ import annotations

import csv
from functools import lru_cache
from pathlib import Path

CSV = Path(__file__).resolve().parent / "data" / "size_priors.csv"


@lru_cache(maxsize=1)
def _table() -> dict[str, dict[str, float]]:
    out: dict[str, dict[str, float]] = {}
    with open(CSV, newline="", encoding="utf-8") as f:
        for r in csv.DictReader(f):
            rng = r["nucleus_diameter_um_range"].replace("–", "-").replace("—", "-")
            try:
                lo, hi = (float(x) for x in rng.split("-"))
                mean = float(r["nucleus_diameter_um_mean"])
            except ValueError:
                continue
            out[r["cell_type"]] = {"mean": mean, "lo": lo, "hi": hi}
    return out


def diameter_mean_um(cell_type: str) -> float:
    return _table()[cell_type]["mean"]


def diameter_range_um(cell_type: str) -> tuple[float, float]:
    e = _table()[cell_type]
    return e["lo"], e["hi"]


def radius_range_um(cell_type: str) -> tuple[float, float]:
    """(min_radius, max_radius) in um for a cell type's nucleus (diameter/2)."""
    lo, hi = diameter_range_um(cell_type)
    return lo / 2.0, hi / 2.0


def cell_types() -> list[str]:
    return list(_table())
