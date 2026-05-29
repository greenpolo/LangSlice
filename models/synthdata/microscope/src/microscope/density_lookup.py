"""Join the Ero-2018 per-region density table to the stamper.

The other session produced celltype_data/density_by_region.csv (955 regions,
keyed to Allen CCFv3 structure IDs) with total + per-class cell densities. For
DAPI we need exactly two numbers per region:

  total_density_cells_per_mm3  -> stamp_nuclei(density_per_mm3=...)
  neuron_density / total       -> make_real_template_bank(neuron_frac=...)

We collapse the 4 classes to neuron-vs-glial (our two shape pools); the glia
breakdown (astro/oligo/microglia) is not needed here. White matter is the case
that matters: neuron_frac -> 0 there but total stays high (glia), so DAPI keeps
WM populated -- the thing Nissl-as-density got wrong.
"""
from __future__ import annotations

import csv
from functools import cache
from pathlib import Path

CSV = (Path(__file__).resolve().parent / "data"
       / "density_by_region.csv")


@cache
def _load() -> dict:
    """acronym(lower) + str(structure_id) -> {total, neuron_frac, name}."""
    table: dict = {}
    with open(CSV, newline="", encoding="utf-8") as f:
        for r in csv.DictReader(f):
            try:
                tot = float(r["total_density_cells_per_mm3"])
                neu = float(r["neuron_density"])
            except (KeyError, ValueError):
                continue
            if tot <= 0:
                continue
            entry = {
                "total": tot,
                "neuron_frac": max(0.0, min(1.0, neu / tot)),
                "acronym": r["acronym"],
                "name": r["name"],
                "structure_id": int(r["allen_structure_id"]),
            }
            table[r["acronym"].lower()] = entry
            table[str(r["allen_structure_id"])] = entry
    return table


def region(key) -> dict:
    """Look up a region by acronym (e.g. 'FOTUgr') or CCFv3 structure id.

    Returns {total, neuron_frac, acronym, name, structure_id}. Raises KeyError
    with the count of available regions if not found.
    """
    table = _load()
    k = str(key).lower()
    if k not in table:
        raise KeyError(f"region {key!r} not in density table ({len(_load())//2} regions)")
    return table[k]


def density_and_neuron_frac(key) -> tuple[float, float]:
    """Convenience: (total_density_cells_per_mm3, neuron_frac) for a region."""
    e = region(key)
    return e["total"], e["neuron_frac"]


def extremes(n: int = 1) -> tuple[list, list]:
    """(sparsest n, densest n) regions by total density -- for demos/QC."""
    uniq = {e["structure_id"]: e for e in _load().values()}
    rows = sorted(uniq.values(), key=lambda e: e["total"])
    return rows[:n], rows[-n:]
