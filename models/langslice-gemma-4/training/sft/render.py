"""Translate langslice-native trace examples to HF chat-template messages + tools."""

from __future__ import annotations

import copy
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

# langslice_harness lives at <repo>/src — make it importable when this module
# is run from inside the gemma-4 training directory (e.g. `python -m sft.train_sft`).
# pytest already injects `src` via `pyproject.toml::tool.pytest.ini_options.pythonpath`,
# so this is defensive for direct script invocation.
_REPO_SRC = Path(__file__).resolve().parents[4] / "src"
if str(_REPO_SRC) not in sys.path:
    sys.path.insert(0, str(_REPO_SRC))

from langslice_harness.atlas.core import (
    get_position_range_mm,
    load_atlas,
    species_from_atlas_name,
)
from langslice_harness.atlas.space import Plane
from langslice_harness.harness.estimation.prompts import build_single_slice_prompt


@dataclass(frozen=True)
class AtlasMeta:
    pos_lo: float
    pos_hi: float
    species: str  # human-readable, e.g. "mouse" / "rat" / "developmental mouse"


class AtlasMetaCache:
    """Memoized (atlas_name, plane) -> AtlasMeta lookup.

    Construct ONE instance per training run and pass it to every consumer
    (dataset, renderer, eval callbacks). Avoid constructing fresh instances
    inside hot paths — `load_atlas` itself is lru-cached but redundant
    instantiation defeats the per-cache invariant other code may rely on.
    """

    def __init__(self) -> None:
        self._cache: dict[tuple[str, Plane], AtlasMeta] = {}

    def get(self, atlas_name: str, plane: Plane) -> AtlasMeta:
        key = (atlas_name, plane)
        if key in self._cache:
            return self._cache[key]
        atlas = load_atlas(atlas_name)
        pos_lo, pos_hi = get_position_range_mm(atlas, plane=plane)
        species = str(atlas.metadata.get("species", "")) if hasattr(atlas, "metadata") else ""
        if not species:
            species = species_from_atlas_name(atlas_name)
        meta = AtlasMeta(
            pos_lo=float(pos_lo),
            pos_hi=float(pos_hi),
            species=species,
        )
        self._cache[key] = meta
        return meta


def build_system_prompt(
    *,
    kind: str,
    atlas_name: str,
    plane: Plane,
    atlas_meta_cache: AtlasMetaCache,
) -> str:
    """Build the system prompt by delegating to the production builders."""
    if kind != "single_slice":
        raise ValueError(f"unknown system_prompt_kind: {kind!r}")
    meta = atlas_meta_cache.get(atlas_name, plane)
    return build_single_slice_prompt(
        atlas_name=atlas_name,
        plane=plane,
        pos_lo=meta.pos_lo,
        pos_hi=meta.pos_hi,
        species=meta.species,
    )


_FETCH_ATLAS_TOOL: dict[str, Any] = {
    "type": "function",
    "function": {
        "name": "fetch_atlas",
        "description": (
            "Fetch atlas reference slices at the given positions (in mm). "
            "Returns up to 8 atlas images per call."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "positions_mm": {
                    "type": "array",
                    "items": {"type": "number"},
                    "description": "Positions in mm at which to render atlas slices.",
                }
            },
            "required": ["positions_mm"],
        },
    },
}

_SUBMIT_ESTIMATE_TOOL: dict[str, Any] = {
    "type": "function",
    "function": {
        "name": "submit_estimate",
        "description": "Submit the final position estimate for the query slice.",
        "parameters": {
            "type": "object",
            "properties": {
                "position_mm": {"type": "number"},
                "reasoning": {"type": "string"},
            },
            "required": ["position_mm", "reasoning"],
        },
    },
}


def build_tools_schema(kind: str) -> list[dict[str, Any]]:
    """Return the HF-format function-schema list for the given kind."""
    if kind == "single_slice":
        return [copy.deepcopy(_FETCH_ATLAS_TOOL), copy.deepcopy(_SUBMIT_ESTIMATE_TOOL)]
    raise ValueError(f"unknown system_prompt_kind: {kind!r}")
