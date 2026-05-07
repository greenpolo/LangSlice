"""Translate langslice-native trace examples to HF chat-template messages + tools."""

from __future__ import annotations

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

from langslice_harness.atlas.core import get_position_range_mm, load_atlas
from langslice_harness.atlas.space import Plane
from langslice_harness.harness.estimation.prompts import build_single_slice_prompt


@dataclass(frozen=True)
class AtlasMeta:
    pos_lo: float
    pos_hi: float
    species: str  # human-readable, e.g. "mouse" / "rat" / "developmental mouse"


_SPECIES_BY_ATLAS_PREFIX: dict[str, str] = {
    "allen_mouse_": "mouse",
    "whs_sd_rat_": "rat",
    "admba_": "developmental mouse",
}


def _infer_species(atlas_name: str) -> str:
    for prefix, species in _SPECIES_BY_ATLAS_PREFIX.items():
        if atlas_name.startswith(prefix):
            return species
    return "unknown"


class AtlasMetaCache:
    """Memoized (atlas_name, plane) -> AtlasMeta lookup. Avoids reloading volumes."""

    def __init__(self) -> None:
        self._cache: dict[tuple[str, str], AtlasMeta] = {}

    def get(self, atlas_name: str, plane: str) -> AtlasMeta:
        key = (atlas_name, plane)
        if key in self._cache:
            return self._cache[key]
        atlas = load_atlas(atlas_name)
        # `plane` is a keyword-only arg on get_position_range_mm.
        pos_lo, pos_hi = get_position_range_mm(atlas, plane=plane)  # type: ignore[arg-type]
        meta = AtlasMeta(
            pos_lo=float(pos_lo),
            pos_hi=float(pos_hi),
            species=_infer_species(atlas_name),
        )
        self._cache[key] = meta
        return meta


def build_system_prompt(
    *,
    kind: str,
    atlas_name: str,
    plane: str,
    atlas_meta_cache: AtlasMetaCache,
) -> str:
    """Build the system prompt by delegating to the production builders."""
    meta = atlas_meta_cache.get(atlas_name, plane)
    plane_typed: Plane = plane  # type: ignore[assignment]  # Plane is a Literal alias
    if kind == "single_slice":
        return build_single_slice_prompt(
            atlas_name=atlas_name,
            plane=plane_typed,
            pos_lo=meta.pos_lo,
            pos_hi=meta.pos_hi,
            species=meta.species,
        )
    raise ValueError(f"unknown system_prompt_kind: {kind!r}")


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
        return [_FETCH_ATLAS_TOOL, _SUBMIT_ESTIMATE_TOOL]
    raise ValueError(f"unknown system_prompt_kind: {kind!r}")
