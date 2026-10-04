"""Deterministic label-map helpers for image-gen registration.

Region ids folded to colour families, crisp label borders, the ventricle
id set and line widths. No fit lives here: the fit after the model's border
correction is the deformable package's (:mod:`langslice.core.nonlinear.border_fit`;
the Elastix residual fit and its report were retired on 2026-10-04).
"""

from __future__ import annotations

from typing import Any

import numpy as np

from langslice.core.atlas.core import get_root_mask
from langslice.core.atlas.recolor import color_lut
from langslice.core.atlas.render import annotation_slice
from langslice.core.atlas.render import family_mapping as _family_mapping
from langslice.core.space import Plane

#: Width of an ARA-style leaf delineation line, at a 2048px canvas.
_LEAF_BORDER_PX = 2.0


def line_width_px(long_edge: int) -> int:
    """Delineation-line width for a render whose long edge is *long_edge* px."""
    return max(1, round(_LEAF_BORDER_PX * long_edge / 2048))


#: Ventricle ids, for callers that want the ventricular system dropped to
#: background. The registration lineup does NOT: it asks the model to deform
#: an atlas plate onto the tissue, where a ventricle is a region to place
#: like any other (and a useful landmark for the fit). Blacking them out was
#: for the retired lineup, where the model painted ON the section and a
#: ventricle was a hole with no tissue to paint.
#: "cerebral aqueduct" and "ventricular systems" are spelled out in full:
#: a bare "aqueduct"/"ventric" would swallow periaqueductal gray and the
#: periventricular nuclei, which are real tissue.
_VENTRICLE_KEYWORDS = (
    "ventricle",
    "central canal",
    "choroid",
    "subependymal",
    "cerebral aqueduct",
    "ventricular systems",
)
_BLACKOUT_PLANES = {"coronal"}
_ventricle_ids_cache: dict[str, frozenset[int]] = {}


def _ventricle_ids(atlas: Any) -> frozenset[int]:
    name = str(getattr(atlas, "atlas_name", id(atlas)))
    cached = _ventricle_ids_cache.get(name)
    if cached is not None:
        return cached
    ids: set[int] = set()
    structures = getattr(atlas, "structures", None)
    try:
        records = list(structures.values()) if structures is not None else []
    except Exception:  # noqa: BLE001 - structure table without .values()
        records = []
    for record in records:
        try:
            if any(k in str(record["name"]).lower() for k in _VENTRICLE_KEYWORDS):
                ids.add(int(record["id"]))
        except Exception:  # noqa: BLE001 - malformed structure records
            continue
    result = frozenset(ids)
    _ventricle_ids_cache[name] = result
    return result


def _annotation_slice(
    atlas: Any,
    position_mm: float,
    *,
    plane: Plane = "coronal",
    pitch_deg: float = 0.0,
    yaw_deg: float = 0.0,
    blackout: bool = False,
) -> np.ndarray:
    """:func:`~langslice.core.atlas.render.annotation_slice`, ventricles kept.

    With ``blackout=True`` the ventricular system is dropped to background
    on the planes in :data:`_BLACKOUT_PLANES` before anything downstream
    sees it. The registration path never asks for that: the model deforms
    an atlas plate rather than painting the section, so a ventricle is a
    region to place, and the render, the classifier palette, the Elastix
    side and the ledger all read the same plane.
    """
    sliced = annotation_slice(atlas, position_mm, plane=plane, pitch_deg=pitch_deg, yaw_deg=yaw_deg)
    if blackout and plane in _BLACKOUT_PLANES:
        vids = _ventricle_ids(atlas)
        if vids:
            # np.where, not in-place: the oblique sampler may hand back cached data
            sliced = np.where(np.isin(sliced, list(vids)), 0, sliced)
    return sliced


def _classified_to_rgb(classified_2d: np.ndarray, atlas: Any) -> np.ndarray:
    """Rebuild a clean RGB map from classified ids: exact palette colors on black.

    Elastix registers this cleaned map against the atlas render — the
    generated image's preserved background (white slide, anything) and any
    color drift would otherwise poison the per-channel metric.
    """
    lut = color_lut(atlas)
    rgb = np.zeros((*classified_2d.shape, 3), dtype=np.uint8)
    for uid in np.unique(classified_2d):
        uid_int = int(uid)
        if uid_int == 0:
            continue
        rgb[classified_2d == uid_int] = lut.get(uid_int, (128, 128, 128))
    return rgb


def _merge_classified(classified_2d: np.ndarray, atlas: Any, merge_eps: float = 40.0) -> np.ndarray:
    """Map region ids onto one representative id per merged family color.

    An image model paints one flat shade per area, so the atlas render's thin
    per-layer shade bands have no counterpart in the generated map; register
    them raw and Elastix folds those bands into nothing (measured 10-36%%
    negative-Jacobian area). Registration inputs and border overlays share
    this granularity; classification, markers, and the ledger keep the full
    palette.
    """
    mapping = _family_mapping((int(u) for u in np.unique(classified_2d)), atlas, merge_eps)
    merged = np.zeros_like(classified_2d)
    for uid, rep_id in mapping.items():
        merged[classified_2d == uid] = rep_id
    return merged


def _extract_borders_from_classified(classified_2d: np.ndarray) -> np.ndarray:
    """Single-pixel region borders from classified ids, by neighbor difference.

    Interior borders mark one side of each id change; the outer silhouette
    marks the foreground pixels touching background. No contours, no
    smoothing — each boundary is one crisp line.
    """
    interior = np.zeros(classified_2d.shape, dtype=bool)
    interior[:, 1:] |= classified_2d[:, 1:] != classified_2d[:, :-1]
    interior[1:, :] |= classified_2d[1:, :] != classified_2d[:-1, :]

    fg = classified_2d != 0
    bg_padded = np.pad(classified_2d == 0, 1, constant_values=True)
    touches_bg = (
        bg_padded[2:, 1:-1] | bg_padded[:-2, 1:-1] | bg_padded[1:-1, 2:] | bg_padded[1:-1, :-2]
    )
    borders = ((interior & fg) | (fg & touches_bg)).astype(np.uint8) * 255
    return borders


#: The atlas silhouette mask now lives with the other atlas slice accessors.
#: Kept under its old name because the registration modules — and their
#: monkeypatching tests — import it from here.
_build_atlas_root_mask = get_root_mask

