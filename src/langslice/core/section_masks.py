"""Section masks: where on a section a fit compares, saved per section.

A mask is a float16 weight image ``(H, W)`` in [0, 1] on the section's fit
grid (the oriented :func:`langslice.core.handoff.prepare_linear_registration`
frame at :data:`~langslice.core.deformation.FIT_LONG_EDGE`). Two kinds, both
the three-class EM sorting of the section into tissue that matches the atlas
image, artifact and background (:func:`langslice.core.deformable.masks.em_weights`):

- ``em``: made by ``set_mask`` at the section's current registration
  (:func:`em_sorting`);
- ``fit``: copied from the applied deformation record's saved sorting
  (:func:`fit_weights`).

Nothing else: no threshold and no drawn mask. A mask is section-side; the
damage marks are atlas-side, always applied, and never mask the section.

A mask is a state entry (``SliceState.masks[name]``) plus a file
``sections/<stem>/masks/<name>.npz`` under the job folder holding ``weights``
(float16), ``grid`` (``[W, H]``) and ``made`` (ISO time). Files are never
deleted: undo hides the state reference, and a mask made again under a name
that exists gets a new file (``<name>.2.npz``) so the undone state still
finds its own. An entry records the digest of the section's registration it
was made at (:func:`registration_digest`); the mask is *stale* once that
differs.
"""

from __future__ import annotations

import hashlib
import json
import re
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Protocol

import cv2
import numpy as np

from langslice.core import appearance as looks
from langslice.core.atlas.images import resolve_image
from langslice.core.atlas.size import brain_scale
from langslice.core.deformable.geometry import WorkingGrid, sample_native
from langslice.core.deformable.masks import (
    EM_MATCH_THRESHOLD,
    MASK_MARGIN_MM,
    SORTING_KEYS,
    dilate,
    em_weights,
)
from langslice.core.deformable.plane_images import (
    native_intensity,
    native_labels,
    normalize_intensity,
)
from langslice.core.deformable.render import composed_native_grid
from langslice.core.deformation import RecordStore, fit_grid, linear_key
from langslice.core.state import SliceState, StackState
from langslice.core.workspace import Workspace

#: What a mask's name may be made of; ``none`` means no mask and is reserved.
NAME_PATTERN = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_-]{0,39}$")
RESERVED_NAMES = ("none",)
KINDS = ("em", "fit")
#: The working resolution of the EM sorting, adult mouse micrometres (times
#: the brain scale): the ``standard`` fit's.
WORKING_UM = 20.0


class _Layout(Protocol):
    """The part of the job's layout a mask file needs (``job.layout.JobLayout``)."""

    folder: Path

    def masks_dir(self, section_id: str) -> Path: ...

    def relative(self, path: str | Path) -> str: ...

    def resolve(self, ref: str | Path) -> Path: ...


@dataclass(frozen=True, eq=False)
class MaskView:
    """A mask as a fit reads it: *weights* float32 on the fit grid, *stale*
    when the section's registration changed since it was made."""

    name: str
    kind: str
    weights: np.ndarray
    stale: bool
    params: dict[str, Any]
    left_out: float


# --- the registration a mask was made at -----------------------------------------------


def registration_parts(state: StackState, record: SliceState) -> dict[str, Any]:
    """What ``ops.inputs.section_inputs`` and :func:`registration_digest` both
    digest: the linear placement, the fit appearance and the damage marks."""
    return {
        "linear": linear_key(state, record),
        "fit_look": looks.preprocessed_settings(state, record.id),
        "damage": {"regions": list(record.damaged_regions)},
    }


def registration_digest(state: StackState, record: SliceState) -> str:
    """A digest of what a mask of *record* depends on: its linear placement,
    its fit appearance, its damage marks and its applied deformation. The same
    content as ``ops.inputs.section_inputs(..., deformation=True)``, which a
    mask stores at creation (core cannot import ops)."""
    held = {**registration_parts(state, record),
            "deformation": (record.deformation or {}).get("key")}
    text = json.dumps(held, sort_keys=True, default=str)
    return hashlib.sha256(text.encode()).hexdigest()


def is_stale(state: StackState, record: SliceState, entry: dict[str, Any]) -> bool:
    """Whether the registration *entry* was made at differs from the current one."""
    return str(entry.get("inputs", "")) != registration_digest(state, record)


def mask_rows(state: StackState, record: SliceState) -> list[dict[str, Any]]:
    """``[{name, kind, stale}]`` of *record*'s masks (the status rows)."""
    return [{"name": name, "kind": str(entry.get("kind")),
             "stale": is_stale(state, record, entry)} for name, entry in record.masks.items()]


# --- making a sorting ------------------------------------------------------------------


def _gray(image: np.ndarray) -> np.ndarray:
    """*image* as float32 brightness in [0, 1]: uint8 scaled, colour averaged."""
    array = np.asarray(image)
    scaled = array.astype(np.float32) / (255.0 if array.dtype == np.uint8 else 1.0)
    if scaled.ndim == 3:
        scaled = scaled[..., :3].mean(axis=-1)
    return np.clip(scaled, 0.0, 1.0).astype(np.float32)


def em_sorting(
    ws: Workspace, state: StackState, record: SliceState, store: RecordStore, *,
    channel_image: np.ndarray, atlas_image: str, params: dict[str, float],
) -> tuple[np.ndarray, dict[str, Any]]:
    """The EM sorting of *record*'s section against *atlas_image* at its CURRENT
    registration: ``(WM, facts)``, WM the float32 P(match) on the fit grid and
    *facts* ``{"params", "left_out"}``.

    *channel_image* is the channel being sorted on the fit grid (brightness
    0-1, or uint8; colour is averaged). The atlas image is placed through the
    section's linear placement and, when it holds one, its applied
    deformation (``store.current``); the footprint is the placed outline
    widened by :data:`~langslice.core.deformable.masks.MASK_MARGIN_MM`. The
    EM runs on the 20 um working grid and its P(match) is upsampled. *params*
    may carry ``match_sd``, ``background_sd``, ``artifact_sd`` (class widths)
    and ``background_mean``, ``artifact_mean`` (brightness 0-1), each
    optional; the values used are in ``facts["params"]`` under the same keys.
    ``left_out`` is the fraction of the footprint with P(match) below 0.5.
    ``ValueError`` for an unknown key, a bad atlas image or no placement.
    """
    unknown = sorted(set(params) - set(SORTING_KEYS))
    if unknown:
        raise ValueError(f"Unknown sorting key(s) {unknown}; the keys are {list(SORTING_KEYS)}")
    grid = fit_grid(state, ws, record)
    width, height = grid.image.size
    placement = grid.placement
    scale = brain_scale(ws.atlas)
    resolved = resolve_image(ws.atlas, atlas_image)

    section = _gray(channel_image)
    if section.shape != (height, width):
        section = cv2.resize(section, (width, height), interpolation=cv2.INTER_AREA)

    labels = native_labels(ws.atlas, placement)
    native = normalize_intensity(native_intensity(ws.atlas, placement, resolved), labels > 0)
    held = store.current(state, record)
    if held is not None and tuple(held.section_size) == (width, height):
        nx, ny = composed_native_grid(held, 1)
        coordinates = np.stack([nx, ny], axis=-1).astype(np.float64)
    else:
        yy, xx = np.indices((height, width), dtype=np.float64)
        inverse = np.linalg.inv(placement.atlas_to_section)
        coordinates = np.stack([inverse[k, 0] * xx + inverse[k, 1] * yy + inverse[k, 2]
                                for k in range(2)], axis=-1)
    placed_image = sample_native(native.astype(np.float32), coordinates, nearest=False)
    placed_labels = sample_native(labels, coordinates)

    mm = placement.section_mm_per_px
    footprint = dilate(placed_labels != 0, MASK_MARGIN_MM * scale / mm)
    working = WorkingGrid.for_section((width, height), mm, WORKING_UM * scale / 1000.0)
    spacing = float(np.mean(working.spacing_mm))
    weights = em_weights(
        working.to_working(section), working.to_working(placed_image),
        working.to_working(footprint.astype(np.uint8), nearest=True) > 0,
        spacing, scale, dict(params) or None)
    match = cv2.resize(weights["WM"], (width, height), interpolation=cv2.INTER_LINEAR)
    match = np.clip(match, 0.0, 1.0).astype(np.float32)
    used = weights["params"]
    left_out = (float((match[footprint] < EM_MATCH_THRESHOLD).mean())
                if footprint.any() else 0.0)
    facts = {
        "params": {"match_sd": float(used["sigmaM"]), "background_sd": float(used["sigmaB"]),
                   "artifact_sd": float(used["sigmaA"]),
                   "background_mean": float(used["muB"]), "artifact_mean": float(used["muA"])},
        "left_out": left_out,
    }
    return match, facts


def fit_weights(store: RecordStore, state: StackState, record: SliceState) -> np.ndarray | None:
    """The applied deformation record's saved sorting as float32 on the fit
    grid, or None when the section holds no deformation or its record saved no
    sorting (``set_mask`` source ``fit``: ``NO_DEFORMATION`` / ``NO_SORTING``)."""
    held = store.current(state, record)
    if held is None or held.weights is None:
        return None
    return np.asarray(held.weights, dtype=np.float32)


# --- files and entries -----------------------------------------------------------------


def _file_for(layout: _Layout, record: SliceState, name: str) -> Path:
    folder = layout.masks_dir(record.id)
    path = folder / f"{name}.npz"
    version = 1
    while path.exists():
        version += 1
        path = folder / f"{name}.{version}.npz"
    return path


def write_mask(
    layout: _Layout, record: SliceState, name: str, weights: np.ndarray, meta: dict[str, Any],
) -> dict[str, Any]:
    """Save *weights* as *record*'s mask *name* and return its state entry
    (also set as ``record.masks[name]``). *meta* carries ``kind`` (``em`` or
    ``fit``), ``channel``, ``atlas_image`` (resolved), ``params``, ``inputs``
    (:func:`registration_digest` at creation) and ``left_out``. ``ValueError``
    for a bad name, kind or weights."""
    if not NAME_PATTERN.match(name) or name in RESERVED_NAMES:
        raise ValueError(f"A mask name is letters, digits, - and _ (up to 40, not "
                         f"{', '.join(RESERVED_NAMES)}); got {name!r}")
    kind = str(meta.get("kind"))
    if kind not in KINDS:
        raise ValueError(f"A mask kind is one of {list(KINDS)}; got {kind!r}")
    array = np.asarray(weights)
    if array.ndim != 2 or not np.isfinite(array).all():
        raise ValueError("A mask is a finite (H, W) weight image")
    stored = np.clip(array.astype(np.float32), 0.0, 1.0).astype(np.float16)
    path = _file_for(layout, record, name)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f"{path.stem}.partial.npz")
    try:
        np.savez_compressed(temporary, weights=stored,
                            grid=np.array([stored.shape[1], stored.shape[0]]),
                            made=np.array(datetime.now(UTC).isoformat()))
        temporary.replace(path)
    finally:
        temporary.unlink(missing_ok=True)
    entry = {
        "kind": kind, "channel": str(meta.get("channel", "")),
        "atlas_image": str(meta.get("atlas_image", "")),
        "params": dict(meta.get("params") or {}), "inputs": str(meta.get("inputs", "")),
        "file": layout.relative(path), "left_out": float(meta.get("left_out", 0.0)),
    }
    record.masks[name] = entry
    return entry


def read_mask(layout: _Layout, state: StackState, record: SliceState, name: str) -> MaskView:
    """The mask *name* of *record*. ``KeyError`` (the op's ``UNKNOWN_MASK``) when
    the section has none by that name; ``ValueError`` when its file cannot be
    read."""
    entry = record.masks[name]
    path = layout.resolve(entry["file"])
    try:
        with np.load(path) as data:
            weights = np.asarray(data["weights"], dtype=np.float32)
    except (OSError, KeyError, ValueError) as exc:
        raise ValueError(f"{record.id}'s mask {name!r} cannot be read ({path}): {exc}") from exc
    return MaskView(name=name, kind=str(entry["kind"]), weights=weights,
                    stale=is_stale(state, record, entry), params=dict(entry.get("params") or {}),
                    left_out=float(entry.get("left_out", 0.0)))


def section_weights(mask: MaskView | None) -> np.ndarray | None:
    """The weights a fit takes (``prepare_fit(section_weights=)``): the mask's,
    or None for no mask."""
    return None if mask is None else mask.weights

