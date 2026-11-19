"""The pair SPOmiAlign sees: a section and its atlas plane on one canvas."""

from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np

from langslice.engines import _blocks
from langslice.engines._pixels import slide_colour
from langslice.engines.spomialign import engine
from langslice.engines.spomialign.settings import ATLAS_WINDOW, CANVAS_EDGE, WORK_H, WORK_W
from langslice.plugin_api import (
    FitFrame,
)


def _canvas_scale(shape: tuple[int, int]) -> int:
    """How many times finer than the grid the canvas is."""
    return max(1, round(CANVAS_EDGE / max(shape)))


def _atlas_picture(image: np.ndarray, labels: np.ndarray) -> np.ndarray:
    """The atlas image as SPOmiAlign's published reference looks, a brightfield Nissl
    plate: white background, dense tissue dark (a percentile window inside the
    brain), uint8 RGB."""
    inside = labels > 0
    values = image[inside] if inside.any() else image.ravel()
    lo, hi = np.percentile(values.astype(np.float32), [ATLAS_WINDOW.low, ATLAS_WINDOW.high])
    scaled = np.clip((image.astype(np.float32) - lo) / max(hi - lo, 1e-6), 0.0, 1.0)
    grey = (255 * (1.0 - scaled)).astype(np.uint8)
    return np.repeat(grey[..., None], 3, axis=2)


@dataclass(frozen=True)
class Pair:
    """Hold the pair SPOmiAlign sees on one canvas and how the canvas maps back.

    Parameters
    ----------
    target
        The atlas image, ``(h, w, 3)`` uint8.
    source
        The section, the same shape.
    keep
        False in the damage zone.
    section_keep
        False where the section mask is below one half; None without a mask.
    to_canvas
        Fit-grid px to canvas px (3 x 3).
    from_canvas
        Canvas px to the grid of the lookup (native atlas px, or with a
        deformation the fit grid, read through ``applied``).
    applied
        The deformation as native atlas px at every fit-grid pixel, when the
        canvas is the fit grid.
    um_per_px
        The atlas micrometres one canvas pixel spans.
    work_um
        The atlas micrometres one pixel of SPOmiAlign's working grid spans.
    smoothing_px
        The field's Gaussian sigma, px of the working grid.
    """

    target: np.ndarray
    source: np.ndarray
    keep: np.ndarray
    section_keep: np.ndarray | None
    to_canvas: np.ndarray
    from_canvas: np.ndarray
    applied: np.ndarray | None
    um_per_px: float
    work_um: float
    smoothing_px: float

    @property
    def on_grid(self) -> bool:
        """Say whether the canvas is the fit grid, through the section's deformation."""
        return self.applied is not None

    def engine_pair(self) -> engine.EnginePair:
        """Return the pair as the engine takes it."""
        return engine.EnginePair(
            target=self.target,
            source=self.source,
            keep=self.keep,
            smoothing_px=self.smoothing_px,
            section_keep=self.section_keep,
        )


@dataclass(frozen=True)
class _Canvas:
    """Hold one section's canvas before the pair is made.

    Parameters
    ----------
    plate
        The atlas plate on the canvas's source grid.
    damage
        The damage zone on the same grid.
    source
        The section on the canvas.
    section_keep
        Where the section mask is at least one half; None without a mask.
    scale
        The fit grid's or the atlas plane's px to canvas px (3 x 3).
    to_canvas
        Fit-grid px to canvas px (3 x 3).
    um_per_px
        The atlas micrometres one canvas pixel spans.
    """

    plate: np.ndarray
    damage: np.ndarray
    source: np.ndarray
    section_keep: np.ndarray | None
    scale: np.ndarray
    to_canvas: np.ndarray
    um_per_px: float


def _grid_canvas(
    frame: FitFrame,
    applied: np.ndarray,
    picture: np.ndarray,
    zone: np.ndarray,
    weights: np.ndarray | None,
) -> _Canvas:
    """The canvas on the fit grid, through the section's deformation: the atlas plate and damage
    zone resampled through it and the section resized to it."""
    stain = _blocks.section_pixels(frame)
    gh, gw = applied.shape[:2]
    k = _canvas_scale((gh, gw))
    size = (gw * k, gh * k)
    plate = cv2.remap(
        picture,
        applied[..., 0],
        applied[..., 1],
        cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=(255, 255, 255),
    )
    damage = cv2.remap(
        zone.astype(np.uint8),
        applied[..., 0],
        applied[..., 1],
        cv2.INTER_NEAREST,
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=0,
    )
    section_keep = (
        None
        if weights is None
        else cv2.resize(weights, size, interpolation=cv2.INTER_LINEAR) >= 0.5
    )
    plane_px = float(np.mean(frame.plane["px_um"]))
    placed = np.asarray(frame.placed_atlas_to_section, np.float64)
    scale = engine.resize_map((gw, gh), size)
    return _Canvas(
        plate=plate,
        damage=damage,
        source=cv2.resize(stain, size, interpolation=cv2.INTER_LINEAR),
        section_keep=section_keep,
        scale=scale,
        to_canvas=scale,
        um_per_px=plane_px / float(np.sqrt(abs(np.linalg.det(placed[:2, :2])))) / k,
    )


def _plane_canvas(
    frame: FitFrame,
    picture: np.ndarray,
    zone: np.ndarray,
    weights: np.ndarray | None,
) -> _Canvas:
    """The canvas on the atlas plane's grid, the section resampled onto it through its linear
    placement."""
    stain = _blocks.section_pixels(frame)
    h, w = frame.atlas["annotation"].shape
    k = _canvas_scale((h, w))
    size = (w * k, h * k)
    scale = engine.resize_map((w, h), size)
    start = np.asarray(frame.placed_atlas_to_section, np.float64)
    to_grid = (start @ np.linalg.inv(scale))[:2]
    source = cv2.warpAffine(
        stain,
        to_grid,
        size,
        flags=cv2.INTER_LINEAR | cv2.WARP_INVERSE_MAP,
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=slide_colour(stain),
    )
    section_keep = (
        None
        if weights is None
        else cv2.warpAffine(
            weights,
            to_grid,
            size,
            flags=cv2.INTER_LINEAR | cv2.WARP_INVERSE_MAP,
            borderMode=cv2.BORDER_REPLICATE,
        )
        >= 0.5
    )
    return _Canvas(
        plate=picture,
        damage=zone.astype(np.uint8),
        source=source,
        section_keep=section_keep,
        scale=scale,
        to_canvas=scale @ np.linalg.inv(start),
        um_per_px=float(np.mean(frame.plane["px_um"])) / k,
    )


def pair_inputs(
    frame: FitFrame,
    plane_image: np.ndarray,
    *,
    weights: np.ndarray | None,
    on_grid: bool,
    smoothness_um: float,
) -> Pair:
    """The pair SPOmiAlign sees, on one canvas: over the atlas plane through the linear
    placement, or (``on_grid``) over the fit grid through the section's deformation. *weights*
    (a section mask on the fit grid) becomes ``section_keep`` on the section's canvas, True where
    the weight is at least one half."""
    labels = frame.atlas["annotation"]
    picture = _atlas_picture(plane_image, labels)
    if frame.excluded is not None:
        picture[frame.excluded] = 255
    zone = frame.damage_zone if frame.damage_zone is not None else np.zeros(labels.shape, bool)
    applied = None if frame.applied_map is None else np.asarray(frame.applied_map, np.float32)
    if on_grid and applied is not None:
        canvas = _grid_canvas(frame, applied, picture, zone, weights)
    else:
        applied = None
        canvas = _plane_canvas(frame, picture, zone, weights)
    size = (canvas.source.shape[1], canvas.source.shape[0])
    keep = ~cv2.resize(canvas.damage, size, interpolation=cv2.INTER_NEAREST).astype(bool)
    work_um = canvas.um_per_px * float(np.mean([size[0] / WORK_W, size[1] / WORK_H]))
    return Pair(
        target=cv2.resize(canvas.plate, size, interpolation=cv2.INTER_LINEAR),
        source=canvas.source,
        keep=keep,
        section_keep=canvas.section_keep,
        to_canvas=canvas.to_canvas,
        from_canvas=np.linalg.inv(canvas.scale),
        applied=applied,
        um_per_px=canvas.um_per_px,
        work_um=work_um,
        smoothing_px=smoothness_um / work_um,
    )
