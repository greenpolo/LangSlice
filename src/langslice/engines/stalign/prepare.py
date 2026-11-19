"""Preparing a section for STalign: its target image, the atlas image and the call's facts."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import cv2
import numpy as np

from langslice.engines import _blocks
from langslice.engines._pixels import (
    area_average,
    placed_matrix,
    placed_outline,
    sample_pixels,
    slide_level,
)
from langslice.engines._sorting import SORTING_KEYS, Sorting
from langslice.engines.stalign import engine
from langslice.engines.stalign.settings import (
    BRIGHTNESS,
    MIN_SCALE,
    RESOLUTIONS,
    SAME_LEVEL,
    SORTING_METHODS,
    SWAP,
)
from langslice.plugin_api import (
    FitFrame,
    RefusalCode,
    ToolContext,
    ToolError,
)


@dataclass(frozen=True)
class _Target:
    """Hold the section image STalign fits to.

    Parameters
    ----------
    image
        Slide 0, tissue 1, ``(1, h, w)`` at about ``target_um``.
    rows_um, cols_um
        Its pixel centres, um (atlas um at the section's linear scale).
    um_per_px
        Micrometres per fit-grid pixel at the section's linear scale.
    size
        The fit grid, ``(width, height)``.
    """

    image: np.ndarray
    rows_um: np.ndarray
    cols_um: np.ndarray
    um_per_px: float
    size: tuple[int, int]


@dataclass(frozen=True)
class _Source:
    """Hold the atlas image STalign bends.

    Parameters
    ----------
    image
        The atlas image and its squared deviation, ``(2, h, w)``, at about
        ``source_um``.
    rows_um, cols_um
        Its pixel centres, um.
    start
        The start affine (3 x 3), source um to target um, row/col order.
    applied
        The section's deformation as a map from fit-grid pixels to native
        atlas pixels, when it holds one (the image is then on the fit grid).
    atlas_px_um
        The native atlas plane's pixel, ``(x, y)`` um.
    """

    image: np.ndarray
    rows_um: np.ndarray
    cols_um: np.ndarray
    start: np.ndarray
    applied: np.ndarray | None
    atlas_px_um: tuple[float, float]

    @property
    def on_grid(self) -> bool:
        """Say whether the image is on the fit grid, through a deformation."""
        return self.applied is not None


def _axis(n: int, scale: float, um: float) -> np.ndarray:
    """Return the centres of *n* down pixels of *scale* original pixels of *um*."""
    return ((np.arange(n) + 0.5) * scale - 0.5) * um


def _scale_um(frame: FitFrame) -> float:
    """Micrometres per fit-grid pixel at the section's linear scale (atlas micrometres)."""
    px = float(np.mean(frame.plane["px_um"]))
    return px / float(np.sqrt(abs(np.linalg.det(placed_matrix(frame)[:2, :2]))))


def section_target(frame: FitFrame) -> _Target:
    """The section's image, slide 0 and tissue 1, at about ``target_um``, and its frame."""
    grey = _blocks.section_pixels(frame).astype(np.float64).mean(axis=2)
    gh, gw = grey.shape
    slide = slide_level(grey)
    outline = placed_outline(frame, (gw, gh))
    tissue = float(np.median(grey[outline])) if outline.any() else float(grey.mean())
    if abs(tissue - slide) < SAME_LEVEL:
        raise ToolError(RefusalCode.FIT_FAILED, "the section's tissue and slide look the same")
    unit = (grey - slide) / (tissue - slide)
    um = _scale_um(frame)
    down, (sx, sy) = area_average(unit.astype(np.float32), RESOLUTIONS.target_um / um)
    return _Target(
        image=down.astype(np.float64)[None],
        rows_um=_axis(down.shape[0], sy, um),
        cols_um=_axis(down.shape[1], sx, um),
        um_per_px=um,
        size=(gw, gh),
    )


def _source_image(image: np.ndarray) -> np.ndarray:
    """The atlas image over its mean, with its squared deviation (the published recipe)."""
    image = image[None] / max(float(np.mean(np.abs(image))), MIN_SCALE)
    return np.concatenate((image, (image - image.mean()) ** 2))


def _source(frame: FitFrame, plane_image: np.ndarray, um: float) -> _Source:
    """The atlas image (damage painted as background) at about ``source_um``, its axes and
    the start affine (source um -> target um, row/col)."""
    atlas = np.asarray(plane_image, np.float32).copy()
    if frame.damage_zone is not None:
        atlas[frame.damage_zone] = 0.0
    px_x, px_y = (float(v) for v in frame.plane["px_um"])
    if frame.applied_map is not None:
        applied = np.asarray(frame.applied_map, np.float32)
        image = cv2.remap(
            atlas,
            applied[..., 0],
            applied[..., 1],
            cv2.INTER_LINEAR,
            borderMode=cv2.BORDER_CONSTANT,
            borderValue=0.0,
        )
        down, (sx, sy) = area_average(image, RESOLUTIONS.source_um / um)
        return _Source(
            image=_source_image(down.astype(np.float64)),
            rows_um=_axis(down.shape[0], sy, um),
            cols_um=_axis(down.shape[1], sx, um),
            start=np.eye(3),
            applied=applied,
            atlas_px_um=(px_x, px_y),
        )
    down, (sx, sy) = area_average(atlas, RESOLUTIONS.source_um / min(px_x, px_y))
    a_xy = np.diag([um, um, 1.0]) @ placed_matrix(frame) @ np.diag([1 / px_x, 1 / px_y, 1.0])
    return _Source(
        image=_source_image(down.astype(np.float64)),
        rows_um=_axis(down.shape[0], sy, px_y),
        cols_um=_axis(down.shape[1], sx, px_x),
        start=SWAP @ a_xy @ SWAP,
        applied=None,
        atlas_px_um=(px_x, px_y),
    )


def grid_query(size: tuple[int, int], um: float) -> np.ndarray:
    """Every fit-grid pixel as (row um, col um), in raster order."""
    gw, gh = size
    yy, xx = np.indices((gh, gw), dtype=np.float64)
    return np.stack([yy.ravel() * um, xx.ravel() * um], -1)


def native_map(source_um: np.ndarray, source: _Source, target: _Target) -> np.ndarray:
    """STalign's source points of every fit-grid pixel as native atlas pixels (x, y)."""
    gw, gh = target.size
    um = target.um_per_px
    pts = source_um.reshape(gh, gw, 2)
    if source.applied is not None:
        grid = np.stack([pts[..., 1] / um, pts[..., 0] / um], -1)
        return sample_pixels(source.applied, grid).astype(np.float64)
    px_x, px_y = source.atlas_px_um
    return np.stack([pts[..., 1] / px_x, pts[..., 0] / px_y], -1)


def sorting_settings(sorting: Mapping[str, Any]) -> tuple[Sorting, str | None]:
    """Read the tool's ``sorting`` argument as STalign's spreads and mixture classes."""
    given = dict(sorting)
    method = str(given.pop("method", "classic"))
    if method not in SORTING_METHODS:
        raise ToolError(
            RefusalCode.BAD_ARGUMENT,
            f"sorting method is one of {', '.join(SORTING_METHODS)}; got {method!r}",
        )
    artifact = bool(given.pop("artifact", False))
    unknown = [key for key in given if key not in SORTING_KEYS]
    if unknown:
        raise ToolError(
            RefusalCode.BAD_ARGUMENT,
            f"unknown sorting keys {unknown}; the keys are method, "
            f"{', '.join(SORTING_KEYS)}, artifact",
        )
    spreads = Sorting.from_given(given)
    classes = None
    if method == "em":
        classes = "MAB" if artifact else "MB"
    elif method == "none":
        classes = "M"
    return spreads, classes


def section_frame(
    ctx: ToolContext, name: str, atlas_image: str, section_image: str
) -> tuple[FitFrame, np.ndarray, str]:
    """The section's fit frame, the atlas image asked for on it and the name that resolved to."""
    frame: FitFrame = _blocks.frame_for(ctx, name, atlas_image, section_image)
    plane_image, resolved = _blocks.atlas_plane(frame, atlas_image)
    return frame, plane_image, resolved


@dataclass(frozen=True)
class LddmmCall:
    """Hold what every section of one ``stalign_lddmm`` call shares.

    Parameters
    ----------
    settings
        The call's settings, as the reply and the deformation record them.
    fit
        STalign's keyword arguments.
    machine
        Where STalign runs.
    atlas_image, section_image
        The images asked for.
    folder
        The call's folder in the job.
    view
        Whether pictures are sent.
    sorts
        Whether STalign's final sorting is saved with the deformation.
    """

    settings: Mapping[str, Any]
    fit: engine.LddmmSettings
    machine: engine.Machine
    atlas_image: str
    section_image: str
    folder: Path
    view: bool
    sorts: bool


@dataclass(frozen=True)
class Prepared:
    """Hold one section ready to fit.

    Parameters
    ----------
    sid
        The section's filename.
    frame
        Its fit frame.
    resolved
        The atlas image that was used.
    inputs
        The ``inputs`` block of its reply.
    digest
        Its registration's digest when the call began.
    target, source
        The two images.
    """

    sid: str
    frame: FitFrame
    resolved: str
    inputs: dict[str, Any]
    digest: str
    target: _Target
    source: _Source


def prepare_section(ctx: ToolContext, call: LddmmCall, name: str) -> Prepared | dict[str, object]:
    """Read one section's frame and images, or the row that says why it cannot be fitted."""
    section = ctx.reads.section(name)
    sid = section.id
    if call.fit.refit_transform and section.deformation:
        refusal = ToolError(
            RefusalCode.PRECONDITION,
            'fit "both" would refit the transform, which '
            'clears the section\'s deformation; fit "warp" to bend on top of it',
        )
        return {"id": sid, **refusal.payload()}
    frame, plane_image, resolved = section_frame(ctx, sid, call.atlas_image, call.section_image)
    if frame.placed_atlas_to_section is None:
        refusal = ToolError(RefusalCode.PRECONDITION, "the section needs a transform first")
        return {"id": sid, **refusal.payload()}
    target = section_target(frame)
    return Prepared(
        sid=sid,
        frame=frame,
        resolved=resolved,
        inputs=_blocks.inputs_block(
            ctx,
            frame,
            section_image=call.section_image,
            atlas_image=call.atlas_image,
            resolved=resolved,
            mask=None,
            brightness=BRIGHTNESS,
        ),
        digest=ctx.reads.section_inputs(sid, deformation=True),
        target=target,
        source=_source(frame, plane_image, target.um_per_px),
    )
