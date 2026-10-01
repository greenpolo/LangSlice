"""The deformable fit as the linear agent uses it: inputs, cache, records, pictures.

``fit_deformable`` (``toolbox.py``) is a thin tool over this module and the
engine package :mod:`langslice.deformable`. What lives here:

- the fit grid: the section's oriented, unframed render at
  :data:`FIT_LONG_EDGE` with its linear placement
  (:func:`langslice.registration_handoff.prepare_linear_registration`, the
  same handoff ``trace_borders`` uses), and the image the fit reads on it —
  the section's ``fit`` appearance (one raw channel is a fit appearance the
  ``preprocess`` tool sets), or the image model's traced lines mapped onto
  that grid;
- one resolved :class:`Choice` per candidate and its engine settings;
- :class:`RecordStore`: results cached by a digest of every input (so
  applying a previewed candidate reuses it), applied records saved under the
  results folder (``deformable/<section>/<key>/``, the parent steps inside),
  and loaded back for ``start="current"`` and for placement pictures;
- :func:`linear_key` and :func:`clear_stale`: a deformation lives on top of
  the linear placement it was fitted at, and any change to that placement
  drops it;
- the picture: the final borders drawn smoothly on the section
  (:func:`langslice.deformable.draw_warped_borders`), included regions
  highlighted, excluded regions in a second color.
"""

from __future__ import annotations

import hashlib
import importlib.util
import json
import logging
import re
from collections import OrderedDict
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Any

import cv2
import numpy as np
from PIL import Image

from langslice.deformable import (
    CandidateFailure,
    DeformableRecord,
    FitSettings,
    Placement,
    draw_warped_borders,
    finish_fit,
    fit_prepared,
    placement_from_handoff,
    prepare_fit,
    resampled_record,
)
from langslice.deformable.atlas_images import native_intensity, native_labels, normalize_intensity
from langslice.deformable.engines import run_engine
from langslice.deformable.render import composed_native_grid
from langslice.deformable.settings import (
    ANTS_STIFFNESS,
    CORRELATION_RADIUS_UM,
    DETAIL,
    ELASTIX_MEAN_SQUARES_BENDING_SCALE,
    ELASTIX_STIFFNESS,
)
from langslice.linear import appearance as looks
from langslice.linear.state import SliceState, StackState

if TYPE_CHECKING:  # ponytail: import cycle — engine builds the toolbox
    from langslice.linear.engine import EngineContext

logger = logging.getLogger(__name__)

#: Long edge of the section grid a fit runs on and its record lives on. The
#: engine works on a coarser grid (the detail level's micrometres); this grid
#: carries the final field, the labels and the masks.
FIT_LONG_EDGE = 1536
#: The detail level every tool fit runs at. The 2026-10-01 ceiling test found
#: detail never changed which candidate looked best, so the tool fixes it;
#: tests set ``coarse`` here for speed.
DETAIL_LEVEL = "standard"
#: Setting variants one call may preview.
MAX_CANDIDATES = 4
#: Fits one call may run (sections x candidates).
MAX_FITS_PER_CALL = 8
#: Fitted records kept in memory for reuse (a record is tens of megabytes).
CACHE_SIZE = 8
#: The atlas images a fit can use, as the agent names them, and the engine's
#: name for each. ``borders`` is the colour-family boundaries the display
#: options draw (and the image model is shown).
ATLAS_CHOICES: dict[str, str] = {"borders": "borders_merged", "ara": "ara", "nissl": "nissl"}
#: The section images a fit can read.
FIT_LOOK = "fit"
TRACED_BORDERS = "traced_borders"
TRACED_LINES = "traced_lines"
TRACED = (TRACED_BORDERS, TRACED_LINES)
SECTION_IMAGES = (FIT_LOOK, *TRACED)
STARTS = ("linear", "current")
ENGINES = ("ants", "elastix")
#: What a candidate may change.
CANDIDATE_KEYS = ("stiffness", "section_image", "atlas_image", "engine")
#: Picture modes: the fit alone, or the fit then what it started from.
MODES = ("borders", "ab")
#: Flags listed per candidate before the rest are only counted.
MAX_FLAGS = 12
#: Flags about the whole section, listed before any per-region flag.
SECTION_FLAGS = ("DISPLACEMENT_OUTSIZED", "FOLDS")
#: Longest one ``fit_deformable`` call waits for running ``trace_borders``
#: calls its traced section images need (an image call takes ~1-3 minutes).
TRACE_WAIT_S = 300.0
#: Run several fits in a process pool; a single fit runs in this process.
#: Tests switch the pool off.
USE_PROCESS_POOL = True


class FitRefusal(Exception):
    """A refusal with its error code and payload fields."""

    def __init__(self, code: str, message: str, **extra: Any) -> None:
        super().__init__(message)
        self.payload = {"status": "error", "error": code, "message": message, **extra}


def ants_available() -> bool:
    return importlib.util.find_spec("ants") is not None


ANTS_MISSING = (
    "ANTs (antspyx) is not installed on this host: install LangSlice's "
    "'registration' extra (pip install 'langslice[registration]')"
)


def _digest(value: object) -> str:
    return hashlib.sha256(json.dumps(value, sort_keys=True, default=str).encode()).hexdigest()


# --- the linear placement a deformation lives on --------------------------


def linear_key(state: StackState, record: SliceState) -> str:
    """Identity of the section's linear placement: what a warp is fitted on."""
    transform = record.transform or {}
    return _digest({
        "atlas": state.atlas, "plane": state.plane, "position_mm": record.position_mm,
        "angles": [state.pitch_deg, state.yaw_deg], "flip": record.flip,
        "rotation_deg": record.rotation_deg,
        "transform": {key: transform.get(key) for key in ("params", "spline", "calibration")},
    })


def clear_stale(state: StackState) -> list[str]:
    """Drop every deformation whose linear placement changed; return those ids."""
    cleared: list[str] = []
    for record in state.in_order():
        held = record.deformation
        if held and held.get("linear_key") != linear_key(state, record):
            record.deformation = None
            cleared.append(record.id)
    return cleared


# --- the grid and the images a fit reads ----------------------------------


@dataclass
class Grid:
    """One section's fit grid: its linear placement and the default image."""

    record: SliceState
    placement: Placement
    handoff: dict[str, Any]
    image: Image.Image


def fit_grid(state: StackState, ctx: EngineContext, record: SliceState) -> Grid:
    """The section's fit grid (``ValueError`` without a usable linear placement)."""
    from langslice.registration_handoff import prepare_linear_registration

    prepared = prepare_linear_registration(state, ctx, record.id, long_edge=FIT_LONG_EDGE)
    return Grid(record=record, placement=placement_from_handoff(prepared),
                handoff=dict(prepared.metadata), image=prepared.image)


def _on_grid(image: Image.Image, grid: Grid) -> Image.Image:
    return image if image.size == grid.image.size else image.resize(
        grid.image.size, Image.Resampling.BILINEAR)


def stain_image(
    ctx: EngineContext, state: StackState, grid: Grid, section_image: str,
) -> tuple[Image.Image, Any]:
    """``(image, identity)`` of the stain a fit reads: the section's fit appearance.

    Traced section images draw their pictures on it too. A raw channel is a
    fit appearance (``preprocess`` target ``fit``), not a section image.
    """
    if section_image not in SECTION_IMAGES:
        raise ValueError(f"Unknown section image {section_image!r}")
    record = grid.record
    look = looks.section_settings(state, "fit", record.id)
    image = looks.fit_image(ctx, state, record, long_edge=FIT_LONG_EDGE)
    return _on_grid(image, grid), {"fit_look": look}


def traced_lines(
    state: StackState, ctx: EngineContext, grid: Grid, *, running: bool = False,
    waited_s: float = TRACE_WAIT_S,
) -> tuple[np.ndarray, str]:
    """The image model's lines from ``trace_borders`` at THIS placement, on the grid.

    Refused unless the section holds a completed trace whose geometry is the
    current one. *running* means the caller waited *waited_s* seconds for the
    section's trace call and it is still running. Returns the line mask and
    the trace's artifact directory.
    """
    from langslice.registration_tool import correction_fingerprint

    record = grid.record
    held = record.image_correction or {}
    if running:
        raise FitRefusal("TRACE_TIMEOUT", f"{record.id}'s trace_borders call is still "
                         f"running after {waited_s:g} s of waiting; call again later.",
                         id=record.id)
    if held.get("status") == "running":
        raise FitRefusal("TRACE_RUNNING", f"{record.id}'s trace_borders result is recorded "
                         "as running, but no call is running in this session; call "
                         "trace_borders again.", id=record.id)
    if held.get("status") == "error":
        raise FitRefusal("TRACE_FAILED", f"{record.id}'s trace_borders call failed: "
                         + str(held.get("message") or held.get("error") or "no image"),
                         id=record.id)
    if held.get("status") != "ok":
        raise FitRefusal("NO_TRACE", f"{record.id} has no completed trace_borders "
                         "result; traced section images need one.", id=record.id)
    if held.get("geometry_fingerprint") != correction_fingerprint(state, ctx, record.id):
        raise FitRefusal("TRACE_STALE", f"{record.id}'s trace_borders result was made at "
                         "a different placement; trace the current placement first.",
                         id=record.id)
    directory = Path(str(held.get("artifact_dir", "")))
    lines_path = directory / "extracted_lines.png"
    request_path = directory / "request.json"
    if not lines_path.exists() or not request_path.exists():
        raise FitRefusal("NO_TRACE", f"{record.id}'s trace_borders artifacts are missing "
                         f"({directory}).", id=record.id)
    atlas_to_canvas = np.asarray(json.loads(request_path.read_text())["atlas_to_canvas"],
                                 dtype=np.float64)
    with Image.open(lines_path) as opened:
        lines = np.asarray(opened.convert("L"), dtype=np.float32) / 255.0
    section_from_canvas = grid.placement.atlas_to_section @ np.linalg.inv(atlas_to_canvas)
    warped = cv2.warpAffine(lines, section_from_canvas[:2], grid.image.size,
                            flags=cv2.INTER_LINEAR, borderValue=0)
    return warped > 0.05, str(directory)


# --- one candidate's choices ----------------------------------------------


@dataclass(frozen=True)
class Choice:
    """One candidate as resolved: what the agent named, and the engine's settings."""

    section_image: str
    atlas_image: str
    engine: str
    stiffness: str

    def settings(self, include: tuple[str, ...], exclude: tuple[str, ...]) -> FitSettings:
        """The engine settings. A stain fit with ANTs adds the automatic tissue
        and ventricle label channels (``labels="auto"``): the 2026-10-02
        stain ceiling test's clearest gain (outline and enlarged ventricles);
        Elastix has no label channels."""
        traced = self.section_image in TRACED
        if self.section_image == TRACED_BORDERS:
            labels = "model"
        elif not traced and self.engine == "ants":
            labels = "auto"
        else:
            labels = "none"
        return FitSettings(
            engine=self.engine,  # type: ignore[arg-type]
            stiffness=self.stiffness,  # type: ignore[arg-type]
            detail=DETAIL_LEVEL,  # type: ignore[arg-type]
            atlas_image=ATLAS_CHOICES[self.atlas_image],  # type: ignore[arg-type]
            section_image="lines" if traced else "stain",
            labels=labels,  # type: ignore[arg-type]
            exclude=exclude, structures=include,
        )

    def echo(self) -> dict[str, str]:
        return {"engine": self.engine, "stiffness": self.stiffness,
                "section_image": self.section_image, "atlas_image": self.atlas_image}


def engine_settings(settings: FitSettings) -> dict[str, Any]:
    """The engine numbers a setting stands for, in physical units."""
    level = DETAIL[settings.detail]
    metric = settings.metric
    used: dict[str, Any] = {"metric": metric, "working_um": level.working_um}
    if metric == "local_correlation":
        used["correlation_radius_um"] = CORRELATION_RADIUS_UM
    if settings.section_image == "stain" and settings.stain_edges:
        used["edge_channel"] = True
    if settings.engine == "ants":
        stiffness = ANTS_STIFFNESS[settings.stiffness]
        used.update(update_sigma_mm=stiffness.update_sigma_mm,
                    total_sigma_mm=stiffness.total_sigma_mm,
                    iterations=list(level.ants_iterations))
    else:
        stiffness_e = ELASTIX_STIFFNESS[settings.stiffness]
        bending = stiffness_e.bending_weight
        if metric == "mean_squares":
            bending *= ELASTIX_MEAN_SQUARES_BENDING_SCALE
        used.update(grid_spacing_mm=stiffness_e.grid_spacing_mm, bending_weight=bending,
                    resolutions=level.elastix_resolutions,
                    iterations=level.elastix_iterations)
    if settings.labels != "none":
        used["label_map"] = settings.labels
    if settings.structures:
        used["neighbourhood_um"] = settings.neighbourhood_um
    return used


def cache_key(
    state: StackState, grid: Grid, settings: FitSettings, image_identity: Any,
    previous_key: str | None,
) -> str:
    """Every input of one fit: the same key means the same result."""
    return _digest({
        "section": grid.record.id, "linear": linear_key(state, grid.record),
        "grid": [FIT_LONG_EDGE, list(grid.image.size)],
        "placement": grid.placement.to_dict(), "image": image_identity,
        "settings": settings.to_dict(), "start": previous_key or "linear",
    })


# --- records: cache, disk, the reference on the state ---------------------


@dataclass
class RecordStore:
    """Fitted records by input key (memory, most recent first) and on disk."""

    root: Path
    cache: OrderedDict[str, DeformableRecord] = field(default_factory=OrderedDict)

    def directory(self, section_id: str, key: str) -> Path:
        name = re.sub(r"[^a-zA-Z0-9._-]", "_", Path(section_id).name)
        return self.root / name / key[:24]

    def get(self, section_id: str, key: str) -> DeformableRecord | None:
        record = self.cache.get(key)
        if record is None:
            folder = self.directory(section_id, key)
            if (folder / "record.json").exists():
                try:
                    record = DeformableRecord.load(folder)
                except Exception:  # an unreadable record is recomputed, not trusted
                    logger.warning("Could not load %s", folder, exc_info=True)
                    return None
        if record is not None:
            self.put(key, record)
        return record

    def put(self, key: str, record: DeformableRecord) -> None:
        self.cache[key] = record
        self.cache.move_to_end(key)
        while len(self.cache) > CACHE_SIZE:
            self.cache.popitem(last=False)

    def save(self, section_id: str, key: str, record: DeformableRecord) -> Path:
        folder = self.directory(section_id, key)
        if not (folder / "record.json").exists():
            record.save(folder)
        return folder

    def current(self, state: StackState, record: SliceState) -> DeformableRecord | None:
        """The section's applied record, when it still sits on its placement."""
        held = record.deformation
        if (not held or "keep_linear" in held
                or held.get("linear_key") != linear_key(state, record)):
            return None
        return self.get(record.id, str(held.get("key", "")))


def summary(record: DeformableRecord, previous: DeformableRecord | None) -> dict[str, Any]:
    """Compact per-candidate numbers: displacement, folds, flags."""
    tissue = record.tissue
    magnitude = np.linalg.norm(record.field_mm, axis=-1)[tissue]
    result: dict[str, Any] = {
        "displacement_mm": {
            "max": round(float(magnitude.max()), 3) if magnitude.size else 0.0,
            "median": round(float(np.median(magnitude)), 3) if magnitude.size else 0.0,
        },
        "fold_fraction": round(float(record.diagnostics.get("fold_fraction", 0.0)), 5),
    }
    if previous is not None and previous.field_mm.shape == record.field_mm.shape:
        step = np.linalg.norm(record.field_mm - previous.field_mm, axis=-1)[tissue]
        result["step_displacement_mm"] = {
            "max": round(float(step.max()), 3) if step.size else 0.0,
            "median": round(float(np.median(step)), 3) if step.size else 0.0,
        }
    flags: list[dict[str, Any]] = []
    # Whole-section flags first, so the per-region cap never hides them.
    for flag in sorted(record.diagnostics.get("flags", []),
                       key=lambda item: item.get("code") not in SECTION_FLAGS):
        if flag.get("code") == "FOLDS":
            flags.append({"code": "FOLDS", "fold_fraction": round(float(flag["fold_fraction"]), 5)})
            continue
        if flag.get("code") == "DISPLACEMENT_OUTSIZED":
            flags.append({"code": "DISPLACEMENT_OUTSIZED",
                          **{key: round(float(flag[key]), 3) for key in (
                              "max_mm", "median_mm", "max_limit_mm", "median_limit_mm")}})
            continue
        ratio = flag.get("area_ratio")
        flags.append({"code": flag["code"], "region": flag.get("acronym"),
                      "area_ratio": None if ratio is None else round(float(ratio), 2),
                      "limits": flag.get("limits")})
    result["flags"] = flags[:MAX_FLAGS]
    if len(flags) > MAX_FLAGS:
        result["more_flags"] = len(flags) - MAX_FLAGS
    return result


def reference(
    *, folder: Path, key: str, linear: str, record: DeformableRecord, choice: Choice,
    include: tuple[str, ...], exclude: tuple[str, ...], start: str,
    previous: dict[str, Any] | None, numbers: dict[str, Any],
) -> dict[str, Any]:
    """What ``SliceState.deformation`` holds for an applied record."""
    step = {"start": start, **choice.echo(), "include": list(include), "exclude": list(exclude)}
    steps = list((previous or {}).get("steps") or []) if start == "current" else []
    return {
        "record": str(folder), "key": key, "linear_key": linear,
        "steps": [*steps, step], "summary": numbers,
        "inverse_source": record.inverse_source,
    }


def provenance(
    grid: Grid, choice: Choice, include: tuple[str, ...], exclude: tuple[str, ...],
    start: str, image_identity: Any, linear: str,
) -> dict[str, Any]:
    return {
        "tool": "fit_deformable", "section_id": grid.record.id,
        "linear_handoff": grid.handoff, "linear_key": linear,
        "section_grid": {"size": list(grid.image.size), "long_edge": FIT_LONG_EDGE,
                         "frame": "oriented rendered section, unframed; not acquisition "
                                  "pixels (see linear_handoff)"},
        "inputs": {**choice.echo(), "include": list(include), "exclude": list(exclude),
                   "start": start, "image": image_identity},
    }


# --- running fits ----------------------------------------------------------


@dataclass
class Job:
    """One fit to run (or a cached result) for one section and one choice."""

    grid: Grid
    choice: Choice
    settings: FitSettings
    key: str
    image_identity: Any
    previous: DeformableRecord | None
    image: Image.Image
    lines: np.ndarray | None
    result: DeformableRecord | CandidateFailure | None = None
    cached: bool = False


def run_jobs(ctx: EngineContext, jobs: list[Job]) -> None:
    """Prepare and run every job without a result, concurrently when several."""
    prepared = []
    waiting: list[Job] = []
    for job in jobs:
        if job.result is not None:
            continue
        try:
            prepared.append(prepare_fit(
                job.image, ctx.atlas, job.grid.placement, job.settings, lines=job.lines,
                previous=job.previous,
                abba=ctx.abba_atlas if job.choice.atlas_image == "nissl" else None,
            ))
            waiting.append(job)
        except Exception as exc:  # noqa: BLE001 - a bad candidate must not sink the rest
            job.result = CandidateFailure(job.settings, f"{type(exc).__name__}: {exc}")
    if not prepared:
        return
    if USE_PROCESS_POOL and len(prepared) > 1:
        results = fit_prepared(prepared)
    else:
        results = []
        for item in prepared:
            try:
                results.append(finish_fit(item, run_engine(item.inputs, item.settings)))
            except Exception as exc:  # noqa: BLE001
                results.append(CandidateFailure(item.settings, f"{type(exc).__name__}: {exc}"))
    for job, result in zip(waiting, results, strict=True):
        job.result = result


# --- the picture ------------------------------------------------------------


@dataclass(frozen=True)
class Style:
    """How one call's pictures are drawn (its display options)."""

    #: Zoom fractions; empty is the whole section.
    zoom: tuple[float, ...]
    #: Regions drawn strong; empty draws every border strong.
    highlight: tuple[str, ...]
    #: Regions drawn in the second color (those excluded from the fit).
    marked: tuple[str, ...]
    outlines: str
    color: tuple[int, int, int]
    thickness: float
    atlas_opacity: float
    #: Long edge of the picture (of the zoomed crop when zoomed), in pixels;
    #: never more than the fit image's own pixels.
    long_edge: int = 512


def picture(
    ctx: EngineContext,
    image: Image.Image,
    record: DeformableRecord,
    *,
    warped: bool,
    style: Style,
    atlas_image: str,
    title: str,
) -> Image.Image:
    """The record's borders on the section at ``style.long_edge``, captioned.

    The crop (the zoom, or the whole fit image) is shown at ``style.long_edge``
    on its long side, or at its own pixels when it has fewer: the fit image
    (:data:`FIT_LONG_EDGE`) is never upsampled.
    """
    from langslice.linear.render import caption, zoom_box

    width, height = record.section_size
    zoom = style.zoom
    box = zoom_box(list(zoom), (width, height)) if zoom else (0, 0, width, height)
    crop = (box[2] - box[0], box[3] - box[1])
    factor = min(1.0, int(style.long_edge) / float(max(crop)))
    full = (max(8, int(round(width * factor))), max(8, int(round(height * factor))))
    shown = (0, 0, full[0], full[1])
    if zoom:
        shown = zoom_box(list(zoom), full)
    small = resampled_record(record, full, shown)
    base = image.convert("RGB").resize(full, Image.Resampling.LANCZOS).crop(shown)
    if style.atlas_opacity > 0 and atlas_image in ("ara", "nissl"):
        base = _blend_atlas(ctx, base, small if warped else _unwarped(small), atlas_image,
                            style.atlas_opacity)
    drawn = draw_warped_borders(
        base, small, ctx.atlas, highlight=style.highlight, marked=style.marked, warped=warped,
        outlines=style.outlines, width_px=style.thickness, color=style.color,
        native=native_labels(ctx.atlas, record.placement),
    )
    return caption(drawn, title)


def trace_picture(
    image: Image.Image, lines: np.ndarray, *, style: Style, title: str,
) -> Image.Image:
    """The image model's traced lines on the section image, at ``style.long_edge``.

    *image* and *lines* share the fit grid (:func:`traced_lines`); the zoom,
    border color and thickness are the call's, and the picture is never drawn
    past the fit image's own pixels, like :func:`picture`.
    """
    from langslice.linear.render import caption, zoom_box

    width, height = image.size
    box = zoom_box(list(style.zoom), (width, height)) if style.zoom else (0, 0, width, height)
    crop = (box[2] - box[0], box[3] - box[1])
    factor = min(1.0, int(style.long_edge) / float(max(crop)))
    shown = (max(8, int(round(crop[0] * factor))), max(8, int(round(crop[1] * factor))))
    base = image.convert("RGB").crop(box).resize(shown, Image.Resampling.LANCZOS)
    mask = np.asarray(lines, dtype=np.float32)[box[1]:box[3], box[0]:box[2]]
    # Area-averaged so a one-pixel line survives the shrink.
    drawn = cv2.resize(mask, shown, interpolation=cv2.INTER_AREA) > 0.02
    radius = int(round((float(style.thickness) - 1.0) / 2.0))
    if radius > 0:
        drawn = cv2.dilate(drawn.astype(np.uint8), np.ones((2 * radius + 1,) * 2, np.uint8)) > 0
    pixels = np.array(base)
    pixels[drawn] = style.color
    return caption(Image.fromarray(pixels), title)


def _unwarped(record: DeformableRecord) -> DeformableRecord:
    from dataclasses import replace

    return replace(record, field_mm=np.zeros_like(record.field_mm))


def _blend_atlas(
    ctx: EngineContext, base: Image.Image, record: DeformableRecord, kind: str, opacity: float,
) -> Image.Image:
    """The atlas image pulled through the record's map, blended under the lines."""
    native = native_intensity(ctx.atlas, record.placement, kind,
                              abba=ctx.abba_atlas if kind == "nissl" else None)
    labels = native_labels(ctx.atlas, record.placement)
    native = normalize_intensity(native, labels > 0)
    nx, ny = composed_native_grid(record, 1)
    gray = cv2.remap(native.astype(np.float32), nx, ny, cv2.INTER_LINEAR,
                     borderMode=cv2.BORDER_CONSTANT, borderValue=0)
    pixels = np.asarray(base, dtype=np.float32)
    mixed = pixels * (1.0 - opacity) + (gray[..., None] * 255.0) * opacity
    return Image.fromarray(np.rint(np.clip(mixed, 0, 255)).astype(np.uint8))
