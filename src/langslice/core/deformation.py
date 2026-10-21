"""The deformable fit as the tools use it: inputs, cache, records, pictures.

``fireants_syn`` and ``trace_borders``' landing (through
:func:`langslice.ops.deformable.fit_deformable`) are thin verbs over this
module and the engine package :mod:`langslice.core.deformable`. What lives
here:

- the fit grid: the section's oriented render at
  :data:`FIT_LONG_EDGE` with its linear placement
  (:func:`langslice.core.handoff.prepare_linear_registration`, the
  same handoff ``trace_borders`` uses), and the image the fit reads on it —
  the section's channel (:mod:`langslice.core.appearance`, a raw channel or a
  derived one such as ``preprocessed``, set by ``set_channel``), or the image
  model's traced lines mapped onto that grid;
- the resolved :class:`Choice` and its engine settings;
- :class:`RecordStore`: results cached by a digest of every input (the same
  fit again reuses it), applied records saved under the section's folder
  (``deformable/<key>/``, each step's parent a sibling folder it names), and loaded back to fit on
  top of and for placement pictures;
- :func:`linear_key` and :func:`clear_stale`: a deformation lives on top of
  the linear placement it was fitted at, and any change to that placement
  drops it;
- the picture: the final borders drawn smoothly on the section
  (:func:`langslice.core.deformable.draw_warped_borders`), included regions
  highlighted, excluded regions in a second color.
"""

from __future__ import annotations

import importlib.util
import json
import logging
import re
import threading
from collections import OrderedDict
from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import cv2
import numpy as np
from PIL import Image

from langslice.core import appearance as looks
from langslice.core.deformable import (
    CandidateFailure,
    DeformableRecord,
    FitSettings,
    Placement,
    draw_warped_borders,
    fit_prepared,
    placement_from_handoff,
    prepare_fit,
    resampled_record,
    warp_from_native_map,
)
from langslice.core.deformable.engines import (
    FIREANTS_LEARNING_RATE,
    FIREANTS_VERSION,
    INVERSE_ITERATIONS,
    TRACED_LEARNING_RATE,
    TRACED_TOTAL_SIGMA_FACTOR,
)
from langslice.core.deformable.fit import run_prepared
from langslice.core.deformable.plane_images import (
    native_intensity,
    native_labels,
    normalize_intensity,
)
from langslice.core.deformable.render import composed_native_grid
from langslice.core.deformable.settings import (
    CORRELATION_RADIUS_UM,
    DEFAULT_SMOOTHNESS_UM,
    DETAIL,
    TOTAL_SIGMA_FRACTION,
    neighbourhood_um_for,
)
from langslice.core.handoff import digest
from langslice.core.state import SliceState, StackState
from langslice.core.workspace import Workspace

logger = logging.getLogger(__name__)

#: Long edge of the section grid a fit runs on and its record lives on. The
#: engine works on a coarser grid (the detail level's micrometres); this grid
#: carries the final field, the labels and the masks.
FIT_LONG_EDGE = 1536
#: The detail level every tool fit runs at: detail never changes which
#: candidate looks best, so the tool fixes it; tests set ``coarse`` here for
#: speed.
DETAIL_LEVEL = "standard"
#: Fitted records kept in memory for reuse (a record is tens of megabytes).
CACHE_SIZE = 8
#: The ``fit_atlas`` that reads the boundaries of the regions the image model
#: is shown (``levels.MODEL_DEPTH``; the engine's ``borders_merged``). Every
#: other ``fit_atlas`` is one of the atlas's images, by its own name
#: (:func:`langslice.core.atlas.images.atlas_images`).
FIT_BORDERS = "borders"
#: What of the section a fit can read (``fit_section``).
FIT_LOOK = "fit"
TRACED_BORDERS = "traced_borders"
FIT_SECTIONS = (FIT_LOOK, TRACED_BORDERS)
#: Flags listed per fit before the rest are only counted.
MAX_FLAGS = 12
#: Flags about the whole section, listed before any per-region flag.
SECTION_FLAGS = ("DISPLACEMENT_OUTSIZED", "FOLDS")
#: Longest one ``fit_deformable`` call waits for running ``trace_borders``
#: calls its traced section images need (an image call takes ~1-3 minutes).
TRACE_WAIT_S = 300.0
#: Run several fits in a process pool; a single fit runs in this process.
#: Tests switch the pool off.
USE_PROCESS_POOL = True
#: Fits that run in this process run one at a time (a job's background work
#: beside the agent's own calls): the engines' seed and thread settings are
#: process-wide.
_ENGINE_LOCK = threading.Lock()


class FitRefusal(Exception):
    """A refusal with its error code and payload fields."""

    def __init__(self, code: str, message: str, **extra: Any) -> None:
        super().__init__(message)
        self.payload = {"status": "error", "error": code, "message": message, **extra}


def engine_available() -> bool:
    """Whether the deformable engine's libraries (FireANTs, torch) are installed."""
    return all(importlib.util.find_spec(name) is not None for name in ("fireants", "torch"))


#: The refusal when :func:`engine_available` is False (model-facing text,
#: pinned by the text snapshot).
ENGINE_MISSING = (
    "The deformable engine cannot run on this host: torch and fireants must both "
    "import. They are LangSlice's dependencies: reinstall LangSlice "
    "(its pyproject.toml pins the FireANTs build)"
)


# --- the linear placement a deformation lives on --------------------------


def linear_key(state: StackState, record: SliceState) -> str:
    """Identity of the section's linear placement: what a warp is fitted on."""
    transform = record.transform or {}
    return digest({
        "atlas": state.atlas, "plane": state.plane, "position_mm": record.position_mm,
        "angles": [record.pitch_deg, record.yaw_deg], "flip": record.flip,
        "rotation_deg": record.rotation_deg,
        "transform": {key: transform.get(key) for key in ("params", "calibration")},
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


def fit_grid(
    state: StackState, ctx: Workspace, record: SliceState,
    transform: dict[str, Any] | None = None,
) -> Grid:
    """The section's fit grid (``ValueError`` without a usable linear placement).

    *transform* stands in for the written one (``elastix_affine`` starts
    from it).
    """
    from langslice.core.handoff import prepare_linear_registration

    prepared = prepare_linear_registration(state, ctx, record.id, long_edge=FIT_LONG_EDGE,
                                           transform=transform)
    return Grid(record=record, placement=placement_from_handoff(prepared),
                handoff=dict(prepared.metadata), image=prepared.image)


def _on_grid(image: Image.Image, grid: Grid) -> Image.Image:
    return image if image.size == grid.image.size else image.resize(
        grid.image.size, Image.Resampling.BILINEAR)


def stain_image(
    ctx: Workspace, state: StackState, grid: Grid, channel: str,
) -> tuple[Image.Image, Any]:
    """``(image, identity)`` of the channel a fit reads on the fit grid
    (:mod:`langslice.core.appearance`): ``preprocessed``, a raw channel or a
    derived one.

    Traced fit sections draw their pictures on the preprocessed channel too.
    """
    if channel in FIT_SECTIONS:  # COMPAT (integrator removes): callers still pass a fit section
        channel = "preprocessed"
    if channel != "preprocessed":  # INTEGRATOR: replace with appearance.channel_image (chunk B)
        raise ValueError(f"Unknown channel {channel!r}")
    record = grid.record
    look = looks.preprocessed_settings(state, record.id)
    image = looks.preprocessed_image(ctx, state, record, long_edge=FIT_LONG_EDGE)
    identity: dict[str, Any] = {"fit_look": look}
    if look is None:  # the default appearance: what it is drawn from, so a
        # saved fit is not reused after the run's preprocessing changes
        identity["default"] = [looks.look_token(ctx, None), ctx.spec.host_preprocessing]
    return _on_grid(image, grid), identity


def traced_lines(
    state: StackState, ctx: Workspace, grid: Grid, *, running: bool = False,
    waited_s: float = TRACE_WAIT_S, root: Path | None = None,
) -> tuple[np.ndarray, str]:
    """The image model's lines from ``trace_borders`` at THIS placement, on the grid.

    Refused unless the section holds a completed trace whose geometry is the
    current one. *running* means the caller waited *waited_s* seconds for the
    section's trace call and it is still running. *root* is the job folder a
    stored relative artifact directory is under. Returns the line mask and
    the trace's artifact directory as stored (so a cache key built on it
    survives a move of the job folder).
    """
    from langslice.core import handoff

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
    if held.get("geometry_fingerprint") != handoff.correction_fingerprint(state, ctx, record.id):
        raise FitRefusal("TRACE_STALE", f"{record.id}'s trace_borders result was made at "
                         "a different placement; trace the current placement first.",
                         id=record.id)
    stored = str(held.get("artifact_dir", ""))
    directory = Path(stored)
    if not directory.is_absolute() and root is not None:
        directory = Path(root) / directory
    lines_path = directory / "extracted_lines.png"
    request_path = directory / "request.json"
    if not lines_path.exists() or not request_path.exists():
        raise FitRefusal("NO_TRACE", f"{record.id}'s trace_borders artifacts are missing "
                         f"({stored}).", id=record.id)
    atlas_to_canvas = np.asarray(json.loads(request_path.read_text())["atlas_to_canvas"],
                                 dtype=np.float64)
    with Image.open(lines_path) as opened:
        lines = np.asarray(opened.convert("L"), dtype=np.float32) / 255.0
    section_from_canvas = grid.placement.atlas_to_section @ np.linalg.inv(atlas_to_canvas)
    warped = cv2.warpAffine(lines, section_from_canvas[:2], grid.image.size,
                            flags=cv2.INTER_LINEAR, borderValue=0)
    return warped > 0.05, stored


# --- one candidate's choices ----------------------------------------------


@dataclass(frozen=True)
class Choice:
    """One candidate as resolved: what the agent named, and the engine's settings."""

    fit_section: str
    fit_atlas: str
    #: The warp's smoothness in micrometres (``FitSettings.smoothness_um``).
    smoothness_um: float = DEFAULT_SMOOTHNESS_UM
    #: The section mask by name (``"none"`` or a mask of the section,
    #: :mod:`langslice.core.section_masks`); its weights reach the fit through
    #: :attr:`Job.section_weights`.
    mask: str = "none"
    #: COMPAT (integrator removes): the former stiffness level, read as
    #: ``smoothness_um`` (soft 50, medium 80, firm 120).
    stiffness: str | None = None

    def __post_init__(self) -> None:
        if self.stiffness is not None:
            object.__setattr__(self, "smoothness_um", {
                "soft": 50.0, "medium": 80.0, "firm": 120.0}[self.stiffness])
            object.__setattr__(self, "stiffness", None)

    def settings(self, include: tuple[str, ...], exclude: tuple[str, ...], *,
                 brain_scale: float = 1.0) -> FitSettings:
        """The engine settings; a restricted fit's neighbourhood is the default
        at the atlas's *brain_scale* (``core.atlas.size.brain_scale``)."""
        traced = self.fit_section == TRACED_BORDERS
        return FitSettings(
            smoothness_um=self.smoothness_um,
            detail=DETAIL_LEVEL,  # type: ignore[arg-type]
            atlas_image="borders_merged" if self.fit_atlas == FIT_BORDERS else self.fit_atlas,
            section_image="lines" if traced else "stain",
            labels="model" if traced else "none",
            exclude=exclude, structures=include,
            neighbourhood_um=neighbourhood_um_for(brain_scale),
        )

    def echo(self) -> dict[str, Any]:
        return {"smoothness_um": self.smoothness_um, "mask": self.mask,
                "fit_section": self.fit_section, "fit_atlas": self.fit_atlas}


def engine_settings(settings: FitSettings, brain_scale: float = 1.0) -> dict[str, Any]:
    """The engine numbers a setting stands for, in physical units, at the
    atlas's *brain_scale* (the adult mouse's numbers times it)."""
    level = DETAIL[settings.detail]
    metric = settings.metric
    used: dict[str, Any] = {"metric": metric, "working_um": level.working_um * brain_scale}
    if metric == "local_correlation":
        used["correlation_radius_um"] = CORRELATION_RADIUS_UM * brain_scale
    if settings.section_image == "stain" and settings.stain_edges:
        used["edge_channel"] = True
    update_sigma_mm = settings.smoothness_um / 1000.0 * brain_scale
    used.update(smoothness_um=settings.smoothness_um * brain_scale,
                update_sigma_mm=update_sigma_mm,
                total_sigma_mm=update_sigma_mm * TOTAL_SIGMA_FRACTION,
                iterations=list(level.iterations))
    if settings.labels != "none":
        used["label_map"] = settings.labels
    if settings.structures:
        used["neighbourhood_um"] = settings.neighbourhood_um
    return used


def cache_key(
    state: StackState, grid: Grid, settings: FitSettings, image_identity: Any,
    previous_key: str | None, brain_scale: float = 1.0, mask_identity: Any = None,
) -> str:
    """Every input of one fit: the same key means the same result. The atlas
    (in the placement) fixes its *brain_scale*; *mask_identity* names the
    section mask the fit was given (its file, ``None`` for no mask)."""
    return digest({
        "section": grid.record.id, "linear": linear_key(state, grid.record),
        "grid": [FIT_LONG_EDGE, list(grid.image.size)],
        "placement": grid.placement.to_dict(), "image": image_identity,
        "settings": settings.to_dict(), "start": previous_key or "linear",
        # The engine and the section mask: a record from another engine or
        # mask is not reused.
        "engine": ["fireants", FIREANTS_VERSION, FIREANTS_LEARNING_RATE, INVERSE_ITERATIONS,
                   list(DETAIL[settings.detail].iterations)],
        **({"traced_engine": [TRACED_LEARNING_RATE, TRACED_TOTAL_SIGMA_FACTOR, "balanced"]}
           if settings.section_image == "lines" else {}),
        **({"section_mask": mask_identity} if mask_identity is not None else {}),
    })


# --- records: cache, disk, the reference on the state ---------------------


def folder_name(key: str) -> str:
    """The folder a record of *key* is saved in: the key's first 24 hex digits."""
    return key[:24]


@dataclass
class RecordStore:
    """Fitted records by input key (memory, most recent first) and on disk.

    On disk a record is ``<folder>/<key[:24]>``: *folder_of* gives a
    section's folder (the job's ``sections/<name>/deformable``); without it,
    ``<root>/<section filename>``. A record's parent is a sibling folder
    there, named in its ``record.json`` (:meth:`DeformableRecord.save`).
    Nothing removes a saved record: undo returns to a parent's key, and a
    child reads its parent from the folder.

    A record put here is rounded to the precision it is saved at
    (:meth:`DeformableRecord.round_to_storage`), so a record from memory and
    the same record read back are equal, and a fit built on either is the same.
    """

    root: Path
    cache: OrderedDict[str, DeformableRecord] = field(default_factory=OrderedDict)
    folder_of: Callable[[str], Path] | None = None

    def directory(self, section_id: str, key: str) -> Path:
        if self.folder_of is not None:
            return Path(self.folder_of(section_id)) / folder_name(key)
        name = re.sub(r"[^a-zA-Z0-9._-]", "_", Path(section_id).name)
        return self.root / name / folder_name(key)

    def get(self, section_id: str, key: str) -> DeformableRecord | None:
        record = self.cache.get(key)
        if record is None:
            folder = self.directory(section_id, key)
            if (folder / "record.json").exists():
                record = DeformableRecord.load(folder)
        if record is not None:
            self.put(key, record)
        return record

    def put(self, key: str, record: DeformableRecord) -> None:
        if record.saved_as is None:  # a record with a folder is at storage precision
            record.round_to_storage()
        record.saved_as = folder_name(key)
        self.cache[key] = record
        self.cache.move_to_end(key)
        while len(self.cache) > CACHE_SIZE:
            self.cache.popitem(last=False)

    def save(self, section_id: str, key: str, record: DeformableRecord) -> Path:
        """Save *record* under *key* (its parent first, when that is not saved
        beside it); a record already saved there is kept."""
        folder = self.directory(section_id, key)
        if not (folder / "record.json").exists():
            record.save(folder)
        record.saved_as = folder.name
        return folder

    def current(self, state: StackState, record: SliceState) -> DeformableRecord | None:
        """The section's applied record, when it still sits on its placement."""
        held = record.deformation
        if not held or held.get("linear_key") != linear_key(state, record):
            return None
        return self.get(record.id, str(held.get("key", "")))


def summary(record: DeformableRecord, previous: DeformableRecord | None) -> dict[str, Any]:
    """Compact per-candidate numbers: displacement, folds, flags."""
    footprint = record.footprint
    magnitude = np.linalg.norm(record.field_mm, axis=-1)[footprint]
    result: dict[str, Any] = {
        "displacement_mm": {
            "max": round(float(magnitude.max()), 3) if magnitude.size else 0.0,
            "median": round(float(np.median(magnitude)), 3) if magnitude.size else 0.0,
        },
        "fold_fraction": round(float(record.diagnostics.get("fold_fraction", 0.0)), 5),
    }
    if previous is not None and previous.field_mm.shape == record.field_mm.shape:
        step = np.linalg.norm(record.field_mm - previous.field_mm, axis=-1)[footprint]
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
    *, folder: Path | str, key: str, linear: str, record: DeformableRecord,
    step: dict[str, Any], previous: dict[str, Any] | None, numbers: dict[str, Any],
) -> dict[str, Any]:
    """What ``SliceState.deformation`` holds for an applied record.

    *folder* is where the record lives as the job stores it: relative to the
    job folder (the caller makes it so). *step* describes this step (its
    ``start`` is ``current`` when it builds on *previous*, whose steps then
    come first, else ``linear``).
    """
    steps = list((previous or {}).get("steps") or []) if step.get("start") == "current" else []
    return {
        "record": str(folder), "key": key, "linear_key": linear,
        "steps": [*steps, step], "summary": numbers,
        "inverse_source": record.inverse_source,
    }


def record_from_map(
    ctx: Workspace, grid: Grid, native_xy: np.ndarray, *, method: str,
    provenance: dict[str, Any],
) -> DeformableRecord:
    """A deformation made elsewhere as an applied record on *grid*: *native_xy*
    ``(H, W, 2)`` is the full map from the fit grid's pixel centres to native
    atlas-plane pixels (``(H, W)`` the grid image's size), stored as the residual
    over the grid's linear placement
    (:func:`langslice.core.deformable.warp_from_native_map`). The record's
    engine is *method*; its provenance is *provenance* with the section grid's
    facts."""
    outcome = warp_from_native_map(ctx.atlas, grid.placement, native_xy, DETAIL_LEVEL,
                                   engine=method)
    outcome.provenance = {
        **provenance, "section_id": grid.record.id, "linear_handoff": grid.handoff,
        "section_grid": {"size": list(grid.image.size), "long_edge": FIT_LONG_EDGE,
                         "frame": "oriented rendered section, unframed; not acquisition "
                                  "pixels (see linear_handoff)"},
    }
    return outcome


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
    #: The section mask's weights (float, the fit grid's ``(H, W)``;
    #: :func:`langslice.core.section_masks.section_weights`); None for no mask.
    section_weights: np.ndarray | None = None


def engine_lock() -> threading.Lock:
    """The lock one in-process deformable fit holds (fits share the engine)."""
    return _ENGINE_LOCK


def run_jobs(ctx: Workspace, jobs: list[Job]) -> None:
    """Prepare and run every job without a result, concurrently when several."""
    prepared = []
    waiting: list[Job] = []
    for job in jobs:
        if job.result is not None:
            continue
        try:
            prepared.append(prepare_fit(
                job.image, ctx.atlas, job.grid.placement, job.settings, lines=job.lines,
                previous=job.previous, section_weights=job.section_weights,
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
                with _ENGINE_LOCK:
                    results.append(run_prepared(item))
            except Exception as exc:  # noqa: BLE001
                results.append(CandidateFailure(item.settings, f"{type(exc).__name__}: {exc}"))
    for job, result in zip(waiting, results, strict=True):
        job.result = result


# --- the picture ------------------------------------------------------------


@dataclass(frozen=True)
class Style:
    """How one call's pictures are drawn (its display options)."""

    #: Zoom in pixels of the unzoomed picture (``display.zoom_fractions``);
    #: empty is the whole section.
    zoom: tuple[float, ...]
    #: Regions drawn strong; empty draws every border strong.
    highlight: tuple[str, ...]
    #: Regions drawn in the second color (those excluded from the fit).
    marked: tuple[str, ...]
    outlines: str
    #: The image model's trace colour (:func:`trace_picture`: plain, read back by colour).
    color: tuple[int, int, int]
    thickness: float
    atlas_opacity: float
    #: Ontology depth of the regions whose boundaries are drawn.
    depth: int
    #: Long edge of the picture (of the zoomed crop when zoomed), in pixels;
    #: never more than the fit image's own pixels.
    long_edge: int = 512
    #: Core colour of the fitted borders in :func:`picture`, ``black`` or
    #: ``white`` (cased with the other).
    core: str = "black"


def picture(
    ctx: Workspace,
    image: Image.Image,
    record: DeformableRecord,
    *,
    warped: bool,
    style: Style,
    atlas_images: tuple[str, ...],
    title: str,
    note: dict[str, Any] | None = None,
) -> Image.Image:
    """The record's borders on the section at ``style.long_edge``, captioned.

    The crop (the zoom, or the whole fit image) is shown at ``style.long_edge``
    on its long side, or at its own pixels when it has fewer: the fit image
    (:data:`FIT_LONG_EDGE`) is never upsampled. *atlas_images* (the atlas's
    images among the call's ``view.atlas_channels``) are pulled through the
    record's map and blended under the lines at ``style.atlas_opacity``.
    With *note* (:func:`langslice.core.layers.note` fields: sections, mode)
    the picture is noted with its frame (:class:`~langslice.core.layers.WarpNote`),
    so the job saves its layers.
    """
    from langslice.core import layers
    from langslice.core.canvas import zoom_box
    from langslice.core.captions import caption
    from langslice.core.display import zoom_fractions

    width, height = record.section_size
    zoom = list(zoom_fractions(style.zoom, unzoomed_size((width, height), style.long_edge)))
    box = zoom_box(zoom, (width, height)) if zoom else (0, 0, width, height)
    crop = (box[2] - box[0], box[3] - box[1])
    factor = min(1.0, int(style.long_edge) / float(max(crop)))
    full = (max(8, int(round(width * factor))), max(8, int(round(height * factor))))
    shown = (0, 0, full[0], full[1])
    if zoom:
        shown = zoom_box(list(zoom), full)
    small = resampled_record(record, full, shown)
    base = image.convert("RGB").resize(full, Image.Resampling.LANCZOS).crop(shown)
    kinds = tuple(kind for kind in atlas_images if kind != "borders")
    if style.atlas_opacity > 0 and kinds:
        base = _blend_atlas(ctx, base, small if warped else _unwarped(small), kinds,
                            style.atlas_opacity)
    native = native_labels(ctx.atlas, record.placement)
    drawn = draw_warped_borders(
        base, small, ctx.atlas, highlight=style.highlight, marked=style.marked, warped=warped,
        outlines=style.outlines, depth=style.depth, width_px=style.thickness,
        core=style.core, native=native,
    )
    captioned = caption(drawn, title)
    if note is not None:
        layers.note(captioned, **note, warp=layers.WarpNote(
            record=small, band=0, warped=warped,
            highlight=tuple(style.highlight), marked=tuple(style.marked),
            outlines=style.outlines, depth=style.depth, width_px=float(style.thickness),
            native=native))
    return captioned


def unzoomed_size(size: tuple[int, int], long_edge: int) -> tuple[int, int]:
    """The content size :func:`picture` and :func:`trace_picture` draw a whole
    image of *size* at: *long_edge* on its long side, never upsampled."""
    factor = min(1.0, int(long_edge) / float(max(size)))
    return (max(8, int(round(size[0] * factor))), max(8, int(round(size[1] * factor))))


def trace_picture(
    image: Image.Image, lines: np.ndarray, *, style: Style, title: str,
) -> Image.Image:
    """The image model's traced lines on the section image, at ``style.long_edge``.

    *image* and *lines* share the fit grid (:func:`traced_lines`); the zoom,
    border color and thickness are the call's, and the picture is never drawn
    past the fit image's own pixels, like :func:`picture`.
    """
    from langslice.core.canvas import zoom_box
    from langslice.core.captions import caption
    from langslice.core.display import zoom_fractions

    width, height = image.size
    zoom = list(zoom_fractions(style.zoom, unzoomed_size((width, height), style.long_edge)))
    box = zoom_box(zoom, (width, height)) if zoom else (0, 0, width, height)
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
    ctx: Workspace, base: Image.Image, record: DeformableRecord, kinds: tuple[str, ...],
    opacity: float,
) -> Image.Image:
    """The atlas images pulled through the record's map, blended under the lines.

    One image is gray; two are added in their colours
    (:func:`langslice.core.appearance.channel_colors`), as everywhere else.
    """
    labels = native_labels(ctx.atlas, record.placement)
    nx, ny = composed_native_grid(record, 1)

    def pulled(kind: str) -> np.ndarray:
        native = native_intensity(ctx.atlas, record.placement, kind)
        native = normalize_intensity(native, labels > 0)
        return cv2.remap(native.astype(np.float32), nx, ny, cv2.INTER_LINEAR,
                         borderMode=cv2.BORDER_CONSTANT, borderValue=0)

    if len(kinds) == 1:
        atlas = pulled(kinds[0])[..., None] * 255.0
    else:
        atlas = sum(pulled(kind)[..., None] * np.asarray(rgb, dtype=np.float32)
                    for kind, _word, rgb in looks.channel_colors(kinds))
    pixels = np.asarray(base, dtype=np.float32)
    mixed = pixels * (1.0 - opacity) + atlas * opacity
    return Image.fromarray(np.rint(np.clip(mixed, 0, 255)).astype(np.uint8))
