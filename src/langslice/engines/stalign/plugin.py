"""STalign as LangSlice tools: the adapter, run in LangSlice's process.

It is a tool of every job (its row of the tool table names
``langslice.engines.stalign.plugin:tools``). Options (``plugin_config``):
``device`` (``auto``: the GPU at float32 when there is one, else the CPU at
float64) and ``threads``; any other key is refused. STalign (GPL-3.0, vendored
as ``langslice._vendor.stalign``) runs in LangSlice's process: ``engine.py``
imports it when a tool runs. Its declaration is its rows of the tool table
(``langslice.plugins.declared``).

Two tools:

- ``stalign_lddmm``: STalign's 2-D fit, one piece of background work per section, building on
  the section's current registration (a landmark warp included);
- ``stalign_plane_search``: STalign 2.0's 3-D plane search, proposing cutting angles and
  positions, nothing written.

Inputs, as every engine takes them: ``section_image`` (a section image by name, default
``preprocessed``), ``atlas_image`` (default ``nissl``, resolved by the job's atlas; the reply
names the image used) and ``mask`` (STalign takes none: it sorts the section itself, and its final
sorting is saved with the deformation). The reply carries ``settings`` and, per section,
``inputs`` and ``estimated``.

Frames. The target is the section image (by name) on the fit grid, grey, scaled so
the slide (median of the image border) is 0 and the tissue (median inside the placed atlas
outline) is 1, whatever the stain's polarity, area-averaged to 10 um. Target micrometres are
atlas micrometres at the section's linear scale. The source is the atlas image on its native
plane, area-averaged to 50 um (the published atlas voxel), I / mean|I| plus (I - mean I)^2 (the
published atlas recipe); on a section that holds a deformation, the atlas resampled onto the
fit grid through it (``FitFrame.applied_map``), so the bend starts on top of a landmark warp.
Marked damage (with the native margin) is painted as atlas background.
"""

from __future__ import annotations

import csv
import json
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from functools import partial
from pathlib import Path
from typing import Any, Literal, TypeVar

import cv2
import numpy as np
from PIL import Image, ImageDraw

from langslice.core.failures import ENGINE_FAILURES
from langslice.core.refusal import Refused
from langslice.engines import _blocks
from langslice.engines._options import device_option, known_options
from langslice.engines._pixels import (
    area_average,
    outlined,
    placed_matrix,
    placed_outline,
    sample_pixels,
    slide_level,
)
from langslice.engines._sorting import SORTING_KEYS, Sorting
from langslice.engines.stalign import engine
from langslice.plugin_api import (
    DeformationProposal,
    FitFrame,
    RefusalCode,
    ToolContext,
    ToolError,
    ToolResult,
    ToolsOn,
    ToolSpec,
    flag_words,
    map_sections,
)
from langslice.plugins.declared import engine_tool

Result = TypeVar("Result")
OPTIONS = ("device", "threads")
FITS = ("warp", "both")
SORTING_METHODS = ("classic", "em", "none")
ENGINE_VERSION = "STalign 2.0.0 8a84bf1f"
SWAP = np.array([[0.0, 1.0, 0.0], [1.0, 0.0, 0.0], [0.0, 0.0, 1.0]])
BRIGHTNESS = "slide 0, tissue 1"
SAME_LEVEL = 1e-6
MIN_SCALE = 1e-12
LOSS_COLUMNS = ("iteration", "total", "image_mismatch", "smoothness", "landmarks")
LOSS_ROW_UM = 1


@dataclass(frozen=True)
class Resolutions:
    """Hold the micrometres the two images are averaged to.

    Parameters
    ----------
    target_um
        The section image's pixel, um.
    source_um
        The atlas image's pixel, um (the published atlas voxel).
    """

    target_um: float = 10.0
    source_um: float = 50.0


@dataclass(frozen=True)
class SearchStarts:
    """Hold the plane search's start grid.

    Parameters
    ----------
    tilt_deg
        The largest tilt about either in-plane axis (starts are -tilt, 0, tilt).
    shift_um
        The largest shift along the slicing axis (starts are -shift, 0, shift).
    """

    tilt_deg: float = 6.0
    shift_um: float = 300.0


@dataclass(frozen=True)
class LossPlot:
    """Hold the size of the mismatch plot, px.

    Parameters
    ----------
    height, width
        The image's size.
    pad
        The margin around the line.
    """

    height: int = 240
    width: int = 640
    pad: int = 30


@dataclass(frozen=True)
class Pictures:
    """Hold the sizes of the sorting picture.

    Parameters
    ----------
    gap_px
        The white strip between the section and the sorting.
    long_edge
        The longest side of the picture sent.
    """

    gap_px: int = 8
    long_edge: int = 1600


RESOLUTIONS = Resolutions()
STARTS = SearchStarts()
LOSS_PLOT = LossPlot()
PICTURES = Pictures()
PUBLISHED_SORTING: dict[str, Any] = {}


MASK_ENGINES = ("fireants_syn", "elastix_affine", "emlddmm_stack", "spomialign")


def _lddmm_description(on: ToolsOn) -> str:
    rough_start = on.say(
        "interactive_transform",
        ": set the orientation and a rough transform with interactive_transform first",
    )
    landmark_kept = on.say(
        "set_landmarks", ", a landmark warp included, so the landmark move is kept"
    )
    damage_first = on.say("mark_damage", "; mark them with mark_damage first")
    mask_from_sorting = on.say(
        "make_mask", ", and make_mask can take it as a mask for another fit (source fit)"
    )
    mask_engines = on.say_any(
        MASK_ENGINES,
        "; the engines that take one are " + on.listed([(name, name) for name in MASK_ENGINES]),
    )
    refine_regions = on.say(
        "fireants_syn",
        "; refine single regions afterwards with fireants_syn, which builds on this fit",
    )
    opened_by = on.say("read_file", ", which read_file opens")
    made_images = on.say("make_image", ", or an image made with make_image")
    return f"""\
Fit the atlas onto sections with STalign, on their current registration.

STalign (Clifton et al. 2023, Nature Communications) is LDDMM registration
(large deformation diffeomorphic metric mapping) built to align single-cell
spatial transcriptomics, as rasterised cell images or histology, to each other
and to the Allen CCF. Its distinctive parts are an estimated contrast map, so
the two images need not share brightness, and its own per-pixel sorting into
matching tissue, background and artifact. It is a gradient descent that finds
the nearest fit, so it needs a close start.

STalign bends the atlas image onto each section's image by gradient
descent on their brightness mismatch, with a linear contrast fit between
the two, a kernel width (kernel_um: tissue closer than that bends
together) and a displacement scale (displacement_um: how far a point may
be carried before the penalty grows). It starts from the section's
current registration (its linear placement, and its deformation when it
has one) and refines it; it does not search from scratch{rough_start}. With fit "warp" the \
transform stays and
only the bend is fitted, on top of the deformation the section holds{landmark_kept}. With fit "both"
it also refits the section's in-plane transform and writes it; refused on a
section that holds a deformation, since a new transform clears it. The
section's marked damage regions are left out automatically{damage_first}.

STalign takes no section mask: it sorts the section's pixels itself. Before
each comparison every pixel is weighted between matching (compared),
background (slide, ignored) and artifact (debris, ignored), from the
mismatch and from the background and artifact means (sorting, below). The
final sorting is saved with the deformation{mask_from_sorting}. A named mask is
refused{mask_engines}. STalign fits the whole
section: it takes no regions{refine_regions}. It keeps its own brightness
scaling: slide 0, typical tissue 1.

Starts one piece of background work per section and returns at once with
their ids, so you can carry on with other sections. A write to a section
whose fit still runs is refused until it lands. When a section's fit
lands, the next tool reply starts with its notice: what was fitted, the
displacement, fold fraction and flags, what STalign used and estimated,
and the file in the job folder that holds STalign's own mismatch record
for the fit{opened_by}; then its pictures. Each
landing is its own undo step. status lists the work still running; submit
waits for it. Judge the overlay against the internal anatomy.

Args:
    sections: One to twenty filenames (the extension may be left off).
    section_image: The section image to fit, by name: "preprocessed", a raw channel{made_images}. \
STalign reads it as grey.
    atlas_image: One of the atlas's images (the job statement lists
        them: "template" and any other reference images the atlas ships,
        by their names), or "nissl", the atlas's Nissl (its template when
        it ships none). Default "nissl"; the reply names the image used.
    mask: "none". STalign takes no section mask (see above).
    kernel_um: The width of the kernel that smooths the bend's velocity,
        in micrometres (STalign's a; not a Gaussian sigma on the
        displacement, as smoothness_um is elsewhere): larger bends broader
        patches as one. Default 500.
    fit: "warp" bends on top of the section's registration and keeps its
        transform; "both" also refits the in-plane transform (turn,
        stretch, shift) and writes it. Default "warp".
    displacement_um: The displacement scale in micrometres; larger allows
        a bigger bend. Default 94.
    sorting: {{"method": "classic" | "em" | "none"}} and any of "match_sd",
        "background_sd", "artifact_sd" (spreads), "background_mean",
        "artifact_mean" (left out: estimated) and "artifact" (true adds the
        artifact bin to "em"). Brightness units: slide 0, typical tissue 1.
        "classic" is STalign's published sorting with fixed spreads; "em"
        measures the matching spread itself and takes the background mean
        from the slide at the image border, for sections with missing
        tissue; "none" compares every pixel. Default {{"method": "classic"}}
        with STalign's own spreads.
    iterations: Gradient-descent steps. Default 2000.
    view: True sends the pictures described below with each notice; false
        sends none.

Returns:
    Per section its work id and what the fit uses (inputs: the section image,
    the atlas image as asked and as used, the mask, the damage, the
    brightness scaling), or why it was not started. Each notice brings,
    per section, the numbers above, what STalign ran at (engine_settings:
    its target and atlas grids in micrometres and its own settings), what
    STalign estimated (its sorting,
    saved with the deformation, and the share of the tissue it left out)
    and the section under its new registration with the atlas borders on
    it; when the sorting set some tissue aside, also the final sorting
    picture (set-aside patches outlined; white compared, black ignored),
    and when it set nothing aside the notice says so and sends no picture."""


def stalign_lddmm(
    sections: list[str],
    section_image: str = "preprocessed",
    atlas_image: str = "nissl",
    mask: str = "none",
    kernel_um: float = 500.0,
    fit: Literal["warp", "both"] = "warp",
    displacement_um: float = 94.0,
    sorting: dict[str, Any] = PUBLISHED_SORTING,
    iterations: int = 2000,
    view: bool = True,
) -> dict[str, Any]: ...


def _plane_search_description(on: ToolsOn) -> str:
    write_with = on.say("position_sections", " with position_sections")
    flip_with = on.say("interactive_transform", " with interactive_transform")
    return f"""\
Estimate the cutting angles with STalign's plane search. Writes nothing.

For each section, STalign tilts the atlas plane the section sits on by -6,
0 and 6 degrees about each in-plane axis and shifts it by -300, 0 and 300
um along the slicing axis (27 trial planes), fits each roughly to the
section's image at low resolution, refines the best, and
reports where that plane lies: its pitch and yaw in this stack's
convention, and the position it implies. It starts from the section's
current position, the stack's cutting angles and its transform, so place
and orient the section first. Marked damage is left out automatically.
One search reaches about 6 degrees from the start; from further away it
lands short, so write the angles and search again from them until two
passes agree within a degree or two.

The stack has one plane. Search sections with plenty of anatomy spread
along the stack, take the middle value of the pitches and of the yaw
magnitudes, leave out a section flagged at_search_edge or whose picture
at the found plane looks wrong, and write the pair{write_with}. Keep the positions you placed \
unless a
section's picture shows its position change is right. A section mounted
mirrored reverses the sign of its yaw and keeps its pitch, and the
hemispheres often cannot be told apart by any other sign, so many
sections of a stack, up to half, may be mirrored: the reply sorts the
sections by the sign of their yaw. Make every section agree on one sign
(mirror the smaller group{flip_with} and search them again) before writing the angles.

Args:
    sections: One to eight filenames (the extension may be left off).
    section_image: The section image to search with, as for the fits.
    atlas_image: One of the atlas's images, or "nissl", as for the fits.
    view: True returns the pictures described below; false returns none.

Returns:
    Per section pitch_deg, yaw_deg and position_mm of the found plane,
    change (the differences from the current pitch, yaw and position)
    and at_search_edge; the sections grouped by the sign of their yaw,
    with the middle pitch and yaw magnitude of the call; and a picture of
    each section with the atlas borders drawn at the found plane, not
    written."""


def stalign_plane_search(
    sections: list[str],
    section_image: str = "preprocessed",
    atlas_image: str = "nissl",
    view: bool = True,
) -> dict[str, Any]: ...


@dataclass(frozen=True)
class Options:
    """Hold the plugin's options (``plugin_config``).

    Parameters
    ----------
    device
        ``auto``, ``cpu``, ``cuda`` or a numbered GPU such as ``cuda:1``.
    threads
        The threads STalign may use, and the sections searched at once.
    """

    device: str = "auto"
    threads: int = 4

    @classmethod
    def from_config(cls, config: Mapping[str, object] | None) -> Options:
        """Read the options a job's ``plugin_config`` gives.

        Parameters
        ----------
        config
            The plugin's entry of ``plugin_config``; None gives the defaults.

        Returns
        -------
        Options
            The options.

        Raises
        ------
        PluginError
            For an option the plugin does not take.
        """
        given = known_options("STalign", config, OPTIONS)
        return cls(
            device=device_option(given.get("device", cls.device)),
            threads=int(str(given.get("threads", cls.threads))),
        )

    @property
    def machine(self) -> engine.Machine:
        """Return where STalign runs."""
        return engine.Machine(self.device, self.threads)


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


def _target(frame: FitFrame) -> _Target:
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


def _grid_query(size: tuple[int, int], um: float) -> np.ndarray:
    """Every fit-grid pixel as (row um, col um), in raster order."""
    gw, gh = size
    yy, xx = np.indices((gh, gw), dtype=np.float64)
    return np.stack([yy.ravel() * um, xx.ravel() * um], -1)


def _native_map(source_um: np.ndarray, source: _Source, target: _Target) -> np.ndarray:
    """STalign's source points of every fit-grid pixel as native atlas pixels (x, y)."""
    gw, gh = target.size
    um = target.um_per_px
    pts = source_um.reshape(gh, gw, 2)
    if source.applied is not None:
        grid = np.stack([pts[..., 1] / um, pts[..., 0] / um], -1)
        return sample_pixels(source.applied, grid).astype(np.float64)
    px_x, px_y = source.atlas_px_um
    return np.stack([pts[..., 1] / px_x, pts[..., 0] / px_y], -1)


def _sorting_settings(sorting: Mapping[str, Any]) -> tuple[Sorting, str | None]:
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


def _frame(
    ctx: ToolContext, name: str, atlas_image: str, section_image: str
) -> tuple[FitFrame, np.ndarray, str]:
    """The section's fit frame, the atlas image asked for on it and the name that resolved to."""
    frame: FitFrame = _blocks.frame_for(ctx, name, atlas_image, section_image)
    plane_image, resolved = _blocks.atlas_plane(frame, atlas_image)
    return frame, plane_image, resolved


def _sorting_picture(
    ctx: ToolContext, frame: FitFrame, weights: np.ndarray, title: str
) -> Image.Image | None:
    """The section's image with every set-aside patch outlined, beside the weight map
    (white compared, black ignored); None when the sorting set nothing aside (a map of
    all white says nothing, so no panel is drawn)."""
    stain = _blocks.section_pixels(frame)
    gh, gw = stain.shape[:2]
    w = cv2.resize(np.asarray(weights, np.float32), (gw, gh), interpolation=cv2.INTER_LINEAR)
    ignored = w < 0.5
    if not ignored.any():
        return None
    left = outlined(stain, ignored)
    right = np.repeat((np.clip(w, 0, 1) * 255).astype(np.uint8)[..., None], 3, axis=2)
    gap = np.full((gh, PICTURES.gap_px, 3), 255, np.uint8)
    image = Image.fromarray(np.concatenate([left, gap, right], axis=1))
    image.thumbnail((PICTURES.long_edge, PICTURES.long_edge))
    return ctx.caption(image, title, sections=[frame.id])


def _loss_picture(energy: np.ndarray, path: Path) -> None:
    """STalign's mismatch record as a small line plot (a file in the job, not sent)."""
    h, w, pad = LOSS_PLOT.height, LOSS_PLOT.width, LOSS_PLOT.pad
    image = Image.new("RGB", (w, h), "white")
    draw = ImageDraw.Draw(image)
    if len(energy) > 1:
        y = energy[:, 1]
        lo, hi = float(y.min()), float(y.max())
        span = max(hi - lo, MIN_SCALE)
        pts = [
            (pad + (w - 2 * pad) * i / (len(y) - 1), h - pad - (h - 2 * pad) * (v - lo) / span)
            for i, v in enumerate(y)
        ]
        draw.line(pts, fill="black", width=1)
        draw.text(
            (pad, 6),
            f"STalign image mismatch per iteration ({len(y)}); min {lo:.4g}, max {hi:.4g}",
            fill="black",
        )
    image.save(path)


def _run_engine(call: Callable[[], Result], folder: Path) -> Result:
    """Run an engine call in this process, its failures refused as ``FIT_FAILED``."""
    folder.mkdir(parents=True, exist_ok=True)
    try:
        return call()
    except ENGINE_FAILURES as exc:
        failure = f"{type(exc).__name__}: {exc}"
        raise ToolError(RefusalCode.FIT_FAILED, f"STalign failed: {failure}") from exc


@dataclass(frozen=True)
class _LddmmCall:
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
class _Prepared:
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


@dataclass(frozen=True)
class _Landing:
    """Fit one section and write the result, as the section's background work."""

    call: _LddmmCall
    section: _Prepared

    def __call__(self, lctx: ToolContext) -> ToolResult:
        """Run the fit, save its files, write the registration and report it."""
        call, section = self.call, self.section
        here = call.folder / section.sid
        request = engine.LddmmRequest(
            source=section.source.image,
            source_rows_um=section.source.rows_um,
            source_cols_um=section.source.cols_um,
            target=section.target.image,
            target_rows_um=section.target.rows_um,
            target_cols_um=section.target.cols_um,
            start=section.source.start,
            query_um=_grid_query(section.target.size, section.target.um_per_px),
            settings=call.fit,
            machine=call.machine,
        )
        result = _run_engine(partial(engine.run_lddmm, request), here)
        native = _native_map(result.source_um, section.source, section.target)
        note = self._note()
        weights = (
            _blocks.on_fit_grid(result.match_weights, section.target.size) if call.sorts else None
        )
        outline = placed_outline(section.frame, section.target.size)
        estimated = _blocks.estimated_block(weights, outline)
        engine_settings = {
            "target_um": RESOLUTIONS.target_um,
            "atlas_um": RESOLUTIONS.source_um,
            **result.chosen,
            "device": result.device,
        }
        blocks = {
            "inputs": section.inputs,
            "estimated": estimated,
            "settings": call.settings,
            "engine_settings": engine_settings,
        }
        self._save(here, result, blocks)
        summary = self._write(lctx, result, native, note, weights, estimated, engine_settings)
        return self._report(lctx, here, result, note, estimated, summary, blocks)

    def _note(self) -> str:
        """Say what the fit did, for the registration's record."""
        start = "the deformation it held" if self.section.source.on_grid else "its linear placement"
        what = "affine and bend" if self.call.fit.refit_transform else "bend"
        return f"STalign {what} on {start}, atlas {self.section.resolved}"

    def _save(self, here: Path, result: engine.LddmmResult, blocks: Mapping[str, Any]) -> None:
        """Save the call's files: the energy record, its plot, the settings and the weights."""
        energy = result.energy
        with open(here / "energy.csv", "w", encoding="utf-8", newline="") as fh:
            writer = csv.writer(fh)
            writer.writerow(LOSS_COLUMNS[: 1 + energy.shape[1]])
            for i, row in enumerate(energy):
                writer.writerow([i, *[f"{v:.6g}" for v in row]])
        _loss_picture(energy, here / "mismatch.png")
        asked = {**self.call.settings, **self.call.fit.keywords()}
        (here / "settings.json").write_text(
            json.dumps(
                {
                    "asked": asked,
                    **blocks,
                    "chosen": result.chosen,
                    "device": result.device,
                    "seconds": result.seconds,
                },
                indent=1,
                default=str,
            ),
            encoding="utf-8",
            newline="\n",
        )
        np.savez_compressed(
            here / "weights.npz",
            WM=result.match_weights,
            WB=result.background_weights,
            WA=result.artifact_weights,
        )

    def _write(
        self,
        lctx: ToolContext,
        result: engine.LddmmResult,
        native: np.ndarray,
        note: str,
        weights: np.ndarray | None,
        estimated: Mapping[str, Any] | None,
        engine_settings: Mapping[str, Any],
    ) -> dict[str, Any]:
        """Write the section's transform (fit ``both``) and its deformation; return the
        deformation's summary."""
        section = self.section
        sid = section.sid
        affine = self.call.fit.refit_transform
        meta = {"inputs": section.inputs, "estimated": estimated}

        def proposal() -> DeformationProposal:
            return lctx.deformation_from_map(
                sid,
                native,
                note=note,
                meta=meta,
                weights=weights,
                start="current" if section.source.on_grid else "linear",
                settings=self.call.settings,
                engine_settings=engine_settings,
                engine_version=ENGINE_VERSION,
            )

        warp = None if affine else proposal()
        with lctx.writing():
            if affine:
                self._write_transform(lctx, result, note)
            if warp is None:
                warp = proposal()
            done = lctx.set_deformations(
                [warp.row()], expect=None if affine else {sid: section.digest}
            )
            if sid not in done.written:
                raise done.refusal(sid)
        return warp.summary

    def _write_transform(self, lctx: ToolContext, result: engine.LddmmResult, note: str) -> None:
        """Write the transform STalign refitted (fit ``both``)."""
        section = self.section
        sid = section.sid
        px_x, px_y = section.source.atlas_px_um
        um = section.target.um_per_px
        a_xy = SWAP @ result.affine @ SWAP
        placed = np.diag([1 / um, 1 / um, 1.0]) @ a_xy @ np.diag([px_x, px_y, 1.0])
        proposal = lctx.transform_from_matrix(sid, np.linalg.inv(placed)[:2], note=note)
        done = lctx.set_transforms([proposal.row()], expect={sid: section.digest})
        if sid not in done.written:
            raise done.refusal(sid)

    def _report(
        self,
        lctx: ToolContext,
        here: Path,
        result: engine.LddmmResult,
        note: str,
        estimated: Mapping[str, Any] | None,
        summary: dict[str, Any],
        blocks: Mapping[str, Any],
    ) -> ToolResult:
        """Word the landing's notice and draw its pictures."""
        sid = self.section.sid
        relative = _blocks.job_path(lctx, here)
        left = ""
        if estimated and estimated["left_out_fraction"] is not None:
            left = (
                f"; its sorting left out {estimated['left_out_fraction']:.0%} of the tissue "
                "and is saved with the deformation"
            )
        waited = result.gpu_wait_s
        queue = {"gpu_wait_s": round(waited)} if waited >= 1.0 else {}
        text = (
            f"{note}; displacement median {summary['displacement_mm']['median']} mm, "
            f"max {summary['displacement_mm']['max']} mm; fold fraction "
            f"{summary.get('fold_fraction', 0)}; flags {flag_words(summary)}{left}. "
            + (f"It waited {round(waited)} s for the machine's GPU queue. " if queue else "")
            + f"STalign's mismatch record: "
            f"{relative}/energy.csv (plot: {relative}/mismatch.png)."
        )
        pictures: list[Image.Image] = []
        if self.call.view:
            pictures = [lctx.caption(lctx.overlay_picture(sid), f"{sid}: {note}")]
            sorting = _sorting_picture(
                lctx,
                self.section.frame,
                result.match_weights,
                f"{sid}: STalign's final sorting (outlined: set aside; right: white compared)",
            )
            if sorting is None:
                text += (
                    " Its final sorting set nothing aside, so every pixel was compared "
                    "and no sorting picture is sent."
                )
            else:
                pictures.append(sorting)
        return ToolResult({"notice": text, "id": sid, **summary, **queue, **blocks}, pictures)


def _prepare(ctx: ToolContext, call: _LddmmCall, name: str) -> _Prepared | dict[str, object]:
    """Read one section's frame and images, or the row that says why it cannot be fitted."""
    section = ctx.section(name)
    sid = section.id
    if call.fit.refit_transform and section.deformation:
        refusal = ToolError(
            RefusalCode.PRECONDITION,
            'fit "both" would refit the transform, which '
            'clears the section\'s deformation; fit "warp" to bend on top of it',
        )
        return {"id": sid, **refusal.payload()}
    frame, plane_image, resolved = _frame(ctx, sid, call.atlas_image, call.section_image)
    if frame.placed_atlas_to_section is None:
        refusal = ToolError(RefusalCode.PRECONDITION, "the section needs a transform first")
        return {"id": sid, **refusal.payload()}
    target = _target(frame)
    return _Prepared(
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
        digest=ctx.section_inputs(sid, deformation=True),
        target=target,
        source=_source(frame, plane_image, target.um_per_px),
    )


def _start(ctx: ToolContext, call: _LddmmCall, section: _Prepared) -> dict[str, object]:
    """Start one section's fit as background work; return its row."""
    try:
        work = ctx.start_work([section.sid], _Landing(call, section))
    except (Refused, *ENGINE_FAILURES) as exc:
        payload = (
            exc.payload()
            if isinstance(exc, Refused)
            else {"error": RefusalCode.ENGINE_UNAVAILABLE, "message": str(exc)}
        )
        failed = {key: value for key, value in payload.items() if key != "status"}
        return {"id": section.sid, "status": "error", **failed}
    return {
        "id": section.sid,
        "status": "started",
        "work": work,
        "start": "deformation" if section.source.on_grid else "linear placement",
        "inputs": section.inputs,
        "estimated": None,
    }


def _lddmm(
    ctx: ToolContext,
    machine: engine.Machine,
    sections: list[str],
    section_image: str,
    atlas_image: str,
    kernel_um: float,
    fit: str,
    displacement_um: float,
    sorting: dict[str, Any],
    iterations: int,
    view: bool,
) -> ToolResult:
    if not sections:
        raise ToolError(RefusalCode.BAD_ARGUMENT, "name one to twenty sections")
    if fit not in FITS:
        raise ToolError(RefusalCode.BAD_ARGUMENT, f"fit is one of {', '.join(FITS)}; got {fit!r}")
    spreads, classes = _sorting_settings(sorting)
    settings = {
        "section_image": section_image,
        "atlas_image": atlas_image,
        "mask": "none",
        "kernel_um": kernel_um,
        "fit": fit,
        "displacement_um": displacement_um,
        "sorting": sorting or {"method": "classic"},
        "iterations": iterations,
        "device": machine.device,
    }
    call = _LddmmCall(
        settings=settings,
        fit=engine.LddmmSettings(
            kernel_um=float(kernel_um),
            displacement_um=float(displacement_um),
            iterations=int(iterations),
            refit_transform=fit == "both",
            sorting=spreads,
            mixture_classes=classes,
        ),
        machine=machine,
        atlas_image=atlas_image,
        section_image=section_image,
        folder=_blocks.call_folder(ctx, "lddmm"),
        view=view,
        sorts=sorting.get("method", "classic") != "none",
    )
    rows: list[dict[str, object]] = []
    for name in sections:
        prepared = _prepare(ctx, call, name)
        rows.append(prepared if isinstance(prepared, dict) else _start(ctx, call, prepared))
    return ToolResult({"rows": rows, "settings": settings})


def _target_to_atlas(frame: FitFrame, um: float) -> np.ndarray:
    """4 x 4: target (z, row um, col um) -> atlas um, through the section's linear placement
    and its plane (z along the unit normal)."""
    to_um = np.asarray(frame.plane["native_to_um"], np.float64)
    from_grid = np.linalg.inv(placed_matrix(frame))
    m = to_um @ from_grid
    d_row, d_col = m[:, 1] / um, m[:, 0] / um
    n = np.cross(d_col, d_row)
    n /= np.linalg.norm(n)
    basis = np.eye(4)
    basis[:3, 0], basis[:3, 1], basis[:3, 2], basis[:3, 3] = n, d_row, d_col, m[:, 2]
    return basis


@dataclass(frozen=True)
class _PlaneCall:
    """One ``stalign_plane_search`` call, as every section searches."""

    ctx: ToolContext
    machine: engine.Machine
    folder: Path
    atlas_name: str
    section_image: str
    atlas_image: str
    view: bool


def _search_section(call: _PlaneCall, name: str) -> ToolResult:
    """Search one section's plane: its row and picture; save its record."""
    ctx = call.ctx
    view_of = ctx.section(name)
    sid = view_of.id
    frame, _plane_image, resolved = _frame(ctx, sid, call.atlas_image, call.section_image)
    inputs = _blocks.inputs_block(
        ctx,
        frame,
        section_image=call.section_image,
        atlas_image=call.atlas_image,
        resolved=resolved,
        mask=None,
        brightness=BRIGHTNESS,
    )
    target = _target(frame)
    gw, gh = target.size
    um = target.um_per_px
    request = engine.PlaneRequest(
        atlas=call.atlas_name,
        atlas_image=resolved,
        target=target.image,
        target_rows_um=target.rows_um,
        target_cols_um=target.cols_um,
        start=np.linalg.inv(_target_to_atlas(frame, um)),
        settings=engine.PlaneSettings(),
        machine=call.machine,
    )
    result = _run_engine(partial(engine.run_plane_search, request), call.folder / sid)
    basis = np.linalg.inv(result.affine)
    middle = [(gh - 1) / 2 * um, (gw - 1) / 2 * um]
    centre = basis @ np.array([0.0, *middle, 1.0])
    normal = np.cross(basis[:3, 2], basis[:3, 1])
    plane = ctx.angles_from_plane(normal.tolist(), centre[:3].tolist())
    saved = {"found": plane, "prefit": result.prefit, "device": result.device}
    (call.folder / sid / "plane.json").write_text(
        json.dumps(saved, indent=1, default=float), encoding="utf-8", newline="\n"
    )
    pitch0, yaw0 = (float(v) for v in view_of.angles)
    change = {
        "pitch_deg": round(plane["pitch_deg"] - pitch0, 2),
        "yaw_deg": round(plane["yaw_deg"] - yaw0, 2),
        "position_mm": round(plane["position_mm"] - view_of.position_mm, 3),
    }
    row = {
        "id": sid,
        "status": "ok",
        "inputs": inputs,
        "estimated": None,
        "pitch_deg": round(plane["pitch_deg"], 2),
        "yaw_deg": round(plane["yaw_deg"], 2),
        "position_mm": round(plane["position_mm"], 3),
        "change": change,
        "at_search_edge": _at_search_edge(result.starts, result.best),
    }
    if not call.view:
        return ToolResult(row)
    proposal = {
        "position_mm": plane["position_mm"],
        "angles": (plane["pitch_deg"], plane["yaw_deg"]),
    }
    caption = (
        f"{sid}: atlas at the plane STalign found (pitch "
        f"{row['pitch_deg']}, yaw {row['yaw_deg']}), not written"
    )
    picture = ctx.overlay_picture(sid, proposal=proposal)
    return ToolResult(row, [ctx.caption(picture, caption)])


def _at_search_edge(starts: list[list[float]], best: int | None) -> bool:
    """Tell whether the winning start is at the edge of the search's starts.

    Parameters
    ----------
    starts
        Every start's tilts (degrees) and shift (micrometres).
    best
        The winning start's index; None without one.

    Returns
    -------
    bool
        True when one of its tilts or its shift is the search's largest.
    """
    if not starts or best is None:
        return False
    tilt_a, tilt_b, shift = starts[best]
    tilted = (
        abs(abs(tilt_a) - STARTS.tilt_deg) < SAME_LEVEL
        or abs(abs(tilt_b) - STARTS.tilt_deg) < SAME_LEVEL
    )
    return tilted or abs(abs(shift) - STARTS.shift_um) < SAME_LEVEL


def _plane_search(
    ctx: ToolContext,
    machine: engine.Machine,
    sections: list[str],
    section_image: str,
    atlas_image: str,
    view: bool,
) -> ToolResult:
    if not sections:
        raise ToolError(RefusalCode.BAD_ARGUMENT, "name one to eight sections")
    call = _PlaneCall(
        ctx,
        machine,
        _blocks.call_folder(ctx, "plane"),
        ctx.atlas().name,
        section_image,
        atlas_image,
        view,
    )
    done = map_sections(partial(_search_section, call), list(sections), machine.threads)
    rows = [one.reply for one in done]
    pictures: list[Image.Image] = []
    for one in done:
        pictures.extend(one.pictures)
    ok = [row for row in rows if row["status"] == "ok"]
    groups = {
        "positive_yaw": [r["id"] for r in ok if r["yaw_deg"] > 0],
        "negative_yaw": [r["id"] for r in ok if r["yaw_deg"] < 0],
        "zero_yaw": [r["id"] for r in ok if r["yaw_deg"] == 0],
    }
    summary = {
        "sections_by_yaw_sign": groups,
        "middle_pitch_deg": round(float(np.median([r["pitch_deg"] for r in ok])), 2)
        if ok
        else None,
        "middle_yaw_magnitude_deg": round(float(np.median(np.abs([r["yaw_deg"] for r in ok]))), 2)
        if ok
        else None,
    }
    return ToolResult(
        {
            "rows": rows,
            "stack": summary,
            "settings": {"section_image": section_image, "atlas_image": atlas_image},
        },
        pictures,
    )


def tools(config: Mapping[str, object] | None = None) -> list[ToolSpec]:
    """Return the plugin's tools.

    Parameters
    ----------
    config
        The plugin's entry of ``plugin_config``; None gives the defaults.

    Returns
    -------
    list of ToolSpec
        ``stalign_lddmm`` and ``stalign_plane_search``.

    Raises
    ------
    PluginError
        For an option the plugin does not take.
    """
    machine = Options.from_config(config).machine
    meta = {
        "method": "STalign 2.0.0",
        "paper": "Clifton et al. 2023, Nat Commun",
        "commit": "8a84bf1f",
        "engine": "langslice.engines.stalign",
    }

    def lddmm(
        ctx: ToolContext,
        sections: list[str],
        section_image: str = "preprocessed",
        atlas_image: str = "nissl",
        mask: str = "none",
        kernel_um: float = 500.0,
        fit: str = "warp",
        displacement_um: float = 94.0,
        sorting: dict[str, Any] | None = None,
        iterations: int = 2000,
        view: bool = True,
    ) -> ToolResult:
        _blocks.refuse_mask("stalign_lddmm", mask, "sorts the section's pixels itself")
        return _lddmm(
            ctx,
            machine,
            sections,
            section_image,
            atlas_image,
            kernel_um,
            fit,
            displacement_um,
            dict(sorting or {}),
            iterations,
            view,
        )

    def plane(
        ctx: ToolContext,
        sections: list[str],
        section_image: str = "preprocessed",
        atlas_image: str = "nissl",
        view: bool = True,
    ) -> ToolResult:
        return _plane_search(ctx, machine, sections, section_image, atlas_image, view)

    return [
        engine_tool(
            "stalign_lddmm",
            stalign_lddmm,
            lddmm,
            description=_lddmm_description,
            version="1",
            meta=meta,
        ),
        engine_tool(
            "stalign_plane_search",
            stalign_plane_search,
            plane,
            description=_plane_search_description,
            version="1",
            meta=meta,
        ),
    ]


__all__ = ["tools"]
