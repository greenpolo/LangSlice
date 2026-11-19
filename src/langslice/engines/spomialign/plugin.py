"""SPOmiAlign as a LangSlice tool: the adapter, run in LangSlice's process.

It is a tool of every job (its row of the tool table names
``langslice.engines.spomialign.plugin:tools``). Option (``plugin_config``):
``device`` (the torch device: ``auto``, the first GPU when torch sees one,
else the CPU; ``cpu``, ``cuda`` or a numbered GPU); any other key is refused.
The method itself runs in this process, in ``engine.py`` and ``matching.py``,
on RoMa (``romatch``, LangSlice's ``engines`` extra), imported only when a call
runs; a call without it is refused ``ENGINE_UNAVAILABLE``. Its declaration is
its row of the tool table (``langslice.plugins.declared``).

Each call puts the section on SPOmiAlign's canvas, runs the engine on (atlas
image, section), and maps the answer back: the linear part through
``transform_from_matrix``, the full map through ``deformation_from_map``.
Canvas: for rigid, affine and affine_warp the atlas plane's grid, upsampled
to about :data:`CANVAS_EDGE` px, with the section resampled onto it through
its current transform (as the published user's hand-picked plate and
rotation did); for warp on a section that holds a deformation, the fit grid,
with the atlas resampled onto it through that deformation
(``FitFrame.applied_map``), so the warp starts on top of a landmark warp, as
``stalign_lddmm`` does. A new transform clears a deformation, so rigid,
affine and affine_warp are refused on a section that holds one
(``PRECONDITION``).
"""

from __future__ import annotations

import csv
import json
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal

import cv2
import numpy as np
from PIL import Image, ImageDraw

from langslice.core.failures import ENGINE_FAILURES
from langslice.engines import _blocks
from langslice.engines._devices import AUTO
from langslice.engines._options import device_option, known_options
from langslice.engines._pixels import outlined, sample_pixels, slide_colour
from langslice.engines.spomialign import engine, matching
from langslice.plugin_api import (
    FitFrame,
    RefusalCode,
    SectionView,
    ToolContext,
    ToolError,
    ToolResult,
    ToolsOn,
    ToolSpec,
)
from langslice.plugins.declared import engine_tool

OPTIONS = ("device",)
MISSING = (
    "SPOmiAlign cannot run here: its matcher, RoMa (the romatch package), "
    "is not installed; install LangSlice's engines extra: "
    'pip install "langslice[engines]". Use another method, or ask the user to install it.'
)
CANVAS_EDGE = 1536
FITS = ("transform", "warp", "both")
WRITES = {
    engine.RIGID: ("affine",),
    engine.AFFINE: ("affine",),
    engine.WARP: ("deformation",),
    engine.AFFINE_WARP: ("affine", "deformation"),
}
ENGINE_VERSION = "SPOmiAlign c7f8b7a"
MATCHER = "RoMa roma_outdoor, coarse resolution 560"
MIN_MATCHES = 3
BRIGHTNESS = "8-bit, Nissl rendered dark on white"
NISSL = _blocks.NISSL
WORK_W, WORK_H = matching.W, matching.H


@dataclass(frozen=True)
class AtlasWindow:
    """Hold the percentile window the atlas plate is shown through.

    Parameters
    ----------
    low, high
        The percentiles of the atlas image inside the brain that map to
        black ink and to white.
    """

    low: float = 0.5
    high: float = 99.5


@dataclass(frozen=True)
class MatchesPicture:
    """Hold the layout of the matches picture saved with a call.

    Parameters
    ----------
    drawn_pairs
        How many of the most confident pairs are drawn.
    gap_px
        The white strip between the two canvases.
    long_edge
        The longest side of the saved picture.
    cross_colour
        The colour of the crosses marking passed-over pairs.
    """

    drawn_pairs: int = 300
    gap_px: int = 16
    long_edge: int = 2400
    cross_colour: tuple[int, int, int] = (220, 30, 30)


ATLAS_WINDOW = AtlasWindow()
MATCHES_PICTURE = MatchesPicture()


def mode_of(fit: str, rigid: bool) -> str:
    """The engine mode of a call's *fit* and *rigid*: ``rigid`` or ``affine``
    for the transform, ``warp``, or ``affine_warp`` for both."""
    if fit == "transform":
        return engine.RIGID if rigid else engine.AFFINE
    return engine.WARP if fit == "warp" else engine.AFFINE_WARP


def _spomialign_description(on: ToolsOn) -> str:
    steps = on.listed(
        [
            ("position_sections", "place the section with position_sections"),
            (
                "interactive_transform",
                "set its flip, quarter turn and rough transform with interactive_transform",
            ),
        ]
    )
    rough_start = f": {steps} first" if steps else ""
    damage_first = on.say(
        "mark_damage",
        "; where tissue is missing or lies folded over other anatomy, mark those regions "
        "with mark_damage first (tissue that is intact but shifted is not damage)",
    )
    landmark_kept = on.say(
        "set_landmarks", ", a landmark warp included, so the landmark move is kept"
    )
    made_images = on.say("make_image", ", or an image made with make_image")
    mask_made = on.say("make_mask", " (made with make_mask)")
    opened_by = on.say("read_file", ", which read_file opens")
    return f"""\
Register sections to the atlas with SPOmiAlign, on their current registration.

SPOmiAlign (Wang et al. 2026, Briefings in Bioinformatics) was built to align
spatial-omics sections (transcriptomics, proteomics, H&E) to each other and to
the Allen Nissl plates. Instead of comparing brightness it matches points with
RoMa (Edstedt et al. 2024), a dense feature matcher pretrained on DINOv2
features, so only the tissue's shape and structure need agree, not its stain.

SPOmiAlign matches distinctive points between a section image (by
name) and an atlas image with a pretrained dense matcher (RoMa), weights
the matches toward tissue edges, keeps the most confident (matches), and
fits the chosen transform to the pairs; it compares no intensities pixel by
pixel and runs no iterations; a call takes under a minute. It starts from
the section's current placement and does not search from scratch{rough_start}. The section's \
marked damage regions are left
out automatically{damage_first}.

fit chooses what is fitted:
- "transform": the section's transform: rotation, shift, scale and shear
  (an affine), or with rigid rotation and shift alone, the section keeping
  its size. Written as the section's transform.
- "warp": a smooth local deformation on top of the registration the section
  holds (its transform, and its deformation when it has one{landmark_kept}); the transform
  stays. Written as the section's deformation.
- "both": an affine, then a warp on top of it (SPOmiAlign's own default).
  Writes both.
A new transform clears the section's deformation, so "transform" and
"both" are refused on a section that holds one: fit "warp" on it instead.

The warp is SPOmiAlign's smoothed match field: each pair's displacement,
filled in between the pairs and smoothed by a Gaussian of smoothness_um.
Where the section and the atlas image share little tissue (a damaged or
partial section), the affine follows wrong matches, because it weighs
every pair equally and rejects none; the sign is an atlas shrunk off the
tissue in the returned overlay. For such a section, fit "transform" with
rigid, check the returned overlay, then fit "warp". The warp bends whole
regions and does not refine individual borders.

SPOmiAlign reads one image pair per section, nothing else, and has no
weighting: it estimates no mask of its own, and a section mask can only
drop matches. With a mask named, every match whose end on the section lies
where the mask is below one half is dropped and the next most confident
take their place; the mask is approximated by dropping matches, not
applied to the pixels. The atlas side's marked damage is dropped the same
way. The section is read as 8-bit RGB with the atlas image drawn as a
brightfield Nissl, dark on white.

Each call is one undoable write.

Args:
    sections: One to twenty filenames (the extension may be left off).
    fit: "transform", "warp" or "both".
    rigid: True fits the transform as a rotation and a shift alone (fit
        "transform" only). Default false.
    section_image: The section image to match, by name: "preprocessed", a raw \
channel{made_images}. Default "preprocessed".
    atlas_image: One of the atlas's images (the job statement lists
        them: "template" and any other reference images the atlas ships,
        by their names), or "nissl", the atlas's Nissl (its template when
        it ships none). Default "nissl"; the reply names the image used.
    mask: "none", or the name of a mask of each section{mask_made}; matches ending where it is \
below one half
        are dropped. Default "none".
    smoothness_um: The width (sigma) of the Gaussian that smooths the
        warp's displacement field, in micrometres of the atlas; larger
        bends broader regions as one. Default 150.
    matches: How many of the most confident matched pairs the fit uses.
        Default 5000 (SPOmiAlign's).
    view: True returns the picture described below; false returns none.

Returns:
    Per section what was fitted and on what, what the fit used (inputs:
    the section image, the atlas image as asked and as used, the mask, the
    damage, the brightness scaling), what SPOmiAlign ran at
    (engine_settings: the matcher, its working grid and micrometres per
    pixel, the smoothing in its pixels, the pairs), the transform written
    (rotation_deg, scale_x, scale_y, shear, translate_x_mm,
    translate_y_mm about the canvas centre) when one was, for a warp its
    displacement (median and max, mm), fold fraction and plausibility
    flags, the pairs used and dropped, and the folder in the job holding
    the call's files (the two images SPOmiAlign saw, its settings and the
    matched pairs as a table and a picture{opened_by}); and a picture of each section under its \
new registration
    with the atlas borders on it. SPOmiAlign estimates no weighting, so
    estimated is null."""


def spomialign(
    sections: list[str],
    fit: Literal["transform", "warp", "both"],
    rigid: bool = False,
    section_image: str = "preprocessed",
    atlas_image: str = "nissl",
    mask: str = "none",
    smoothness_um: float = 150.0,
    matches: int = 5000,
    view: bool = True,
) -> dict[str, Any]: ...


@dataclass(frozen=True)
class Options:
    """Hold the plugin's options (``plugin_config``).

    Parameters
    ----------
    device
        The torch device: ``auto`` (the first GPU when torch sees one, else
        the CPU), ``cpu``, ``cuda`` or a numbered GPU.
    """

    device: str = AUTO

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
        given = known_options("SPOmiAlign", config, OPTIONS)
        return cls(device=device_option(given.get("device", cls.device)))


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
class _Pair:
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


def _inputs(
    frame: FitFrame,
    plane_image: np.ndarray,
    *,
    weights: np.ndarray | None,
    on_grid: bool,
    smoothness_um: float,
) -> _Pair:
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
    return _Pair(
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


def _engine(options: Options, mode: str, matches: int, pairs: list[_Pair]) -> engine.EngineRun:
    """Run the engine on every section's pair in this process (one model load)."""
    absent = matching.weights_missing()
    try:
        return engine.fit_pairs(
            [pair.engine_pair() for pair in pairs],
            mode=mode,
            top_k=matches,
            device_name=options.device,
        )
    except ENGINE_FAILURES as error:
        missing = [name for name in matching.weights_missing() if name in absent]
        if missing:
            raise ToolError(
                RefusalCode.ENGINE_UNAVAILABLE,
                "SPOmiAlign's matcher weights are not in torch's "
                "hub cache and could not be downloaded: "
                + ", ".join(matching.WEIGHTS[name] for name in missing)
                + ". Use another method, or ask the user to install it.",
            ) from error
        first = str(error).strip().splitlines()[:1] or ["no message"]
        raise ToolError(
            RefusalCode.FIT_FAILED, f"SPOmiAlign did not run: {type(error).__name__}: {first[0]}"
        ) from error


def gpu_words(waited_s: float, retries: int) -> dict[str, Any]:
    """What a row says of the machine's GPU queue, as fireants_syn's rows say it:
    ``gpu_wait_s`` from a second's wait, ``retried`` when the GPU ran out of memory."""
    out: dict[str, Any] = {}
    if waited_s >= 1.0:
        out["gpu_wait_s"] = round(waited_s)
    if retries:
        times = "once" if retries == 1 else f"{retries} times"
        out["retried"] = (
            f"The GPU ran out of memory {times} (another process held it); "
            "the fit waited and ran again from the start, so this result is "
            "whole."
        )
    return out


def _to_native(pair: _Pair, canvas_xy: np.ndarray) -> np.ndarray:
    """Target-canvas points (2 x N) as native atlas px (2 x N)."""
    grid = (pair.from_canvas @ np.vstack([canvas_xy, np.ones(canvas_xy.shape[1])]))[:2]
    if pair.applied is None:
        return grid
    return sample_pixels(pair.applied, grid.T).T


def _full_map(frame: FitFrame, pair: _Pair, matrix: np.ndarray, residual: np.ndarray) -> np.ndarray:
    """Native atlas px of every fit-grid pixel under SPOmiAlign's full map.

    A fit-grid pixel u sits at canvas px c = to_canvas u; SPOmiAlign takes c to
    M c + R(c) on the target canvas (R zero off the canvas, as ITK's field is), which
    :func:`_to_native` reads as native atlas px."""
    gh, gw = _blocks.section_pixels(frame).shape[:2]
    ys, xs = np.mgrid[0:gh, 0:gw]
    u = np.stack([xs.ravel(), ys.ravel(), np.ones(xs.size)]).astype(np.float64)
    c = (pair.to_canvas @ u)[:2]
    linear = engine.homogeneous(matrix)
    moved = linear[:2, :2] @ c + linear[:2, 2:3]
    cx = c[0].reshape(gh, gw).astype(np.float32)
    cy = c[1].reshape(gh, gw).astype(np.float32)
    for axis in range(2):
        moved[axis] += cv2.remap(
            np.ascontiguousarray(residual[..., axis]),
            cx,
            cy,
            interpolation=cv2.INTER_LINEAR,
            borderMode=cv2.BORDER_CONSTANT,
            borderValue=0.0,
        ).ravel()
    return _to_native(pair, moved).T.reshape(gh, gw, 2)


def _pairs_picture(
    pair: _Pair,
    used: np.ndarray,
    dropped: np.ndarray,
    path: Path,
) -> None:
    """The two canvases side by side, the most confident pairs as dots of one colour on both
    (coloured by where they sit on the section) and the pairs passed over in the damage zone
    as red crosses on the atlas (a file in the job, not sent)."""
    left, right = pair.target, pair.source
    gap = MATCHES_PICTURE.gap_px
    image = Image.fromarray(
        np.concatenate([left, np.full((left.shape[0], gap, 3), 255, np.uint8), right], axis=1)
    )
    draw = ImageDraw.Draw(image)
    shift = left.shape[1] + gap
    r = max(3, left.shape[1] // 250)
    height = right.shape[0]
    shown = MATCHES_PICTURE.drawn_pairs
    for ax, ay, sx, sy, _ in used[:shown]:
        hue = np.uint8(170 * sy / max(height, 1)), np.uint8(255), np.uint8(230)
        rgb = cv2.cvtColor(np.array([[hue]], np.uint8), cv2.COLOR_HSV2RGB)[0, 0]
        colour = tuple(int(v) for v in rgb)
        draw.ellipse([ax - r, ay - r, ax + r, ay + r], fill=colour, outline="black")
        draw.ellipse([sx + shift - r, sy - r, sx + shift + r, sy + r], fill=colour, outline="black")
    cross = MATCHES_PICTURE.cross_colour
    for ax, ay, *_ in dropped:
        draw.line([(ax - r, ay - r), (ax + r, ay + r)], fill=cross, width=2)
        draw.line([(ax - r, ay + r), (ax + r, ay - r)], fill=cross, width=2)
    draw.text(
        (6, 6),
        f"atlas (left) and section (right) as SPOmiAlign saw them; the "
        f"{min(shown, len(used))} most confident of {len(used)} pairs, each a dot of "
        f"one colour on both; red crosses: "
        f"{len(dropped)} passed over (marked damage or the mask)",
        fill="black",
    )
    edge = MATCHES_PICTURE.long_edge
    image.thumbnail((edge, edge))
    image.save(path)


def _files(here: Path, pair: _Pair, got: engine.EngineAnswer, settings: Mapping[str, Any]) -> None:
    """Save the call's files for one section: the two canvases, the settings and the pairs."""
    here.mkdir(parents=True, exist_ok=True)
    Image.fromarray(outlined(pair.target, ~pair.keep)).save(here / "atlas.png")
    Image.fromarray(pair.source).save(here / "section.png")
    (here / "settings.json").write_text(
        json.dumps(settings, indent=1, default=str), encoding="utf-8", newline="\n"
    )
    none = np.zeros((0, 5), np.float32)
    used = none if got.used is None else got.used
    dropped = none if got.dropped_pairs is None else got.dropped_pairs
    with open(here / "matches.csv", "w", encoding="utf-8", newline="") as fh:
        writer = csv.writer(fh)
        writer.writerow(["atlas_x", "atlas_y", "section_x", "section_y", "confidence", "dropped"])
        for rows, flag in ((used, 0), (dropped, 1)):
            for row in rows:
                writer.writerow([*(f"{v:.2f}" for v in row[:4]), f"{row[4]:.4g}", flag])
    _pairs_picture(pair, used, dropped, here / "matches.png")


def _refusal(view_of: SectionView, frame: FitFrame, mode: str) -> dict[str, object] | None:
    """Why the engine *mode* cannot start on this section, before the engine runs."""
    sid = view_of.id
    frozen = [prop for prop in WRITES[mode] if prop in view_of.frozen]
    if frozen:
        refusal = ToolError(
            RefusalCode.FROZEN, f"the section's {' and '.join(frozen)} may not be changed"
        )
        return {"id": sid, **refusal.payload()}
    if frame.placed_atlas_to_section is None:
        refusal = ToolError(
            RefusalCode.PRECONDITION,
            "SPOmiAlign starts from the section's placement: give it a transform first",
        )
        return {"id": sid, **refusal.payload()}
    if mode != engine.WARP and view_of.deformation:
        refusal = ToolError(
            RefusalCode.PRECONDITION,
            "this fit would write a new transform, which clears the "
            'section\'s deformation; fit "warp" to build on it',
        )
        return {"id": sid, **refusal.payload()}
    return None


@dataclass(frozen=True)
class _Call:
    """Hold what every section of one ``spomialign`` call shares.

    Parameters
    ----------
    mode
        The engine's mode (``rigid``, ``affine``, ``warp`` or ``affine_warp``).
    section_image, atlas_image, mask
        The images and mask asked for.
    smoothness_um
        The warp's smoothing, um.
    matches
        How many of the most confident pairs are used.
    view
        Whether pictures are sent.
    options
        The plugin's options.
    settings
        The call's settings, as the reply and the deformation record them.
    """

    mode: str
    section_image: str
    atlas_image: str
    mask: str
    smoothness_um: float
    matches: int
    view: bool
    options: Options
    settings: dict[str, Any]


@dataclass(frozen=True)
class _Ready:
    """Hold one section ready to match.

    Parameters
    ----------
    sid
        The section's filename.
    frame
        Its fit frame.
    pair
        The pair SPOmiAlign sees.
    digest
        Its registration's digest when the call began.
    inputs
        The ``inputs`` block of its reply.
    resolved
        The atlas image that was used.
    """

    sid: str
    frame: FitFrame
    pair: _Pair
    digest: str
    inputs: dict[str, Any]
    resolved: str


def _call(
    options: Options,
    fit: str,
    rigid: bool,
    section_image: str,
    atlas_image: str,
    mask: str,
    smoothness_um: float,
    matches: int,
    view: bool,
    sections: list[str],
) -> _Call:
    """Check a call's arguments and gather them."""
    if fit not in FITS:
        raise ToolError(RefusalCode.BAD_ARGUMENT, f"fit is one of {', '.join(FITS)}; got {fit!r}")
    if rigid and fit != "transform":
        raise ToolError(
            RefusalCode.BAD_ARGUMENT,
            "rigid fits the transform alone: SPOmiAlign's warp "
            "follows an affine or the registration the section holds; give fit "
            '"transform" with rigid, or leave rigid false',
        )
    if not sections:
        raise ToolError(RefusalCode.BAD_ARGUMENT, "name one to twenty sections")
    if int(matches) < MIN_MATCHES:
        raise ToolError(RefusalCode.BAD_ARGUMENT, f"matches is at least {MIN_MATCHES}")
    if not float(smoothness_um) > 0:
        raise ToolError(RefusalCode.BAD_ARGUMENT, "smoothness_um is above 0")
    settings = {
        "fit": fit,
        "rigid": bool(rigid),
        "section_image": section_image,
        "atlas_image": atlas_image,
        "mask": mask,
        "smoothness_um": float(smoothness_um),
        "matches": int(matches),
    }
    return _Call(
        mode=mode_of(fit, bool(rigid)),
        section_image=section_image,
        atlas_image=atlas_image,
        mask=mask,
        smoothness_um=float(smoothness_um),
        matches=int(matches),
        view=view,
        options=options,
        settings=settings,
    )


def _read_sections(
    ctx: ToolContext, call: _Call, sections: list[str]
) -> tuple[list[dict[str, object]], list[_Ready]]:
    """Read every section's pair; a section that cannot start has its refusal as a row."""
    rows: list[dict[str, object]] = []
    ready: list[_Ready] = []
    for name in sections:
        view_of = ctx.section(name)
        frame = _blocks.frame_for(ctx, view_of.id, call.atlas_image, call.section_image)
        plane_image, resolved = _blocks.atlas_plane(frame, call.atlas_image)
        refused = _refusal(view_of, frame, call.mode)
        if refused:
            rows.append(refused)
            continue
        held = _blocks.section_mask(ctx, view_of.id, call.mask)
        on_grid = call.mode == engine.WARP and frame.applied_map is not None
        pair = _inputs(
            frame,
            plane_image,
            weights=None if held is None else np.asarray(held.weights, np.float32),
            on_grid=on_grid,
            smoothness_um=call.smoothness_um,
        )
        inputs = _blocks.inputs_block(
            ctx,
            frame,
            section_image=call.section_image,
            atlas_image=call.atlas_image,
            resolved=resolved,
            mask=held,
            brightness=BRIGHTNESS,
        )
        digest = ctx.section_inputs(view_of.id, deformation=on_grid)
        ready.append(_Ready(view_of.id, frame, pair, digest, inputs, resolved))
    return rows, ready


def _land(
    ctx: ToolContext,
    call: _Call,
    folder: Path,
    section: _Ready,
    got: engine.EngineAnswer,
    wait_s: float,
) -> tuple[dict[str, object], Image.Image | None]:
    """Save one section's files, write its registration and give its row and picture."""
    sid, pair, inputs = section.sid, section.pair, section.inputs
    mode = call.mode
    here = folder / Path(sid).stem
    start = "the deformation it held" if pair.on_grid else "its linear placement"
    by_mask = got.dropped_by_mask
    in_damage = got.dropped - by_mask
    _files(
        here,
        pair,
        got,
        {
            **call.settings,
            "inputs": inputs,
            "atlas_image_resolved": section.resolved,
            "smoothing_px": pair.smoothing_px,
            "start": start,
            "canvas_px": pair.target.shape[:2],
            "device": call.options.device,
            "pairs_used": got.pairs,
            "pairs_dropped_in_damage": in_damage,
            "pairs_dropped_by_mask": by_mask,
        },
    )
    relative = _blocks.job_path(ctx, here)
    warps = mode in (engine.WARP, engine.AFFINE_WARP)
    matrix, residual = got.matrix, got.residual
    if got.error or matrix is None or (warps and residual is None):
        failed = ToolError(RefusalCode.FIT_FAILED, got.error)
        return {**failed.payload(section_id=sid), "files": relative}, None
    what = mode.replace("_", " and ") if mode != engine.WARP else f"warp on {start}"
    note = (
        f"SPOmiAlign {what}, atlas {section.resolved}, {got.pairs} pairs, "
        f"{in_damage} in marked damage dropped"
        + (f", {by_mask} outside the mask dropped" if call.mask != "none" else "")
    )
    engine_settings = {
        "matcher": MATCHER,
        "mode": mode,
        "working_grid_px": [WORK_W, WORK_H],
        "working_um": round(float(pair.work_um), 3),
        "smoothness_um": call.smoothness_um,
        "smoothing_sigma_px": round(float(pair.smoothing_px), 3),
        "matches": call.matches,
        "canvas_px": list(pair.target.shape[:2]),
    }
    row: dict[str, Any] = {
        "id": sid,
        "status": "ok",
        "fitted": note,
        "files": relative,
        "inputs": inputs,
        "estimated": None,
        "engine_settings": engine_settings,
        **gpu_words(wait_s, got.oom_retries),
    }
    if call.mask != "none":
        row["mask_note"] = "approximated by dropping matches"
    if mode != engine.WARP:
        to_native = (pair.from_canvas @ engine.homogeneous(matrix) @ pair.to_canvas)[:2]
        proposal = ctx.transform_from_matrix(sid, to_native, note=note)
        written = ctx.set_transforms([proposal.row()], expect={sid: section.digest})
        if sid not in written.written:
            return {**written.rows[0], "id": sid}, None
        row["transform"] = proposal.knobs
    if warps and residual is not None:
        native = _full_map(section.frame, pair, matrix, residual)
        warp = ctx.deformation_from_map(
            sid,
            native,
            note=note,
            meta={"inputs": inputs, "estimated": None},
            start="current" if pair.on_grid else "linear",
            settings=call.settings,
            engine_settings=engine_settings,
            engine_version=ENGINE_VERSION,
        )
        written = ctx.set_deformations(
            [warp.row()], expect={sid: section.digest} if mode == engine.WARP else None
        )
        if sid not in written.written:
            return {**written.rows[0], "id": sid}, None
        row.update(warp.summary)
    picture = ctx.caption(ctx.overlay_picture(sid), f"{sid}: {note}") if call.view else None
    return row, picture


def _run(
    ctx: ToolContext,
    options: Options,
    sections: list[str],
    fit: str,
    rigid: bool,
    section_image: str,
    atlas_image: str,
    mask: str,
    smoothness_um: float,
    matches: int,
    view: bool,
) -> ToolResult:
    call = _call(
        options,
        fit,
        rigid,
        section_image,
        atlas_image,
        mask,
        smoothness_um,
        matches,
        view,
        sections,
    )
    rows, ready = _read_sections(ctx, call, sections)
    if not ready:
        return ToolResult({"rows": rows, "settings": call.settings})
    folder = _blocks.call_folder(ctx, call.mode)
    run = _engine(options, call.mode, call.matches, [section.pair for section in ready])
    pictures: list[Image.Image] = []
    with ctx.writing():
        for section, got in zip(ready, run.answers, strict=True):
            row, picture = _land(ctx, call, folder, section, got, run.gpu_wait_s)
            rows.append(row)
            if picture is not None:
                pictures.append(picture)
    return ToolResult({"rows": rows, "settings": call.settings}, pictures)


def tools(config: Mapping[str, object] | None = None) -> list[ToolSpec]:
    """Return the plugin's tool.

    Parameters
    ----------
    config
        The plugin's entry of ``plugin_config``; None gives the defaults.

    Returns
    -------
    list of ToolSpec
        ``spomialign``.

    Raises
    ------
    PluginError
        For an option the plugin does not take.
    """
    options = Options.from_config(config)

    def handler(
        ctx: ToolContext,
        sections: list[str],
        fit: str,
        rigid: bool = False,
        section_image: str = "preprocessed",
        atlas_image: str = NISSL,
        mask: str = "none",
        smoothness_um: float = 150.0,
        matches: int = 5000,
        view: bool = True,
    ) -> ToolResult:
        if not matching.is_installed():
            raise ToolError(RefusalCode.ENGINE_UNAVAILABLE, MISSING)
        return _run(
            ctx,
            options,
            sections,
            fit,
            rigid,
            section_image,
            atlas_image,
            mask,
            smoothness_um,
            matches,
            view,
        )

    return [
        engine_tool(
            "spomialign",
            spomialign,
            handler,
            description=_spomialign_description,
            version="2",
            meta={
                "method": "SPOmiAlign",
                "paper": "Wang et al. 2026, Brief Bioinform",
                "commit": "c7f8b7a",
                "engine": "langslice.engines.spomialign",
            },
        )
    ]
