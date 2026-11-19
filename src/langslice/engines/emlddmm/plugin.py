"""emlddmm as LangSlice tools: the adapter, run in LangSlice's process.

It is a tool of every job (its row of the tool table names
``langslice.engines.emlddmm.plugin:tools``). Options (``plugin_config``):
``device`` (``auto``: the GPU when there is one, else the CPU), ``dtype``
(``float64``), ``section_um`` (58.88), ``section_weight`` (``gap`` or a
thickness in um), ``memory_limit_gb`` (24); any other key is refused. emlddmm
(Apache-2.0, vendored as ``langslice._vendor.emlddmm``) runs in LangSlice's
process: ``engine.py`` imports it when a tool runs. Its declaration is its rows of
the tool table (``langslice.plugins.declared``).

Two tools:

- ``emlddmm_stack``: emlddmm's joint fit of the placed sections against the 3-D atlas, one piece
  of background work over the whole stack, on the stack's current plane: each section's in-plane
  rigid motion, the stack's shared scale and shear, and one 3-D warp read in each section's own
  plane (``out_of_plane=False``, ``lock_slice_axis``), landing as one undo step;
- ``emlddmm_plane``: the same fit with the plane free and no warp, proposing the stack's cutting
  angles and spacing, nothing written.

Frames. The stack frame has z along the planes' normal and (row, column) along the displayed
atlas rows and columns, origin at the most anterior section's plane centre. Section k sits at its
distance from that plane, snapped to a slot grid ``slot_um`` apart along the slicing axis (the
agent's choice, no default; emlddmm needs evenly spaced slices, an empty slot carries no data,
``W0`` 0). Each section enters already placed: its section image (by name), grey, read as it is in
[0, 1], area-averaged to ``section_um``, resampled through the
section's current registration (its
deformation included, ``FitFrame.applied_map``) onto the stack's in-plane grid, which is the union
of the sections' footprints. Marked damage (with the native margin) and everything outside the
image get weight 0. The background mean muB is 1 on a white slide (the median of the image
border over 0.5: brightfield, with muA 0) or 0 on a black one (fluorescence). The atlas
image is read from BrainGlobe at 50 um
(``_atlas_images.py``), divided by its mean.

Files, under ``plugins/<tool>/calls/<n>_<tool>/``: ``settings.json`` (asked, chosen, plugin
options, what emlddmm chose, whether its energy record was captured), ``sections.csv`` (the
stack's per-section results; the plane's are in ``A.json``), ``A.json`` (emlddmm's own matrices),
``energy.csv`` and ``energy.png``, ``images/`` (each section as the engine saw it, its final
sorting and the atlas under the fit), ``weights.npz`` and ``log.txt``
(what emlddmm printed). Replies name them; their numbers never go into the
reply.
"""

from __future__ import annotations

import csv
import json
import math
from collections.abc import Mapping
from dataclasses import dataclass, field, replace
from pathlib import Path
from typing import Any, Literal, TypeVar

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont

from langslice.core.failures import ENGINE_FAILURES
from langslice.core.refusal import Refused
from langslice.engines import _blocks
from langslice.engines._devices import resolve_device
from langslice.engines._options import device_option, known_options
from langslice.engines._pixels import (
    area_average,
    placed_matrix,
    placed_outline,
    sample_pixels,
    slide_level,
)
from langslice.engines._sorting import SORTING_KEYS, Sorting
from langslice.engines.emlddmm import engine
from langslice.job.settings.tool_table import TOOLS
from langslice.plugin_api import (
    DeformationProposal,
    FitFrame,
    PluginError,
    RefusalCode,
    SectionView,
    ToolContext,
    ToolError,
    ToolResult,
    ToolsOn,
    ToolSpec,
    flag_words,
)
from langslice.plugins.declared import engine_tool

Array = np.ndarray
Result = TypeVar("Result")
OPTIONS = ("device", "dtype", "section_um", "section_weight", "memory_limit_gb")
DTYPES = ("float32", "float64")
FITS = ("transform", "warp", "both")
ENGINE_VERSION = "emlddmm 9c4dc80 + e426ece + 6ff62b0"
NEEDS_TRANSFORM = "needs a transform first"
ANGLE_TOLERANCE_DEG = 1.0
LATTICE_PX = 8
UM_PER_MM = 1000.0
OVERLAYS = TOOLS["emlddmm_stack"].limits["pictures"]
SCALE_FIRST = 1.0
SCALES_HALF, SCALES_QUARTER = 2, 4
MIN_ITERATIONS = 4
GETS = {
    "deformation": "its transform kept, the rigid change folded into its deformation",
    "none": "nothing (its transform or deformation is frozen)",
}


@dataclass(frozen=True)
class PublishedExample:
    """Hold the constants of emlddmm's published mouse serial-section example.

    Each section is weighed as the gap it stands for: the warp's sigma times
    sqrt(k), the warp's and the affine's steps times k, with k the section's
    thickness over the slot spacing, and the slice step times the example's
    slice thickness over the slot spacing.

    Parameters
    ----------
    iterations
        The steps at its three scales.
    kernel_um
        The warp's kernel width, um.
    warp_freedom
        The warp's freedom (``sigmaR`` before the weighing).
    velocity_size_um
        The velocity field's grid spacing (``dv``), um.
    warp_step
        The warp's gradient step (``ev``).
    affine_step
        The stack affine's gradient step (``eA``).
    slice_step
        Each slice's in-plane gradient step (``eA2d``), at ``slice_um``.
    slice_um
        The example's slice spacing, um.
    no_warp
        A warp start past every step: the warp stays off.
    """

    iterations: tuple[int, int, int] = (100, 50, 25)
    kernel_um: float = 500.0
    warp_freedom: float = 10000.0
    velocity_size_um: float = 1000.0
    warp_step: float = 1e-2
    affine_step: float = 1e7
    slice_step: float = 1e5
    slice_um: float = 20.0
    no_warp: int = 10**6


@dataclass(frozen=True)
class Memory:
    """Hold the memory model of the stack's arrays and the slot search.

    Parameters
    ----------
    gb_per_megavoxel
        The memory one million voxels of the stack take at float64.
    search_max_um, search_step_um
        The slot spacings tried, to say which one would fit.
    """

    gb_per_megavoxel: float = 0.4
    search_max_um: float = 5000.0
    search_step_um: float = 5.0


@dataclass(frozen=True)
class Geometry:
    """Hold the numbers of the stack grid and of reading a section onto it.

    Parameters
    ----------
    edge_samples
        The points sampled along each side of a section's fit grid to find
        the stack's footprint.
    fixed_point_steps
        The iterations that invert a deformation's map.
    margin_pixels
        The empty margin around the sections' footprints, in grid pixels.
    byte_range
        A section image whose maximum is above this is read as 0-255, else
        as 0-1.
    white_slide
        The slide's level over its range above which it is white.
    """

    edge_samples: int = 24
    fixed_point_steps: int = 20
    margin_pixels: int = 2
    byte_range: float = 1.5
    white_slide: float = 0.5


@dataclass(frozen=True)
class SheetLayout:
    """Hold the layout of a section's sheet of the images the engine saw.

    Parameters
    ----------
    display_max
        The brightness shown as white for the section as sent.
    percentiles
        The atlas image's window, percentiles inside the tissue.
    enlarge
        How many times the sheet is enlarged.
    band
        The caption band's height, px.
    gap
        The white strip between the three tiles, px.
    font
        The caption font's size.
    """

    display_max: float = 1.6
    percentiles: tuple[float, float] = (1.0, 99.0)
    enlarge: int = 3
    band: int = 22
    gap: int = 4
    font: int = 11


@dataclass(frozen=True)
class EnergyPlot:
    """Hold the size of the energy plot, px, and its font.

    Parameters
    ----------
    height, width
        The image's size.
    pad
        The margin around the lines.
    font
        The label font's size.
    """

    height: int = 260
    width: int = 720
    pad: int = 34
    font: int = 9


EXAMPLE = PublishedExample()
MEMORY = Memory()
GEOMETRY = Geometry()
SHEET = SheetLayout()
PLOT = EnergyPlot()
PUBLISHED_SORTING: dict[str, Any] = {}


def _stack_description(on: ToolsOn) -> str:
    local_mismatch = on.say_any(
        ("fireants_syn", "landmark_triangulation", "trace_borders"),
        "; for a local mismatch on one section (a ventricle, a displaced piece) "
        "use a per-section warp",
    )
    landmark_kept = on.say("set_landmarks", ", so a landmark move is kept")
    damage_first = on.say("mark_damage", "; mark them with mark_damage first")
    mask_from_sorting = on.say(
        "make_mask", ", and make_mask can take it as a mask for another fit (source fit)"
    )
    zoom_in = on.say("zoom", ", zoom")
    opened_by = on.say("read_file", ", which read_file opens")
    made_images = on.say("make_image", ", or an image made with make_image")
    mask_made = on.say("make_mask", " (made with make_mask)")
    return f"""\
Fit the whole stack to the atlas at once with emlddmm, on its current plane.

emlddmm (Tward et al. 2025, Nature Communications) maps whole-brain datasets
of any modality into the Allen CCF. It models every section as an image of one
deformed 3-D atlas, with an EM sorting of missing tissue and artifacts and a
contrast fit in small blocks.

Use this when you have most of a brain's series, closely and evenly
spaced, and want placements that agree across the stack and fit the
brain's overall shape. It bends the brain as a whole, never one section
alone{local_mismatch}.

emlddmm explains every placed section as a picture of the same 3-D atlas:
one smooth 3-D warp of the atlas, one shared scale and shear for the stack,
and for each section its own in-plane turn and shift (its mounting). It
fits all of them together against the sections' images (one section
image, by name), with a contrast fit in small blocks and an EM sorting of every
pixel into matching tissue, background and artifact, so holes, debris and
empty slide are set aside. Neighbouring sections constrain each other: a
section with little anatomy is bent and shifted the way the stack is. The
cutting angles and positions stay as they are: put the sections in order
and within about one interval of their place first. emlddmm has no
centring step and cannot recover from a start 1 to 3 mm off the tissue,
so have the atlas outline over each section's tissue, placed and
roughly aligned, before calling; each section is
refined from its current registration, its deformation included{landmark_kept}. The section's \
marked damage regions are
left out automatically{damage_first}.

emlddmm takes a section mask as soft weights: the mask multiplies each
pixel's weight in the fit, so a pixel at 0.3 counts at 0.3. Besides the
mask, emlddmm estimates its own sorting during the fit; the final sorting
of each section it writes a deformation for is saved with that
deformation{mask_from_sorting}. It keeps its own brightness scaling: the section image as it
is, 0 to 1.

Writes, per section: with fit "transform" or "both" its transform (turn,
shift, and the stack's scale and shear) and with fit "warp" or "both" its
deformation, the section's cut of the 3-D warp. A section that holds a deformation or a
frozen transform keeps its transform and gets the rigid change folded
into its deformation. A section whose result cannot be written still
shapes the stack. The whole landing is one undo step.

The sections are put on a grid of slots slot_um apart along the slicing
axis, each moved to its nearest slot for the fit (at most half a slot),
and the memory and time grow as the grid gets finer. The call is refused
when the stack would not fit the memory, and says which spacing would.

Starts one piece of background work over the stack and returns at once
with its id. Every section in the stack refuses writes until it lands;
you can look{zoom_in} and work on sections you left out. When it lands, the
next tool reply starts with its notice: how many sections were written,
those fitted for the stack and not written with the reason, the sections
whose warp has plausibility flags with their codes, and the overlays of
up to four sections (flagged ones first, then those that moved most) with
how far each moved. Judge the rest with look. emlddmm's own energy record,
the images it saw and the per-section results (movement, warp
displacement, fold fraction, flags, tissue set aside) are files in the
job{opened_by}.

Args:
    sections: Filenames; empty means every section with a position and a
        transform. At least two.
    slot_um: The slot spacing in micrometres; you set it, there is no
        default. Get the order right and each section within about one
        interval of its place first. Then set slot_um to the section
        thickness, or to about half the interval between neighbouring
        sections. Finer costs memory that many machines lack; coarser
        moves sections further from where you placed them and limits what
        the fit can resolve along the slicing axis. Two sections may not
        share a slot.
    section_image: The section image to fit, by name: "preprocessed", a raw channel{made_images}. \
Read as grey, as it
        is, 0 to 1. Default "preprocessed".
    atlas_image: One of the atlas's images (the job statement lists
        them), or "nissl", the atlas's Nissl (its template when it ships
        none). Default "nissl"; the reply names the image used.
    mask: "none", or the name of a mask that every section in the stack
        has{mask_made}; its weights multiply the
        fit's weight on that section. Default "none".
    kernel_um: The width of the kernel that smooths the 3-D warp's
        velocity, in micrometres, across sections too (emlddmm's a; not a
        Gaussian sigma on the displacement, as smoothness_um is
        elsewhere): tissue closer than this bends together. Default 500.
    fit: "transform" refits each section's in-plane turn and shift and
        the stack's scale and shear and writes them; "warp" keeps every
        transform and fits the 3-D warp alone; "both" fits both. Default
        "both".
    warp_freedom: How freely the warp may bend; larger bends more. Start
        large and reduce if the warps look unrealistic. Default 10000.
    iterations: Steps at the 200 um scale; the 100 and 50 um scales take
        half and a quarter as many. Default 100.
    sorting: Any of "match_sd", "background_sd", "artifact_sd",
        "background_mean", "artifact_mean" (brightness from 0, black, to 1,
        white); keys left out take emlddmm's values. emlddmm's sorting is
        its own EM with an artifact class: it takes no "method" or
        "artifact".
    view: True sends the overlays with the notice; false sends none.

Returns:
    The work id, the number of sections in the stack, those left out with
    why, the sections that will not get a transform and what they get, the
    furthest a section moves to its slot, the device, and per section what
    the fit uses (inputs: the section image, the atlas image as asked and as
    used, the mask, the damage, the brightness scaling); or why it was not
    started. The notice that follows brings, per section, what emlddmm
    estimated (its sorting and the share of the tissue it left out)."""


def emlddmm_stack(
    sections: list[str],
    slot_um: float,
    section_image: str = "preprocessed",
    atlas_image: str = "nissl",
    mask: str = "none",
    kernel_um: float = 500.0,
    fit: Literal["transform", "warp", "both"] = "both",
    warp_freedom: float = 10000.0,
    iterations: int = 100,
    sorting: dict[str, Any] = PUBLISHED_SORTING,
    view: bool = True,
) -> dict[str, Any]: ...


def _plane_description(on: ToolsOn) -> str:
    write_with = on.say("position_sections", " with position_sections")
    flip_with = on.say("interactive_transform", " with interactive_transform")
    opened_by = on.say("read_file", ", which read_file opens")
    like_stack = on.say("emlddmm_stack", ", as for emlddmm_stack")
    return f"""\
Estimate the stack's cutting angles and spacing with emlddmm. Writes nothing.

Fits emlddmm's 3-D affine to the whole placed stack at once, with the
plane free and no warp: every section votes on one pitch and yaw, and
the spacing may stretch and shift as a whole (a section cannot move on
its own). Starts from the current angles and positions, so place and
orient every section first, with the atlas outline over its tissue (emlddmm
cannot recover from a start 1 to 3 mm off); one call is one descent and lands short
from far away, so write the angles{write_with} and run it again until two passes agree within a
degree or two. emlddmm cannot flip a section: a section mounted mirrored
and not yet flipped pulls the yaw toward zero and stands out from the
others in its residual turn or shift here; flip it{flip_with} and run again before writing. Keep the
positions you placed unless the spacing change looks right in the
pictures. emlddmm's energy record and the images it saw are files in the
job{opened_by}.

Args:
    sections: Filenames; empty means every section with a position. At
        least two.
    slot_um: The slot spacing in micrometres{like_stack}: you
        set it, there is no default. With the order right and each section
        within about one interval of its place, set it to the section
        thickness, or to about half the interval between neighbouring
        sections.
    section_image: The section image to fit{like_stack}.
    atlas_image: One of the atlas's images or "nissl", as for the fit.
    iterations: Steps at the coarsest scale{like_stack}. Default
        100.
    view: True returns the pictures described below; false returns none.

Returns:
    pitch_deg, yaw_deg and their change from the stack's angles; the
    spacing scale and shift_mm; per section what the fit used (inputs), its
    proposed position_mm, the
    change, and the residual turn (deg) and shift (mm) it needed at the
    found plane beyond what the stack as a whole needed; and the overlays of up to four sections \
drawn at the found plane, not written."""


def emlddmm_plane(
    sections: list[str],
    slot_um: float,
    section_image: str = "preprocessed",
    atlas_image: str = "nissl",
    iterations: int = 100,
    view: bool = True,
) -> dict[str, Any]: ...


@dataclass(frozen=True)
class Options:
    """Hold the plugin's options (``plugin_config``).

    Parameters
    ----------
    device
        ``auto``, ``cpu``, ``cuda`` or a numbered GPU such as ``cuda:1``.
    dtype
        ``float32`` or ``float64``.
    section_um
        The pixel of the stack grid, um.
    section_weight
        The thickness one section stands for, um; None is the gap between
        neighbouring slots.
    memory_limit_gb
        The most memory a stack may need.
    """

    device: str = "auto"
    dtype: str = "float64"
    section_um: float = 58.88
    section_weight: float | None = None
    memory_limit_gb: float = 24.0

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
            For an option the plugin does not take, or a dtype that is not
            ``float32`` or ``float64``.
        """
        given = known_options("emlddmm", config, OPTIONS)
        dtype = str(given.get("dtype", cls.dtype))
        if dtype not in DTYPES:
            raise PluginError(f"emlddmm's dtype is one of {', '.join(DTYPES)}; got {dtype!r}")
        weight = given.get("section_weight", "gap")
        return cls(
            device=device_option(given.get("device", cls.device)),
            dtype=dtype,
            section_um=float(str(given.get("section_um", cls.section_um))),
            section_weight=None if weight == "gap" else float(str(weight)),
            memory_limit_gb=float(str(given.get("memory_limit_gb", cls.memory_limit_gb))),
        )

    def settings(self) -> dict[str, object]:
        """Return the options as the call's files record them."""
        return {
            "device": self.device,
            "dtype": self.dtype,
            "section_um": self.section_um,
            "section_weight": "gap" if self.section_weight is None else self.section_weight,
            "memory_limit_gb": self.memory_limit_gb,
        }

    def resolved(self) -> Options:
        """Return the options with ``auto`` replaced by the device it names."""
        return replace(self, device=resolve_device(self.device))


def _unit(v: Array) -> Array:
    """Return a vector scaled to length one."""
    v = np.asarray(v, np.float64)
    return v / np.linalg.norm(v)


@dataclass(frozen=True)
class _Section:
    """Hold what one section contributes, taken from its fit frame (which is then dropped).

    Parameters
    ----------
    view
        The section as the job holds it.
    placed
        The linear placement, native atlas pixels to fit-grid pixels (3 x 3).
    has_transform
        Whether the section has a written transform.
    applied
        The section's deformation as native atlas pixels at every fit-grid
        pixel, when it holds one.
    size
        The fit grid, ``(width, height)``.
    white_slide
        Whether the slide is white.
    soft
        The section mask's weights at the stack grid's pixel, or None.
    resolved
        The atlas image that was used.
    inputs
        The ``inputs`` block of the section's reply.
    down
        The section image in [0, 1], area-averaged to the stack's pixel.
    down_scale
        ``(sx, sy)``: the fit-grid pixels one down pixel spans.
    centre, dx, dy
        The plane's centre and its column and row directions, atlas um.
    to_plane, from_plane
        Native atlas pixels to in-plane um (2 x 2), and back.
    shift_um
        The in-plane um of the native origin about the plane's centre.
    native_mm
        The native plane's pixel, mm.
    tissue_native
        The atlas's annotated plane.
    outline
        The placed atlas outline on the fit grid.
    damage
        The marked damage on the native plane, or None.
    """

    view: SectionView
    placed: Array
    has_transform: bool
    applied: np.ndarray | None
    size: tuple[int, int]
    white_slide: bool
    soft: np.ndarray | None
    resolved: str
    inputs: dict[str, Any]
    down: np.ndarray
    down_scale: tuple[float, float]
    centre: Array
    dx: Array
    dy: Array
    to_plane: Array
    from_plane: Array
    shift_um: Array
    native_mm: float
    tissue_native: np.ndarray
    outline: np.ndarray
    damage: np.ndarray | None

    @property
    def id(self) -> str:
        """Return the section's filename."""
        return self.view.id

    def registered(self, pts: Array) -> Array:
        """Carry fit-grid pixels ``(..., 2)`` through the current registration to native atlas
        pixels."""
        if self.applied is not None:
            return sample_pixels(self.applied, pts).astype(np.float64)
        inv = np.linalg.inv(self.placed)
        return np.asarray(pts, np.float64) @ inv[:2, :2].T + inv[:2, 2]

    def unregistered(self, p: Array) -> Array:
        """Carry native atlas pixels to the fit-grid pixels the registration sends there.

        Linear: the placement. With a deformation: a fixed-point iteration from the placement.
        """
        lin = self.placed[:2, :2]
        g = p @ lin.T + self.placed[:2, 2]
        if self.applied is None:
            return g
        for _ in range(GEOMETRY.fixed_point_steps):
            g = g - (self.registered(g) - p) @ lin.T
        return g

    def plane_um(self, p: Array) -> Array:
        """Carry native atlas pixels to in-plane micrometres (column, row) about the plane's
        centre."""
        flat = np.asarray(p, np.float64).reshape(-1, 2).T
        return (self.to_plane @ flat).T.reshape(p.shape) + self.shift_um

    def native_from_atlas(self, atlas_um: Array, shape: tuple[int, int]) -> tuple[Array, Array]:
        """Carry atlas um ``(N, 3)`` to native pixels of the section's own plane ``(nh, nw, 2)``,
        and the distance off that plane, um."""
        d = np.asarray(atlas_um, np.float64) - self.centre
        uv = np.stack([d @ self.dx, d @ self.dy], -1)
        normal = _unit(np.cross(self.dy, self.dx))
        return ((uv - self.shift_um) @ self.from_plane.T).reshape(*shape, 2), d @ normal


@dataclass(frozen=True)
class _Ask:
    """Hold what a call asked of the stack.

    Parameters
    ----------
    names
        The sections named; empty means every section.
    slot_um
        The slot spacing, um.
    section_image, atlas_image
        The images asked for.
    mask
        The mask asked for.
    need_transform
        Whether every section must have a transform.
    """

    names: list[str]
    slot_um: float
    section_image: str
    atlas_image: str
    mask: str
    need_transform: bool


def _read_section(ctx: ToolContext, view: SectionView, ask: _Ask, section_um: float) -> _Section:
    """Read one section: its registration, its plane, its image in [0, 1] area-averaged to the
    stack's pixel, its tissue outline and its damage."""
    sid = view.id
    frame: FitFrame = _blocks.frame_for(ctx, sid, ask.atlas_image, ask.section_image)
    _plane_image, resolved = _blocks.atlas_plane(frame, ask.atlas_image)
    held = _blocks.section_mask(ctx, sid, ask.mask)
    placed = placed_matrix(frame)
    grey = _blocks.section_pixels(frame).astype(np.float64).mean(axis=2)
    gh, gw = grey.shape
    slide = slide_level(grey)
    unit = 255.0 if grey.max() > GEOMETRY.byte_range else 1.0
    plane = frame.plane
    px_um = float(np.mean(plane["px_um"])) / float(np.sqrt(abs(np.linalg.det(placed[:2, :2]))))
    down, scale = area_average((grey / unit).astype(np.float32), section_um / px_um)
    soft = None
    if held is not None:
        soft = area_average(np.asarray(held.weights, np.float32), section_um / px_um)[0]
    m = np.asarray(plane["native_to_um"], np.float64)
    centre, dx, dy = (np.asarray(plane[k], np.float64) for k in ("center_um", "dx", "dy"))
    to_plane = np.array([[dx @ m[:, 0], dx @ m[:, 1]], [dy @ m[:, 0], dy @ m[:, 1]]])
    return _Section(
        view=view,
        placed=placed,
        has_transform=frame.placed_atlas_to_section is not None,
        applied=None if frame.applied_map is None else np.asarray(frame.applied_map, np.float32),
        size=(gw, gh),
        white_slide=slide / unit > GEOMETRY.white_slide,
        soft=soft,
        resolved=resolved,
        inputs=_blocks.inputs_block(
            ctx,
            frame,
            section_image=ask.section_image,
            atlas_image=ask.atlas_image,
            resolved=resolved,
            mask=held,
            brightness="as it is, 0-1",
        ),
        down=down,
        down_scale=scale,
        centre=centre,
        dx=dx,
        dy=dy,
        to_plane=to_plane,
        from_plane=np.linalg.inv(to_plane),
        shift_um=np.array([dx @ (m[:, 2] - centre), dy @ (m[:, 2] - centre)]),
        native_mm=float(np.mean(plane["px_um"])) / UM_PER_MM,
        tissue_native=frame.atlas["annotation"] > 0,
        outline=placed_outline(frame, (gw, gh)),
        damage=None if frame.damage_zone is None else np.asarray(frame.damage_zone, bool),
    )


@dataclass(frozen=True)
class _Placed:
    """Hold one section on the stack: where it sits and what it holds.

    Parameters
    ----------
    section
        The section.
    offset_um
        The section's plane centre's place in the stack's (column, row) um.
    slot
        The slot it sits in.
    snap_um
        How far it moved to its slot, um along its normal.
    tissue
        The atlas tissue on the stack grid where the section holds data.
    """

    section: _Section
    offset_um: Array
    slot: int
    snap_um: float
    tissue: np.ndarray

    @property
    def id(self) -> str:
        """Return the section's filename."""
        return self.section.id


@dataclass(frozen=True)
class _Stack:
    """Hold the stack of the placed sections: geometry, slots and images on the stack grid.

    Parameters
    ----------
    sections
        The sections in order along the slicing axis.
    left_out
        The sections that could not enter, with the reason.
    normal, dx, dy
        The stack's cutting direction and its column and row directions.
    origin_um
        The most anterior section's plane centre, atlas um.
    spacing_um
        The slot spacing along the normal, um.
    depths_um
        Each section's distance from the first plane, um.
    slots
        Each section's slot.
    slot_um
        The slot spacing asked for, um.
    n_slots
        The number of slots.
    tilt_deg
        The largest angle between two sections' planes.
    rows_um, cols_um
        The stack grid's coordinates, um.
    images, weights
        The sections on the grid and where they hold data.
    memory_gb
        The memory the stack is estimated to take.
    weight_um
        The thickness one section stands for, um.
    white_slide
        Whether the first section's slide is white.
    resolved
        The atlas image that was used.
    section_um
        The stack grid's pixel, um.
    """

    sections: list[_Placed]
    left_out: list[dict[str, Any]]
    normal: Array
    dx: Array
    dy: Array
    origin_um: Array
    spacing_um: float
    depths_um: Array
    slots: np.ndarray
    slot_um: float
    n_slots: int
    tilt_deg: float
    rows_um: Array
    cols_um: Array
    images: np.ndarray
    weights: np.ndarray
    memory_gb: float
    weight_um: float
    white_slide: bool
    resolved: str
    section_um: float

    @property
    def rotation(self) -> Array:
        """Return the stack's axes as rows: slot, row and column direction in atlas um."""
        return np.stack([self.normal, self.dy, self.dx])

    @property
    def slot_depths_um(self) -> Array:
        """Return each slot's coordinate along the slicing axis, um."""
        return np.arange(self.n_slots) * self.spacing_um

    def images_for_engine(self) -> engine.StackImages:
        """Return the stack grid as the engine takes it."""
        return engine.StackImages(
            images=self.images,
            weights=self.weights,
            depths_um=self.slot_depths_um,
            rows_um=self.rows_um,
            cols_um=self.cols_um,
        )


def _slots(depths: Array, slot_um: float, cos: float) -> tuple[np.ndarray, float]:
    """Snap depths to slots ``slot_um * cos`` apart: the slot of each and that spacing."""
    spacing = slot_um * cos
    return np.rint(depths / spacing).astype(np.intp), spacing


def _memory_gb(n_slots: int, ny: int, nx: int, dtype: str) -> float:
    """Estimate the memory of a stack of ``n_slots`` slots of ``ny`` x ``nx`` pixels."""
    megavoxels = n_slots * ny * nx / 1e6
    return megavoxels * MEMORY.gb_per_megavoxel / (2.0 if dtype == "float32" else 1.0)


def _gather(
    ctx: ToolContext, options: Options, ask: _Ask
) -> tuple[list[_Section], list[dict[str, Any]]]:
    """Read every section that can enter the stack; a section that cannot (no placement) is left
    out with its reason."""
    left_out: list[dict[str, Any]] = []
    sections: list[_Section] = []
    for view in [ctx.section(name) for name in ask.names] if ask.names else ctx.sections():
        try:
            if ask.need_transform and view.affine is None:
                raise ToolError(RefusalCode.PRECONDITION, NEEDS_TRANSFORM)
            section = _read_section(ctx, view, ask, options.section_um)
            if ask.need_transform and not section.has_transform:
                raise ToolError(RefusalCode.PRECONDITION, NEEDS_TRANSFORM)
        except ToolError as exc:
            if exc.code == RefusalCode.BAD_ARGUMENT:
                raise
            left_out.append({"id": view.id, "why": exc.code, "message": exc.message})
            continue
        sections.append(section)
    if len(sections) < 2:
        raise ToolError(
            RefusalCode.PRECONDITION,
            "emlddmm fits a stack: at least two sections with "
            "positions" + (" and transforms" if ask.need_transform else ""),
            left_out=left_out,
        )
    return sections, left_out


@dataclass(frozen=True)
class _Axes:
    """Hold the one cutting direction of a stack and how far its sections' planes differ.

    Parameters
    ----------
    dx, dy, normal
        The column direction, the row direction and the normal.
    tilt_deg
        The largest angle between a section's directions and the mean.
    """

    dx: Array
    dy: Array
    normal: Array
    tilt_deg: float


def _cutting_axes(sections: list[_Section]) -> _Axes:
    """Find the stack's one cutting direction; refuse a stack whose planes differ."""
    dxs, dys = (np.array([_unit(getattr(s, k)) for s in sections]) for k in ("dx", "dy"))
    dx, dy = _unit(dxs.mean(0)), _unit(dys.mean(0))
    dy = _unit(dy - (dy @ dx) * dx)
    cosine = float(np.clip(min((dxs @ dx).min(), (dys @ dy).min()), -1, 1))
    tilt = max(math.degrees(math.acos(cosine)), 0.0)
    if tilt > ANGLE_TOLERANCE_DEG:
        raise ToolError(
            RefusalCode.PRECONDITION,
            f"the sections' planes differ by {tilt:.1f} degrees: "
            "emlddmm needs one cutting angle for the stack",
        )
    return _Axes(dx, dy, _unit(np.cross(dy, dx)), tilt)


def _footprint_axes(
    sections: list[_Section], offsets: Array, section_um: float
) -> tuple[Array, Array]:
    """The stack grid's column and row coordinates: the union of the sections' footprints with a
    margin, on a grid ``section_um`` apart."""
    n = GEOMETRY.edge_samples
    corners = []
    for section, off in zip(sections, offsets, strict=True):
        w, h = section.size
        along_w, along_h = np.linspace(0, w - 1, n), np.linspace(0, h - 1, n)
        edge = np.concatenate(
            [
                np.stack([along_w, np.zeros(n)], 1),
                np.stack([along_w, np.full(n, h - 1.0)], 1),
                np.stack([np.zeros(n), along_h], 1),
                np.stack([np.full(n, w - 1.0), along_h], 1),
            ]
        )
        corners.append(section.plane_um(section.registered(edge)) + off)
    all_corners = np.concatenate(corners)
    margin = GEOMETRY.margin_pixels * section_um
    lo = np.floor(all_corners.min(0) / section_um) * section_um - margin
    hi = np.ceil(all_corners.max(0) / section_um) * section_um + margin
    cols = np.arange(lo[0], hi[0] + 1e-6, section_um)
    rows = np.arange(lo[1], hi[1] + 1e-6, section_um)
    return cols, rows


def _check_fits(
    sections: list[_Section],
    slots: np.ndarray,
    depths: Array,
    cos: float,
    shape: tuple[int, int],
    options: Options,
    slot_um: float,
) -> float:
    """Refuse a stack with two sections in a slot or one that needs more memory than the limit;
    return the memory it needs, GB."""
    ny, nx = shape
    n_slots = int(slots.max()) + 1
    memory = _memory_gb(n_slots, ny, nx, options.dtype)
    taken: dict[int, str] = {}
    for k, section in zip(slots.tolist(), sections, strict=True):
        if k in taken:
            raise ToolError(
                RefusalCode.BAD_ARGUMENT,
                f"{taken[k]} and {section.id} share a {slot_um:g} um "
                "slot; use a finer slot_um (or leave one of them out)",
            )
        taken[k] = section.id
    limit = options.memory_limit_gb
    if memory <= limit:
        return memory
    fits = next(
        (
            float(s)
            for s in np.arange(slot_um, MEMORY.search_max_um, MEMORY.search_step_um)
            if _memory_gb(int(_slots(depths, float(s), cos)[0].max()) + 1, ny, nx, options.dtype)
            <= limit
        ),
        None,
    )
    advice = "fewer sections or a smaller field"
    if fits is not None:
        shared = len(set(_slots(depths, fits, cos)[0].tolist())) != len(sections)
        advice = f"a slot_um of {fits:g} or more would fit" + (
            ", but then two sections share a slot: leave some sections out" if shared else ""
        )
    raise ToolError(
        RefusalCode.BAD_ARGUMENT,
        f"{n_slots} slots of {ny} x {nx} pixels "
        f"need about {memory:.0f} GB, over the {limit:g} GB limit; {advice}",
        estimate_gb=round(memory, 1),
    )


def _paint(
    section: _Section,
    offset: Array,
    cols: Array,
    rows: Array,
    image: np.ndarray,
    weight: np.ndarray,
) -> np.ndarray:
    """Paint one section onto its slot's image and weight; return its tissue on the stack grid."""
    mesh_x, mesh_y = np.meshgrid(cols, rows)
    uv = np.stack([mesh_x - offset[0], mesh_y - offset[1]], -1) - section.shift_um
    p = uv @ section.from_plane.T
    g = section.unregistered(p)
    sx, sy = section.down_scale
    at = (g + 0.5) / np.array([sx, sy]) - 0.5
    w, h = section.size
    inside = (
        (g[..., 0] >= -0.5) & (g[..., 0] <= w - 0.5) & (g[..., 1] >= -0.5) & (g[..., 1] <= h - 0.5)
    )
    image[:] = np.where(inside, sample_pixels(section.down, at), float(section.white_slide))
    keep = inside.copy()
    if section.damage is not None:
        keep &= ~(sample_pixels(section.damage.astype(np.uint8), p, cv2.INTER_NEAREST) > 0)
    weight[:] = keep
    if section.soft is not None:
        weight *= np.clip(sample_pixels(section.soft, at), 0.0, 1.0)
    tissue = sample_pixels(section.tissue_native.astype(np.uint8), p, cv2.INTER_NEAREST) > 0
    return tissue & keep


def _stack(ctx: ToolContext, options: Options, ask: _Ask) -> _Stack:
    """The stack of the placed sections: geometry, slots, memory check, images on the stack grid."""
    if not ask.slot_um > 0:
        raise ToolError(RefusalCode.BAD_ARGUMENT, "slot_um is a positive number of micrometres")
    gathered, left_out = _gather(ctx, options, ask)
    axes = _cutting_axes(gathered)
    centres = np.array([s.centre for s in gathered])
    order = np.argsort(centres @ axes.normal)
    sections = [gathered[i] for i in order]
    centres = centres[order]
    origin = centres[0]
    rel = centres - origin
    far = rel[-1]
    cos = abs(float(_unit(far) @ axes.normal)) if np.linalg.norm(far) > 0 else 1.0
    depths = rel @ axes.normal
    slots, spacing = _slots(depths, ask.slot_um, cos)
    offsets = np.stack([rel @ axes.dx, rel @ axes.dy], 1)
    cols, rows = _footprint_axes(sections, offsets, options.section_um)
    memory = _check_fits(sections, slots, depths, cos, (len(rows), len(cols)), options, ask.slot_um)
    n_slots = int(slots.max()) + 1
    gap = float(slots.max() - slots.min()) * spacing / max(len(sections) - 1, 1)
    thick = gap if options.section_weight is None else options.section_weight
    images = np.zeros((n_slots, len(rows), len(cols)), np.float32)
    weights = np.zeros((n_slots, len(rows), len(cols)), np.float32)
    placed = []
    for i, (section, off, k) in enumerate(zip(sections, offsets, slots, strict=True)):
        tissue = _paint(section, off, cols, rows, images[k], weights[k])
        placed.append(_Placed(section, off, int(k), float((k * spacing - depths[i]) / cos), tissue))
    return _Stack(
        sections=placed,
        left_out=left_out,
        normal=axes.normal,
        dx=axes.dx,
        dy=axes.dy,
        origin_um=origin,
        spacing_um=spacing,
        depths_um=depths,
        slots=slots,
        slot_um=float(ask.slot_um),
        n_slots=n_slots,
        tilt_deg=axes.tilt_deg,
        rows_um=rows,
        cols_um=cols,
        images=images,
        weights=weights,
        memory_gb=memory,
        weight_um=thick,
        white_slide=sections[0].white_slide,
        resolved=sections[0].resolved,
        section_um=options.section_um,
    )


def _schedule(
    stack: _Stack,
    iterations: list[int],
    kernel_um: float,
    warp_freedom: float,
    *,
    rigid: bool,
    warp: bool,
) -> engine.Schedule:
    """emlddmm's arguments per scale: the published mouse serial-section example, with each
    section weighed as the gap it stands for."""
    k = stack.weight_um / stack.spacing_um
    return engine.Schedule(
        iterations=tuple(int(i) for i in iterations),
        kernel_um=float(kernel_um),
        velocity_size_um=EXAMPLE.velocity_size_um,
        warp_start=0 if warp else EXAMPLE.no_warp,
        warp_step=EXAMPLE.warp_step * k,
        affine_step=EXAMPLE.affine_step * k if rigid else 0.0,
        slice_step=EXAMPLE.slice_step * EXAMPLE.slice_um / stack.spacing_um if rigid else 0.0,
        warp_sigma=float(warp_freedom) * math.sqrt(k),
    )


def scale_iterations(iterations: int) -> list[int]:
    """Return emlddmm's steps at its three scales (200, 100, 50 um) from the first scale's: that,
    half of it and a quarter, as the published example's [100, 50, 25] runs."""
    first = int(iterations)
    return [first, max(1, first // 2), max(1, first // 4)]


def _check_iterations(iterations: Any) -> list[int]:
    """The three scales' steps of *iterations*, the first scale's (:func:`scale_iterations`)."""
    if (
        isinstance(iterations, bool)
        or not isinstance(iterations, int)
        or iterations < MIN_ITERATIONS
    ):
        raise ToolError(
            RefusalCode.BAD_ARGUMENT,
            "iterations is a whole number of steps at the coarsest "
            "scale, 4 or more (the finer scales take half and a quarter)",
        )
    return scale_iterations(iterations)


def _check_sorting(sorting: dict[str, Any]) -> dict[str, float]:
    held = [key for key in sorting if key in ("method", "artifact")]
    if held:
        raise ToolError(
            RefusalCode.BAD_ARGUMENT,
            f"emlddmm's sorting is its own EM with an artifact class: "
            f"it takes no {' or '.join(held)}; its keys are "
            f"{', '.join(SORTING_KEYS)}",
        )
    unknown = [key for key in sorting if key not in SORTING_KEYS]
    if unknown:
        raise ToolError(
            RefusalCode.BAD_ARGUMENT,
            f"unknown sorting keys {unknown}; the keys are {', '.join(SORTING_KEYS)}",
        )
    return {key: float(value) for key, value in sorting.items()}


def _run_engine(request: engine.FitRequest, folder: Path) -> engine.FitResult:
    """Run emlddmm in this process; what it printed is kept in *folder*'s ``log.txt``."""
    folder.mkdir(parents=True, exist_ok=True)
    try:
        result = engine.run(request)
    except ENGINE_FAILURES as exc:
        failure = f"{type(exc).__name__}: {exc}"
        raise ToolError(RefusalCode.FIT_FAILED, f"emlddmm failed: {failure}") from exc
    (folder / "log.txt").write_text(result.log, encoding="utf-8", newline="\n")
    return result


def _request(
    options: Options,
    atlas: str,
    stack: _Stack,
    schedule: engine.Schedule,
    *,
    sorting: Sorting,
    mode: engine.Mode,
    read_warp: bool,
    points: tuple[Array, ...],
) -> engine.FitRequest:
    """Build the engine's request for a stack."""
    return engine.FitRequest(
        atlas=atlas,
        atlas_image=stack.resolved,
        stack=stack.images_for_engine(),
        frame=engine.StackFrame(
            rotation=stack.rotation,
            origin_um=stack.origin_um,
            slots=tuple(placed.slot for placed in stack.sections),
        ),
        schedule=schedule,
        sorting=sorting,
        white_slide=stack.white_slide,
        mode=mode,
        read_warp=read_warp,
        points=points,
        device=options.device,
        single_precision=options.dtype == "float32",
        up_vector=tuple((-stack.dy).tolist()) if mode == "plane" else None,
    )


def _font(size: int) -> ImageFont.ImageFont | ImageFont.FreeTypeFont:
    try:
        return ImageFont.load_default(size)
    except TypeError:
        return ImageFont.load_default()


def _sheet_picture(
    stack: _Stack, i: int, placed: _Placed, result: engine.FitResult, path: Path
) -> None:
    """One section as the engine saw it, its final sorting (white compared, black ignored) and
    the atlas under the fit; a file in the job, not sent."""
    k = placed.slot
    data = stack.weights[k] > 0
    seen = np.clip(stack.images[k] / SHEET.display_max, 0, 1)
    sort = np.clip(np.asarray(result.match_weights[k], np.float32), 0, 1) * data
    atl = np.asarray(result.atlas_cut[i], np.float32)
    lo, hi = np.percentile(atl[data], SHEET.percentiles) if data.any() else (0, 1)
    atl = np.clip((atl - lo) / max(hi - lo, 1e-9), 0, 1)
    tiles = [(t * 255).astype(np.uint8) for t in (seen, sort, atl)]
    gap = np.full((tiles[0].shape[0], SHEET.gap), 255, np.uint8)
    row = np.concatenate([tiles[0], gap, tiles[1], gap, tiles[2]], 1)
    image = Image.fromarray(row).convert("RGB")
    size = (image.width * SHEET.enlarge, image.height * SHEET.enlarge)
    image = image.resize(size, Image.Resampling.LANCZOS)
    band = Image.new("RGB", (image.width, image.height + SHEET.band), "white")
    band.paste(image, (0, SHEET.band))
    ImageDraw.Draw(band).text(
        (6, 5),
        f"{placed.id}  slot {k}: the section as sent | the final "
        "sorting (white compared) | the atlas under the fit",
        fill="black",
        font=_font(SHEET.font),
    )
    band.save(path)


def _energy_files(energy: Array, folder: Path) -> None:
    """emlddmm's own energy record as ``energy.csv`` and a small line plot (files in the job)."""
    with open(folder / "energy.csv", "w", encoding="utf-8", newline="") as fh:
        writer = csv.writer(fh)
        writer.writerow(["scale", "iteration", "total", "mismatch", "regularization"])
        for row in energy:
            writer.writerow([int(row[0]), int(row[1]), *[f"{v:.6g}" for v in row[2:]]])
    h, w, pad = PLOT.height, PLOT.width, PLOT.pad
    image = Image.new("RGB", (w, h), "white")
    draw = ImageDraw.Draw(image)
    if len(energy) > 1:
        for col, shade, name in (
            (2, (0, 0, 0), "total"),
            (3, (110, 110, 110), "mismatch"),
            (4, (190, 190, 190), "regularization"),
        ):
            y = energy[:, col]
            lo, hi = float(y.min()), float(y.max())
            span = max(hi - lo, 1e-12)
            pts = [
                (pad + (w - 2 * pad) * j / (len(y) - 1), h - pad - (h - 2 * pad) * (v - lo) / span)
                for j, v in enumerate(y)
            ]
            draw.line(pts, fill=shade, width=1)
            draw.text(
                (pad, 4 + 10 * (col - 2)),
                f"{name}: min {lo:.4g}, max {hi:.4g} (each scaled to its own range)",
                fill=shade,
                font=_font(PLOT.font),
            )
    image.save(folder / "energy.png")


def _expand(native: Array, w: int, h: int) -> Array:
    """A map read on the lattice (nh, nw, 2) as a full fit-grid map (h, w, 2)."""
    nh, nw = native.shape[:2]
    xs = (np.arange(w) * (nw - 1) / max(w - 1, 1)).astype(np.float32)
    ys = (np.arange(h) * (nh - 1) / max(h - 1, 1)).astype(np.float32)
    mx, my = np.meshgrid(xs, ys)
    return sample_pixels(native.astype(np.float32), np.stack([mx, my], -1)).astype(np.float64)


@dataclass(frozen=True)
class _Lattice:
    """Where the map is read on one section: a lattice of fit-grid pixels, each carried through
    the section's current registration to its place in the stack frame.

    Parameters
    ----------
    shape
        The lattice, ``(rows, cols)``.
    pixels
        Its fit-grid pixels ``(rows, cols, 2)`` as (x, y).
    native
        The native atlas pixels the current registration sends them to.
    points
        Their places in the stack frame, ``(n, 2)`` (row, col) um.
    """

    shape: tuple[int, int]
    pixels: Array
    native: Array
    points: Array


def _lattice(placed: _Placed) -> _Lattice:
    """Lay the lattice the map is read on over one section."""
    section = placed.section
    w, h = section.size
    nw, nh = int(math.ceil((w - 1) / LATTICE_PX)) + 1, int(math.ceil((h - 1) / LATTICE_PX)) + 1
    xs, ys = np.linspace(0, w - 1, nw), np.linspace(0, h - 1, nh)
    pixels = np.stack(np.meshgrid(xs, ys), -1)
    native = section.registered(pixels)
    uv = section.plane_um(native) + placed.offset_um
    return _Lattice(
        (nh, nw), pixels, native, np.stack([uv[..., 1].ravel(), uv[..., 0].ravel()], -1)
    )


def _flags(summary: dict[str, Any]) -> list[str]:
    return list(dict.fromkeys(str(flag.get("code")) for flag in summary.get("flags") or []))


def _set_aside(result: engine.FitResult, placed: _Placed) -> dict[str, float]:
    """The share of the section's tissue (the placed atlas outline, where it holds data) the
    EM sorting set aside as background (WB) and as artifact (WA)."""
    inside = placed.tissue
    maps = {"WB": result.background_weights, "WA": result.artifact_weights}
    return {
        key: round(float((np.asarray(weights[placed.slot], np.float32) > 0.5)[inside].mean()), 3)
        if inside.any()
        else 0.0
        for key, weights in maps.items()
    }


def _weights_on_grid(stack: _Stack, placed: _Placed, stack_weights: np.ndarray) -> np.ndarray:
    """emlddmm's final sorting of one section (its slot of ``WM``, on the stack grid) as a
    float16 weight map on the section's fit grid, read through its current registration."""
    section = placed.section
    w, h = section.size
    gxy = np.stack(np.meshgrid(np.arange(w), np.arange(h)), -1).astype(np.float64)
    uv = section.plane_um(section.registered(gxy)) + placed.offset_um
    ix = ((uv[..., 0] - stack.cols_um[0]) / stack.section_um).astype(np.float32)
    iy = ((uv[..., 1] - stack.rows_um[0]) / stack.section_um).astype(np.float32)
    got = sample_pixels(stack_weights.astype(np.float32), np.stack([ix, iy], -1))
    return np.clip(got, 0.0, 1.0).astype(np.float16)


def _route(view: SectionView, *, rigid: bool, warp: bool) -> str:
    """How one section's result is written: ``transform`` (its transform, then the warp on top),
    ``deformation`` (the rigid change folded into its deformation) or ``none``."""
    free_a, free_d = "affine" not in view.frozen, "deformation" not in view.frozen
    if rigid and free_a and not view.deformation:
        return "transform" if (not warp or free_d) else "none"
    return "deformation" if (warp or rigid) and free_d else "none"


def _chosen(stack: _Stack, options: Options, schedule: engine.Schedule) -> dict[str, Any]:
    """What the plugin picked from the stack, kept in settings.json."""
    return {
        "slot_um": stack.slot_um,
        "slot_spacing_along_normal_um": round(stack.spacing_um, 3),
        "n_slots": stack.n_slots,
        "stack_shape": list(stack.images.shape),
        "largest_plane_tilt_deg": round(stack.tilt_deg, 3),
        "section_px_um": stack.section_um,
        "section_stands_for_um": round(stack.weight_um, 2),
        "device": options.device,
        "dtype": options.dtype,
        "memory_estimate_gb": round(stack.memory_gb, 2),
        "memory_limit_gb": options.memory_limit_gb,
        "emlddmm": schedule.keywords(),
    }


@dataclass(frozen=True)
class _FileRecord:
    """Hold what the call's files keep besides the fit's own results.

    Parameters
    ----------
    asked
        What the call asked.
    chosen
        What the plugin picked from the stack.
    plugin
        The plugin's options.
    extra
        What ``A.json`` adds to the matrices.
    """

    asked: Mapping[str, Any]
    chosen: Mapping[str, Any]
    plugin: Mapping[str, Any]
    extra: Mapping[str, Any]


def _write_files(
    stack: _Stack, result: engine.FitResult, folder: Path, record: _FileRecord
) -> None:
    """The call's readable files (see the module text); nothing here goes into a reply."""
    _energy_files(result.energy, folder)
    (folder / "images").mkdir(exist_ok=True)
    for i, placed in enumerate(stack.sections):
        _sheet_picture(
            stack, i, placed, result, folder / "images" / f"{i:02d}_{Path(placed.id).stem}.png"
        )
    np.savez_compressed(
        folder / "weights.npz",
        WM=result.match_weights,
        WB=result.background_weights,
        WA=result.artifact_weights,
        slot=np.asarray(stack.slots),
    )
    matrices = {
        "A": result.affine.tolist(),
        "A_start": result.start_affine.tolist(),
        "A2d": result.slice_affines.tolist(),
        **record.extra,
    }
    (folder / "A.json").write_text(
        json.dumps(matrices, indent=1, default=float), encoding="utf-8", newline="\n"
    )
    (folder / "settings.json").write_text(
        json.dumps(
            {
                "asked": record.asked,
                "chosen": record.chosen,
                "plugin": record.plugin,
                "emlddmm": result.chosen,
                "device": result.device,
                "energy": "captured" if result.energy_captured else "not captured",
                "peak_gpu_gb": result.peak_gpu_gb,
                "seconds": result.seconds,
            },
            indent=1,
            default=str,
        ),
        encoding="utf-8",
        newline="\n",
    )


@dataclass(frozen=True)
class _StackCall:
    """Hold what one ``emlddmm_stack`` call shares across its landing.

    Parameters
    ----------
    options
        The plugin's options, the device resolved.
    stack
        The stack.
    request
        The engine's request.
    lattices
        Where each section's map is read.
    settings
        The call's settings, as the reply and the deformation record them.
    asked
        What the call asked, for ``settings.json``.
    chosen
        What the plugin picked, for ``settings.json`` and the deformation record.
    routes
        How each section's result is written.
    digests
        Each section's registration digest when the call began.
    rigid, warp
        Whether the in-plane transform and the warp are fitted.
    view
        Whether pictures are sent.
    folder
        The call's folder in the job.
    note
        The words the registration's record carries.
    """

    options: Options
    stack: _Stack
    request: engine.FitRequest
    lattices: list[_Lattice]
    settings: dict[str, Any]
    asked: dict[str, Any]
    chosen: dict[str, Any]
    routes: dict[str, str]
    digests: dict[str, str]
    rigid: bool
    warp: bool
    view: bool
    folder: Path
    note: str


@dataclass(frozen=True)
class _Record:
    """Hold one section's result of the fit, read on its own plane.

    Parameters
    ----------
    placed
        The section on the stack.
    native
        The native atlas pixels at the lattice, ``(nh, nw, 2)``, with the warp if fitted.
    matrix
        The in-plane transform the fit implies, fit-grid pixels to native atlas pixels.
    weights
        The final sorting on the fit grid, or None without a warp.
    estimated
        The ``estimated`` block.
    off_plane_um
        The most the section's points lie off its own plane, um.
    moved_mm
        The median and the largest move inside the tissue, mm.
    set_aside
        The share of the tissue set aside as background and as artifact.
    """

    placed: _Placed
    native: Array
    matrix: Array
    weights: np.ndarray | None
    estimated: dict[str, Any] | None
    off_plane_um: float
    moved_mm: tuple[float, float]
    set_aside: dict[str, float]

    @property
    def id(self) -> str:
        """Return the section's filename."""
        return self.placed.id


@dataclass(frozen=True)
class _Row:
    """Hold what one section's landing wrote.

    Parameters
    ----------
    record
        The section's result.
    summary
        The written deformation's summary, empty without one.
    wrote
        What was written, in words.
    problem
        Why it was not (fully) written; empty when it was.
    """

    record: _Record
    summary: dict[str, Any]
    wrote: str
    problem: str


@dataclass
class _Outcome:
    """Keep what the writes of one landing gave."""

    written: dict[str, str] = field(default_factory=dict)
    failed: dict[str, dict[str, Any]] = field(default_factory=dict)
    summaries: dict[str, dict[str, Any]] = field(default_factory=dict)


@dataclass(frozen=True)
class _StackLanding:
    """Fit the stack and write each section's result, as the stack's background work."""

    call: _StackCall

    def __call__(self, lctx: ToolContext) -> ToolResult:
        """Run the fit, read it per section, save the files, write and report."""
        call = self.call
        result = _run_engine(call.request, call.folder)
        records = self._records(result)
        _write_files(
            call.stack,
            result,
            call.folder,
            _FileRecord(call.asked, call.chosen, call.options.settings(), {}),
        )
        outcome = self._write(lctx, records)
        rows = self._rows(records, outcome)
        self._sections_csv(rows)
        return self._report(lctx, result, rows, outcome)

    def _records(self, result: engine.FitResult) -> list[_Record]:
        """Read the fit on each section's own plane."""
        call = self.call
        records = []
        for i, (placed, lattice) in enumerate(zip(call.stack.sections, call.lattices, strict=True)):
            section = placed.section
            n_points = len(lattice.points)
            nat_full, off_plane = section.native_from_atlas(result.warped_um[i], lattice.shape)
            nat_lin, _ = section.native_from_atlas(result.linear_um[i], lattice.shape)
            gxy = np.concatenate([lattice.pixels.reshape(-1, 2), np.ones((n_points, 1))], 1)
            matrix = np.linalg.lstsq(gxy, nat_lin.reshape(-1, 2), rcond=None)[0].T
            w, h = section.size
            inside = section.outline[
                np.clip(lattice.pixels[..., 1].round().astype(int), 0, h - 1),
                np.clip(lattice.pixels[..., 0].round().astype(int), 0, w - 1),
            ]
            used = nat_full if call.warp else nat_lin
            moved = np.hypot(*(used - lattice.native).transpose(2, 0, 1)) * section.native_mm
            moved = moved[inside] if inside.any() else moved.ravel()
            weights = (
                _weights_on_grid(call.stack, placed, result.match_weights[placed.slot])
                if call.warp
                else None
            )
            records.append(
                _Record(
                    placed=placed,
                    native=used,
                    matrix=matrix,
                    weights=weights,
                    estimated=_blocks.estimated_block(weights, section.outline),
                    off_plane_um=float(np.abs(off_plane).max()),
                    moved_mm=(float(np.median(moved)), float(moved.max())),
                    set_aside=_set_aside(result, placed),
                )
            )
        return records

    def _proposal(
        self, lctx: ToolContext, record: _Record, *, from_current: bool
    ) -> DeformationProposal:
        """Propose the section's cut of the 3-D warp as its deformation."""
        call = self.call
        section = record.placed.section
        sid = record.id
        current = from_current and section.applied is not None
        return lctx.deformation_from_map(
            sid,
            _expand(record.native, *section.size),
            note=call.note,
            meta={"inputs": section.inputs, "estimated": record.estimated},
            weights=record.weights,
            settings=call.settings,
            engine_settings=call.chosen,
            engine_version=ENGINE_VERSION,
            start="current" if current else "linear",
        )

    def _write(self, lctx: ToolContext, records: list[_Record]) -> _Outcome:
        """Write the transforms, then the deformations, as one undo step."""
        call = self.call
        routes, digests = call.routes, call.digests
        outcome = _Outcome()
        transforms = {
            rec.id: lctx.transform_from_matrix(rec.id, rec.matrix, note=call.note).row()
            for rec in records
            if routes[rec.id] == "transform"
        }
        proposals = {
            rec.id: self._proposal(lctx, rec, from_current=True)
            for rec in records
            if routes[rec.id] == "deformation"
        }
        with lctx.writing():
            if transforms:
                done = lctx.set_transforms(
                    list(transforms.values()), expect={sid: digests[sid] for sid in transforms}
                )
                for row in done.rows:
                    if row.get("status") == "ok":
                        outcome.written[row["id"]] = "transform"
                    else:
                        outcome.failed[row["id"]] = row
            for rec in records:
                if routes[rec.id] == "transform" and call.warp and rec.id not in outcome.failed:
                    proposals[rec.id] = self._proposal(lctx, rec, from_current=False)
            if proposals:
                done = lctx.set_deformations(
                    [p.row() for p in proposals.values()],
                    expect={sid: digests[sid] for sid in proposals if routes[sid] == "deformation"},
                )
                for row in done.rows:
                    if row.get("status") == "ok":
                        outcome.summaries[row["id"]] = proposals[row["id"]].summary
                        outcome.written[row["id"]] = self._words(
                            outcome.written.get(row["id"]) == "transform"
                        )
                    else:
                        outcome.failed[row["id"]] = row
        return outcome

    def _words(self, transform_too: bool) -> str:
        """Say what a section's landing wrote when its deformation was written."""
        if transform_too:
            return "transform and deformation"
        return "deformation (rigid folded in)" if self.call.rigid else "deformation"

    def _rows(self, records: list[_Record], outcome: _Outcome) -> list[_Row]:
        """One row per section: what was written and why not, if it was not."""
        rows = []
        for rec in records:
            sid = rec.id
            code = outcome.failed.get(sid, {}).get("error")
            if sid in outcome.written and sid not in outcome.failed:
                problem, wrote = "", outcome.written[sid]
            elif sid in outcome.written:
                problem, wrote = f"transform only, deformation {code}", "transform only"
            else:
                problem, wrote = code or RefusalCode.FROZEN, "not written"
            rows.append(_Row(rec, outcome.summaries.get(sid, {}), wrote, problem))
        return rows

    def _sections_csv(self, rows: list[_Row]) -> None:
        """Save the per-section results as ``sections.csv``."""
        with open(self.call.folder / "sections.csv", "w", encoding="utf-8", newline="") as fh:
            writer = csv.writer(fh)
            writer.writerow(
                [
                    "section",
                    "slot",
                    "snap_um",
                    "moved_median_mm",
                    "moved_max_mm",
                    "off_plane_max_um",
                    "displacement_median_mm",
                    "displacement_max_mm",
                    "fold_fraction",
                    "flags",
                    "set_aside_background",
                    "set_aside_artifact",
                    "written",
                    "problem",
                ]
            )
            for r in rows:
                rec = r.record
                disp = r.summary.get("displacement_mm", {})
                writer.writerow(
                    [
                        rec.id,
                        rec.placed.slot,
                        round(rec.placed.snap_um, 2),
                        round(rec.moved_mm[0], 4),
                        round(rec.moved_mm[1], 4),
                        round(rec.off_plane_um, 1),
                        disp.get("median", ""),
                        disp.get("max", ""),
                        r.summary.get("fold_fraction", ""),
                        " ".join(_flags(r.summary)),
                        rec.set_aside["WB"],
                        rec.set_aside["WA"],
                        r.wrote,
                        r.problem,
                    ]
                )

    def _report(
        self, lctx: ToolContext, result: engine.FitResult, rows: list[_Row], outcome: _Outcome
    ) -> ToolResult:
        """Word the landing's notice and draw the overlays."""
        call = self.call
        problems = [f"{r.record.id} ({r.problem})" for r in rows if r.problem]
        flagged = [f"{r.record.id} ({flag_words(r.summary)})" for r in rows if _flags(r.summary)]
        shown = sorted(
            (r for r in rows if r.record.id in outcome.written),
            key=lambda r: (not _flags(r.summary), -r.record.moved_mm[0]),
        )[:OVERLAYS]
        drawn = ", ".join(f"{r.record.id} ({r.record.moved_mm[0]:.2f})" for r in shown)
        relative = _blocks.job_path(lctx, call.folder)
        left = [
            f"{r.record.id} ({r.record.estimated['left_out_fraction']:.0%})"
            for r in rows
            if r.record.estimated and r.record.estimated["left_out_fraction"] is not None
        ]
        energy = (
            f"emlddmm's energy record: {relative}/energy.csv (plot: {relative}/energy.png)"
            if result.energy_captured
            else "emlddmm's energy record was not captured"
        )
        waited = result.gpu_wait_s
        text = (
            f"{call.note}: written for {len(outcome.written)} of {len(rows)} sections"
            + (
                f"; it waited {round(waited)} s for the machine's GPU queue"
                if waited >= 1.0
                else ""
            )
            + (f"; not (fully) written: {', '.join(problems)}" if problems else "")
            + (f"; warp flags: {', '.join(flagged)}" if flagged else "")
            + (
                f"; the sorting emlddmm estimated left out, of the tissue, {', '.join(left)}, "
                "saved with each deformation"
                if left
                else ""
            )
            + (
                f". Overlays, with how far each moved (median, mm): {drawn}"
                if call.view and shown
                else ""
            )
            + f". {energy}; per-section results: {relative}/sections.csv; "
            f"the images it saw: {relative}/images/."
        )
        pictures = (
            [
                lctx.caption(lctx.overlay_picture(r.record.id), f"{r.record.id}: after emlddmm")
                for r in shown
            ]
            if call.view
            else []
        )
        return ToolResult(
            {
                "notice": text,
                "files": relative,
                "settings": call.settings,
                **({"gpu_wait_s": round(waited)} if waited >= 1.0 else {}),
                "rows": [
                    {
                        "id": r.record.id,
                        "inputs": r.record.placed.section.inputs,
                        "estimated": r.record.estimated,
                    }
                    for r in rows
                ],
            },
            pictures,
        )


def _start(ctx: ToolContext, call: _StackCall) -> ToolResult:
    """Start the stack's fit as background work and answer at once."""
    stack = call.stack
    try:
        work = ctx.start_work([placed.id for placed in stack.sections], _StackLanding(call))
    except (Refused, *ENGINE_FAILURES) as exc:
        payload = (
            exc.payload()
            if isinstance(exc, Refused)
            else {"error": RefusalCode.ENGINE_UNAVAILABLE, "message": str(exc)}
        )
        details = {k: v for k, v in payload.items() if k not in ("status", "error", "message")}
        raise ToolError(
            str(payload.get("error")), str(payload.get("message") or exc), **details
        ) from exc
    other = [
        {"id": sid, "gets": GETS[route]}
        for sid, route in call.routes.items()
        if route != "transform"
    ]
    device = call.options.device
    return ToolResult(
        {
            "work": work,
            "stack": len(stack.sections),
            "left_out": stack.left_out,
            "settings": call.settings,
            "rows": [
                {"id": placed.id, "inputs": placed.section.inputs, "estimated": None}
                for placed in stack.sections
            ],
            **({"other_routes": other} if other else {}),
            "largest_snap_um": round(max(abs(p.snap_um) for p in stack.sections), 1),
            "device": device + (" (the fit takes tens of minutes)" if device == "cpu" else ""),
            "files": _blocks.job_path(ctx, call.folder),
        }
    )


def _stack_tool(
    ctx: ToolContext,
    options: Options,
    ask: _Ask,
    kernel_um: float,
    fit: str,
    warp_freedom: float,
    iterations: int,
    sorting: dict[str, Any],
    view: bool,
) -> ToolResult:
    if fit not in FITS:
        raise ToolError(RefusalCode.BAD_ARGUMENT, f"fit is one of {', '.join(FITS)}; got {fit!r}")
    rigid, warp = fit in ("transform", "both"), fit in ("warp", "both")
    steps = _check_iterations(iterations)
    sort = _check_sorting(sorting)
    options = options.resolved()
    stack = _stack(ctx, options, ask)
    schedule = _schedule(stack, steps, kernel_um, warp_freedom, rigid=rigid, warp=warp)
    lattices = [_lattice(placed) for placed in stack.sections]
    request = _request(
        options,
        ctx.atlas().name,
        stack,
        schedule,
        sorting=Sorting.from_given(sort),
        mode="stack",
        read_warp=warp,
        points=tuple(lattice.points for lattice in lattices),
    )
    settings = {
        "section_image": ask.section_image,
        "atlas_image": ask.atlas_image,
        "mask": ask.mask,
        "kernel_um": kernel_um,
        "fit": fit,
        "slot_um": ask.slot_um,
        "warp_freedom": warp_freedom,
        "iterations": int(iterations),
        "sorting": sort,
    }
    call = _StackCall(
        options=options,
        stack=stack,
        request=request,
        lattices=lattices,
        settings=settings,
        asked={"sections": ask.names, **settings, "view": view},
        chosen=_chosen(stack, options, schedule),
        routes={
            placed.id: _route(placed.section.view, rigid=rigid, warp=warp)
            for placed in stack.sections
        },
        digests={
            placed.id: ctx.section_inputs(placed.id, deformation=True) for placed in stack.sections
        },
        rigid=rigid,
        warp=warp,
        view=view,
        folder=_blocks.call_folder(ctx, "stack"),
        note=f"emlddmm stack fit, atlas {stack.resolved}, {len(stack.sections)} sections",
    )
    return _start(ctx, call)


@dataclass(frozen=True)
class _FoundPlane:
    """Hold the plane emlddmm found for the stack.

    Parameters
    ----------
    pitch_deg, yaw_deg
        The stack's cutting angles.
    positions
        Each section's plane at the found affine (position, angles).
    scale, shift_mm
        The spacing's stretch and shift as a whole.
    """

    pitch_deg: float
    yaw_deg: float
    positions: list[dict[str, float]]
    scale: float
    shift_mm: float


def _found_plane(ctx: ToolContext, stack: _Stack, result: engine.FitResult) -> _FoundPlane:
    """Read the found plane of every section from the fit's affine."""
    atlas = ctx.atlas()
    centre = (np.asarray(atlas.shape, float) - 1) / 2.0 * np.asarray(atlas.resolution_um)
    inverse = np.linalg.inv(result.affine)
    found = [
        ctx.angles_from_plane(
            result.affine[0, :3].tolist(),
            ((inverse @ np.array([z, 0.0, 0.0, 1.0]))[:3] + centre).tolist(),
        )
        for z in stack.depths_um
    ]
    old = np.array([float(p.section.view.position_mm) for p in stack.sections])
    new = np.array([f["position_mm"] for f in found])
    scale, shift_mm = (
        np.polyfit(old, new, 1) if np.ptp(old) > 0 else (1.0, float(np.mean(new - old)))
    )
    return _FoundPlane(
        found[0]["pitch_deg"], found[0]["yaw_deg"], found, float(scale), float(shift_mm)
    )


def _plane_tool(
    ctx: ToolContext,
    options: Options,
    ask: _Ask,
    iterations: int,
    view: bool,
) -> ToolResult:
    steps = _check_iterations(iterations)
    options = options.resolved()
    stack = _stack(ctx, options, ask)
    schedule = _schedule(
        stack, steps, EXAMPLE.kernel_um, EXAMPLE.warp_freedom, rigid=True, warp=False
    )
    folder = _blocks.call_folder(ctx, "plane")
    request = _request(
        options,
        ctx.atlas().name,
        stack,
        schedule,
        sorting=Sorting(),
        mode="plane",
        read_warp=False,
        points=tuple(np.zeros((0, 2)) for _ in stack.sections),
    )
    result = _run_engine(request, folder)
    plane = _found_plane(ctx, stack, result)
    secs = stack.sections
    pitch0, yaw0 = (float(v) for v in secs[0].section.view.angles)
    old = np.array([float(p.section.view.position_mm) for p in secs])
    a2d = result.slice_affines[[p.slot for p in secs]]
    turns = np.degrees(np.arctan2(a2d[:, 1, 0], a2d[:, 0, 0]))
    shifts = a2d[:, :2, 2] / UM_PER_MM
    turns, shifts = turns - np.median(turns), shifts - np.median(shifts, axis=0)
    rows = [
        {
            "id": placed.id,
            "inputs": placed.section.inputs,
            "estimated": None,
            "position_mm": round(f["position_mm"], 3),
            "change_mm": round(f["position_mm"] - float(was), 3),
            "residual": {"turn_deg": round(float(t), 2), "shift_mm": round(float(np.hypot(*s)), 3)},
        }
        for placed, f, was, t, s in zip(secs, plane.positions, old, turns, shifts, strict=True)
    ]
    asked = {
        "sections": ask.names,
        "slot_um": ask.slot_um,
        "section_image": ask.section_image,
        "atlas_image": ask.atlas_image,
        "iterations": int(iterations),
        "view": view,
    }
    proposal = {
        "pitch_deg": plane.pitch_deg,
        "yaw_deg": plane.yaw_deg,
        "spacing_scale": plane.scale,
        "spacing_shift_mm": plane.shift_mm,
    }
    _write_files(
        stack,
        result,
        folder,
        _FileRecord(
            asked,
            _chosen(stack, options, schedule),
            options.settings(),
            {"proposed_plane": proposal, "sections": rows},
        ),
    )
    pictures: list[Image.Image] = []
    if view:
        picks = {int(round(v)) for v in np.linspace(0, len(secs) - 1, min(OVERLAYS, len(secs)))}
        for j in sorted(picks):
            found = plane.positions[j]
            caption = (
                f"{rows[j]['id']}: atlas at the plane emlddmm found (pitch {plane.pitch_deg:.2f}, "
                f"yaw {plane.yaw_deg:.2f}), not written"
            )
            overlay = ctx.overlay_picture(
                str(rows[j]["id"]),
                proposal={
                    "position_mm": found["position_mm"],
                    "angles": (plane.pitch_deg, plane.yaw_deg),
                },
            )
            pictures.append(ctx.caption(overlay, caption))
    return ToolResult(
        {
            "pitch_deg": round(plane.pitch_deg, 2),
            "yaw_deg": round(plane.yaw_deg, 2),
            "change": {
                "pitch_deg": round(plane.pitch_deg - pitch0, 2),
                "yaw_deg": round(plane.yaw_deg - yaw0, 2),
            },
            "spacing": {"scale": round(plane.scale, 4), "shift_mm": round(plane.shift_mm, 3)},
            "rows": rows,
            "left_out": stack.left_out,
            **({} if result.energy_captured else {"energy": "not captured"}),
            "files": _blocks.job_path(ctx, folder),
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
        ``emlddmm_stack`` and ``emlddmm_plane``.

    Raises
    ------
    PluginError
        For an option the plugin does not take.
    """
    options = Options.from_config(config)
    meta = {
        "method": "emlddmm (Tward et al. 2025, Nat Commun)",
        "commit": "9c4dc80 + e426ece + 6ff62b0",
        "engine": "langslice.engines.emlddmm",
    }

    def stack(
        ctx: ToolContext,
        sections: list[str],
        slot_um: float,
        section_image: str = "preprocessed",
        atlas_image: str = "nissl",
        mask: str = "none",
        kernel_um: float = 500.0,
        fit: str = "both",
        warp_freedom: float = 10000.0,
        iterations: int = 100,
        sorting: dict[str, Any] | None = None,
        view: bool = True,
    ) -> ToolResult:
        ask = _Ask(
            list(dict.fromkeys(sections)),
            float(slot_um),
            section_image,
            atlas_image,
            mask,
            need_transform=True,
        )
        return _stack_tool(
            ctx,
            options,
            ask,
            float(kernel_um),
            str(fit),
            float(warp_freedom),
            iterations,
            dict(sorting or {}),
            view,
        )

    def plane(
        ctx: ToolContext,
        sections: list[str],
        slot_um: float,
        section_image: str = "preprocessed",
        atlas_image: str = "nissl",
        iterations: int = 100,
        view: bool = True,
    ) -> ToolResult:
        ask = _Ask(
            list(dict.fromkeys(sections)),
            float(slot_um),
            section_image,
            atlas_image,
            "none",
            need_transform=False,
        )
        return _plane_tool(ctx, options, ask, iterations, view)

    return [
        engine_tool(
            "emlddmm_stack",
            emlddmm_stack,
            stack,
            description=_stack_description,
            version="1",
            meta=meta,
        ),
        engine_tool(
            "emlddmm_plane",
            emlddmm_plane,
            plane,
            description=_plane_description,
            version="1",
            meta=meta,
        ),
    ]


__all__ = ["tools"]
