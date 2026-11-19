"""Preparing the stack: each section read from its fit frame onto the stack's grid."""

from __future__ import annotations

import math
from dataclasses import dataclass
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
from langslice.engines.emlddmm import engine
from langslice.engines.emlddmm.settings import (
    ANGLE_TOLERANCE_DEG,
    GEOMETRY,
    MEMORY,
    NEEDS_TRANSFORM,
    UM_PER_MM,
    Array,
    Options,
)
from langslice.plugin_api import (
    FitFrame,
    RefusalCode,
    SectionView,
    ToolContext,
    ToolError,
)


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
class Ask:
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


def _read_section(ctx: ToolContext, view: SectionView, ask: Ask, section_um: float) -> _Section:
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
class Placed:
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
class Stack:
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

    sections: list[Placed]
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
    ctx: ToolContext, options: Options, ask: Ask
) -> tuple[list[_Section], list[dict[str, Any]]]:
    """Read every section that can enter the stack; a section that cannot (no placement) is left
    out with its reason."""
    left_out: list[dict[str, Any]] = []
    sections: list[_Section] = []
    for view in (
        [ctx.reads.section(name) for name in ask.names] if ask.names else ctx.reads.sections()
    ):
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


def build_stack(ctx: ToolContext, options: Options, ask: Ask) -> Stack:
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
        placed.append(Placed(section, off, int(k), float((k * spacing - depths[i]) / cos), tissue))
    return Stack(
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
