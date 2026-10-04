"""Section pixels to atlas micrometres: the linear frame, the residual and the maps.

The geometry behind the job folder's public files (formats phase,
2026-10-04): ``registration.json`` and, per section, ``coords.tif``,
``labels.tif`` and ``residual.tif`` (written by :mod:`langslice.job.formats`).
One path from the stored parameters to the atlas, the one every picture and
fit uses: the six stored numbers are placed by
:func:`langslice.core.handoff.linear_placement_matrix` on the section's
:data:`~langslice.core.sections.PREVIEW_LONG_EDGE` working frame (the frame
they are normalized against), the native atlas plane is mapped to BrainGlobe
micrometres by :func:`langslice.core.oblique.plane_index_affine` times the
resolution (as :func:`langslice.core.layers.frame_record` does for a
picture), and an applied deformation is the record's own field and placement
(:class:`langslice.core.deformable.record.DeformableRecord`), composed exactly as
its pictures compose it (residual first, then the linear placement undone).

Conventions (Nash 2026-10-03, BrainGlobe's): atlas micrometres in the atlas's
own axis order, voxel ``i``'s centre at ``i * resolution``; image pixels are
``[row, col]`` with pixel centres at integers, row 0 the top of the image
FILE as stored (no rotation or flip applied: those are part of the mapping).

- :class:`SectionFrame` / :func:`section_frame`: one placed section's linear
  map, ``pixel_to_atlas_um(size)`` for the file's pixels or any resize of
  them (memoized on the workspace, so every write can rewrite
  ``registration.json`` cheaply).
- :func:`native_points`: native atlas-plane ``(x, y)`` of section points,
  linear or through an applied deformation.
- :func:`section_maps`: the per-pixel maps on a grid (the working copy by
  default, or the file's own pixels): coordinates (NaN outside the footprint or
  the atlas volume), atlas ids (nearest neighbour on the native plane, as the
  pictures' labels layer and the deformable record's labels), and the
  residual ``d`` such that ``coords[p] = pixel_to_atlas_um @ [p + d(p), 1]``.
- :func:`residual_markers`: VisuAlign-style ``[x, y, nx, ny]`` markers of
  the residual on a regular grid, in the file's continuous pixel coordinates.
"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, cast

import numpy as np

from langslice.core.affine import pixel_center_map
from langslice.core.handoff import linear_placement_matrix
from langslice.core.image_prep import prepared_size, working_size
from langslice.core.sections import PREVIEW_LONG_EDGE
from langslice.core.space import Plane

if TYPE_CHECKING:
    from langslice.core.deformable.record import DeformableRecord
    from langslice.core.state import SliceState, StackState
    from langslice.core.workspace import Workspace

#: ``[x, y, 1]`` <-> ``[row, col, 1]``.
SWAP = np.array([[0.0, 1.0, 0.0], [1.0, 0.0, 0.0], [0.0, 0.0, 1.0]])
#: Rows per block when a map is computed (bounds the float64 temporaries).
BLOCK_ROWS = 256
#: Marker grid of :func:`residual_markers`: 1/36 of the file's long edge,
#: 32 px at least (the density ``nonlinear.border_registration`` exports).
MARKER_DIVISIONS = 36
MARKER_MIN_SPACING_PX = 32


def rowcol(matrix: np.ndarray) -> np.ndarray:
    """A 3x3 on ``[x, y, 1]`` as the same map on ``[row, col, 1]``."""
    return SWAP @ np.asarray(matrix, dtype=np.float64) @ SWAP


def apply(matrix: np.ndarray, x: np.ndarray, y: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """A 3x3 affine on ``[x, y, 1]`` applied to coordinate arrays."""
    m = np.asarray(matrix, dtype=np.float64)
    return m[0, 0] * x + m[0, 1] * y + m[0, 2], m[1, 0] * x + m[1, 1] * y + m[1, 2]


def orientation_matrix(size: tuple[int, int], rotation_deg: int, flip: bool) -> np.ndarray:
    """3x3 on ``[x, y, 1]``: pixel centres of an unturned render of *size*
    (width, height) onto the render as turned and flipped
    (:func:`langslice.core.sections.render_slice`: ROTATE first, then FLIP).
    PIL's quarter turns are counter-clockwise."""
    width, height = float(size[0]), float(size[1])
    rotation = int(rotation_deg) % 360
    if rotation == 90:
        turn = np.array([[0.0, 1.0, 0.0], [-1.0, 0.0, width - 1.0], [0.0, 0.0, 1.0]])
        turned_width = height
    elif rotation == 180:
        turn = np.array([[-1.0, 0.0, width - 1.0], [0.0, -1.0, height - 1.0], [0.0, 0.0, 1.0]])
        turned_width = width
    elif rotation == 270:
        turn = np.array([[0.0, -1.0, height - 1.0], [1.0, 0.0, 0.0], [0.0, 0.0, 1.0]])
        turned_width = height
    else:
        turn = np.eye(3)
        turned_width = width
    if flip:
        turn = np.array([[-1.0, 0.0, turned_width - 1.0], [0.0, 1.0, 0.0],
                         [0.0, 0.0, 1.0]]) @ turn
    return turn


def unturned(size: tuple[int, int], rotation_deg: int) -> tuple[int, int]:
    """The size before a quarter turn of *rotation_deg* gave *size*."""
    return (size[1], size[0]) if int(rotation_deg) % 180 == 90 else (size[0], size[1])


@dataclass(frozen=True)
class SectionFrame:
    """One placed section's linear map from its image file to the atlas.

    ``file_to_render`` takes the file's pixel centres (``[x, y, 1]``) onto the
    oriented working frame the stored transform is normalized against
    (``render_size``, at ``render_um_per_px``); ``atlas_to_render`` takes the
    native atlas plane onto that frame
    (:func:`~langslice.core.handoff.linear_placement_matrix`);
    ``native_to_um`` takes a native plane pixel ``[row, col, 1]`` to atlas
    micrometres. ``annotation`` is the native plane's atlas ids (shared,
    read only).
    """

    section_id: str
    file_size: tuple[int, int]
    working_size: tuple[int, int]
    working_factor: float
    render_size: tuple[int, int]
    render_um_per_px: float
    file_um_per_px: float
    calibration_source: str
    position_mm: float
    plane: str
    pitch_deg: float
    yaw_deg: float
    rotation_deg: int
    flip: bool
    file_to_render: np.ndarray
    atlas_to_render: np.ndarray
    native_to_index: np.ndarray
    native_to_um: np.ndarray
    annotation: np.ndarray
    atlas: dict[str, Any]

    def file_to_native(self) -> np.ndarray:
        """3x3 on ``[x, y, 1]``: file pixel centres -> native plane pixel centres."""
        return np.linalg.inv(self.atlas_to_render) @ self.file_to_render

    def grid_to_file(self, size: tuple[int, int] | None = None) -> np.ndarray:
        """3x3 on ``[x, y, 1]``: a resize of the file to *size* -> file pixels."""
        if size is None or tuple(size) == tuple(self.file_size):
            return np.eye(3)
        return pixel_center_map(tuple(size), self.file_size)  # type: ignore[arg-type]

    def grid_to_native(self, size: tuple[int, int] | None = None) -> np.ndarray:
        """3x3 on ``[x, y, 1]``: grid pixel centres -> native plane pixel centres."""
        return self.file_to_native() @ self.grid_to_file(size)

    def pixel_to_atlas_um(self, size: tuple[int, int] | None = None) -> np.ndarray:
        """3x3: a pixel ``[row, col, 1]`` of the file (or of its resize to
        *size*, width and height) -> atlas micrometres, the linear placement."""
        return self.native_to_um @ rowcol(self.grid_to_native(size))

    def grid_um_per_px(self, size: tuple[int, int] | None = None) -> float:
        """Micrometres per pixel (along x) of the file resized to *size*."""
        if size is None:
            return self.file_um_per_px
        return self.file_um_per_px * self.file_size[0] / float(size[0])


def _file_stamp(path: str) -> tuple[int, int] | None:
    try:
        info = os.stat(path)
    except OSError:
        return None
    return (info.st_size, info.st_mtime_ns)


#: The six stored numbers of a section without a transform: the identity,
#: as every placement picture draws it (``core.placement.stored_placement``).
IDENTITY_PARAMS = (1.0, 0.0, 0.0, 0.0, 1.0, 0.0)


def stored_params(record: SliceState) -> tuple[float, ...]:
    """The section's six stored numbers (the identity without a transform)."""
    transform = record.transform or {}
    params = transform.get("params") if transform else IDENTITY_PARAMS
    return tuple(float(v) for v in params)  # type: ignore[union-attr]


def placement_problem(state: StackState, record: SliceState) -> str | None:
    """Why *record* has no linear map yet (None: it has one). A section
    without a transform is placed by the identity, as its pictures show it."""
    if record.position_mm is None or not np.isfinite(record.position_mm):
        return "no position"
    transform = record.transform or {}
    if not transform:
        return None
    if transform.get("spline"):
        return "a landmark spline (not an affine placement)"
    if transform.get("stale"):
        return "the stored transform is marked stale"
    params = transform.get("params")
    try:
        values = np.asarray(params, dtype=np.float64)
    except (TypeError, ValueError):
        return "the stored transform has no six numbers"
    if values.shape != (6,) or not np.isfinite(values).all():
        return "the stored transform has no six numbers"
    return None


def section_frame(state: StackState, workspace: Workspace, record: SliceState) -> SectionFrame:
    """The section's linear frame (``ValueError`` when it has none:
    :func:`placement_problem`). Memoized on the workspace
    (``Workspace.frame_cache``) by everything it depends on."""
    problem = placement_problem(state, record)
    if problem is not None:
        raise ValueError(f"{record.id}: {problem}")
    if state.atlas != workspace.spec.atlas or state.plane != workspace.spec.plane:
        raise ValueError("State and workspace disagree about the atlas or plane")
    transform = record.transform or {}
    path = workspace.image_path(record.id)
    spec = workspace.spec
    key = (
        record.id, _file_stamp(path), bool(record.flip), int(record.rotation_deg),
        float(record.position_mm),  # type: ignore[arg-type]
        state.atlas, state.plane, state.pitch_deg, state.yaw_deg,
        stored_params(record),
        json.dumps(transform.get("calibration"), sort_keys=True, default=str),
        json.dumps((spec.inputs or {}).get("pixel_size_um"), default=str),
        json.dumps(spec.host_preprocessing, sort_keys=True, default=str),
    )
    held = workspace.frame_cache.get(key)
    if held is None:
        held = _section_frame(state, workspace, record)
        workspace.frame_cache[key] = held
    return cast(SectionFrame, held)


def _section_frame(state: StackState, workspace: Workspace, record: SliceState) -> SectionFrame:
    from PIL import Image

    from langslice.core.layers import atlas_facts
    from langslice.core.oblique import plane_index_affine

    path = workspace.image_path(record.id)
    with Image.open(path) as handle:
        file_size = (int(handle.size[0]), int(handle.size[1]))
    cached = workspace.source_cache.get(record.id)
    if cached is not None:
        working, factor = (int(cached[0].size[0]), int(cached[0].size[1])), float(cached[1])
    else:
        found = working_size(path, pages=workspace.spec.host_preprocessing is not None)
        if found is None:
            source, factor = workspace.working_source(record.id)
            working = (int(source.size[0]), int(source.size[1]))
        else:
            working, factor = found
    unturned_render = prepared_size(working, max_long_edge=PREVIEW_LONG_EDGE)
    render_scale = factor * working[0] / float(unturned_render[0])
    file_um, source = workspace.calibration(record.id)
    transform = record.transform or {}
    if file_um is not None:
        render_um = float(file_um) * render_scale
    else:
        try:
            render_um = float((transform.get("calibration") or {})["section_um_per_px"])
        except (KeyError, TypeError, ValueError) as exc:
            raise ValueError(f"{record.id}: no pixel size (neither the file, the host nor "
                             "the stored transform gives one)") from exc
        source = str((transform.get("calibration") or {}).get("source", "stored"))
        file_um = render_um / render_scale
    if not np.isfinite(render_um) or render_um <= 0:
        raise ValueError(f"{record.id}: the pixel size must be finite and positive")
    rotation = int(record.rotation_deg)
    orient = orientation_matrix(unturned_render, rotation, bool(record.flip))
    render_size = ((unturned_render[1], unturned_render[0]) if rotation % 180 == 90
                   else unturned_render)
    file_to_render = orient @ pixel_center_map(file_size, unturned_render)
    plane = cast(Plane, state.plane)
    position = float(record.position_mm)  # type: ignore[arg-type]
    atlas = workspace.atlas
    atlas_to_render, geometry = linear_placement_matrix(
        render_size, render_um, atlas, position, plane, state.pitch_deg, state.yaw_deg,
        stored_params(record))
    facts = atlas_facts(atlas)
    index = plane_index_affine(atlas, position, plane, state.pitch_deg, state.yaw_deg)
    to_um = np.diag(facts["resolution_um"]) @ index
    return SectionFrame(
        section_id=record.id, file_size=file_size, working_size=working,
        working_factor=float(factor), render_size=render_size,
        render_um_per_px=float(render_um), file_um_per_px=float(file_um),
        calibration_source=str(source or ""), position_mm=position, plane=state.plane,
        pitch_deg=state.pitch_deg, yaw_deg=state.yaw_deg, rotation_deg=rotation,
        flip=bool(record.flip), file_to_render=file_to_render,
        atlas_to_render=atlas_to_render, native_to_index=index, native_to_um=to_um,
        annotation=np.asarray(geometry.annotation), atlas=facts,
    )


# --- through a deformation ------------------------------------------------------------


def native_points(
    frame: SectionFrame, warp: DeformableRecord | None, x: np.ndarray, y: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Native atlas-plane ``(x, y)`` of file points ``(x, y)`` (pixel centres
    at integers; 2-D arrays). Without *warp* the linear placement; with it,
    the record's own: the point carried onto its fit grid (the oriented
    render it was fitted on), displaced by its field (bilinear, edges
    replicated, as its pictures sample it), then its placement undone."""
    if warp is None:
        return apply(frame.file_to_native(), x, y)
    import cv2

    fit_size = (int(warp.section_size[0]), int(warp.section_size[1]))
    fit_unturned = unturned(fit_size, frame.rotation_deg)
    to_fit = (orientation_matrix(fit_unturned, frame.rotation_deg, frame.flip)
              @ pixel_center_map(frame.file_size, fit_unturned))
    qx, qy = apply(to_fit, x, y)
    map_x = np.ascontiguousarray(qx, dtype=np.float32)
    map_y = np.ascontiguousarray(qy, dtype=np.float32)
    scale = 1.0 / warp.mm_per_px
    dx = cv2.remap(np.ascontiguousarray(warp.field_mm[..., 0], dtype=np.float32), map_x,
                   map_y, cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE)
    dy = cv2.remap(np.ascontiguousarray(warp.field_mm[..., 1], dtype=np.float32), map_x,
                   map_y, cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE)
    px = qx + dx.astype(np.float64) * scale
    py = qy + dy.astype(np.float64) * scale
    return apply(np.linalg.inv(warp.placement.atlas_to_section), px, py)


# --- the per-pixel maps -----------------------------------------------------------------


@dataclass(frozen=True)
class SectionMaps:
    """One section's maps on one grid (the file resized to ``grid_size``).

    ``coords`` (rows, cols, 3) float32 atlas micrometres, NaN outside the
    section's footprint or the atlas volume; ``labels`` (rows, cols) uint32
    atlas ids, 0 there; ``residual`` (rows, cols, 2) float32 ``(drow, dcol)``
    in grid pixels, None without a deformation, such that ``coords[r, c] =
    pixel_to_atlas_um @ [r + drow, c + dcol, 1]``; ``footprint`` the filled
    outline the maps cover, ``tissue`` the threshold's own tissue estimate
    (saved as ``tissue.png`` for scripts that mask with it).
    """

    section_id: str
    grid_size: tuple[int, int]
    full_resolution: bool
    um_per_px: float
    pixel_to_atlas_um: np.ndarray
    coords: np.ndarray
    labels: np.ndarray
    residual: np.ndarray | None
    footprint: np.ndarray
    tissue: np.ndarray
    tissue_found: bool


#: A gap in the section's outline narrower than twice this is closed when
#: the footprint is drawn (a tear, a fissure, a fringe of dim white matter
#: the threshold dropped at the edge of a ventricle). Measured 2026-10-04 on
#: LSD_910 M02_B_06 (fluorescent, dim fibre tracts beside enlarged
#: ventricles, two tears reaching the ventral surface): 0.03 mm left a dark
#: band beside the hippocampus open, 0.06 mm left the tears as notches,
#: 0.12 mm closed both and kept the real ventral notches; 0.4 mm began to
#: fill those.
FOOTPRINT_CLOSING_MM = 0.15


def section_footprint(
    workspace: Workspace, frame: SectionFrame, size: tuple[int, int],
) -> tuple[np.ndarray, np.ndarray, bool]:
    """``(footprint, tissue, found)`` on a grid of *size* (nearest).

    *tissue* is the foreground rule of the deformable fit
    (:func:`langslice.core.deformable.masks.tissue_masks`, its raw mask) on the
    working copy; dim tissue (fibre tracts, white matter) may fall outside
    it. *footprint* is the section's filled outline: that mask closed over
    gaps of up to ``2 * FOOTPRINT_CLOSING_MM`` and every hole filled, so
    nothing inside the outline is cut, dark regions and tears included.
    Every pixel when no tissue separates from the background (*found*
    False)."""
    import cv2
    from scipy import ndimage as ndi

    from langslice.core.deformable.masks import tissue_masks

    source, _factor = workspace.working_source(frame.section_id)
    try:
        _filled, raw = tissue_masks(source)
    except ValueError:
        full = np.ones((size[1], size[0]), dtype=bool)
        return full, full, False
    working_um = frame.file_um_per_px * frame.working_factor
    radius = max(1, int(round(FOOTPRINT_CLOSING_MM * 1000.0 / working_um)))
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * radius + 1,) * 2)
    padded = np.pad(raw.astype(np.uint8), radius + 1)
    closed = cv2.morphologyEx(padded, cv2.MORPH_CLOSE, kernel)
    footprint = ndi.binary_fill_holes(closed[radius + 1:-radius - 1, radius + 1:-radius - 1] > 0)

    def on_grid(mask: np.ndarray) -> np.ndarray:
        if mask.shape[::-1] != tuple(size):
            mask = cv2.resize(mask.astype(np.uint8), tuple(size),
                              interpolation=cv2.INTER_NEAREST) > 0
        return np.asarray(mask, dtype=bool)

    return on_grid(np.asarray(footprint)), on_grid(raw), True


def section_maps(
    workspace: Workspace, frame: SectionFrame, warp: DeformableRecord | None, *,
    full_resolution: bool = False,
) -> SectionMaps:
    """The section's coordinate, label and residual maps (see
    :class:`SectionMaps`) on its working copy's grid (what every picture is
    drawn from), or with *full_resolution* on the file's own pixels."""
    from langslice.core.deformable.geometry import sample_native

    size = frame.file_size if full_resolution else frame.working_size
    width, height = int(size[0]), int(size[1])
    footprint, tissue, found = section_footprint(workspace, frame, (width, height))
    to_file = frame.grid_to_file((width, height))
    grid_to_native = frame.grid_to_native((width, height))
    native_to_grid = np.linalg.inv(grid_to_native)
    shape = np.asarray(frame.atlas["shape"], dtype=np.float64)
    coords = np.full((height, width, 3), np.nan, dtype=np.float32)
    labels = np.zeros((height, width), dtype=np.uint32)
    residual = np.zeros((height, width, 2), dtype=np.float32) if warp is not None else None
    cols = np.arange(width, dtype=np.float64)
    for r0 in range(0, height, BLOCK_ROWS):
        r1 = min(height, r0 + BLOCK_ROWS)
        gy, gx = np.meshgrid(np.arange(r0, r1, dtype=np.float64), cols, indexing="ij")
        fx, fy = apply(to_file, gx, gy)
        nx, ny = native_points(frame, warp, fx, fy)
        index = (frame.native_to_index[:, 0:1, None] * ny[None]
                 + frame.native_to_index[:, 1:2, None] * nx[None]
                 + frame.native_to_index[:, 2:3, None])
        inside = np.all((index >= -0.5) & (index < shape[:, None, None] - 0.5), axis=0)
        keep = inside & footprint[r0:r1]
        um = (frame.native_to_um[:, 0:1, None] * ny[None] + frame.native_to_um[:, 1:2, None]
              * nx[None] + frame.native_to_um[:, 2:3, None])
        block = np.moveaxis(um, 0, -1).astype(np.float32)
        block[~keep] = np.nan
        coords[r0:r1] = block
        ids = sample_native(frame.annotation, np.stack([nx, ny], axis=-1)).astype(np.uint32)
        ids[~keep] = 0
        labels[r0:r1] = ids
        if residual is not None:
            lx, ly = apply(native_to_grid, nx, ny)
            residual[r0:r1, :, 0] = (ly - gy).astype(np.float32)
            residual[r0:r1, :, 1] = (lx - gx).astype(np.float32)
    return SectionMaps(
        section_id=frame.section_id, grid_size=(width, height),
        full_resolution=bool(full_resolution), um_per_px=frame.grid_um_per_px((width, height)),
        pixel_to_atlas_um=frame.pixel_to_atlas_um((width, height)), coords=coords,
        labels=labels, residual=residual, footprint=footprint, tissue=tissue,
        tissue_found=found,
    )


def residual_markers(frame: SectionFrame, warp: DeformableRecord) -> list[list[float]]:
    """VisuAlign markers of the residual: ``[x, y, nx, ny]`` per point of a
    regular grid over the file (1/36 of its long edge apart, 32 px at
    least, both edges included), in the file's continuous pixel coordinates
    (the top-left corner at 0; a pixel centre at index + 0.5).

    A section point ``(nx, ny)`` corresponds to the point ``(x, y)`` of the
    linearly anchored atlas plane (VisuAlign's direction: its triangulation
    is built on ``(nx, ny)`` and maps them to ``(x, y)``, which the
    anchoring then takes into the atlas)."""
    width, height = frame.file_size
    spacing = max(MARKER_MIN_SPACING_PX, round(max(width, height) / MARKER_DIVISIONS))
    xs = np.unique(np.append(np.arange(0, width, spacing), width - 1)).astype(np.float64)
    ys = np.unique(np.append(np.arange(0, height, spacing), height - 1)).astype(np.float64)
    gx, gy = np.meshgrid(xs, ys)
    nx, ny = native_points(frame, warp, gx, gy)
    lx, ly = apply(np.linalg.inv(frame.file_to_native()), nx, ny)
    markers = np.stack([lx + 0.5, ly + 0.5, gx + 0.5, gy + 0.5], axis=-1).reshape(-1, 4)
    if not np.isfinite(markers).all():
        raise ValueError(f"{frame.section_id}: the residual markers are not finite")
    return [[float(v) for v in row] for row in markers]
