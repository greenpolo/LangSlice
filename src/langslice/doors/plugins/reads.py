"""The reads of a plugin context: sections, their images and layers.

:class:`ContextReads` is the base of
:class:`langslice.doors.plugins.context.ToolContext`: it holds the run, the
caller and the call, finds a section by filename, and reads what the job
holds of it (:class:`SectionView`, the images by name, a
:class:`SectionRaster`, the layers).
"""

from __future__ import annotations

import copy
from collections.abc import Sequence
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

import numpy as np
from PIL import Image

from langslice.core.pictures.sizes import PICTURE_EDGES
from langslice.core.plugins import ToolError
from langslice.core.refusal import RefusalCode, Refused
from langslice.core.stack.codec import deformation_to_json, physical_to_json
from langslice.core.stack.section import Section
from langslice.doors.plugins.caller import Caller
from langslice.job.settings import frozen as freeze
from langslice.tools.refusal import unknown_sections
from langslice.tools.stale import section_inputs

if TYPE_CHECKING:
    from langslice.tools.run import Run


@dataclass(frozen=True)
class AffineView:
    """A section's in-plane transform: ``kind`` (``interactive``, ``elastix``,
    ``plugin:<tool>``...), the six normalized numbers ``[a, b, tx, c, d, ty]``
    (``docs/reference/file_formats.md``), the knobs they are in (``rotation_deg``,
    ``scale_x``, ``scale_y``, ``shear``, ``translate_x_mm``, ``translate_y_mm``,
    ``pivot``), and whether it mirrors the section."""

    kind: str | None
    matrix: tuple[float, ...] | None
    knobs: dict[str, Any] | None
    mirrored: bool | None


@dataclass(frozen=True)
class SectionView:
    """One section as the job holds it now."""

    id: str
    #: Its place in the stack (0-based, in corrected order).
    index: int
    #: The image file's size in pixels, ``(width, height)``.
    file_size: tuple[int, int]
    pixel_size_um: float | None
    #: ``file``, ``host`` or ``""`` (neither says).
    pixel_size_source: str
    position_mm: float | None
    #: ``(pitch_deg, yaw_deg)`` of its atlas plane.
    angles: tuple[float, float]
    flip: bool
    rotation_deg: int
    affine: AffineView | None
    #: The applied deformation's summary (``key``, ``steps``...), or None.
    deformation: dict[str, Any] | None
    damaged_regions: tuple[str, ...]
    damage_note: str
    #: Names of the layers it shows.
    layers: tuple[str, ...]
    #: The properties frozen on it
    #: (``langslice.job.settings.frozen.PROPERTIES``).
    frozen: tuple[str, ...]


@dataclass(frozen=True)
class AtlasView:
    name: str
    #: Voxel size in micrometres along the atlas's axes.
    resolution_um: tuple[float, ...]
    orientation: str
    species: str
    plane: str
    shape: tuple[int, ...]


@dataclass(frozen=True)
class SectionRaster:
    """A section image together with its frame.

    ``array`` is the section as its orientation sets it (quarter turn, then
    left-right flip) and at most the asked size: its pixels are NOT the image
    file's. ``section_px`` (what layers' points and lines are in) is the file's
    own pixels, unturned, unflipped and full size. :meth:`to_section_px` and
    :meth:`from_section_px` convert coordinates both ways;
    :meth:`mask_to_section_px` turns a raster computed on ``array`` into one in
    the file's orientation (a raster covers the whole file whatever its size, so
    needs no scaling); ``Layer.labels(mask, frame="oriented")`` does that when
    the layer is added.
    """

    id: str
    array: np.ndarray
    #: The image file's size, ``(width, height)``.
    file_size: tuple[int, int]
    rotation_deg: int
    flip: bool

    def _matrix(self) -> np.ndarray:
        """3x3: oriented edge fractions to file edge fractions."""
        from langslice.core.fits.landmarks import affine_from_points
        from langslice.core.sections.native import unoriented_fractions

        corners = np.array([[0.0, 0.0], [1.0, 0.0], [0.0, 1.0]])
        x, y = unoriented_fractions(corners[:, 0], corners[:, 1], self.rotation_deg, self.flip)
        return affine_from_points(np.stack([x, y], axis=1))

    def to_section_px(self, xy: Any) -> np.ndarray:
        """``[x, y]`` pixels of ``array`` as ``section_px`` (file pixels)."""
        from langslice.core.fits.landmarks import transform

        height, width = self.array.shape[:2]
        fractions = (np.asarray(xy, dtype=np.float64) + 0.5) / np.array([width, height])
        theirs = transform(self._matrix(), fractions)
        return theirs * np.array(self.file_size) - 0.5

    def from_section_px(self, xy: Any) -> np.ndarray:
        """``section_px`` (file pixels) as ``[x, y]`` pixels of ``array``."""
        from langslice.core.fits.landmarks import transform

        height, width = self.array.shape[:2]
        fractions = (np.asarray(xy, dtype=np.float64) + 0.5) / np.array(self.file_size)
        return transform(np.linalg.inv(self._matrix()), fractions) * np.array(
            [width, height]) - 0.5

    def mask_to_section_px(self, raster: Any) -> np.ndarray:
        """A raster in ``array``'s orientation as one in the file's."""
        return unorient_raster(np.asarray(raster), self.rotation_deg, self.flip)


def unorient_raster(raster: np.ndarray, rotation_deg: int, flip: bool) -> np.ndarray:
    """*raster*, drawn as a section's orientation sets it, in the file's orientation."""
    from langslice.core.sections.native import unorient

    kind = "F" if np.issubdtype(raster.dtype, np.floating) else "I"
    image = Image.fromarray(raster.astype(np.float32 if kind == "F" else np.int32), mode=kind)
    return np.asarray(unorient(image, rotation_deg, flip)).astype(raster.dtype)


@dataclass(frozen=True)
class LayerRef:
    """A layer registered on a section."""

    id: str
    section: str
    name: str
    kind: str
    frame: str
    file: str


class ContextReads:
    """Read the job's sections for one plugin call or script.

    The classes stacked on this one (frames, proposals, pictures, writes,
    fits) share its footing, which they read as their own: ``_run`` (the
    run: job, workspace, spec), ``_caller``, ``call``, ``_record`` (a
    section by filename, after ``_fresh``), ``_fresh`` (a script picks up
    what another writer saved), ``_view``, the per-call caches ``_grids``,
    and the write bookkeeping ``_depth`` and ``_step`` that ``writes.py``
    keeps. ``frames.py``'s ``_grid`` is the fit grid the classes above it
    read too. A private name not listed here is its own class's.

    Parameters
    ----------
    run
        The run the call belongs to: its job, workspace and spec.
    caller
        Who the context works for, a plugin tool or a script.
    call
        The call's id, which the layers it makes record.
    """

    def __init__(self, run: Run, caller: Caller, call: str) -> None:
        self._run = run
        self._caller = caller
        #: The call's id, as the layers it makes record it.
        self.call = call
        self._grids: dict[tuple[str, str], Any] = {}
        self._depth: int = 0  # open ``writing()`` blocks
        #: The call's undo step so far: the mark it began at, ``job.pushed``
        #: after it and the undo entry on top then; None before any write.
        self._step: tuple[int, int, dict[str, Any]] | None = None

    def _fresh(self) -> None:
        """A script's read first picks up what another writer saved (a plugin
        call is inside a tool call that did)."""
        if not self._caller.in_tool_call:
            self._run.job.sync()

    def _record(self, section: object) -> Section:
        self._fresh()
        state = self._run.state
        record = state.resolve(section)
        if record is None:
            raise as_tool_error(unknown_sections(state, [str(section)]))
        return record

    def _view(self, record: Section) -> SectionView:
        from langslice.core.maps.section_maps import render_sizes

        workspace = self._run.workspace
        transform = record.transform
        affine = None
        if transform is not None:
            params = transform.params
            knobs = None
            if transform.physical is not None:
                knobs = physical_to_json(transform.physical)
            affine = AffineView(
                kind=transform.kind,
                matrix=tuple(float(v) for v in params) if params else None,
                knobs=knobs, mirrored=transform.mirrored)
        size = render_sizes(workspace, record.id).file_size
        pixel_size, source = workspace.calibration(record.id)
        held = None if record.deformation is None else deformation_to_json(record.deformation)
        return SectionView(
            id=record.id, index=int(record.index_corrected), file_size=(size[0], size[1]),
            pixel_size_um=pixel_size, pixel_size_source=source,
            position_mm=record.position_mm, angles=record.angles, flip=bool(record.flip),
            rotation_deg=int(record.rotation_deg), affine=affine,
            deformation=None if held is None else {
                key: copy.deepcopy(held[key]) for key in ("key", "kind", "steps")
                if key in held},
            damaged_regions=tuple(record.damaged_regions), damage_note=record.damage_note,
            layers=tuple(ref.name for ref in record.layers),
            frozen=tuple(name for name in freeze.PROPERTIES
                         if name in self._run.spec.frozen_for(record.id)))

    def sections(self) -> list[SectionView]:
        """Every section, in stack order."""
        self._fresh()
        return [self._view(record) for record in self._run.state.in_order()]

    def section(self, section: object) -> SectionView:
        """One section by filename (the extension may be left off)."""
        return self._view(self._record(section))

    def atlas(self) -> AtlasView:
        """The job's atlas."""
        atlas = self._run.workspace.atlas
        return AtlasView(
            name=str(self._run.state.atlas),
            resolution_um=tuple(float(v) for v in np.asarray(atlas.resolution, dtype=float)),
            orientation=str(getattr(atlas, "orientation", "")),
            species=self._run.workspace.species,
            plane=str(self._run.state.plane),
            shape=tuple(int(v) for v in np.shape(atlas.annotation)))

    def channels(self, section: object) -> tuple[str, ...]:
        """The names of the section's raw channels, in file order (``red``,
        ``green``, ``blue``, or ``gray``, for a colour file; the host's names or
        ``ch1``, ``ch2``... for a multi-page file)."""
        record = self._record(section)
        names, _planes = self._run.workspace.section_channels(record.id)
        return tuple(str(name) for name in names)

    def image_names(self, section: object) -> tuple[str, ...]:
        """Every section image a fit or a picture can read on the section, by
        name: its raw channels (:meth:`channels`), then ``preprocessed`` and
        the images made since (``make_image``, :meth:`add_image`)."""
        from langslice.core.sections.images import image_names

        return tuple(image_names(self._run.workspace, self._record(section)))

    def raw_plane(self, section: object, name: str) -> np.ndarray:
        """Raw channel *name* of the section, ``uint8`` ``(H, W)`` on its working
        grid in the image file's orientation: the frame :meth:`add_image` takes.
        ``ToolError`` ``UNKNOWN_IMAGE`` for a name that is no raw channel."""
        from langslice.core.sections.images import raw_plane

        record = self._record(section)
        try:
            plane = raw_plane(self._run.workspace, record.id, str(name))
            return np.array(plane)
        except ValueError as exc:
            raise ToolError(RefusalCode.UNKNOWN_IMAGE, str(exc)) from exc

    def working_rgb(self, section: object) -> np.ndarray:
        """The section's working copy as read, ``uint8`` ``(H, W, 3)`` in the image
        file's orientation (a colour file's red, green and blue: what a colour
        deconvolution reads): the frame :meth:`add_image` takes."""
        from langslice.core.sections.reading import read_working_image

        record = self._record(section)
        workspace = self._run.workspace
        image, _factor = read_working_image(workspace.image_path(record.id))
        width, height = workspace.working_source(record.id)[0].size
        if image.size != (width, height):
            image = image.resize((width, height))
        return np.asarray(image.convert("RGB")).copy()

    def working_grid(self, section: object) -> tuple[int, int]:
        """``(H, W)`` of the section's working grid: the shape :meth:`add_image`
        takes, the file's orientation."""
        record = self._record(section)
        width, height = self._run.workspace.working_source(record.id)[0].size
        return height, width

    def mask(self, section: object, name: str) -> Any:
        """The section's mask *name* (``make_mask`` made it) as a
        :class:`langslice.core.sections.masks.MaskView`: ``weights`` (float
        ``(H, W)`` on the fit grid, 0 to 1), ``kind`` (``em`` / ``fit``),
        ``params`` and ``left_out``. ``ToolError`` ``UNKNOWN_MASK`` for a name the
        section does not hold, ``STALE_MASK`` when the section changed since
        the mask was made."""
        from langslice.tools import masks as ops_masks

        record = self._record(section)
        try:
            view = ops_masks.read(self._run.job, record, str(name))
        except Refused as refusal:
            raise as_tool_error(refusal) from refusal
        if view.stale:
            raise ToolError(RefusalCode.STALE_MASK, ops_masks.stale_mask(
                self._run.job, record.id, str(name)))
        return view

    def section_inputs(self, section: object, *, deformation: bool = False) -> str:
        """A digest of what a computation on the section reads (its placement,
        appearance and damage marks; with *deformation* its deformation too).
        Take it before a long computation and hand it back as ``expect=`` to a
        write: a section that changed meanwhile comes back ``STALE_INPUT``."""
        return section_inputs(self._run.state, self._record(section), deformation=deformation)

    def section_image(self, section: object, *, level: str | int = "auto",
                      images: Sequence[str] | None = None, raw: bool = False) -> np.ndarray:
        """The pixels of :meth:`section_raster`. Their frame is NOT the image
        file's (turned and flipped as the orientation sets it, and reduced):
        convert coordinates with :meth:`section_raster`'s methods before they
        go in a ``section_px`` layer."""
        return self.section_raster(section, level=level, images=images, raw=raw).array

    def section_raster(self, section: object, *, level: str | int = "auto",
                       images: Sequence[str] | None = None, raw: bool = False,
                       ) -> SectionRaster:
        """The section as its orientation sets it (RGB ``uint8`` ``array``,
        never larger than the source) with the frame it is in
        (:class:`SectionRaster`). *level* is the long edge in pixels, or ``low`` /
        ``medium`` / ``high`` / ``auto`` (the job's picture size). *images*: one
        section image by name (:meth:`image_names`; gray, three equal planes),
        or several raw channels drawn together in their colours; None is the
        ``preprocessed`` image, what the fits read by default.

        *raw* gives the file's own pixels, turned and flipped as the
        orientation sets it and reduced to *level*: with no *images*, the
        file as an image viewer shows it (8-bit RGB; what a delivered PNG
        is), no preprocessing; with exactly one raw channel name, that raw
        channel as gray RGB, as read from the file.
        The pixels come from the working copy (a file longer than 3072
        pixels is already reduced), so *level* at or above the file's long
        edge gives the file itself for a file of that size or less."""
        from langslice.core.maps.section_maps import render_sizes
        from langslice.core.pictures.sizes import picture_edge
        from langslice.core.sections.images import kind_of
        from langslice.core.sections.renders import render_slice

        record = self._record(section)
        workspace = self._run.workspace
        if isinstance(level, int):
            edge = int(level)
        elif level == "auto":
            edge = picture_edge(workspace)
        elif level in PICTURE_EDGES:
            edge = PICTURE_EDGES[level][1]
        else:
            raise ToolError(RefusalCode.BAD_ARGS,
                            f"level is a pixel size or one of "
                            f"{', '.join(PICTURE_EDGES)}; got {level!r}")
        names = [str(name) for name in images or ()]
        known = self.image_names(record.id)
        unknown = [name for name in names if name not in known]
        if unknown:
            raise ToolError(RefusalCode.UNKNOWN_IMAGE,
                            f"{record.id} has no section image "
                            f"{', '.join(map(repr, unknown))}; its images: {', '.join(known)}")
        look: Any = None
        if raw:
            if len(names) > 1 or (names and kind_of(workspace, record, names[0]) != "raw"):
                raise ToolError(RefusalCode.BAD_ARGS,
                                "raw takes no image (the file as it is) or one "
                                "raw channel name")
            look = {"image": names[0]} if names else {"file": True}
        elif len(names) == 1:
            look = {"image": names[0]}
        elif names:
            if any(kind_of(workspace, record, name) != "raw" for name in names):
                raise ToolError(RefusalCode.BAD_ARGS,
                                "several images are drawn together only when all "
                                "are raw channels")
            look = {"overlay": names}
        try:
            image = render_slice(workspace, record, long_edge=edge, look=look)
        except ValueError as exc:
            raise ToolError(RefusalCode.BAD_ARGS, str(exc)) from exc
        return SectionRaster(
            id=record.id, array=np.asarray(image.convert("RGB")),
            file_size=tuple(render_sizes(workspace, record.id).file_size),  # type: ignore[arg-type]
            rotation_deg=int(record.rotation_deg), flip=bool(record.flip))

    def layers(self, section: object) -> list[LayerRef]:
        """The layers the section shows now."""
        record = self._record(section)
        return [LayerRef(id=ref.id, section=record.id, name=ref.name, kind=ref.kind,
                         frame=ref.frame, file=ref.file) for ref in record.layers]


def as_tool_error(refusal: Refused) -> ToolError:
    """Turn a refusal into the :class:`ToolError` a plugin catches.

    Parameters
    ----------
    refusal
        The refusal an operation raised.

    Returns
    -------
    ToolError
        The same code, the refusal's message first and its other facts
        after it.
    """
    return ToolError(refusal.code, str(refusal), **{
        key: value for key, value in refusal.facts.items() if key != "message"})
