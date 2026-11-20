"""The reads of a plugin context (``ctx.reads``): sections, their images and layers.

:class:`Reads` reads what the job holds of a section
(:class:`SectionView`, the images by name, a :class:`SectionRaster`, the
layers). :class:`Records` is how every part of a context finds a section by
filename.
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
from langslice.core.refusal import RefusalCode
from langslice.core.space import Plane
from langslice.core.stack.section import Section
from langslice.job.settings import frozen as freeze
from langslice.plugins.caller import Caller
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
    ``pivot``), and whether it mirrors the section: ``kind``, ``matrix``
    (None when it holds no numbers), ``knobs`` (None when it was not made
    from them) and ``mirrored``."""

    kind: str | None
    matrix: tuple[float, ...] | None
    knobs: dict[str, Any] | None
    mirrored: bool | None


@dataclass(frozen=True)
class SectionView:
    """One section as the job holds it now (``ctx.reads.section``).

    Attributes
    ----------
    id
        The section's filename.
    index
        Its place in the stack's order, from 0.
    file_size
        The image file's ``(width, height)`` in pixels.
    pixel_size_um
        Micrometres per file pixel; None when the job knows none.
    pixel_size_source
        Where that size came from: ``host`` (given when the job was made),
        ``file`` (the image file's own tags) or empty (none).
    position_mm
        Atlas millimetres along the slicing axis from the anterior edge.
    angles
        The section's cutting angles ``(pitch_deg, yaw_deg)``.
    flip, rotation_deg
        Its orientation: mirrored left-right, then turned 0, 90, 180 or 270.
    affine
        Its in-plane transform (:class:`AffineView`); None before it has one.
    deformation
        The applied deformation's ``key``, ``kind`` and ``steps``; None
        without one.
    damaged_regions, damage_note
        The atlas regions ``mark_damage`` marked as lost, and its note.
    layers
        The names of the layers it shows.
    frozen
        The properties the job does not let anyone change on it.
    """

    id: str
    index: int
    file_size: tuple[int, int]
    pixel_size_um: float | None
    pixel_size_source: str
    position_mm: float
    angles: tuple[float, float]
    flip: bool
    rotation_deg: int
    affine: AffineView | None
    deformation: dict[str, Any] | None
    damaged_regions: tuple[str, ...]
    damage_note: str
    layers: tuple[str, ...]
    frozen: tuple[str, ...]


@dataclass(frozen=True)
class AtlasView:
    """The job's atlas (``ctx.reads.atlas()``).

    Attributes
    ----------
    name
        The BrainGlobe atlas name, such as ``allen_mouse_25um``.
    resolution_um
        Micrometres per voxel along each of the atlas's axes.
    orientation
        BrainGlobe's three-letter axis order (``asr``: anterior to posterior,
        superior to inferior, right to left).
    species
        The atlas's species.
    plane
        The cutting plane of the job's sections.
    shape
        The volume's shape in voxels, in the atlas's axis order.
    """

    name: str
    resolution_um: tuple[float, ...]
    orientation: str
    species: str
    plane: Plane
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
    the layer is added. ``rotation_deg`` and ``flip`` are the orientation
    ``array`` was turned by. ``file_size`` is the file's ``(width, height)``,
    ``pixel_size_um`` the micrometres of one file pixel (None when the job
    knows none) and :attr:`um_per_px` those of one pixel of ``array``.
    """

    id: str
    array: np.ndarray
    file_size: tuple[int, int]
    rotation_deg: int
    flip: bool
    pixel_size_um: float | None = None

    @property
    def um_per_px(self) -> float | None:
        """Micrometres of one pixel of ``array``; None without a pixel size."""
        if self.pixel_size_um is None:
            return None
        return self.pixel_size_um * max(self.file_size) / max(self.array.shape[:2])

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
        return transform(np.linalg.inv(self._matrix()), fractions) * np.array([width, height]) - 0.5

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
    """A layer registered on a section: its ``id``, the ``section``'s
    filename, its ``name`` (what ``look``'s ``layers`` takes), its ``kind``
    (``points``, ``lines``, ``labels``, ``scalar`` or ``picture``), the
    ``frame`` its coordinates are in (``section_px``, ``atlas_um`` or
    ``oriented``) and the ``file`` that holds it, relative to the job
    folder."""

    id: str
    section: str
    name: str
    kind: str
    frame: str
    file: str


class Records:
    """Finds the job's sections by filename for one caller.

    Every part of a context that takes a section names it through one
    :class:`Records`; a script's reads first pick up what another writer
    saved, a plugin call is inside a tool call that did.

    Parameters
    ----------
    run
        The run the call belongs to: its job, workspace and spec.
    caller
        Who the context works for, a plugin tool or a script.
    """

    def __init__(self, run: Run, caller: Caller) -> None:
        self._run = run
        self._caller = caller

    def fresh(self) -> None:
        """A script's read first picks up what another writer saved."""
        if not self._caller.in_tool_call:
            self._run.job.sync()

    def get(self, section: object) -> Section:
        """The section named *section* (a filename, the extension may be left
        off), after :meth:`fresh`; ``unknown_sections`` when it names none."""
        self.fresh()
        state = self._run.state
        record = state.resolve(section)
        if record is None:
            raise unknown_sections(state, [str(section)])
        return record


class Reads:
    """What the job holds of its sections, the atlas and the section images:
    ``ctx.reads``.

    Parameters
    ----------
    run
        The run the call belongs to: its job, workspace and spec.
    records
        How a section is found by filename.
    """

    def __init__(self, run: Run, records: Records) -> None:
        self._run = run
        self._records = records

    def _view(self, record: Section) -> SectionView:
        from langslice.core.maps.section_maps import render_sizes

        workspace = self._run.workspace
        transform = record.transform
        affine = None
        if transform is not None:
            params = transform.params
            knobs = None
            if transform.physical is not None:
                knobs = transform.physical.to_json()
            affine = AffineView(
                kind=transform.kind,
                matrix=tuple(float(v) for v in params) if params else None,
                knobs=knobs,
                mirrored=transform.mirrored,
            )
        size = render_sizes(workspace, record.id).file_size
        pixel_size, source = workspace.calibration(record.id)
        held = None if record.deformation is None else record.deformation.to_json()
        return SectionView(
            id=record.id,
            index=int(record.index_corrected),
            file_size=(size[0], size[1]),
            pixel_size_um=pixel_size,
            pixel_size_source=source,
            position_mm=record.position_mm,
            angles=record.angles,
            flip=bool(record.flip),
            rotation_deg=int(record.rotation_deg),
            affine=affine,
            deformation=None
            if held is None
            else {key: copy.deepcopy(held[key]) for key in ("key", "kind", "steps") if key in held},
            damaged_regions=tuple(record.damaged_regions),
            damage_note=record.damage_note,
            layers=tuple(ref.name for ref in record.layers),
            frozen=tuple(
                name for name in freeze.PROPERTIES if name in self._run.spec.frozen_for(record.id)
            ),
        )

    def sections(self) -> list[SectionView]:
        """Every section, in stack order."""
        self._records.fresh()
        return [self._view(record) for record in self._run.state.in_order()]

    def section(self, section: object) -> SectionView:
        """One section by filename (the extension may be left off)."""
        return self._view(self._records.get(section))

    def atlas(self) -> AtlasView:
        """The job's atlas."""
        atlas = self._run.workspace.atlas
        return AtlasView(
            name=str(self._run.state.atlas),
            resolution_um=tuple(float(v) for v in np.asarray(atlas.resolution, dtype=float)),
            orientation=str(getattr(atlas, "orientation", "")),
            species=self._run.workspace.species,
            plane=self._run.state.plane,
            shape=tuple(int(v) for v in np.shape(atlas.annotation)),
        )

    def channels(self, section: object) -> tuple[str, ...]:
        """The names of the section's raw channels, in file order (``red``,
        ``green``, ``blue``, or ``gray``, for a colour file; the host's names or
        ``ch1``, ``ch2``... for a multi-page file)."""
        record = self._records.get(section)
        names, _planes = self._run.workspace.section_channels(record.id)
        return tuple(str(name) for name in names)

    def image_names(self, section: object) -> tuple[str, ...]:
        """Every section image a fit or a picture can read on the section, by
        name: its raw channels (:meth:`channels`), then ``preprocessed`` and
        the images made since (``make_image``, ``ctx.writes.add_image``)."""
        from langslice.core.sections.images import image_names

        return tuple(image_names(self._run.workspace, self._records.get(section)))

    def raw_plane(self, section: object, name: str) -> np.ndarray:
        """Raw channel *name* of the section, ``uint8`` ``(H, W)`` on its working
        grid in the image file's orientation: the frame ``ctx.writes.add_image`` takes.
        ``ToolError`` ``NOT_FOUND`` for a name that is no raw channel."""
        from langslice.core.sections.images import raw_plane

        record = self._records.get(section)
        try:
            plane = raw_plane(self._run.workspace, record.id, str(name))
            return np.array(plane)
        except ValueError as exc:
            raise ToolError(RefusalCode.NOT_FOUND, str(exc)) from exc

    def working_rgb(self, section: object) -> np.ndarray:
        """The section's working copy as read, ``uint8`` ``(H, W, 3)`` in the image
        file's orientation (a colour file's red, green and blue: what a colour
        deconvolution reads): the frame ``ctx.writes.add_image`` takes."""
        from langslice.core.sections.reading import read_working_image

        record = self._records.get(section)
        workspace = self._run.workspace
        image, _factor = read_working_image(workspace.image_path(record.id))
        width, height = workspace.working_source(record.id)[0].size
        if image.size != (width, height):
            image = image.resize((width, height))
        return np.asarray(image.convert("RGB")).copy()

    def working_grid(self, section: object) -> tuple[int, int]:
        """``(H, W)`` of the section's working grid: the shape ``ctx.writes.add_image``
        takes, the file's orientation."""
        record = self._records.get(section)
        width, height = self._run.workspace.working_source(record.id)[0].size
        return height, width

    def mask(self, section: object, name: str) -> Any:
        """The section's mask *name* (``make_mask`` made it) as a
        :class:`langslice.core.sections.masks.MaskView`: ``weights`` (float
        ``(H, W)`` on the fit grid, 0 to 1), ``kind`` (``em`` / ``fit``),
        ``params`` and ``left_out``. ``ToolError`` ``NOT_FOUND`` for a name the
        section does not hold, ``STALE_INPUT`` when the section changed since
        the mask was made."""
        from langslice.tools.changing import masks as ops_masks

        record = self._records.get(section)
        view = ops_masks.read(self._run.job, record, str(name))
        if view.stale:
            raise ToolError(
                RefusalCode.STALE_INPUT, ops_masks.stale_mask(self._run.job, record.id, str(name))
            )
        return view

    def section_inputs(self, section: object, *, deformation: bool = False) -> str:
        """A digest of what a computation on the section reads (its placement,
        appearance and damage marks; with *deformation* its deformation too).
        Take it before a long computation and hand it back as ``expect=`` to a
        write: a section that changed meanwhile comes back ``STALE_INPUT``."""
        return section_inputs(self._run.state, self._records.get(section), deformation=deformation)

    def section_image(
        self,
        section: object,
        *,
        level: str | int = "auto",
        images: Sequence[str] | None = None,
        raw: bool = False,
    ) -> np.ndarray:
        """The pixels of ``ctx.reads.section_raster``. Their frame is NOT the image
        file's (turned and flipped as the orientation sets it, and reduced):
        convert coordinates with ``ctx.reads.section_raster``'s methods before they
        go in a ``section_px`` layer."""
        return self.section_raster(section, level=level, images=images, raw=raw).array

    def section_raster(
        self,
        section: object,
        *,
        level: str | int = "auto",
        images: Sequence[str] | None = None,
        raw: bool = False,
    ) -> SectionRaster:
        """The section as its orientation sets it (RGB ``uint8`` ``array``,
        never larger than the source) with the frame it is in
        (:class:`SectionRaster`). *level* is the long edge in pixels, or ``low`` /
        ``medium`` / ``high`` / ``auto`` (the job's picture size). *images*: one
        section image by name (``ctx.reads.image_names``; gray, three equal planes),
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

        record = self._records.get(section)
        workspace = self._run.workspace
        if isinstance(level, int):
            edge = int(level)
        elif level == "auto":
            edge = picture_edge(workspace)
        elif level in PICTURE_EDGES:
            edge = PICTURE_EDGES[level][1]
        else:
            raise ToolError(
                RefusalCode.BAD_ARGUMENT,
                f"level is a pixel size or one of {', '.join(PICTURE_EDGES)}; got {level!r}",
            )
        names = [str(name) for name in images or ()]
        known = self.image_names(record.id)
        unknown = [name for name in names if name not in known]
        if unknown:
            raise ToolError(
                RefusalCode.NOT_FOUND,
                f"{record.id} has no section image "
                f"{', '.join(map(repr, unknown))}; its images: {', '.join(known)}",
            )
        look: Any = None
        if raw:
            if len(names) > 1 or (names and kind_of(workspace, record, names[0]) != "raw"):
                raise ToolError(
                    RefusalCode.BAD_ARGUMENT,
                    "raw takes no image (the file as it is) or one raw channel name",
                )
            look = {"image": names[0]} if names else {"file": True}
        elif len(names) == 1:
            look = {"image": names[0]}
        elif names:
            if any(kind_of(workspace, record, name) != "raw" for name in names):
                raise ToolError(
                    RefusalCode.BAD_ARGUMENT,
                    "several images are drawn together only when all are raw channels",
                )
            look = {"overlay": names}
        try:
            image = render_slice(workspace, record, long_edge=edge, look=look)
        except ValueError as exc:
            raise ToolError(RefusalCode.BAD_ARGUMENT, str(exc)) from exc
        pixel_size, _source = workspace.calibration(record.id)
        return SectionRaster(
            id=record.id,
            array=np.asarray(image.convert("RGB")),
            file_size=render_sizes(workspace, record.id).file_size,
            rotation_deg=int(record.rotation_deg),
            flip=bool(record.flip),
            pixel_size_um=None if pixel_size is None else float(pixel_size),
        )

    def layers(self, section: object) -> list[LayerRef]:
        """The layers the section shows now."""
        record = self._records.get(section)
        return [
            LayerRef(
                id=ref.id,
                section=record.id,
                name=ref.name,
                kind=ref.kind,
                frame=ref.frame,
                file=ref.file,
            )
            for ref in record.layers
        ]
