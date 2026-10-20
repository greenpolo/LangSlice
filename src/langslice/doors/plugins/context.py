"""The context a plugin tool's handler gets, and a script's ``job.context``:
what it may read and write.

One :class:`ToolContext` per plugin call (a plugin must not keep it); a
script's (:class:`langslice.doors.library.ScriptContext`) lives as long as
its job handle. Who is calling is a :class:`Caller`: a plugin tool
(:meth:`Caller.of_tool`) or a script (:meth:`Caller.script`). Reads give
plain data in LangSlice's documented units and frames; writes go through the
same operations the native tools use (``ops.positions``, ``ops.transforms``,
``ops.deformable``, ``ops.damage``), so every write is one undo step that checkpoints the state
and rewrites ``registration.json``, frozen properties are refused (``FROZEN``),
and a section that changed since a digest was taken is refused (``STALE_INPUT``).
A tool declared ``read`` cannot write; a ``write`` tool writes the section
properties named in its :attr:`~langslice.core.plugins.ToolSpec.writes` only;
a script writes any property the job does not freeze. A plugin's files go
under :meth:`ToolContext.work_dir` and nowhere else.
"""

from __future__ import annotations

import contextlib
import copy
import types
from collections.abc import Callable, Iterator, Mapping, Sequence
from dataclasses import dataclass, field, replace
from pathlib import Path
from typing import TYPE_CHECKING, Any, cast

import numpy as np
from PIL import Image

from langslice.core import frozen as freeze
from langslice.core.plugins import SCRIPT_CALLER, Layer, ToolError, ToolResult, ToolSpec
from langslice.core.sizes import PICTURE_EDGES
from langslice.core.state import SliceState, StackState
from langslice.ops.inputs import STALE_INPUT, section_inputs, stale_row
from langslice.ops.refusal import Refused, frozen_row, unknown_sections

if TYPE_CHECKING:
    from langslice.doors.tools.door import Door


# --- what a handler reads ---------------------------------------------------------------


@dataclass(frozen=True)
class AffineView:
    """A section's in-plane transform: ``kind`` (``interactive``, ``elastix``,
    ``plugin:<tool>``...), the six normalized numbers ``[a, b, tx, c, d, ty]``
    (``docs/file_formats.md``), the knobs they are in (``rotation_deg``,
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
    #: The properties frozen on it (``langslice.core.frozen.PROPERTIES``).
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
class FitFrame:
    """A section and the atlas plane it is placed on, on the grid LangSlice's
    fits work on.

    ``image`` is the section as its orientation sets it, at its current plane,
    with no in-plane transform (RGB, ``uint8``).
    ``atlas`` holds the plane at the section's position and cutting angles on
    the native atlas grid: ``annotation`` (region ids) and the atlas's images
    asked for (``template`` and any additional reference the BrainGlobe atlas
    ships, by its name). ``atlas_to_section`` (3x3 on ``[x, y, 1]``)
    maps native atlas pixel centres to ``image`` pixel centres. ``plane`` says
    where the native plane lies in atlas micrometres: ``center_um``, ``dx``,
    ``dy`` (the column and row directions), ``px_um``, ``shape`` and
    ``resolution_um``. ``placed_atlas_to_section`` is the same map under the
    section's stored in-plane transform (its current linear placement, where an
    engine that refines it starts), None while it has no transform.
    ``excluded`` is the native plane's pixels of the section's marked damage
    regions (sides included), the regions every native fit leaves out; None
    while the section has none marked. ``damage_zone`` is ``excluded`` widened
    by the margin the native fits also leave out of their comparison
    (:data:`langslice.core.deformable.masks.EXCLUSION_MARGIN_MM`, scaled to the
    atlas's brain size); None with ``excluded``. ``stain`` is the section's
    preprocessed channel on the same grid as ``image`` (RGB ``uint8``), the
    picture the native fits read; None unless asked for.
    """

    id: str
    image: np.ndarray
    atlas: dict[str, np.ndarray]
    atlas_to_section: np.ndarray
    plane: dict[str, Any]
    position_mm: float
    angles: tuple[float, float]
    placed_atlas_to_section: np.ndarray | None = None
    applied_map: np.ndarray | None = None
    excluded: np.ndarray | None = None
    damage_zone: np.ndarray | None = None
    stain: np.ndarray | None = None


@dataclass(frozen=True)
class AtlasPlane:
    """An atlas plane at a placement of your choosing
    (:meth:`ToolContext.atlas_plane`), on its native grid: ``atlas`` the
    planes sampled (``annotation`` and the atlas's images asked for), ``plane``
    where it lies in atlas micrometres (as :attr:`FitFrame.plane`:
    ``native_to_um``, ``px_um``, ``shape``, ...), at ``position_mm`` and
    ``angles`` ``(pitch_deg, yaw_deg)``."""

    position_mm: float
    angles: tuple[float, float]
    atlas: dict[str, np.ndarray]
    plane: dict[str, Any]


@dataclass(frozen=True)
class TransformProposal:
    """A transform a method found, in the forms LangSlice stores it:
    ``params`` are the six normalized numbers, ``knobs`` the same transform as
    ``interactive_transform`` takes it about ``pivot``, ``record`` the stored
    record. :meth:`row` is a row for :meth:`ToolContext.set_transforms`."""

    id: str
    params: tuple[float, ...]
    knobs: dict[str, float]
    pivot: tuple[float, float]
    mirrored: bool
    calibration: dict[str, Any]
    record: dict[str, Any]

    def row(self) -> dict[str, Any]:
        return {"id": self.id, "transform": self.record}


@dataclass(frozen=True)
class DeformationProposal:
    """A deformation a method found, as LangSlice stores it, not written yet
    (:meth:`ToolContext.deformation_from_map`): ``summary`` holds its numbers
    (``displacement_mm`` max and median over the placed atlas footprint, in
    millimetres, ``fold_fraction``, ``flags``) and ``inverse_consistency_mm``
    how well the inverse LangSlice computed undoes it. :meth:`row` is a row
    for :meth:`ToolContext.set_deformations`;
    ``overlay_picture(id, proposal={"deformation": proposal})`` draws it."""

    id: str
    key: str
    summary: dict[str, Any]
    inverse_consistency_mm: dict[str, float]
    _warp: Any = field(repr=False, compare=False)

    def row(self) -> dict[str, Any]:
        return {"id": self.id, "deformation": self}


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
        from langslice.core.landmarks import affine_from_points
        from langslice.core.native import unoriented_fractions

        corners = np.array([[0.0, 0.0], [1.0, 0.0], [0.0, 1.0]])
        x, y = unoriented_fractions(corners[:, 0], corners[:, 1], self.rotation_deg, self.flip)
        return affine_from_points(np.stack([x, y], axis=1))

    def to_section_px(self, xy: Any) -> np.ndarray:
        """``[x, y]`` pixels of ``array`` as ``section_px`` (file pixels)."""
        from langslice.core.landmarks import transform

        height, width = self.array.shape[:2]
        fractions = (np.asarray(xy, dtype=np.float64) + 0.5) / np.array([width, height])
        theirs = transform(self._matrix(), fractions)
        return theirs * np.array(self.file_size) - 0.5

    def from_section_px(self, xy: Any) -> np.ndarray:
        """``section_px`` (file pixels) as ``[x, y]`` pixels of ``array``."""
        from langslice.core.landmarks import transform

        height, width = self.array.shape[:2]
        fractions = (np.asarray(xy, dtype=np.float64) + 0.5) / np.array(self.file_size)
        return transform(np.linalg.inv(self._matrix()), fractions) * np.array(
            [width, height]) - 0.5

    def mask_to_section_px(self, raster: Any) -> np.ndarray:
        """A raster in ``array``'s orientation as one in the file's."""
        return unorient_raster(np.asarray(raster), self.rotation_deg, self.flip)


def unorient_raster(raster: np.ndarray, rotation_deg: int, flip: bool) -> np.ndarray:
    """*raster*, drawn as a section's orientation sets it, in the file's orientation."""
    from langslice.core.native import unorient

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


@dataclass
class WriteResult:
    """What a write did. ``ok`` is False when the call was refused whole
    (``error`` is the code, ``message`` the words; nothing was written);
    otherwise ``rows`` has one row per section asked, ``status`` ``ok`` or
    ``error`` with the code (``FROZEN``, ``STALE_INPUT``, ``UNKNOWN_SLICE_IDS``,
    ...), and ``written`` the ids written (one undo step)."""

    ok: bool = True
    error: str | None = None
    message: str = ""
    written: list[str] = field(default_factory=list)
    rows: list[dict[str, Any]] = field(default_factory=list)
    details: dict[str, Any] = field(default_factory=dict)

    @classmethod
    def refused(cls, refusal: Refused) -> WriteResult:
        return cls(ok=False, error=refusal.code, message=str(refusal),
                   details={key: value for key, value in refusal.details.items()
                            if key != "message"})

    def reply(self) -> dict[str, Any]:
        """The result as reply fields: ``written`` and ``rows``, or the refusal."""
        if not self.ok:
            return {"status": "error", "error": self.error, "message": self.message,
                    **self.details}
        return {"written": list(self.written), "rows": [dict(row) for row in self.rows]}


def _plain(value: Any) -> Any:
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, Mapping):
        return {str(key): _plain(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_plain(item) for item in value]
    return value


# --- who is calling ---------------------------------------------------------------------


@dataclass(frozen=True)
class Caller:
    """Who a :class:`ToolContext` works for: a plugin tool, or a script.

    ``name`` names it where a layer or background work records its maker
    (the tool's name, ``script``); ``label`` is what its writes are recorded
    as (a transform's ``kind``, a deformation's ``method`` and engine:
    ``plugin:<tool>`` or ``script``); ``kind`` and ``writes`` what it may
    write; ``version`` the plugin's own. ``work_dir``: it has a folder of
    its own (``plugins/<tool>/``); ``in_tool_call``: the tool door around it
    brings the job up to date before the call and drops the deformations
    its writes made stale after it; a script is in no tool call, so its
    reads sync first and its placement writes drop them themselves (in the
    same undo step); ``merge_calls``: its outermost writes join one undo
    step (one tool call) unless another write came between; ``background``:
    it may start background work."""

    name: str
    label: str
    kind: str
    writes: tuple[str, ...]
    version: str | None = None
    work_dir: bool = True
    in_tool_call: bool = True
    merge_calls: bool = True
    background: bool = True

    @classmethod
    def of_tool(cls, tool: ToolSpec) -> Caller:
        """A plugin tool's call."""
        return cls(name=tool.name, label=f"plugin:{tool.name}", kind=tool.kind,
                   writes=tuple(tool.writes), version=tool.version)

    @classmethod
    def script(cls) -> Caller:
        """A script's ``job.context``: writes every property the job leaves free,
        recorded as ``script``; no folder of its own (a script keeps its
        files in ``scripts/`` and ``scratch/``) and no background work (the
        script is the computation)."""
        return cls(name=SCRIPT_CALLER, label=SCRIPT_CALLER, kind="write",
                   writes=tuple(freeze.PROPERTIES), work_dir=False, in_tool_call=False,
                   merge_calls=False, background=False)

    @property
    def is_script(self) -> bool:
        return self.name == SCRIPT_CALLER

    def provenance(self, call: str, note: str, meta: Any) -> dict[str, Any]:
        """What a deformation record keeps of the caller that wrote it:
        ``plugin`` (tool, call, version, note, meta) or ``script`` (call,
        note, meta, and ``argv0``, the running script's path as started)."""
        if self.is_script:
            import sys

            return {"script": {"call": call, "note": note, "meta": meta,
                               "argv0": str(sys.argv[0]) if sys.argv else ""}}
        return {"plugin": {"tool": self.name, "call": call, "version": self.version,
                           "note": note, "meta": meta}}


def _not_for_scripts(what: str) -> ToolError:
    return ToolError("NOT_FOR_SCRIPTS", what)


# --- the context -------------------------------------------------------------------------


class ToolContext:
    """What one plugin tool call, or a script, can read and write (see the
    module text)."""

    def __init__(self, door: Door, caller: Caller, call: str) -> None:
        self._door = door
        self._caller = caller
        #: The call's id, as the layers it makes record it.
        self.call = call
        self._grids: dict[tuple[str, str], Any] = {}
        self._depth = 0  # open ``writing()`` blocks
        #: The call's undo step so far: the mark it began at, ``job.pushed``
        #: after it and the undo entry on top then; None before any write.
        self._step: tuple[int, int, dict[str, Any]] | None = None

    # --- the job ----------------------------------------------------------------------

    @property
    def tool(self) -> str:
        """The tool's name (``script`` for a script)."""
        return self._caller.name

    @property
    def spec(self) -> Mapping[str, Any]:
        """The job's spec (``JobSpec.to_dict``), read-only."""
        return types.MappingProxyType(copy.deepcopy(self._door.spec.to_dict()))

    @property
    def job_dir(self) -> Path:
        """The job folder."""
        return Path(self._door.job.folder)

    def work_dir(self) -> Path:
        """``<job>/plugins/<tool>/``: the one place this tool writes files
        (scratch, logs, an engine's inputs and outputs). Made on first use.
        A script has none (``NOT_FOR_SCRIPTS``): it keeps its files in the
        job folder's ``scripts/`` and ``scratch/``."""
        if not self._caller.work_dir:
            raise _not_for_scripts("a script keeps its files in the job folder's scripts/ "
                                   "and scratch/")
        path = self._door.job.layout.plugin_dir(self._caller.name)
        path.mkdir(parents=True, exist_ok=True)
        return path

    def task_on(self, task: str) -> bool:
        """Whether the job's *task* (``position``, ``transform``, ``nonlinear``) is on."""
        return bool(self._door.spec.has(task))

    # --- reading ------------------------------------------------------------------------

    def _fresh(self) -> None:
        """A script's read first picks up what another writer saved (a plugin
        call is inside a tool call that did)."""
        if not self._caller.in_tool_call:
            self._door.job.sync()

    def _record(self, section: object) -> SliceState:
        self._fresh()
        state = self._door.state
        record = state.resolve(section)
        if record is None:
            raise ToolError("UNKNOWN_SLICE_IDS", **unknown_sections(state, [str(section)]).details)
        return record

    def _view(self, record: SliceState) -> SectionView:
        from langslice.core.maps import render_sizes

        workspace = self._door.ctx
        transform = record.transform or {}
        affine = None
        if transform:
            params = transform.get("params")
            affine = AffineView(
                kind=transform.get("kind"),
                matrix=tuple(float(v) for v in params) if params else None,
                knobs=copy.deepcopy(transform.get("physical")) if transform.get("physical")
                else None,
                mirrored=transform.get("mirrored"))
        size = render_sizes(workspace, record.id).file_size
        pixel_size, source = workspace.calibration(record.id)
        held = record.deformation
        return SectionView(
            id=record.id, index=int(record.index_corrected), file_size=(size[0], size[1]),
            pixel_size_um=pixel_size, pixel_size_source=source,
            position_mm=record.position_mm, angles=record.angles, flip=bool(record.flip),
            rotation_deg=int(record.rotation_deg), affine=affine,
            deformation=None if held is None else {
                key: copy.deepcopy(held[key]) for key in ("key", "kind", "steps")
                if key in held},
            damaged_regions=tuple(record.damaged_regions), damage_note=record.damage_note,
            layers=tuple(str(ref["name"]) for ref in record.layers),
            frozen=tuple(name for name in freeze.PROPERTIES
                         if name in self._door.spec.frozen_for(record.id)))

    def sections(self) -> list[SectionView]:
        """Every section, in stack order."""
        self._fresh()
        return [self._view(record) for record in self._door.state.in_order()]

    def section(self, section: object) -> SectionView:
        """One section by filename (the extension may be left off)."""
        return self._view(self._record(section))

    def atlas(self) -> AtlasView:
        """The job's atlas."""
        atlas = self._door.ctx.atlas
        return AtlasView(
            name=str(self._door.state.atlas),
            resolution_um=tuple(float(v) for v in np.asarray(atlas.resolution, dtype=float)),
            orientation=str(getattr(atlas, "orientation", "")),
            species=self._door.ctx.species, plane=str(self._door.state.plane),
            shape=tuple(int(v) for v in np.shape(atlas.annotation)))

    def channels(self, section: object) -> tuple[str, ...]:
        """The names of the section's raw channels, in file order (``red``,
        ``green``, ``blue``, or ``gray``, for a colour file; the host's names or
        ``ch1``, ``ch2``... for a multi-page file): what ``section_raster(...,
        channels=[name])`` takes."""
        names, _planes = self._door.ctx.section_channels(self._record(section).id)
        return tuple(str(name) for name in names)

    def section_inputs(self, section: object, *, deformation: bool = False) -> str:
        """A digest of what a computation on the section reads (its placement,
        appearance and damage marks; with *deformation* its deformation too).
        Take it before a long computation and hand it back as ``expect=`` to a
        write: a section that changed meanwhile comes back ``STALE_INPUT``."""
        return section_inputs(self._door.state, self._record(section), deformation=deformation)

    def section_image(self, section: object, *, level: str | int = "auto",
                      channels: Sequence[str] | None = None, raw: bool = False) -> np.ndarray:
        """The pixels of :meth:`section_raster`. Their frame is NOT the image
        file's (turned and flipped as the orientation sets it, and reduced):
        convert coordinates with :meth:`section_raster`'s methods before they
        go in a ``section_px`` layer."""
        return self.section_raster(section, level=level, channels=channels, raw=raw).array

    def section_raster(self, section: object, *, level: str | int = "auto",
                       channels: Sequence[str] | None = None, raw: bool = False,
                       ) -> SectionRaster:
        """The section as its orientation sets it (RGB ``uint8`` ``array``,
        never larger than the source) with the frame it is in
        (:class:`SectionRaster`). *level* is the long edge in pixels, or ``low`` /
        ``medium`` / ``high`` / ``auto`` (the job's picture size). *channels*:
        raw channel names (drawn in their colours with the channel display
        properties the agent set), or ``["preprocessed"]`` for the channel the
        fits read; None is the default appearance (the preprocessed blend
        when the job preprocesses).

        *raw* gives the file's own pixels, turned and flipped as the
        orientation sets it and reduced to *level*: with no *channels*, the
        file as an image viewer shows it (8-bit RGB; what a delivered PNG
        is), no preprocessing and no channel properties; with exactly one
        channel name, that raw channel as gray RGB (three equal planes), as
        read from the file, whatever display properties the agent set.
        The pixels come from the working copy (a file longer than 3072
        pixels is already reduced), so *level* at or above the file's long
        edge gives the file itself for a file of that size or less."""
        from langslice.core.appearance import preprocessed_settings
        from langslice.core.channels import with_properties
        from langslice.core.maps import render_sizes
        from langslice.core.sections import render_slice
        from langslice.core.sizes import picture_edge

        record = self._record(section)
        if isinstance(level, int):
            edge = int(level)
        elif level == "auto":
            edge = picture_edge(self._door.ctx)
        elif level in PICTURE_EDGES:
            edge = PICTURE_EDGES[level][1]
        else:
            raise ToolError("BAD_ARGS", f"level is a pixel size or one of "
                            f"{', '.join(PICTURE_EDGES)}; got {level!r}")
        look: Any = None
        if raw:
            names = [str(name) for name in channels or ()]
            if len(names) > 1 or names == ["preprocessed"]:
                raise ToolError("BAD_ARGS", "raw takes no channel (the file as it is) or one "
                                "raw channel name")
            look = {"channel": names[0]} if names else {"file": True}
        elif channels:
            names = [str(name) for name in channels]
            look = (preprocessed_settings(self._door.state, record.id)
                    if names == ["preprocessed"]
                    else with_properties(self._door.state, {"overlay": names}))
        try:
            image = render_slice(self._door.ctx, record, long_edge=edge, look=look)
        except ValueError as exc:  # a channel the section does not have
            raise ToolError("BAD_ARGS", str(exc)) from exc
        return SectionRaster(
            id=record.id, array=np.asarray(image.convert("RGB")),
            file_size=tuple(render_sizes(self._door.ctx, record.id).file_size),  # type: ignore[arg-type]
            rotation_deg=int(record.rotation_deg), flip=bool(record.flip))

    def _grid(self, record: SliceState) -> Any:
        from langslice.core import deformation
        from langslice.core.affine import IDENTITY_PARAMS

        if record.position_mm is None:
            raise ToolError("NO_POSITION", f"{record.id} has no position yet")
        # Keyed by what the grid is drawn from: a script's context outlives a
        # placement change.
        key = (record.id, section_inputs(self._door.state, record))
        if key not in self._grids:
            try:
                self._grids[key] = deformation.fit_grid(
                    self._door.state, self._door.ctx, record,
                    transform={"params": list(IDENTITY_PARAMS)})
            except ValueError as exc:
                raise ToolError("INVALID_LINEAR_PLACEMENT", f"{record.id}: {exc}") from exc
        return self._grids[key]

    def fit_frame(self, section: object, *, atlas: Sequence[str] = (
            "template", "annotation"), stain: bool = False) -> FitFrame:
        """The section and its atlas plane on the fit grid (:class:`FitFrame`).
        Needs a position. *atlas* picks the atlas planes to sample:
        ``annotation`` and any of the atlas's images
        (:func:`langslice.core.atlas.images.atlas_images`); a name the atlas
        does not have is skipped. *stain* adds the section's preprocessed
        channel on the grid."""
        from langslice.core import deformation
        from langslice.core.atlas.images import atlas_images
        from langslice.core.atlas.size import brain_scale
        from langslice.core.damage import exclusions
        from langslice.core.deformable.masks import EXCLUSION_MARGIN_MM, dilate
        from langslice.core.deformable.plane_images import (
            native_intensity,
            native_labels,
            placement_left,
            regions_mask,
        )

        record = self._record(section)
        grid = self._grid(record)
        image = grid.image.convert("RGB")
        placement = grid.placement
        workspace = self._door.ctx
        planes: dict[str, np.ndarray] = {}
        if "annotation" in atlas:
            planes["annotation"] = np.asarray(native_labels(workspace.atlas, placement))
        for kind in atlas_images(workspace.atlas):
            if kind in atlas:
                planes[kind] = np.asarray(native_intensity(workspace.atlas, placement, kind),
                                          dtype=np.float32)
        placed = None
        if record.transform:
            try:
                placed = np.asarray(deformation.fit_grid(
                    self._door.state, workspace, record).placement.atlas_to_section,
                    dtype=np.float64)
            except ValueError:  # a stored transform no fit can use (stale, turned over)
                placed = None
        applied = None
        if record.deformation:
            from langslice.core.deformable.render import composed_native_grid

            held = self._door.job.deformations.current(self._door.state, record)
            if held is not None and tuple(held.section_size) == image.size:
                applied = np.stack(composed_native_grid(held, 1), axis=-1)
        excluded = zone = None
        _, marked = exclusions(record)
        if marked:
            labels = planes.get("annotation")
            if labels is None:
                labels = np.asarray(native_labels(workspace.atlas, placement))
            excluded = regions_mask(workspace.atlas, labels, marked,
                                    placement_left(workspace.atlas, placement, marked))
            spacing_mm = float(np.mean(_plane_frame(workspace.atlas, placement)["px_um"])) / 1000.0
            zone = dilate(excluded, EXCLUSION_MARGIN_MM * brain_scale(workspace.atlas)
                          / spacing_mm)
        return FitFrame(
            id=record.id, image=np.asarray(image), atlas=planes,
            atlas_to_section=np.asarray(placement.atlas_to_section, dtype=np.float64),
            plane=_plane_frame(workspace.atlas, placement),
            position_mm=float(placement.position_mm),
            angles=(float(placement.pitch_deg), float(placement.yaw_deg)),
            placed_atlas_to_section=placed, applied_map=applied, excluded=excluded,
            damage_zone=zone,
            stain=(np.asarray(deformation.stain_image(
                workspace, self._door.state, grid, deformation.FIT_LOOK)[0].convert("RGB"))
                if stain else None))

    def angles_from_plane(self, normal_um: Sequence[float], point_um: Sequence[float],
                          ) -> dict[str, float]:
        """The placement of the plane through *point_um* with normal
        *normal_um* (atlas micrometres, the atlas's axis order, as
        :attr:`FitFrame.plane` gives a plane): ``position_mm`` (where it crosses
        the volume's in-plane centre, the stack's position convention),
        ``pitch_deg`` and ``yaw_deg`` (the stack's cutting-angle convention).
        Nothing is written. ``ToolError`` ``BAD_PLANE`` for a plane tilted
        past what a cutting angle can say."""
        from langslice.core.import_geometry import plane_angles, plane_position_mm
        from langslice.core.space import Plane

        atlas = self._door.ctx.atlas
        plane = cast(Plane, self._door.state.plane)
        normal = np.asarray(normal_um, dtype=np.float64)
        point = np.asarray(point_um, dtype=np.float64)
        if normal.shape != (3,) or point.shape != (3,) or not (
                np.isfinite(normal).all() and np.isfinite(point).all()) or not normal.any():
            raise ToolError("BAD_PLANE", "normal_um and point_um are three finite numbers, "
                            "the normal not zero")
        try:
            pitch, yaw = plane_angles(normal, atlas, plane)
        except ValueError as exc:
            raise ToolError("BAD_PLANE", str(exc)) from exc
        return {"position_mm": float(plane_position_mm(point, normal, atlas, plane)),
                "pitch_deg": float(pitch), "yaw_deg": float(yaw)}

    def atlas_slice(self, section: object, *, kind: str = "template") -> np.ndarray:
        """The atlas plane at the section's position and cutting angles, on
        the native atlas grid (:class:`FitFrame` ``atlas``): *kind* is
        ``annotation`` or one of the atlas's images."""
        from langslice.core.atlas.images import atlas_images

        offered = ("annotation", *atlas_images(self._door.ctx.atlas))
        if kind not in offered:
            raise ToolError("BAD_ARGS", f"kind is one of {', '.join(offered)}; got {kind!r}")
        frame = self.fit_frame(section, atlas=(kind,))
        return frame.atlas[kind]

    def transform_from_matrix(self, section: object, matrix: Any, *,
                              note: str = "") -> TransformProposal:
        """A method's linear result as a transform LangSlice stores: *matrix*
        is the 2x3 map from the fit grid's pixels (:attr:`FitFrame.image`) to
        native atlas pixels (:attr:`FitFrame.atlas`), the form registration
        engines return. The result is a refinement of the section's current
        placement exactly as an Elastix fit is, in the six normalized numbers
        and the knobs about the section's stored pivot (else the canvas
        centre). ``ToolError`` ``BAD_MATRIX`` for a degenerate map."""
        from langslice.core.affine import IDENTITY_PARAMS, denormalized_affine, normalized_affine

        record = self._record(section)
        grid = self._grid(record)
        width, height = grid.image.size
        m = np.vstack([np.asarray(matrix, dtype=np.float64).reshape(2, 3), [0.0, 0.0, 1.0]])
        if not np.isfinite(m).all() or abs(np.linalg.det(m[:2, :2])) < 1e-9:
            raise ToolError("BAD_MATRIX", "the map is degenerate or not finite")
        start = np.vstack([denormalized_affine(list(IDENTITY_PARAMS), (width, height)),
                           [0.0, 0.0, 1.0]])
        step = np.asarray(grid.placement.atlas_to_section, dtype=np.float64) @ m
        params = normalized_affine((start @ step)[:2], (width, height))
        return self.transform_from_params(record.id, params, note=note)

    def transform_from_params(self, section: object, params: Sequence[float], *,
                              note: str = "") -> TransformProposal:
        """The stored transform for the six normalized numbers *params*
        (``[a, b, tx, c, d, ty]``, on the section as its orientation sets it):
        the knobs about the section's stored pivot (else the canvas centre),
        its calibration and whether it mirrors the section."""
        from langslice.core.affine import decompose_affine, denormalized_affine
        from langslice.core.canvas import canvas_geometry
        from langslice.core.sections import PREVIEW_LONG_EDGE, render_slice
        from langslice.core.transform import calibrate, physical_params

        record = self._record(section)
        values = [float(v) for v in params]
        if len(values) != 6 or not np.isfinite(values).all():
            raise ToolError("BAD_ARGS", "a transform is six finite numbers [a, b, tx, c, d, ty]")
        if record.position_mm is None:
            raise ToolError("NO_POSITION", f"{record.id} has no position yet")
        workspace, state = self._door.ctx, self._door.state
        rendered = render_slice(workspace, record, long_edge=PREVIEW_LONG_EDGE)
        um_per_px, source = calibrate(state, workspace, record, rendered)
        try:
            geometry = canvas_geometry(rendered.size, um_per_px, workspace.atlas,
                                       record.position_mm, state.plane,  # type: ignore[arg-type]
                                       record.pitch_deg, record.yaw_deg)
        except ValueError as exc:
            raise ToolError("INVALID_LINEAR_PLACEMENT", f"{record.id}: {exc}") from exc
        in_section = denormalized_affine(values, rendered.size)
        ox, oy = (float(v) for v in geometry.section_offset)
        shift = np.array([[1.0, 0.0, ox], [0.0, 1.0, oy], [0.0, 0.0, 1.0]])
        back = np.array([[1.0, 0.0, -ox], [0.0, 1.0, -oy], [0.0, 0.0, 1.0]])
        on_canvas = (shift @ np.vstack([in_section, [0.0, 0.0, 1.0]]) @ back)[:2]
        canvas_w, canvas_h = geometry.size
        held_pivot = ((record.transform or {}).get("physical") or {}).get("pivot")
        pivot = (float(held_pivot[0]), float(held_pivot[1])) if held_pivot else (0.5, 0.5)
        knobs = physical_params(on_canvas, pivot=(canvas_w * pivot[0], canvas_h * pivot[1]),
                                um_per_px=um_per_px)
        mirrored = bool(decompose_affine(values)["mirrored"])
        calibration = {"section_um_per_px": round(float(um_per_px), 4), "source": source}
        stored = {
            "kind": self._caller.label, "params": values,
            "physical": {**knobs, "pivot": list(pivot)}, "calibration": calibration,
            "mirrored": mirrored, "note": str(note or "").strip(),
        }
        return TransformProposal(id=record.id, params=tuple(values), knobs=dict(knobs),
                                 pivot=pivot, mirrored=mirrored, calibration=calibration,
                                 record=stored)

    def deformation_from_map(self, section: object, native_xy: Any, *, note: str = "",
                             meta: Mapping[str, Any] | None = None) -> DeformationProposal:
        """A method's deformable result as the deformation LangSlice stores:
        *native_xy* is a float ``(H, W, 2)`` array giving, for each pixel
        centre ``(x, y)`` of the fit grid (:attr:`FitFrame.image`, ``H`` by
        ``W``), the native atlas pixel ``(x, y)`` (:attr:`FitFrame.atlas`) it
        corresponds to: the whole map, linear placement and deformation
        composed, in the frames :meth:`transform_from_matrix` takes. It is
        stored as every deformation is, the residual over the section's
        current linear placement, with an inverse LangSlice computes; it
        replaces whatever deformation the section holds. *note* and *meta*
        (JSON values) are kept with it. ``ToolError`` ``BAD_FIELD`` for an
        array of another shape, not float or not finite;
        ``INVALID_LINEAR_PLACEMENT`` for a section without a position or a
        transform."""
        import json

        from langslice.ops import deformable

        record = self._record(section)
        plain_meta = _plain(dict(meta or {}))
        try:
            json.dumps(plain_meta)
        except (TypeError, ValueError) as exc:
            raise ToolError("BAD_ARGS", f"meta is JSON values: {exc}") from exc
        text = str(note or "").strip()
        try:
            warp = deformable.warp_from_map(
                self._door.job, self._door.ctx, record.id, native_xy,
                method=self._caller.label, step={"note": text} if text else {},
                provenance={**self._caller.provenance(self.call, text, plain_meta),
                            "input": "full map: fit-grid pixel centres to native atlas "
                                     "pixels"})
        except Refused as refusal:
            raise ToolError(refusal.code, str(refusal), **{
                key: value for key, value in refusal.details.items()
                if key != "message"}) from refusal
        consistency = warp.record.diagnostics.get("inverse_consistency_mm") or {}
        return DeformationProposal(
            id=record.id, key=warp.key, summary=copy.deepcopy(warp.numbers),
            inverse_consistency_mm={k: round(float(v), 4) for k, v in consistency.items()},
            _warp=warp)

    def read_registration(self, path: str | Path, *, sections: Sequence[str] = (),
                          at: Mapping[object, Mapping[str, Any]] | None = None,
                          registered_on: str = "file") -> dict[str, Any]:
        """A linear registration file made elsewhere (QuickNII or VisuAlign
        JSON/XML, DeepSlice CSV/JSON/XML, a LangSlice ``registration.json``)
        as placements for this job's sections. Nothing is written. Returns
        ``{"placements": [...], "refused": {id: reason}, "missing": [id],
        "unmatched": [file names]}``; a placement is ``{"id", "position_mm",
        "pitch_deg", "yaw_deg", "rotation_deg", "flip", "params",
        "pixel_size_um", "pixel_size_source", "out_of_plane_um",
        "in_plane"}``.

        Each section is read in its CURRENT orientation (quarter turn and
        flip as the agent set them), so ``rotation_deg`` and ``flip`` are the
        section's own and ``params`` (six normalized numbers ``[a, b, tx, c,
        d, ty]``) are for the oriented render: pass them to
        :meth:`set_transforms` as ``{"id", "params"}`` (or
        :meth:`transform_from_params`) once the section holds the placement
        they were read at. ``in_plane`` reads them as rotation, scales and
        shear on that render (:func:`~langslice.core.affine.decompose_affine`).
        *at* ``{section: {"position_mm", "angles": (pitch, yaw)}}`` reads the
        file's map on a plane of your choosing (the section's final position
        and angles, e.g. a stack's propagated angle): ``params`` are then the
        file's map projected onto that plane and ``out_of_plane_um`` is how
        far the map lies off it; without *at* the file's own plane is used
        and ``out_of_plane_um`` is about 0. ``params`` is None, with the
        reason in ``params_note``, when the job has no pixel size for the
        section (the six numbers are relative to a pixel size, so they mean
        nothing to a job that would calibrate otherwise).

        *registered_on* is the image the file's entries were made on:
        ``"file"`` (default) the image file as stored, ``"oriented"`` the
        section as its orientation turns and flips it, which is what a method
        sees when the plugin hands it :meth:`section_raster`; the six numbers
        are for the oriented render either way."""
        from langslice.job.imports import import_placements

        self._fresh()
        state = self._door.state
        names = [self._record(ref).id for ref in sections] or [r.id for r in state.slices]
        planes: dict[str, tuple[float, float, float]] = {}
        for ref, plane in (at or {}).items():
            name = self._record(ref).id
            record = state.by_id(name)
            assert record is not None
            position = plane.get("position_mm", record.position_mm)
            angles = plane.get("angles") or (record.pitch_deg, record.yaw_deg)
            if position is None:
                raise ToolError("NO_POSITION", f"{name} has no position to read the file at")
            planes[name] = (float(position), float(angles[0]), float(angles[1]))
        result = import_placements(
            Path(path), self._door.ctx, section_ids=names,
            orientations={r.id: (r.rotation_deg, r.flip) for r in state.slices},
            planes=planes, registered_on=registered_on)
        rows = []
        for p in result.placements:
            calibrated = p.pixel_size_source != "imported"
            rows.append({
                "id": p.section_id, "position_mm": round(float(p.position_mm), 4),
                "pitch_deg": round(float(p.pitch_deg), 3),
                "yaw_deg": round(float(p.yaw_deg), 3),
                "rotation_deg": int(p.rotation_deg), "flip": bool(p.flip),
                "params": [float(v) for v in p.params] if calibrated else None,
                **({} if calibrated else {
                    "params_note": "the job has no pixel size for this section; the file's "
                                   "numbers imply one, which this job would not use"}),
                "pixel_size_um": float(p.pixel_size_um),
                "pixel_size_source": p.pixel_size_source,
                "out_of_plane_um": round(float(p.placement.out_of_plane_um), 3),
                "in_plane": {k: (round(float(v), 5) if isinstance(v, float) else v)
                             for k, v in p.placement.in_plane.items()},
            })
        return {"placements": rows, "refused": dict(result.refused),
                "missing": list(result.missing), "unmatched": list(result.unmatched)}

    # --- an atlas plane anywhere -------------------------------------------------------------

    def atlas_plane(self, position_mm: float, *, pitch_deg: float = 0.0, yaw_deg: float = 0.0,
                    kinds: Sequence[str] = ("template", "annotation")) -> AtlasPlane:
        """The atlas plane at *position_mm* (atlas millimetres along the
        slicing axis, the stack's convention) and cutting angles *pitch_deg*
        / *yaw_deg*, whichever section holds it, on the native atlas grid
        (:class:`AtlasPlane`): ``annotation`` and the atlas's images among
        *kinds* (a name the atlas does not have is skipped). Nothing is
        written."""
        from langslice.core.atlas.images import atlas_images
        from langslice.core.deformable.geometry import Placement
        from langslice.core.deformable.plane_images import native_intensity, native_labels

        workspace = self._door.ctx
        values = [float(position_mm), float(pitch_deg), float(yaw_deg)]
        if not np.isfinite(values).all():
            raise ToolError("BAD_ARGS", "position_mm, pitch_deg and yaw_deg are finite numbers")
        placement = Placement(atlas_name=str(self._door.state.atlas), position_mm=values[0],
                              plane=self._door.state.plane,  # type: ignore[arg-type]
                              pitch_deg=values[1], yaw_deg=values[2],
                              atlas_to_section=np.eye(3), section_mm_per_px=1.0)
        planes: dict[str, np.ndarray] = {}
        if "annotation" in kinds:
            planes["annotation"] = np.asarray(native_labels(workspace.atlas, placement))
        for kind in atlas_images(workspace.atlas):
            if kind in kinds:
                planes[kind] = np.asarray(native_intensity(workspace.atlas, placement, kind),
                                          dtype=np.float32)
        return AtlasPlane(position_mm=values[0], angles=(values[1], values[2]), atlas=planes,
                          plane=_plane_frame(workspace.atlas, placement))

    def atlas_volume(self, kind: str = "template") -> np.ndarray:
        """The atlas's whole volume *kind* (``annotation`` or one of its
        images) on its own grid, the atlas's axis order (read only: copy it
        before changing it)."""
        from langslice.core.atlas.images import atlas_images
        from langslice.core.atlas.images import atlas_volume as volume

        atlas = self._door.ctx.atlas
        if kind == "annotation":
            return np.asarray(atlas.annotation)
        if kind not in atlas_images(atlas):
            raise ToolError("BAD_ARGS", f"kind is annotation or one of "
                            f"{', '.join(atlas_images(atlas))}; got {kind!r}")
        return volume(atlas, kind)

    # --- points through the registration ----------------------------------------------------

    def to_atlas_um(self, section: object, xy: Any, *, frame: str = "file") -> np.ndarray:
        """Section points *xy* (``(N, 2)`` ``[x, y]`` pixels: of the image file
        as stored with *frame* ``file``, the ``section_px`` of layers; of the
        fit grid, :attr:`FitFrame.image`, with ``fit``) in atlas micrometres,
        ``(N, 3)`` in the atlas's axis order, through the section's current
        registration, its deformation included; NaN outside the atlas volume."""
        from langslice.ops.points import atlas_points

        record = self._record(section)
        return self._points(atlas_points, record.id, xy, frame).um

    def from_atlas_um(self, section: object, xyz_um: Any, *, frame: str = "file",
                      ) -> np.ndarray:
        """Atlas points *xyz_um* (``(N, 3)`` micrometres, the atlas's axis
        order) as ``[x, y]`` pixels of the section (*frame* as
        :meth:`to_atlas_um`), each projected onto the section's atlas plane
        and carried back through its current registration, its deformation's
        inverse included. :meth:`plane_distance_um` says how far off the
        plane each point lies."""
        from langslice.ops.points import section_points

        record = self._record(section)
        return self._points(section_points, record.id, xyz_um, frame).xy

    def plane_distance_um(self, section: object, xyz_um: Any) -> np.ndarray:
        """How far each atlas point *xyz_um* lies off the section's atlas
        plane, in micrometres, ``(N,)``."""
        from langslice.ops.points import section_points

        record = self._record(section)
        return self._points(section_points, record.id, xyz_um, "file").off_plane_um

    def region_at(self, section: object, xy: Any, *, frame: str = "file",
                  ) -> list[dict[str, Any] | None]:
        """The atlas region under each section point *xy* (*frame* as
        :meth:`to_atlas_um`) through the section's current registration:
        ``{"id", "acronym", "name"}`` per point, None outside the atlas."""
        from langslice.ops.points import atlas_points

        record = self._record(section)
        found = self._points(atlas_points, record.id, xy, frame)
        structures = self._door.ctx.atlas.structures
        out: list[dict[str, Any] | None] = []
        for value in found.ids.tolist():
            if not value:
                out.append(None)
                continue
            try:
                entry = structures[int(value)]
                out.append({"id": int(value), "acronym": str(entry["acronym"]),
                            "name": str(entry["name"])})
            except KeyError:
                out.append({"id": int(value), "acronym": None, "name": None})
        return out

    def _points(self, function: Callable[..., Any], name: str, values: Any, frame: str) -> Any:
        try:
            return function(self._door.job, self._door.ctx, name, values, frame=frame)
        except Refused as refusal:
            raise _as_tool_error(refusal) from refusal

    # --- an outside engine's files -----------------------------------------------------------

    def write_frame(self, section: object, folder: str | Path, *, format: str = "tif",
                    ) -> dict[str, Path]:
        """Write what an engine outside LangSlice registers, into *folder*:
        on the fit grid (:attr:`FitFrame.image`'s pixels), ``section``
        (RGB), ``stain`` (the preprocessed channel the native fits read,
        gray) and ``placed_template`` / ``placed_annotation`` (the atlas
        plane drawn on that grid at the section's current placement, so the
        two images already overlap where the placement puts them); on the
        native atlas grid ``template`` and ``annotation``; and
        ``frame.json`` (the sizes, ``placed_atlas_to_section``,
        ``atlas_to_section``, ``native_to_um``, the fit grid's millimetres
        per pixel, the files). *format* is the images' suffix (``tif``, or
        ``nii.gz``, ``nii``, ``mha``, ``nrrd`` through SimpleITK); every
        image has spacing 1 and origin 0, so an engine's answer is in pixels
        (:meth:`map_from_itk` reads it back). Returns ``{name: path}``.
        Writes nothing in the job."""
        import json

        import cv2

        from langslice.core.itk_io import IMAGE_FORMATS, write_image

        if format not in IMAGE_FORMATS:
            raise ToolError("BAD_ARGS", f"format is one of {', '.join(IMAGE_FORMATS)}; "
                            f"got {format!r}")
        record = self._record(section)
        f = self.fit_frame(record.id, stain=True)
        height, width = f.image.shape[:2]
        placed = (f.placed_atlas_to_section if f.placed_atlas_to_section is not None
                  else f.atlas_to_section)
        target = Path(folder)
        target.mkdir(parents=True, exist_ok=True)
        stain = np.asarray(Image.fromarray(np.asarray(f.stain)).convert("L"))
        images: dict[str, np.ndarray] = {
            "section": np.asarray(f.image, dtype=np.uint8),
            "stain": stain,
            "template": np.asarray(f.atlas["template"], dtype=np.float32),
            "annotation": np.asarray(f.atlas["annotation"]).astype(np.uint32),
            "placed_template": cv2.warpAffine(
                np.asarray(f.atlas["template"], dtype=np.float32), placed[:2], (width, height),
                flags=cv2.INTER_LINEAR, borderValue=0.0),
            "placed_annotation": cv2.warpAffine(
                np.asarray(f.atlas["annotation"]).astype(np.float32), placed[:2],
                (width, height), flags=cv2.INTER_NEAREST, borderValue=0.0).astype(np.uint32),
        }
        written = {name: write_image(target / f"{name}.{format}", array)
                   for name, array in images.items()}
        grid = self._grid(record)
        info = {
            "section": record.id,
            "fit_size": [int(width), int(height)],
            "native_size": [int(f.atlas["annotation"].shape[1]),
                            int(f.atlas["annotation"].shape[0])],
            "fit_mm_per_px": float(grid.placement.section_mm_per_px),
            "placed_atlas_to_section": np.asarray(placed).tolist(),
            "atlas_to_section": np.asarray(f.atlas_to_section).tolist(),
            "native_to_um": f.plane["native_to_um"],
            "position_mm": f.position_mm, "angles": list(f.angles),
            "files": {name: path.name for name, path in written.items()},
            "convention": "Pixel centres at integers, [x, y] = [column, row]; every image "
                          "spacing 1, origin 0. section, stain and placed_* on the fit grid; "
                          "template and annotation on the native atlas grid. "
                          "placed_atlas_to_section (3x3 on [x, y, 1]) takes a native atlas "
                          "pixel to the fit-grid pixel the current placement puts it on; "
                          "native_to_um a native pixel [x, y, 1] to atlas micrometres.",
        }
        frame_file = target / "frame.json"
        frame_file.write_text(json.dumps(info, indent=1) + "\n", encoding="utf-8")
        return {**written, "frame": frame_file}

    def map_from_itk(self, section: object, source: Any, *, fixed: str = "section",
                     atlas: str = "placed", spacing: float = 1.0) -> np.ndarray:
        """An outside engine's answer as the full map
        :meth:`deformation_from_map` and :meth:`set_deformations` take
        (``(H, W, 2)`` native atlas pixels of every fit-grid pixel centre).

        *source*: a displacement-field image file (``.nii.gz``, ``.mha``,
        ``.nrrd``...), a transform file (``.mat``, ``.txt``, ``.tfm``,
        ``.h5``: ANTs' and ITK's), a ``SimpleITK`` transform or vector image,
        or a float array ``(H, W, 2)`` of displacements ``(dx, dy)`` in
        fit-grid pixels. Read in ITK's convention: it maps each FIXED image
        point to the MOVING point it corresponds to. *fixed* says which image
        was fixed: ``section`` (the section's ``stain`` or ``section`` of
        :meth:`write_frame`) or ``atlas`` (then the map is inverted, on the
        fit grid). *atlas* says which atlas image the engine used:
        ``placed`` (``placed_template``, on the fit grid) or ``native``
        (``template``; with *fixed* ``section`` only). *spacing* is the
        images' spacing per pixel when they were not :meth:`write_frame`'s."""
        from langslice.core.itk_io import inverted, mapped_points

        if fixed not in ("section", "atlas") or atlas not in ("placed", "native"):
            raise ToolError("BAD_ARGS", "fixed is 'section' or 'atlas'; atlas is 'placed' or "
                            "'native'")
        if fixed == "atlas" and atlas == "native":
            raise ToolError("BAD_ARGS", "a field with the atlas fixed is read on the fit grid: "
                            "register against placed_template (atlas='placed')")
        record = self._record(section)
        width, height = self._grid(record).image.size
        try:
            points = mapped_points(source, (width, height), spacing=float(spacing))
            if fixed == "atlas":
                points = inverted(points)
        except (ValueError, RuntimeError) as exc:
            raise ToolError("BAD_FIELD", f"{record.id}: {exc}") from exc
        if atlas == "native":
            return points
        to_native = np.linalg.inv(self._placed(record))
        return np.stack([to_native[0, 0] * points[..., 0] + to_native[0, 1] * points[..., 1]
                         + to_native[0, 2],
                         to_native[1, 0] * points[..., 0] + to_native[1, 1] * points[..., 1]
                         + to_native[1, 2]], axis=-1)

    def transform_from_itk(self, section: object, source: Any, *, fixed: str = "section",
                           atlas: str = "placed", spacing: float = 1.0,
                           note: str = "") -> TransformProposal:
        """An outside engine's affine (a transform file such as ANTs'
        ``0GenericAffine.mat``, or a ``SimpleITK`` transform; *fixed*,
        *atlas* and *spacing* as :meth:`map_from_itk`, either way round) as
        a :class:`TransformProposal`, like :meth:`transform_from_matrix`.
        ``BAD_MATRIX`` for a transform that is not affine."""
        from langslice.core.itk_io import linear_matrix

        if fixed not in ("section", "atlas") or atlas not in ("placed", "native"):
            raise ToolError("BAD_ARGS", "fixed is 'section' or 'atlas'; atlas is 'placed' or "
                            "'native'")
        record = self._record(section)
        size = self._grid(record).image.size
        try:
            matrix = linear_matrix(source, (int(size[0]), int(size[1])), spacing=float(spacing))
        except (ValueError, RuntimeError) as exc:
            raise ToolError("BAD_MATRIX", f"{record.id}: {exc}") from exc
        if fixed == "atlas":
            if abs(np.linalg.det(matrix[:2, :2])) < 1e-12:
                raise ToolError("BAD_MATRIX", f"{record.id}: the map is degenerate")
            matrix = np.linalg.inv(matrix)
        if atlas == "placed":
            matrix = np.linalg.inv(self._placed(record)) @ matrix
        return self.transform_from_matrix(record.id, matrix[:2], note=note)

    def _placed(self, record: SliceState) -> np.ndarray:
        """3x3: native atlas pixels -> fit-grid pixels under the section's
        current placement (its stored transform, else none)."""
        from langslice.core import deformation

        if record.transform:
            try:
                return np.asarray(deformation.fit_grid(
                    self._door.state, self._door.ctx, record).placement.atlas_to_section,
                    dtype=np.float64)
            except ValueError as exc:
                raise ToolError("INVALID_LINEAR_PLACEMENT", f"{record.id}: {exc}") from exc
        return np.asarray(self._grid(record).placement.atlas_to_section, dtype=np.float64)

    # --- the native engines, with settings of your own --------------------------------------

    def fit_fireants(self, section: object, *, settings: Any = None, mask: Any = None,
                     start: str = "linear", em: bool = True, marked: bool = True,
                     note: str = "") -> DeformationProposal:
        """Run FireANTs SyN (the engine ``fireants_syn`` runs) on the section
        with settings of your own and return the result as a
        :class:`DeformationProposal`, NOT written (:meth:`set_deformations`
        writes it, recorded as this caller's, with the settings in its
        provenance ``meta``).

        *settings*: a :class:`langslice.core.deformable.FitSettings` or a
        mapping of its fields over ``fireants_syn``'s ``medium`` defaults
        (``stiffness`` ``soft`` / ``medium`` / ``firm``, ``detail``
        ``coarse`` / ``standard``, ``atlas_image``, ``stain_metric``,
        ``stain_edges``, ``structures`` and ``neighbourhood_um`` to fit
        some regions only, ``exclude``). *mask*: bool, the fit grid's
        ``(H, W)`` (:attr:`FitFrame.image`), True where the section is
        compared (left out: a tear, a bubble). *start* ``linear`` fits from
        the linear placement, ``current`` on top of the applied deformation.
        *em* False skips the EM section mask's second run; *marked* False
        leaves the section's marked damage regions in the fit."""
        from dataclasses import fields as dataclass_fields

        from langslice.core import deformation
        from langslice.core.deformable import FitSettings
        from langslice.core.deformable.render import composed_native_grid
        from langslice.ops.deformable import fit_record

        record = self._record(section)
        workspace = self._door.ctx
        base = deformation.Choice(fit_section=deformation.FIT_LOOK, fit_atlas="template",
                                  stiffness="medium").settings(
            (), (), brain_scale=workspace.brain_scale)
        if settings is None:
            chosen = base
        elif isinstance(settings, FitSettings):
            chosen = settings
        elif isinstance(settings, Mapping):
            known = {item.name for item in dataclass_fields(FitSettings)}
            unknown = sorted(set(settings) - known)
            if unknown:
                raise ToolError("BAD_ARGS", f"unknown settings {unknown}; FitSettings has "
                                f"{sorted(known)}")
            try:
                chosen = replace(base, **dict(settings))  # checked as it is made
            except (TypeError, ValueError) as exc:
                raise ToolError("BAD_ARGS", str(exc)) from exc
        else:
            raise ToolError("BAD_ARGS", "settings is a FitSettings or a mapping of its fields")
        keep = None
        if mask is not None:
            keep = np.asarray(mask).astype(bool)
        try:
            fitted = fit_record(self._door.job, workspace, record.id, chosen, start=start,
                                mask=keep, em=bool(em), marked=bool(marked))
        except Refused as refusal:
            raise _as_tool_error(refusal) from refusal
        native_xy = np.stack(composed_native_grid(fitted, 1), axis=-1).astype(np.float64)
        meta = {"engine": "fireants", "settings": fitted.settings.to_dict()
                if fitted.settings is not None else chosen.to_dict(),
                "start": start, "em": bool(em), "marked": bool(marked),
                "mask_pixels": None if keep is None else int(keep.sum()),
                "runtime_s": round(float(fitted.engine.get("runtime_s", 0.0)), 1)}
        return self.deformation_from_map(record.id, native_xy,
                                         note=note or "FireANTs SyN, settings of the caller's",
                                         meta=meta)

    def fit_elastix(self, section: object, *, restrict_to: Sequence[str] = (),
                    exclude: Sequence[str] = (), atlas_image: str = "template",
                    mask: Any = None, marked: bool = True,
                    note: str = "") -> TransformProposal:
        """Run the Elastix affine (the engine ``elastix_affine`` runs) on the
        section from its current placement and return the result as a
        :class:`TransformProposal`, NOT written (:meth:`set_transforms`
        writes it; its record keeps the call's settings under ``fit``).
        *restrict_to* / *exclude*: region entries (``"CTX:left"`` for one
        side); *mask* (bool, the fit grid's ``(H, W)``) True where the
        section is compared; *marked* False keeps the section's marked
        damage regions in the fit."""
        from langslice.ops.transforms import elastix_fit

        record = self._record(section)
        keep = None if mask is None else np.asarray(mask).astype(bool)
        try:
            fitted = elastix_fit(self._door.job, self._door.ctx, record.id,
                                 restrict_to=list(restrict_to), exclude=list(exclude),
                                 atlas_image=str(atlas_image), mask=keep, marked=bool(marked))
        except Refused as refusal:
            raise _as_tool_error(refusal) from refusal
        proposal = self.transform_from_params(record.id, fitted.params,
                                              note=note or "Elastix affine, settings of the "
                                              "caller's")
        stored = {**proposal.record, "fit": {
            "engine": "elastix", "restrict_to": list(fitted.report.get("include") or ()),
            "exclude": list(fitted.report.get("exclude") or ()), "atlas_image": atlas_image,
            "marked": bool(marked), "mask_pixels": None if keep is None else int(keep.sum())}}
        return replace(proposal, record=stored)

    # --- output for the agent ---------------------------------------------------------------

    def overlay_picture(self, section: object, *, proposal: Mapping[str, Any] | None = None,
                        resolution: int | None = None) -> Image.Image:
        """The overlay ``look`` draws of the section (borders on the section,
        captioned), as it stands, or with a *proposal* not yet written:
        ``{"transform": record or params, "position_mm": mm, "angles":
        (pitch, yaw)}``, any of them, or ``{"deformation": proposal or map}``
        alone (a :class:`DeformationProposal`, or the map
        :meth:`deformation_from_map` takes), drawn on the section's current
        placement. Nothing is saved or written here; the picture is the
        handler's to return."""
        from langslice.core import look as looks

        record = self._record(section)
        state = self._door.state
        store = self._door.job.deformations
        if proposal and proposal.get("deformation") is not None:
            if set(proposal) - {"deformation"}:
                raise ToolError("BAD_ARGS", "a deformation proposal is drawn on the "
                                "section's current placement: give it alone")
            warped = proposal["deformation"]
            if not isinstance(warped, DeformationProposal):
                warped = self.deformation_from_map(record.id, warped)
            elif warped.id != record.id:
                raise ToolError("BAD_ARGS", f"the proposal is {warped.id}'s, not {record.id}'s")
            store = self._proposal_store(warped)
            state = StackState.from_dict(copy.deepcopy(state.to_dict()))
            held = state.by_id(record.id)
            assert held is not None
            held.deformation = {"key": warped.key, "linear_key": warped._warp.linear_key,
                                "steps": [dict(warped._warp.step)]}
            record = held
        elif proposal:
            state = StackState.from_dict(copy.deepcopy(state.to_dict()))
            held = state.by_id(record.id)
            assert held is not None
            transform = proposal.get("transform")
            if isinstance(transform, TransformProposal):
                transform = transform.record
            elif transform is not None and not isinstance(transform, Mapping):
                transform = self.transform_from_params(record.id, transform).record
            if transform is not None:
                held.transform = dict(transform)
                held.deformation = None
            if proposal.get("position_mm") is not None:
                held.position_mm = float(proposal["position_mm"])
                held.deformation = None
            if proposal.get("angles") is not None:
                held.cutting_angles_deg = {"pitch": float(proposal["angles"][0]),
                                           "yaw": float(proposal["angles"][1])}
                held.deformation = None
            record = held
        edge = int(resolution) if resolution else None
        request = looks.LookRequest(mode="overlay", sections=(record.id,), long_edge=edge)
        try:
            drawn = looks.look(self._door.ctx, state, request, store=store)
        except looks.LookError as exc:
            raise ToolError(exc.code, str(exc)) from exc
        return drawn[0].image

    def positioning_picture(self, ids: Sequence[object], *,
                            proposal: Mapping[object, Mapping[str, Any]] | None = None,
                            resolution: int | None = None, part: int | None = None,
                            ) -> list[Image.Image]:
        """The positioning picture ``position_sections`` shows (``look`` mode
        ``positioning``: the sections in order beside the atlas planes at
        their positions, captioned), as it stands, or with a *proposal* not
        yet written: ``{section: {"position_mm": mm, "angles": (pitch,
        yaw)}}``, either key, for any of the sections. A proposed picture has
        a last caption band saying it is not written. A long stack is several
        pictures (parts of the stack): all are returned, in order, or with
        *part* (0-based) only that one. Sections without a position are left
        out of the atlas column as in ``look``. *resolution* is the long edge
        of each section's tile (None: the job's level). Nothing is saved or
        written here; the pictures are the handler's to return. Refused
        (``POSITIONING_OFF``) where the job disables ``look:positioning``."""
        from langslice.core import look as looks
        from langslice.doors.tools.changing import _positions

        if self._door.spec.off("look:positioning"):
            raise ToolError("POSITIONING_OFF", "the positioning picture is switched off for "
                            "this job")
        names = [self._record(ref).id for ref in ids]
        state = self._door.state
        if proposal:
            state = StackState.from_dict(copy.deepcopy(state.to_dict()))
            for ref, change in proposal.items():
                held = state.by_id(self._record(ref).id)
                assert held is not None
                unknown = set(change) - {"position_mm", "angles"}
                if unknown:
                    raise ToolError("BAD_ARGS", "a positioning proposal gives position_mm and/or "
                                    f"angles per section; got {sorted(unknown)}")
                if change.get("position_mm") is not None:
                    held.position_mm = float(change["position_mm"])
                if change.get("angles") is not None:
                    held.cutting_angles_deg = {"pitch": float(change["angles"][0]),
                                               "yaw": float(change["angles"][1])}
                held.deformation = None
        records = [state.by_id(name) for name in names]
        positions = _positions([float(r.position_mm) for r in records
                                if r is not None and r.position_mm is not None])
        edge = int(resolution) if resolution else None
        request = looks.LookRequest(mode="positioning", sections=tuple(names),
                                    positions_mm=tuple(positions), long_edge=edge,
                                    part=part)
        try:
            drawn = looks.look(self._door.ctx, state, request, store=self._door.job.deformations)
        except looks.LookError as exc:
            raise ToolError(exc.code, str(exc)) from exc
        images = [picture.image for picture in drawn]
        if proposal:
            images = [self.caption(image, "proposal: positions and angles drawn as proposed, "
                                   "NOT written") for image in images]
        return images

    def _proposal_store(self, proposal: DeformationProposal) -> Any:
        """A record store that holds *proposal*'s record (in memory only) and
        reads every other from the job's."""
        from langslice.core.deformation import RecordStore

        held = self._door.job.deformations
        store = RecordStore(root=held.root, folder_of=held.folder_of)
        store.put(proposal.key, proposal._warp.record)
        return store

    @staticmethod
    def caption(image: Image.Image, text: str) -> Image.Image:
        """*image* with *text* in a caption band below it (pictures reach a
        model as bare attachments: what they show is said in the picture)."""
        from langslice.core.captions import caption

        return caption(image, text)

    # --- background work --------------------------------------------------------------------

    def start_work(self, sections: Sequence[object],
                   land: Callable[[ToolContext], ToolResult]) -> str:
        """Start *land* as a piece of the job's background work on *sections*
        and return its id at once (``w3``); the tool returns meanwhile.

        On a work thread ``land(ctx)`` computes and writes through *ctx*, a
        context of its own for the same tool (``with ctx.writing():`` makes
        its writes one undo step; pass ``expect=`` digests taken at the
        start, as a long tool does). Its :class:`ToolResult` is the work's
        notice: ``reply["notice"]`` (one paragraph) its text, the reply its
        result and the pictures saved and numbered with the job's. A
        :class:`ToolError` or :class:`Refused` it raises fails the work with
        its message. The next tool reply carries the notice
        (``background``); ``status`` lists running work; ``submit`` waits
        for it. While it runs, a write to any of *sections*, by any tool, is
        refused ``SECTION_BUSY``; so is starting it while other work runs
        there."""
        from langslice.job.background import DONE, FAILED, Landed

        if not self._caller.background:
            raise _not_for_scripts("a script runs its own computation: compute, then write "
                                   "with expect=")
        job = self._door.job
        ids = [self._record(section).id for section in sections]
        busy = job.busy_refusal(ids)
        if busy is not None:
            raise Refused(busy.pop("error"), **busy)
        door, caller = self._door, self._caller

        def landing() -> Landed:
            context = ToolContext(door, caller, f"{self.call}-bg")
            try:
                result = land(context)
            except (ToolError, Refused) as exc:
                return Landed(status=FAILED, text=str(getattr(exc, "message", "") or exc))
            reply = dict(result.reply)
            return Landed(status=DONE, text=str(reply.pop("notice", "") or "done."),
                          pictures=[item if isinstance(item, Image.Image)
                                    else Image.fromarray(np.asarray(item))
                                    for item in result.pictures],
                          result=_plain(reply))

        return job.background.start(caller.name, ids, landing, wait_for_images=False).id

    # --- writing ----------------------------------------------------------------------------

    def _require(self, *properties: str) -> None:
        caller = self._caller
        if caller.kind != "write":
            raise ToolError("READ_TOOL_WRITES", f"{caller.name} is declared a read tool and "
                            "writes nothing; declare kind='write'")
        outside = [name for name in properties if name not in caller.writes]
        if outside:
            raise ToolError("WRITE_NOT_DECLARED", f"{caller.name} writes {outside} but "
                            f"declares writes={list(caller.writes)}")

    @contextlib.contextmanager
    def writing(self) -> Iterator[ToolContext]:
        """Hold the job's write lock around several writes: everything
        written inside is ONE undo step. A plugin call's writes are one step
        too, unless another write landed on the job between two of the
        call's (then they stay separate steps, so undo never takes back
        someone else's write); a script's outermost writes are a step each.
        A long tool computes first and applies inside, passing ``expect=``
        to the writes so a section that changed meanwhile is refused."""
        job = self._door.job
        with job.writing():
            outermost = self._depth == 0
            mark = job.pushed
            if outermost and self._step is not None and self._caller.merge_calls:
                start, pushed, top = self._step
                if job.pushed == pushed and job.undo_stack and job.undo_stack[-1] is top:
                    mark = start  # nothing else wrote since the call's last write
            self._depth += 1
            try:
                yield self
            finally:
                self._depth -= 1
                job.merge_steps(mark)
                if outermost and job.pushed > mark and job.undo_stack:
                    self._step = (mark, job.pushed, job.undo_stack[-1])

    def _stale(self, ids: Sequence[str], expect: Mapping[str, str] | None,
               ) -> dict[str, dict[str, Any]]:
        """The rows of sections whose inputs differ from *expect*."""
        rows: dict[str, dict[str, Any]] = {}
        for name in ids:
            if expect is None or name not in expect:
                continue
            record = self._door.state.by_id(name)
            if record is None:
                continue
            now = {section_inputs(self._door.state, record, deformation=flag)
                   for flag in (False, True)}  # taken with or without the deformation
            if expect[name] not in now:
                rows[name] = stale_row(name)
        return rows

    def set_positions(self, rows: Sequence[Mapping[str, Any]], *,
                      cutting_angles: Mapping[str, float] | None = None,
                      expect: Mapping[str, str] | None = None) -> WriteResult:
        """Set positions (``rows``: ``{"id", "position_mm"}``; atlas
        millimetres, clamped into the atlas range and reported) and, with
        *cutting_angles* (``{"pitch_deg", "yaw_deg"}``), the stack's angles.
        The stack then runs in the order of the positions. Needs ``position``
        (and ``angles``) in the tool's ``writes``."""
        from langslice.ops import positions

        self._require("position", *(["angles"] if cutting_angles is not None else []))
        job, workspace = self._door.job, self._door.ctx
        with self.writing():
            out: dict[str, dict[str, Any]] = {}
            todo: list[Mapping[str, Any]] = []
            ids = [self._resolve_id(row.get("id")) for row in rows]
            stale = self._stale([name for name in ids if name], expect)
            for row, name in zip(rows, ids, strict=True):
                if not name:
                    out[str(row.get("id"))] = {"id": str(row.get("id")), "status": "error",
                                               "error": "UNKNOWN_SLICE_IDS"}
                elif name in stale:
                    out[name] = stale[name]
                elif (frozen := frozen_row(job, name, "position")) is not None:
                    out[name] = frozen
                else:
                    todo.append(row)
            try:
                done = positions.position_sections(job, workspace, todo, cutting_angles) \
                    if todo or cutting_angles is not None else None
            except Refused as refusal:
                return WriteResult.refused(refusal)
            written = [] if done is None else [name for name, _mm in done.written]
            for name, mm in ([] if done is None else done.written):
                out[name] = {"id": name, "status": "ok", "position_mm": round(float(mm), 4)}
            for name, asked, used in ([] if done is None else done.clamped):
                out[name]["clamped_from_mm"] = round(float(asked), 4)
                out[name]["position_mm"] = round(float(used), 4)
            cleared = self._cleared()
        reply = WriteResult(written=written, rows=list(out.values()))
        if done is not None and done.cutting_angles is not None:
            reply.details["cutting_angles"] = dict(done.cutting_angles)
        if cleared:
            reply.details["deformation_cleared"] = cleared
        return reply

    def _cleared(self) -> list[str]:
        """Outside a tool call (a script), the deformations a placement write
        made stale, dropped in its undo step (inside one the tool door does
        it after the call)."""
        if self._caller.in_tool_call:
            return []
        return self._door.job.clear_stale_deformations()

    def _resolve_id(self, ref: object) -> str:
        record = self._door.state.resolve(ref)
        return record.id if record is not None else ""

    def set_transforms(self, rows: Sequence[Mapping[str, Any]], *,
                       expect: Mapping[str, str] | None = None) -> WriteResult:
        """Set in-plane transforms. A row is ``{"id", "transform": record}``
        (a :class:`TransformProposal`'s ``record``, or :meth:`TransformProposal.row`)
        or ``{"id", "params": [six numbers], "note"?}``. A frozen affine, or
        a transform that would turn over a section whose orientation is
        frozen, comes back as that section's row (``FROZEN``); the others are
        written as one undo step. Needs ``affine`` in the tool's ``writes``."""
        from langslice.ops import transforms

        self._require("affine")
        job = self._door.job
        with self.writing():
            out: dict[str, dict[str, Any]] = {}
            ids = [self._resolve_id(row.get("id")) for row in rows]
            stale = self._stale([name for name in ids if name], expect)
            written: list[str] = []
            for row, name in zip(rows, ids, strict=True):
                if not name:
                    out[str(row.get("id"))] = {"id": str(row.get("id")), "status": "error",
                                               "error": "UNKNOWN_SLICE_IDS"}
                    continue
                if name in stale:
                    out[name] = stale[name]
                    continue
                record = row.get("transform")
                try:
                    if record is None:
                        if "params" not in row:
                            raise ToolError("BAD_ARGS", "a row is {id, transform} or "
                                            "{id, params}")
                        record = self.transform_from_params(
                            name, row["params"], note=str(row.get("note") or "")).record
                    done = transforms.set_transforms(job, {name: dict(record)})
                except ToolError as exc:
                    out[name] = {"id": name, "status": "error", "error": exc.code,
                                 "message": exc.message}
                    continue
                except Refused as refusal:
                    out[name] = {"id": name, "status": "error", **refusal.payload()}
                    out[name].pop("ids", None)
                    continue
                written += done
                out[name] = {"id": name, "status": "ok"}
            cleared = self._cleared()
        reply = WriteResult(written=written, rows=list(out.values()))
        if cleared:
            reply.details["deformation_cleared"] = cleared
        return reply

    def set_deformations(self, rows: Sequence[Mapping[str, Any]], *,
                         expect: Mapping[str, str] | None = None) -> WriteResult:
        """Set deformations. A row is ``{"id", "map": native_xy, "note"?,
        "meta"?}`` (the full map :meth:`deformation_from_map` takes) or
        ``{"id", "deformation": proposal}`` (:meth:`DeformationProposal.row`).
        Each replaces the section's deformation, as the deformation a native
        fit applies is stored (one undo step for the call; a later change to
        the section's placement clears it). Per section: a frozen deformation
        is ``FROZEN``, a map that cannot be stored ``BAD_FIELD``, a section
        without a position or transform ``INVALID_LINEAR_PLACEMENT``, one whose
        inputs differ from *expect* (or whose placement moved since its
        proposal was made) ``STALE_INPUT``. A map is converted before the
        job's write lock is taken, so a long tool holds it only to write.
        The ok rows carry the numbers ``DeformationProposal.summary`` has.
        Needs ``deformation`` in the tool's ``writes``."""
        from langslice.ops import deformable

        self._require("deformation")
        job = self._door.job
        out: dict[str, dict[str, Any]] = {}
        proposals: list[DeformationProposal] = []
        for row in rows:
            name = self._resolve_id(row.get("id"))
            if not name:
                out[str(row.get("id"))] = {"id": str(row.get("id")), "status": "error",
                                           "error": "UNKNOWN_SLICE_IDS"}
                continue
            frozen = frozen_row(job, name, "deformation")
            if frozen is not None:
                out[name] = frozen
                continue
            proposal = row.get("deformation")
            try:
                if proposal is None:
                    if "map" not in row:
                        raise ToolError("BAD_ARGS", "a row is {id, map, note?, meta?} or "
                                        "{id, deformation}")
                    proposal = self.deformation_from_map(
                        name, row["map"], note=str(row.get("note") or ""),
                        meta=row.get("meta"))
                elif not isinstance(proposal, DeformationProposal) or proposal.id != name:
                    raise ToolError("BAD_ARGS", "deformation is this section's "
                                    "DeformationProposal (deformation_from_map)")
            except ToolError as exc:
                out[name] = {"id": name, "status": "error", "error": exc.code,
                             "message": exc.message}
                continue
            out[name] = {"id": name}
            proposals.append(proposal)
        written: list[str] = []
        with self.writing():
            stale = self._stale([proposal.id for proposal in proposals], expect)
            out.update(stale)
            done = deformable.apply_warps(
                job, [proposal._warp for proposal in proposals if proposal.id not in stale])
            for applied in done.rows:
                out[applied["id"]] = applied
            written = list(done.written)
        return WriteResult(written=written, rows=list(out.values()))

    def mark_damage(self, rows: Sequence[Mapping[str, Any]], *,
                    expect: Mapping[str, str] | None = None) -> WriteResult:
        """Mark atlas regions a section has lost (``rows``: ``{"id", "regions",
        "note"?}``; empty regions clear the marks), as ``mark_damage`` does."""
        from langslice.ops import damage

        self._require()
        job, workspace = self._door.job, self._door.ctx
        with self.writing():
            out: list[dict[str, Any]] = []
            written: list[str] = []
            for row in rows:
                name = self._resolve_id(row.get("id"))
                if not name:
                    out.append({"id": str(row.get("id")), "status": "error",
                                "error": "UNKNOWN_SLICE_IDS"})
                    continue
                stale = self._stale([name], expect)
                if name in stale:
                    out.append(stale[name])
                    continue
                try:
                    done = damage.mark_damage(job, workspace, name, list(row.get("regions") or ()),
                                              str(row.get("note") or ""))
                except Refused as refusal:
                    out.append({"id": name, "status": "error", **refusal.payload()})
                    continue
                out.append({"id": name, "status": "ok", "regions": list(done.regions),
                            "written": done.written})
                if done.written:
                    written.append(name)
        return WriteResult(written=written, rows=out)

    def add_layer(self, section: object, layer: Layer) -> LayerRef:
        """Leave *layer* on the section so ``look`` can draw it (``layers=``)
        and ``status`` lists it. The layer's file goes under this tool's
        folder (a script's under ``plugins/script/``); the reference joins
        the section's state, so undoing the call (a script's: the write)
        that made it hides it. A layer of the same name on the section
        replaces the earlier one. Only a ``write`` tool leaves layers."""
        from langslice.job.plugin_store import register_layer

        self._require()
        layer.check()
        record_id = self._record(section).id
        job = self._door.job
        if layer.frame == "oriented":  # a raster computed on a section_image array
            held = self._record(section)
            layer = replace(layer, frame="section_px", raster=unorient_raster(
                np.asarray(layer.raster), held.rotation_deg, held.flip))
        with self.writing():
            record = self._door.state.by_id(record_id)
            assert record is not None
            before = job.snapshot()
            ref = register_layer(job.layout, record, self._caller.name, self.call, layer,
                                 persist=job.persist)
            job.commit(before)
        return LayerRef(id=ref["id"], section=record.id, name=ref["name"], kind=ref["kind"],
                        frame=ref["frame"], file=ref["file"])

    def layers(self, section: object) -> list[LayerRef]:
        """The layers the section shows now."""
        record = self._record(section)
        return [LayerRef(id=ref["id"], section=record.id, name=ref["name"], kind=ref["kind"],
                         frame=ref["frame"], file=ref["file"]) for ref in record.layers]


def _as_tool_error(refusal: Refused) -> ToolError:
    return ToolError(refusal.code, str(refusal), **{
        key: value for key, value in refusal.details.items() if key != "message"})


def _plane_frame(atlas: Any, placement: Any) -> dict[str, Any]:
    """The native atlas plane's frame in atlas micrometres (``center_um``,
    ``dx``, ``dy``: the display column and row directions, ``px_um``,
    ``shape``, ``resolution_um``; ``native_to_um``: 3x3 from a native pixel
    centre ``[x, y, 1]`` to its point in atlas micrometres, exactly the plane
    sampled, voxel centres at index times resolution), the plane
    :func:`langslice.core.oblique.sample_oblique_plane` samples."""
    from langslice.core.oblique import build_rotation_matrix, plane_axes

    normal_axis, row_axis, col_axis = plane_axes(atlas, placement.plane)
    rotation = build_rotation_matrix(placement.pitch_deg, placement.yaw_deg,
                                     row_axis=row_axis, col_axis=col_axis)
    res = np.asarray(atlas.resolution, dtype=np.float64)
    shape = tuple(int(v) for v in np.shape(atlas.annotation))
    centre = np.zeros(3)
    centre[normal_axis] = float(placement.position_mm) * 1000.0
    centre[row_axis] = (shape[row_axis] - 1) / 2.0 * res[row_axis]
    centre[col_axis] = (shape[col_axis] - 1) / 2.0 * res[col_axis]
    unit = np.eye(3)
    d_row, d_col = rotation @ unit[row_axis], rotation @ unit[col_axis]
    if placement.plane == "coronal":
        dx, dy = d_col, d_row
        px = [float(res[col_axis]), float(res[row_axis])]
        size = [shape[row_axis], shape[col_axis]]
    else:
        dx, dy = d_row, d_col
        px = [float(res[row_axis]), float(res[col_axis])]
        size = [shape[col_axis], shape[row_axis]]
    from langslice.core.oblique import plane_index_affine

    to_index = plane_index_affine(atlas, float(placement.position_mm), placement.plane,
                                  float(placement.pitch_deg), float(placement.yaw_deg))
    # [x, y, 1] -> [row, col, 1]
    swap = np.array([[0.0, 1.0, 0.0], [1.0, 0.0, 0.0], [0.0, 0.0, 1.0]])
    native_to_um = np.diag(res) @ to_index @ swap
    return {"center_um": centre.tolist(), "dx": dx.tolist(), "dy": dy.tolist(),
            "px_um": px, "shape": size, "resolution_um": res.tolist(),
            "native_to_um": native_to_um.tolist()}


__all__ = [
    "AffineView", "AtlasPlane", "AtlasView", "Caller", "DeformationProposal", "FitFrame",
    "LayerRef", "SectionRaster", "SectionView", "ToolContext",
    "TransformProposal", "WriteResult", "STALE_INPUT",
]
