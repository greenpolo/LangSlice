"""Pictures of a job's current registration, drawn for the agent viewer.

The agent viewer shows each section under its current registration with the
atlas borders (the focus panels), the sections and the atlas as thumbnails
along the cutting axis (the positioning strip) and the strip's facts.
:class:`JobRenderer` draws them for one watched job with the core's
renderers, from the job folder as the watcher reads it, WITHOUT opening the
job: the spec and the workspace come through the doors as the agent CLI
reads a job (``read_spec``, ``context``), the applied deformations from the
sections' folders (the core's ``RecordStore``) and the stack from the
watcher. Nothing is written to the job folder; the atlas's own disk cache
is the only file the core may write, as for any reader.

Every picture is a JPEG (the core's ``encode_jpeg``, the one every door
sends) with its caption band cut off: the page labels it. The encoded
pictures are kept in a bounded LRU per job, keyed by kind, section or
position, edge and :func:`render_key`, so a picture is drawn again only
when what it shows changed. :class:`RenderPool` runs every job's drawing
on one pool of two threads; a job draws one picture at a time (a
workspace's caches are not made for concurrent writers), and concurrent
requests for one picture share one drawing.
"""

from __future__ import annotations

import hashlib
import json
import logging
import math
import threading
from collections import OrderedDict
from collections.abc import Callable, Mapping
from concurrent.futures import Executor, Future, ThreadPoolExecutor
from dataclasses import dataclass
from functools import partial
from pathlib import Path
from typing import TypeVar

from PIL import Image

from langslice.core.fits.deformation import RecordStore
from langslice.core.pictures.captions import view_angles_label
from langslice.core.pictures.jpeg import encode_jpeg
from langslice.core.pictures.look import (
    snapshot,
)
from langslice.core.refusal import Refused
from langslice.core.stack.section import Section
from langslice.core.stack.stack import Stack
from langslice.core.workspace import Workspace
from langslice.doors.jobs import JobContext, NoJob, context, read_spec
from langslice.hosts.gui.calm import draw_calm_overlay, draw_calm_thumb
from langslice.hosts.gui.drawing import draw_atlas as _atlas_picture
from langslice.hosts.gui.drawing import draw_overlay as _overlay_picture
from langslice.hosts.gui.drawing import draw_thumb as _thumb_picture
from langslice.hosts.gui.space import (
    AtlasShell,
    ShellError,
    Space,
    atlas_shell,
    job_space,
)
from langslice.job.folder.layout import JOB_FILE

logger = logging.getLogger(__name__)

MIN_EDGE_PX = 64
MAX_EDGE_PX = 2048
EDGE_STEP_PX = 64
DEFAULT_EDGE_PX = 512
CACHE_ENTRIES = 96
WAIT_S = 60.0
WORKERS = 2
KEY_DIGITS = 12
NOMINAL_STEP_MM = 0.5
ROUND_STEPS_MM = (0.1, 0.2, 0.25, 0.5, 1.0)
MIN_THUMBNAILS = 12
MAX_THUMBNAILS = 32
KEPT_SOURCES = 8
KEPT_PLANES = 16
KEPT_RENDERS = 64
KEPT_JOBS = 3
NO_STATE = (
    "This job has no state this LangSlice reads yet, so there is nothing "
    "to draw."
)
ATLAS_FAILED = "The atlas {atlas} could not be loaded: {reason}"
DRAWING_FAILED = "The picture could not be drawn: {reason}"
BUSY = (
    "The picture is still being drawn; it will be ready on the next request."
)

_K = TypeVar("_K")
_Key = tuple[str, str, int, str]


class RenderError(Exception):
    """A picture that cannot be drawn; the message says why."""


class UnknownSection(LookupError):
    """A section the job does not hold, or an atlas position past the strip."""


class RenderBusy(TimeoutError):
    """A picture not drawn within :data:`WAIT_S`."""


class _Superseded(Exception):
    """A queued picture of a section whose registration has changed since."""


@dataclass(frozen=True)
class Axis:
    """The cutting axis the strip is drawn along."""

    low: str
    high: str
    min_mm: float
    max_mm: float


@dataclass(frozen=True)
class AtlasThumb:
    """One atlas thumbnail of the strip: its index and position."""

    k: int
    position_mm: float


@dataclass(frozen=True)
class StripSection:
    """One section on the strip, in stack order."""

    id: str
    order: int
    position_mm: float
    position_source: str
    render_key: str
    thumb_key: str


@dataclass(frozen=True)
class Strip:
    """The positioning strip's facts (``GET jobs/<n>/strip``)."""

    axis: Axis
    step_mm: float
    atlas: tuple[AtlasThumb, ...]
    angles: str
    sections: tuple[StripSection, ...]


@dataclass(frozen=True)
class _Opened:
    """A job read for drawing: its workspace, records and strip axis."""

    workspace: JobContext
    store: RecordStore
    axis: Axis
    step_mm: float
    positions_mm: tuple[float, ...]
    stamp: tuple[int, int] | None


def edge_px(requested: int) -> int:
    """Round a requested long edge to the sizes pictures are drawn at.

    Parameters
    ----------
    requested
        The long edge the page asks for, in pixels.

    Returns
    -------
    int
        *requested* rounded up to a multiple of :data:`EDGE_STEP_PX`,
        between :data:`MIN_EDGE_PX` and :data:`MAX_EDGE_PX`.
    """
    steps = math.ceil(max(requested, 1) / EDGE_STEP_PX)
    return min(MAX_EDGE_PX, max(MIN_EDGE_PX, steps * EDGE_STEP_PX))


def render_key(state: Stack, section_id: str) -> str:
    """Digest everything a picture of one section depends on.

    Parameters
    ----------
    state
        The job's stack as the viewer reads it.
    section_id
        The section's filename.

    Returns
    -------
    str
        :data:`KEY_DIGITS` hex digits of the section's look snapshot (the
        section fields a picture is drawn from, its made images'
        digests included): it changes exactly when what an overlay of the
        section depends on changes.
    """
    text = json.dumps(snapshot(state, [section_id]), sort_keys=True)
    return hashlib.sha256(text.encode()).hexdigest()[:KEY_DIGITS]


def thumb_key(state: Stack, section_id: str) -> str:
    """Digest what a section's thumbnail depends on.

    Parameters
    ----------
    state
        The job's stack as the viewer reads it.
    section_id
        The section's filename.

    Returns
    -------
    str
        :data:`KEY_DIGITS` hex digits of the section's flip, quarter turn and
        made images: a new position, transform or deformation leaves it
        unchanged, so the strip does not draw the thumbnail again.
    """
    section = _section(state, section_id)
    images = sorted(section.images.items())
    facts = {
        "flip": section.flip,
        "rotation_deg": section.rotation_deg,
        "images": {name: ref.digest for name, ref in images},
    }
    text = json.dumps(facts, sort_keys=True, default=str)
    return hashlib.sha256(text.encode()).hexdigest()[:KEY_DIGITS]


def atlas_positions(
    min_mm: float, max_mm: float, step_mm: float
) -> tuple[float, ...]:
    """Place the strip's atlas thumbnails along the axis.

    Parameters
    ----------
    min_mm, max_mm
        The atlas's position range along the cutting axis.
    step_mm
        The spacing between thumbnails.

    Returns
    -------
    tuple of float
        Every multiple of *step_mm* strictly inside the range (its ends
        cut through the edge of the volume, where there is no brain).
    """
    first = math.floor(min_mm / step_mm + 1e-6) + 1
    last = math.ceil(max_mm / step_mm - 1e-6) - 1
    return tuple(round(k * step_mm, 4) for k in range(first, last + 1))


def strip_step(min_mm: float, max_mm: float, brain_scale: float) -> float:
    """Choose the spacing of the strip's atlas thumbnails.

    Parameters
    ----------
    min_mm, max_mm
        The atlas's position range along the cutting axis.
    brain_scale
        The atlas brain's size next to the adult mouse's.

    Returns
    -------
    float
        The round spacing (:data:`ROUND_STEPS_MM`) nearest
        :data:`NOMINAL_STEP_MM` times *brain_scale* among those that put
        :data:`MIN_THUMBNAILS` to :data:`MAX_THUMBNAILS` thumbnails on the
        axis, else the one closest to that count.
    """
    nominal = NOMINAL_STEP_MM * brain_scale
    return min(
        ROUND_STEPS_MM,
        key=lambda step: _step_rank(step, min_mm, max_mm, nominal),
    )


def _step_rank(
    step_mm: float, min_mm: float, max_mm: float, nominal_mm: float
) -> tuple[int, float]:
    count = len(atlas_positions(min_mm, max_mm, step_mm))
    outside = max(MIN_THUMBNAILS - count, count - MAX_THUMBNAILS, 0)
    return outside, abs(math.log(step_mm / nominal_mm))


class JobRenderer:
    """The pictures of one watched job (see the module text).

    Parameters
    ----------
    folder
        The job folder the pictures are drawn from.
    stack
        Returns the job's current stack as the watcher reads it, None
        while the folder holds no state this LangSlice reads.
    executor
        The pool every picture is drawn on, shared by the server's jobs.
    atlas_loader
        Loads an atlas by name; None is BrainGlobe's
        (``core.atlas.load.load_atlas``).
    """

    def __init__(
        self,
        folder: Path,
        stack: Callable[[], Stack | None],
        executor: Executor,
        *,
        atlas_loader: Callable[[str], object] | None = None,
    ) -> None:
        self.folder = folder
        self._stack = stack
        self._executor = executor
        self._atlas_loader = atlas_loader
        self._opened: _Opened | None = None
        self._opening = threading.Lock()
        self._drawing = threading.Lock()
        self._lock = threading.Lock()
        self._pictures: OrderedDict[_Key, bytes] = OrderedDict()
        self._flying: dict[_Key, Future[bytes]] = {}
        self._shell: tuple[_Opened, AtlasShell] | None = None

    def strip(self) -> Strip:
        """Describe the positioning strip of the job as it stands.

        Returns
        -------
        Strip
            The axis, the atlas thumbnails' positions and the view angles
            they are drawn at, and every section in stack order with its
            position and render key (none while the job has no state).

        Raises
        ------
        RenderError
            When the job's spec, images or atlas cannot be read.
        """
        opened = self._open()
        state = self._stack()
        thumbs = tuple(
            AtlasThumb(k, mm) for k, mm in enumerate(opened.positions_mm)
        )
        if state is None:
            return Strip(opened.axis, opened.step_mm, thumbs, "", ())
        sections = tuple(
            StripSection(
                section.id,
                order,
                _rounded_mm(section.position_mm),
                section.position_source,
                render_key(state, section.id),
                thumb_key(state, section.id),
            )
            for order, section in enumerate(state.in_order(), 1)
        )
        angles = state.view_angles
        words = view_angles_label(angles, median=state.drawn_at_median(angles))
        return Strip(
            opened.axis, opened.step_mm, thumbs, words.strip(), sections
        )

    def overlay(
        self, section_id: str, edge: int, *, calm: bool = False
    ) -> bytes:
        """Draw a section under its current registration.

        Parameters
        ----------
        section_id
            The section's filename.
        edge
            The picture's long edge in pixels (:func:`edge_px`).
        calm
            The Simple view's quiet picture (:mod:`.calm`): the main stain
            alone, the borders soft; else the raw channels in colour.

        Returns
        -------
        bytes
            A JPEG: the section under its stored transform and applied
            deformation with the atlas borders (``look``'s ``overlay``).

        Raises
        ------
        UnknownSection
            When the job holds no section *section_id*.
        RenderError
            When the picture cannot be drawn, with the reason.
        RenderBusy
            When it was not drawn within :data:`WAIT_S`.
        """
        state = self._state()
        section = _section(state, section_id)
        opened = self._open()
        kind = "calm-overlay" if calm else "overlay"
        key = (kind, section_id, edge, render_key(state, section_id))
        drawer = draw_calm_overlay if calm else _overlay_picture
        draw = partial(drawer, opened, state, section, edge)
        still = self._still(render_key, section_id, key[3])
        return self._picture(key, opened, draw, current=still)

    def thumb(
        self, section_id: str, edge: int, *, calm: bool = False
    ) -> bytes:
        """Draw a section's thumbnail: its raw channels, oriented.

        Parameters
        ----------
        section_id
            The section's filename.
        edge
            The thumbnail's long edge in pixels, at most the working
            copy's (nothing is upsampled).
        calm
            The Simple view's quiet picture: the main stain alone.

        Returns
        -------
        bytes
            A JPEG of the section alone, turned and flipped as it is held.

        Raises
        ------
        UnknownSection
            When the job holds no section *section_id*.
        RenderError
            When the picture cannot be drawn, with the reason.
        RenderBusy
            When it was not drawn within :data:`WAIT_S`.
        """
        state = self._state()
        section = _section(state, section_id)
        opened = self._open()
        kind = "calm-thumb" if calm else "thumb"
        key = (kind, section_id, edge, thumb_key(state, section_id))
        drawer = draw_calm_thumb if calm else _thumb_picture
        draw = partial(drawer, opened, section, edge)
        still = self._still(thumb_key, section_id, key[3])
        return self._picture(key, opened, draw, current=still)

    def atlas_thumb(self, k: int, edge: int) -> bytes:
        """Draw the strip's *k*-th atlas thumbnail.

        Parameters
        ----------
        k
            The thumbnail's index in :meth:`strip`'s ``atlas``.
        edge
            The thumbnail's long edge in pixels.

        Returns
        -------
        bytes
            A JPEG of the atlas's default image (its Nissl) at that
            position and the stack's view angles, framed to its anatomy,
            without borders.

        Raises
        ------
        UnknownSection
            When the strip has no *k*-th position.
        RenderError
            When the picture cannot be drawn, with the reason.
        RenderBusy
            When it was not drawn within :data:`WAIT_S`.
        """
        state = self._state()
        opened = self._open()
        if not 0 <= k < len(opened.positions_mm):
            raise UnknownSection(f"The strip has no atlas thumbnail {k}.")
        position_mm = opened.positions_mm[k]
        pitch, yaw = state.view_angles
        key = ("atlas", f"{position_mm:.4f}", edge, f"{pitch:.4f} {yaw:.4f}")
        draw = partial(_atlas_picture, opened, state, position_mm, edge)
        return self._picture(key, opened, draw)

    def shell(self) -> AtlasShell:
        """Read the job's atlas brain mesh for the 3D view, once per atlas.

        Returns
        -------
        AtlasShell
            The atlas's root mesh.

        Raises
        ------
        RenderError
            When the job's spec, images or atlas cannot be read, or the
            atlas has no brain mesh.
        """
        opened = self._open()
        kept = self._shell
        if kept is not None and kept[0] is opened:
            return kept[1]
        with self._drawing:
            try:
                shell = atlas_shell(opened.workspace)
            except ShellError as exc:
                raise RenderError(str(exc)) from exc
        self._shell = (opened, shell)
        return shell

    def space(self) -> Space:
        """Place the job's sections in the atlas for the 3D view.

        Returns
        -------
        Space
            Each section's thumbnail corners in atlas micrometres (None
            while unplaced) and the section thickness.

        Raises
        ------
        RenderError
            When the job has no state, or its spec, images or atlas cannot
            be read.
        """
        opened = self._open()
        state = self._state()
        thickness_um = opened.workspace.spec.position.thickness_um
        with self._drawing:
            return job_space(
                opened.workspace, state, thickness_um=thickness_um
            )

    def release(self) -> None:
        """Let go of the workspace and pictures until the next drawing."""
        with self._opening:
            self._opened = None
            self._shell = None
        with self._lock:
            self._pictures.clear()

    def _state(self) -> Stack:
        state = self._stack()
        if state is None:
            raise RenderError(NO_STATE)
        return state

    def _open(self) -> _Opened:
        """Read the job for drawing again when its ``job.json`` changed."""
        stamp = _stamp(self.folder / JOB_FILE)
        with self._opening:
            opened = self._opened
            if opened is None or opened.stamp != stamp:
                opened = _open_job(self.folder, stamp, self._atlas_loader)
                self._opened = opened
                with self._lock:
                    self._pictures.clear()
            return opened

    def _still(
        self, keyed: Callable[[Stack, str], str], section_id: str, digest: str
    ) -> Callable[[], bool]:
        """Make the check that a section's picture is still the one to draw."""
        return partial(_is_current, self._stack, keyed, section_id, digest)

    def _picture(
        self,
        key: _Key,
        opened: _Opened,
        draw: Callable[[], Image.Image],
        *,
        current: Callable[[], bool] | None = None,
    ) -> bytes:
        """Return the picture *key* from the cache, else drawn.

        One drawing per key, however many requests wait for it. A picture
        whose section has changed by the time its turn comes is not drawn
        (*current* says so): the request is answered busy, and asked again it
        draws the section as it is then. A fast run's queue so holds only
        the pictures still worth drawing.
        """
        with self._lock:
            ready = self._pictures.get(key)
            if ready is not None:
                self._pictures.move_to_end(key)
                return ready
            future = self._flying.get(key)
            if future is None:
                make = partial(self._make, key, opened, draw, current=current)
                future = self._executor.submit(make)
                self._flying[key] = future
        try:
            return future.result(timeout=WAIT_S)
        except (TimeoutError, _Superseded) as exc:
            raise RenderBusy(BUSY) from exc

    def _make(
        self,
        key: _Key,
        opened: _Opened,
        draw: Callable[[], Image.Image],
        *,
        current: Callable[[], bool] | None = None,
    ) -> bytes:
        """Draw and encode one picture on the pool, then keep it."""
        data: bytes | None = None
        try:
            with self._drawing:
                if current is not None and not current():
                    raise _Superseded(key)
                try:
                    picture = _drawn_safely(draw, key)
                finally:
                    _trim(opened.workspace)
            data = encode_jpeg(picture)
            return data
        finally:
            with self._lock:
                self._flying.pop(key, None)
                if data is not None and self._opened is opened:
                    self._pictures[key] = data
                while len(self._pictures) > CACHE_ENTRIES:
                    self._pictures.popitem(last=False)


class RenderPool:
    """The drawing of one server: one thread pool, one renderer per job.

    Parameters
    ----------
    atlas_loader
        Loads an atlas by name for every job; None is BrainGlobe's.
    workers
        How many pictures are drawn at once, over all jobs.
    """

    def __init__(
        self,
        *,
        atlas_loader: Callable[[str], object] | None = None,
        workers: int = WORKERS,
    ) -> None:
        self._executor = ThreadPoolExecutor(
            max_workers=workers, thread_name_prefix="langslice-render"
        )
        self._atlas_loader = atlas_loader
        self._renderers: OrderedDict[Path, JobRenderer] = OrderedDict()
        self._lock = threading.Lock()

    def renderer(
        self, folder: Path, stack: Callable[[], Stack | None]
    ) -> JobRenderer:
        """Return the renderer of one job, made on first use.

        Parameters
        ----------
        folder
            The job folder, which names the job.
        stack
            Returns the job's current stack (the watcher's), used when the
            renderer is made.

        Returns
        -------
        JobRenderer
            The same renderer for every call with *folder*. The renderers
            of the jobs used longest ago, past :data:`KEPT_JOBS`, let go of
            their workspace and pictures.
        """
        with self._lock:
            known = self._renderers.get(folder)
            if known is None:
                known = JobRenderer(
                    folder,
                    stack,
                    self._executor,
                    atlas_loader=self._atlas_loader,
                )
                self._renderers[folder] = known
            self._renderers.move_to_end(folder)
            idle = list(self._renderers.values())[:-KEPT_JOBS]
        for renderer in idle:
            renderer.release()
        return known

    def close(self) -> None:
        """Stop drawing: queued pictures are dropped, none is waited for."""
        self._executor.shutdown(wait=False, cancel_futures=True)


def _open_job(
    folder: Path,
    stamp: tuple[int, int] | None,
    atlas_loader: Callable[[str], object] | None,
) -> _Opened:
    """Read *folder*'s job for drawing, as the agent CLI reads it.

    The job is not opened, so nothing is written; :class:`RenderError`
    with the reason when its spec, images or atlas cannot be read.
    """
    try:
        spec = read_spec(folder)
        workspace = context(spec, folder, atlas_loader=atlas_loader)
    except (NoJob, OSError, ValueError) as exc:
        raise RenderError(str(exc)) from exc
    layout = workspace.layout
    store = RecordStore(
        root=layout.sections_dir, folder_of=layout.deformable_dir
    )
    try:
        min_mm, max_mm = workspace.position_range
        low, high = workspace.axis_ends
        step_mm = strip_step(min_mm, max_mm, workspace.brain_scale)
    except Exception as exc:
        raise RenderError(
            ATLAS_FAILED.format(atlas=spec.atlas, reason=exc)
        ) from exc
    axis = Axis(low, high, round(min_mm, 4), round(max_mm, 4))
    positions = atlas_positions(min_mm, max_mm, step_mm)
    return _Opened(workspace, store, axis, step_mm, positions, stamp)


def _drawn_safely(draw: Callable[[], Image.Image], key: _Key) -> Image.Image:
    """*draw*'s picture; a refusal or a failure as :class:`RenderError`."""
    try:
        return draw()
    except Refused as refusal:
        raise RenderError(str(refusal)) from refusal
    except Exception as exc:
        logger.warning("The viewer could not draw %s", key, exc_info=True)
        raise RenderError(DRAWING_FAILED.format(reason=exc)) from exc


def _is_current(
    stack: Callable[[], Stack | None],
    keyed: Callable[[Stack, str], str],
    section_id: str,
    digest: str,
) -> bool:
    """Tell whether a section's picture keyed *digest* is still current.

    *stack* reads the job's stack now; *keyed* digests what the picture
    depends on (:func:`render_key`, :func:`thumb_key`).
    """
    state = stack()
    try:
        return state is not None and keyed(state, section_id) == digest
    except UnknownSection:
        return False


def _section(state: Stack, section_id: str) -> Section:
    section = state.by_id(section_id)
    if section is None:
        raise UnknownSection(f"The job has no section {section_id!r}.")
    return section


def _rounded_mm(position_mm: float) -> float:
    return round(position_mm, 3)


def _stamp(path: Path) -> tuple[int, int] | None:
    try:
        info = path.stat()
    except OSError:
        return None
    return info.st_size, info.st_mtime_ns


def _trim(workspace: Workspace) -> None:
    """Keep the workspace's per-section caches short, oldest out.

    Called with the job's drawing lock held, so no drawing reads them.
    """
    for name in _oldest(workspace.source_cache, KEPT_SOURCES):
        del workspace.source_cache[name]
    for name in _oldest(workspace.channel_cache, KEPT_SOURCES):
        del workspace.channel_cache[name]
    for plane in _oldest(workspace.image_cache, KEPT_PLANES):
        del workspace.image_cache[plane]
    for render in _oldest(workspace.render_cache, KEPT_RENDERS):
        del workspace.render_cache[render]
    for render in _oldest(workspace.render_scale, KEPT_RENDERS):
        del workspace.render_scale[render]


def _oldest(cache: Mapping[_K, object], keep: int) -> list[_K]:
    """List the keys of *cache* past its newest *keep*, oldest first."""
    return list(cache)[: max(0, len(cache) - keep)]
