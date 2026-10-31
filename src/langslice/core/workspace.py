"""The core workspace: one job's sections and atlas, and their caches.

:class:`Workspace` is everything the core operations (renders, atlas sections,
fits, deformations, the image-correction handoff) need that is not stack state:
the few settings the core reads (:class:`ImageSettings`), the image folder,
the atlas, each section's working copy, raw channels and pixel size, and the
render caches. It knows nothing of the agent: no model, no message images,
and nothing of the job's form beyond its :class:`ImageSettings`. The agent
driver wraps it (:class:`langslice.doors.agent.engine.EngineContext` adds the
model and the cache of encoded message images), and every core module takes
a ``Workspace``, so a script can build one and call them without the agent
framework loaded.
"""

from __future__ import annotations

import logging
import os
from collections import OrderedDict
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any

from PIL import Image

from langslice.core.atlas.load import get_position_range_mm, load_atlas
from langslice.core.sections.reading import (
    channel_planes,
    host_preprocess,
    normalize_image,
    read_pixel_size_um,
    read_working_image,
    read_working_pages,
)
from langslice.core.space import Plane, atlas_space_context, slice_axis_ends

logger = logging.getLogger(__name__)

#: A host's channel-blend settings as it gave them (``{"mode": "auto"}``,
#: :func:`~langslice.core.sections.reading.host_preprocess`).
HostPreprocessing = Mapping[str, str | bool | float | Sequence[float]]


def log_progress(message: str) -> None:
    """The default progress sink: the log."""
    logger.info(message)


@dataclass(frozen=True)
class ImageSettings:
    """The settings of a job the core reads, and only those.

    The job makes one from its form and hands it to the workspace; the core
    knows nothing else of the job's settings (its tasks, what is frozen or
    disabled).
    """

    #: The BrainGlobe atlas name.
    atlas: str
    #: The section plane.
    plane: Plane
    #: How the sections are shown and read by the fits: ``auto`` (the
    #: adaptive preprocess) or ``none`` (the file as it is).
    preprocess: str
    #: A host's blend of a multi-page snapshot's channels, which is then each
    #: section's default appearance; None: the file, through *preprocess*.
    host_preprocessing: HostPreprocessing | None
    #: The micrometres per pixel the host gave, as it gave them (one number,
    #: a mapping of filenames to numbers, or None), for the keys of what
    #: depends on them.
    supplied_pixel_size: object
    #: The host's micrometres per pixel for every file, or None.
    stack_pixel_size_um: float | None
    #: The host's micrometres per pixel by filename, over
    #: *stack_pixel_size_um*.
    file_pixel_sizes_um: Mapping[str, float]
    #: The host's name for each page of a multi-page snapshot, or None.
    channel_names: tuple[str, ...] | None
    #: The size level of the pictures the agent is shown (``low``,
    #: ``medium``, ``high`` or ``auto``).
    picture_level: str
    #: The settings a border trace's saved reply depends on, as (name,
    #: value) pairs: a trace made under others is stale.
    trace_settings: tuple[tuple[str, object], ...]
    #: All of the job's settings as one sorted JSON text: what frames shared
    #: across a fork must agree on.
    settings_text: str
    #: What a fit says of a section with no written in-plane transform (the
    #: job words it: for a job whose in-plane task is off, what to supply
    #: instead).
    missing_transform: str


@dataclass
class Workspace:
    """One job's sections and atlas: everything a core operation needs that
    is not stack state."""

    settings: ImageSettings
    image_folder: str
    emit: Callable[[str], None] = log_progress
    #: Atlas accessor, injectable so tests (and offline hosts) can supply one.
    atlas_loader: Callable[[str], Any] = load_atlas
    #: Rendered sections, keyed ``(id, flip, rotation, long_edge, preprocess)``.
    #: The user's files never change during a run, and a correction
    #: or a different size is a different key. Cached images are shared —
    #: callers read them, never mutate them.
    render_cache: dict[tuple[str, bool, int, int, str], Image.Image] = field(
        default_factory=dict, repr=False
    )
    #: Same keys as ``render_cache``: how much that render shrank the file's
    #: pixels, so a known micrometres-per-pixel can follow the image down.
    render_scale: dict[tuple[str, bool, int, int, str], float] = field(
        default_factory=dict, repr=False
    )
    #: Each file's working copy
    #: (:func:`langslice.core.sections.reading.read_working_image`) and how
    #: many file pixels one of its pixels spans. Every render above is drawn
    #: from it, so a whole-slide scan is read once, small.
    source_cache: dict[str, tuple[Image.Image, float]] = field(
        default_factory=dict, repr=False
    )
    #: Each file's raw channels at working size: names and 8-bit planes
    #: (:func:`langslice.core.sections.reading.channel_planes`). Shared: read
    #: only.
    channel_cache: dict[str, tuple[tuple[str, ...], list[Any]]] = field(
        default_factory=dict, repr=False
    )
    #: Each file's raw channels' map from file intensities to their 8-bit
    #: planes, by channel name
    #: (:func:`langslice.core.sections.channels.intensity_ranges`).
    intensity_cache: dict[str, dict[str, Any]] = field(default_factory=dict, repr=False)
    #: Each placed section's linear frame
    #: (:func:`langslice.core.maps.section_maps.section_frame`: file pixels to
    #: atlas micrometres), keyed by everything it depends on, so
    #: ``registration.json`` is rewritten on every write without recomputing
    #: the sections that did not move: section id -> (key, frame), one per
    #: section, least recently used first (bounded by
    #: ``core.maps.section_maps.FRAME_CACHE_SECTIONS``). Shared: read only.
    frame_cache: OrderedDict[str, tuple[tuple[Any, ...], Any]] = field(
        default_factory=OrderedDict, repr=False)
    #: Atlas template planes
    #: (:func:`langslice.core.atlas.planes.atlas_section`), keyed ``(position,
    #: plane, pitch, yaw, frame)``, least recently used first (bounded by
    #: ``core.atlas.planes.PLANE_CACHE_SIZE``): the opening and the pictures
    #: size and draw the same plane more than once. Shared: read only.
    plane_cache: OrderedDict[tuple[float, str, float, float, bool], Image.Image] = field(
        default_factory=OrderedDict, repr=False)
    #: Section images at the working copy's size, uint8 (raw planes, made
    #: images' files, the opening image): ``(section id, what)`` -> array,
    #: least recently used first (bounded by
    #: ``core.sections.images.IMAGE_CACHE_SIZE``). Shared: read only.
    image_cache: OrderedDict[tuple[str, str], Any] = field(
        default_factory=OrderedDict, repr=False)
    #: The job folder whose files the made section images are read from (set by
    #: the job when it opens on this workspace; None: no job, and only the raw
    #: channels and the opening image can be drawn).
    job_folder: str | None = None
    _atlas: Any = field(default=None, repr=False)
    _range: tuple[float, float] | None = field(default=None, repr=False)
    _pixel_sizes: dict[str, float | None] = field(default_factory=dict, repr=False)

    def progress(self, message: str) -> None:
        self.emit(message)
        logger.info(message)

    def image_path(self, slice_id: str) -> str:
        """Absolute path of a section image. Ids are folder-relative names."""
        return os.path.join(self.image_folder, slice_id)

    def working_source(self, slice_id: str) -> tuple[Image.Image, float]:
        """``(working copy, file pixels per working pixel)`` of one section.

        Shared: read it, never mutate it in place.
        """
        cached = self.source_cache.get(slice_id)
        if cached is None:
            settings = self.settings.host_preprocessing
            if settings is not None:
                # A host's snapshot, one page per channel: the default
                # appearance is the automatic blend of the pages, drawn at
                # working size. The pages stay readable as channels.
                pages, factor = read_working_pages(self.image_path(slice_id))
                blended = host_preprocess(pages, settings).convert("L")
                cached = (normalize_image(blended), factor)
            else:
                cached = read_working_image(self.image_path(slice_id))
            self.source_cache[slice_id] = cached
        return cached

    def section_channels(self, slice_id: str) -> tuple[tuple[str, ...], list[Any]]:
        """``(names, raw 8-bit planes)`` of one section at working size.

        A single colour file is red/green/blue (``gray`` when the three are
        equal); a multi-page file is one plane per page, named by the host's
        ``inputs["channel_names"]`` when it gave one per page, else ch1, ch2...
        """
        cached = self.channel_cache.get(slice_id)
        if cached is None:
            pages, _factor = read_working_pages(self.image_path(slice_id))
            names = self.settings.channel_names
            listed = None if names is None else list(names)
            cached = channel_planes(pages, listed)
            self.channel_cache[slice_id] = cached
        return cached

    @property
    def atlas(self) -> Any:
        if self._atlas is None:
            self._atlas = self.atlas_loader(self.settings.atlas)
        return self._atlas

    @property
    def brain_scale(self) -> float:
        """The atlas brain's size next to the adult mouse's
        (:func:`langslice.core.atlas.brain_size.brain_scale`): what the
        deformable
        fits and the positioning ruler multiply their distances by."""
        from langslice.core.atlas.brain_size import brain_scale

        return brain_scale(self.atlas)

    @property
    def position_range(self) -> tuple[float, float]:
        if self._range is None:
            self._range = get_position_range_mm(
                self.atlas, plane=self.settings.plane
            )
        return self._range

    @property
    def axis_ends(self) -> tuple[str, str]:
        """(low, high) anatomical ends of this run's slicing axis."""
        return slice_axis_ends(
            atlas_space_context(self.atlas), self.settings.plane
        )

    def calibration(self, slice_id: str) -> tuple[float | None, str]:
        """``(micrometres per pixel of the FILE, source)`` for one section.

        The host's pixel size wins when it is given — that is what
        ``--pixel-size-um`` is for — for the whole stack or per file
        (``ImageSettings.file_pixel_sizes_um`` over its
        ``stack_pixel_size_um``); otherwise the file's own tags answer
        (:func:`langslice.core.sections.reading.read_pixel_size_um`). ``(None,
        "")`` when nothing says; the caller then reports the section as
        uncalibrated.
        """
        host = self.settings.file_pixel_sizes_um.get(
            slice_id, self.settings.stack_pixel_size_um)
        if host is not None:
            return host, "host"
        if slice_id not in self._pixel_sizes:
            self._pixel_sizes[slice_id] = read_pixel_size_um(self.image_path(slice_id))
        from_file = self._pixel_sizes[slice_id]
        return (from_file, "file") if from_file else (None, "")

    @property
    def species(self) -> str:
        metadata = getattr(self.atlas, "metadata", None)
        return str((metadata or {}).get("species", "mouse"))
