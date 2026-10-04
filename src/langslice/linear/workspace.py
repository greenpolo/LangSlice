"""The core workspace: one job's sections and atlas, and their caches.

:class:`Workspace` is everything the core operations (renders, atlas
sections, fits, deformations, the image-correction handoff) need that is not
stack state: the job spec, the image folder, the atlas and ABBA's Nissl
volume, each section's working copy, raw channels and pixel size, and the
render caches. It knows nothing of the agent: no model, no message images.
The agent driver wraps it (:class:`langslice.linear.engine.EngineContext`
adds the model and the cache of encoded message images), and every core
module takes a ``Workspace``, so a script can build one and call them
without the agent framework loaded.
"""

from __future__ import annotations

import logging
import os
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any, cast

from PIL import Image

from langslice.atlas.core import get_position_range_mm, load_atlas
from langslice.image_prep import (
    channel_planes,
    host_preprocess,
    normalize_image,
    read_pixel_size_um,
    read_working_image,
    read_working_pages,
)
from langslice.linear.spec import JobSpec
from langslice.space import Plane, atlas_space_context, slice_axis_ends

logger = logging.getLogger(__name__)


def log_progress(message: str) -> None:
    """The default progress sink: the log."""
    logger.info(message)


@dataclass
class Workspace:
    """One job's sections and atlas: everything a core operation needs that
    is not stack state."""

    spec: JobSpec
    image_folder: str
    emit: Callable[[str], None] = log_progress
    #: Atlas accessor, injectable so tests (and offline hosts) can supply one.
    atlas_loader: Callable[[str], Any] = load_atlas
    #: Rendered sections, keyed ``(id, flip, rotation, long_edge, preprocess,
    #: frame)``. The user's files never change during a run, and a correction
    #: or a different size is a different key. Cached images are shared —
    #: callers read them, never mutate them.
    render_cache: dict[tuple[str, bool, int, int, str, bool], Image.Image] = field(
        default_factory=dict, repr=False
    )
    #: Same keys as ``render_cache``: how much that render shrank the file's
    #: pixels, so a known micrometres-per-pixel can follow the image down.
    render_scale: dict[tuple[str, bool, int, int, str, bool], float] = field(
        default_factory=dict, repr=False
    )
    #: Each file's working copy (:func:`langslice.image_prep.read_working_image`)
    #: and how many file pixels one of its pixels spans. Every render above
    #: is drawn from it, so a whole-slide scan is read once, small.
    source_cache: dict[str, tuple[Image.Image, float]] = field(
        default_factory=dict, repr=False
    )
    #: Each file's raw channels at working size: names and 8-bit planes
    #: (:func:`langslice.image_prep.channel_planes`). Shared: read only.
    channel_cache: dict[str, tuple[tuple[str, ...], list[Any]]] = field(
        default_factory=dict, repr=False
    )
    _atlas: Any = field(default=None, repr=False)
    #: ABBA's cached Allen atlas when it matches this run's atlas (the
    #: ``nissl`` atlas image); looked up once.
    _abba: Any = field(default=None, repr=False)
    _abba_checked: bool = field(default=False, repr=False)
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
            settings = self.spec.host_preprocessing
            if settings is not None:
                # A host's snapshot, one page per channel: the default
                # appearance is the host's blend of the pages (what
                # ``preprocess.preview`` shows), drawn at working size. The
                # pages stay readable as channels.
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
            names = (self.spec.inputs or {}).get("channel_names")
            cached = channel_planes(pages, list(names) if isinstance(names, list) else None)
            self.channel_cache[slice_id] = cached
        return cached

    @property
    def abba_atlas(self) -> Any:
        """ABBA's cached Allen atlas when present and matching, else None."""
        if not self._abba_checked:
            self._abba_checked = True
            try:
                from langslice.deformable.abba_atlas import AbbaAtlas

                found = AbbaAtlas.find()
                self._abba = found if found is not None and found.compatible(self.atlas) else None
            except Exception:  # an unreadable cache or a non-BrainGlobe atlas: no Nissl
                logger.debug("ABBA atlas lookup failed", exc_info=True)
                self._abba = None
        return self._abba

    @property
    def atlas(self) -> Any:
        if self._atlas is None:
            self._atlas = self.atlas_loader(self.spec.atlas)
        return self._atlas

    @property
    def position_range(self) -> tuple[float, float]:
        if self._range is None:
            self._range = get_position_range_mm(
                self.atlas, plane=cast(Plane, self.spec.plane)
            )
        return self._range

    @property
    def axis_ends(self) -> tuple[str, str]:
        """(low, high) anatomical ends of this run's slicing axis."""
        return slice_axis_ends(
            atlas_space_context(self.atlas), cast(Plane, self.spec.plane)
        )

    def calibration(self, slice_id: str) -> tuple[float | None, str]:
        """``(micrometres per pixel of the FILE, source)`` for one section.

        The host's ``inputs["pixel_size_um"]`` wins when it is given — that is
        what ``--pixel-size-um`` is for — otherwise the file's own tags answer
        (:func:`langslice.image_prep.read_pixel_size_um`). ``(None, "")`` when
        nothing says; the caller estimates and says that it estimated.
        """
        host = (self.spec.inputs or {}).get("pixel_size_um")
        if host:
            try:
                value = float(host)
            except (TypeError, ValueError):
                value = 0.0
            if value > 0:
                return value, "host"
        if slice_id not in self._pixel_sizes:
            self._pixel_sizes[slice_id] = read_pixel_size_um(self.image_path(slice_id))
        from_file = self._pixel_sizes[slice_id]
        return (from_file, "file") if from_file else (None, "")

    @property
    def species(self) -> str:
        metadata = getattr(self.atlas, "metadata", None)
        return str((metadata or {}).get("species", "mouse"))
