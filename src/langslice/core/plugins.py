"""Tool plugins, the core's part: the data types.

A tool plugin adds tools to a job's toolbox at run time
(``docs/developers/plugins.md``). This module holds what every layer must
know about one: the declaration of a tool (:class:`ToolSpec`), what its
handler returns (:class:`ToolResult`), the layer a tool leaves on a section
(:class:`Layer`) and the one error a tool raises (:class:`ToolError`). The
checks a declaration must pass and the registry of loaded plugins are the
job's; nothing here imports a module by name.

The public import path of the types is ``langslice.plugin_api``.
"""

from __future__ import annotations

import re
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any

import numpy as np

from langslice.core.refusal import RefusalCode, Refused
from langslice.core.tool_kinds import Group, Kind

#: The plugin API's version. A plugin that names another is refused.
PLUGIN_API_VERSION = 1

#: A tool name: a Python identifier, so a script can call it as a method.
NAME_PATTERN = re.compile(r"[A-Za-z_][A-Za-z0-9_]*\Z")
#: What a script's own writes through ``job.context`` are recorded as (a
#: transform's ``kind``, a deformation's ``method``, the maker of its layers
#: and pictures): no plugin tool takes the name.
SCRIPT_CALLER = "script"

#: The kinds of layer.
LAYER_KINDS: tuple[str, ...] = ("points", "lines", "labels", "scalar", "picture")
#: The frames each kind may have (the first is the default).
KIND_FRAMES: dict[str, tuple[str, ...]] = {
    "points": ("section_px", "atlas_um"),
    "lines": ("section_px", "atlas_um"),
    "labels": ("section_px", "oriented"),
    "scalar": ("section_px", "oriented"),
    "picture": ("none",),
}

#: Most characters of a layer name.
MAX_LAYER_NAME = 48


class ToolError(Refused):
    """Raised by a plugin handler to refuse a call: nothing is written for it.

    The plugin's name for :class:`~langslice.core.refusal.Refused`: ``code``
    is the machine-readable reason (a ``RefusalCode`` member, or a code of
    the plugin's own), ``message`` says why (the code when it is empty) and
    ``details`` are plain facts; the reply is ``{"status": "error", "error":
    code, "message": ..., **details}``.
    """

    def __init__(self, code: str, message: str = "", **details: Any) -> None:
        super().__init__(str(code), str(message or code), **details)
        self.details = details


class PluginError(ValueError):
    """A plugin that cannot be loaded or a tool declaration that fails the
    checks: the message is one line saying why."""


@dataclass(frozen=True, eq=False)
class ToolSpec:
    """One tool a plugin adds (``docs/developers/plugins.md``).

    ``stub`` is a function whose signature is the tool's arguments and whose
    docstring is its description, written like the stubs of the native tools'
    declarations (needs marks allowed); ``handler`` is called as ``handler(ctx,
    **arguments)`` with a :class:`langslice.doors.plugins.ToolContext` and
    returns a :class:`ToolResult`. A ``write`` tool writes section properties
    only through the context, and only those named in ``writes``.
    """

    name: str
    stub: Callable[..., Any]
    handler: Callable[..., ToolResult]
    kind: Kind
    group: Group
    #: ``position`` | ``transform`` | ``nonlinear`` | None: the tool exists only
    #: while that task is on; ``angles``: while the agent may write the cutting
    #: angles (as ``position_sections``).
    task: str | None = None
    #: The section properties the tool may write (position, angles,
    #: orientation, affine, deformation).
    writes: tuple[str, ...] = ()
    #: Computes outside the job's write lock and applies under it.
    long: bool = False
    #: The most one call takes, by what it counts (``sections``: the length of
    #: the ``sections`` argument).
    limits: Mapping[str, int] = field(default_factory=dict)
    #: What the tool needs while it is on (groups of names; as ``ToolDef.needs``).
    needs: tuple[tuple[str, ...], ...] = ()
    #: The plugin's own version, recorded in the job's provenance.
    version: str = ""
    #: The plugin API version the tool was written for.
    api_version: int = PLUGIN_API_VERSION
    #: Anything the plugin wants kept with the tool in the job's provenance
    #: (JSON values), e.g. the text of a tool description it was given.
    meta: Mapping[str, Any] = field(default_factory=dict)
    #: The atlases the tool is offered on: shell-style patterns for the job's
    #: atlas name (``("allen_mouse_*um", "ccfv3augmented_mouse_25um")``); empty
    #: means every atlas. On any other atlas the tool does not exist (it is off
    #: as a disabled tool is: not declared, not listed, named in no text).
    atlases: tuple[str, ...] = ()
    #: The section planes the tool is offered on (``coronal``, ``sagittal``,
    #: ``horizontal``); empty means every plane.
    planes: tuple[str, ...] = ()


@dataclass
class ToolResult:
    """What a handler returns: the reply (JSON values) and its pictures."""

    reply: dict[str, Any] = field(default_factory=dict)
    #: Pictures (PIL images, captions burned in) sent with the reply and saved
    #: in the job folder like every picture a tool shows.
    pictures: list[Any] = field(default_factory=list)


@dataclass(frozen=True)
class Layer:
    """Something a tool leaves on a section that the agent and the user can
    see: points, lines, a label or scalar raster, or a finished picture.

    Build one with :meth:`points`, :meth:`lines`, :meth:`labels`,
    :meth:`scalar` or :meth:`picture`. ``frame`` says what the coordinates
    are: ``section_px`` (pixels of the section's image file, ``[x, y]``,
    pixel centres at integers, ``(0, 0)`` the centre of the first pixel; a
    raster covers the whole file whatever its size; a raster computed on a
    ``section_image`` array is in frame ``oriented``, which :meth:`ToolContext.add_layer`
    turns back to the file's orientation) or ``atlas_um``
    (``[x, y, z]`` in BrainGlobe atlas micrometres, the atlas's axis order).
    """

    kind: str
    frame: str
    name: str = ""
    #: points: ``(N, 2)`` (``section_px``) or ``(N, 3)`` (``atlas_um``).
    xy: np.ndarray | None = None
    #: points: one text label per point; lines: one per line.
    tags: tuple[str, ...] = ()
    #: points: one number per point.
    values: tuple[float, ...] = ()
    #: lines: ``(coordinates, closed)`` each; coordinates ``(M, 2|3)``.
    paths: tuple[tuple[np.ndarray, bool], ...] = ()
    #: labels: a 2-D integer raster; scalar: a 2-D float raster.
    raster: np.ndarray | None = None
    #: scalar: the level drawn as an outline on a picture.
    level: float = 0.5
    #: picture: a PIL image.
    image: Any = None

    @classmethod
    def points(cls, points: Any, *, labels: Sequence[str] = (),
                     values: Sequence[float] = (), frame: str = "section_px",
                     name: str = "") -> Layer:
        array = np.asarray(points, dtype=np.float64)
        layer = cls("points", frame, name=name, xy=array,
                    tags=tuple(str(item) for item in labels),
                    values=tuple(float(item) for item in values))
        layer.check()
        return layer

    @classmethod
    def lines(cls, lines: Sequence[Any], *, closed: bool | Sequence[bool] = False,
                    labels: Sequence[str] = (), frame: str = "section_px",
                    name: str = "") -> Layer:
        flags = list(closed) if isinstance(closed, Sequence) and not isinstance(
            closed, (str, bytes)) else [bool(closed)] * len(lines)
        made = tuple((np.asarray(line, dtype=np.float64), bool(flag))
                     for line, flag in zip(lines, flags, strict=True))
        layer = cls("lines", frame, name=name, paths=made,
                    tags=tuple(str(item) for item in labels))
        layer.check()
        return layer

    @classmethod
    def labels(cls, raster: Any, *, frame: str, name: str = "") -> Layer:
        layer = cls("labels", frame, name=name, raster=np.asarray(raster))
        layer.check()
        return layer

    @classmethod
    def scalar(cls, raster: Any, *, frame: str, level: float = 0.5, name: str = "") -> Layer:
        layer = cls("scalar", frame, name=name,
                    raster=np.asarray(raster, dtype=np.float32), level=float(level))
        layer.check()
        return layer

    @classmethod
    def picture(cls, image: Any, *, name: str = "") -> Layer:
        layer = cls("picture", "none", name=name, image=image)
        layer.check()
        return layer

    def check(self) -> None:
        """``ToolError`` ``BAD_LAYER`` naming what is wrong."""
        if self.kind not in LAYER_KINDS:
            raise ToolError(RefusalCode.BAD_LAYER,
                            f"a layer's kind is one of {', '.join(LAYER_KINDS)}; "
                            f"got {self.kind!r}")
        if self.frame not in KIND_FRAMES[self.kind]:
            raise ToolError(RefusalCode.BAD_LAYER,
                            f"a {self.kind} layer's frame is "
                            f"{' or '.join(KIND_FRAMES[self.kind])}; got {self.frame!r}")
        if len(self.name) > MAX_LAYER_NAME:
            raise ToolError(RefusalCode.BAD_LAYER,
                            f"a layer's name is at most {MAX_LAYER_NAME} characters")
        width = 2 if self.frame == "section_px" else 3
        if self.kind == "points":
            if self.xy is None or self.xy.ndim != 2 or self.xy.shape[1] != width:
                raise ToolError(RefusalCode.BAD_LAYER,
                                f"points in {self.frame} are an (N, {width}) array")
            if not np.isfinite(self.xy).all():
                raise ToolError(RefusalCode.BAD_LAYER, "points must be finite")
            if self.tags and len(self.tags) != len(self.xy):
                raise ToolError(RefusalCode.BAD_LAYER,
                                "points take one label each, or none")
            if self.values and len(self.values) != len(self.xy):
                raise ToolError(RefusalCode.BAD_LAYER,
                                "points take one value each, or none")
        elif self.kind == "lines":
            for coordinates, _closed in self.paths:
                if coordinates.ndim != 2 or coordinates.shape[1] != width \
                        or len(coordinates) < 2 or not np.isfinite(coordinates).all():
                    raise ToolError(RefusalCode.BAD_LAYER,
                                    f"each line in {self.frame} is a finite "
                                    f"(M, {width}) array with at least two points")
            if self.tags and len(self.tags) != len(self.paths):
                raise ToolError(RefusalCode.BAD_LAYER,
                                "lines take one label each, or none")
        elif self.kind in ("labels", "scalar"):
            raster = self.raster
            if raster is None or raster.ndim != 2 or raster.size == 0:
                raise ToolError(RefusalCode.BAD_LAYER,
                                f"a {self.kind} layer is a 2-D raster")
            if self.kind == "labels" and not np.issubdtype(raster.dtype, np.integer):
                raise ToolError(RefusalCode.BAD_LAYER,
                                "a labels raster holds integers")
            if self.kind == "labels" and raster.size and int(raster.min()) < 0:
                raise ToolError(RefusalCode.BAD_LAYER,
                                "label ids are not negative")
        elif self.image is None:
            raise ToolError(RefusalCode.BAD_LAYER,
                            "a picture layer holds an image")
