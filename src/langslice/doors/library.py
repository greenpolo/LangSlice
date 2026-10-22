"""The script door: ``langslice.open_job(folder)`` and the verbs as methods.

A script (or a coding agent's Python) opens a job folder and calls the same
verbs the agent tools and the CLI offer, by the same names and arguments
(:mod:`langslice.doors.declarations`)::

    import langslice

    job = langslice.open_job("/data/brain1")       # the job folder or its images
    job.status()["rows"]
    reply = job.position_sections(sections=[{"id": "s01.tif", "position_mm": 5.2}])
    reply["artifacts"]                              # the pictures' files
    reply.images                                    # the pictures, as PIL images

Each method returns the tool's reply as a plain, JSON-safe dict
(:class:`Reply`): the result the agent CLI gives under ``result`` (whole,
not shortened), every status row with every field (null where absent), and
under ``artifacts`` the files of its pictures, saved in the job folder with
their layers as every door saves them, as the CLI lists them (``path``,
``kind``, ``index``, ``label``). A method returns once its pictures are on
disk; the pictures themselves, as PIL images, are on the reply's
``images`` attribute (not a key). ``look``'s ``resolution`` takes any size from
128 px to the source's own pixels. Writes go through the job: one undo step each,
checkpointed, and picked up by an agent working on the same folder
(``Job.sync``), whose writes this handle picks up before each call.

A script makes a job with :func:`create_job` (a folder of sections and the
settings ``langslice-job FOLDER init`` takes, or a :class:`JobSpec`), the
same ingest every host uses, and gets the same handle. Both take
``image_model=``: the model ``trace_borders`` calls, a provider name, a
model profile (:func:`langslice.providers.profiles.image_model`) or a model
of the caller's own (``docs/library.md``).

An image-model verb (``trace_borders``) and a long engine tool (``fireants_syn``,
a plugin's ``stalign_lddmm``) return once the call has started; it lands in the
background and :meth:`JobHandle.wait` returns the landing result (notice,
pictures). :meth:`JobHandle.close` (or leaving the
``with`` block) waits for it, and a script that exits without either has
every open job closed for it at exit, while threads can still start
(:func:`langslice.job.views.at_exit`).

A method of the script's own goes into the job through :attr:`JobHandle.context`
(:class:`ScriptContext`): the context a tool plugin's handler gets
(:class:`langslice.doors.plugins.context.ToolContext`), its writes recorded as
``script``, each outermost write one undo step, every read synced first, and
two pictures of its own: a proposal drawn and saved, not written.

Nothing here loads the agent framework or a model client.
"""

from __future__ import annotations

import contextlib
import functools
import logging
import math
import os
import weakref
from collections.abc import Callable, Iterator, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any

from langslice.core.plugins import SCRIPT_CALLER, ToolError
from langslice.core.spec import JobSpec
from langslice.doors.jobs import Opened, open_folder
from langslice.doors.plugins.context import Caller, DeformationProposal, ToolContext, WriteResult

if TYPE_CHECKING:
    from PIL import Image

    from langslice.doors.tools.door import Door
    from langslice.providers.registry import ImageModel

logger = logging.getLogger(__name__)

#: ``job.json``'s record of the image model a library job was made with
#: (provider, model, profile, tested, the profile prompt's digest).
IMAGE_MODEL_KEY = "image_model"


class Reply(dict):  # type: ignore[type-arg]
    """A verb's reply: a plain dict of JSON values (``json.dumps(reply)``
    works), its pictures' files under ``artifacts``; the pictures
    themselves, as PIL images in the order a model receives them, on
    :attr:`images`, which is not a key."""

    images: list[Any]

    def __init__(self, body: Mapping[str, Any] | None = None,
                 images: Sequence[Any] = ()) -> None:
        super().__init__(body or {})
        self.images = list(images)


def plain(value: Any) -> Any:
    """*value* as JSON values: mappings with text keys, lists, numbers,
    text, booleans and null; a numpy value as its Python value, anything
    else as its text (as the agent CLI prints it)."""
    if value is None or isinstance(value, (bool, int, float, str)):
        return value
    if isinstance(value, Mapping):
        return {str(key): plain(item) for key, item in value.items()}
    if isinstance(value, (list, tuple, set, frozenset)):
        items = sorted(value, key=str) if isinstance(value, (set, frozenset)) else value
        return [plain(item) for item in items]
    import numpy as np

    if isinstance(value, np.ndarray):
        return plain(value.tolist())
    if isinstance(value, np.generic):
        return plain(value.item())
    return str(value)


def library_reply(verb: str, reply: Any, saved: Sequence[Any]) -> Reply:
    """A tool's reply for a script: the pictures out of it (their files
    under ``artifacts``, :func:`langslice.job.views.artifacts`; the images
    on :attr:`Reply.images`; their lines of text under ``media_texts``), the
    status rows uniform (:func:`langslice.core.status.with_uniform_rows`),
    everything else JSON values (:func:`plain`); a picture that could not be
    saved is a line under ``warnings``."""
    from langslice.core.status import with_uniform_rows
    from langslice.doors.tools import TOOL_MEDIA_PARTS_KEY
    from langslice.job.views import artifacts

    body = dict(reply) if isinstance(reply, Mapping) else {"result": reply}
    media = body.pop(TOOL_MEDIA_PARTS_KEY, None)
    media = media if isinstance(media, list) else []
    texts = [item for item in media if isinstance(item, str)]
    if texts:
        body["media_texts"] = texts
    out = plain(with_uniform_rows(body))
    out["artifacts"], unsaved = artifacts(saved, verb)
    if unsaved:
        out["warnings"] = unsaved
    return Reply(out, [item for item in media if not isinstance(item, str)])


# --- a script's own method ----------------------------------------------------------------


@dataclass(frozen=True)
class SavedPicture:
    """A picture a script's context saved in the job folder: ``path`` (its
    ``view.jpg``, absolute), ``index`` (its picture number, what ``zoom``'s
    ``picture`` takes), ``caption`` and the ``image`` (PIL)."""

    path: Path
    index: int
    caption: str
    image: Image.Image


#: The caption band of a preview: what it draws and that it is not written.
PREVIEW_CAPTION = "proposal: {what} drawn as proposed, NOT written"


class ScriptContext(ToolContext):
    """A script's ``job.context``: the tool plugin context
    (``docs/plugins.md``) for a script, one per job handle.

    Its writes are recorded as ``script`` (a transform's ``kind``, a
    deformation's ``method`` and engine, the record's provenance ``script``:
    ``call``, ``note``, ``meta``, ``argv0``) and logged as one
    ``script_write`` line each in ``logs/events.jsonl``; each outermost write
    is one undo step (``with job.context.writing():`` makes several one);
    each read picks up what another writer saved first; ``work_dir`` and
    ``start_work`` are refused (``NOT_FOR_SCRIPTS``). A script may write
    every property the job does not freeze: the job's ``disabled`` list
    switches tools off and does not gate a script's writes.
    :meth:`preview_overlay` and :meth:`preview_positioning` save a
    proposal's picture as a numbered picture of the job, not written.

    How (``c = job.context``; ``docs/library.md``, "Your own method", has a
    worked example):

    - ``f = c.fit_frame("<file>", stain=True)``: ``f.image`` the section on
      the fit grid (H, W, 3), ``f.stain`` its ``preprocessed`` image (what the
      native fits read by default; ``image="<name>"`` gives another section image
      as ``f.section_image``), ``f.atlas["template"]`` / ``["annotation"]`` the
      atlas plane on its native grid, ``f.placed_atlas_to_section`` 3x3 native
      atlas px -> fit-grid px (the current placement).
    - Take ``d = c.section_inputs("<file>")`` before computing and pass
      ``expect={"<file>": d}`` to the write: a section changed meanwhile is
      that row's ``STALE_INPUT``, nothing written for it.
    - An affine: ``c.set_transforms([c.transform_from_matrix("<file>", M,
      note="...").row()], expect=...)``, M the 2x3 map fit-grid px -> native px.
    - A deformation: ``c.set_deformations([{"id": "<file>", "map": xy,
      "note": "..."}], expect=...)``, xy a float (H, W, 2) array giving each
      fit-grid pixel's native atlas (x, y) under the whole map.
    - Look before writing: ``c.preview_overlay("<file>", {"deformation":
      xy}).path``; the written result: ``job.look(mode="overlay",
      sections=["<file>"])``.
    - An outside engine: ``c.write_frame("<file>", "scratch/frame")`` writes
      its inputs, ``c.map_from_itk`` / ``c.transform_from_itk`` read its
      answer (an ITK/ANTs field or transform file).
    - Points: ``c.to_atlas_um``, ``c.from_atlas_um``, ``c.region_at``;
      atlas planes anywhere: ``c.atlas_plane``, ``c.atlas_volume``.
    - The native engines with settings of your own: ``c.fit_fireants``,
      ``c.fit_elastix`` (proposals; written by ``set_*``).
    - An image of your own for the fits and pictures: ``c.add_image("<file>",
      "<name>", array, note="...")``, array ``(H, W)`` = ``c.working_grid("<file>")``
      in the image file's orientation, computed from ``c.raw_plane("<file>",
      "red")`` or ``c.working_rgb("<file>")`` (never from ``section_raster``,
      which is turned as the section is), uint8 or 0-1 floats; then
      ``section_image="<name>"`` in any fit.
    - Marks of your own on the pictures: ``c.add_layer`` (a
      :class:`langslice.plugin_api.Layer`), drawn by ``job.look(...,
      layers=[name])``.
    """

    def __init__(self, door: Door) -> None:
        super().__init__(door, Caller.script(), f"{SCRIPT_CALLER}-0")
        self._calls = 0
        self._pending = False  # a call id taken by a proposal its write has not used yet

    # --- call ids: one per outermost write ----------------------------------------------

    def _begin(self) -> None:
        if self._depth == 0 and not self._pending:
            self._calls += 1
            self.call = f"{SCRIPT_CALLER}-{os.getpid()}-{self._calls}"
            self._pending = True

    @contextlib.contextmanager
    def writing(self) -> Iterator[ToolContext]:
        outer = self._depth == 0
        if outer:
            self._begin()
        try:
            with super().writing() as held:
                yield held
        finally:
            if outer:
                self._pending = False

    def _logged(self, properties: Sequence[str], run: Callable[[], WriteResult],
                notes: Callable[[WriteResult], dict[str, str]] | None = None) -> WriteResult:
        """Run one write; a write that landed is a ``script_write`` event."""
        import sys

        outer = self._depth == 0
        self._begin()
        try:
            result = run()
        finally:
            if outer:
                self._pending = False
        if result.ok and result.written:
            self._door.job.layout.log_event(
                "script_write", property=list(properties), sections=list(result.written),
                notes=notes(result) if notes is not None else {},
                argv0=str(sys.argv[0]) if sys.argv else "", call=self.call)
        return result

    def deformation_from_map(self, section: object, native_xy: Any, *, note: str = "",
                             meta: Mapping[str, Any] | None = None, weights: Any = None,
                             provenance: Mapping[str, Any] | None = None,
                             start: str = "linear",
                             settings: Mapping[str, Any] | None = None,
                             engine_settings: Mapping[str, Any] | None = None,
                             engine_version: str = "") -> DeformationProposal:
        self._begin()
        return super().deformation_from_map(section, native_xy, note=note, meta=meta,
                                            weights=weights, provenance=provenance, start=start,
                                            settings=settings, engine_settings=engine_settings,
                                            engine_version=engine_version)

    def set_positions(self, rows: Sequence[Mapping[str, Any]], *,
                      cutting_angles: Mapping[str, float] | None = None,
                      expect: Mapping[str, str] | None = None) -> WriteResult:
        return self._logged(
            ["position", *(["angles"] if cutting_angles is not None else [])],
            lambda: super(ScriptContext, self).set_positions(
                rows, cutting_angles=cutting_angles, expect=expect))

    def set_transforms(self, rows: Sequence[Mapping[str, Any]], *,
                       expect: Mapping[str, str] | None = None) -> WriteResult:
        def notes(result: WriteResult) -> dict[str, str]:
            out: dict[str, str] = {}
            for row in rows:
                record = self._door.state.resolve(row.get("id"))
                text = str(row.get("note") or (row.get("transform") or {}).get("note") or "")
                if record is not None and record.id in result.written and text:
                    out[record.id] = text
            return out

        return self._logged(["affine"], lambda: super(ScriptContext, self).set_transforms(
            rows, expect=expect), notes)

    def set_deformations(self, rows: Sequence[Mapping[str, Any]], *,
                         expect: Mapping[str, str] | None = None) -> WriteResult:
        def notes(result: WriteResult) -> dict[str, str]:
            out: dict[str, str] = {}
            for row in rows:
                record = self._door.state.resolve(row.get("id"))
                held = row.get("deformation")
                text = str(row.get("note") or (held._warp.step.get("note") if isinstance(
                    held, DeformationProposal) else "") or "")
                if record is not None and record.id in result.written and text:
                    out[record.id] = text
            return out

        return self._logged(["deformation"], lambda: super(ScriptContext, self)
                            .set_deformations(rows, expect=expect), notes)

    def mark_damage(self, rows: Sequence[Mapping[str, Any]], *,
                    expect: Mapping[str, str] | None = None) -> WriteResult:
        return self._logged(["damage"], lambda: super(ScriptContext, self).mark_damage(
            rows, expect=expect), lambda result: {
                str(row.get("id")): str(row.get("note")) for row in rows
                if row.get("note") and str(row.get("id")) in result.written})

    def add_layer(self, section: object, layer: Any) -> Any:
        outer = self._depth == 0
        self._begin()
        try:
            ref = super().add_layer(section, layer)
        finally:
            if outer:
                self._pending = False
        import sys

        self._door.job.layout.log_event(
            "script_write", property=["layer"], sections=[ref.section], notes={},
            layer=ref.name, argv0=str(sys.argv[0]) if sys.argv else "", call=self.call)
        return ref

    # --- previews: a proposal's picture, saved, not written --------------------------------

    def preview_overlay(self, section: object, proposal: Mapping[str, Any], *,
                        resolution: int | None = None) -> SavedPicture:
        """The overlay ``look`` draws of the section (atlas borders drawn on
        the section) with *proposal* drawn and NOT written (as
        :meth:`overlay_picture` takes it: ``{"deformation": proposal or
        map}``, or ``{"transform": ..., "position_mm": ..., "angles": ...}``),
        a caption band saying so, saved as a numbered picture of the job
        (tool ``script``, mode ``overlay``) before this returns. ``zoom``
        crops it from the saved picture. A linear proposal's picture keeps
        its layers (``labels.tif``, ``borders.png``, the frame in
        ``view.json``); a deformation's has none (its record is not in the
        job). The section as it stands is ``job.look(mode="overlay")``."""
        from dataclasses import replace

        from langslice.core.layers import collecting, note, note_for

        if not proposal:
            raise ToolError("BAD_ARGS", "a preview draws a proposal; job.look(mode='overlay', "
                            "sections=[...]) draws the section as it stands")
        record = self._record(section)
        what = ("deformation" if proposal.get("deformation") is not None
                else ", ".join(key for key in ("transform", "position_mm", "angles")
                               if proposal.get(key) is not None) or "nothing")
        with collecting() as drawn:
            image = self.overlay_picture(record.id, proposal=proposal, resolution=resolution)
        held = note_for(image, drawn)
        text = PREVIEW_CAPTION.format(what=what)
        picture = self.caption(image, text)
        linear = what != "deformation" and held is not None and held.panel is not None

        def noted(shown_picture: Image.Image) -> None:
            note(shown_picture, sections=(record.id,), mode="overlay",
                 frame=held.frame if linear and held is not None else None,
                 panel=replace(held.panel, size=shown_picture.size)
                 if linear and held is not None and held.panel is not None else None,
                 recipe={"renderer": "transport_snapshot", "args": {}, "proposal": what},
                 caption=f"{text}: {held.caption}" if held is not None and held.caption
                 else text,
                 # a proposal not written: no registration the section holds
                 extra={"registrations": {}})

        return self._save([picture], noted, {"section": record.id, "proposal": what})[0]

    def preview_positioning(self, proposal: Mapping[object, Mapping[str, Any]], *,
                            sections: Sequence[object] | None = None,
                            resolution: int | None = None,
                            part: int | None = None) -> list[SavedPicture]:
        """The positioning picture (``look`` mode ``positioning``) of
        *sections* (default: every section) with *proposal* ``{section:
        {"position_mm", "angles"}}`` drawn and NOT written
        (:meth:`positioning_picture`), saved as numbered pictures of the job
        (tool ``script``, mode ``positioning``) before this returns."""
        from langslice.core.layers import note

        if not proposal:
            raise ToolError("BAD_ARGS", "a preview draws a proposal; job.look(mode="
                            "'positioning') draws the stack as it stands")
        names = [self._record(ref).id for ref in sections] if sections else [
            view.id for view in self.sections()]
        pictures = self.positioning_picture(names, proposal=proposal, resolution=resolution,
                                            part=part)

        def noted(shown_picture: Image.Image) -> None:
            note(shown_picture, sections=tuple(names), mode="positioning",
                 recipe={"renderer": "transport_snapshot", "args": {},
                         "proposal": "positions and angles"},
                 caption=PREVIEW_CAPTION.format(what="positions and angles"),
                 extra={"registrations": {}})

        return self._save(pictures, noted, {"sections": names, "proposal": "positioning"})

    def _save(self, pictures: Sequence[Image.Image], noted: Callable[[Image.Image], None],
              arguments: dict[str, Any]) -> list[SavedPicture]:
        """Save *pictures* as the job's numbered pictures, tool ``script``,
        each noted by *noted*; written before this returns."""
        from langslice.job.views import PICTURE_FILE, captured

        job = self._door.job
        with captured() as saved:
            with job.views.shown(SCRIPT_CALLER, atlas_of=lambda: self._door.ctx.atlas) as shown:
                for picture in pictures:
                    noted(picture)
                shown.show(list(pictures), arguments=plain(arguments), call_id=self.call)
        job.views.flush()
        captions = list(shown.captions) + [None] * len(pictures)
        return [SavedPicture(path=item.folder / PICTURE_FILE, index=item.seq,
                             caption=str(captions[index] or ""), image=picture)
                for index, (item, picture) in enumerate(zip(saved, pictures, strict=False))]


#: The jobs the library holds open, closed at interpreter exit (:func:`_close_open`).
_OPEN: dict[int, weakref.ref[Opened]] = {}
_EXIT_HOOKED = False


def _track(opened: Opened) -> None:
    global _EXIT_HOOKED
    if not _EXIT_HOOKED:
        from langslice.job.views import at_exit

        at_exit(_close_open)
        _EXIT_HOOKED = True
    key = id(opened)
    _OPEN[key] = weakref.ref(opened, lambda _ref: _OPEN.pop(key, None))


def _close_open() -> None:
    """Close every job a script left open: its image-model calls land and
    its pictures are written before the interpreter stops taking work."""
    for ref in list(_OPEN.values()):
        opened = ref()
        if opened is None:
            continue
        try:
            opened.close()
        except Exception:  # an exiting script has no one to raise to
            logger.warning("Could not close the job %s at exit", opened.job.folder,
                           exc_info=True)
    _OPEN.clear()


class JobHandle:
    """One open job folder; every verb its settings have is a method."""

    def __init__(self, opened: Opened, imported: dict[str, Any] | None = None) -> None:
        self._opened = opened
        self._tools = {tool.__name__: self._method(tool) for tool in opened.tools().tools}
        self._imported = imported
        self._context: ScriptContext | None = None
        _track(opened)

    def _method(self, tool: Callable[..., Any]) -> Callable[..., Reply]:
        """The verb *tool* as a method: the tool's call, then its pictures
        written, then its reply for a script (:func:`library_reply`)."""
        from langslice.job.views import captured

        job = self._opened.job
        name = tool.__name__

        @functools.wraps(tool)
        def method(*args: Any, **kwargs: Any) -> Reply:
            with captured() as saved:
                reply = tool(*args, **kwargs)
            job.views.flush()  # the pictures are on disk before the method returns
            return library_reply(name, reply, saved)

        return method

    # --- the verbs ------------------------------------------------------------------

    @property
    def verbs(self) -> list[str]:
        """The verbs this job has (its tasks decide), in registry order."""
        return self._opened.listed_verbs()

    def __getattr__(self, name: str) -> Callable[..., Reply]:
        tools = self.__dict__.get("_tools") or {}
        if name in tools:
            return tools[name]
        from langslice.ops.registry import verbs_for

        if name in verbs_for(self._opened.spec):
            spec = self._opened.spec
            raise AttributeError(f"This job's settings have no {name!r} (tasks "
                                 f"{spec.tasks}"
                                 + (f", disabled {spec.disabled}" if spec.disabled else "")
                                 + "); see .verbs")
        raise AttributeError(name)

    def __dir__(self) -> list[str]:
        return sorted({*super().__dir__(), *self.verbs})

    # --- the job ----------------------------------------------------------------------

    @property
    def context(self) -> ScriptContext:
        """The job as a method of the script's own reads and writes it
        (:class:`ScriptContext`, ``docs/library.md``): the same reads,
        conversions and writes a tool plugin has, recorded as ``script``."""
        if self._context is None:
            door = self._opened.tools().door
            assert door is not None  # build_tools keeps the door it built
            self._context = ScriptContext(door)
        return self._context

    @property
    def folder(self) -> str:
        """The job folder."""
        return str(self._opened.job.folder)

    @property
    def image_model(self) -> Any:
        """The image model ``trace_borders`` calls when one was handed in
        (:class:`langslice.providers.registry.ImageModel`), else None."""
        return self._opened.image_model

    @property
    def imported(self) -> dict[str, Any] | None:
        """What :func:`create_job`'s *registration* file placed (the import
        report: format, per section how it matched and where it was placed,
        ``unmatched`` entries, ``missing`` sections, ``refused`` ones with
        the reason, the pixel size, ``warnings``); None for a job made
        without one, or opened with :func:`open_job`."""
        return self._imported

    @property
    def job(self) -> Any:
        """The job itself (:class:`langslice.job.job.Job`: state, undo, gates)."""
        return self._opened.job

    @property
    def state(self) -> Any:
        """The stack as it stands (:class:`langslice.core.state.StackState`);
        read it, write through the verbs."""
        return self._opened.job.state

    @property
    def workspace(self) -> Any:
        """The atlas and the section files (:class:`langslice.core.workspace.Workspace`)."""
        return self._opened.ctx

    def section_maps(self, section: str, *, full_resolution: bool = False) -> Any:
        """One section's maps in memory, written nowhere: the arrays
        ``export_maps`` writes to ``sections/<name>/``
        (:class:`langslice.core.maps.SectionMaps`). ``coords`` ``(rows, cols,
        3)`` float32 atlas micrometres (NaN outside the atlas volume, as
        ``coords.tif``), ``labels`` ``(rows, cols)`` uint32 atlas ids, ``residual``
        ``(rows, cols, 2)`` float32 ``(d_row, d_col)`` grid pixels over the
        whole grid (None without a deformation), ``pixel_to_atlas_um`` (3x3 on
        ``[row, col, 1]``: ``coords = pixel_to_atlas_um @ [row + d_row, col +
        d_col, 1]``), ``grid_size`` ``(width, height)``, ``um_per_px``,
        ``full_resolution``. The grid is the section's working copy, or with
        *full_resolution* the image file's own pixels. A section without a
        placement, or whose deformation record or file cannot be read, raises
        ``ValueError``; an unknown name, ``UNKNOWN_SLICE_IDS``
        (:class:`langslice.ops.refusal.Refused`, a ``ValueError``)."""
        from langslice.ops.exports import section_maps

        return section_maps(self._opened.job, self._opened.ctx, section,
                            full_resolution=full_resolution)

    def export_stack(self) -> list[str]:
        """Write what ``submit`` writes after its step, without submitting: the
        stack's exports, ``registration.json`` and each completed trace reply
        (``ops.exports.export_stack``), never the per-section maps. Returns
        the files written."""
        from langslice.ops.exports import export_stack

        return [path for path, _kind in export_stack(self._opened.job, self._opened.ctx).files]

    def wait(self, timeout: float | None = None) -> Reply:
        """Wait for the job's background work (a fit or trace a tool started and
        answered ``started`` for) to land, and return its landing result as the
        agent CLI's ``langslice-job FOLDER VERB`` does: ``background`` holds each
        finished work's notice (what it wrote, its numbers, its picture numbers),
        ``work`` its summaries (id, kind, sections, status), the pictures on
        ``artifacts`` and ``images``. Work that finished meanwhile is delivered
        once, here or at the head of the next reply. *timeout* seconds (None: as
        long as it takes) leave work still running listed under ``running``."""
        from langslice.doors.tools.toolbox import with_notices, work_pictures

        job = self._opened.job
        job.background.wait_all(timeout)
        job.views.flush()
        before = {work.id for work in job.background.all() if work.delivered}
        reply = with_notices(job, {"status": "ok"})
        done = [work for work in job.background.all()
                if work.delivered and work.id not in before]
        reply.setdefault("background", [])
        reply["work"] = [work.summary() for work in done]
        reply["running"] = [work.id for work in job.background.running()]
        out = library_reply("wait", reply, [])
        out["artifacts"] = work_pictures(job, reply, start=0)
        return out

    def close(self) -> None:
        """Finish the job's background work: its image-model calls land and
        its pictures are written. The job stays usable; a script that exits
        without closing has this done for it."""
        self._opened.close()

    def __enter__(self) -> JobHandle:
        return self

    def __exit__(self, *_exc: object) -> None:
        self.close()

    def __repr__(self) -> str:
        return f"<langslice-job {self.folder} ({len(self.state.slices)} sections)>"


def as_image_model(value: Any) -> ImageModel | None:
    """*value* as the image model the tools call: None stays None; a provider
    name, an ``ImageModel``, an object with a ``call`` or a function goes
    through :func:`langslice.providers.profiles.image_model`."""
    if value is None:
        return None
    from langslice.providers.profiles import image_model

    return image_model(value)


def profile_record(model: ImageModel) -> dict[str, Any]:
    """What ``job.json`` records of the image model a job was made with."""
    import hashlib

    record: dict[str, Any] = {"provider": model.provider, "model": model.model,
                              "profile": model.profile, "tested": bool(model.tested)}
    if model.prompt is not None:
        record["prompt_sha256"] = hashlib.sha256(model.prompt.encode("utf-8")).hexdigest()
    if model.photograph_first is not None:
        record["photograph_first"] = model.photograph_first
    return record


def _check_model_fits(spec: JobSpec, model: ImageModel | None) -> None:
    if model is not None and not spec.nonlinear.uses_image_model:
        raise ValueError("This job runs without an image model (nonlinear provider 'none'); "
                         "create it with image_model= to trace borders")


def open_job(
    folder: str | os.PathLike[str], *,
    image_model: Any = None,
    atlas_loader: Callable[[str], Any] | None = None,
    emit: Callable[[str], None] | None = None,
) -> JobHandle:
    """Open the job in *folder* (the job folder, or the image folder beside
    it). Create one with :func:`create_job` or ``langslice-job <images> init``.

    *image_model* is the model ``trace_borders`` calls (a provider name, a
    profile, a model of your own: :func:`langslice.providers.profiles.image_model`);
    None: the job's provider as its settings name it. A job made with an
    untested profile (``job.json`` ``image_model``) opened without one offers
    no ``trace_borders``, so a reopen never traces with another prompt than
    the one the job was made with. *atlas_loader* replaces BrainGlobe's
    (tests, offline hosts)."""
    from langslice.job.layout import read_job_file

    model = as_image_model(image_model)
    opened = open_folder(folder, atlas_loader=atlas_loader, emit=emit, image_model=model)
    try:
        _check_model_fits(opened.spec, model)
        held = (read_job_file(opened.job.layout) or {}).get(IMAGE_MODEL_KEY) or {}
        if model is None and held and not held.get("tested", True):
            opened.traces_off = True
            opened.ctx.progress(f"[job] made with the untested image-model profile "
                                f"{held.get('profile')!r}: pass image_model= to trace borders")
        return JobHandle(opened)
    except BaseException:
        opened.close()
        raise


def _transform(name: str, value: Any) -> dict[str, Any]:
    """A supplied in-plane transform: LangSlice's normalized six numbers
    ``[a, b, tx, c, d, ty]`` (``langslice.core.affine``), or a transform
    dictionary kept as given."""
    if isinstance(value, Mapping):
        return dict(value)
    try:
        params = [float(item) for item in value]
    except (TypeError, ValueError) as exc:
        raise ValueError(f"transforms[{name!r}] must be six numbers [a, b, tx, c, d, ty] "
                         "or a transform dictionary") from exc
    if len(params) != 6 or not all(math.isfinite(item) for item in params):
        raise ValueError(f"transforms[{name!r}] must be six finite numbers [a, b, tx, c, d, ty]")
    return {"kind": "interactive", "params": params,
            "mirrored": params[0] * params[4] - params[1] * params[3] < 0}


def _angles(value: Mapping[str, Any]) -> dict[str, Any]:
    """Supplied cutting angles in either form (``inputs.angles``), checked
    (:func:`langslice.core.spec.supplied_angles`) and as plain floats."""
    from langslice.core.spec import supplied_angles

    stack, sections = supplied_angles(dict(value))
    if stack is not None:
        return {"pitch": stack[0], "yaw": stack[1]}
    return {name: {"pitch": pitch, "yaw": yaw} for name, (pitch, yaw) in sections.items()}


def default_tasks(images: str | os.PathLike[str], transforms: Mapping[str, Any]) -> list[str]:
    """:func:`create_job`'s tasks when none are given: ``nonlinear``, plus
    ``transform`` unless every section in *images* has a supplied
    transform."""
    from langslice.core.discovery import discover_slices

    names = {os.path.basename(path) for path in discover_slices(os.fspath(images))}
    return ["nonlinear"] if names and names <= set(transforms) else ["transform", "nonlinear"]


def create_job(
    images: str | os.PathLike[str] | JobSpec,
    *,
    atlas: str = "allen_mouse_25um",
    plane: str = "coronal",
    tasks: Sequence[str] | None = None,
    image_model: Any = None,
    job_dir: str | os.PathLike[str] | None = None,
    positions: Mapping[str, float] | None = None,
    transforms: Mapping[str, Any] | None = None,
    angles: Mapping[str, Any] | None = None,
    orientation: Mapping[str, Mapping[str, Any]] | None = None,
    pixel_size_um: float | Mapping[str, float] | None = None,
    inputs: Mapping[str, Any] | None = None,
    registration: str | os.PathLike[str] | None = None,
    fresh: bool = False,
    plugins: Any = None,
    atlas_loader: Callable[[str], Any] | None = None,
    emit: Callable[[str], None] | None = None,
    **settings: Any,
) -> JobHandle:
    """Create (or continue) the job of a folder of section images; the same
    handle as :func:`open_job`.

    *images* is the folder of sections (or a :class:`JobSpec`, used as it
    is: then only *image_model*, *atlas_loader* and *emit* apply). The job
    folder is *job_dir*, else ``<images>/langslice``.

    The registration you supply, by section filename: *positions* (atlas
    millimetres from the anterior edge of the volume), *transforms* (six
    numbers ``[a, b, tx, c, d, ty]``, LangSlice's normalized in-plane affine
    on the oriented section, or a transform dictionary), *orientation*
    (``{"flip": bool, "rotation_deg": 0|90|180|270}``), the cutting
    *angles* (``{"pitch": deg, "yaw": deg}`` for the whole stack, or
    ``{filename: {"pitch": deg, "yaw": deg}}`` for each section its own; a
    section the per-section form does not name is flat:
    :func:`langslice.core.spec.supplied_angles`) and *pixel_size_um* (one
    number, or ``{filename: um}``); *inputs*
    takes any other key of ``JobSpec.inputs``. *tasks* None: ``nonlinear``,
    plus ``transform`` (``elastix_affine``) unless every section has a supplied
    transform.

    Or *registration*: a linear registration made elsewhere (a QuickNII or
    VisuAlign JSON/XML, a DeepSlice CSV/JSON/XML, a LangSlice
    ``registration.json``), imported as every section's position, cutting
    angles, orientation and transform (``doors.jobs.with_registration``; not
    with *positions*, *transforms*, *angles* or *orientation*; VisuAlign
    markers are not imported). *tasks* None is then ``["nonlinear"]``: the
    imported placement kept as it is. The handle's :attr:`JobHandle.imported`
    says what was placed; the warnings go to *emit* as well.

    *plugins* adds tool plugins to the job's toolbox (``docs/plugins.md``):
    ``module:factory`` references, factory functions defined at module level,
    or :class:`~langslice.plugin_api.ToolSpec` objects, one or a list; the job
    records the references (``plugin_config`` in *settings* gives a factory its
    options), so reopening it loads the same plugins. Loading runs their code.

    *image_model* as in :func:`open_job`; None makes a job
    without one (nonlinear provider ``none``). *fresh* starts over instead
    of continuing the folder's job. *settings* are further
    :class:`JobSpec` fields (``preprocess``, ``interval_um`` lives in
    ``position``...).
    """
    from langslice.doors.api.setup import load_credentials
    from langslice.doors.jobs import REGISTRATION_EXCLUDES, create, with_registration
    from langslice.job.layout import write_job_file

    load_credentials()  # the image model's keys, as open_job loads them
    model = as_image_model(image_model)
    if plugins is not None:
        from langslice.doors.plugins import as_refs

        refs = as_refs(plugins)
        if isinstance(images, JobSpec):
            raise ValueError("create_job takes a JobSpec as it is: set plugins on the spec "
                             "instead")
        settings["plugins"] = [*list(settings.get("plugins") or ()), *refs]
    if isinstance(images, JobSpec):
        given = {name: value for name, value in {
            "positions": positions, "transforms": transforms, "angles": angles,
            "orientation": orientation, "pixel_size_um": pixel_size_um, "inputs": inputs,
            "tasks": tasks, "job_dir": job_dir,
            "registration": registration}.items() if value is not None}
        if given or settings or fresh:
            raise ValueError("create_job takes a JobSpec as it is: set "
                             f"{sorted({*given, *settings})} on the spec instead")
        spec = images
    else:
        folder = Path(os.path.abspath(os.path.expanduser(os.fspath(images))))
        supplied: dict[str, Any] = dict(inputs or {})
        if positions:
            supplied["positions"] = {str(name): float(mm) for name, mm in positions.items()}
        if transforms:
            supplied["transforms"] = {str(name): _transform(str(name), value)
                                      for name, value in transforms.items()}
        if angles:
            supplied["angles"] = _angles(angles)
        if orientation:
            supplied["orientation"] = {str(name): dict(value)
                                       for name, value in orientation.items()}
        if pixel_size_um is not None:
            # JobSpec reads it (core.spec.supplied_pixel_sizes) and refuses a bad size.
            supplied["pixel_size_um"] = (dict(pixel_size_um) if isinstance(pixel_size_um, Mapping)
                                         else pixel_size_um)
        if registration is not None:
            clashes = sorted(key for key in REGISTRATION_EXCLUDES if supplied.get(key))
            if clashes:
                raise ValueError(f"registration= supplies every section's placement; "
                                 f"it cannot be combined with {clashes}")
        nonlinear = dict(settings.pop("nonlinear", None) or {})
        if model is not None:
            nonlinear.update(provider=model.provider, image_model=model.model)
        else:
            nonlinear.setdefault("provider", "none")
        spec = JobSpec.from_dict({
            "atlas": atlas, "plane": plane, "debrief": False, **settings,
            "image_folder": str(folder),
            "tasks": list(tasks) if tasks is not None else ["nonlinear"]
            if registration is not None else default_tasks(
                folder, supplied.get("transforms") or {}),
            "nonlinear": nonlinear, "inputs": supplied,
            "job_dir": None if job_dir is None else os.path.abspath(
                os.path.expanduser(os.fspath(job_dir))),
            "resume": not fresh,
        })
    _check_model_fits(spec, model)
    imported: dict[str, Any] | None = None
    if registration is not None:
        spec, imported = with_registration(spec, registration, atlas_loader=atlas_loader,
                                           emit=emit)
    opened = create(spec, atlas_loader=atlas_loader, emit=emit, image_model=model)
    try:
        if model is not None:
            write_job_file(opened.job.layout, **{IMAGE_MODEL_KEY: profile_record(model)})
        return JobHandle(opened, imported)
    except BaseException:
        opened.close()
        raise
