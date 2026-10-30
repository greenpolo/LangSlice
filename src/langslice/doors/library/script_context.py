"""A script's ``job.context``."""

from __future__ import annotations

import contextlib
import os
from collections.abc import Callable, Iterator, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any

from langslice.core.jsonable import to_jsonable
from langslice.core.plugins import SCRIPT_CALLER, ToolError
from langslice.core.refusal import RefusalCode
from langslice.doors.plugins.caller import Caller
from langslice.doors.plugins.context import ToolContext
from langslice.doors.plugins.proposals import DeformationProposal
from langslice.doors.plugins.writes import WriteResult

if TYPE_CHECKING:
    from PIL import Image

    from langslice.tools.run import Run


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
    (``docs/developers/plugins.md``) for a script, one per job handle.

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

    How (``c = job.context``; ``docs/developers/library.md``, "Your own method", has a
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

    def __init__(self, run: Run) -> None:
        super().__init__(run, Caller.script(), f"{SCRIPT_CALLER}-0")
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
            self._run.job.layout.log_event(
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
                record = self._run.state.resolve(row.get("id"))
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
                record = self._run.state.resolve(row.get("id"))
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

        self._run.job.layout.log_event(
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

        from langslice.core.pictures.drawn import collecting, note, note_for

        if not proposal:
            raise ToolError(RefusalCode.BAD_ARGS,
                            "a preview draws a proposal; job.look(mode='overlay', "
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
        from langslice.core.pictures.drawn import note

        if not proposal:
            raise ToolError(RefusalCode.BAD_ARGS,
                            "a preview draws a proposal; job.look(mode="
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
        from langslice.job.pictures import PICTURE_FILE, captured

        job = self._run.job
        with captured() as saved:
            with job.views.shown(SCRIPT_CALLER, atlas_of=lambda: self._run.ctx.atlas) as shown:
                for picture in pictures:
                    noted(picture)
                shown.show(list(pictures), arguments=to_jsonable(arguments), call_id=self.call)
        job.views.flush()
        captions = list(shown.captions) + [None] * len(pictures)
        return [SavedPicture(path=item.folder / PICTURE_FILE, index=item.seq,
                             caption=str(captions[index] or ""), image=picture)
                for index, (item, picture) in enumerate(zip(saved, pictures, strict=False))]
