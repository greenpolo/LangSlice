"""A script's ``job.context``."""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any

from langslice.core.jsonable import to_jsonable
from langslice.core.plugins import SCRIPT_CALLER, ToolError
from langslice.core.refusal import RefusalCode
from langslice.plugins.caller import Caller
from langslice.plugins.context import ToolContext

if TYPE_CHECKING:
    from PIL import Image

    from langslice.tools.run import Run


@dataclass(frozen=True)
class SavedPicture:
    """A picture a script's context saved in the job folder: ``path`` (its
    ``view.jpg``, absolute), ``index`` (its picture number, what ``zoom``'s
    ``picture`` takes), ``caption`` and the ``image`` (PIL)."""

    path: Path
    index: int
    caption: str
    image: Image.Image


PREVIEW_CAPTION = "proposal: {what} drawn as proposed, NOT written"


class ScriptContext(ToolContext):
    """A script's ``job.context``: the tool plugin context
    (``docs/developers/plugins.md``) for a script, one per job handle.

    Its writes are recorded as ``script`` (a transform's ``kind``, a
    deformation's ``method`` and engine, the record's provenance ``script``:
    ``call``, ``note``, ``meta``, ``argv0``) and logged as one
    ``script_write`` line each in ``logs/events.jsonl``; each outermost write
    is one undo step (``with job.context.writes.writing():`` makes several one);
    each read picks up what another writer saved first; ``work_dir`` and
    ``start_work`` are refused (``TOOL_OFF``). A script may write
    every property the job does not freeze: the job's ``disabled`` list
    switches tools off and does not gate a script's writes.
    :meth:`preview_overlay` and :meth:`preview_positioning` save a
    proposal's picture as a numbered picture of the job, not written.

    The calls are in parts (``help(c.reads)``...): ``c.reads`` (sections, atlas,
    images, masks), ``c.frames`` (fit frame, atlas planes, points), ``c.proposals``
    (a transform or a deformation as LangSlice stores it, not written), ``c.fits``
    (an outside engine's files, the native engines, registration files),
    ``c.writes`` and ``c.pictures``.

    How (``c = job.context``; ``docs/developers/library.md``, "Your own method", has a
    worked example):

    - ``f = c.frames.fit_frame("<file>", stain=True)``: ``f.image`` the section on
      the fit grid (H, W, 3), ``f.stain`` its ``preprocessed`` image (what the
      native fits read by default; ``image="<name>"`` gives another section image
      as ``f.section_image``), ``f.atlas["template"]`` / ``["annotation"]`` the
      atlas plane on its native grid, ``f.placed_atlas_to_section`` 3x3 native
      atlas px -> fit-grid px (the current placement).
    - Take ``d = c.reads.section_inputs("<file>")`` before computing and pass
      ``expect={"<file>": d}`` to the write: a section changed meanwhile is
      that row's ``STALE_INPUT``, nothing written for it.
    - An affine: ``c.writes.set_transforms([c.proposals.transform_from_matrix("<file>", M,
      note="...").row()], expect=...)``, M the 2x3 map fit-grid px -> native px.
    - A deformation: ``c.writes.set_deformations([{"id": "<file>", "map": xy,
      "note": "..."}], expect=...)``, xy a float (H, W, 2) array giving each
      fit-grid pixel's native atlas (x, y) under the whole map.
    - Look before writing: ``c.preview_overlay("<file>", {"deformation":
      xy}).path``; the written result: ``job.look(mode="overlay",
      sections=["<file>"])``.
    - An outside engine: ``c.fits.write_frame("<file>", "scratch/frame")`` writes
      its inputs, ``c.fits.map_from_itk`` / ``c.fits.transform_from_itk`` read its
      answer (an ITK/ANTs field or transform file).
    - Points: ``c.frames.to_atlas_um``, ``c.frames.from_atlas_um``, ``c.frames.region_at``;
      atlas planes anywhere: ``c.frames.atlas_plane``, ``c.frames.atlas_volume``.
    - The native engines with settings of your own: ``c.fits.fit_fireants``,
      ``c.fits.fit_elastix`` (proposals; written by ``set_*``).
    - An image of your own for the fits and pictures: ``c.writes.add_image("<file>",
      "<name>", array, note="...")``, array ``(H, W)`` = ``c.reads.working_grid("<file>")``
      in the image file's orientation, computed from ``c.reads.raw_plane("<file>",
      "red")`` or ``c.reads.working_rgb("<file>")`` (never from ``section_raster``,
      which is turned as the section is), uint8 or 0-1 floats; then
      ``section_image="<name>"`` in any fit.
    - Marks of your own on the pictures: ``c.writes.add_layer`` (a
      :class:`langslice.plugin_api.Layer`), drawn by ``job.look(...,
      layers=[name])``.
    """

    def __init__(self, run: Run) -> None:
        super().__init__(run, Caller.script(), f"{SCRIPT_CALLER}-0")

    def preview_overlay(
        self, section: object, proposal: Mapping[str, Any], *, resolution: int | None = None
    ) -> SavedPicture:
        """The overlay ``look`` draws of the section (atlas borders drawn on
        the section) with *proposal* drawn and NOT written (as
        ``ctx.pictures.overlay_picture`` takes it: ``{"deformation": proposal or
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
            raise ToolError(
                RefusalCode.BAD_ARGUMENT,
                "a preview draws a proposal; job.look(mode='overlay', "
                "sections=[...]) draws the section as it stands",
            )
        record = self._records.get(section)
        what = (
            "deformation"
            if proposal.get("deformation") is not None
            else ", ".join(
                key
                for key in ("transform", "position_mm", "angles")
                if proposal.get(key) is not None
            )
            or "nothing"
        )
        with collecting() as drawn:
            image = self.pictures.overlay_picture(
                record.id, proposal=proposal, resolution=resolution
            )
        held = note_for(image, drawn)
        text = PREVIEW_CAPTION.format(what=what)
        picture = self.pictures.caption(image, text)
        linear = what != "deformation" and held is not None and held.panel is not None

        def noted(shown_picture: Image.Image) -> None:
            note(
                shown_picture,
                sections=(record.id,),
                mode="overlay",
                frame=held.frame if linear and held is not None else None,
                panel=replace(held.panel, size=shown_picture.size)
                if linear and held is not None and held.panel is not None
                else None,
                recipe={"renderer": "transport_snapshot", "args": {}, "proposal": what},
                caption=f"{text}: {held.caption}" if held is not None and held.caption else text,
                extra={"registrations": {}},
            )

        return self._save([picture], noted, {"section": record.id, "proposal": what})[0]

    def preview_positioning(
        self,
        proposal: Mapping[object, Mapping[str, Any]],
        *,
        sections: Sequence[object] | None = None,
        resolution: int | None = None,
        part: int | None = None,
    ) -> list[SavedPicture]:
        """The positioning picture (``look`` mode ``positioning``) of
        *sections* (default: every section) with *proposal* ``{section:
        {"position_mm", "angles"}}`` drawn and NOT written
        (``ctx.pictures.positioning_picture``), saved as numbered pictures of the job
        (tool ``script``, mode ``positioning``) before this returns."""
        from langslice.core.pictures.drawn import note

        if not proposal:
            raise ToolError(
                RefusalCode.BAD_ARGUMENT,
                "a preview draws a proposal; job.look(mode="
                "'positioning') draws the stack as it stands",
            )
        names = (
            [self._records.get(ref).id for ref in sections]
            if sections
            else [view.id for view in self.reads.sections()]
        )
        pictures = self.pictures.positioning_picture(
            names, proposal=proposal, resolution=resolution, part=part
        )

        def noted(shown_picture: Image.Image) -> None:
            note(
                shown_picture,
                sections=tuple(names),
                mode="positioning",
                recipe={
                    "renderer": "transport_snapshot",
                    "args": {},
                    "proposal": "positions and angles",
                },
                caption=PREVIEW_CAPTION.format(what="positions and angles"),
                extra={"registrations": {}},
            )

        return self._save(pictures, noted, {"sections": names, "proposal": "positioning"})

    def _save(
        self,
        pictures: Sequence[Image.Image],
        noted: Callable[[Image.Image], None],
        arguments: dict[str, Any],
    ) -> list[SavedPicture]:
        """Save *pictures* as the job's numbered pictures, tool ``script``,
        each noted by *noted*; written before this returns."""
        from langslice.job.pictures import PICTURE_FILE, captured

        job = self._run.job
        with captured() as saved:
            workspace = self._run.workspace
            with job.views.shown(SCRIPT_CALLER, atlas_of=lambda: workspace.atlas) as shown:
                for picture in pictures:
                    noted(picture)
                shown.show(list(pictures), arguments=to_jsonable(arguments), call_id=self.call)
        job.views.flush()
        captions = list(shown.captions) + [None] * len(pictures)
        return [
            SavedPicture(
                path=item.folder / PICTURE_FILE,
                index=item.seq,
                caption=str(captions[index] or ""),
                image=picture,
            )
            for index, (item, picture) in enumerate(zip(saved, pictures, strict=False))
        ]
