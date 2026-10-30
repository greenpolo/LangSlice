"""The writes of a plugin context: each one undo step through the tools' operations.

:class:`ContextWrites` writes positions, transforms, deformations, damage
marks, made images and layers through the operations the native tools use,
refusing a frozen property (``FROZEN``) and a section changed since its
digest (``STALE_INPUT``), and returns a :class:`WriteResult`.
"""

from __future__ import annotations

import contextlib
from collections.abc import Iterator, Mapping, Sequence
from dataclasses import dataclass, field, replace
from typing import TYPE_CHECKING, Any, Self, cast

import numpy as np

from langslice.core.jsonable import JsonObject, to_jsonable
from langslice.core.plugins import Layer, ToolError
from langslice.core.refusal import RefusalCode, Refused
from langslice.core.stack.codec import Written, affine_from_json
from langslice.doors.plugins.proposals import (
    ContextProposals,
    DeformationProposal,
)
from langslice.doors.plugins.reads import (
    LayerRef,
    as_tool_error,
    unorient_raster,
)
from langslice.tools.refusal import (
    refused_row,
    section_refusal,
    unknown_sections,
)
from langslice.tools.stale import section_inputs, stale_refusal

if TYPE_CHECKING:
    from langslice.tools.warps import WarpApplied


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
                   details={key: value for key, value in refusal.facts.items()
                            if key != "message"})

    def reply(self) -> dict[str, Any]:
        """The result as reply fields: ``written`` and ``rows``, or the refusal."""
        if not self.ok:
            return Refused(str(self.error), message=self.message, **self.details).payload()
        return {"written": list(self.written), "rows": [dict(row) for row in self.rows]}


class ContextWrites(ContextProposals):
    """Write a section's properties, made images and layers."""

    def add_image(self, section: object, name: str, array: Any, note: str = "") -> dict[str, Any]:
        """Register *array* as the section's made image *name*, made by this
        script (*note* says how; it is recorded as the image's source): a
        ``(H, W)`` array on the section's working grid in the image FILE's
        orientation, before the section's quarter turn and flip, as
        :meth:`raw_plane` and :meth:`working_rgb` give the section (never
        :meth:`section_raster`, whose pixels are turned and flipped as the
        section is: a flipped section's raster handed back here would be a
        mirror image); ``uint8``, or floats from 0 to 1. Saved as
        ``sections/<stem>/images/<name>.png``, one undo step; every fit and
        ``look`` can then read it by name. Returns its entry (``source``,
        ``file``, ``digest``). ``ToolError`` ``BAD_ARGS`` for a raw channel's
        name, another shape or values."""
        from langslice.tools import images as ops_images

        self._require()
        record = self._record(section)
        try:
            with self.writing():
                live = self._run.state.by_id(record.id) or record
                # the script's array, checked by add_image
                given = cast("object", array)
                return ops_images.add_image(self._run.job, self._run.workspace, live, str(name),
                                            given, note)
        except Refused as refusal:
            raise as_tool_error(refusal) from refusal

    def _require(self, *properties: str) -> None:
        caller = self._caller
        if caller.kind != "write":
            raise ToolError(RefusalCode.READ_TOOL_WRITES,
                            f"{caller.name} is declared a read tool and "
                            "writes nothing; declare kind='write'")
        outside = [name for name in properties if name not in caller.writes]
        if outside:
            raise ToolError(RefusalCode.WRITE_NOT_DECLARED,
                            f"{caller.name} writes {outside} but "
                            f"declares writes={list(caller.writes)}")

    @contextlib.contextmanager
    def writing(self) -> Iterator[Self]:
        """Hold the job's write lock around several writes: everything
        written inside is ONE undo step. A plugin call's writes are one step
        too, unless another write landed on the job between two of the
        call's (then they stay separate steps, so undo never takes back
        someone else's write); a script's outermost writes are a step each.
        A long tool computes first and applies inside, passing ``expect=``
        to the writes so a section that changed meanwhile is refused."""
        job = self._run.job
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
            record = self._run.state.by_id(name)
            if record is None:
                continue
            now = {section_inputs(self._run.state, record, deformation=flag)
                   for flag in (False, True)}  # taken with or without the deformation
            if expect[name] not in now:
                rows[name] = stale_refusal().payload(section_id=name)
        return rows

    def set_positions(self, rows: Sequence[Mapping[str, Any]], *,
                      cutting_angles: Mapping[str, float] | None = None,
                      expect: Mapping[str, str] | None = None) -> WriteResult:
        """Set positions (``rows``: ``{"id", "position_mm"}``; atlas
        millimetres, clamped into the atlas range and reported) and, with
        *cutting_angles* (``{"pitch_deg", "yaw_deg"}``), the stack's angles.
        The stack then runs in the order of the positions. Needs ``position``
        (and ``angles``) in the tool's ``writes``."""
        from langslice.tools import positions

        self._require("position", *(["angles"] if cutting_angles is not None else []))
        job, workspace = self._run.job, self._run.workspace
        with self.writing():
            out: dict[str, dict[str, Any]] = {}
            todo: list[Mapping[str, Any]] = []
            ids = [self._resolve_id(row.get("id")) for row in rows]
            stale = self._stale([name for name in ids if name], expect)
            for row, name in zip(rows, ids, strict=True):
                if not name:
                    ref = str(row.get("id"))
                    out[ref] = self._unknown(ref).payload(section_id=ref)
                elif name in stale:
                    out[name] = stale[name]
                elif frozen := section_refusal(job, name, "position"):
                    out[name] = frozen.payload(section_id=name)
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
        return self._run.job.clear_stale_deformations()

    def _resolve_id(self, ref: object) -> str:
        record = self._run.state.resolve(ref)
        return record.id if record is not None else ""

    def _unknown(self, ref: str) -> Refused:
        """Return the refusal of a write row whose ``id`` names no section."""
        return unknown_sections(self._run.state, [ref])

    def set_transforms(self, rows: Sequence[Mapping[str, Any]], *,
                       expect: Mapping[str, str] | None = None) -> WriteResult:
        """Set in-plane transforms. A row is ``{"id", "transform": record}``
        (a :class:`TransformProposal`'s ``record``, or :meth:`TransformProposal.row`)
        or ``{"id", "params": [six numbers], "note"?}``. A frozen affine, or
        a transform that would turn over a section whose orientation is
        frozen, comes back as that section's row (``FROZEN``); the others are
        written as one undo step. Needs ``affine`` in the tool's ``writes``."""
        from langslice.tools import transforms

        self._require("affine")
        job = self._run.job
        with self.writing():
            out: dict[str, dict[str, Any]] = {}
            ids = [self._resolve_id(row.get("id")) for row in rows]
            stale = self._stale([name for name in ids if name], expect)
            written: list[str] = []
            for row, name in zip(rows, ids, strict=True):
                if not name:
                    ref = str(row.get("id"))
                    out[ref] = self._unknown(ref).payload(section_id=ref)
                    continue
                if name in stale:
                    out[name] = stale[name]
                    continue
                record = row.get("transform")
                try:
                    if record is None:
                        if "params" not in row:
                            raise ToolError(RefusalCode.BAD_ARGS,
                                            "a row is {id, transform} or "
                                            "{id, params}")
                        note = str(row.get("note") or "")
                        stored = self.transform_from_params(
                            name, row["params"], note=note).affine
                    else:  # reportAny: a plugin's plain dict
                        stored = affine_from_json(cast("JsonObject", record))
                    assert stored is not None
                    done = transforms.set_transforms(job, {name: stored})
                except ToolError as exc:
                    out[name] = Refused(exc.code, message=exc.message).payload(section_id=name)
                    continue
                except Refused as refusal:
                    out[name] = refusal.payload(section_id=name)
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
        Each is stored as the deformation a native fit applies is (one undo
        step for the call; a later change to the section's placement clears
        it): it replaces the section's deformation, or, a proposal made on top
        of it (``deformation_from_map(start="current")``), keeps its steps
        before its own. Per section: a frozen deformation
        is ``FROZEN``, a map that cannot be stored ``BAD_FIELD``, a section
        without a position or transform ``INVALID_LINEAR_PLACEMENT``, one whose
        inputs differ from *expect* (or whose placement moved since its
        proposal was made) ``STALE_INPUT``. A map is converted before the
        job's write lock is taken, so a long tool holds it only to write.
        The ok rows carry the numbers ``DeformationProposal.summary`` has.
        Needs ``deformation`` in the tool's ``writes``."""
        from langslice.tools import warps

        self._require("deformation")
        job = self._run.job
        out: dict[str, dict[str, Any]] = {}
        proposals: list[DeformationProposal] = []
        for row in rows:
            name = self._resolve_id(row.get("id"))
            if not name:
                ref = str(row.get("id"))
                out[ref] = self._unknown(ref).payload(section_id=ref)
                continue
            frozen = section_refusal(job, name, "deformation")
            if frozen is not None:
                out[name] = frozen.payload(section_id=name)
                continue
            proposal = row.get("deformation")
            try:
                if proposal is None:
                    if "map" not in row:
                        raise ToolError(RefusalCode.BAD_ARGS,
                                        "a row is {id, map, note?, meta?} or "
                                        "{id, deformation}")
                    proposal = self.deformation_from_map(
                        name, row["map"], note=str(row.get("note") or ""),
                        meta=row.get("meta"))
                elif not isinstance(proposal, DeformationProposal) or proposal.id != name:
                    raise ToolError(RefusalCode.BAD_ARGS,
                                    "deformation is this section's "
                                    "DeformationProposal (deformation_from_map)")
            except ToolError as exc:
                out[name] = Refused(exc.code, message=exc.message).payload(section_id=name)
                continue
            out[name] = {"id": name}
            proposals.append(proposal)
        written: list[str] = []
        with self.writing():
            stale = self._stale([proposal.id for proposal in proposals], expect)
            out.update(stale)
            done = warps.apply_warps(
                job, [proposal._warp for proposal in proposals if proposal.id not in stale])
            for applied in done.rows:
                out[applied.id] = _warp_row(applied)
            written = list(done.written)
        return WriteResult(written=written, rows=list(out.values()))

    def mark_damage(self, rows: Sequence[Mapping[str, Any]], *,
                    expect: Mapping[str, str] | None = None) -> WriteResult:
        """Mark atlas regions a section has lost (``rows``: ``{"id", "regions",
        "note"?}``; empty regions clear the marks), as ``mark_damage`` does."""
        from langslice.tools import damage

        self._require()
        job, workspace = self._run.job, self._run.workspace
        with self.writing():
            out: list[dict[str, Any]] = []
            written: list[str] = []
            for row in rows:
                name = self._resolve_id(row.get("id"))
                if not name:
                    ref = str(row.get("id"))
                    out.append(self._unknown(ref).payload(section_id=ref))
                    continue
                stale = self._stale([name], expect)
                if name in stale:
                    out.append(stale[name])
                    continue
                try:
                    done = damage.mark_damage(job, workspace, name, list(row.get("regions") or ()),
                                              str(row.get("note") or ""))
                except Refused as refusal:
                    out.append(refusal.payload(section_id=name))
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
        job = self._run.job
        if layer.frame == "oriented":  # a raster computed on a section_image array
            held = self._record(section)
            layer = replace(layer, frame="section_px", raster=unorient_raster(
                np.asarray(layer.raster), held.rotation_deg, held.flip))
        with self.writing():
            record = self._run.state.by_id(record_id)
            assert record is not None
            before = job.snapshot()
            ref = register_layer(job.layout, record, self._caller.name, self.call, layer,
                                 persist=job.persist)
            job.commit(before)
        return LayerRef(id=ref["id"], section=record.id, name=ref["name"], kind=ref["kind"],
                        frame=ref["frame"], file=ref["file"])


def _warp_row(applied: WarpApplied) -> Written:
    """Return one warp's row: its refusal, or ``ok`` with its numbers."""
    if applied.refusal is not None:
        return refused_row(applied.id, applied.refusal)
    row: Written = {"id": applied.id, "status": "ok"}
    row.update(to_jsonable(applied.numbers or {}))
    return row
