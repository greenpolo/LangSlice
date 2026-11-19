"""The pictures of a plugin context (``ctx.pictures``): what ``look`` draws, with a
proposal drawn.

:class:`Pictures` draws a section's overlay and the positioning
picture as they stand or with a proposal not yet written, and captions a
picture for the agent. Nothing here is saved or written.
"""

from __future__ import annotations

import copy
from collections.abc import Mapping, Sequence
from typing import TYPE_CHECKING, Any, cast

from PIL import Image

from langslice.core.jsonable import JsonObject
from langslice.core.plugins import ToolError
from langslice.core.refusal import RefusalCode
from langslice.core.stack.affine import StoredAffine
from langslice.plugins.proposals import DeformationProposal, Proposals, TransformProposal
from langslice.plugins.reads import Records

if TYPE_CHECKING:
    from langslice.tools.run import Run


def _no_registration(image: Image.Image) -> None:
    """Mark the picture drawn of a proposal not written: its saved index line
    records no registration the section holds (``extra["registrations"]``
    empty, :meth:`langslice.job.pictures.ViewStore._registrations`)."""
    from langslice.core.pictures.drawn import note_of

    held = note_of(image)
    if held is not None:
        held.extra = {**held.extra, "registrations": {}}


class Pictures:
    """Draw a section's pictures, with a proposal or as they stand:
    ``ctx.pictures``.

    Parameters
    ----------
    run
        The run the call belongs to: its job, workspace and spec.
    records
        How a section is found by filename.
    proposals
        The context's proposals, which a map or numbers are turned into.
    """

    def __init__(self, run: Run, records: Records, proposals: Proposals) -> None:
        self._run = run
        self._records = records
        self._proposals = proposals

    def overlay_picture(
        self,
        section: object,
        *,
        proposal: Mapping[str, Any] | None = None,
        resolution: int | None = None,
    ) -> Image.Image:
        """The overlay ``look`` draws of the section (borders on the section,
        captioned), as it stands, or with a *proposal* not yet written:
        ``{"transform": record or params, "position_mm": mm, "angles":
        (pitch, yaw)}``, any of them, or ``{"deformation": proposal or map}``
        alone (a :class:`DeformationProposal`, or the map
        ``ctx.proposals.deformation_from_map`` takes), drawn on the section's current
        placement. Nothing is saved or written here; the picture is the
        handler's to return."""
        from langslice.core.pictures import look as looks

        record = self._records.get(section)
        state = self._run.state
        store = self._run.job.deformations
        if proposal and proposal.get("deformation") is not None:
            if set(proposal) - {"deformation"}:
                raise ToolError(
                    RefusalCode.BAD_ARGUMENT,
                    "a deformation proposal is drawn on the "
                    "section's current placement: give it alone",
                )
            warped = proposal["deformation"]
            if not isinstance(warped, DeformationProposal):
                warped = self._proposals.deformation_from_map(record.id, warped)
            elif warped.id != record.id:
                raise ToolError(
                    RefusalCode.BAD_ARGUMENT, f"the proposal is {warped.id}'s, not {record.id}'s"
                )
            store = self._proposal_store(warped)
            state = copy.deepcopy(state)
            held = state.by_id(record.id)
            assert held is not None
            held.deformation = warped.reference()
            record = held
        elif proposal:
            state = copy.deepcopy(state)
            held = state.by_id(record.id)
            assert held is not None
            transform = proposal.get("transform")
            if isinstance(transform, TransformProposal):
                transform = transform.record
            elif transform is not None and not isinstance(transform, Mapping):
                transform = self._proposals.transform_from_params(record.id, transform).affine
            if isinstance(transform, StoredAffine):
                held.transform = transform
                held.deformation = None
            elif transform is not None:
                held.transform = StoredAffine.from_json(cast("JsonObject", transform))
                held.deformation = None
            if proposal.get("position_mm") is not None:
                held.position_mm = float(proposal["position_mm"])
                held.deformation = None
            if proposal.get("angles") is not None:
                angles = cast("Sequence[float]", proposal["angles"])
                held.angles = (float(angles[0]), float(angles[1]))
                held.deformation = None
            record = held
        edge = int(resolution) if resolution else None
        request = looks.LookRequest(mode="overlay", sections=(record.id,), long_edge=edge)
        drawn = looks.look(self._run.workspace, state, request, store=store)
        if proposal:
            _no_registration(drawn[0].image)
        return drawn[0].image

    def positioning_picture(
        self,
        ids: Sequence[object],
        *,
        proposal: Mapping[object, Mapping[str, Any]] | None = None,
        resolution: int | None = None,
        part: int | None = None,
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
        written here; the pictures are the handler's to return."""
        from langslice.core.pictures import look as looks
        from langslice.tools.changing.positions import distinct_positions

        names = [self._records.get(ref).id for ref in ids]
        state = self._run.state
        if proposal:
            state = copy.deepcopy(state)
            for ref, change in proposal.items():
                held = state.by_id(self._records.get(ref).id)
                assert held is not None
                unknown = set(change) - {"position_mm", "angles"}
                if unknown:
                    raise ToolError(
                        RefusalCode.BAD_ARGUMENT,
                        "a positioning proposal gives position_mm and/or "
                        f"angles per section; got {sorted(unknown)}",
                    )
                if change.get("position_mm") is not None:
                    held.position_mm = float(change["position_mm"])
                if change.get("angles") is not None:
                    angles = cast("Sequence[float]", change["angles"])
                    held.angles = (float(angles[0]), float(angles[1]))
                held.deformation = None
        records = [state.by_id(name) for name in names]
        positions = distinct_positions([r.position_mm for r in records if r is not None])
        edge = int(resolution) if resolution else None
        request = looks.LookRequest(
            mode="positioning",
            sections=tuple(names),
            positions_mm=tuple(positions),
            long_edge=edge,
            part=part,
        )
        drawn = looks.look(self._run.workspace, state, request, store=self._run.job.deformations)
        images = [picture.image for picture in drawn]
        if proposal:
            for image in images:
                _no_registration(image)
            images = [
                self.caption(image, "proposal: positions and angles drawn as proposed, NOT written")
                for image in images
            ]
        return images

    def _proposal_store(self, proposal: DeformationProposal) -> Any:
        """A record store that holds *proposal*'s record (in memory only) and
        reads every other from the job's."""
        from langslice.core.fits.deformation import RecordStore

        held = self._run.job.deformations
        store = RecordStore(root=held.root, folder_of=held.folder_of)
        store.put(proposal.key, proposal.warp.record)
        return store

    @staticmethod
    def caption(image: Image.Image, text: str, *, sections: Sequence[object] = ()) -> Image.Image:
        """*image* with *text* in a caption band below it (pictures reach a
        model as bare attachments: what they show is said in the picture, and in the
        caption its entry in the reply's ``pictures`` carries). The captioned
        picture shows the sections *image* was drawn of (an
        :meth:`overlay_picture`'s section), or *sections* (filenames) for a
        picture of the handler's own: its saved index line names them and the
        registration each is drawn under, which ``restore_registration``'s
        list reads."""
        from langslice.core.pictures.captions import caption
        from langslice.core.pictures.drawn import annotate, note_of

        drawn = note_of(image)
        shown = tuple(str(name) for name in sections) or (
            tuple(drawn.sections) if drawn is not None else ()
        )
        fields: dict[str, Any] = {"caption": text}
        if shown:
            fields["sections"] = shown
            fields["mode"] = drawn.mode if drawn is not None and drawn.mode else "plugin"
        captioned = annotate(caption(image, text), **fields)
        if drawn is not None and "registrations" in drawn.extra:
            held = note_of(captioned)
            if held is not None:
                held.extra["registrations"] = dict(drawn.extra["registrations"])
        return captioned
