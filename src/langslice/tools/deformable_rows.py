"""One section's answer from a deformable fit, and its row in a reply.

:func:`langslice.tools.deformable.fit_deformable` answers each section with
a :class:`DeformableRow`: the fit it applied (:class:`AppliedFit`) or why
not. :func:`deformable_row_to_json` is the row's one conversion to the reply
(``fireants_syn``'s ``results``, a packaged trace's landing).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import cast

from langslice.core.fits.inputs import EstimatedBlock, InputsBlock
from langslice.core.jsonable import JsonObject, to_jsonable
from langslice.core.refusal import Refused, ReplyStatus
from langslice.core.stack.codec import Written
from langslice.tools.refusal import refused_row


@dataclass(frozen=True)
class AppliedFit:
    """What one fit applied gave, as its section's row reports it."""

    engine_settings: JsonObject
    numbers: JsonObject
    runtime_s: float
    cached: bool
    gpu: JsonObject
    inputs: InputsBlock | None = None
    estimated: EstimatedBlock | None = None


@dataclass(frozen=True)
class DeformableRow:
    """One section's answer from :func:`fit_deformable`: its fit, or why not.

    A row without a refusal holds a fit that was applied
    (:func:`deformable_row_to_json` is its reply).
    """

    id: str
    settings: JsonObject | None = None
    trace_regions: JsonObject | None = None
    refusal: Refused | None = None
    queued: bool = False
    applied: AppliedFit | None = None
    steps: int = 0
    image_indexes: tuple[int, ...] | None = None


def deformable_row_to_json(row: DeformableRow) -> Written:
    """Return one section's row of a deformable fit's reply.

    Parameters
    ----------
    row
        The section's answer.

    Returns
    -------
    dict
        A section refused before its fit was set up: its refusal
        (:func:`langslice.tools.refusal.refused_row`); one refused while it
        was set up: ``id``, ``status``, ``settings``, ``error`` and the
        facts. Otherwise ``id``, ``trace_regions`` (a traced fit's own),
        ``settings``, then the refusal, or ``status`` ``ok``, the engine
        numbers, the displacement summary, ``runtime_s``, ``cached``, the
        GPU queue, ``inputs`` and ``estimated`` (a fit of a section image),
        then the refusal of a record that could not be saved, or
        ``written``, ``steps`` and the pictures' ``image_indexes``.
    """
    if row.settings is None:
        assert row.refusal is not None
        return refused_row(row.id, row.refusal)
    out: Written = {"id": row.id}
    if not row.queued:
        out["status"] = ReplyStatus.ERROR
    if row.trace_regions is not None:
        out["trace_regions"] = to_jsonable(row.trace_regions)
    out["settings"] = to_jsonable(row.settings)
    applied = row.applied
    if applied is not None:
        out.update(
            status="ok",
            engine_settings=to_jsonable(applied.engine_settings),
            **to_jsonable(applied.numbers),
            runtime_s=applied.runtime_s,
            cached=applied.cached,
        )
        out.update(to_jsonable(applied.gpu))
        if applied.inputs is not None:
            out["inputs"] = to_jsonable(applied.inputs.payload())
            out["estimated"] = (
                None
                if applied.estimated is None
                else to_jsonable(applied.estimated.payload())
            )
    if row.refusal is not None:
        out.update(cast("Written", row.refusal.payload()))
        return out
    out.update(written=True, steps=row.steps)
    if row.image_indexes is not None:
        out["image_indexes"] = list(row.image_indexes)
    return out
