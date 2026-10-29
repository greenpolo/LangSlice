"""A section's applied deformation, as the stack refers to it.

The deformation itself is a saved fit on disk (a ``DeformableRecord`` in
its folder, ``record.json`` and its warp). The section holds a
:class:`DeformationRef` to it: where it is, the linear placement it was
fitted on (any change to that placement clears it), the steps that made
it and a summary of its displacement.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field

from langslice.core.jsonable import Json, JsonObject, empty_object


@dataclass(frozen=True)
class LandmarkWarp:
    """Hold what a landmark warp was built from.

    Parameters
    ----------
    revision
        The digest of the pairs at the application.
    pairs
        Each pair's id and the key of its endpoints then.
    folds
        The pairs whose triangles folded, by id, with what folded.
    baseline_key
        The deformation the landmark warp was composed on; None when it
        was built on the linear placement.
    baseline_steps
        The steps of that deformation.
    holds
        The automatic hold points a warp built with them kept, as section
        fractions; None for a warp of the pairs and the four corners alone.
    """

    revision: str
    pairs: Mapping[str, str]
    folds: JsonObject
    baseline_key: str | None
    baseline_steps: tuple[JsonObject, ...]
    holds: Sequence[Json] | None = None


@dataclass(frozen=True)
class DeformationRef:
    """Hold a section's reference to the deformation it holds.

    Parameters
    ----------
    key
        The saved fit's key (its digest); None for a kept-linear marker.
    linear_key
        The digest of the linear placement it was fitted on.
    steps
        The steps that made it, first first: each step's method, its
        start (``linear`` or ``current``) and its settings; None for a
        kept-linear marker.
    record
        The saved fit's folder, relative to the job folder; None for a
        proposal drawn before it is saved.
    summary
        Its displacement and fold numbers; None for a proposal.
    inverse_source
        How the saved fit's inverse was made; None for a proposal.
    landmark
        What a landmark warp was built from; None for every other fit.
    kept_linear
        The note of a kept-linear marker, a reference with no saved fit
        that a job folder of format 3 may hold under ``keep_linear`` (the
        section was left on its linear placement); None for a fit.
    extra
        Other keys the reference carried (an earlier LangSlice's), kept
        as they came so the state file reads back the same.
    """

    key: str | None
    linear_key: str
    steps: tuple[JsonObject, ...] | None
    record: str | None = None
    summary: JsonObject | None = None
    inverse_source: str | None = None
    landmark: LandmarkWarp | None = None
    kept_linear: str | None = None
    extra: JsonObject = field(default_factory=empty_object)
