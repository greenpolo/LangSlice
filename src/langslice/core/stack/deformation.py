"""A section's applied deformation, as the stack refers to it.

The deformation itself is a saved fit on disk (a ``DeformableRecord`` in
its folder, ``record.json`` and its warp). The section holds a
:class:`DeformationRef` to it: where it is, the linear placement it was
fitted on (any change to that placement clears it), the steps that made
it and a summary of its displacement.
"""

from __future__ import annotations

from typing import Any

from pydantic import model_validator

from langslice.core.stack.part import Object, Part, omitted


class LandmarkWarp(Part):
    """Hold what a landmark warp was built from.

    Stored beside the deformation's own keys, each under its name with
    ``landmark_`` before it.

    Parameters
    ----------
    revision
        The digest of the pairs at the application.
    pairs
        Each pair's id and the key of its endpoints then.
    folds
        The pairs whose triangles folded, by id, with what folded.
    holds
        The automatic hold points a warp built with them kept, as section
        fractions; None for a warp of the pairs and the four corners alone.
    baseline_key
        The deformation the landmark warp was composed on; None when it
        was built on the linear placement.
    baseline_steps
        The steps of that deformation.
    """

    revision: str
    pairs: dict[str, str]
    folds: Object
    holds: list[Any] | None = omitted()
    baseline_key: str | None
    baseline_steps: tuple[Object, ...]


LANDMARK_PREFIX = "landmark_"


class DeformationRef(Part):
    """Hold a section's reference to the deformation it holds.

    Parameters
    ----------
    record
        The saved fit's folder, relative to the job folder; None for a
        proposal drawn before it is saved.
    key
        The saved fit's key (its digest).
    linear_key
        The digest of the linear placement it was fitted on.
    steps
        The steps that made it, first first: each step's method, its
        start (``linear`` or ``current``) and its settings.
    summary
        Its displacement and fold numbers; None for a proposal.
    inverse_source
        How the saved fit's inverse was made; None for a proposal.
    landmark
        What a landmark warp was built from; None for every other fit.
        Stored flat among the reference's keys.

    Other keys the reference carried are kept as they came.
    """

    record: str | None = omitted()
    key: str
    linear_key: str
    steps: tuple[Object, ...]
    summary: Object | None = omitted()
    inverse_source: str | None = omitted()
    landmark: LandmarkWarp | None = omitted()

    @model_validator(mode="before")
    @classmethod
    def nested(cls, data: Any) -> Any:
        """Gather the flat ``landmark_*`` keys into the landmark warp."""
        if not isinstance(data, dict) or f"{LANDMARK_PREFIX}revision" not in data:
            return data
        warp = {
            name: data[LANDMARK_PREFIX + name]
            for name in LandmarkWarp.model_fields
            if LANDMARK_PREFIX + name in data
        }
        rest = {
            key: value
            for key, value in data.items()
            if not (key.startswith(LANDMARK_PREFIX) and key[len(LANDMARK_PREFIX) :] in warp)
        }
        return {**rest, "landmark": warp}

    def arranged(self, data: dict[str, Any]) -> dict[str, Any]:
        """Write a landmark warp's key first and its steps after its facts."""
        warp = data.pop("landmark", None)
        if warp is None:
            return data
        head = ("key", "record", "linear_key", "summary", "inverse_source")
        first = {key: data.pop(key) for key in head if key in data}
        steps = data.pop("steps")
        flat = {LANDMARK_PREFIX + name: value for name, value in warp.items()}
        return {**first, **flat, "steps": steps, **data}
