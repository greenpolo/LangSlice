"""The sorting an engine's mixture is given: the spreads and brightnesses of its classes."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, fields


@dataclass(frozen=True)
class Sorting:
    """Hold the values a call sets for STalign's and emlddmm's pixel classes.

    A value left out (None) is the engine's own. Brightness runs from 0, black,
    to 1, white (STalign: slide 0, typical tissue 1).

    Parameters
    ----------
    match_sd
        The matching class's spread (``sigmaM``).
    background_sd
        The background class's spread (``sigmaB``).
    artifact_sd
        The artifact class's spread (``sigmaA``).
    background_mean
        The background class's brightness (``muB``).
    artifact_mean
        The artifact class's brightness (``muA``).
    """

    match_sd: float | None = None
    background_sd: float | None = None
    artifact_sd: float | None = None
    background_mean: float | None = None
    artifact_mean: float | None = None

    @classmethod
    def from_given(cls, given: Mapping[str, object]) -> Sorting:
        """Read the tool's ``sorting`` argument (keys already checked) as numbers."""
        return cls(**{key: float(str(value)) for key, value in given.items()})

    def keywords(self) -> dict[str, float]:
        """Return the engines' keywords for the values that are set."""
        return {
            KEYWORDS[field.name]: value
            for field in fields(self)
            if (value := getattr(self, field.name)) is not None
        }


KEYWORDS = {
    "match_sd": "sigmaM",
    "background_sd": "sigmaB",
    "artifact_sd": "sigmaA",
    "background_mean": "muB",
    "artifact_mean": "muA",
}
SORTING_KEYS = tuple(KEYWORDS)
