"""Pure filtering helpers for SFT corpus curation.

These stateless helpers are shared training-core utilities after the
expert-iteration runtime was retired.
"""

from __future__ import annotations

from collections.abc import Iterable
from typing import Any


def best_of_n(rollouts: Iterable[dict[str, Any]]) -> list[dict[str, Any]]:
    """Per ``section_id``, return the single rollout with the lowest ``abs_err_mm``.

    Stable: ties are broken by first occurrence in input order.
    """
    best_per_section: dict[str, dict[str, Any]] = {}
    for row in rollouts:
        sid = row["section_id"]
        err = float(row["abs_err_mm"])
        existing = best_per_section.get(sid)
        if existing is None or err < float(existing["abs_err_mm"]):
            best_per_section[sid] = row
    seen: list[str] = []
    seen_set: set[str] = set()
    for row in rollouts:
        sid = row["section_id"]
        if sid in seen_set:
            continue
        seen.append(sid)
        seen_set.add(sid)
    return [best_per_section[sid] for sid in seen if sid in best_per_section]


def threshold_accept(
    rollouts: Iterable[dict[str, Any]],
    *,
    threshold_pct: float,
) -> list[dict[str, Any]]:
    """Keep rollouts where ``abs_err_mm <= threshold_pct * plane_extent_mm``."""
    if threshold_pct < 0:
        raise ValueError(f"threshold_pct must be >= 0, got {threshold_pct}")
    kept: list[dict[str, Any]] = []
    for row in rollouts:
        extent = float(row["plane_extent_mm"])
        if extent <= 0:
            continue
        tol = threshold_pct * extent
        if float(row["abs_err_mm"]) <= tol:
            kept.append(row)
    return kept

__all__ = ["best_of_n", "threshold_accept"]
