"""Pure filter functions for expert-iteration rollouts.

A "scored rollout" row is a dict with at least:
    - ``section_id`` (str): unique key for the prompt
    - ``generation_idx`` (int): which of the N rollouts for that prompt
    - ``abs_err_mm`` (float): plane-aware absolute error (from
      ``slicebench.score.score_predictions``)
    - ``plane_extent_mm`` (float): per-plane atlas extent in mm
    - ``plane`` (str), ``atlas`` (str): used downstream for trace formatting

The filters here are deliberately stateless and side-effect-free so the
driver can compose them, test them in isolation, and feed the results
straight into trace formatting.
"""

from __future__ import annotations

from collections.abc import Iterable
from typing import Any


def best_of_n(rollouts: Iterable[dict[str, Any]]) -> list[dict[str, Any]]:
    """Per ``section_id``, return the single rollout with the lowest ``abs_err_mm``.

    Stable: ties are broken by the first occurrence in the input order
    (this matches the typical "lowest-index generation wins on tie" intuition
    and keeps the function deterministic).
    """
    best_per_section: dict[str, dict[str, Any]] = {}
    for row in rollouts:
        sid = row["section_id"]
        err = float(row["abs_err_mm"])
        existing = best_per_section.get(sid)
        if existing is None or err < float(existing["abs_err_mm"]):
            best_per_section[sid] = row
    # Preserve input ordering of section_ids for deterministic downstream IO.
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
    """Keep every rollout where ``abs_err_mm <= threshold_pct * plane_extent_mm``.

    ``threshold_pct`` is a fraction of the plane extent (e.g. ``0.015`` for
    1.5% — matching ``plane_tolerance_mm`` used in the distilled corpus).
    Each rollout supplies its own ``plane_extent_mm`` so a single threshold
    fraction handles coronal/sagittal/horizontal in one pass.
    """
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
