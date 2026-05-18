"""Phase C — curriculum weight updates from per-bin slicebench MAE.

Phase A computed ``section_id → bin_idx`` (see :mod:`bins`); Phase B wired
:class:`WeightedRowDataset` and the trainer subclasses that consume it
(see :mod:`sampler`). Phase C closes the loop: between expert-iteration
rounds, read slicebench's ``per_coord_bin`` MAE block, compute a weight
per ``(plane, bin_label)``, expand to a per-section dict, and push it
back into the weighted dataset.

Formula
-------
For each ``(plane, bin_label)`` with ``mae_bin`` mm we compute:

    w_bin = (mae_bin / baseline_mae) ** alpha

where ``baseline_mae`` defaults to the mean of all per-bin MAE values
(skipping empty bins). Weights are then expanded to a per-section dict
``{section_id: w}`` via the ``section_bins`` mapping (sections whose bin
isn't represented in the eval get the baseline weight 1.0).

Smoothing + safety
------------------
- 0.5-EMA smoothing against ``prev_weights`` BEFORE the cap is applied
  so a single noisy slicebench run can't yank the curriculum sideways.
- Per-update ratio cap at ``max_weight_change`` (default 3×) — we clamp
  the ratio ``w_new/w_prev`` into ``[1/cap, cap]``. Without this guard
  a bin that briefly spikes 10× over baseline could starve every other
  bin of training signal.
- Floor at ``0.1 × baseline_weight`` (= 0.1) to keep every bin
  represented in the sampler — fully starving a bin makes its eval
  signal noisier on the next round, which is exactly what curriculum
  is trying to avoid.

The function is pure: it never reads slicebench output unless the
caller passes a path. The caller composes:

    per_bin_mae = read_per_bin_mae(summary_path)
    section_bins = compute_section_bins(allocation, ...)
    weights = compute_weights(per_bin_mae, section_bins, prev_weights=prev)
    update_weighted_dataset(dataset, weights)
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:  # pragma: no cover — typing only
    from .sampler import WeightedRowDataset


PerBinMAE = dict[tuple[str, str], float]
"""``{(plane, bin_label): mae_mm}``. ``bin_label`` is ``q1..q5``."""

# Slicebench's labelling — see ``slicebench/score.py:_coord_bin_label``.
_BIN_LABELS: tuple[str, ...] = ("q1", "q2", "q3", "q4", "q5")


def read_per_bin_mae(slicebench_summary_path: Path | str) -> PerBinMAE:
    """Read ``per_coord_bin[plane][q1..q5].mae_mm`` from a slicebench summary.

    Empty bins (``n == 0``) are skipped — slicebench writes them as
    ``{"n": 0}`` with no ``mae_mm`` field, so they'd raise a KeyError if
    we tried to read blindly.
    """
    path = Path(slicebench_summary_path)
    summary = json.loads(path.read_text(encoding="utf-8"))
    per_coord_bin = summary.get("per_coord_bin") or {}
    out: PerBinMAE = {}
    for plane, bins_for_plane in per_coord_bin.items():
        if not isinstance(bins_for_plane, dict):
            continue
        for label, stats in bins_for_plane.items():
            if not isinstance(stats, dict):
                continue
            n = stats.get("n", 0)
            if not n:
                continue
            mae = stats.get("mae_mm")
            if mae is None:
                continue
            out[(str(plane), str(label))] = float(mae)
    return out


def compute_weights(
    per_bin_mae: PerBinMAE,
    section_bins: dict[str, tuple[str, str] | int],
    *,
    baseline_mae: float | None = None,
    alpha: float = 1.0,
    max_weight_change: float = 3.0,
    prev_weights: dict[str, float] | None = None,
    floor_fraction: float = 0.1,
    smoothing: float = 0.5,
) -> dict[str, float]:
    """Compute per-section curriculum weights.

    Parameters
    ----------
    per_bin_mae:
        ``{(plane, bin_label): mae_mm}`` — output of :func:`read_per_bin_mae`.
    section_bins:
        ``{section_id: (plane, bin_label) | int}``. Two shapes are accepted:

        - ``(plane, bin_label)`` tuples (e.g. ``("coronal", "q3")``) —
          hand-built from :func:`bins.compute_section_bins` plus the
          plane lookup.
        - Plain ``int`` bin indices — the legacy
          :func:`bins.compute_section_bins` return shape. In this case
          per-bin MAE must be averaged across planes (we fall back to a
          plane-agnostic mean MAE per ``q_idx``). Prefer the tuple shape.
    baseline_mae:
        Reference MAE that maps to ``w == 1.0``. Defaults to the
        unweighted mean of all ``per_bin_mae`` values. ``None`` with an
        empty ``per_bin_mae`` -> uniform weights (everyone gets 1.0).
    alpha:
        Sharpness exponent. ``1.0`` = linear up-weighting (default).
        ``0.5`` would be square-root (gentler), ``2.0`` would be
        quadratic (more aggressive).
    max_weight_change:
        Maximum per-update multiplicative change vs ``prev_weights``.
        Default 3× both ways. Set to ``inf`` to disable the cap.
    prev_weights:
        Optional ``{section_id: w}`` from the previous round. When
        present, the new weight is EMA-smoothed against it (see
        ``smoothing``) BEFORE the cap is applied.
    floor_fraction:
        Lower bound for any weight, expressed as a fraction of the
        baseline weight (1.0). ``0.1`` keeps every bin at least at
        ``0.1 × baseline``. Set to ``0`` to disable.
    smoothing:
        EMA factor in ``[0, 1]``. ``0.5`` (default) means
        ``w_new = 0.5 * w_computed + 0.5 * w_prev``. ``0.0`` disables
        smoothing (use the raw computed weight).

    Returns
    -------
    dict[section_id, float]
        One entry per section in ``section_bins``. Sections whose bin
        isn't present in ``per_bin_mae`` get the baseline weight 1.0
        (subject to the cap + floor).
    """
    if not 0.0 <= smoothing <= 1.0:
        raise ValueError(f"smoothing must be in [0, 1], got {smoothing}")
    if alpha <= 0:
        raise ValueError(f"alpha must be positive, got {alpha}")
    if max_weight_change < 1.0:
        raise ValueError(
            f"max_weight_change must be >= 1.0 (got {max_weight_change}); "
            "values <1 would forbid any weight from changing at all."
        )
    if floor_fraction < 0.0:
        raise ValueError(f"floor_fraction must be >= 0, got {floor_fraction}")

    # Baseline defaults to mean per-bin MAE (skipping empty bins).
    if baseline_mae is None:
        if per_bin_mae:
            baseline_mae = sum(per_bin_mae.values()) / len(per_bin_mae)
        else:
            baseline_mae = 0.0
    baseline_mae = float(baseline_mae)

    # Plane-agnostic fallback for legacy ``int`` bin indices.
    plane_agnostic_by_qidx: dict[int, float] = {}
    if any(isinstance(v, int) for v in section_bins.values()):
        # Average each q_idx across planes that have it.
        accum: dict[int, list[float]] = {}
        for (_plane, label), mae in per_bin_mae.items():
            try:
                qi = _BIN_LABELS.index(label)
            except ValueError:
                continue
            accum.setdefault(qi, []).append(mae)
        for qi, vals in accum.items():
            plane_agnostic_by_qidx[qi] = sum(vals) / len(vals)

    floor = floor_fraction  # baseline weight is 1.0 by construction
    out: dict[str, float] = {}
    for section_id, bin_key in section_bins.items():
        # Resolve to a numeric MAE for this section's bin.
        mae_for_bin: float | None = None
        if isinstance(bin_key, tuple):
            mae_for_bin = per_bin_mae.get(bin_key)
        elif isinstance(bin_key, int):
            mae_for_bin = plane_agnostic_by_qidx.get(bin_key)

        # Compute the raw target weight. Sections whose bin isn't in
        # the eval keep the baseline weight (1.0).
        if mae_for_bin is None or baseline_mae <= 0.0:
            w_computed = 1.0
        else:
            ratio = mae_for_bin / baseline_mae
            w_computed = ratio**alpha

        # EMA smooth against prev_weights, then apply ratio cap.
        if prev_weights is not None and section_id in prev_weights:
            prev = float(prev_weights[section_id])
            if smoothing > 0.0 and prev > 0.0:
                w_smoothed = smoothing * w_computed + (1.0 - smoothing) * prev
            else:
                w_smoothed = w_computed
            # Cap the ratio of new vs prev within [1/cap, cap].
            if prev > 0.0:
                ratio_to_prev = w_smoothed / prev
                if ratio_to_prev > max_weight_change:
                    w_capped = prev * max_weight_change
                elif ratio_to_prev < 1.0 / max_weight_change:
                    w_capped = prev / max_weight_change
                else:
                    w_capped = w_smoothed
            else:
                w_capped = w_smoothed
        else:
            w_capped = w_computed

        # Floor.
        w_final = max(w_capped, floor)
        out[section_id] = float(w_final)
    return out


def update_weighted_dataset(
    dataset: "WeightedRowDataset",
    weights: dict[str, float],
    *,
    section_id_key: str = "section_id",
    baseline_weight: float = 1.0,
) -> int:
    """Push per-section weights onto ``dataset._weights``.

    Maps each row's ``section_id`` (read from
    ``dataset._rows[i][section_id_key]``) to the corresponding entry in
    ``weights``. Rows whose section isn't in ``weights`` get
    ``baseline_weight`` (default 1.0). Returns the number of rows that
    matched a weight (for caller logging).

    Notes
    -----
    The RLVR :class:`rlvr.dataset.RowDataset` uses
    ``_image_paths`` / ``_apply_clahes`` underscore keys for lazy decode,
    but ``section_id`` should land in the public row dict so the env
    factory can pick it up via ``reset(**row)``. This helper assumes
    the row carries it; callers building rows should add it.
    """
    n = len(dataset)
    if n == 0:
        return 0
    new_weights: list[float] = []
    n_matched = 0
    for i in range(n):
        row = dataset._rows[i]  # noqa: SLF001 — internal access by design
        sid = row.get(section_id_key)
        if sid is not None and sid in weights:
            new_weights.append(float(weights[sid]))
            n_matched += 1
        else:
            new_weights.append(float(baseline_weight))
    dataset.set_weights(new_weights)
    return n_matched


def write_weights_json(
    path: Path | str,
    weights: dict[str, float],
    *,
    metadata: dict[str, Any] | None = None,
) -> None:
    """Persist ``weights`` to ``path`` as JSON. Round to keep diffs small.

    Format::

        {"_metadata": {...}, "<section_id>": <weight>, ...}

    The ``_metadata`` key is namespaced with a leading underscore so it
    can never collide with a real ``section_id`` (which we never prefix
    with underscore). Callers can read the file straight back with
    :func:`read_weights_json` or just ``json.loads(...)`` if they don't
    care about metadata.
    """
    payload: dict[str, Any] = {sid: round(float(w), 6) for sid, w in weights.items()}
    if metadata:
        payload["_metadata"] = dict(metadata)
    out_path = Path(path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")


def read_weights_json(path: Path | str) -> dict[str, float]:
    """Inverse of :func:`write_weights_json`. Drops ``_metadata`` keys."""
    data = json.loads(Path(path).read_text(encoding="utf-8"))
    return {k: float(v) for k, v in data.items() if not k.startswith("_")}


__all__ = [
    "PerBinMAE",
    "compute_weights",
    "read_per_bin_mae",
    "read_weights_json",
    "update_weighted_dataset",
    "write_weights_json",
]
