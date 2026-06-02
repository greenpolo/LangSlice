"""Synthetic terminal-trace corpus generator for expert-iteration SFT.

Converts manifest allocation rows into synthetic SFT JSONL rows by:

1. Calling ``generate_trace(strategy="lane_a_prefix")`` to build a procedural
   multi-step prefix from the empirical distributions in
   ``langslice_traces._empirical``.
2. Injecting a synthetic ``FinalAnswer(position_mm=GT, reasoning=<canned text>)``
   so the prefix becomes a complete trace.
3. Deciding per-row whether to render as ``sft_full`` (multi-turn; controlled by
   ``multi_turn_floor``) or ``sft_answer_only`` (single-turn / answer-only).
4. Serializing back to the langslice-native JSONL row shape that
   ``sft.dataset._validate_row`` expects.

None of this touches real model weights or Docker — it is pure-Python and
suitable for unit testing without GPU or SigLIP.

Key design decisions
--------------------
- We bypass the ``RenderedExample`` (HF chat-template) path entirely and build
  the native JSONL row from the ``CanonicalTrace`` directly.  ``RenderedExample``
  requires PIL image loading from disk (``_common._user_turn`` calls
  ``hydrate_image``); for corpus generation we only need the string paths.
- For ``sft_answer_only`` rows the ``trace`` list contains exactly one ``submit``
  step (no tool_call steps at all).  ``sft_full`` rows include the full prefix of
  ``ToolStep`` records plus the terminal submit.
- ``sft_answer_only`` hardcodes ``reasoning=""`` in its renderer, but
  ``_validate_row`` rejects empty reasoning.  We therefore write the reasoning
  directly from the injected ``FinalAnswer`` rather than routing through the
  renderer's HF message layer.
"""

from __future__ import annotations

import json
import logging
import math
import random
from copy import deepcopy
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import Any, Literal

# Single source of truth for manifest→brainglobe atlas-name aliases.
from langslice_harness.atlas.core import _ATLAS_ALIASES as _BG_ATLAS_ALIASES

_LOGGER = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Reasoning composition
#
# Each row's submit step needs a non-empty ``reasoning`` string (validator
# rejects empty).  We prefer an anatomically-grounded composition listing the
# regions visible at the GT position (looked up via brainglobe), and fall back
# to the canned string only when brainglobe is unavailable or the lookup fails.
#
# The region list is a high-signal pseudo-label: training on
# ``"Visible structures: hippocampus; thalamus; ..."`` routes gradient back
# through the vision projector for each named structure, far richer than the
# canned generic sentence.  Per-row cost is ~0.5ms after the first atlas load
# (~425ms one-time).
# ---------------------------------------------------------------------------
_FALLBACK_REASONING: str = "Submitting estimate based on slice morphology."
_THINKING_FALLBACK: str = "Visible clues: atlas signature unavailable."
_THINKING_SIGNATURE_TOP_K = 16
SyntheticReasoningMode = Literal["region_dump", "thinking_signature"]


@lru_cache(maxsize=32)
def _load_bg_atlas(atlas_name: str) -> Any | None:
    """Return a cached BrainGlobeAtlas instance, or None if unavailable.

    Failures are logged at DEBUG so they don't spam normal runs; callers
    silently fall back to the canned reasoning string.
    """
    try:
        from brainglobe_atlasapi import BrainGlobeAtlas  # noqa: PLC0415
    except ImportError:
        _LOGGER.debug("brainglobe_atlasapi not installed; reasoning fallback")
        return None
    resolved = _BG_ATLAS_ALIASES.get(atlas_name, atlas_name)
    try:
        return BrainGlobeAtlas(resolved)
    except Exception as exc:  # noqa: BLE001
        _LOGGER.debug("brainglobe load failed for %r: %s", resolved, exc)
        return None


def _region_paths_at_position(
    atlas: Any, plane: str, position_mm: float,
) -> list[tuple[tuple[int, ...], int]]:
    """Region ``(structure_id_path, voxel_count)`` tuples for a 2D slice.

    Sorted by voxel area descending (most-prominent structures first).
    Returns ``[]`` if any step fails — caller falls back to the canned string.

    Returning paths (not just leaf names) lets the reasoning composer roll
    each region up its hierarchy to match a token budget.
    """
    import numpy as np  # noqa: PLC0415

    from langslice_harness.atlas.space import (  # noqa: PLC0415
        atlas_space_context,
        slice_axis_index,
    )

    if atlas is None:
        return []
    try:
        ctx = atlas_space_context(atlas)
        axis = slice_axis_index(ctx, plane)  # type: ignore[arg-type]
    except Exception as exc:  # noqa: BLE001
        _LOGGER.debug("atlas space resolve failed for %r/%r: %s", atlas, plane, exc)
        return []

    res_mm = ctx.resolution_um[axis] / 1000.0
    voxel_idx = int(round(position_mm / res_mm))
    if voxel_idx < 0 or voxel_idx >= ctx.shape[axis]:
        return []

    if axis == 0:
        ann_slice = atlas.annotation[voxel_idx, :, :]
    elif axis == 1:
        ann_slice = atlas.annotation[:, voxel_idx, :]
    else:
        ann_slice = atlas.annotation[:, :, voxel_idx]

    ids, counts = np.unique(ann_slice, return_counts=True)
    order = np.argsort(-counts)

    out: list[tuple[tuple[int, ...], int]] = []
    for sid_arr, c_arr in zip(ids[order], counts[order], strict=True):
        sid = int(sid_arr)
        if sid == 0:
            continue
        try:
            path_raw = atlas.structures[sid]["structure_id_path"]
            path = tuple(int(p) for p in path_raw)
        except (KeyError, TypeError, ValueError):
            continue
        out.append((path, int(c_arr)))
    return out


def _names_at_depth(
    paths_counts: list[tuple[tuple[int, ...], int]],
    depth_offset: int,
    structures: Any,
) -> list[str]:
    """Roll each region up ``depth_offset`` levels from its leaf.

    ``depth_offset=0`` → leaf names (finest); ``=1`` → immediate parent; etc.
    Voxel counts are accumulated across regions that collapse to the same
    ancestor, then names are emitted ordered by total accumulated area desc.
    Multiple leaf regions with the same ancestor become one entry — the
    natural way the hierarchy expresses coarser anatomical groupings.
    """
    rolled: dict[int, int] = {}
    for path, count in paths_counts:
        idx = max(0, len(path) - 1 - depth_offset)
        anc_id = path[idx]
        rolled[anc_id] = rolled.get(anc_id, 0) + count
    names: list[str] = []
    for anc_id, _count in sorted(rolled.items(), key=lambda kv: -kv[1]):
        try:
            name = str(structures[anc_id]["name"])
        except (KeyError, TypeError):
            continue
        if name:
            names.append(name)
    return names


def _region_names_at_position(atlas_name: str, plane: str, position_mm: float) -> list[str]:
    """Leaf-level region names at slice (thin wrapper, retained for callers)."""
    atlas = _load_bg_atlas(atlas_name)
    if atlas is None:
        return []
    paths = _region_paths_at_position(atlas, plane, position_mm)
    return _names_at_depth(paths, 0, atlas.structures)


def _compose_synth_reasoning(
    atlas_name: str,
    plane: str,
    gt_mm: float,
    fetch_positions: list[float] | None = None,
    char_budget: int = 7000,
) -> str:
    """Compose the reasoning string for one synthetic submit step.

    Answer-only rows (no ``fetch_positions``)::

        Target coronal allen_mouse_25um slice contains: A; B; C; ...
        Therefore, the target coronal allen_mouse_25um slice is at 8.40 mm.

    Multi-turn rows (with ``fetch_positions``) add contrastive lines naming
    the regions present at the anteriormost and posteriormost fetched atlas
    positions that are **not** in the target slice — high-signal supervision
    for "why those atlas images are NOT the target" using set-difference
    against the target's region list.

    Length control: when the leaf-level rendering would exceed ``char_budget``
    (set so the row fits ``max_seq_length`` after image/system/tool overhead),
    walks up the brainglobe atlas hierarchy uniformly across all three position
    lookups, replacing leaf regions with their parent/grandparent until the
    rendering fits.  Uniform depth across positions preserves the semantic
    meaning of the anterior/posterior set differences.

    Returns the canned fallback only when brainglobe is unavailable or the
    target-position lookup fails (validator-safe; always non-empty).
    """
    label = f"{plane} {atlas_name}"
    atlas = _load_bg_atlas(atlas_name)
    if atlas is None:
        return _FALLBACK_REASONING

    target_paths = _region_paths_at_position(atlas, plane, gt_mm)
    if not target_paths:
        return _FALLBACK_REASONING

    anterior_pos: float | None = None
    posterior_pos: float | None = None
    anterior_paths: list[tuple[tuple[int, ...], int]] = []
    posterior_paths: list[tuple[tuple[int, ...], int]] = []
    if fetch_positions:
        anterior_pos = min((p for p in fetch_positions if p < gt_mm), default=None)
        posterior_pos = max((p for p in fetch_positions if p > gt_mm), default=None)
        if anterior_pos is not None:
            anterior_paths = _region_paths_at_position(atlas, plane, anterior_pos)
        if posterior_pos is not None:
            posterior_paths = _region_paths_at_position(atlas, plane, posterior_pos)

    structures = atlas.structures
    rendered = ""
    for offset in range(20):  # 20 is generous; brainglobe trees are ~5-10 deep
        target_names = _names_at_depth(target_paths, offset, structures)
        if not target_names:
            continue
        target_set = set(target_names)

        anterior_extra: list[str] = []
        if anterior_paths:
            anterior_extra = [
                n for n in _names_at_depth(anterior_paths, offset, structures)
                if n not in target_set
            ]
        posterior_extra: list[str] = []
        if posterior_paths:
            posterior_extra = [
                n for n in _names_at_depth(posterior_paths, offset, structures)
                if n not in target_set
            ]

        parts: list[str] = [
            f"Target {label} slice contains: " + "; ".join(target_names) + ".",
        ]
        if anterior_pos is not None and anterior_extra:
            parts.append(
                f"Anterior {label} atlas at {anterior_pos:.2f} mm contains: "
                + "; ".join(anterior_extra)
                + ", which are not present in the target slice."
            )
        if posterior_pos is not None and posterior_extra:
            parts.append(
                f"Posterior {label} atlas at {posterior_pos:.2f} mm contains: "
                + "; ".join(posterior_extra)
                + ", which are not present in the target slice."
            )
        parts.append(f"Therefore, the target {label} slice is at {gt_mm:.2f} mm.")
        rendered = "\n\n".join(parts)
        if len(rendered) <= char_budget:
            return rendered

    # Fully collapsed and still doesn't fit: return whatever the last (root-most)
    # rendering produced — it will be very short anyway.
    return rendered or _FALLBACK_REASONING


# ---------------------------------------------------------------------------
# SectionSpec — mirrors what generate_trace needs plus a few corpus fields
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class SectionSpec:
    """All fields needed to generate + serialize one synthetic row.

    Mirrors the information available from the RLVR allocation join
    (``rlvr.dataset.SingleSliceExample``) plus the few extra fields the
    langslice-native JSONL schema requires.
    """
    plane: str
    dataset: str
    subject_id: str
    section_id: str
    image_path: str          # repo-relative path to the query image
    ground_truth_mm: float
    atlas_name: str
    atlas_version: str


# ---------------------------------------------------------------------------
# Row builder — CanonicalTrace → langslice-native JSONL dict
# ---------------------------------------------------------------------------

def _canonical_atlas_path(atlas_name: str, plane: str, position_mm: float) -> str:
    """Canonical repo-relative atlas image path (without the data/atlas/ prefix)."""
    return f"models/langslice-gemma-4/data/atlas/{atlas_name}/{plane}/{position_mm:.2f}mm.jpg"


def _build_native_row(
    spec: SectionSpec,
    *,
    tool_steps_included: bool,
    ground_truth_mm: float,
    tool_step_positions: list[list[float]],
    atlas_name: str,
    plane: str,
    atlas_version: str,
    reasoning: str,
) -> dict[str, Any]:
    """Construct the langslice-native JSONL dict from synthesized trace components.

    For ``sft_answer_only`` (``tool_steps_included=False``) the trace is just
    the terminal submit step.  For ``sft_full`` (``tool_steps_included=True``)
    the trace includes all ToolStep fetch_atlas calls followed by submit.

    Image paths in tool_result use the canonical
    ``models/langslice-gemma-4/data/atlas/…`` prefix so they resolve correctly
    from any corpus root that sets ``dataset_root`` to the repo root.
    """
    trace: list[dict[str, Any]] = []

    if tool_steps_included:
        for positions in tool_step_positions:
            image_paths = [_canonical_atlas_path(atlas_name, plane, p) for p in positions]
            n = len(positions)
            noun = "atlas section" if n == 1 else "atlas sections"
            formatted = ", ".join(f"{p:.2f} mm" for p in positions)
            text = f"Fetched {n} {noun}: {formatted}"
            trace.append({
                "tool_call": {
                    "name": "fetch_atlas",
                    "args": {"positions_mm": list(positions)},
                },
                "tool_result": {
                    "image_paths": image_paths,
                    "text": text,
                },
            })

    # Terminal submit step (always present, non-empty reasoning).
    trace.append({
        "submit": {
            "name": "submit_estimate",
            "args": {
                "position_mm": ground_truth_mm,
                "reasoning": reasoning,
            },
        }
    })

    user_prompt = (
        f"Determine this {plane} slice's "
        f"{'AP' if plane == 'coronal' else 'ML' if plane == 'sagittal' else 'DV'} "
        f"position in the {atlas_name} atlas."
    )

    return {
        "bucket": 1,
        "atlas_name": atlas_name,
        "atlas_version": atlas_version,
        "plane": plane,
        "subject_id": spec.subject_id,
        # Phase 6: top-level section_id enables curriculum weight keying in
        # train_sft.py's WeightedRandomSampler (prefers section_id over subject_id).
        "section_id": spec.section_id,
        "system_prompt_kind": "single_slice",
        "query_image_paths": [spec.image_path],
        "user_prompt_text": user_prompt,
        "trace": trace,
        "quality": {
            "source": "synthetic_terminal",
            "strategy": "lane_a_prefix" if tool_steps_included else "answer_only",
            "section_id": spec.section_id,
            "dataset": spec.dataset,
        },
    }


def _build_fetch_step(
    atlas_name: str,
    plane: str,
    positions: list[float],
) -> dict[str, Any]:
    """Build one completed fetch step with tool_call + tool_result."""
    image_paths = [_canonical_atlas_path(atlas_name, plane, p) for p in positions]
    n = len(positions)
    noun = "atlas section" if n == 1 else "atlas sections"
    formatted = ", ".join(f"{p:.2f} mm" for p in positions)
    text = f"Fetched {n} {noun}: {formatted}"
    return {
        "tool_call": {
            "name": "fetch_atlas",
            "args": {"positions_mm": list(positions)},
        },
        "tool_result": {
            "image_paths": image_paths,
            "text": text,
        },
    }


def _compose_thinking_signature(
    atlas_name: str,
    plane: str,
    position_mm: float,
    *,
    comparison_positions: list[float] | None = None,
) -> str:
    """Return compact thinking text grounded in visible-clues signatures."""
    try:
        from langslice_training.corpus.atlas_signature import (  # noqa: PLC0415
            compose_visible_clues,
        )
    except Exception:  # noqa: BLE001
        return _THINKING_FALLBACK

    try:
        target = compose_visible_clues(
            atlas_name,
            plane,
            position_mm,
            comparison_positions=None,
            top_k=_THINKING_SIGNATURE_TOP_K,
        ).strip()
        if not comparison_positions:
            return target or _THINKING_FALLBACK
        compared = compose_visible_clues(
            atlas_name,
            plane,
            position_mm,
            comparison_positions=list(comparison_positions),
            top_k=_THINKING_SIGNATURE_TOP_K,
        ).strip()
    except Exception:  # noqa: BLE001
        return _THINKING_FALLBACK

    if not compared:
        return target or _THINKING_FALLBACK
    return compared


def _build_thinking_signature_rows(
    spec: SectionSpec,
    *,
    tool_steps_included: bool,
    ground_truth_mm: float,
    tool_step_positions: list[list[float]],
    atlas_name: str,
    plane: str,
    atlas_version: str,
) -> list[dict[str, Any]]:
    """Build turn-split rows with per-action sibling thinking strings."""

    def _new_row(strategy: str) -> dict[str, Any]:
        user_prompt = (
            f"Determine this {plane} slice's "
            f"{'AP' if plane == 'coronal' else 'ML' if plane == 'sagittal' else 'DV'} "
            f"position in the {atlas_name} atlas."
        )
        return {
            "bucket": 1,
            "atlas_name": atlas_name,
            "atlas_version": atlas_version,
            "plane": plane,
            "subject_id": spec.subject_id,
            "section_id": spec.section_id,
            "thinking_mode": True,
            "system_prompt_kind": "single_slice",
            "query_image_paths": [spec.image_path],
            "user_prompt_text": user_prompt,
            "trace": [],
            "quality": {
                "source": "synthetic_terminal",
                "strategy": strategy,
                "section_id": spec.section_id,
                "dataset": spec.dataset,
            },
        }

    if not tool_steps_included:
        row = _new_row("answer_only")
        row["trace"] = [{
            "submit": {
                "name": "submit_estimate",
                "args": {"position_mm": ground_truth_mm},
            },
            "thinking": _compose_thinking_signature(
                atlas_name, plane, ground_truth_mm, comparison_positions=None,
            ),
        }]
        return [row]

    rows: list[dict[str, Any]] = []
    history: list[dict[str, Any]] = []
    fetched_positions: list[float] = []

    for idx, positions in enumerate(tool_step_positions):
        row = _new_row("lane_a_prefix")
        if idx == 0:
            thinking = _compose_thinking_signature(
                atlas_name, plane, ground_truth_mm, comparison_positions=None,
            )
        else:
            thinking = _compose_thinking_signature(
                atlas_name,
                plane,
                ground_truth_mm,
                comparison_positions=list(fetched_positions),
            )
        terminal = {
            "tool_call": {
                "name": "fetch_atlas",
                "args": {"positions_mm": list(positions)},
            },
            "thinking": thinking,
        }
        row["trace"] = deepcopy(history) + [terminal]
        rows.append(row)

        history.append(_build_fetch_step(atlas_name, plane, positions))
        fetched_positions.extend(float(p) for p in positions)

    final = _new_row("lane_a_prefix")
    final["trace"] = deepcopy(history) + [{
        "submit": {
            "name": "submit_estimate",
            "args": {"position_mm": ground_truth_mm},
        },
        "thinking": _compose_thinking_signature(
            atlas_name,
            plane,
            ground_truth_mm,
            comparison_positions=list(fetched_positions),
        ),
    }]
    rows.append(final)
    return rows


# ---------------------------------------------------------------------------
# Image-count cap (position-level)
#
# Unlike ``langslice_traces.trace_ops.trim_trace`` which drops *whole tool
# steps* (and frequently collapses a 2-fetch multi-turn row into 1 fetch when
# the cap bites), this helper drops *individual positions* inside each step
# while preserving the multi-step trace shape.  Tool-calling competence is
# the reason we keep a multi-turn floor at all, so keeping both fetch turns
# alive is more important than matching the empirical n_positions per step.
# ---------------------------------------------------------------------------

def _cap_canonical_tool_step_positions(
    tool_step_positions: list[list[float]],
    *,
    max_total_images: int,
    gt_position_mm: float,
) -> list[list[float]]:
    """Drop positions farthest from GT until ``sum(len(step)) ≤ max_total_images``.

    Never drops the last surviving position in any step, so every tool step
    that was present originally still has ≥1 fetched position after the cap.
    No-op when the input already fits under the cap or is empty.

    Operates on the canonical-trace positions list (the same shape stored on
    ``CanonicalTrace.tool_steps[*].call_args["positions_mm"]``), so the
    downstream ``_build_native_row`` call naturally re-renders the
    ``image_paths`` + ``text`` fields from the capped positions.
    """
    if not tool_step_positions or max_total_images <= 0:
        return tool_step_positions
    total = sum(len(s) for s in tool_step_positions)
    if total <= max_total_images:
        return tool_step_positions

    candidates: list[tuple[float, int, int]] = []
    for sidx, positions in enumerate(tool_step_positions):
        for pidx, pos in enumerate(positions):
            try:
                dist = abs(float(pos) - gt_position_mm)
            except (TypeError, ValueError):
                dist = float("inf")
            candidates.append((dist, sidx, pidx))

    candidates.sort(key=lambda c: -c[0])
    step_size = [len(s) for s in tool_step_positions]
    drops: list[set[int]] = [set() for _ in tool_step_positions]
    n_to_drop = total - max_total_images
    for _dist, sidx, pidx in candidates:
        if n_to_drop == 0:
            break
        if step_size[sidx] - len(drops[sidx]) <= 1:
            continue
        drops[sidx].add(pidx)
        n_to_drop -= 1

    return [
        [p for i, p in enumerate(positions) if i not in drops[sidx]]
        for sidx, positions in enumerate(tool_step_positions)
    ]


# ---------------------------------------------------------------------------
# Main generation function
# ---------------------------------------------------------------------------

def generate_synthetic_rows(
    specs: list[SectionSpec],
    multi_turn_floor: float,
    rng: random.Random,
    *,
    grid: list[float] | None = None,
    atlas_grid_cache_dir: Path | None = None,
    max_total_images: int | None = None,
    synthetic_reasoning_mode: SyntheticReasoningMode = "region_dump",
) -> list[dict[str, Any]]:
    """Generate synthetic JSONL rows from section specs.

    Parameters
    ----------
    specs:
        Section specs to generate rows for. In ``region_dump`` mode each spec
        produces exactly one row. In ``thinking_signature`` mode answer-only
        specs produce one row and multi-turn specs produce one row per
        supervised assistant action.
    multi_turn_floor:
        Fraction of rows rendered as ``sft_full`` (multi-turn, with tool_steps).
        The rest are rendered as ``sft_answer_only`` (trace = submit only).
        0.0 → all answer-only; 1.0 → all multi-turn.
    rng:
        Seeded ``random.Random`` instance for reproducibility.
    grid:
        Optional pre-built atlas grid (list of sorted position floats) shared
        across all specs.  When provided, ``atlas_grid_cache_dir`` is ignored.
        Typically used in unit tests to avoid touching disk.
    atlas_grid_cache_dir:
        Directory containing precomputed SigLIP embedding caches whose basenames
        encode available atlas positions (``<atlas>_<plane>.pt``).  Used to load
        the grid when ``grid`` is None.  If both are None the generator falls
        back to a minimal synthetic grid centred on the ground-truth position.
    max_total_images:
        Optional VRAM cap on atlas images in multi-turn rows.  Positions
        farthest from GT are dropped first; multi-step structure is preserved.
        ``None`` (default) leaves the empirical distribution untouched.
        Answer-only rows are unaffected (they have no atlas images).
    synthetic_reasoning_mode:
        ``"region_dump"`` (default) keeps the legacy one-row-per-spec behavior
        with submit-step ``reasoning`` text. ``"thinking_signature"`` emits
        sibling ``thinking`` fields and splits multi-turn traces into one row
        per assistant action.

    Returns
    -------
    list of dicts ready to be written as JSONL via ``write_synthetic_jsonl``.
    """
    from langslice_traces.generator import (  # noqa: PLC0415
        generate_trace,
        load_atlas_grid,
    )

    if synthetic_reasoning_mode not in ("region_dump", "thinking_signature"):
        raise ValueError(
            "synthetic_reasoning_mode must be 'region_dump' or 'thinking_signature'"
        )

    rows: list[dict[str, Any]] = []

    # Grid cache: one list per (atlas_name, plane) pair.
    _grid_cache: dict[tuple[str, str], list[float]] = {}

    for spec in specs:
        key = (spec.atlas_name, spec.plane)

        # Resolve grid for this (atlas, plane) pair.
        if grid is not None:
            g = grid
        elif key not in _grid_cache:
            if atlas_grid_cache_dir is not None:
                try:
                    g = load_atlas_grid(
                        atlas_grid_cache_dir,
                        spec.atlas_name,
                        spec.plane,  # type: ignore[arg-type]
                    )
                    _grid_cache[key] = g
                except (FileNotFoundError, Exception):  # noqa: BLE001
                    # Fall back to a synthetic grid around GT.
                    g = _fallback_grid(spec.ground_truth_mm)
                    _grid_cache[key] = g
            else:
                g = _fallback_grid(spec.ground_truth_mm)
                _grid_cache[key] = g
        else:
            g = _grid_cache[key]

        # Generate the procedural prefix.
        canonical = generate_trace(
            image_path=spec.image_path,
            ground_truth_mm=spec.ground_truth_mm,
            plane=spec.plane,  # type: ignore[arg-type]
            atlas_name=spec.atlas_name,
            atlas_version=spec.atlas_version,
            subject_id=spec.subject_id,
            grid=g,
            strategy="lane_a_prefix",
            rng=rng,
        )
        # Decide rendering mode for this row first — multi-turn rows get the
        # contrastive "anterior/posterior atlas at X contains ..." reasoning
        # because the model sees those atlas images in the prompt and needs
        # to learn why they're NOT the target; answer-only rows don't have
        # those images so the contrastive lines would be ungrounded.
        use_full = rng.random() < multi_turn_floor
        tool_step_positions = [
            list(step.call_args["positions_mm"]) for step in canonical.tool_steps
        ]
        all_fetch_positions = (
            [p for positions in tool_step_positions for p in positions]
            if use_full else None
        )

        if use_full and max_total_images is not None and max_total_images > 0:
            capped_positions = _cap_canonical_tool_step_positions(
                tool_step_positions,
                max_total_images=max_total_images,
                gt_position_mm=spec.ground_truth_mm,
            )
        else:
            capped_positions = tool_step_positions

        if synthetic_reasoning_mode == "region_dump":
            # Compose the submit-step reasoning using brainglobe.  Cached
            # implicitly via ``_load_bg_atlas``; falls back to canned when missing.
            reasoning = _compose_synth_reasoning(
                spec.atlas_name, spec.plane, spec.ground_truth_mm,
                fetch_positions=all_fetch_positions,
            )

            # final_answer is None for lane_a_prefix; inject a synthetic one.
            from langslice_traces.schema import FinalAnswer  # noqa: PLC0415
            canonical.final_answer = FinalAnswer(
                name="submit_estimate",
                position_mm=spec.ground_truth_mm,
                reasoning=reasoning,
            )
            row = _build_native_row(
                spec,
                tool_steps_included=use_full,
                ground_truth_mm=spec.ground_truth_mm,
                tool_step_positions=capped_positions,
                atlas_name=spec.atlas_name,
                plane=spec.plane,
                atlas_version=spec.atlas_version,
                reasoning=reasoning,
            )
            rows.append(row)
        else:
            rows.extend(_build_thinking_signature_rows(
                spec,
                tool_steps_included=use_full,
                ground_truth_mm=spec.ground_truth_mm,
                tool_step_positions=capped_positions,
                atlas_name=spec.atlas_name,
                plane=spec.plane,
                atlas_version=spec.atlas_version,
            ))

    return rows


def _fallback_grid(ground_truth_mm: float, span: float = 3.0, step: float = 0.1) -> list[float]:
    """Minimal synthetic grid used when no embedding cache is available.

    Produces a uniform grid from ``gt - span/2`` to ``gt + span/2`` at
    ``step`` mm intervals, then appends the GT itself so it is always present.
    The grid is sorted and deduplicated.
    """
    lo = max(0.0, ground_truth_mm - span / 2.0)
    hi = ground_truth_mm + span / 2.0
    n = max(2, round((hi - lo) / step) + 1)
    raw = [lo + i * (hi - lo) / (n - 1) for i in range(n)]
    raw.append(round(ground_truth_mm, 2))
    seen: set[float] = set()
    out: list[float] = []
    for v in sorted(raw):
        rv = round(v, 4)
        if rv not in seen:
            seen.add(rv)
            out.append(rv)
    return out


# ---------------------------------------------------------------------------
# JSONL writer
# ---------------------------------------------------------------------------

def write_synthetic_jsonl(rows: list[dict[str, Any]], output_path: Path) -> None:
    """Append ``rows`` to ``output_path`` as newline-delimited JSON."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("a", encoding="utf-8") as fh:
        for row in rows:
            fh.write(json.dumps(row, ensure_ascii=False) + "\n")


# ---------------------------------------------------------------------------
# Allocation loader
# ---------------------------------------------------------------------------

def load_specs_from_allocation(
    allocation_root: Path,
    plane: str,
    n: int | None,
    rng: random.Random,
    *,
    repo_root: Path | None = None,
    seen_ids: set[str] | None = None,
    section_weights: dict[str, float] | None = None,
) -> list[SectionSpec]:
    """Sample ``n`` SectionSpec instances from the RLVR allocation for ``plane``.

    Reads ``allocation_root/allocations/<plane>/rlvr.jsonl`` (tombstone-aware),
    then joins with ``allocation_root/shards/<plane>/<dataset>.jsonl`` for GT
    position + image path.  Silently drops rows whose shard is missing or whose
    ``section_id`` is absent from the shard.

    When ``repo_root`` is provided, ``image_path`` values are resolved relative
    to ``repo_root``; otherwise they are left repo-relative.

    n:
        Sample size.  When ``None``, returns the full eligible pool unsampled —
        used by ``_phase_synthetic`` to combine per-plane pools before doing a
        single curriculum-aware weighted sample across the union (per-plane
        sampling would destroy multi-plane curriculum bias).
    seen_ids:
        Optional set of section_ids to exclude from the sampling pool.  Used by
        iSFT with ``--use-seen-ledger`` so synth picks unseen sections each round.
    section_weights:
        Optional ``{section_id: weight}`` mapping.  When provided and ``n`` is
        not ``None``, samples ``n`` SectionSpec instances **without replacement**
        weighted by these values (sections with higher weights are more likely
        to be picked).  When None, samples uniformly.  Used by iSFT with
        ``--curriculum-weights-dir`` to bias synth toward weak bins after a
        per-bin MAE update.
    """
    from langslice_training.rl.common.dataset import (  # noqa: PLC0415
        _index_shard_by_section,
        _resolve_active_allocation,
    )

    alloc_path = allocation_root / "allocations" / plane / "rlvr.jsonl"
    if not alloc_path.is_file():
        return []

    active = _resolve_active_allocation(plane, allocation_root)
    if not active:
        return []

    # Group by dataset so we open each shard at most once.
    by_dataset: dict[str, list[dict[str, Any]]] = {}
    with alloc_path.open("r", encoding="utf-8") as fh:
        for raw in fh:
            stripped = raw.strip()
            if not stripped:
                continue
            entry = json.loads(stripped)
            sid = entry["section_id"]
            if sid not in active:
                continue
            ds = entry.get("dataset")
            if not ds:
                continue
            by_dataset.setdefault(ds, []).append(entry)

    specs: list[SectionSpec] = []
    for dataset, alloc_entries in by_dataset.items():
        shard_path = allocation_root / "shards" / plane / f"{dataset}.jsonl"
        if not shard_path.is_file():
            continue
        shard_index = _index_shard_by_section(shard_path)
        for entry in alloc_entries:
            sid = entry["section_id"]
            row = shard_index.get(sid)
            if row is None:
                continue
            image_path = str(row.get("image_path", ""))
            if not image_path:
                continue
            atlas_name = str(row.get("atlas", ""))
            # Best-effort atlas_version — fall back to empty string.
            atlas_version = str(row.get("atlas_version", ""))
            if seen_ids is not None and sid in seen_ids:
                continue
            specs.append(SectionSpec(
                plane=plane,
                dataset=dataset,
                subject_id=str(row.get("subject_id", sid)),
                section_id=sid,
                image_path=image_path,
                ground_truth_mm=float(row["position_mm"]),
                atlas_name=atlas_name,
                atlas_version=atlas_version,
            ))

    if n is None:
        return list(specs)
    return weighted_sample_without_replacement(specs, n, rng, section_weights)


def weighted_sample_without_replacement(
    specs: list[SectionSpec],
    n: int,
    rng: random.Random,
    section_weights: dict[str, float] | None,
) -> list[SectionSpec]:
    """Sample ``n`` distinct specs, weighted by ``section_weights`` when given.

    Uses Efraimidis-Spirakis A-Res: each item gets a key ``U^(1/w)``; the
    top-``n`` keys win.  Items with higher weights have higher expected keys,
    so curriculum-weighted sections are preferentially picked without
    duplicates.  When ``section_weights`` is ``None``, falls back to uniform
    ``rng.sample`` (slightly faster and well-tested).
    """
    if len(specs) <= n:
        return list(specs)
    if section_weights is None:
        return rng.sample(specs, n)
    keys: list[float] = []
    for spec in specs:
        w = max(1e-9, float(section_weights.get(spec.section_id, 1.0)))
        u = max(1e-12, rng.random())
        keys.append(math.pow(u, 1.0 / w))
    top_idx = sorted(range(len(specs)), key=lambda i: -keys[i])[:n]
    return [specs[i] for i in top_idx]
