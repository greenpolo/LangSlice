"""Trace step manipulation primitives (trim, etc.)."""

from __future__ import annotations


def trim_trace(
    trace: list[dict],
    *,
    max_fetch_calls: int = 2,
    max_total_images: int = 12,
) -> tuple[list[dict], dict]:
    """Distill a trace by keeping only the fetch_atlas calls closest to the
    final submission. Preserves alternation and never alters the submit.

    Pruning policy (per codex's distillation spec):
      * keep at most `max_fetch_calls` fetch_atlas turns;
      * cap total tool_result images at `max_total_images`;
      * prefer fetches whose positions are nearest to the final submission;
      * leave the submit step (and its reasoning) untouched;
      * ensure tool_call/tool_result alternation is preserved (we keep whole
        steps, never partial).
    """
    if not trace:
        return trace, {"trim_reason": "empty"}
    submit_step = trace[-1]
    if "submit" not in submit_step:
        return trace, {"trim_reason": "no_submit"}

    submitted_pos = float(submit_step["submit"]["args"]["position_mm"])

    # Collect fetch_atlas steps with their (closest-fetched-pos -> submit) distance.
    indexed_fetches: list[tuple[float, int, dict, int]] = []
    original_total_images = 0
    for i, step in enumerate(trace[:-1]):
        if "tool_call" not in step:
            continue
        positions = (step["tool_call"].get("args") or {}).get("positions_mm") or []
        n_images = len(step.get("tool_result", {}).get("image_paths") or [])
        original_total_images += n_images
        if positions:
            try:
                min_dist = min(abs(float(p) - submitted_pos) for p in positions)
            except (TypeError, ValueError):
                min_dist = float("inf")
        else:
            min_dist = float("inf")
        indexed_fetches.append((min_dist, i, step, n_images))

    if not indexed_fetches:
        return trace, {
            "trim_reason": "no_fetches",
            "original_num_fetch_calls": 0,
            "kept_num_fetch_calls": 0,
            "original_num_images": 0,
            "kept_num_images": 0,
            "final_position_mm": submitted_pos,
            "min_distance_from_kept_atlas_position_to_final_mm": None,
        }

    # Greedy: nearest fetches first, respect both caps. Always keep at least
    # the closest fetch (so the trimmed trace still demonstrates tool use).
    by_distance = sorted(indexed_fetches, key=lambda t: t[0])
    kept_indices: set[int] = set()
    kept_images = 0
    for _min_dist, idx, _step, n_images in by_distance:
        if len(kept_indices) >= max_fetch_calls:
            break
        if kept_indices and kept_images + n_images > max_total_images:
            continue
        kept_indices.add(idx)
        kept_images += n_images

    overall_min = min(
        f[0] for f in indexed_fetches if f[1] in kept_indices
    ) if kept_indices else None

    # Reassemble in chronological order — order of tool calls matters for the
    # alternation invariant the trainer's render.py expects.
    trimmed = [step for i, step in enumerate(trace[:-1]) if i in kept_indices]
    trimmed.append(submit_step)

    n_kept = len(kept_indices)
    n_orig = len(indexed_fetches)
    return trimmed, {
        "trim_reason": (
            "no_op" if n_kept == n_orig and kept_images == original_total_images
            else "pruned"
        ),
        "original_num_fetch_calls": n_orig,
        "kept_num_fetch_calls": n_kept,
        "original_num_images": original_total_images,
        "kept_num_images": kept_images,
        "final_position_mm": round(submitted_pos, 4),
        "min_distance_from_kept_atlas_position_to_final_mm": (
            round(overall_min, 4) if overall_min is not None else None
        ),
    }
