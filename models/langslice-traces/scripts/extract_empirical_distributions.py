"""Extract empirical distributions of trace shape from the real SFT corpus.

One-off extractor. Reads the Gemini-authored SFT JSONL and dumps a paste-block
that becomes ``models/langslice-gemma-4/training/langslice_traces/_empirical.py``.
Downstream synthesis
(Lane A prefix generator) samples from these frozen distributions so its output
is statistically indistinguishable from the real corpus on the metrics a code
review flagged: anchor-to-GT correlation, position roundness, varied fetch
counts.

Computes:
- Per-step number-of-fetches (n positions per fetch_atlas call) distribution
  for step 0 (broad) and step 1 (narrow).
- Step-span distribution (max - min positions) in mm.
- Step-1 center-offset from the final submit position.
- Per-position roundness class probabilities (int / half / tenth / raw).
- Number-of-tool-steps distribution (1, 2, 3 ...).
- Step-0 anchor calibration: σ for the slope-1 model
  ``center_0 = gt + N(0, σ)``, computed as the population std of
  ``(center_0 - gt)``. Reported overall + per-plane. The observed Pearson r
  between center_0 and gt is reported alongside as a diagnostic.

Usage::

    python models/langslice-traces/scripts/extract_empirical_distributions.py \
        --corpus models/langslice-gemma-4/data/sft_examples.jsonl \
        --output _local/empirical_dist_dump.py

Standalone — no langslice_traces import. The output is meant to be diffed
against the live ``_empirical.py`` and copy-pasted forward when the corpus is
materially changed.
"""

from __future__ import annotations

import argparse
import json
import math
import statistics
import sys
from collections import Counter, defaultdict
from datetime import date
from pathlib import Path
from typing import Any

_REPO = Path(__file__).resolve().parents[1]
_DEFAULT_CORPUS = _REPO / "models" / "langslice-gemma-4" / "data" / "sft_examples.jsonl"
_DEFAULT_OUTPUT = _REPO / "_local" / "empirical_dist_dump.py"


# -- corpus loading ------------------------------------------------------------


def load_corpus(path: Path) -> list[dict[str, Any]]:
    """Read the JSONL corpus into a list of row dicts."""
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as fh:
        for line_no, line in enumerate(fh, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError as exc:
                raise RuntimeError(
                    f"failed to parse JSON at {path}:{line_no}: {exc}"
                ) from exc
    return rows


# -- step extraction -----------------------------------------------------------


def _tool_steps_and_submit(
    trace: list[dict[str, Any]],
) -> tuple[list[list[float]], float | None]:
    """Split a trace into (list of positions-per-tool-step, submit position).

    Returns the positions list per fetch_atlas call (in order) and the final
    submit ``position_mm`` (or None if missing).
    """
    tool_positions: list[list[float]] = []
    submit_position: float | None = None
    for step in trace:
        if "tool_call" in step:
            call = step["tool_call"]
            args = call.get("args") or {}
            positions = args.get("positions_mm") or []
            # Ensure they are floats.
            tool_positions.append([float(p) for p in positions])
        elif "submit" in step:
            sub_args = step["submit"].get("args") or {}
            if "position_mm" in sub_args:
                submit_position = float(sub_args["position_mm"])
    return tool_positions, submit_position


# -- roundness classification --------------------------------------------------

_TOL = 1e-6


def classify_roundness(p: float) -> str:
    """Return 'int' / 'half' / 'tenth' / 'raw' — tightest matching class."""
    if abs(p - round(p)) < _TOL:
        return "int"
    if abs(p * 2 - round(p * 2)) < _TOL:
        return "half"
    if abs(p * 10 - round(p * 10)) < _TOL:
        return "tenth"
    return "raw"


# -- distribution helpers ------------------------------------------------------


def normalize_counts(counts: dict[Any, int]) -> list[tuple[Any, float]]:
    """Convert raw counts to sorted (value, probability) list."""
    total = sum(counts.values())
    if total == 0:
        return []
    return [(k, counts[k] / total) for k in sorted(counts.keys())]


def bin_floats(values: list[float], n_bins: int, lo: float, hi: float) -> dict[float, int]:
    """Histogram floats into n_bins between [lo, hi]. Bin centers are returned
    as keys. Values outside [lo, hi] are clamped to the nearest bin."""
    if n_bins <= 0:
        return {}
    width = (hi - lo) / n_bins
    counts: Counter[float] = Counter()
    for v in values:
        if v <= lo:
            idx = 0
        elif v >= hi:
            idx = n_bins - 1
        else:
            idx = int((v - lo) / width)
            if idx >= n_bins:
                idx = n_bins - 1
        center = lo + (idx + 0.5) * width
        # Round bin center for readability.
        counts[round(center, 4)] += 1
    return dict(counts)


def pearson_r(xs: list[float], ys: list[float]) -> float:
    """Pearson correlation coefficient."""
    n = len(xs)
    if n < 2:
        return 0.0
    mx = sum(xs) / n
    my = sum(ys) / n
    num = sum((x - mx) * (y - my) for x, y in zip(xs, ys))
    dx = math.sqrt(sum((x - mx) ** 2 for x in xs))
    dy = math.sqrt(sum((y - my) ** 2 for y in ys))
    if dx == 0 or dy == 0:
        return 0.0
    return num / (dx * dy)


def linear_regression(xs: list[float], ys: list[float]) -> tuple[float, float, float]:
    """OLS slope, intercept, residual std (population std, ddof=0).

    Diagnostic-only here: we report the OLS slope to confirm it's ~1.0, which
    is what makes the slope-1 sigma (see ``slope_one_residual_std``) the right
    parameter for downstream sampling.
    """
    n = len(xs)
    mx = sum(xs) / n
    my = sum(ys) / n
    num = sum((x - mx) * (y - my) for x, y in zip(xs, ys))
    den = sum((x - mx) ** 2 for x in xs)
    if den == 0:
        return 0.0, my, statistics.pstdev(ys) if n >= 1 else 0.0
    slope = num / den
    intercept = my - slope * mx
    residuals = [y - (slope * x + intercept) for x, y in zip(xs, ys)]
    resid_std = math.sqrt(sum(r * r for r in residuals) / n) if n >= 1 else 0.0
    return slope, intercept, resid_std


def slope_one_residual_std(xs: list[float], ys: list[float]) -> float:
    """Population std of ``(y - x)`` — the σ for the slope-1 noise model.

    Downstream synthesis samples ``center_0 = gt + N(0, σ)``; this is the σ
    that reproduces that exact shape on the corpus. Equivalent to OLS
    residual std only when the empirical slope is ~1.0, which our corpus
    confirms (1.002).
    """
    n = len(xs)
    if n == 0:
        return 0.0
    diffs = [y - x for x, y in zip(xs, ys)]
    return statistics.pstdev(diffs)


# -- formatting ----------------------------------------------------------------


def _format_int_dist(label: str, dist: list[tuple[int, float]]) -> str:
    body = ",\n".join(f"    ({k}, {v:.6f})" for k, v in dist)
    return f"{label}: list[tuple[int, float]] = [\n{body},\n]"


def _format_float_dist(label: str, dist: list[tuple[float, float]]) -> str:
    body = ",\n".join(f"    ({k:.4f}, {v:.6f})" for k, v in dist)
    return f"{label}: list[tuple[float, float]] = [\n{body},\n]"


def _format_roundness(label: str, classes: dict[str, float]) -> str:
    # Always emit all four keys, in canonical order, even if some are zero.
    order = ["int", "half", "tenth", "raw"]
    pieces = ", ".join(f'"{k}": {classes.get(k, 0.0):.6f}' for k in order)
    return f"{label}: dict[str, float] = {{{pieces}}}"


def build_paste_block(
    *,
    p_n_steps: list[tuple[int, float]],
    p_nfetch_step0: list[tuple[int, float]],
    p_nfetch_step1: list[tuple[int, float]],
    p_span_step0: list[tuple[float, float]],
    p_span_step1: list[tuple[float, float]],
    p_center_offset_step1: list[tuple[float, float]],
    p_roundness_step0: dict[str, float],
    p_roundness_step1: dict[str, float],
    sigma_anchor_mm: float,
    observed_r: float,
    observed_int_rate_step0: float,
    n_rows: int,
    extraction_date: str,
    sigma_anchor_per_plane: dict[str, float],
    r_per_plane: dict[str, float],
    corpus_path: str,
) -> str:
    """Assemble the full ``_empirical.py``-shaped paste-block."""
    lines: list[str] = []
    lines.append(
        f'"""Empirical distributions of real Gemini-authored trace shape.\n'
        f"\n"
        f"Frozen at extraction time {extraction_date} from\n"
        f"{corpus_path} (n={n_rows}).\n"
        f"\n"
        f"Regenerate by running models/langslice-traces/scripts/extract_empirical_distributions.py if the\n"
        f"SFT corpus is materially changed.\n"
        f"\n"
        f"All distributions are lists of (value, weight) tuples where weights sum to 1.0.\n"
        f"Use the sample() helper for reproducible draws via a passed random.Random.\n"
        f'"""'
    )
    lines.append("")
    lines.append("from __future__ import annotations")
    lines.append("")
    lines.append("import random")
    lines.append("from typing import TypeVar")
    lines.append("")
    lines.append('T = TypeVar("T")')
    lines.append("")
    lines.append("# Number of tool steps per trace (after dropping the terminal submit step).")
    lines.append(_format_int_dist("P_N_TOOL_STEPS", p_n_steps))
    lines.append("")
    lines.append("# Number of positions per fetch, step 0 (broad).")
    lines.append(_format_int_dist("P_NFETCH_STEP0", p_nfetch_step0))
    lines.append("")
    lines.append("# Number of positions per fetch, step 1 (narrow).")
    lines.append(_format_int_dist("P_NFETCH_STEP1", p_nfetch_step1))
    lines.append("")
    lines.append("# Span (max - min) per step, in mm. Bins of (span_mm, weight).")
    lines.append(_format_float_dist("P_SPAN_STEP0", p_span_step0))
    lines.append("")
    lines.append(_format_float_dist("P_SPAN_STEP1", p_span_step1))
    lines.append("")
    lines.append("# Signed offset between step-1 center and submit position (mm).")
    lines.append(_format_float_dist("P_CENTER_OFFSET_STEP1", p_center_offset_step1))
    lines.append("")
    lines.append("# Roundness class probabilities per step.")
    lines.append(_format_roundness("P_ROUNDNESS_STEP0", p_roundness_step0))
    lines.append(_format_roundness("P_ROUNDNESS_STEP1", p_roundness_step1))
    lines.append("")
    lines.append(
        "# Sigma for the step-0 anchor in the slope-1 model: sampling"
    )
    lines.append(
        "# ``center_0 = gt + N(0, SIGMA_ANCHOR_MM)`` reproduces the observed"
    )
    lines.append(
        "# distribution of step-0 fetch centers around GT (population std of"
    )
    lines.append(
        f"# center_0 - gt; observed Pearson r ~ {observed_r:.3f} on the corpus)."
    )
    lines.append(f"SIGMA_ANCHOR_MM: float = {sigma_anchor_mm:.6f}")
    lines.append("")
    lines.append("# Per-plane breakdown of the same sigma (planes have different extents).")
    plane_sigma_pieces = ", ".join(
        f'"{k}": {v:.6f}' for k, v in sorted(sigma_anchor_per_plane.items())
    )
    lines.append(
        f"SIGMA_ANCHOR_MM_BY_PLANE: dict[str, float] = {{{plane_sigma_pieces}}}"
    )
    lines.append("")
    lines.append("# Diagnostic - what the source corpus actually shows.")
    lines.append(f"OBSERVED_R_STEP0_GT: float = {observed_r:.6f}")
    r_plane_pieces = ", ".join(
        f'"{k}": {v:.6f}' for k, v in sorted(r_per_plane.items())
    )
    lines.append(
        f"OBSERVED_R_STEP0_GT_BY_PLANE: dict[str, float] = {{{r_plane_pieces}}}"
    )
    lines.append(
        f"OBSERVED_INTEGER_RATE_STEP0: float = {observed_int_rate_step0:.6f}"
    )
    lines.append(f"OBSERVED_N: int = {n_rows}")
    lines.append("")
    lines.append("")
    lines.append("def sample(rng: random.Random, dist: list[tuple[T, float]]) -> T:")
    lines.append('    """Draw one value from a weighted (value, prob) distribution.')
    lines.append("")
    lines.append(
        '    Uses ``rng.random()`` for reproducibility; weights must sum to ~1.0 but'
    )
    lines.append(
        '    we use cumulative comparison so non-normalized inputs still work.'
    )
    lines.append('    """')
    lines.append("    total = sum(w for _, w in dist)")
    lines.append("    target = rng.random() * total")
    lines.append("    cumulative = 0.0")
    lines.append("    for value, weight in dist:")
    lines.append("        cumulative += weight")
    lines.append("        if cumulative >= target:")
    lines.append("            return value")
    lines.append("    return dist[-1][0]  # numerical-precision fallback")
    lines.append("")
    lines.append("")
    lines.append("def sample_roundness(rng: random.Random, dist: dict[str, float]) -> str:")
    lines.append(
        '    """Draw a roundness class (\'int\'|\'half\'|\'tenth\'|\'raw\') from a dict dist."""'
    )
    lines.append("    return sample(rng, [(k, v) for k, v in dist.items()])")
    lines.append("")
    return "\n".join(lines)


# -- main pipeline -------------------------------------------------------------


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument(
        "--corpus",
        type=Path,
        default=_DEFAULT_CORPUS,
        help=f"Path to SFT JSONL (default: {_DEFAULT_CORPUS})",
    )
    ap.add_argument(
        "--output",
        type=Path,
        default=_DEFAULT_OUTPUT,
        help=f"Where to write the paste-block (default: {_DEFAULT_OUTPUT})",
    )
    args = ap.parse_args(argv)

    rows = load_corpus(args.corpus)
    print(f"Loaded {len(rows)} rows from {args.corpus}", file=sys.stderr)

    n_steps_counter: Counter[int] = Counter()
    nfetch_step: list[Counter[int]] = [Counter(), Counter()]
    span_step: list[list[float]] = [[], []]
    center_offset_step1: list[float] = []
    roundness_step: list[Counter[str]] = [Counter(), Counter()]

    # For anchor calibration: center_0 vs gt (submit position).
    gts: list[float] = []
    centers_0: list[float] = []
    plane_gts: dict[str, list[float]] = defaultdict(list)
    plane_centers_0: dict[str, list[float]] = defaultdict(list)

    near_zero_step1_offset = 0

    for row in rows:
        trace = row.get("trace") or []
        plane = row.get("plane", "unknown")
        tool_positions, submit_pos = _tool_steps_and_submit(trace)
        n_steps_counter[len(tool_positions)] += 1

        for step_idx, positions in enumerate(tool_positions):
            if step_idx >= 2:
                break  # Track only step 0 and step 1.
            if positions:
                nfetch_step[step_idx][len(positions)] += 1
                span_step[step_idx].append(max(positions) - min(positions))
                for p in positions:
                    roundness_step[step_idx][classify_roundness(p)] += 1

        # Step-1 offset wrt submit.
        if (
            len(tool_positions) >= 2
            and submit_pos is not None
            and tool_positions[1]
        ):
            center_1 = sum(tool_positions[1]) / len(tool_positions[1])
            offset = center_1 - submit_pos
            center_offset_step1.append(offset)
            if abs(offset) < 0.01:
                near_zero_step1_offset += 1

        # Step-0 anchor: align mean(step0.positions) with submit position (gt).
        if (
            tool_positions
            and tool_positions[0]
            and submit_pos is not None
        ):
            center_0 = sum(tool_positions[0]) / len(tool_positions[0])
            gts.append(submit_pos)
            centers_0.append(center_0)
            plane_gts[plane].append(submit_pos)
            plane_centers_0[plane].append(center_0)

    # ---- assemble distributions ----

    p_n_steps = normalize_counts(dict(n_steps_counter))
    p_nfetch_step0 = normalize_counts(dict(nfetch_step[0]))
    p_nfetch_step1 = normalize_counts(dict(nfetch_step[1]))

    # Span histograms.
    span0_max = max(span_step[0]) if span_step[0] else 1.0
    span1_max = max(span_step[1]) if span_step[1] else 1.0
    # Round up to next 0.5 for niceness.
    span0_hi = math.ceil(span0_max * 2) / 2
    span1_hi = math.ceil(span1_max * 2) / 2
    span0_counts = bin_floats(span_step[0], n_bins=10, lo=0.0, hi=max(span0_hi, 0.5))
    span1_counts = bin_floats(span_step[1], n_bins=10, lo=0.0, hi=max(span1_hi, 0.5))
    p_span_step0 = normalize_counts(span0_counts)
    p_span_step1 = normalize_counts(span1_counts)

    # Step-1 center offset histogram (~20 bins between -2 and +2).
    offset_lo = -2.0
    offset_hi = 2.0
    offset_counts = bin_floats(
        center_offset_step1, n_bins=20, lo=offset_lo, hi=offset_hi
    )
    p_center_offset_step1 = normalize_counts(offset_counts)

    # Roundness fractions.
    def _normalize_roundness(c: Counter[str]) -> dict[str, float]:
        total = sum(c.values())
        if total == 0:
            return {"int": 0.0, "half": 0.0, "tenth": 0.0, "raw": 0.0}
        return {k: c.get(k, 0) / total for k in ["int", "half", "tenth", "raw"]}

    p_round0 = _normalize_roundness(roundness_step[0])
    p_round1 = _normalize_roundness(roundness_step[1])

    # ---- Pearson r + sigma anchor ----

    observed_r = pearson_r(gts, centers_0)
    # Sigma for the slope-1 model: population std of (center_0 - gt). That's
    # exactly the σ downstream code uses when sampling
    # ``center_0 = gt + N(0, σ)``. We also fit OLS to confirm the empirical
    # slope is ~1.0; if it weren't, the slope-1 sigma would no longer be the
    # right parameter.
    slope, intercept, _resid_std = linear_regression(gts, centers_0)
    sigma_anchor_mm = slope_one_residual_std(gts, centers_0)

    sigma_per_plane: dict[str, float] = {}
    r_per_plane: dict[str, float] = {}
    for plane in sorted(plane_gts):
        if len(plane_gts[plane]) < 5:
            continue
        r_per_plane[plane] = pearson_r(plane_gts[plane], plane_centers_0[plane])
        sigma_per_plane[plane] = slope_one_residual_std(
            plane_gts[plane], plane_centers_0[plane]
        )

    # Step-0 integer rate (observed).
    total_step0_positions = sum(roundness_step[0].values())
    observed_int_rate_step0 = (
        roundness_step[0].get("int", 0) / total_step0_positions
        if total_step0_positions else 0.0
    )

    # ---- diagnostics to stderr ----

    print(
        f"n_steps: {dict(n_steps_counter)}",
        file=sys.stderr,
    )
    print(
        f"step0 nfetch: {dict(nfetch_step[0])}",
        file=sys.stderr,
    )
    print(
        f"step1 nfetch: {dict(nfetch_step[1])}",
        file=sys.stderr,
    )
    print(
        f"span0 max={max(span_step[0]) if span_step[0] else 0:.3f}mm "
        f"mean={sum(span_step[0]) / len(span_step[0]) if span_step[0] else 0:.3f}mm",
        file=sys.stderr,
    )
    print(
        f"span1 max={max(span_step[1]) if span_step[1] else 0:.3f}mm "
        f"mean={sum(span_step[1]) / len(span_step[1]) if span_step[1] else 0:.3f}mm",
        file=sys.stderr,
    )
    print(
        f"roundness step0: {p_round0}",
        file=sys.stderr,
    )
    print(
        f"roundness step1: {p_round1}",
        file=sys.stderr,
    )
    print(
        f"r(center_0, gt) = {observed_r:.4f} (n={len(gts)})",
        file=sys.stderr,
    )
    print(
        f"sigma_anchor_mm = {sigma_anchor_mm:.4f} (slope={slope:.3f}, intercept={intercept:.3f})",
        file=sys.stderr,
    )
    for plane in sorted(sigma_per_plane):
        print(
            f"  {plane}: r={r_per_plane[plane]:.4f} sigma={sigma_per_plane[plane]:.4f} "
            f"(n={len(plane_gts[plane])})",
            file=sys.stderr,
        )
    print(
        f"step-1 near-zero offset rate (|off|<0.01): "
        f"{near_zero_step1_offset / len(center_offset_step1) if center_offset_step1 else 0:.4f}",
        file=sys.stderr,
    )

    # ---- assemble paste-block ----

    # Prefer a repo-relative path in the generated docstring so it's
    # readable / portable; fall back to the raw value if outside the repo.
    try:
        corpus_for_docstring = str(
            Path(args.corpus).resolve().relative_to(_REPO).as_posix()
        )
    except ValueError:
        corpus_for_docstring = str(args.corpus)

    paste_block = build_paste_block(
        p_n_steps=p_n_steps,
        p_nfetch_step0=p_nfetch_step0,
        p_nfetch_step1=p_nfetch_step1,
        p_span_step0=p_span_step0,
        p_span_step1=p_span_step1,
        p_center_offset_step1=p_center_offset_step1,
        p_roundness_step0=p_round0,
        p_roundness_step1=p_round1,
        sigma_anchor_mm=sigma_anchor_mm,
        observed_r=observed_r,
        observed_int_rate_step0=observed_int_rate_step0,
        n_rows=len(rows),
        extraction_date=str(date.today()),
        sigma_anchor_per_plane=sigma_per_plane,
        r_per_plane=r_per_plane,
        corpus_path=corpus_for_docstring,
    )

    # ---- write output + stdout ----

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(paste_block, encoding="utf-8")
    print(f"Wrote paste-block to {args.output}", file=sys.stderr)

    print(paste_block)

    return 0


if __name__ == "__main__":
    sys.exit(main())
