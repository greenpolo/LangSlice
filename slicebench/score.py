"""Scoring + aggregation for a SliceBench run.

Reads ``predictions.jsonl`` from a run dir, computes plane-relative error
(no tolerances), and emits ``scored.jsonl`` + ``summary.json``.

PRIMARY METRIC is plane-relative error percentage:
    err_pct_of_plane = abs_err_mm / plane_extent_mm * 100   (clipped to [0,100])

Why pct-of-plane and not mm? Mouse coronal extent is ~13 mm, rat coronal ~17 mm,
sagittal ~11 mm, horizontal ~8 mm. A "1 mm error" means very different things
across planes — 7% of mouse coronal but 12% of horizontal. Pct-of-plane is the
only comparable summary across planes/species/atlases. Use mm only when you
specifically need absolute distances (e.g., per-plane comparisons within one
atlas, or when comparing against literature that uses mm).

Both abs_err_mm and accuracy_pct (100 - err_pct_of_plane) are still emitted to
``summary.json`` for backwards compatibility with curriculum.weights.read_per_bin_mae
and any downstream consumer.

Sagittal positions are canonicalized to a single hemisphere before computing
err so mirror flips don't show as errors.
"""

from __future__ import annotations

import argparse
import json
import statistics
import sys
from collections import Counter, defaultdict
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Iterable

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from langslice_harness.harness.estimation.trace_collection import canonicalize_positions


# ──────────────────────────────────────────────────────────────────────────
# Per-row scoring
# ──────────────────────────────────────────────────────────────────────────


def _abs_err_mm(plane: str, atlas: str, truth_mm: float, predicted_mm: float) -> float:
    """Plane-aware absolute error. Sagittal is canonicalized per hemisphere first."""
    if plane == "sagittal":
        truth_canon = canonicalize_positions([truth_mm], atlas, plane)[0]  # type: ignore[index]
        pred_canon = canonicalize_positions([predicted_mm], atlas, plane)[0]  # type: ignore[index]
        return abs(truth_canon - pred_canon)
    return abs(truth_mm - predicted_mm)


def _accuracy_pct(abs_err_mm: float, plane_extent_mm: float) -> float:
    if plane_extent_mm <= 0:
        return 0.0
    return max(0.0, 100.0 * (1.0 - abs_err_mm / plane_extent_mm))


def _err_pct_of_plane(abs_err_mm: float, plane_extent_mm: float) -> float:
    """Plane-relative error as a percentage of plane extent (the PRIMARY metric).

    Returns abs_err / plane_extent * 100, clipped to [0, 100]. 0% = perfect,
    100% = worst (predicted at opposite end of plane). Mathematically:
    err_pct_of_plane + accuracy_pct = 100.
    """
    if plane_extent_mm <= 0:
        return 100.0
    return min(100.0, max(0.0, 100.0 * abs_err_mm / plane_extent_mm))


def _decode_tps(row: dict) -> float | None:
    """Tokens generated (output) per second of wall-clock for one rollout.

    Reads ``usage.candidates_token_count`` (the harness's name for output
    tokens, set by AgentTraceRecorder) and ``elapsed_s``. Returns None if
    either is missing or zero — common for failed rollouts (no submit, no
    completion tokens).
    """
    usage = row.get("usage") or {}
    out_tokens = usage.get("candidates_token_count")
    elapsed = row.get("elapsed_s")
    if not out_tokens or not elapsed or float(elapsed) <= 0:
        return None
    return float(out_tokens) / float(elapsed)


def _total_tps(row: dict) -> float | None:
    """Total tokens (input + output) processed per second of wall-clock."""
    usage = row.get("usage") or {}
    total = usage.get("total_token_count")
    elapsed = row.get("elapsed_s")
    if not total or not elapsed or float(elapsed) <= 0:
        return None
    return float(total) / float(elapsed)


def _classify_error(err_str: str | None) -> str:
    """Bucket a slicebench prediction error string for eval-health reporting."""
    if not err_str:
        return "ok"
    s = err_str
    if "Connection error" in s:
        return "connection"
    if "Context" in s or "context length" in s or "ContextWindowExceeded" in s:
        return "context_window"
    if "BadRequest" in s:
        return "bad_request"
    if "Timeout" in s.lower() or "timeout" in s.lower():
        return "timeout"
    if "RateLimit" in s or "429" in s:
        return "rate_limit"
    if "InternalServer" in s:
        return "server_500"
    return "other"


def _eval_health(predictions: list[dict]) -> dict:
    """Diagnose eval infrastructure issues that contaminate headline numbers.

    Specifically detects vllm cold-start bursts: a cluster of connection errors
    within ~30 seconds of run start, which slicebench will silently count as
    100%-error rows (since predicted_mm is None) and inflate the mean.

    Returns a dict with:
      - n_total, n_failed, n_ok
      - failures_by_type: counter of error categories
      - cold_start_burst: detected? if yes, count + window_sec
      - contamination_warning: human-readable flag if any concerning pattern
    """
    n_total = len(predictions)
    failed = [r for r in predictions if r.get("predicted_mm") is None]
    by_type = Counter(_classify_error(r.get("error")) for r in failed)

    # Cold-start detection: count connection errors in the first 30 sec of the run.
    cold_burst: dict = {"detected": False}
    timestamps: list[datetime] = []
    for r in predictions:
        ts = r.get("ts")
        if isinstance(ts, str):
            try:
                timestamps.append(datetime.fromisoformat(ts.replace("Z", "+00:00")))
            except ValueError:
                pass
    if timestamps:
        run_start = min(timestamps)
        early_window_sec = 30.0
        early_conn_errs = 0
        for r in failed:
            if _classify_error(r.get("error")) != "connection":
                continue
            ts = r.get("ts")
            if not isinstance(ts, str):
                continue
            try:
                t = datetime.fromisoformat(ts.replace("Z", "+00:00"))
            except ValueError:
                continue
            if (t - run_start).total_seconds() <= early_window_sec:
                early_conn_errs += 1
        if early_conn_errs >= 5:
            cold_burst = {
                "detected": True,
                "n_connection_errors_first_30s": early_conn_errs,
                "advice": (
                    "Pre-warm vllm before launching slicebench (5+ chat-completion "
                    "pings). These rows count as 100% error and will inflate "
                    "mean_err_pct. Consider rerunning."
                ),
            }

    warnings = []
    if cold_burst.get("detected"):
        warnings.append(
            f"COLD-START contamination: {cold_burst['n_connection_errors_first_30s']} "
            f"connection errors in first 30 sec — likely vllm cold-start, NOT model behavior."
        )
    if by_type.get("connection", 0) > 0 and not cold_burst.get("detected"):
        warnings.append(
            f"{by_type['connection']} connection errors scattered through the run "
            f"(not a cold-start cluster). May indicate flaky network / vllm instability."
        )
    if by_type.get("context_window", 0) >= 5:
        warnings.append(
            f"{by_type['context_window']} context-window failures — consider raising "
            f"vllm --max-model-len. These are real failures (model couldn't read prompt) "
            f"and the rows count as 100% error in headline mean."
        )

    return {
        "n_total": n_total,
        "n_failed": len(failed),
        "n_ok": n_total - len(failed),
        "failures_by_type": dict(by_type),
        "cold_start_burst": cold_burst,
        "contamination_warnings": warnings,
    }


def score_predictions(predictions: Iterable[dict]) -> list[dict]:
    """Augment each prediction row with abs_err_mm + accuracy_pct + decode_tps."""
    out = []
    for row in predictions:
        if row.get("predicted_mm") is None:
            # No-submit / parse failure. Treat as worst-case (full plane extent error).
            err = float(row["plane_extent_mm"])
        else:
            err = _abs_err_mm(
                row["plane"], row["atlas"], float(row["truth_mm"]), float(row["predicted_mm"])
            )
        scored = dict(row)
        scored["abs_err_mm"] = round(err, 4)
        scored["accuracy_pct"] = round(_accuracy_pct(err, float(row["plane_extent_mm"])), 4)
        scored["err_pct_of_plane"] = round(_err_pct_of_plane(err, float(row["plane_extent_mm"])), 4)
        decode_tps = _decode_tps(row)
        total_tps = _total_tps(row)
        scored["decode_tps"] = round(decode_tps, 2) if decode_tps is not None else None
        scored["total_tps"] = round(total_tps, 2) if total_tps is not None else None
        out.append(scored)
    return out


# ──────────────────────────────────────────────────────────────────────────
# Aggregation
# ──────────────────────────────────────────────────────────────────────────


def _stat_block(rows: list[dict]) -> dict:
    if not rows:
        return {"n": 0}
    accs = [r["accuracy_pct"] for r in rows]
    err_pcts = [r.get("err_pct_of_plane", 100.0 - r["accuracy_pct"]) for r in rows]
    errs = [r["abs_err_mm"] for r in rows]
    sorted_errs = sorted(errs)
    sorted_err_pcts = sorted(err_pcts)

    def p(q: float) -> float:
        if not sorted_errs:
            return 0.0
        idx = int(q * (len(sorted_errs) - 1))
        return sorted_errs[idx]

    def pp(q: float) -> float:
        if not sorted_err_pcts:
            return 0.0
        idx = int(q * (len(sorted_err_pcts) - 1))
        return sorted_err_pcts[idx]

    block = {
        "n": len(rows),
        # Primary metric: plane-relative error percentage (0=perfect, 100=worst).
        # Comparable across planes/atlases/species; the only fair summary.
        "mean_err_pct": round(statistics.mean(err_pcts), 3),
        "median_err_pct": round(statistics.median(err_pcts), 3),
        "p90_err_pct": round(pp(0.90), 3),
        "p99_err_pct": round(pp(0.99), 3),
        # Legacy / supporting metrics. Kept for backwards compat (curriculum
        # weights, dashboards, literature comparisons in mm).
        "mean_accuracy_pct": round(statistics.mean(accs), 3),
        "median_accuracy_pct": round(statistics.median(accs), 3),
        "mae_mm": round(statistics.mean(errs), 4),
        "median_abs_err_mm": round(statistics.median(errs), 4),
        "p90_abs_err_mm": round(p(0.90), 4),
        "p99_abs_err_mm": round(p(0.99), 4),
    }

    # Throughput stats — drop rows without usage / elapsed (e.g., failed rollouts)
    decode_tps = [r["decode_tps"] for r in rows if r.get("decode_tps") is not None]
    total_tps = [r["total_tps"] for r in rows if r.get("total_tps") is not None]
    if decode_tps:
        sorted_dtps = sorted(decode_tps)
        block["mean_decode_tps"] = round(statistics.mean(decode_tps), 2)
        block["median_decode_tps"] = round(statistics.median(decode_tps), 2)
        block["p10_decode_tps"] = round(sorted_dtps[int(0.10 * (len(sorted_dtps) - 1))], 2)
    if total_tps:
        block["mean_total_tps"] = round(statistics.mean(total_tps), 2)
        block["median_total_tps"] = round(statistics.median(total_tps), 2)
    return block


def _coord_bin_label(idx: int) -> str:
    return f"q{idx + 1}"  # q1..q5


def _bin_index(truth_mm: float, plane_extent_mm: float, n_bins: int = 5) -> int:
    """Return 0..n_bins-1 quintile index for truth_mm within [0, plane_extent_mm]."""
    if plane_extent_mm <= 0:
        return 0
    frac = max(0.0, min(0.999999, truth_mm / plane_extent_mm))
    return min(n_bins - 1, int(frac * n_bins))


def _collapse_by_section(scored: list[dict]) -> tuple[list[dict], list[dict], int]:
    """Collapse multi-generation rows into best-of-N and mean-of-N per section.

    Returns ``(best_of_n_rows, mean_of_n_rows, num_generations)``. When the
    predictions came from a num_generations==1 run both lists equal the
    input and num_generations is 1.
    """
    by_section: dict[str, list[dict]] = defaultdict(list)
    for row in scored:
        by_section[row["section_id"]].append(row)
    max_gens = max((len(v) for v in by_section.values()), default=1)
    if max_gens <= 1:
        return scored, scored, 1

    best_rows: list[dict] = []
    mean_rows: list[dict] = []
    for sid, group in by_section.items():
        # Best-of-N: lowest abs_err among the section's generations.
        best = min(group, key=lambda r: r["abs_err_mm"])
        best_rows.append(best)
        # Mean-of-N: synthesize one row carrying the mean error + accuracy
        # of the section's generations. Keep all the section-level fields.
        mean_err = statistics.mean(r["abs_err_mm"] for r in group)
        mean_acc = statistics.mean(r["accuracy_pct"] for r in group)
        synthesized = dict(group[0])
        synthesized["abs_err_mm"] = round(mean_err, 4)
        synthesized["accuracy_pct"] = round(mean_acc, 4)
        synthesized["section_id"] = sid
        # Stash spread for downstream tooling — std of error across gens.
        if len(group) > 1:
            synthesized["err_std_mm"] = round(statistics.stdev(r["abs_err_mm"] for r in group), 4)
        else:
            synthesized["err_std_mm"] = 0.0
        mean_rows.append(synthesized)
    return best_rows, mean_rows, max_gens


def summarize(scored: list[dict], *, n_coord_bins: int = 5) -> dict:
    """Produce nested aggregation: overall + per-plane + per-coord-bin + per-brain
    + per-species + per-imaging + per-staining.

    When the predictions include multiple generations per section, also
    emits ``best_of_n`` and ``mean_of_n`` aggregations alongside the
    "all samples" view that treats every (section, generation) pair
    independently.
    """
    by_plane: dict[str, list[dict]] = defaultdict(list)
    by_coord_bin: dict[str, dict[str, list[dict]]] = defaultdict(lambda: defaultdict(list))
    by_brain: dict[str, list[dict]] = defaultdict(list)
    by_species: dict[str, list[dict]] = defaultdict(list)
    by_species_plane: dict[str, list[dict]] = defaultdict(list)
    by_imaging: dict[str, list[dict]] = defaultdict(list)
    by_staining: dict[str, list[dict]] = defaultdict(list)

    for row in scored:
        plane = row["plane"]
        species = row.get("species") or "unknown"
        by_plane[plane].append(row)
        bin_idx = _bin_index(float(row["truth_mm"]), float(row["plane_extent_mm"]), n_coord_bins)
        by_coord_bin[plane][_coord_bin_label(bin_idx)].append(row)
        by_brain[f"{plane}/{row['dataset']}/{row['subject_id']}"].append(row)
        by_species[species].append(row)
        by_species_plane[f"{species}/{plane}"].append(row)
        by_imaging[row.get("imaging") or "unknown"].append(row)
        by_staining[row.get("staining") or "unknown"].append(row)

    # per-coord-bin needs the bin's mm range labelled too
    coord_bin_summary: dict = {}
    for plane, bin_groups in by_coord_bin.items():
        if not bin_groups:
            continue
        # Use the first row's plane_extent_mm as the canonical extent for this plane.
        # All rows in the same plane should have the same atlas extent; if not, they
        # were already binned per-row by fractional position so the labels are still
        # correct, but the mm range shown here is per the first row's atlas.
        sample = next(iter(bin_groups.values()))[0]
        extent = float(sample["plane_extent_mm"])
        bin_width = extent / n_coord_bins
        coord_bin_summary[plane] = {}
        for i in range(n_coord_bins):
            label = _coord_bin_label(i)
            stats = _stat_block(bin_groups.get(label, []))
            stats["range_mm"] = [round(i * bin_width, 4), round((i + 1) * bin_width, 4)]
            coord_bin_summary[plane][label] = stats

    summary: dict = {
        "overall": _stat_block(scored),
        "per_plane": {p: _stat_block(rs) for p, rs in by_plane.items()},
        "per_coord_bin": coord_bin_summary,
        "per_brain": {k: _stat_block(rs) for k, rs in by_brain.items()},
        "per_species": {k: _stat_block(rs) for k, rs in by_species.items()},
        "per_species_plane": {k: _stat_block(rs) for k, rs in by_species_plane.items()},
        "per_imaging": {k: _stat_block(rs) for k, rs in by_imaging.items()},
        "per_staining": {k: _stat_block(rs) for k, rs in by_staining.items()},
    }

    best_rows, mean_rows, num_generations = _collapse_by_section(scored)
    summary["num_generations"] = num_generations
    if num_generations > 1:
        # When multiple generations exist per section, expose two collapsed
        # views: best-of-N (oracle ceiling) and mean-of-N (expected value).
        summary["best_of_n"] = {
            "overall": _stat_block(best_rows),
            "per_plane": {
                p: _stat_block([r for r in best_rows if r["plane"] == p])
                for p in by_plane.keys()
            },
        }
        summary["mean_of_n"] = {
            "overall": _stat_block(mean_rows),
            "per_plane": {
                p: _stat_block([r for r in mean_rows if r["plane"] == p])
                for p in by_plane.keys()
            },
        }
    return summary


# ──────────────────────────────────────────────────────────────────────────
# I/O
# ──────────────────────────────────────────────────────────────────────────


def score_run(run_dir: Path) -> dict:
    """Read predictions.jsonl, score, write scored.jsonl + summary.json. Returns summary."""
    predictions_path = run_dir / "predictions.jsonl"
    if not predictions_path.exists():
        raise FileNotFoundError(f"missing {predictions_path}")

    predictions = []
    with predictions_path.open("r", encoding="utf-8") as fh:
        for raw in fh:
            stripped = raw.strip()
            if not stripped:
                continue
            predictions.append(json.loads(stripped))

    scored = score_predictions(predictions)
    scored_path = run_dir / "scored.jsonl"
    with scored_path.open("w", encoding="utf-8") as fh:
        for row in scored:
            fh.write(json.dumps(row) + "\n")

    summary = summarize(scored)
    summary["eval_health"] = _eval_health(predictions)
    summary_path = run_dir / "summary.json"
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    return summary


def _print_table(summary: dict) -> None:
    """Pretty-print summary to stdout."""
    health = summary.get("eval_health", {})
    warnings = health.get("contamination_warnings", [])
    if warnings:
        print("\n!!! EVAL HEALTH WARNINGS !!!")
        for w in warnings:
            print(f"  ⚠  {w}")
        bt = health.get("failures_by_type", {})
        if bt:
            print(f"  failures_by_type: {bt}")
        print()

    o = summary["overall"]
    n_gens = summary.get("num_generations", 1)
    label = "all-samples (every (section, gen) pair)" if n_gens > 1 else "overall"
    print(f"\n=== {label} (n={o['n']}) ===")
    print(f"  mean_err_pct        = {o.get('mean_err_pct', 0):.2f}%   ← PRIMARY (plane-relative)")
    print(f"  median_err_pct      = {o.get('median_err_pct', 0):.2f}%")
    print(f"  p90 / p99 err_pct   = {o.get('p90_err_pct', 0):.2f}% / {o.get('p99_err_pct', 0):.2f}%")
    print(f"  -- supporting (mm-absolute):")
    print(f"  mae_mm              = {o.get('mae_mm', 0):.4f}")
    print(f"  median_abs_err_mm   = {o.get('median_abs_err_mm', 0):.4f}")
    print(f"  p90 / p99 abs_err   = {o.get('p90_abs_err_mm', 0):.4f} / {o.get('p99_abs_err_mm', 0):.4f}")
    if "mean_decode_tps" in o:
        print(f"  decode tok/s        = mean {o['mean_decode_tps']:.1f}  "
              f"median {o['median_decode_tps']:.1f}  "
              f"p10 {o.get('p10_decode_tps', 0):.1f}  (output tokens / wall second)")
    if "mean_total_tps" in o:
        print(f"  total tok/s         = mean {o['mean_total_tps']:.1f}  "
              f"median {o['median_total_tps']:.1f}  (input+output / wall second)")

    if n_gens > 1:
        bo = summary["best_of_n"]["overall"]
        mo = summary["mean_of_n"]["overall"]
        print(f"\n=== best-of-{n_gens} (oracle ceiling — lowest err per section) ===")
        print(f"  mean_err_pct = {bo.get('mean_err_pct', 0):.2f}%  "
              f"median_err_pct = {bo.get('median_err_pct', 0):.2f}%  "
              f"(mae_mm = {bo.get('mae_mm', 0):.4f})")
        print(f"\n=== mean-of-{n_gens} (expected per-section error) ===")
        print(f"  mean_err_pct = {mo.get('mean_err_pct', 0):.2f}%  "
              f"median_err_pct = {mo.get('median_err_pct', 0):.2f}%  "
              f"(mae_mm = {mo.get('mae_mm', 0):.4f})")

    print("\n=== per_plane ===")
    for plane, s in sorted(summary["per_plane"].items()):
        print(f"  {plane:11s} n={s['n']:4d}  mean_err={s['mean_err_pct']:.2f}%  median={s['median_err_pct']:.2f}%  (mae_mm={s['mae_mm']:.4f})")

    print("\n=== per_coord_bin (quintiles by fractional position) ===")
    for plane, bins in sorted(summary["per_coord_bin"].items()):
        print(f"  {plane}:")
        for label, s in sorted(bins.items()):
            r = s.get("range_mm", [0, 0])
            print(f"    {label} [{r[0]:5.2f}-{r[1]:5.2f}mm] n={s['n']:4d}  mean_err={s.get('mean_err_pct', 0):.2f}%  (mae_mm={s.get('mae_mm', 0):.4f})")

    print("\n=== per_brain ===")
    for brain, s in sorted(summary["per_brain"].items()):
        print(f"  {brain:55s} n={s['n']:3d}  mean_err={s['mean_err_pct']:.2f}%  (mae_mm={s['mae_mm']:.4f})")

    print("\n=== per_species ===")
    for k, s in sorted(summary["per_species"].items()):
        print(f"  {k:15s} n={s['n']:4d}  mean_err={s['mean_err_pct']:.2f}%  (mae_mm={s['mae_mm']:.4f})")

    print("\n=== per_species_plane (cross-cut — use to compare same species×plane across models) ===")
    for k, s in sorted(summary["per_species_plane"].items()):
        print(f"  {k:22s} n={s['n']:4d}  mean_err={s['mean_err_pct']:.2f}%  (mae_mm={s['mae_mm']:.4f})")

    print("\n=== per_imaging ===")
    for k, s in sorted(summary["per_imaging"].items()):
        print(f"  {k:15s} n={s['n']:4d}  mean_err={s['mean_err_pct']:.2f}%  (mae_mm={s['mae_mm']:.4f})")

    print("\n=== per_staining ===")
    for k, s in sorted(summary["per_staining"].items()):
        print(f"  {k:35s} n={s['n']:4d}  mean_err={s['mean_err_pct']:.2f}%  (mae_mm={s['mae_mm']:.4f})")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Score a SliceBench run.")
    parser.add_argument("run_dir", type=Path, help="Directory containing predictions.jsonl")
    parser.add_argument("--quiet", action="store_true", help="Suppress stdout table")
    args = parser.parse_args(argv)

    summary = score_run(args.run_dir)
    if not args.quiet:
        _print_table(summary)
        print(f"\nwrote {args.run_dir / 'scored.jsonl'}")
        print(f"wrote {args.run_dir / 'summary.json'}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
