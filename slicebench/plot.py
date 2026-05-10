"""Comparison plots for SliceBench runs.

Reads ``summary.json`` and ``scored.jsonl`` from each run directory and emits
GraphPad-Prism-styled PNGs. Each panel saved separately + a combined figure.
"""

from __future__ import annotations

import argparse
import json
import random
import sys
import warnings
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import pandas as pd
import plotnine as p9

from slicebench.prism_theme import PRISM_PALETTE, theme_prism

warnings.filterwarnings("ignore", category=FutureWarning, module="plotnine")


# ──────────────────────────────────────────────────────────────────────────
# Run loading
# ──────────────────────────────────────────────────────────────────────────


def _load_run(run_dir: Path) -> tuple[dict, pd.DataFrame]:
    summary = json.loads((run_dir / "summary.json").read_text(encoding="utf-8"))
    rows = []
    with (run_dir / "scored.jsonl").open("r", encoding="utf-8") as fh:
        for raw in fh:
            stripped = raw.strip()
            if not stripped:
                continue
            rows.append(json.loads(stripped))
    df = pd.DataFrame(rows)
    return summary, df


def _model_label(run_dir: Path, summary: dict) -> str:
    """Human-friendly model label. Prefer run_meta.model, fall back to dir name."""
    meta_path = run_dir / "run_meta.json"
    if meta_path.exists():
        meta = json.loads(meta_path.read_text(encoding="utf-8"))
        model = meta.get("model")
        if model:
            return str(model)
    # fallback
    return run_dir.name


# ──────────────────────────────────────────────────────────────────────────
# Bootstrap CI
# ──────────────────────────────────────────────────────────────────────────


def _bootstrap_ci(values: list[float], n_iter: int = 1000, alpha: float = 0.05,
                  seed: int = 7) -> tuple[float, float]:
    if not values:
        return 0.0, 0.0
    rng = random.Random(seed)
    n = len(values)
    means = []
    for _ in range(n_iter):
        sample = [values[rng.randrange(n)] for _ in range(n)]
        means.append(sum(sample) / n)
    means.sort()
    lo = means[int((alpha / 2) * n_iter)]
    hi = means[int((1 - alpha / 2) * n_iter)]
    return lo, hi


# ──────────────────────────────────────────────────────────────────────────
# Plot builders
# ──────────────────────────────────────────────────────────────────────────


def _palette_for(models: list[str]) -> dict[str, str]:
    return {m: PRISM_PALETTE[i % len(PRISM_PALETTE)] for i, m in enumerate(models)}


def plot_headline_accuracy(combined: pd.DataFrame, out: Path) -> None:
    """Bar chart of mean accuracy per model with bootstrap 95% CI error bars."""
    rows = []
    for model, group in combined.groupby("model"):
        accs = group["accuracy_pct"].tolist()
        mean = sum(accs) / len(accs)
        lo, hi = _bootstrap_ci(accs)
        rows.append({"model": model, "mean": mean, "lo": lo, "hi": hi})
    df = pd.DataFrame(rows).sort_values("mean", ascending=False).reset_index(drop=True)
    palette = _palette_for(df["model"].tolist())

    plot = (
        p9.ggplot(df, p9.aes(x="model", y="mean", fill="model"))
        + p9.geom_bar(stat="identity", color="black", size=0.4, width=0.6)
        + p9.geom_errorbar(p9.aes(ymin="lo", ymax="hi"), width=0.2, size=0.6, color="black")
        + p9.scale_fill_manual(values=palette)
        + p9.scale_y_continuous(limits=[0, 100], expand=(0, 0))
        + p9.labs(x="Model", y="Mean accuracy (%)", title="SliceBench: headline accuracy")
        + theme_prism()
        + p9.theme(legend_position="none", figure_size=(max(5, 1.2 * len(df)), 4.5))
    )
    plot.save(out, dpi=200, verbose=False)


def plot_accuracy_by_plane(combined: pd.DataFrame, out: Path) -> None:
    """Mean accuracy per (model × plane), faceted by plane."""
    rows = []
    for (model, plane), group in combined.groupby(["model", "plane"]):
        accs = group["accuracy_pct"].tolist()
        rows.append({"model": model, "plane": plane,
                     "mean": sum(accs) / len(accs), "n": len(accs)})
    df = pd.DataFrame(rows)
    palette = _palette_for(sorted(df["model"].unique()))

    plot = (
        p9.ggplot(df, p9.aes(x="model", y="mean", fill="model"))
        + p9.geom_bar(stat="identity", color="black", size=0.4, width=0.6)
        + p9.scale_fill_manual(values=palette)
        + p9.scale_y_continuous(limits=[0, 100], expand=(0, 0))
        + p9.facet_wrap("plane")
        + p9.labs(x="", y="Mean accuracy (%)", title="SliceBench: accuracy per plane")
        + theme_prism()
        + p9.theme(
            legend_position="bottom",
            axis_text_x=p9.element_text(rotation=30, ha="right"),
            figure_size=(10, 4.5),
        )
    )
    plot.save(out, dpi=200, verbose=False)


def plot_accuracy_by_coord_bin(combined: pd.DataFrame, out: Path) -> None:
    """Line+dot per model showing accuracy along plane axis (anterior→posterior etc).

    x = quintile bin (q1..q5), faceted by plane.
    """
    # Build coord bins from truth_mm/plane_extent_mm. Match score._bin_index.
    df = combined.copy()
    frac = (df["truth_mm"] / df["plane_extent_mm"]).clip(lower=0.0, upper=0.999999)
    df["bin"] = (frac * 5).astype(int).clip(upper=4)
    df["bin_label"] = df["bin"].map(lambda i: f"q{i + 1}")

    rows = []
    for (model, plane, bin_label), group in df.groupby(["model", "plane", "bin_label"]):
        accs = group["accuracy_pct"].tolist()
        rows.append({"model": model, "plane": plane, "bin_label": bin_label,
                     "mean": sum(accs) / len(accs), "n": len(accs)})
    agg = pd.DataFrame(rows)
    if agg.empty:
        return
    palette = _palette_for(sorted(agg["model"].unique()))
    bin_order = ["q1", "q2", "q3", "q4", "q5"]
    agg["bin_label"] = pd.Categorical(agg["bin_label"], categories=bin_order, ordered=True)

    plot = (
        p9.ggplot(agg, p9.aes(x="bin_label", y="mean", color="model", group="model"))
        + p9.geom_line(size=0.9)
        + p9.geom_point(size=2.5)
        + p9.scale_color_manual(values=palette)
        + p9.scale_y_continuous(limits=[0, 100], expand=(0, 0))
        + p9.facet_wrap("plane")
        + p9.labs(
            x="Position quintile (anterior/medial/dorsal → posterior/lateral/ventral)",
            y="Mean accuracy (%)",
            title="SliceBench: accuracy by coordinate range",
        )
        + theme_prism()
        + p9.theme(legend_position="bottom", figure_size=(11, 4.5))
    )
    plot.save(out, dpi=200, verbose=False)


def plot_error_distribution(combined: pd.DataFrame, out: Path) -> None:
    """Boxplot of abs_err_mm per model, log-y."""
    df = combined.copy()
    # plotnine's log scale chokes on zeros; floor at 0.001mm
    df["abs_err_mm_safe"] = df["abs_err_mm"].clip(lower=0.001)
    palette = _palette_for(sorted(df["model"].unique()))

    plot = (
        p9.ggplot(df, p9.aes(x="model", y="abs_err_mm_safe", fill="model"))
        + p9.geom_boxplot(color="black", size=0.4, width=0.6, outlier_size=1.0)
        + p9.scale_fill_manual(values=palette)
        + p9.scale_y_log10()
        + p9.labs(x="Model", y="Absolute error (mm, log scale)",
                  title="SliceBench: error distribution")
        + theme_prism()
        + p9.theme(
            legend_position="none",
            axis_text_x=p9.element_text(rotation=30, ha="right"),
            figure_size=(max(5, 1.2 * df["model"].nunique()), 4.5),
        )
    )
    plot.save(out, dpi=200, verbose=False)


def plot_error_by_plane(combined: pd.DataFrame, out: Path) -> None:
    """Boxplot of abs_err_mm per model, faceted by plane."""
    df = combined.copy()
    df["abs_err_mm_safe"] = df["abs_err_mm"].clip(lower=0.001)
    palette = _palette_for(sorted(df["model"].unique()))

    plot = (
        p9.ggplot(df, p9.aes(x="model", y="abs_err_mm_safe", fill="model"))
        + p9.geom_boxplot(color="black", size=0.4, width=0.6, outlier_size=1.0)
        + p9.scale_fill_manual(values=palette)
        + p9.scale_y_log10()
        + p9.facet_wrap("plane")
        + p9.labs(x="", y="Absolute error (mm, log scale)",
                  title="SliceBench: error distribution per plane")
        + theme_prism()
        + p9.theme(
            legend_position="bottom",
            axis_text_x=p9.element_text(rotation=30, ha="right"),
            figure_size=(11, 4.5),
        )
    )
    plot.save(out, dpi=200, verbose=False)


def plot_cost_vs_accuracy(summaries: dict[str, dict], runs_meta: dict[str, dict], out: Path) -> None:
    """Scatter of mean accuracy vs estimated cost per section. Skipped if no cost info."""
    rows = []
    for model, summary in summaries.items():
        meta = runs_meta.get(model, {})
        cost_total = meta.get("estimated_cost_usd")
        n = summary["overall"]["n"]
        if cost_total is None or not n:
            continue
        rows.append({
            "model": model,
            "cost_per_section_usd": cost_total / n,
            "mean_accuracy_pct": summary["overall"]["mean_accuracy_pct"],
        })
    if not rows:
        # Nothing to plot. Write a placeholder so callers know.
        return
    df = pd.DataFrame(rows)
    palette = _palette_for(df["model"].tolist())

    plot = (
        p9.ggplot(df, p9.aes(x="cost_per_section_usd", y="mean_accuracy_pct",
                              color="model", label="model"))
        + p9.geom_point(size=4)
        + p9.geom_text(nudge_y=2, size=10, fontweight="bold", show_legend=False)
        + p9.scale_color_manual(values=palette)
        + p9.scale_x_log10()
        + p9.scale_y_continuous(limits=[0, 100], expand=(0, 0))
        + p9.labs(x="Cost per section (USD, log scale)", y="Mean accuracy (%)",
                  title="SliceBench: cost vs accuracy")
        + theme_prism()
        + p9.theme(legend_position="none", figure_size=(7, 5))
    )
    plot.save(out, dpi=200, verbose=False)


# ──────────────────────────────────────────────────────────────────────────
# Driver
# ──────────────────────────────────────────────────────────────────────────


def build_plots(run_dirs: list[Path], out_dir: Path) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)

    summaries: dict[str, dict] = {}
    runs_meta: dict[str, dict] = {}
    frames: list[pd.DataFrame] = []
    for run in run_dirs:
        summary, df = _load_run(run)
        model_label = _model_label(run, summary)
        df = df.assign(model=model_label)
        frames.append(df)
        summaries[model_label] = summary
        meta_path = run / "run_meta.json"
        if meta_path.exists():
            runs_meta[model_label] = json.loads(meta_path.read_text(encoding="utf-8"))

    if not frames:
        print("no runs supplied; nothing to plot", file=sys.stderr)
        return

    combined = pd.concat(frames, ignore_index=True)

    plot_headline_accuracy(combined, out_dir / "01_headline_accuracy.png")
    plot_accuracy_by_plane(combined, out_dir / "02_accuracy_by_plane.png")
    plot_accuracy_by_coord_bin(combined, out_dir / "03_accuracy_by_coord_bin.png")
    plot_error_distribution(combined, out_dir / "04_error_distribution.png")
    plot_error_by_plane(combined, out_dir / "05_error_by_plane.png")
    plot_cost_vs_accuracy(summaries, runs_meta, out_dir / "06_cost_vs_accuracy.png")

    print(f"wrote {len(list(out_dir.glob('*.png')))} PNG(s) to {out_dir}")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Build SliceBench comparison plots.")
    parser.add_argument("--runs", nargs="+", required=True, type=Path,
                        help="One or more run directories (each must have summary.json + scored.jsonl)")
    parser.add_argument("--out", required=True, type=Path,
                        help="Output directory for PNGs")
    args = parser.parse_args(argv)

    missing = [r for r in args.runs if not (r / "summary.json").exists()]
    if missing:
        for r in missing:
            print(f"ERROR: {r}/summary.json not found — run score.py on it first", file=sys.stderr)
        return 1

    build_plots(args.runs, args.out)
    return 0


if __name__ == "__main__":
    sys.exit(main())
