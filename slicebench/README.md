# SliceBench

Multi-model position-estimation benchmark for LangSlice.

The bench is `data/slicebench.json`:
- `tiny=64` — 8 evenly-spaced sections per brain across the small-8 brains, for fast FT-checkpoint iteration (~10 min on a local llama.cpp Gemma).
- `small=317` — 8 brains, all eval sections, for fast model comparison.
- `large=1051` — 24 brains, all eval sections, for publication-grade reporting.

`tiny ⊂ small ⊂ large` by brain composition. Truth comes from
`data/manifest/shards/<plane>/<dataset>.jsonl` joined to the active eval
allocation in `data/manifest/allocations/<plane>/eval.jsonl`. `tiny`'s
per-brain section selection is **deterministic and reproducible** — the loader
sorts each brain's eval sections by `truth_mm` and picks N evenly-spaced
indices. No RNG seed needed.

## Scoring

Pure plane-relative accuracy. No tolerances, no thresholds, no rescues.

```
accuracy_pct = max(0, 100 * (1 - abs_err_mm / plane_extent_mm))
```

`plane_extent_mm` is the atlas extent along the slice-normal axis (sagittal
halved per canonical-hemisphere convention). A 0.5mm error on the 13.2mm
coronal AP axis scores 96.2%; the same 0.5mm error on a 5.5mm sagittal
canonical hemisphere scores 90.9%. Sagittal positions are canonicalized
before computing absolute error so mirror flips don't show as errors.

## Workflow

```powershell
# 1. Run each model on small-8 (~317 sections)
python slicebench/run.py --model gemini-3-flash-preview --bench small --out slicebench/runs/small/gemini-3-flash-preview/
python slicebench/run.py --model gemma-4-31b-it          --bench small --out slicebench/runs/small/gemma-4-31b-it/
python slicebench/run.py --model gemma-4-26b-a4b-it      --bench small --out slicebench/runs/small/gemma-4-e4b/

# 2. Once the fine-tuned model is ready, serve it via llama.cpp + litellm-proxy:
.\llama-server.exe -m langslice-gemma-4-e4b-ft.gguf --port 8080 --alias langslice-gemma-4-e4b-ft
$env:LANGSLICE_LITELLM_PROXY_BASE = "http://localhost:8080/v1"
$env:LANGSLICE_LITELLM_PROXY_KEY  = "sk-langslice-local"
python slicebench/run.py --model litellm-proxy:langslice-gemma-4-e4b-ft --bench small --out slicebench/runs/small/gemma-4-e4b-ft/

# 3. Score each run (idempotent + cheap; re-run any time after run.py)
python slicebench/score.py slicebench/runs/small/gemini-3-flash-preview/
python slicebench/score.py slicebench/runs/small/gemma-4-31b-it/
python slicebench/score.py slicebench/runs/small/gemma-4-e4b/

# 4. Plot all runs together
python slicebench/plot.py --runs slicebench/runs/small/* --out slicebench/figures/small/

# 5. Once everything passes on small, run the full large-24 (~1051 sections, ~5x cost)
python slicebench/run.py --model gemini-3-flash-preview --bench large --out slicebench/runs/large/gemini-3-flash-preview/
# ...etc.
```

`run.py` is **resumable** — re-launching with the same `--out` dir picks up
where it left off (skips section_ids already in `predictions.jsonl`).

## CLI reference

### `run.py`

```
python slicebench/run.py --model <id> --bench {small,large} --out <dir> [--concurrency N] [--limit N]
```

- `--model` — passed straight through `langslice_harness.harness.estimation.model_resolver`. Native
  Gemini strings work directly (`gemini-3-flash-preview`); Gemma family routes through ADK's `Gemma`
  wrapper (`gemma-4-31b-it`); local llama.cpp / openrouter / ollama via the relevant prefixes.
- `--bench` — `small` or `large`.
- `--out` — run output directory. Created if missing.
- `--concurrency` — parallel workers (default 1). Pro must stay 1; Flash tolerates 6–8.
- `--limit` — cap to first N rows for dev/smoke.

Outputs:

- `predictions.jsonl` — one row per section: `{section_id, dataset, subject_id, plane, atlas,
  species, staining, imaging, image_path, plane_extent_mm, truth_mm, predicted_mm, reasoning,
  error, elapsed_s, model, run_id, ts}`
- `run_meta.json` — model, bench, start/end, totals, failure count.

Note on `gemini-3-flash-preview`: there is a known hidden-quota 429 issue with this preview model.
If 429s appear, surface the failure and decide whether to back off — **never auto-fall-back to
Gemini 2.x**. Only Gemini 3 models are valid for this bench.

### `score.py`

```
python slicebench/score.py <run_dir> [--quiet]
```

Reads `predictions.jsonl`, applies plane-relative accuracy, writes:

- `scored.jsonl` — predictions + `abs_err_mm` + `accuracy_pct`
- `summary.json` — overall + per-plane + per-coord-bin (5 quintiles) + per-brain + per-species
  + per-imaging + per-staining

Each breakdown reports `n`, `mean_accuracy_pct`, `median_accuracy_pct`, `mae_mm`,
`median_abs_err_mm`, `p90_abs_err_mm`, `p99_abs_err_mm`.

### `plot.py`

```
python slicebench/plot.py --runs <run_dir>... --out <fig_dir>
```

Reads `summary.json` + `scored.jsonl` from each run and emits 6 PNGs in
GraphPad-Prism style (white bg, no grid, inward ticks, bold sans-serif):

1. `01_headline_accuracy.png` — mean accuracy bars per model, bootstrap 95% CI.
2. `02_accuracy_by_plane.png` — same, faceted by plane.
3. `03_accuracy_by_coord_bin.png` — line+dot per model along the plane axis (anterior→posterior, etc).
4. `04_error_distribution.png` — boxplot of `abs_err_mm`, log-y.
5. `05_error_by_plane.png` — same, faceted by plane.
6. `06_cost_vs_accuracy.png` — scatter of mean accuracy vs $/section (skipped if no costs in `run_meta.json`).

## Files

```
slicebench/
  __init__.py        # package marker
  loader.py          # bench loader → list[EvalRow]
  run.py             # CLI: run one model through one bench
  score.py           # plane-relative accuracy + aggregations
  plot.py            # plotnine plots
  prism_theme.py     # theme_prism() + PRISM_PALETTE
  README.md          # this file
```

## Design notes

- **Reuses the production agent loop** — `estimate_position()` is called directly. No shadow runner.
- **Bench is sourced from three places** — `data/slicebench.json` for the brain list,
  per-shard JSONL for ground-truth positions, allocations for split membership. The loader joins
  all three and drops anything that fails the join with a stderr warning.
- **Atlas extents cached on each `EvalRow`** so scoring doesn't reload atlases per section.
- **Sagittal canonicalization** happens at scoring time, not in the runner, so raw predictions
  are preserved verbatim in `predictions.jsonl` for later inspection.
- **No tolerances anywhere.** Tolerances are SFT-data-filtering concepts; using them on a
  benchmark biases the numbers in our favor.
