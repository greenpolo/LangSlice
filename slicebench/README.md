# SliceBench

A position-estimation benchmark for histological brain sections. Given an image
of a slice, a model predicts its anterior-posterior coordinate (mm) in the
matching brain atlas. Built alongside the
[LangSlice](https://github.com/greenpolo/LangSlice) slice-registration
pipeline; this is the benchmark
[`langslice-gemma-4-E4B`](https://huggingface.co/greenpolo/langslice-gemma-4-E4B)
v1.0 is evaluated on.

## Quick start

Evaluate the published `langslice-gemma-4-E4B` on the small coronal tier
(serve the model via vLLM or llama.cpp first; see the
[model card](https://huggingface.co/greenpolo/langslice-gemma-4-E4B) for the
serving recipe):

```powershell
python slicebench/run.py `
    --model litellm-proxy:langslice-gemma-4-E4B `
    --bench small_coronal `
    --out slicebench/runs/small_coronal/langslice-gemma-4-E4B/
python slicebench/score.py slicebench/runs/small_coronal/langslice-gemma-4-E4B/
```

That reproduces the v1.0 model-card numbers (~1.23 mm MAE on the mouse subset).

## Bench tiers

The bench is defined in [`bench.json`](./bench.json):

- `tiny` — 7 brains × 8 evenly-spaced sections = 56 sections. Deterministic
  per-brain sampling. Fast FT-checkpoint iteration (~10 min on a local
  llama.cpp Gemma).
- `small` — 7 brains, all eval-allocated sections (mostly mouse coronal + a
  few sagittal / horizontal entries for OOD signal). Fast model comparison.
- `large` — 20 brains, all eval-allocated sections. Publication-grade
  reporting across coronal / sagittal / horizontal planes, mouse + rat.

`tiny ⊂ small ⊂ large` by brain composition. Truth comes from
[`slicebench/data/shards/<plane>/<dataset>.jsonl`](./data/shards/) joined to
the active eval allocation in
[`slicebench/data/allocations/<plane>/eval.jsonl`](./data/allocations/).
`tiny`'s per-brain section selection is **deterministic and reproducible** —
the loader sorts each brain's eval sections by `truth_mm` and picks N
evenly-spaced indices. No RNG seed needed.

The bench is **self-contained** — `slicebench/` ships the bench definition,
ground-truth shards, eval allocations, and downscaled section images. No
external data download is required to run or reproduce the benchmark.

## Scoring

Pure plane-relative accuracy. No tolerances, no thresholds, no rescues.

```
accuracy_pct = max(0, 100 * (1 - abs_err_mm / plane_extent_mm))
```

`plane_extent_mm` is the atlas extent along the slice-normal axis (sagittal
halved per canonical-hemisphere convention). A 0.5 mm error on the 13.2 mm
coronal AP axis scores 96.2 %; the same 0.5 mm error on a 5.5 mm sagittal
canonical hemisphere scores 90.9 %. Sagittal positions are canonicalized
before computing absolute error so mirror flips don't show as errors.

## Multi-model workflow

```powershell
# 1. Run each model on small (~200 sections)
python slicebench/run.py --model gemini-3-flash-preview              --bench small --out slicebench/runs/small/gemini-3-flash-preview/
python slicebench/run.py --model unsloth/gemma-4-E4B-it              --bench small --out slicebench/runs/small/base-gemma-4-e4b/
python slicebench/run.py --model litellm-proxy:langslice-gemma-4-E4B --bench small --out slicebench/runs/small/langslice-gemma-4-e4b/

# 2. To run a custom fine-tune via llama.cpp + litellm-proxy:
.\llama-server.exe -m my-finetune.gguf --port 8080 --alias my-finetune
$env:LANGSLICE_LITELLM_PROXY_BASE = "http://localhost:8080/v1"
$env:LANGSLICE_LITELLM_PROXY_KEY  = "sk-langslice-local"
python slicebench/run.py --model litellm-proxy:my-finetune --bench small --out slicebench/runs/small/my-finetune/

# 3. Score each run (idempotent + cheap; re-run any time after run.py)
python slicebench/score.py slicebench/runs/small/gemini-3-flash-preview/
python slicebench/score.py slicebench/runs/small/base-gemma-4-e4b/
python slicebench/score.py slicebench/runs/small/langslice-gemma-4-e4b/

# 4. Plot all runs together
python slicebench/plot.py --runs slicebench/runs/small/* --out slicebench/figures/small/

# 5. Once everything passes on small, run the full large-24 (~1000 sections, ~5x cost)
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

- `--model` — passed straight through
  `langslice_harness.harness.estimation.model_resolver`. Native Gemini strings
  work directly (`gemini-3-flash-preview`); the Gemma family routes through
  the ADK `Gemma` wrapper; local llama.cpp / openrouter / ollama via their
  respective prefixes.
- `--bench` — `small` or `large`.
- `--out` — run output directory. Created if missing.
- `--concurrency` — parallel workers (default 1). Pro models must stay at 1;
  Flash tolerates 6–8.
- `--limit` — cap to first N rows for dev / smoke.

Outputs:

- `predictions.jsonl` — one row per section:
  `{section_id, dataset, subject_id, plane, atlas, species, staining, imaging,
  image_path, plane_extent_mm, truth_mm, predicted_mm, reasoning, error,
  elapsed_s, model, run_id, ts}`
- `run_meta.json` — model, bench, start / end timestamps, totals, failure
  count.

Note on `gemini-3-flash-preview`: there is a known hidden-quota 429 issue
with this preview model. If 429s appear, surface the failure and decide
whether to back off — **never auto-fall-back to Gemini 2.x**. Only Gemini 3
models are valid for this bench.

### `score.py`

```
python slicebench/score.py <run_dir> [--quiet]
```

Reads `predictions.jsonl`, applies plane-relative accuracy, writes:

- `scored.jsonl` — predictions + `abs_err_mm` + `accuracy_pct`
- `summary.json` — overall + per-plane + per-coord-bin (5 quintiles) +
  per-brain + per-species + per-species-plane + per-imaging + per-staining

Each breakdown reports `n`, `mean_accuracy_pct`, `median_accuracy_pct`,
`mae_mm`, `median_abs_err_mm`, `p90_abs_err_mm`, `p99_abs_err_mm`.

### `plot.py`

```
python slicebench/plot.py --runs <run_dir>... --out <fig_dir>
```

Reads `summary.json` + `scored.jsonl` from each run and emits 6 PNGs in
GraphPad-Prism style (white bg, no grid, inward ticks, bold sans-serif):

1. `01_headline_accuracy.png` — mean accuracy bars per model, bootstrap
   95 % CI.
2. `02_accuracy_by_plane.png` — same, faceted by plane.
3. `03_accuracy_by_coord_bin.png` — line + dot per model along the plane
   axis (anterior→posterior, etc).
4. `04_error_distribution.png` — boxplot of `abs_err_mm`, log-y.
5. `05_error_by_plane.png` — same, faceted by plane.
6. `06_cost_vs_accuracy.png` — scatter of mean accuracy vs $/section
   (skipped if no costs in `run_meta.json`).

## Files

```
slicebench/
  __init__.py            # package marker
  loader.py              # bench loader → list[EvalRow]
  run.py                 # CLI: run one model through one bench
  score.py               # plane-relative accuracy + aggregations
  plot.py                # plotnine plots
  prism_theme.py         # theme_prism() + PRISM_PALETTE
  pricing.py             # per-model $/section pricing
  bench.json             # brain composition (tiny / small / large)
  LICENSE                # MIT (code only)
  NOTICE                 # master attribution index
  LICENSES_CORONAL.md    # per-dataset license verification
  CITATION.cff           # structured citation metadata
  README.md              # this file
  data/
    shards/<plane>/<dataset>.jsonl       # per-section GT rows
    allocations/<plane>/eval.jsonl       # active eval section_ids
    images/<plane>/<dataset>/<subject>/  # downscaled JPEGs (long-edge ≤ 2048)
```

## Design notes

- **Reuses the production agent loop** — `estimate_position()` is called
  directly. No shadow runner.
- **Bench is sourced from three places** — `bench.json` for the brain list,
  per-shard JSONL for ground-truth positions, allocations for split
  membership. The loader joins all three and drops anything that fails the
  join with a stderr warning.
- **Atlas extents cached on each `EvalRow`** so scoring doesn't reload
  atlases per section.
- **Sagittal canonicalization** happens at scoring time, not in the runner,
  so raw predictions are preserved verbatim in `predictions.jsonl` for later
  inspection.
- **No tolerances anywhere.** Tolerances are SFT-data-filtering concepts;
  using them on a benchmark biases the numbers in our favor.

## Citation

If you use SliceBench in your work, please cite:

```bibtex
@misc{walshlab2026slicebench,
  title  = {SliceBench: a benchmark for position estimation on histological brain sections},
  author = {Walsh Lab and Baughman, Nicholas},
  year   = {2026},
  month  = {May},
  url    = {https://github.com/greenpolo/LangSlice/tree/main/slicebench}
}
```

See also [`CITATION.cff`](./CITATION.cff) for a structured citation.

## License

SliceBench is distributed under MIT for the *code* (see [`LICENSE`](./LICENSE)).
The brain-section *data* it loads is governed by the per-dataset licenses
documented in [`LICENSES_CORONAL.md`](./LICENSES_CORONAL.md) and the master
[`NOTICE`](./NOTICE) file. The data is noncommercial as bundled because the
Allen Institute brain images carry the Allen Terms of Use, which dominates
the mixed license matrix.

## Related

- The
  [`langslice-gemma-4-E4B`](https://huggingface.co/greenpolo/langslice-gemma-4-E4B)
  model card cites SliceBench small coronal numbers as its headline
  benchmark.
- The [LangSlice](https://github.com/greenpolo/LangSlice) parent repo
  contains the production agent loop, the manifest infrastructure, and the
  data adapters used to source the brain section images.
