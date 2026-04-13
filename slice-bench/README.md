# slice-bench

Model-agnostic benchmark for evaluating AP position estimation quality on histological brain slices.

Produces standardized metrics (MAE, median error, % within threshold) that allow direct comparison across any backend: Gemini API, Gemma fine-tuned, Gemma base, or future models.

## Usage

```bash
python -m slice_bench.run \
    --adapter gemini \
    --dataset references/TestImages/M01 \
    --ground-truth references/TestImages/M01/ground_truth.json
```

## Structure

```
slice-bench/
├── slice_bench/
│   ├── __init__.py
│   ├── run.py              # CLI entry point
│   ├── metrics.py          # MAE, median, percentile, threshold scoring
│   └── adapters/
│       ├── __init__.py
│       ├── base.py         # Abstract adapter interface
│       ├── gemini.py       # Gemini API adapter (existing LangSlice pipeline)
│       └── gemma_local.py  # Local Gemma inference adapter
├── datasets/               # Symlinks or pointers to test image sets
├── results/                # Benchmark run outputs (gitignored)
└── README.md
```

## Adapters

Each adapter implements a simple interface:

```python
class BaseAdapter:
    async def estimate_positions(
        self, images: list[Path], atlas: str
    ) -> dict[str, float]:
        """Return {filename: estimated_ap_mm} for all images."""
        ...
```

This decouples the benchmark from any specific model backend.

## Datasets

- **M01**: 20 slices, Allen Mouse Brain CCF, ground truth from expert annotation
- **M02**: (planned) second mouse brain, different staining

## Relationship to eval/

The existing `eval/eval_brain.py` is tightly coupled to the LangSlice whole-brain pipeline. `slice-bench` is the generalized, model-agnostic successor that can evaluate any estimation backend independently.
