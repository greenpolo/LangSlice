"""Backward-compatible shim. The curriculum package now lives at training.adaptive.curriculum.
This shim re-exports the public surface; remove once all callers are updated."""

from __future__ import annotations

__all__ = [
    "compute_section_bins",
    "WeightedRowDataset",
    "CurriculumGRPOTrainer",
    "CurriculumSFTTrainer",
    "CurriculumLogger",
    # Phase C
    "compute_weights",
    "read_per_bin_mae",
    "update_weighted_dataset",
    "read_weights_json",
    "write_weights_json",
]

_WEIGHTS_NAMES = {
    "compute_weights",
    "read_per_bin_mae",
    "update_weighted_dataset",
    "read_weights_json",
    "write_weights_json",
}


def __getattr__(name: str):  # noqa: D401 — lazy re-exports via shim
    """Lazy re-exports — mirrors the original package's __getattr__ pattern."""
    if name == "compute_section_bins":
        from adaptive.curriculum.bins import compute_section_bins as _f
        return _f
    if name in {"WeightedRowDataset", "CurriculumGRPOTrainer", "CurriculumSFTTrainer"}:
        from adaptive.curriculum import sampler as _s
        return getattr(_s, name)
    if name == "CurriculumLogger":
        from adaptive.curriculum.log import CurriculumLogger as _c
        return _c
    if name in _WEIGHTS_NAMES:
        from adaptive.curriculum import weights as _w
        return getattr(_w, name)
    raise AttributeError(name)
