"""Curriculum scaffolding for the LangSlice Gemma 4 trainers.

Phase A (read-only): :func:`bins.compute_section_bins` assigns a quintile
``bin_idx`` to every section in an allocation pool, using slicebench's
existing :func:`slicebench.score._bin_index`. Logging hooks live in
:mod:`log` so per-section reward / abs-error events can stream to JSONL.

Phase B (uniform-weight sampler): :class:`sampler.WeightedRowDataset`
extends the RLVR :class:`rlvr.dataset.RowDataset` with a per-row
``_weights`` field initialized to all 1.0, plus
:class:`sampler.CurriculumGRPOTrainer` /
:class:`sampler.CurriculumSFTTrainer` subclasses that wire a
:class:`torch.utils.data.WeightedRandomSampler` whenever the dataset
exposes ``_weights``. With uniform weights the sampling distribution is
statistically indistinguishable from the default RandomSampler — so the
plumbing can ship before any real curriculum policy lands.

Phase C (real weight updates) is intentionally NOT in this module yet;
it will land alongside the expert-iteration loop that reads slicebench
``per_coord_bin`` MAE.
"""

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


def __getattr__(name: str):  # noqa: D401 — lazy re-exports
    """Lazy re-exports so importing the package doesn't pull TRL/torch eagerly."""
    if name == "compute_section_bins":
        from .bins import compute_section_bins as _f

        return _f
    if name in {"WeightedRowDataset", "CurriculumGRPOTrainer", "CurriculumSFTTrainer"}:
        from . import sampler as _s

        return getattr(_s, name)
    if name == "CurriculumLogger":
        from .log import CurriculumLogger as _c

        return _c
    if name in _WEIGHTS_NAMES:
        from . import weights as _w

        return getattr(_w, name)
    raise AttributeError(name)
