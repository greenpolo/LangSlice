"""Curriculum-aware datasets + trainer subclasses.

Phase B exposes a :class:`WeightedRowDataset` that wraps the existing RLVR
:class:`rlvr.dataset.RowDataset` with a mutable ``_weights`` list, plus
:class:`CurriculumGRPOTrainer` and :class:`CurriculumSFTTrainer` subclasses
that swap in a :class:`torch.utils.data.WeightedRandomSampler` whenever the
training dataset exposes ``_weights``. With weights initialised to all 1.0
the sampling distribution is statistically indistinguishable from the
default RandomSampler, so the plumbing is safe to ship before the Phase C
weight-update policy lands.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import TYPE_CHECKING, Any

# Import the parent ``RowDataset`` lazily to keep the curriculum package
# importable in environments without the full RLVR dataset loader (e.g. the
# unit-test container where TRL/unsloth aren't installed). The class itself
# has no heavy imports at module import time, so this is just defensive.
if TYPE_CHECKING:  # pragma: no cover - typing only
    pass

from langslice_training.rl_core.dataset import RowDataset


class WeightedRowDataset(RowDataset):
    """``RowDataset`` + a per-row ``_weights`` list.

    The trainer subclasses below switch to a ``WeightedRandomSampler`` IFF
    ``hasattr(dataset, "_weights")``, so any dataset that provides this
    attribute opts in to curriculum sampling. Default weights are all 1.0
    which makes ``WeightedRandomSampler`` produce a uniform distribution
    over rows (with replacement), statistically equivalent to
    ``RandomSampler`` for sampling purposes.

    Phase B keeps weights uniform. Phase C will mutate them between
    expert-iteration rounds via :meth:`set_weights` from per-bin slicebench
    MAE estimates.
    """

    def __init__(self, rows: list[dict[str, Any]]) -> None:
        super().__init__(rows)
        self._weights: list[float] = [1.0] * len(rows)

    def set_weights(self, weights: Sequence[float]) -> None:
        """Replace the per-row sampling weights.

        ``len(weights)`` must equal ``len(self)``. Weights must be
        non-negative and at least one must be strictly positive — otherwise
        :class:`torch.utils.data.WeightedRandomSampler` raises a
        ``RuntimeError`` at sampling time.
        """
        n = len(self)
        if len(weights) != n:
            raise ValueError(f"set_weights: expected {n} values (len(dataset)), got {len(weights)}")
        coerced = [float(w) for w in weights]
        if any(w < 0.0 for w in coerced):
            raise ValueError("set_weights: weights must be non-negative")
        if not any(w > 0.0 for w in coerced):
            raise ValueError("set_weights: at least one weight must be > 0")
        self._weights = coerced


def _build_weighted_dataloader(trainer: Any):
    """Construct the curriculum-aware DataLoader for ``trainer``.

    Used by both :class:`CurriculumGRPOTrainer` and
    :class:`CurriculumSFTTrainer`. Mirrors the default Trainer dataloader
    knobs that downstream code relies on (``per_device_train_batch_size``,
    ``data_collator``, ``dataloader_num_workers``).
    """
    import torch  # local import — keep package importable without torch

    dataset = trainer.train_dataset
    sampler = torch.utils.data.WeightedRandomSampler(
        weights=list(dataset._weights),
        num_samples=len(dataset),
        replacement=True,
    )
    args = trainer.args
    batch_size = int(getattr(args, "per_device_train_batch_size", 1) or 1)
    num_workers = int(getattr(args, "dataloader_num_workers", 0) or 0)
    return torch.utils.data.DataLoader(
        dataset,
        sampler=sampler,
        batch_size=batch_size,
        collate_fn=getattr(trainer, "data_collator", None),
        num_workers=num_workers,
    )


def _maybe_curriculum_dataloader(trainer: Any):
    """Return a weighted DataLoader if the dataset opts in, else ``None``.

    ``None`` signals "fall back to ``super()._get_train_dataloader()``" so
    we preserve the parent class's default plumbing for any non-curriculum
    dataset (e.g. the existing RLVR pipeline pre-Component 3).
    """
    dataset = getattr(trainer, "train_dataset", None)
    if dataset is None:
        return None
    if not hasattr(dataset, "_weights"):
        return None
    return _build_weighted_dataloader(trainer)


def _curriculum_grpo_trainer_cls() -> type:
    """Build :class:`CurriculumGRPOTrainer` against the installed TRL.

    Done lazily because TRL pulls heavy deps (transformers, accelerate, …)
    that we don't want to require for ``import curriculum``.
    """
    from trl import GRPOTrainer  # noqa: PLC0415

    class CurriculumGRPOTrainer(GRPOTrainer):  # type: ignore[misc, valid-type]
        """``GRPOTrainer`` that opts into ``WeightedRandomSampler`` when the
        training dataset exposes ``_weights``. Falls back to the parent's
        default dataloader otherwise."""

        def _get_train_dataloader(self):  # type: ignore[override]
            dl = _maybe_curriculum_dataloader(self)
            if dl is not None:
                return dl
            return super()._get_train_dataloader()

    return CurriculumGRPOTrainer


def _curriculum_sft_trainer_cls() -> type:
    """Build :class:`CurriculumSFTTrainer` against the installed TRL."""
    from trl import SFTTrainer  # noqa: PLC0415

    class CurriculumSFTTrainer(SFTTrainer):  # type: ignore[misc, valid-type]
        """``SFTTrainer`` analog of :class:`CurriculumGRPOTrainer`."""

        def _get_train_dataloader(self):  # type: ignore[override]
            dl = _maybe_curriculum_dataloader(self)
            if dl is not None:
                return dl
            return super()._get_train_dataloader()

    return CurriculumSFTTrainer


def __getattr__(name: str):  # noqa: D401 — module-level lazy class loaders
    """Lazy class construction so ``import curriculum.sampler`` doesn't
    drag TRL into every consumer (e.g. unit tests for ``bins.py``).
    """
    if name == "CurriculumGRPOTrainer":
        return _curriculum_grpo_trainer_cls()
    if name == "CurriculumSFTTrainer":
        return _curriculum_sft_trainer_cls()
    raise AttributeError(name)
