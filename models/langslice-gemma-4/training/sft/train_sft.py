"""Driver script: CLI entry point for SFT training of Gemma 4 E4B via Unsloth + TRL.

Usage:
    cd models/langslice-gemma-4/training && python -m sft.train_sft \
        --config configs/sft_default.toml \
        --dataset ../../../models/langslice-gemma-4/data/sft_examples.jsonl \
        --output-dir ../../../out/sft/run0

Heavy deps (unsloth, trl) are imported lazily inside main() so unit tests can
import sibling modules without a runtime install.
"""

from __future__ import annotations

import argparse
import logging
import sys
import tomllib
from pathlib import Path
from typing import Any

from .collate import LangSliceCollator
from .dataset import load_examples, split_subject_aware
from .eval import AgentLoopEvalCallback, BaselineEvalCallback
from .render import AtlasMetaCache, render_example

logger = logging.getLogger(__name__)


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="SFT for Gemma 4 E4B on langslice-native trace JSONL."
    )
    p.add_argument("--config", type=Path, required=True,
                   help="TOML config with [sft], [lora], [data] tables.")
    p.add_argument("--dataset", type=Path, required=True,
                   help="JSONL of langslice-native trace examples.")
    p.add_argument("--output-dir", type=Path, required=True,
                   help="Where to save the LoRA adapter + tokenizer/processor.")
    p.add_argument("--test-images-root", type=Path,
                   default=Path("references/TestImages"),
                   help="Root containing M01-M09 ground-truth-labeled test images.")
    p.add_argument("--seed", type=int, default=None,
                   help="Override config's seed.")
    p.add_argument("--dry-run", action="store_true",
                   help="Load everything but do not train (smoke for wiring).")
    return p.parse_args(argv)


def _load_config(path: Path) -> dict[str, dict[str, Any]]:
    with path.open("rb") as f:
        return tomllib.load(f)


def _build_datasets(args: argparse.Namespace, data_cfg: dict[str, Any]) -> tuple[list, list]:
    examples = load_examples(args.dataset)
    if args.dry_run and len({e.subject_id for e in examples}) < 2:
        logger.warning("--dry-run with <2 subjects: skipping subject-aware split")
        return examples, []
    train, eval_ = split_subject_aware(
        examples,
        holdout_fraction=float(data_cfg["holdout_fraction"]),
        seed=args.seed if args.seed is not None else 0,
    )
    logger.info(
        "Loaded %d examples (%d train, %d eval) from %s",
        len(examples), len(train), len(eval_), args.dataset,
    )
    return train, eval_


class _RenderedDataset:
    """torch.utils.data.Dataset shim: lazily renders examples on __getitem__."""

    def __init__(self, examples: list, atlas_meta_cache: AtlasMetaCache) -> None:
        self.examples = examples
        self.cache = atlas_meta_cache

    def __len__(self) -> int:
        return len(self.examples)

    def __getitem__(self, idx: int):
        return render_example(self.examples[idx], atlas_meta_cache=self.cache)


def main(argv: list[str] | None = None) -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )
    args = _parse_args(argv)
    config = _load_config(args.config)
    seed = args.seed if args.seed is not None else int(config["sft"].get("seed", 0))

    train_examples, eval_examples = _build_datasets(args, config["data"])

    cache = AtlasMetaCache()
    train_ds = _RenderedDataset(train_examples, cache)
    eval_ds = _RenderedDataset(eval_examples, cache)

    if args.dry_run:
        logger.info("--dry-run: skipping model load + training")
        return

    _train(args, config, train_ds, eval_ds, cache, seed)


def _train(args, config, train_ds, eval_ds, cache, seed: int) -> None:
    """Heavy-import path. Defined separately so dry-run never reaches it."""
    raise NotImplementedError("filled in by Task 13")


if __name__ == "__main__":
    main(sys.argv[1:])
