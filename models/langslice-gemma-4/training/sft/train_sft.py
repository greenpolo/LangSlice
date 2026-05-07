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
from pathlib import Path
from typing import Any

import tomllib

from .collate import LangSliceCollator
from .dataset import load_examples, split_subject_aware
from .render import AtlasMetaCache, render_example

logger = logging.getLogger(__name__)
_REPO_ROOT = Path(__file__).resolve().parents[4]


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


def _resolve_test_images_root(path: Path) -> Path:
    if path.is_absolute() or path.exists():
        return path
    repo_relative = _REPO_ROOT / path
    if repo_relative.exists():
        return repo_relative
    return path


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
        return {"rendered": render_example(self.examples[idx], atlas_meta_cache=self.cache)}


def main(argv: list[str] | None = None) -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )
    args = _parse_args(argv)
    args.test_images_root = _resolve_test_images_root(args.test_images_root)
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
    """Heavy-import path. Loads model, builds trainer, runs trainer.train()."""
    sft_cfg = dict(config["sft"])
    lora_cfg = dict(config["lora"])

    # Lazy imports — keep dataset/render/collate unit tests cheap
    from unsloth import FastVisionModel  # noqa: I001 - Unsloth must patch before TRL import
    from trl import SFTConfig, SFTTrainer  # noqa: I001
    from rlvr.atlas_grid import build_atlas_grid

    from .eval import AgentLoopEvalCallback, BaselineEvalCallback

    # Pre-render the atlas grid once for the eval callbacks
    pairs = {(ex.atlas_name, ex.plane) for ex in train_ds.examples + eval_ds.examples}
    atlas_grid = build_atlas_grid(pairs)

    # Load base model + processor
    model, processor = FastVisionModel.from_pretrained(
        sft_cfg["base_model"],
        load_in_4bit=bool(sft_cfg.get("load_in_4bit", True)),
        max_seq_length=int(sft_cfg.get("max_seq_length", 16384)),
    )
    model = FastVisionModel.get_peft_model(
        model,
        finetune_vision_layers=bool(lora_cfg.get("finetune_vision_layers", False)),
        finetune_language_layers=bool(lora_cfg.get("finetune_language_layers", True)),
        finetune_attention_modules=bool(lora_cfg.get("finetune_attention_modules", True)),
        finetune_mlp_modules=bool(lora_cfg.get("finetune_mlp_modules", True)),
        r=int(lora_cfg.get("r", 16)),
        lora_alpha=int(lora_cfg.get("lora_alpha", 32)),
        use_gradient_checkpointing=lora_cfg.get("use_gradient_checkpointing", "unsloth"),
        random_state=seed,
    )
    FastVisionModel.for_training(model)

    collator = LangSliceCollator(
        processor=processor,
        max_seq_length=int(sft_cfg.get("max_seq_length", 16384)),
    )

    sft_config_kwargs = {
        k: v for k, v in sft_cfg.items()
        if k not in (
            "base_model",
            "load_in_4bit",
            "max_seq_length",
            "agent_eval_steps",
            "seed",
            "chat_template_kwargs",
            "dataset_kwargs",
            "remove_unused_columns",
        )
    }
    training_args = SFTConfig(
        output_dir=str(args.output_dir),
        seed=seed,
        # Critical: do not let TRL try to apply text-only assistant-only loss
        assistant_only_loss=False,
        # Critical for VLMs: no TRL truncation. The custom collator rejects
        # examples beyond max_seq_length before they reach the trainer.
        max_length=None,
        dataset_kwargs={"skip_prepare_dataset": True},
        remove_unused_columns=False,
        **sft_config_kwargs,
    )

    callbacks = [
        BaselineEvalCallback(
            processor=processor,
            atlas_grid=atlas_grid,
            test_images_root=args.test_images_root,
        ),
        AgentLoopEvalCallback(
            processor=processor,
            atlas_grid=atlas_grid,
            test_images_root=args.test_images_root,
            agent_eval_steps=int(sft_cfg.get("agent_eval_steps", 200)),
        ),
    ]

    trainer = SFTTrainer(
        model=model,                  # already a PeftModel — do NOT pass peft_config
        processing_class=processor,
        args=training_args,
        train_dataset=train_ds,
        eval_dataset=eval_ds,
        data_collator=collator,
        callbacks=callbacks,
    )
    # Unsloth can replace the data collator while wrapping SFTTrainer. The
    # LangSlice collator owns VLM chat-template rendering and assistant masking,
    # so force it back before the first dataloader fetch.
    trainer.data_collator = collator
    trainer.train()
    trainer.save_model(str(args.output_dir))
    processor.save_pretrained(str(args.output_dir))
    logger.info("Saved adapter + processor to %s", args.output_dir)


if __name__ == "__main__":
    main(sys.argv[1:])
