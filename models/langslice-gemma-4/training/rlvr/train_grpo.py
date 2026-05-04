"""Driver for the LangSlice multi-turn GRPO run.

Ports ``examples/scripts/openenv/carla_vlm_gemma.py`` from TRL: Unsloth
``FastVisionModel`` + LoRA, ``GRPOTrainer`` with our env factory, three
custom reward functions. Heavy deps (unsloth, trl, datasets) are imported
lazily inside ``main`` so unit tests can import sibling modules without a
runtime install.

Usage
-----
    python -m rlvr.train_grpo \\
        --config models/langslice-gemma-4/training/configs/grpo_default.toml \\
        --sft-model out/sft/gemma4-e4b-langslice \\
        --output-dir out/rlvr/run0 \\
        --test-images-root references/TestImages

The TOML config supplies all GRPOConfig fields; CLI flags only carry paths
that change between runs.
"""

from __future__ import annotations

import argparse
import logging
import random
import sys
from pathlib import Path
from typing import Any

import tomllib

from langslice_harness.atlas.space import Plane

from .atlas_grid import build_atlas_grid
from .dataset import (
    build_rlvr_rows,
    load_test_images,
    make_group_examples,
    to_hf_dataset,
)
from .env import LangSliceEstimateEnv
from .rewards import (
    format_compliance_reward,
    length_budget_reward,
    position_accuracy_reward,
)

logger = logging.getLogger(__name__)


def _load_config(path: Path) -> dict[str, Any]:
    with path.open("rb") as f:
        return tomllib.load(f)


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Run multi-turn GRPO on the LangSlice estimation env.")
    p.add_argument("--config", type=Path, required=True, help="TOML config with GRPOConfig fields.")
    p.add_argument(
        "--sft-model",
        type=Path,
        required=True,
        help="Path or HF id of the post-SFT Gemma 4 E4B checkpoint to start from.",
    )
    p.add_argument("--output-dir", type=Path, required=True, help="Where to save adapter + logs.")
    p.add_argument(
        "--test-images-root",
        type=Path,
        default=Path("references/TestImages"),
        help="Root containing M01-M09 (ground_truth.json) subdirectories.",
    )
    p.add_argument("--seed", type=int, default=0)
    return p.parse_args(argv)


def main(argv: list[str] | None = None) -> None:
    args = _parse_args(argv)
    config = _load_config(args.config)
    rng = random.Random(args.seed)

    grpo_cfg = dict(config.get("grpo", {}))
    lora_cfg = dict(config.get("lora", {}))
    data_cfg = dict(config.get("data", {}))

    # --- Dataset assembly ------------------------------------------------
    singles = load_test_images(args.test_images_root)
    if not singles:
        raise RuntimeError(
            f"No single-slice examples found under {args.test_images_root}; "
            "expected M01..M09 subdirs each with ground_truth.json."
        )
    groups = make_group_examples(
        singles,
        group_size=int(data_cfg.get("group_size", 4)),
        rng=rng,
        thickness_um=int(data_cfg.get("thickness_um", 30)),
    )
    rows = build_rlvr_rows(
        single_examples=singles,
        group_examples=groups,
        single_fraction=float(data_cfg.get("single_fraction", 0.7)),
        seed=args.seed,
    )
    train_dataset = to_hf_dataset(rows)
    logger.info(
        "Assembled %d rows (%d single, %d group)",
        len(rows),
        sum(1 for r in rows if r["kind"] == "single"),
        sum(1 for r in rows if r["kind"] == "group"),
    )

    # --- Pre-render atlas grid for every (atlas, plane) pair in use ------
    pairs: set[tuple[str, Plane]] = {(r["atlas_name"], r["plane"]) for r in rows}
    logger.info("Pre-rendering atlas grid for %d (atlas, plane) pair(s)", len(pairs))
    atlas_grid = build_atlas_grid(pairs)

    # --- Heavy imports happen here so unit tests can skip them -----------
    from trl import GRPOConfig, GRPOTrainer  # noqa: PLC0415
    from unsloth import FastVisionModel  # noqa: PLC0415

    model, tokenizer = FastVisionModel.from_pretrained(
        str(args.sft_model),
        load_in_4bit=bool(grpo_cfg.pop("load_in_4bit", True)),
        fast_inference=False,  # spec §2 — Unsloth-native, no vLLM
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
        random_state=args.seed,
    )

    training_args = GRPOConfig(output_dir=str(args.output_dir), seed=args.seed, **grpo_cfg)

    trainer = GRPOTrainer(
        model=model,
        processing_class=tokenizer,
        train_dataset=train_dataset,
        reward_funcs=[position_accuracy_reward, format_compliance_reward, length_budget_reward],
        args=training_args,
        environment_factory=lambda: LangSliceEstimateEnv(atlas_grid=atlas_grid),
    )
    trainer.train()
    trainer.save_model(str(args.output_dir))
    logger.info("Saved adapter to %s", args.output_dir)


if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )
    main(sys.argv[1:])
