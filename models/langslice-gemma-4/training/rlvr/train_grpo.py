"""Driver for the LangSlice multi-turn GRPO run.

Ports ``examples/scripts/openenv/carla_vlm_gemma.py`` from TRL: Unsloth
``FastVisionModel`` + LoRA, ``GRPOTrainer`` with our env factory, a single
gated-linear closeness reward (``rewards.position_reward``). Heavy deps
(unsloth, trl, datasets) are imported lazily inside ``main`` so unit tests
can import sibling modules without a runtime install.

Usage
-----
    # Phase A - single-slice only, from the repo root:
    python -m langslice_rlvr \
        --config models/langslice-gemma-4/training/configs/grpo_pilot.toml \
        --sft-model out/sft/gemma4-e4b-langslice \
        --output-dir out/rlvr/phase_a \
        --test-images-root references/TestImages

    # Phase B — mixed single+group, resume from Phase A adapter:
    python -m langslice_rlvr \
        --config models/langslice-gemma-4/training/configs/grpo_phase_b.toml \
        --sft-model out/sft/gemma4-e4b-langslice \
        --resume-from-adapter out/rlvr/phase_a \
        --output-dir out/rlvr/phase_b \
        --test-images-root references/TestImages

The repo-root ``langslice_rlvr`` shim makes ``python -m langslice_rlvr`` work
without installing ``rlvr`` as a package. The installed console script still
uses the equivalent ``src/langslice_rlvr`` shim. The TOML config supplies all
GRPOConfig fields; CLI flags carry only the paths that change between runs.
"""

from __future__ import annotations

import argparse
import json
import logging
import random
import sys
from pathlib import Path
from typing import Any, Literal

import tomllib

from .atlas_grid import build_atlas_grid
from .dataset import (
    build_rlvr_rows,
    load_test_images,
    make_group_examples,
    split_subjects_for_holdout,
    to_hf_dataset,
)
from .env import LangSliceEstimateEnv
from .rewards import make_position_reward

Plane = Literal["coronal", "sagittal", "horizontal"]

logger = logging.getLogger(__name__)


def _load_config(path: Path) -> dict[str, Any]:
    with path.open("rb") as f:
        return tomllib.load(f)


def _adapter_base_model_name(adapter_dir: Path) -> str | None:
    """Return the PEFT adapter's base model id, or None for non-adapter paths."""
    adapter_config_path = adapter_dir / "adapter_config.json"
    if not adapter_config_path.is_file():
        return None
    try:
        config = json.loads(adapter_config_path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise ValueError(f"Invalid PEFT adapter config: {adapter_config_path}") from exc
    base_model = config.get("base_model_name_or_path")
    if not isinstance(base_model, str) or not base_model.strip():
        raise ValueError(
            f"PEFT adapter config missing base_model_name_or_path: {adapter_config_path}"
        )
    return base_model


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Run multi-turn GRPO on the LangSlice estimation env.")
    p.add_argument("--config", type=Path, required=True, help="TOML config with GRPOConfig fields.")
    p.add_argument(
        "--sft-model",
        type=Path,
        required=True,
        help="Path or HF id of the post-SFT Gemma 4 E4B checkpoint to start from.",
    )
    p.add_argument(
        "--resume-from-adapter",
        type=Path,
        default=None,
        help=(
            "Optional path to a previously-saved LoRA adapter. When set, the "
            "post-SFT base is loaded from --sft-model and the adapter is "
            "attached as a trainable PEFT adapter, allowing Phase B to start "
            "from the Phase A adapter."
        ),
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


def build_datasets(
    *,
    test_images_root: Path,
    data_cfg: dict[str, Any],
    seed: int,
) -> tuple[Any, Any, list[dict[str, Any]], list[dict[str, Any]], set[tuple[str, Plane]]]:
    """Assemble train + eval HF datasets with strict subject-level holdout.

    Returns ``(train_dataset, eval_dataset, train_rows, eval_rows, atlas_pairs)``.
    The eval dataset may be ``None`` if ``eval_holdout_every`` is non-positive.
    """
    rng = random.Random(seed)
    singles = load_test_images(test_images_root)
    if not singles:
        raise RuntimeError(
            f"No single-slice examples found under {test_images_root}; "
            "expected M01..M09 subdirs each with ground_truth.json."
        )

    eval_every = int(data_cfg.get("eval_holdout_every", 5))
    train_subjects, eval_subjects = split_subjects_for_holdout(
        singles, eval_holdout_every=eval_every
    )
    train_singles = [s for s in singles if s.subject_id in train_subjects]
    eval_singles = [s for s in singles if s.subject_id in eval_subjects]

    group_size = int(data_cfg.get("group_size", 4))
    thickness_um = int(data_cfg.get("thickness_um", 30))
    train_groups = make_group_examples(
        train_singles, group_size=group_size, rng=rng, thickness_um=thickness_um
    )
    eval_groups = make_group_examples(
        eval_singles, group_size=group_size, rng=rng, thickness_um=thickness_um
    )

    single_fraction = float(data_cfg.get("single_fraction", 0.7))
    train_rows = build_rlvr_rows(
        single_examples=train_singles,
        group_examples=train_groups,
        single_fraction=single_fraction,
        seed=seed,
    )
    eval_rows = build_rlvr_rows(
        single_examples=eval_singles,
        group_examples=eval_groups,
        single_fraction=single_fraction,
        seed=seed + 1,
    )

    train_dataset = to_hf_dataset(train_rows) if train_rows else None
    eval_dataset = to_hf_dataset(eval_rows) if eval_rows else None
    atlas_pairs: set[tuple[str, Plane]] = {
        (r["atlas_name"], r["plane"]) for r in train_rows + eval_rows
    }
    return train_dataset, eval_dataset, train_rows, eval_rows, atlas_pairs


def main(argv: list[str] | None = None) -> None:
    args = _parse_args(argv)
    config = _load_config(args.config)

    grpo_cfg = dict(config.get("grpo", {}))
    lora_cfg = dict(config.get("lora", {}))
    data_cfg = dict(config.get("data", {}))
    reward_cfg = dict(config.get("reward", {}))

    # --- Dataset assembly with strict subject-level holdout --------------
    train_dataset, eval_dataset, train_rows, eval_rows, atlas_pairs = build_datasets(
        test_images_root=args.test_images_root,
        data_cfg=data_cfg,
        seed=args.seed,
    )
    if train_dataset is None:
        raise RuntimeError("Train dataset is empty after holdout — adjust eval_holdout_every.")
    logger.info(
        "Assembled %d train rows (%d single, %d group), %d eval rows",
        len(train_rows),
        sum(1 for r in train_rows if r["kind"] == "single"),
        sum(1 for r in train_rows if r["kind"] == "group"),
        len(eval_rows),
    )

    # --- Pre-render atlas grid for every (atlas, plane) pair in use ------
    logger.info("Pre-rendering atlas grid for %d (atlas, plane) pair(s)", len(atlas_pairs))
    atlas_grid = build_atlas_grid(atlas_pairs)

    # --- Reward function bound to configured window ----------------------
    window_mm = float(reward_cfg.get("window_mm", 0.100))
    position_reward = make_position_reward(window_mm=window_mm)

    # --- Heavy imports happen here so unit tests can skip them -----------
    from trl import GRPOConfig, GRPOTrainer  # noqa: PLC0415
    from unsloth import FastVisionModel  # noqa: PLC0415

    max_seq_length = int(grpo_cfg.pop("max_seq_length", 4096))
    load_in_4bit = bool(grpo_cfg.pop("load_in_4bit", True))
    sft_adapter_base = _adapter_base_model_name(args.sft_model)
    model_name_or_path = sft_adapter_base or str(args.sft_model)
    model, processor = FastVisionModel.from_pretrained(
        model_name_or_path,
        max_seq_length=max_seq_length,
        load_in_4bit=load_in_4bit,
        fast_inference=False,  # spec §2 — Unsloth-native, no vLLM
    )
    # When resuming Phase B, attach the saved Phase A adapter as trainable
    # weights on the post-SFT base. Otherwise create a fresh LoRA adapter.
    if args.resume_from_adapter is not None:
        from peft import PeftModel  # noqa: PLC0415

        logger.info("Resuming trainable LoRA adapter from %s", args.resume_from_adapter)
        model = PeftModel.from_pretrained(
            model,
            str(args.resume_from_adapter),
            is_trainable=True,
        )
    elif sft_adapter_base is not None:
        from peft import PeftModel  # noqa: PLC0415

        logger.info("Loading SFT LoRA adapter from %s", args.sft_model)
        model = PeftModel.from_pretrained(
            model,
            str(args.sft_model),
            is_trainable=True,
        )
    else:
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

    trainer_kwargs: dict[str, Any] = {
        "model": model,
        "processing_class": processor,
        "train_dataset": train_dataset,
        "reward_funcs": [position_reward],
        "args": training_args,
        "environment_factory": lambda: LangSliceEstimateEnv(atlas_grid=atlas_grid),
    }
    if eval_dataset is not None:
        trainer_kwargs["eval_dataset"] = eval_dataset

    trainer = GRPOTrainer(**trainer_kwargs)
    trainer.train()
    trainer.save_model(str(args.output_dir))
    logger.info("Saved adapter to %s", args.output_dir)


if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )
    main(sys.argv[1:])
