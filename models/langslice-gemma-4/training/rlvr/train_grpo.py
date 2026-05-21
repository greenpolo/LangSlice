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

RLVR is parked as of 2026-05-09 — kept for reference, not actively used. See
``README.md`` in this directory. The TOML config supplies all GRPOConfig
fields; CLI flags carry only the paths that change between runs.
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
    load_rlvr_allocation,
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


def _install_optional_dep_stubs() -> None:
    """TRL's ``grpo_trainer`` unconditionally imports a handful of optional
    third-party packages (Huawei Ascend NPU bindings, mergekit, etc.) that
    aren't part of our NVIDIA + Unsloth stack. On a clean install those
    imports raise ``ModuleNotFoundError`` before any training code runs.
    We pre-populate ``sys.modules`` with empty stubs (each with a real
    ``ModuleSpec`` so ``importlib.util.find_spec`` works) so the imports
    succeed. None of these code paths are exercised on our hardware.
    """
    import importlib.machinery
    import sys
    import types

    def _stub_module(name: str, *, is_package: bool) -> types.ModuleType:
        if name in sys.modules:
            return sys.modules[name]
        mod = types.ModuleType(name)
        spec = importlib.machinery.ModuleSpec(name, loader=None, is_package=is_package)
        mod.__spec__ = spec
        if is_package:
            mod.__path__ = []  # type: ignore[attr-defined]
        sys.modules[name] = mod
        return mod

    # vllm_ascend — Huawei NPU collectives, imported UNCONDITIONALLY by
    # trl/extras/vllm_client.py. Stub it so that import succeeds; we never
    # exercise the Ascend code path on this hardware.
    if "vllm_ascend" not in sys.modules:
        _stub_module("vllm_ascend", is_package=True)
        _stub_module("vllm_ascend.distributed", is_package=True)
        _stub_module("vllm_ascend.distributed.device_communicators", is_package=True)
        pyhccl = _stub_module(
            "vllm_ascend.distributed.device_communicators.pyhccl", is_package=False
        )

        class _PyHcclStub:
            def __init__(self, *args: Any, **kwargs: Any) -> None:
                raise RuntimeError("vllm_ascend stub — not available on this hardware")

        pyhccl.PyHcclCommunicator = _PyHcclStub  # type: ignore[attr-defined]

    # TRL's import_utils.py runs lines like:
    #     _mergekit_available = _is_package_available("mergekit")
    # The bundled transformers' `_is_package_available` returns a tuple
    # `(False, None)` when the package is missing — and the tuple is TRUTHY
    # in Python. So `is_<x>_available()` returns truthy and triggers
    # conditional imports inside callbacks/extras that then hard-fail.
    # Sweep every `_<x>_available` flag and coerce it to a plain bool so
    # all the conditionals in TRL's optional-dep gates work correctly.
    # Wrapped in try/except so unit tests that mock `trl` via sys.modules
    # (where trl is not a real package) don't blow up.
    try:
        import trl.import_utils as _trl_iu  # noqa: PLC0415

        for _attr in list(vars(_trl_iu).keys()):
            if _attr.startswith("_") and _attr.endswith("_available"):
                _val = getattr(_trl_iu, _attr)
                if isinstance(_val, tuple):
                    setattr(_trl_iu, _attr, bool(_val[0]))
        # This RLVR path uses Unsloth/Transformers native generation, not vLLM.
        # The base image's vLLM build can be present but API-incompatible with
        # bleeding-edge TRL, and TRL imports vLLM helpers during module import
        # whenever it thinks vLLM is available. Force those optional gates off.
        for _attr in ("_vllm_available", "_vllm_ascend_available"):
            if hasattr(_trl_iu, _attr):
                setattr(_trl_iu, _attr, False)
    except (ImportError, AttributeError):
        pass


def _filter_grpo_config_for_installed_trl(
    grpo_config_cls: type,
    grpo_cfg: dict[str, Any],
) -> dict[str, Any]:
    """Drop config keys unsupported by the installed TRL build.

    We run close to TRL main for agent features. Some fields land in different
    releases at different times; unsupported keys should not prevent a smoke
    run from reaching the trainer.

    Walks the full MRO and unions the explicit init parameters at every level
    that doesn't expose ``**kwargs``. Unsloth's compiled GRPOConfig subclass
    accepts ``**kwargs`` itself but forwards to TRL's stricter base, so we
    can't trust a single-level signature inspection.
    """
    import inspect

    # Walk the MRO and stop at the FIRST class whose __init__ does not accept
    # **kwargs. Unsloth's compiled GRPOConfig subclass declares **kwargs but
    # delegates to TRL's stricter base via super().__init__, and that base
    # is the one whose TypeError we have to dodge.
    accepted: set[str] | None = None
    try:
        mro = inspect.getmro(grpo_config_cls)
    except (AttributeError, TypeError):
        return grpo_cfg

    for cls in mro:
        try:
            sig = inspect.signature(cls.__init__)
        except (TypeError, ValueError):
            continue
        params = list(sig.parameters.values())
        if any(p.kind == inspect.Parameter.VAR_KEYWORD for p in params):
            # Permissive level — keep walking up to a stricter base.
            continue
        accepted = {p.name for p in params if p.name != "self"}
        break

    if accepted is None:
        # Every level was permissive (truly **kwargs all the way). Pass through.
        return grpo_cfg

    out: dict[str, Any] = {}
    dropped: list[str] = []
    for key, value in grpo_cfg.items():
        if key in accepted:
            out[key] = value
        else:
            dropped.append(key)
    if dropped:
        logger.warning(
            "Dropping unsupported GRPOConfig key(s) for installed TRL: %s",
            ", ".join(sorted(dropped)),
        )
    return out


# Backward-compatible alias retained so existing callers keep working.
_install_vllm_ascend_stub = _install_optional_dep_stubs


def _wrap_reward_with_curriculum_log(
    reward_fn: Any,
    *,
    logger: Any,
    section_bins: dict[str, int],
    round_idx: int,
) -> Any:
    """Return a reward callable that emits a JSONL log row per env.

    The wrapped callable matches the TRL reward contract
    ``func(completions, environments, **kwargs) -> list[float]``. After
    delegating to ``reward_fn`` it walks ``environments`` and records
    one row per env carrying ``(round, section_id, bin_idx, plane,
    abs_err_mm, accuracy_pct)``. The hook fires once per rollout, which
    is the same cadence GRPOTrainer's reward callbacks see.

    abs_err_mm is computed plane-locally (mean of per-position absolute
    error against the canonicalized truth). Accuracy_pct is the
    plane-relative metric slicebench reports.
    """

    def _logging_reward(
        completions: list[Any] | None = None,
        environments: list[Any] | None = None,
        **kwargs: Any,
    ) -> list[float]:
        rewards = reward_fn(
            completions=completions, environments=environments, **kwargs
        )
        envs = environments or []
        for env in envs:
            try:
                s = env._state  # noqa: SLF001 — same private read the reward already does
                section_id = getattr(s, "section_id", "") or ""
                if not section_id:
                    continue
                preds = s.submitted_positions_mm
                truths = s.ground_truth_positions_mm
                axis_span = float(s.pos_hi) - float(s.pos_lo)
                if preds is None or len(preds) != len(truths) or axis_span <= 0:
                    abs_err = axis_span if axis_span > 0 else 0.0
                else:
                    diffs = [abs(float(p) - float(t)) for p, t in zip(preds, truths, strict=True)]
                    abs_err = sum(diffs) / max(len(diffs), 1)
                accuracy_pct = (
                    max(0.0, 100.0 * (1.0 - abs_err / axis_span))
                    if axis_span > 0
                    else 0.0
                )
                logger.append(
                    {
                        "round": int(round_idx),
                        "section_id": section_id,
                        "bin_idx": int(section_bins.get(section_id, -1)),
                        "plane": str(s.plane),
                        "abs_err_mm": round(float(abs_err), 4),
                        "accuracy_pct": round(float(accuracy_pct), 3),
                    }
                )
            except (AttributeError, ValueError, TypeError, OSError):  # pragma: no cover
                # Logging must never break training; swallow per-env errors.
                continue
        return rewards

    _logging_reward.__name__ = getattr(reward_fn, "__name__", "position_reward")
    _logging_reward.__qualname__ = getattr(reward_fn, "__qualname__", "position_reward")
    return _logging_reward


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
        "--manifest-root",
        type=Path,
        default=Path("data/manifest"),
        help=(
            "Root containing allocations/<plane>/rlvr.jsonl and "
            "shards/<plane>/<dataset>.jsonl. Defaults to data/manifest."
        ),
    )
    p.add_argument(
        "--repo-root",
        type=Path,
        default=Path("."),
        help=(
            "Repo root used to resolve relative image_path entries from shard "
            "rows. Defaults to current working directory."
        ),
    )
    p.add_argument("--seed", type=int, default=0)
    p.add_argument(
        "--report-to",
        type=str,
        default=None,
        help="Override GRPOConfig report_to for smoke/debug runs. Use 'none' to disable.",
    )
    p.add_argument(
        "--curriculum-weights",
        type=Path,
        default=None,
        help=(
            "Optional path to a JSON file mapping section_id -> sampling weight. "
            "When set and the file exists, wraps the train dataset in a "
            "WeightedRowDataset and uses CurriculumGRPOTrainer with a "
            "WeightedRandomSampler. When unset or missing, training is "
            "byte-identical to the non-curriculum path."
        ),
    )
    return p.parse_args(argv)


def build_datasets(
    *,
    manifest_root: Path,
    repo_root: Path,
    data_cfg: dict[str, Any],
    seed: int,
    curriculum: bool = False,
) -> tuple[Any, Any, list[dict[str, Any]], list[dict[str, Any]], set[tuple[str, Plane]]]:
    """Assemble train + eval HF datasets with strict subject-level holdout.

    Returns ``(train_dataset, eval_dataset, train_rows, eval_rows, atlas_pairs)``.
    The eval dataset may be ``None`` if ``eval_holdout_every`` is non-positive.

    When ``curriculum=True`` the train dataset is built as a
    :class:`curriculum.sampler.WeightedRowDataset` (with all weights at 1.0
    by default — the caller is responsible for calling
    :func:`curriculum.weights.update_weighted_dataset` if a weights file
    is available).
    """
    rng = random.Random(seed)
    singles = load_rlvr_allocation(manifest_root, repo_root=repo_root)
    if not singles:
        raise RuntimeError(
            f"No single-slice examples found via {manifest_root}/allocations/"
            f"<plane>/rlvr.jsonl; check that the data agent has populated the "
            "RLVR allocation."
        )

    # Optional dataset cap for smoke runs. RowDataset decodes lazily, so the
    # full 14k allocation no longer pressures RAM — leave this unset for
    # production runs.
    max_examples = data_cfg.get("max_examples")
    if max_examples is not None and int(max_examples) > 0 and len(singles) > int(max_examples):
        rng.shuffle(singles)
        singles = singles[: int(max_examples)]

    eval_every = int(data_cfg.get("eval_holdout_every", 5))
    train_subjects, eval_subjects = split_subjects_for_holdout(
        singles, eval_holdout_every=eval_every
    )
    train_singles = [s for s in singles if s.subject_id in train_subjects]
    eval_singles = [s for s in singles if s.subject_id in eval_subjects]

    single_fraction = float(data_cfg.get("single_fraction", 1.0))
    clahe_fraction = float(data_cfg.get("clahe_fraction", 0.25))

    # Multi-slice (group) rollouts are off the table for now (see
    # feedback_multi_slice_direction_bug). When single_fraction==1.0 we skip
    # the group bin/window pass entirely so we don't pay for unused work.
    train_groups: list[Any] = []
    eval_groups: list[Any] = []
    if single_fraction < 1.0:
        group_size = int(data_cfg.get("group_size", 4))
        thickness_um = int(data_cfg.get("thickness_um", 30))
        train_groups = make_group_examples(
            train_singles, group_size=group_size, rng=rng, thickness_um=thickness_um
        )
        eval_groups = make_group_examples(
            eval_singles, group_size=group_size, rng=rng, thickness_um=thickness_um
        )

    train_rows = build_rlvr_rows(
        single_examples=train_singles,
        group_examples=train_groups,
        single_fraction=single_fraction,
        seed=seed,
        clahe_fraction=clahe_fraction,
    )
    eval_rows = build_rlvr_rows(
        single_examples=eval_singles,
        group_examples=eval_groups,
        single_fraction=single_fraction,
        seed=seed + 1,
        clahe_fraction=clahe_fraction,
    )

    if curriculum and train_rows:
        # Lazy import — keep curriculum out of the import path for non-curriculum runs.
        from curriculum.sampler import WeightedRowDataset  # noqa: PLC0415

        train_dataset = WeightedRowDataset(train_rows)
    else:
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
    if args.report_to is not None:
        grpo_cfg["report_to"] = "none" if args.report_to.lower() == "none" else args.report_to
    lora_cfg = dict(config.get("lora", {}))
    data_cfg = dict(config.get("data", {}))
    reward_cfg = dict(config.get("reward", {}))

    # --- Curriculum opt-in: only when --curriculum-weights points to an
    # existing file. Otherwise fall through to the default RowDataset
    # path so the pre-curriculum behaviour is byte-identical.
    curriculum_enabled = (
        args.curriculum_weights is not None and args.curriculum_weights.is_file()
    )
    if args.curriculum_weights is not None and not curriculum_enabled:
        logger.warning(
            "--curriculum-weights=%s does not exist; falling back to uniform sampling.",
            args.curriculum_weights,
        )

    # --- Dataset assembly with strict subject-level holdout --------------
    train_dataset, eval_dataset, train_rows, eval_rows, atlas_pairs = build_datasets(
        manifest_root=args.manifest_root,
        repo_root=args.repo_root,
        data_cfg=data_cfg,
        seed=args.seed,
        curriculum=curriculum_enabled,
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

    # --- Apply curriculum weights, if any --------------------------------
    if curriculum_enabled:
        from curriculum.weights import (  # noqa: PLC0415
            read_weights_json,
            update_weighted_dataset,
        )

        weights_map = read_weights_json(args.curriculum_weights)
        n_matched = update_weighted_dataset(train_dataset, weights_map)
        logger.info(
            "Loaded %d curriculum weights from %s (matched %d/%d train rows)",
            len(weights_map),
            args.curriculum_weights,
            n_matched,
            len(train_rows),
        )

    # --- Pre-render atlas grid for every (atlas, plane) pair in use ------
    logger.info("Pre-rendering atlas grid for %d (atlas, plane) pair(s)", len(atlas_pairs))
    atlas_grid = build_atlas_grid(atlas_pairs)

    # --- Reward function bound to configured normalized schedule ---------
    cutoff_frac = float(reward_cfg.get("cutoff_frac", 0.10))
    sigma_frac = float(reward_cfg.get("sigma_frac", 0.035))
    position_reward = make_position_reward(
        cutoff_frac=cutoff_frac,
        sigma_frac=sigma_frac,
    )

    # --- Optional per-rollout curriculum logging --------------------------
    # When curriculum is enabled, also wire a logger that writes one
    # JSONL row per rollout (round/section_id/bin_idx/plane/abs_err_mm/
    # accuracy_pct) to <output_dir>/curriculum_log.jsonl. Phase C reads
    # this back between rounds via the iterate.py hook.
    section_bin_lookup: dict[str, int] = {}
    curriculum_round = int(data_cfg.get("curriculum_round", 0))
    if curriculum_enabled:
        try:
            from curriculum.bins import compute_section_bins  # noqa: PLC0415
            from rlvr.dataset import SingleSliceExample as _SSE  # noqa: PLC0415

            # Reconstruct a per-row stub so we can run compute_section_bins
            # without re-running the (slow) load_rlvr_allocation walk. Single
            # rows carry section_id+atlas_name+plane+ground_truth_positions_mm
            # — group rows are skipped here (group section_ids are composite
            # and the per-bin curriculum currently targets singles only).
            singles_stub = [
                _SSE(
                    image_path=Path("/dev/null"),
                    atlas_name=str(r["atlas_name"]),
                    plane=str(r["plane"]),  # type: ignore[arg-type]
                    ap_mm=float(r["ground_truth_positions_mm"][0]),
                    subject_id=str(r["subject_id"]),
                    section_id=str(r["section_id"]),
                )
                for r in train_rows
                if r["kind"] == "single"
            ]
            section_bin_lookup = compute_section_bins(singles_stub)
        except (ImportError, ModuleNotFoundError, RuntimeError, OSError, KeyError) as exc:
            logger.warning(
                "Curriculum logging: bin computation skipped (%s); "
                "log rows will record bin_idx=-1.",
                exc,
            )

        from curriculum.log import CurriculumLogger  # noqa: PLC0415

        log_path = args.output_dir / "curriculum_log.jsonl"
        curriculum_logger = CurriculumLogger(log_path)
        position_reward = _wrap_reward_with_curriculum_log(
            position_reward,
            logger=curriculum_logger,
            section_bins=section_bin_lookup,
            round_idx=curriculum_round,
        )
        logger.info(
            "Curriculum logging enabled: round=%d -> %s (section_bins=%d)",
            curriculum_round,
            log_path,
            len(section_bin_lookup),
        )

    # --- Heavy imports happen here so unit tests can skip them -----------
    from unsloth import FastVisionModel  # noqa: PLC0415
    _install_optional_dep_stubs()
    from trl import GRPOConfig, GRPOTrainer  # noqa: PLC0415

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

    grpo_cfg = _filter_grpo_config_for_installed_trl(GRPOConfig, grpo_cfg)
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

    if curriculum_enabled:
        from curriculum.sampler import CurriculumGRPOTrainer  # noqa: PLC0415

        trainer = CurriculumGRPOTrainer(**trainer_kwargs)
    else:
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
