"""Held-out single-turn eval for Lane A.

Reads a terminal-state JSONL (the same shape :mod:`dataset` consumes for
training), samples N completions per row from the given checkpoint, then
reports the diagnostics the plan says to gate any >100-step GRPO run on:

* greedy (n=1) MAE
* best-of-N MAE
* parse failure rate
* out-of-range rate
* reward mean / std (with the active schedule)
* per-plane breakdown

The metrics functions are pure-Python and unit-testable. The generation loop
uses Unsloth-native ``FastVisionModel`` inference because the GRPO trainer
already requires ``fast_inference=False`` for Gemma 4 — running eval the same
way avoids any vLLM-vs-Unsloth tokenizer drift.

Usage
-----
::

    PYTHONPATH=models/langslice-gemma-4/training \\
    python -m single_turn_rl.eval_single_turn \\
        --model out/rlvr_single_turn/terminal_smoke \\
        --sft-model out/sft/docker-sft-1011-merged-bf16 \\
        --eval-states out/single_turn_rl/terminal_eval.jsonl \\
        --num-generations 1 4 8

If ``--model`` is a PEFT adapter, also pass ``--sft-model`` so the base can
be loaded and the adapter attached. If ``--model`` is a merged checkpoint,
pass it as ``--sft-model`` and omit the flag.
"""

from __future__ import annotations

import argparse
import json
import logging
import math
import statistics
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any

from rlvr.train_grpo import _adapter_base_model_name

from .dataset import load_atlas_reference_image
from .prompts import (
    ATLAS_CAPTION_TEMPLATE,
    TARGET_CAPTION,
    USER_INSTRUCTION,
    build_single_turn_system_prompt,
)
from .rewards import (
    DEFAULT_CUTOFF_FRAC,
    DEFAULT_FORMAT_PENALTY,
    DEFAULT_OUT_OF_RANGE_REWARD,
    DEFAULT_SIGMA_FRAC,
    _ParseError,
    parse_position_mm,
    score_completion,
)
from .terminal_states import TerminalState, read_terminal_states

logger = logging.getLogger(__name__)


# --- Metrics ---------------------------------------------------------------


def _maybe_parse(text: str) -> float | None:
    try:
        return parse_position_mm(text)
    except _ParseError:
        return None


def aggregate_metrics(
    per_state_results: list[dict[str, Any]],
    *,
    cutoff_frac: float,
    sigma_frac: float,
    format_penalty: float,
    out_of_range_reward: float,
) -> dict[str, Any]:
    """Aggregate per-state generation results into eval-grade metrics.

    Each ``per_state_results`` entry is ``{section_id, plane, gt, valid_range,
    completions: [text, ...], scores: [float, ...], predictions: [float|None, ...]}``.
    """
    overall_abs_errs_greedy: list[float] = []
    overall_abs_errs_best: list[float] = []
    parse_failures = 0
    out_of_range = 0
    total_completions = 0
    rewards: list[float] = []
    per_plane: dict[str, dict[str, list[float]]] = defaultdict(
        lambda: {"greedy_abs_err": [], "best_abs_err": [], "rewards": []}
    )

    for r in per_state_results:
        plane = str(r["plane"])
        gt = float(r["gt"])
        pos_lo, pos_hi = float(r["valid_range"][0]), float(r["valid_range"][1])
        preds: list[float | None] = list(r["predictions"])
        scores: list[float] = list(r["scores"])

        # Greedy = first sample
        first_pred = preds[0] if preds else None
        if first_pred is None:
            greedy_err = pos_hi - pos_lo  # parse-fail penalty: full plane span
        elif first_pred < pos_lo or first_pred > pos_hi:
            greedy_err = max(abs(first_pred - gt), pos_hi - pos_lo)
        else:
            greedy_err = abs(first_pred - gt)
        overall_abs_errs_greedy.append(greedy_err)
        per_plane[plane]["greedy_abs_err"].append(greedy_err)

        # Best-of-N: minimum abs error across in-range parses; if none parse,
        # fall back to greedy_err so the metric remains comparable.
        valid_errs: list[float] = []
        for pred in preds:
            if pred is None:
                continue
            if pred < pos_lo or pred > pos_hi:
                continue
            valid_errs.append(abs(pred - gt))
        best_err = min(valid_errs) if valid_errs else greedy_err
        overall_abs_errs_best.append(best_err)
        per_plane[plane]["best_abs_err"].append(best_err)

        for pred in preds:
            total_completions += 1
            if pred is None:
                parse_failures += 1
            elif pred < pos_lo or pred > pos_hi:
                out_of_range += 1
        rewards.extend(scores)
        per_plane[plane]["rewards"].extend(scores)

    def _mean(xs: list[float]) -> float:
        return float(statistics.mean(xs)) if xs else float("nan")

    def _stdev(xs: list[float]) -> float:
        return float(statistics.pstdev(xs)) if len(xs) >= 1 else float("nan")

    overall = {
        "n_states": len(per_state_results),
        "n_completions": total_completions,
        "mae_greedy_mm": _mean(overall_abs_errs_greedy),
        "mae_best_of_n_mm": _mean(overall_abs_errs_best),
        "parse_failure_rate": parse_failures / max(total_completions, 1),
        "out_of_range_rate": out_of_range / max(total_completions, 1),
        "reward_mean": _mean(rewards),
        "reward_std": _stdev(rewards),
    }
    by_plane: dict[str, dict[str, float]] = {}
    for plane, buckets in per_plane.items():
        by_plane[plane] = {
            "n_states": len(buckets["greedy_abs_err"]),
            "mae_greedy_mm": _mean(buckets["greedy_abs_err"]),
            "mae_best_of_n_mm": _mean(buckets["best_abs_err"]),
            "reward_mean": _mean(buckets["rewards"]),
        }
    return {
        "overall": overall,
        "per_plane": by_plane,
        "schedule": {
            "cutoff_frac": cutoff_frac,
            "sigma_frac": sigma_frac,
            "format_penalty": format_penalty,
            "out_of_range_reward": out_of_range_reward,
        },
    }


# --- Generation loop -------------------------------------------------------


def _build_chat_for_inference(state: TerminalState) -> list[dict[str, Any]]:
    """Mirror the dataset's chat layout but as a flat message list for inference."""
    from rlvr.dataset import species_from_atlas_name  # noqa: PLC0415

    species = species_from_atlas_name(state.atlas_name)
    system_prompt = build_single_turn_system_prompt(
        atlas_name=state.atlas_name,
        plane=state.plane,
        pos_lo=state.valid_range_mm[0],
        pos_hi=state.valid_range_mm[1],
        species=species,
    )
    user_blocks: list[dict[str, Any]] = [
        {"type": "image"},
        {"type": "text", "text": TARGET_CAPTION},
    ]
    for pos_mm in state.fetched_positions_mm[: len(state.atlas_image_paths)]:
        user_blocks.append({"type": "image"})
        user_blocks.append(
            {"type": "text", "text": ATLAS_CAPTION_TEMPLATE.format(position_mm=pos_mm)}
        )
    user_blocks.append({"type": "text", "text": USER_INSTRUCTION})
    return [
        {"role": "system", "content": [{"type": "text", "text": system_prompt}]},
        {"role": "user", "content": user_blocks},
    ]


def _decode_images(state: TerminalState, repo_root: Path) -> list[Any]:
    from rlvr.dataset import (  # noqa: PLC0415
        _atlas_in_plane_long_edge,
        preprocess_query_image,
    )

    long_edge = _atlas_in_plane_long_edge(state.atlas_name, state.plane)
    images = [
        preprocess_query_image(
            repo_root / state.query_image_path,
            atlas_long_edge=long_edge,
            apply_clahe=False,
        )
    ]
    for atlas_rel in state.atlas_image_paths:
        images.append(load_atlas_reference_image(repo_root / atlas_rel))
    return images


def _run_generation(
    *,
    model: Any,
    processor: Any,
    state: TerminalState,
    repo_root: Path,
    n: int,
    temperature: float,
    top_p: float,
    max_new_tokens: int,
) -> list[str]:
    """Sample ``n`` completions from the model for one terminal state."""
    chat = _build_chat_for_inference(state)
    images = _decode_images(state, repo_root)
    text = processor.apply_chat_template(chat, add_generation_prompt=True)
    inputs = processor(
        text=text,
        images=images,
        return_tensors="pt",
    )
    # Move to model device.
    inputs = {k: v.to(model.device) if hasattr(v, "to") else v for k, v in inputs.items()}
    completions: list[str] = []
    do_sample = n > 1 or temperature > 0
    for _ in range(n):
        out_ids = model.generate(
            **inputs,
            do_sample=do_sample,
            temperature=temperature if do_sample else 1.0,
            top_p=top_p if do_sample else 1.0,
            max_new_tokens=max_new_tokens,
        )
        # Strip the prompt prefix.
        prompt_len = inputs["input_ids"].shape[-1]
        gen_ids = out_ids[0][prompt_len:]
        completions.append(processor.decode(gen_ids, skip_special_tokens=True))
    return completions


def evaluate(
    *,
    model: Any,
    processor: Any,
    states: list[TerminalState],
    repo_root: Path,
    num_generations: int,
    temperature: float,
    top_p: float,
    max_new_tokens: int,
    cutoff_frac: float,
    sigma_frac: float,
    format_penalty: float,
    out_of_range_reward: float,
) -> list[dict[str, Any]]:
    """Run generation + scoring for every state. Returns per-state result rows."""
    per_state: list[dict[str, Any]] = []
    for i, state in enumerate(states):
        completions = _run_generation(
            model=model,
            processor=processor,
            state=state,
            repo_root=repo_root,
            n=num_generations,
            temperature=temperature,
            top_p=top_p,
            max_new_tokens=max_new_tokens,
        )
        scores: list[float] = []
        predictions: list[float | None] = []
        for c in completions:
            scores.append(
                score_completion(
                    completion=c,
                    ground_truth_mm=state.ground_truth_mm,
                    valid_range_mm=state.valid_range_mm,
                    cutoff_frac=cutoff_frac,
                    sigma_frac=sigma_frac,
                    format_penalty=format_penalty,
                    out_of_range_reward=out_of_range_reward,
                )
            )
            predictions.append(_maybe_parse(c))
        per_state.append({
            "section_id": state.section_id,
            "plane": state.plane,
            "gt": state.ground_truth_mm,
            "valid_range": state.valid_range_mm,
            "completions": completions,
            "predictions": predictions,
            "scores": scores,
        })
        if (i + 1) % 25 == 0 or i == 0:
            print(f"[eval] state {i + 1}/{len(states)}", file=sys.stderr, flush=True)
    return per_state


# --- CLI -------------------------------------------------------------------


def _print_metrics(metrics: dict[str, Any], n_gen: int) -> None:
    overall = metrics["overall"]
    print(f"\n=== num_generations={n_gen} ===")
    print(f"  states: {overall['n_states']}")
    print(f"  completions: {overall['n_completions']}")
    print(f"  greedy MAE (mm): {overall['mae_greedy_mm']:.4f}")
    print(f"  best-of-N MAE (mm): {overall['mae_best_of_n_mm']:.4f}")
    print(f"  parse-failure rate: {overall['parse_failure_rate']:.3%}")
    print(f"  out-of-range rate: {overall['out_of_range_rate']:.3%}")
    print(f"  reward mean: {overall['reward_mean']:.4f}")
    print(f"  reward std: {overall['reward_std']:.4f}")
    if metrics["per_plane"]:
        print("  per plane:")
        for plane, p in metrics["per_plane"].items():
            print(
                f"    {plane}: n={p['n_states']} "
                f"greedy={p['mae_greedy_mm']:.4f} mm "
                f"best={p['mae_best_of_n_mm']:.4f} mm "
                f"reward_mean={p['reward_mean']:.4f}"
            )


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Held-out single-turn eval for Lane A.")
    p.add_argument("--model", type=Path, required=True,
                   help="Path or HF id of the model to evaluate. May be a "
                   "merged checkpoint or a PEFT adapter directory.")
    p.add_argument("--sft-model", type=Path, default=None,
                   help="Required if --model is a PEFT adapter: the base SFT "
                   "checkpoint the adapter attaches to.")
    p.add_argument("--eval-states", type=Path, required=True,
                   help="Path to the held-out terminal-state JSONL.")
    p.add_argument("--repo-root", type=Path, default=Path("."),
                   help="Repo root for resolving repo-relative image paths.")
    p.add_argument("--num-generations", type=int, nargs="+", default=[1, 4, 8],
                   help="N values to sweep (default: 1 4 8).")
    p.add_argument("--temperature", type=float, default=1.0)
    p.add_argument("--top-p", type=float, default=1.0)
    p.add_argument("--max-new-tokens", type=int, default=128)
    p.add_argument("--cutoff-frac", type=float, default=DEFAULT_CUTOFF_FRAC)
    p.add_argument("--sigma-frac", type=float, default=DEFAULT_SIGMA_FRAC)
    p.add_argument("--format-penalty", type=float, default=DEFAULT_FORMAT_PENALTY)
    p.add_argument("--out-of-range-reward", type=float, default=DEFAULT_OUT_OF_RANGE_REWARD)
    p.add_argument("--max-states", type=int, default=None,
                   help="Optional cap for smoke evals.")
    p.add_argument("--max-seq-length", type=int, default=4096)
    p.add_argument("--load-in-4bit", action="store_true", default=True)
    p.add_argument("--no-load-in-4bit", dest="load_in_4bit", action="store_false")
    p.add_argument("--output-dir", type=Path, default=None,
                   help="Optional directory to write per-N JSON metric reports.")
    return p.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = _parse_args(argv)
    states = list(read_terminal_states(args.eval_states))
    if args.max_states is not None:
        states = states[: int(args.max_states)]
    if not states:
        raise RuntimeError(f"No terminal states in {args.eval_states}")
    logger.info("Loaded %d terminal states", len(states))

    # Heavy imports lazy so unit tests can import sibling functions.
    from unsloth import FastVisionModel  # noqa: PLC0415

    sft_adapter_base = _adapter_base_model_name(args.model)
    if sft_adapter_base is not None and args.sft_model is None:
        # The model is an adapter but no SFT base was provided.
        base_model_name = sft_adapter_base
    elif args.sft_model is not None:
        base_model_name = str(args.sft_model)
    else:
        base_model_name = str(args.model)

    model, processor = FastVisionModel.from_pretrained(
        base_model_name,
        max_seq_length=int(args.max_seq_length),
        load_in_4bit=bool(args.load_in_4bit),
        fast_inference=False,
    )
    if sft_adapter_base is not None or (
        args.sft_model is not None and Path(args.model).is_dir()
        and (Path(args.model) / "adapter_config.json").is_file()
    ):
        from peft import PeftModel  # noqa: PLC0415

        logger.info("Loading adapter %s on top of base %s", args.model, base_model_name)
        model = PeftModel.from_pretrained(model, str(args.model), is_trainable=False)
    FastVisionModel.for_inference(model)

    if args.output_dir is not None:
        args.output_dir.mkdir(parents=True, exist_ok=True)

    for n_gen in args.num_generations:
        per_state = evaluate(
            model=model,
            processor=processor,
            states=states,
            repo_root=args.repo_root,
            num_generations=int(n_gen),
            temperature=float(args.temperature),
            top_p=float(args.top_p),
            max_new_tokens=int(args.max_new_tokens),
            cutoff_frac=float(args.cutoff_frac),
            sigma_frac=float(args.sigma_frac),
            format_penalty=float(args.format_penalty),
            out_of_range_reward=float(args.out_of_range_reward),
        )
        metrics = aggregate_metrics(
            per_state,
            cutoff_frac=float(args.cutoff_frac),
            sigma_frac=float(args.sigma_frac),
            format_penalty=float(args.format_penalty),
            out_of_range_reward=float(args.out_of_range_reward),
        )
        _print_metrics(metrics, n_gen)
        if args.output_dir is not None:
            (args.output_dir / f"eval_n{n_gen}.json").write_text(
                json.dumps(metrics, indent=2)
            )

    return 0


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    raise SystemExit(main(sys.argv[1:]))
