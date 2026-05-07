"""SFT-time evaluation: agent-loop callbacks (baseline + periodic) and metric utilities."""

from __future__ import annotations

import json
import logging
import re
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any

from transformers import TrainerCallback

# Make the rlvr package importable from the SFT package when this module is
# loaded outside pytest (driver script invocation).
_TRAINING_ROOT = Path(__file__).resolve().parents[1]
if str(_TRAINING_ROOT) not in sys.path:
    sys.path.insert(0, str(_TRAINING_ROOT))

# Renderer helpers — imported into module namespace so eval-callback tests can
# monkeypatch ``AtlasMetaCache`` (the agent-loop helper constructs one inline).
from sft.render import AtlasMetaCache, build_system_prompt, build_tools_schema

if TYPE_CHECKING:
    from rlvr.env import LangSliceEstimateEnv  # noqa: F401

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class ParsedSubmit:
    is_parseable: bool
    position_mm: float | None = None


def parse_submit_call(arguments_json: str, *, expected_kind: str) -> ParsedSubmit:
    """Parse a submit_*_estimate call's arguments string. Returns is_parseable=False on any failure."""
    try:
        args = json.loads(arguments_json)
    except (json.JSONDecodeError, TypeError):
        return ParsedSubmit(is_parseable=False)
    if expected_kind == "single_slice":
        pos = args.get("position_mm")
        reasoning = args.get("reasoning")
        if not isinstance(pos, (int, float)) or isinstance(pos, bool):
            return ParsedSubmit(is_parseable=False)
        if not isinstance(reasoning, str) or not reasoning.strip():
            return ParsedSubmit(is_parseable=False)
        return ParsedSubmit(is_parseable=True, position_mm=float(pos))
    raise ValueError(f"unknown expected_kind: {expected_kind!r}")


def compute_position_mae_mm(predicted: list[float], truth: list[float]) -> float:
    """Mean absolute error in mm. predicted and truth must be the same length."""
    if len(predicted) != len(truth):
        raise ValueError(f"length mismatch: {len(predicted)} vs {len(truth)}")
    if not predicted:
        raise ValueError("empty prediction list")
    errors = [abs(p - t) for p, t in zip(predicted, truth)]
    return sum(errors) / len(errors)


@dataclass
class EvalRun:
    subject_id: str
    predicted_mm: list[float] | None  # None if not parseable / no submit
    truth_mm: list[float]
    parseable: bool
    n_turns: int


def summarize_eval_runs(runs: list[EvalRun]) -> dict[str, float]:
    """Aggregate metrics over a held-out eval run.

    Returns a dict with four keys:
    - position_mae_mm: mean absolute error across parseable runs (mm).
      ``float('nan')`` when no run produced a parseable submit.
      Callers logging this to trackio/W&B/TensorBoard should guard with
      ``math.isnan`` before aggregating, or skip logging on NaN.
    - tool_call_parseability_rate: fraction of runs with a parseable submit.
    - no_submit_rate: fraction of runs that emitted no submit at all.
    - mean_trace_length: mean number of turns per run.
    """
    if not runs:
        raise ValueError("no eval runs to summarize")
    # Collect (predicted, truth) pairs from runs that both parsed and emitted a submit.
    # This filtering also narrows predicted_mm from list[float] | None to list[float]
    # without needing a type cast or ignore.
    parseable_pairs: list[tuple[list[float], list[float]]] = [
        (r.predicted_mm, r.truth_mm)
        for r in runs
        if r.parseable and r.predicted_mm is not None
    ]
    n_submits = sum(1 for r in runs if r.predicted_mm is not None)
    if parseable_pairs:
        per_run_maes = [compute_position_mae_mm(pred, truth) for pred, truth in parseable_pairs]
        position_mae = sum(per_run_maes) / len(per_run_maes)
    else:
        position_mae = float("nan")
    return {
        "position_mae_mm": position_mae,
        "tool_call_parseability_rate": len(parseable_pairs) / len(runs),
        "no_submit_rate": 1.0 - (n_submits / len(runs)),
        "mean_trace_length": sum(r.n_turns for r in runs) / len(runs),
    }


# --- Agent-loop eval (baseline + periodic) ----------------------------------


def _load_test_images_with_truth(test_images_root: Path) -> list[dict[str, Any]]:
    """Read references/TestImages/M0[1-9]/ground_truth.json into a list of eval rows.

    Each ground_truth.json maps image_filename -> {ap_mm, atlas, slice_index, ...}.
    Returns one row per (subject, image_filename) pair. Plane is hardcoded to
    "coronal" (the only test images in this corpus are coronal).
    """
    rows: list[dict[str, Any]] = []
    for sub in sorted(test_images_root.glob("M0[1-9]")):
        gt_path = sub / "ground_truth.json"
        if not gt_path.is_file():
            continue
        gt = json.loads(gt_path.read_text(encoding="utf-8"))
        for image_filename, meta in gt.items():
            image_path = sub / image_filename
            if not image_path.is_file():
                continue  # skip missing files
            rows.append({
                "subject_id": sub.name,
                "image_path": image_path,
                "atlas_name": meta["atlas"],
                "plane": "coronal",
                "ground_truth_position_mm": float(meta["ap_mm"]),
            })
    return rows


def _extract_tool_call_from_decoded(text: str) -> dict[str, Any] | None:
    """Pull the first tool_call out of Gemma 4's decoded assistant turn.

    TODO: This pattern is a PLACEHOLDER. Gemma 4's actual tool-call output format
    must be verified against a real generation in the Task 14 smoke run; expected
    format per Task 1 verification notes uses ``<|tool_call|>`` markers, but the
    exact wrapping (JSON-in-markers vs. tool_calls-array, separator chars) needs
    empirical confirmation before paid training.

    Returns a dict shaped like HF tool_calls items, or None if no parseable call.
    """
    # Placeholder pattern — replace after Task 14 smoke generation reveals format.
    m = re.search(r"<tool_call>(.*?)</tool_call>", text, flags=re.DOTALL)
    if not m:
        return None
    try:
        payload = json.loads(m.group(1).strip())
    except json.JSONDecodeError:
        return None
    return {
        "id": f"call_{abs(hash(text)) % 100000}",
        "type": "function",
        "function": {
            "name": payload.get("name", ""),
            "arguments": json.dumps(payload.get("arguments", {})),
        },
    }


def _run_agent_loop_for_one(
    *,
    model: Any,
    processor: Any,
    eval_row: dict[str, Any],
    atlas_grid: Any,
    max_turns: int = 16,
) -> EvalRun:
    """Run the SFT model through the agent loop on one eval row, return EvalRun.

    Reuses LangSliceEstimateEnv (RLVR scaffolding) for tool execution.
    """
    from PIL import Image
    from rlvr.env import LangSliceEstimateEnv

    env = LangSliceEstimateEnv(atlas_grid=atlas_grid)
    query_image = Image.open(eval_row["image_path"]).convert("RGB")
    system_prompt_kind = "single_slice"
    truth = [eval_row["ground_truth_position_mm"]]
    cache = AtlasMetaCache()
    system_prompt = build_system_prompt(
        kind=system_prompt_kind,
        atlas_name=eval_row["atlas_name"],
        plane=eval_row["plane"],
        atlas_meta_cache=cache,
    )
    tools = build_tools_schema(system_prompt_kind)
    meta = cache.get(eval_row["atlas_name"], eval_row["plane"])
    messages: list[dict[str, Any]] = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": [
            {"type": "image", "image": query_image},
            {"type": "text", "text": "Estimate the position of this slice."},
        ]},
    ]
    env.reset(
        atlas_name=eval_row["atlas_name"],
        plane=eval_row["plane"],
        valid_range_mm=(meta.pos_lo, meta.pos_hi),
        ground_truth_positions_mm=truth,
        kind="single",
        prompt=messages,
        image=query_image,
        subject_id=eval_row["subject_id"],
    )

    n_turns = 0
    parseable = False
    predicted: list[float] | None = None
    for _ in range(max_turns):
        n_turns += 1
        gen_out = model.generate(
            **processor.apply_chat_template(
                messages,
                tools=tools,
                chat_template_kwargs={"enable_thinking": False},
                add_generation_prompt=True,
                tokenize=True,
                return_dict=True,
                return_tensors="pt",
            ).to(model.device),
            max_new_tokens=512,
            do_sample=False,
        )
        text = processor.decode(gen_out[0], skip_special_tokens=False)
        tool_call = _extract_tool_call_from_decoded(text)
        messages.append({"role": "assistant", "tool_calls": [tool_call] if tool_call else []})
        if tool_call is None:
            break
        if tool_call["function"]["name"].startswith("submit"):
            parsed = parse_submit_call(
                tool_call["function"]["arguments"], expected_kind=system_prompt_kind
            )
            parseable = parsed.is_parseable
            if parsed.is_parseable and parsed.position_mm is not None:
                predicted = [parsed.position_mm]
            break
        result = env.fetch_atlas(**json.loads(tool_call["function"]["arguments"]))
        tool_msg = {
            "role": "tool",
            "tool_call_id": tool_call["id"],
            "content": result.get("content", [{"type": "text", "text": str(result)}]),
        }
        messages.append(tool_msg)

    return EvalRun(
        subject_id=eval_row["subject_id"],
        predicted_mm=predicted,
        truth_mm=truth,
        parseable=parseable,
        n_turns=n_turns,
    )


class _AgentLoopEvalBase(TrainerCallback):
    """Shared logic for baseline + periodic eval callbacks."""

    def __init__(
        self,
        *,
        processor: Any,
        atlas_grid: Any,
        test_images_root: Path,
        log_prefix: str,
    ) -> None:
        self.processor = processor
        self.atlas_grid = atlas_grid
        self.test_rows = _load_test_images_with_truth(test_images_root)
        self.log_prefix = log_prefix

    def _run(self, model: Any, step: int) -> dict[str, float]:
        runs: list[EvalRun] = []
        for row in self.test_rows:
            try:
                runs.append(
                    _run_agent_loop_for_one(
                        model=model,
                        processor=self.processor,
                        eval_row=row,
                        atlas_grid=self.atlas_grid,
                    )
                )
            except Exception:
                logger.exception("agent-loop eval failed for %s", row["subject_id"])
                runs.append(
                    EvalRun(
                        subject_id=row["subject_id"],
                        predicted_mm=None,
                        truth_mm=[row["ground_truth_position_mm"]],
                        parseable=False,
                        n_turns=0,
                    )
                )
        summary = summarize_eval_runs(runs)
        prefixed = {f"{self.log_prefix}/{k}": v for k, v in summary.items()}
        logger.info("eval@step=%d: %s", step, prefixed)
        return prefixed


class BaselineEvalCallback(_AgentLoopEvalBase):
    """Run the agent-loop eval once before training starts."""

    def __init__(self, **kwargs: Any) -> None:
        super().__init__(log_prefix="baseline", **kwargs)
        self._has_run = False

    def on_train_begin(self, args, state, control, model=None, **kwargs):  # noqa: ANN001, ANN003
        if self._has_run or model is None:
            return
        self._has_run = True
        metrics = self._run(model, step=0)
        if hasattr(state, "log_history"):
            state.log_history.append({"step": 0, **metrics})


class AgentLoopEvalCallback(_AgentLoopEvalBase):
    """Run the agent-loop eval every ``agent_eval_steps`` optimizer steps."""

    def __init__(self, *, agent_eval_steps: int, **kwargs: Any) -> None:
        super().__init__(log_prefix="eval", **kwargs)
        self.agent_eval_steps = agent_eval_steps

    def on_step_end(self, args, state, control, model=None, **kwargs):  # noqa: ANN001, ANN003
        if model is None or state.global_step == 0:
            return
        if state.global_step % self.agent_eval_steps != 0:
            return
        # Switch to inference mode for generation, then back to training. Both
        # toggles are best-effort — Unsloth may be absent at test time, and the
        # for_inference call may no-op for non-PEFT models.
        try:
            from unsloth import FastVisionModel
            FastVisionModel.for_inference(model)
        except Exception:
            pass
        try:
            metrics = self._run(model, step=state.global_step)
            state.log_history.append({"step": state.global_step, **metrics})
        finally:
            try:
                from unsloth import FastVisionModel
                FastVisionModel.for_training(model)
            except Exception:
                pass
