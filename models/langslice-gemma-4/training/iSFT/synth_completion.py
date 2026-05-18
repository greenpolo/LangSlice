"""Source C: synth-prefix + Gemma-completes-submit + filter against GT.

Sits between Source A (full multi-turn rollouts; ~22s/section; rich Gemma
reasoning) and Source B (pure-CPU hand-injection; ~0.5ms/section; canned text):

  * Build a procedural tool-call prefix via ``langslice_traces.generate_trace``
    — coarse_sweep + fine_sweep positions chosen from empirical distributions
    (no model call yet).
  * Render the prefix as an OpenAI chat conversation that ends right after the
    last tool_result.
  * POST to vLLM and ask Gemma to complete with one ``submit_estimate`` call.
    That is the *only* model call this path makes per row — coarse/fine results
    are pre-baked, so we skip the 2 expensive Gemma generations Source A pays for.
  * Filter on ``|pred_mm - GT| <= accept_threshold_mm``.  Accepted rows carry
    Gemma's own reasoning text plus the real position_mm we trust (the GT).

Wall-time per row: ~5–7s vs Source A's ~22s — about 3–5x faster — with
real-Gemma reasoning text as a bonus.  Not currently wired into the iSFT
driver; ``generate_completion_rows`` is the entry point for future wiring.
"""

from __future__ import annotations

import base64
import json
import logging
import random
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any

import requests

from iSFT.synth_corpus import SectionSpec, _build_native_row, _fallback_grid

_LOGGER = logging.getLogger(__name__)


def _img_to_data_uri(path: Path) -> str:
    ext = path.suffix.lower().lstrip(".")
    mime = "image/png" if ext == "png" else "image/jpeg"
    return f"data:{mime};base64,{base64.b64encode(path.read_bytes()).decode('ascii')}"


def _build_completion_messages(
    canonical: Any, repo_root: Path,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """Render a CanonicalTrace prefix as OpenAI chat messages + tools schema.

    Ends after the last tool_result; vLLM's next assistant turn is what Gemma
    is being asked to generate (we expect a submit_estimate tool_call).
    """
    from langslice_traces.renderers._common import (  # noqa: PLC0415
        AtlasMetaCache, build_system_prompt, build_tools_schema,
    )

    plane = canonical.plane
    system = build_system_prompt(
        kind=canonical.system_prompt_kind, atlas_name=canonical.atlas_name,
        plane=plane, atlas_meta_cache=AtlasMetaCache(),
    )

    user_content: list[dict[str, Any]] = []
    for qp in canonical.query_image_paths:
        user_content.append({
            "type": "image_url",
            "image_url": {"url": _img_to_data_uri(repo_root / qp)},
        })
    user_content.append({"type": "text", "text": canonical.user_prompt_text})

    messages: list[dict[str, Any]] = [
        {"role": "system", "content": system},
        {"role": "user", "content": user_content},
    ]
    for i, step in enumerate(canonical.tool_steps):
        call_id = f"call_{i}"
        messages.append({
            "role": "assistant",
            "content": None,
            "tool_calls": [{
                "id": call_id, "type": "function",
                "function": {
                    "name": step.call_name,
                    "arguments": json.dumps(step.call_args),
                },
            }],
        })
        tool_content: list[dict[str, Any]] = [
            {"type": "image_url", "image_url": {"url": _img_to_data_uri(repo_root / p)}}
            for p in step.result_image_paths
        ]
        tool_content.append({"type": "text", "text": step.result_text})
        messages.append({
            "role": "tool", "tool_call_id": call_id, "content": tool_content,
        })

    return messages, build_tools_schema(canonical.system_prompt_kind)


def _do_one_completion(
    spec: SectionSpec, *, vllm_url: str, served_name: str,
    accept_threshold_mm: float, grid: list[float], repo_root: Path,
    rng_seed: int, temperature: float, timeout_s: float,
) -> dict[str, Any] | None:
    """One synth-prefix → Gemma-submit → filter cycle. None on reject/error."""
    from langslice_traces.generator import generate_trace  # noqa: PLC0415

    canonical = generate_trace(
        image_path=spec.image_path, ground_truth_mm=spec.ground_truth_mm,
        plane=spec.plane, atlas_name=spec.atlas_name,
        atlas_version=spec.atlas_version, subject_id=spec.subject_id,
        grid=grid, strategy="lane_a_prefix", rng=random.Random(rng_seed),
    )
    messages, tools = _build_completion_messages(canonical, repo_root)

    try:
        response = requests.post(
            f"{vllm_url.rstrip('/')}/chat/completions",
            json={
                "model": served_name, "messages": messages, "tools": tools,
                "tool_choice": "auto", "temperature": temperature,
                "max_tokens": 2048,
            },
            timeout=timeout_s,
        )
        response.raise_for_status()
    except Exception as exc:  # noqa: BLE001
        _LOGGER.debug("vllm completion failed for %s: %s", spec.section_id, exc)
        return None

    msg = response.json()["choices"][0]["message"]
    submit_call = next(
        (tc for tc in (msg.get("tool_calls") or [])
         if tc.get("function", {}).get("name") == "submit_estimate"),
        None,
    )
    if submit_call is None:
        return None
    try:
        args = json.loads(submit_call["function"]["arguments"])
        pred_mm = float(args["position_mm"])
        reasoning = str(args.get("reasoning", "")).strip()
    except (KeyError, ValueError, TypeError, json.JSONDecodeError):
        return None
    if not reasoning or abs(pred_mm - spec.ground_truth_mm) > accept_threshold_mm:
        return None

    return _build_native_row(
        spec, tool_steps_included=True,
        ground_truth_mm=spec.ground_truth_mm,
        tool_step_positions=[
            list(s.call_args["positions_mm"]) for s in canonical.tool_steps
        ],
        atlas_name=spec.atlas_name, plane=spec.plane,
        atlas_version=spec.atlas_version, reasoning=reasoning,
    )


def generate_completion_rows(
    specs: list[SectionSpec],
    *,
    vllm_url: str = "http://127.0.0.1:8000/v1",
    served_name: str = "langslice-ft",
    accept_threshold_mm: float = 0.5,
    grid: list[float] | None = None,
    atlas_grid_cache_dir: Path | None = None,
    concurrency: int = 4,
    repo_root: Path | None = None,
    rng_seed: int = 42,
    temperature: float = 0.4,
    timeout_s: float = 300.0,
) -> list[dict[str, Any]]:
    """Source C entrypoint: returns accepted native JSONL rows (one per spec, at most).

    Rejected and errored rows are dropped silently (DEBUG-logged).  Per-row
    cost is one vLLM completion (~5–7s) plus filter.  Use a generous spec
    pool (say, 1.5–2× the number of rows you want) since rejection rate is
    bounded by the current model's tolerance-hit rate.
    """
    from langslice_traces.generator import load_atlas_grid  # noqa: PLC0415

    root = (repo_root or Path.cwd()).resolve()
    grid_cache: dict[tuple[str, str], list[float]] = {}

    def _grid_for(spec: SectionSpec) -> list[float]:
        if grid is not None:
            return grid
        key = (spec.atlas_name, spec.plane)
        if key not in grid_cache:
            try:
                grid_cache[key] = load_atlas_grid(
                    atlas_grid_cache_dir, spec.atlas_name, spec.plane,  # type: ignore[arg-type]
                ) if atlas_grid_cache_dir is not None else _fallback_grid(spec.ground_truth_mm)
            except Exception:  # noqa: BLE001
                grid_cache[key] = _fallback_grid(spec.ground_truth_mm)
        return grid_cache[key]

    rows: list[dict[str, Any]] = []
    with ThreadPoolExecutor(max_workers=concurrency) as pool:
        futs = [
            pool.submit(
                _do_one_completion, spec,
                vllm_url=vllm_url, served_name=served_name,
                accept_threshold_mm=accept_threshold_mm,
                grid=_grid_for(spec), repo_root=root,
                rng_seed=rng_seed + i, temperature=temperature,
                timeout_s=timeout_s,
            )
            for i, spec in enumerate(specs)
        ]
        for fut in as_completed(futs):
            row = fut.result()
            if row is not None:
                rows.append(row)
    _LOGGER.info(
        "synth_completion: %d/%d accepted (threshold=%.2fmm)",
        len(rows), len(specs), accept_threshold_mm,
    )
    return rows
