"""Async wrapper around ``run_single_slice_session`` for expert-iteration rollouts.

Each rollout produces one ``RolloutResult`` carrying:
    * ``submitted_position_mm``: the model's final answer
    * ``reasoning``: the submit step's reasoning text
    * ``events``: the raw ADK event log captured by ``AgentTraceRecorder``
    * ``run_dir``: where atlas tool images were persisted (parent of
      ``tool_artifact_dir``); ``trace_format._convert_trace`` resolves
      atlas image paths against this directory.
    * ``error``: ``None`` on success, exception string on failure (so the
      driver can drop bad rollouts without bringing down the whole batch)

We use ``ThreadPoolExecutor`` to fan out concurrent rollouts. Each rollout
runs ``asyncio.run(run_single_slice_session(...))`` in a worker thread —
ADK is async-only and we want CPU-parallel concurrency without nesting
event loops in the driver.
"""

from __future__ import annotations

import asyncio
import logging
import uuid
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from PIL import Image

from langslice_harness.harness.estimation.runner import run_single_slice_session
from langslice_harness.harness.estimation.trace_collection import AgentTraceRecorder

logger = logging.getLogger(__name__)


@dataclass
class RolloutSpec:
    """One (prompt, generation) pair to roll out."""

    section_id: str
    subject_id: str
    image_path: Path
    atlas_name: str
    plane: str
    truth_mm: float
    plane_extent_mm: float
    generation_idx: int


@dataclass
class RolloutResult:
    spec: RolloutSpec
    submitted_position_mm: float | None
    reasoning: str
    events: list[dict[str, Any]]
    run_dir: Path
    error: str | None = None
    extra: dict[str, Any] = field(default_factory=dict)


def _do_rollout_sync(
    spec: RolloutSpec,
    *,
    artifacts_root: Path,
    model: str,
    temperature: float,
    max_iterations: int,
    max_retries: int,
    apply_clahe: bool,
    media_resolution: str,
    multimodal_history: str,
) -> RolloutResult:
    """Run one rollout in a worker thread; returns a RolloutResult."""
    section_id_safe = spec.section_id.replace("/", "_").replace(":", "_")
    rollout_id = f"{section_id_safe}_g{spec.generation_idx}_{uuid.uuid4().hex[:8]}"
    run_dir = artifacts_root / rollout_id
    tool_artifact_dir = run_dir / "tool_artifacts"
    tool_artifact_dir.mkdir(parents=True, exist_ok=True)

    recorder = AgentTraceRecorder(tool_artifact_dir=tool_artifact_dir)
    try:
        with Image.open(spec.image_path) as raw:
            image = raw.copy()
        result = asyncio.run(
            run_single_slice_session(
                image=image,
                atlas_name=spec.atlas_name,
                plane=spec.plane,  # type: ignore[arg-type]
                model=model,
                max_iterations=max_iterations,
                max_retries=max_retries,
                temperature=temperature,
                apply_clahe=apply_clahe,
                media_resolution=media_resolution,
                multimodal_history=multimodal_history,
                trace_recorder=recorder,
            )
        )
    except Exception as exc:  # noqa: BLE001 — capture for caller filtering
        logger.exception(
            "Rollout failed for %s gen=%d: %s",
            spec.section_id,
            spec.generation_idx,
            exc,
        )
        return RolloutResult(
            spec=spec,
            submitted_position_mm=None,
            reasoning="",
            events=list(recorder.raw_events),
            run_dir=run_dir,
            error=f"{type(exc).__name__}: {exc}",
        )

    return RolloutResult(
        spec=spec,
        submitted_position_mm=float(result.position_mm),
        reasoning=str(result.reasoning),
        events=list(recorder.raw_events),
        run_dir=run_dir,
    )


def run_rollouts(
    specs: list[RolloutSpec],
    *,
    artifacts_root: Path,
    model: str = "litellm-proxy:langslice-ft",
    temperature: float = 0.9,
    max_iterations: int = 20,
    max_retries: int = 2,
    apply_clahe: bool = True,
    media_resolution: str = "medium",
    multimodal_history: str = "persistent",
    concurrency: int = 4,
) -> list[RolloutResult]:
    """Fan out rollouts via ``ThreadPoolExecutor`` and collect results.

    Order of returned results is **not** guaranteed; sort by ``(section_id,
    generation_idx)`` if downstream code needs determinism.
    """
    artifacts_root.mkdir(parents=True, exist_ok=True)
    results: list[RolloutResult] = []
    with ThreadPoolExecutor(max_workers=concurrency) as pool:
        futures = [
            pool.submit(
                _do_rollout_sync,
                spec,
                artifacts_root=artifacts_root,
                model=model,
                temperature=temperature,
                max_iterations=max_iterations,
                max_retries=max_retries,
                apply_clahe=apply_clahe,
                media_resolution=media_resolution,
                multimodal_history=multimodal_history,
            )
            for spec in specs
        ]
        for fut in as_completed(futures):
            results.append(fut.result())
    return results
