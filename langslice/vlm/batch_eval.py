"""Offline Batch API helpers for LangSlice evaluation workloads."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Sequence

from google.genai import types

import langslice.vlm.config as vlm_config


@dataclass(frozen=True)
class APBatchCase:
    """Single-turn AP evaluation request backed by a staged image URI."""

    key: str
    image_uri: str
    prompt: str
    mime_type: str = "image/jpeg"
    metadata: dict[str, str] = field(default_factory=dict)


def build_ap_batch_requests(
    cases: Sequence[APBatchCase],
    *,
    model: str,
    system_instruction: str | None = None,
) -> list[types.InlinedRequest]:
    """Build inlined Batch API requests for one-shot AP experiments.

    This intentionally targets single-turn evaluation, not the live multi-turn tool loop.
    """

    requests: list[types.InlinedRequest] = []
    for case in cases:
        parts = [
            types.Part(text=case.prompt),
            types.Part.from_uri(file_uri=case.image_uri, mime_type=case.mime_type),
        ]
        content = types.Content(role="user", parts=parts)
        request_config = (
            types.GenerateContentConfig(system_instruction=system_instruction)
            if system_instruction
            else None
        )
        requests.append(
            types.InlinedRequest(
                model=model,
                contents=[content],
                metadata={"key": case.key, **case.metadata},
                config=request_config,
            )
        )
    return requests


def create_ap_batch_job(
    cases: Sequence[APBatchCase],
    *,
    model: str,
    dest_gcs_uri: str,
    system_instruction: str | None = None,
    display_name: str | None = None,
) -> Any:
    """Create a Vertex Batch API job for offline AP prompt/model evaluation."""

    if not vlm_config.supports_batch_api():
        raise RuntimeError("Batch API requires a Vertex backend.")

    client = vlm_config.create_batch_client()
    try:
        requests = build_ap_batch_requests(
            cases,
            model=model,
            system_instruction=system_instruction,
        )
        batch_config = types.CreateBatchJobConfig(dest=dest_gcs_uri, display_name=display_name)
        return client.batches.create(model=model, src=requests, config=batch_config)
    finally:
        close = getattr(client, "close", None)
        if callable(close):
            close()
