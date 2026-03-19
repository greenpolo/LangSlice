"""LangSlice VLM module - Gemini-based visual estimation."""

from langslice.vlm.config import (
    MODEL_NAME,
    TEMPERATURE,
    THINKING_LEVEL,
    create_batch_client,
    close_client,
    count_tokens_enabled,
    get_api_key,
    get_client,
    set_temperature,
)
from langslice.vlm.batch_eval import APBatchCase, build_ap_batch_requests, create_ap_batch_job
from langslice.vlm.estimator import (
    APResult,
    estimate_ap,
    estimate_position,
)

__all__ = [
    "get_api_key",
    "get_client",
    "create_batch_client",
    "close_client",
    "MODEL_NAME",
    "TEMPERATURE",
    "THINKING_LEVEL",
    "count_tokens_enabled",
    "set_temperature",
    "estimate_ap",
    "estimate_position",
    "APResult",
    "APBatchCase",
    "build_ap_batch_requests",
    "create_ap_batch_job",
]
