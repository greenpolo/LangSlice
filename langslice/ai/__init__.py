"""LangSlice AI module - Gemini-based visual estimation."""

from langslice.ai.batch_eval import APBatchCase, build_ap_batch_requests, create_ap_batch_job
from langslice.ai.config import (
    AVAILABLE_THINKING_LEVELS,
    CODE_EXECUTION_ENABLED,
    MODEL_NAME,
    TEMPERATURE,
    THINKING_LEVEL,
    close_client,
    count_tokens_enabled,
    create_batch_client,
    get_api_key,
    get_client,
    set_code_execution_enabled,
    set_temperature,
    set_thinking_level,
    supports_code_execution,
)
from langslice.ai.estimator import (
    APResult,
    estimate_position,
)
from langslice.ai.estimator_image_gen import estimate_position_image_gen

# Backwards-compatible alias (estimate_ap was identical to estimate_position)
estimate_ap = estimate_position

__all__ = [
    "get_api_key",
    "get_client",
    "create_batch_client",
    "close_client",
    "AVAILABLE_THINKING_LEVELS",
    "CODE_EXECUTION_ENABLED",
    "MODEL_NAME",
    "TEMPERATURE",
    "THINKING_LEVEL",
    "count_tokens_enabled",
    "set_code_execution_enabled",
    "set_thinking_level",
    "set_temperature",
    "supports_code_execution",
    "estimate_ap",
    "estimate_position",
    "estimate_position_image_gen",
    "APResult",
    "APBatchCase",
    "build_ap_batch_requests",
    "create_ap_batch_job",
]
