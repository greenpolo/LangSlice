"""LangSlice estimation module - single-slice AP estimation.

Provider-agnostic entry point. Consumers should import from here rather
than from ``estimation.google.*`` or ``estimation.openai.*`` directly, so
that switching providers only requires changing this file.
"""

from langslice.estimation.google.ap_image_gen import estimate_position_image_gen
from langslice.estimation.google.ap_tool_use import (
    APResult,
    estimate_position,
)
from langslice.estimation.google.batch_eval import (
    APBatchCase,
    build_ap_batch_requests,
    create_ap_batch_job,
)

# Backwards-compatible alias (estimate_ap was identical to estimate_position)
estimate_ap = estimate_position

__all__ = [
    "APResult",
    "estimate_ap",
    "estimate_position",
    "estimate_position_image_gen",
    "APBatchCase",
    "build_ap_batch_requests",
    "create_ap_batch_job",
]
