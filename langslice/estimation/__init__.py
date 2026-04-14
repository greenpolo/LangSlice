"""LangSlice estimation module - single-slice AP estimation.

Provider-agnostic entry point. Consumers should import from here rather
than from ``estimation.google.*`` or ``estimation.openai.*`` directly, so
that switching providers only requires changing this file.
"""

from langslice.estimation._types import APResult, MultiSliceResult
from langslice.estimation.google.ap_image_gen import estimate_position_image_gen
from langslice.estimation.google.ap_multi_slice import estimate_group
from langslice.estimation.google.ap_single_slice import estimate_position
from langslice.estimation.google.batch_eval import (
    APBatchCase,
    build_ap_batch_requests,
    create_ap_batch_job,
)

__all__ = [
    "APResult",
    "MultiSliceResult",
    "estimate_group",
    "estimate_position",
    "estimate_position_image_gen",
    "APBatchCase",
    "build_ap_batch_requests",
    "create_ap_batch_job",
]
