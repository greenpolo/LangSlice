"""LangSlice VLM module - Gemini-based visual estimation."""

from langslice.vlm.config import (
    MODEL_NAME,
    THINKING_LEVEL,
    close_client,
    get_api_key,
    get_client,
)
from langslice.vlm.estimator import (
    APResult,
    AffineResult,
    estimate_affine,
    estimate_ap,
    estimate_position,
)

__all__ = [
    "get_api_key",
    "get_client",
    "close_client",
    "MODEL_NAME",
    "THINKING_LEVEL",
    "estimate_ap",
    "estimate_position",
    "estimate_affine",
    "APResult",
    "AffineResult",
]
