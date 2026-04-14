"""Multi-slice group AP estimation via OpenAI-compatible Chat Completions — not yet implemented.

Imports below mirror the Google implementation's dependencies to establish
the expected interface. Suppress F401 since these are intentional placeholders.
"""
# ruff: noqa: F401

from __future__ import annotations

import logging
from collections.abc import Callable
from typing import Any

from PIL import Image

from langslice.agent_trace import (
    image_part_from_pil,
    json_part,
    model_event,
    runtime_event,
)
from langslice.image_prep import normalize_image, prepare_image_for_vlm
from langslice.retry import retry_with_backoff, run_with_progress_heartbeat

logger = logging.getLogger(__name__)
