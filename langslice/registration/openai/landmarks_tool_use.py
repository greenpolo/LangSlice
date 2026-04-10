"""OpenAI landmark tool-use registration — not yet implemented.

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
    tool_call_event,
    tool_result_event,
)
from langslice.registration.types import (
    LandmarkAnnotation,
    RegistrationAnnotationSession,
    render_landmark_annotations,
)

logger = logging.getLogger(__name__)
