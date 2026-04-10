"""OpenAI landmark image-gen registration — not yet implemented.

Imports below mirror the Google implementation's dependencies to establish
the expected interface. Suppress F401 since these are intentional placeholders.
"""
# ruff: noqa: F401

from __future__ import annotations

import logging
from collections.abc import Callable
from typing import Any

from PIL import Image

from langslice.agent_trace import image_part_from_pil, runtime_event
from langslice.registration.types import RegistrationAnnotationSession

logger = logging.getLogger(__name__)
