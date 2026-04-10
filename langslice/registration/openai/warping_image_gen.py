"""OpenAI warping (colored segmentation) registration — not yet implemented.

Imports below mirror the Google implementation's dependencies to establish
the expected interface. Suppress F401 since these are intentional placeholders.
"""
# ruff: noqa: F401

from __future__ import annotations

import logging
from collections.abc import Callable
from typing import Any

import cv2
import numpy as np
from PIL import Image

from langslice.agent_trace import image_part_from_pil, json_part, runtime_event
from langslice.atlas import (
    get_reference_slice,
    load_atlas,
    position_mm_to_index,
)
from langslice.registration.types import RegistrationAnnotationSession

logger = logging.getLogger(__name__)
