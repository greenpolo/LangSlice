"""OpenAI tool definitions for AP estimation — not yet implemented.

Imports below mirror the Google implementation's dependencies to establish
the expected interface. Suppress F401 since these are intentional placeholders.
"""
# ruff: noqa: F401

from __future__ import annotations

import json
from collections.abc import Callable
from typing import Any

import numpy as np
from PIL import Image

from langslice.agent_trace import (
    json_part,
    tool_call_event,
    tool_result_event,
)
from langslice.image_prep import normalize_image
