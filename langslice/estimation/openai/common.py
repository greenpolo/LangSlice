"""Shared helpers for OpenAI-compatible estimation workflows — not yet implemented.

Imports below mirror the Google implementation's dependencies to establish
the expected interface. Suppress F401 since these are intentional placeholders.
"""
# ruff: noqa: F401

from __future__ import annotations

import logging
from typing import Any

from langslice.agent_trace import json_part, model_event, runtime_event
from langslice.estimation._types import APResult

logger = logging.getLogger(__name__)
