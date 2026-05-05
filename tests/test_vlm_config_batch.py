"""Verify supports_batch_api() now allows AI Studio in addition to Vertex."""
from __future__ import annotations

import os
from unittest.mock import patch

from langslice_harness import vlm_config


def test_supports_batch_api_ai_studio():
    with patch.dict(os.environ, {"LANGSLICE_GENAI_BACKEND": "ai_studio"}, clear=False):
        assert vlm_config.supports_batch_api() is True


def test_supports_batch_api_vertex_adc():
    with patch.dict(os.environ, {"LANGSLICE_GENAI_BACKEND": "vertex_adc"}, clear=False):
        assert vlm_config.supports_batch_api() is True


def test_supports_batch_api_vertex_api_key_unsupported():
    # Vertex API-key mode does not currently route batch - keep false.
    with patch.dict(os.environ, {"LANGSLICE_GENAI_BACKEND": "vertex_api_key"}, clear=False):
        assert vlm_config.supports_batch_api() is False
