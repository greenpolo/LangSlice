# Estimation Subsystem Guide

## Overview

`langslice/estimation/` is responsible for single-slice and group AP estimation.
Provider-specific implementations live in `google/` and `openai/` subdirectories.
Shared types (`_types.py`) and debug helpers (`debug.py`) live at the estimation root.
Shared Gemini helpers live in `google/common.py`.

## Files

- `_types.py` - provider-agnostic result types (`APResult`, `MultiSliceResult`)
- `google/ap_single_slice.py` - Gemini single-slice AP tool loop, retries, File API/cache support, trace emission
- `google/ap_image_gen.py` - Gemini image-gen nano-banana multi-pass zoom AP estimation
- `google/ap_multi_slice.py` - Gemini multi-slice group tool-use AP estimation (2-8 consecutive slices)
- `google/common.py` - shared helpers: `_APLoopState`, `_GroupLoopState`, image/trace utilities, shared `fetch_atlas` handler
- `google/tool_definitions.py` - Gemini tool definitions and tool-response construction helpers
- `google/batch_eval.py` - Gemini offline Batch API helpers for one-shot AP evaluation
- `google/ap_tool_use.py` - backward-compatibility shim (re-exports from `common.py`)
- `openai/ap_tool_use.py` - OpenAI tool-use AP estimation (stub, not yet implemented)
- `openai/ap_image_gen.py` - OpenAI image-gen AP estimation (stub, not yet implemented)
- `openai/tool_definitions.py` - OpenAI tool definitions (stub, not yet implemented)
- `debug.py` - shared debug-artifact writing helpers
- `__init__.py` - public exports including `estimate_position(...)`, `estimate_position_image_gen(...)`, `estimate_group(...)`, `estimate_ap(...)`, `MultiSliceResult`, and batch helpers

## Current Runtime Facts

- `estimate_position(...)` is the active Gemini tool-use single-slice AP estimator.
- `estimate_position_image_gen(...)` is the Gemini image-gen nano-banana estimator.
- `estimate_group(...)` is the Gemini tool-use multi-slice group estimator (2-8 consecutive slices).
- `estimate_ap(...)` is just a thin alias to `estimate_position(...)`.
- The tool-use estimators use manual function calling and inject atlas images into tool responses.
- The tool names are `fetch_atlas` and `submit_estimate`.
- The single-slice estimator can optionally use Gemini File API transport and cached content.
- All estimation uses `generate_content` (the Interactions API was removed from estimation).
- Debug traces are written only when a debug directory is available.
- OpenAI stubs contain imports only; implementations will follow.

## Config Facts

- VLM configuration lives in `langslice/vlm_config.py` (shared with registration).
- The active backend values are `ai_studio`, `vertex_api_key`, and `vertex_adc`.
- `AVAILABLE_MODELS` and `AVAILABLE_THINKING_LEVELS` are defined in `vlm_config.py`.
- Batch API support is currently guarded by `supports_batch_api()`, which only returns `True` for `vertex_adc`.

## Local Anti-Patterns

- Do not describe affine registration as living here; that logic lives in `langslice/registration/`.
- Do not inline secrets or bypass backend selection in `vlm_config.py`.
- Do not remove retry/backoff or the manual tool-loop structure without replacing their behavior explicitly.

## Verify After Edits

- `python -m pytest tests/smoke_test.py`
- `python -m pytest tests/test_registration_agents.py`
