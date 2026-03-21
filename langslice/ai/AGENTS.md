# AI Subsystem Guide

## Overview

`langslice/ai/` is currently responsible for Gemini configuration and AP estimation.
It does not own the active registration runtime.

## Files

- `config.py` - backend selection, client creation, model list, thinking budgets, feature flags
- `estimator.py` - AP tool loop, retries, File API/cache/interactions support, trace emission
- `estimator_tools.py` - tool definitions and tool-response construction helpers
- `estimator_debug.py` - debug-artifact writing helpers
- `batch_eval.py` - offline Batch API helpers for one-shot AP evaluation
- `__init__.py` - public exports including `estimate_position(...)`, `estimate_ap(...)`, and batch helpers

## Current Runtime Facts

- `estimate_position(...)` is the active AP estimator.
- `estimate_ap(...)` is just a thin alias to `estimate_position(...)`.
- The estimator uses manual function calling and injects atlas images into tool responses.
- The tool names are `fetch_atlas_slice`, `fetch_multiple_atlas_slices`, `get_atlas_info`, `get_region_names`, and `submit_estimate`.
- The estimator can optionally use Gemini File API transport, cached content, and the Interactions API pilot path.
- Debug traces are written only when a debug directory is available.

## Config Facts

- The active backend values are `ai_studio`, `vertex_api_key`, and `vertex_adc`.
- `AVAILABLE_MODELS` and `AVAILABLE_THINKING_BUDGETS` are defined in `config.py`.
- Batch API support is currently guarded by `supports_batch_api()`, which only returns `True` for `vertex_adc`.

## Local Anti-Patterns

- Do not describe affine registration as living here; that logic now lives in `langslice/registration/`.
- Do not inline secrets or bypass backend selection in `config.py`.
- Do not remove retry/backoff or the manual tool-loop structure without replacing their behavior explicitly.

## Verify After Edits

- `python -m pytest tests/smoke_test.py`
- `python -m pytest tests/test_registration_agents.py`
