# LangSlice Package Guide

## Overview

`langslice/` is the active Python package.
All runtime behavior described by the maintained docs lives here.

## Package Surface

- `cli.py` -- CLI parser for `version`, `register`, `estimate`, `estimate-group`, and `estimate-brain`
- `vlm_config.py` -- Gemini client configuration, backend selection, runtime settings
- `atlas/` -- BrainGlobe atlas access, AP conversion, slice helpers, colored region and boundary helpers
- `estimation/` -- Single-slice and group AP estimation: `_types.py` (result types), `google/` (Gemini), `openai/` (stubs), `debug.py` (shared)
- `whole_brain/` -- Multi-slice whole-brain AP estimation pipeline
- `registration/` -- registration workflows: `google/` (Gemini), `openai/` (stubs), shared router/solver/types
- `ml/` -- Non-LLM machine learning tools (GPU-accelerated target selection, etc.)
- `image_prep.py` -- normalization, metadata pixel-size detection, VLM image preparation
- `agent_trace.py` -- structured event builders for trace inspection
- `retry.py` -- shared retry with backoff and progress heartbeat infrastructure
- `export.py` -- QUINT/ABBA-compatible JSON export helpers with VisuAlign marker support

## Local Conventions

- Keep imports under `langslice.*`.
- Keep atlas orientation assumptions centralized in `atlas/space.py`.
- Keep AP estimation logic in `estimation/`.
- Keep registration runtime behavior in `registration/`, not in `estimation/`.
- Keep image ingest and VLM resize rules in `image_prep.py`.
- Keep export math in `export.py`.

## Local Anti-Patterns

- Do not move runtime logic into ad hoc root scripts.
- Do not describe aspirational behavior in docs when the code does something simpler.

## Verify After Edits

- `python -m pytest`
- `python -m langslice version`
