# LangSlice Package Guide

## Overview

`langslice/` is the active Python package.
All runtime behavior described by the maintained docs lives here.

## Package Surface

- `cli.py` -- CLI parser for `version`, `register`, and `estimate`
- `atlas/` -- BrainGlobe atlas access, AP conversion, slice helpers, colored region and boundary helpers
- `ai/` -- Gemini config, AP estimator, offline batch helpers
- `registration/` -- registration workflows (colored segmentation, legacy image-gen, tool loop), runtime wrapper, affine/TPS solvers, result types
- `image_prep.py` -- normalization, metadata pixel-size detection, VLM image preparation
- `agent_trace.py` -- structured event builders for trace inspection
- `export.py` -- QUINT/ABBA-compatible JSON export helpers with VisuAlign marker support

## Local Conventions

- Keep imports under `langslice.*`.
- Keep atlas orientation assumptions centralized in `atlas/space.py`.
- Keep AP estimation logic in `ai/estimator.py`.
- Keep registration runtime behavior in `registration/`, not in `ai/`.
- Keep image ingest and VLM resize rules in `image_prep.py`.
- Keep export math in `export.py`.

## Local Anti-Patterns

- Do not move runtime logic into ad hoc root scripts.
- Do not describe aspirational behavior in docs when the code does something simpler.

## Verify After Edits

- `python -m pytest`
- `python -m langslice version`
