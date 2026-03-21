# Repository Map

This map is intentionally short and stable so humans and agents can find the active code quickly.

## Active Code Paths

- `langslice/cli.py` - CLI entry point for `langslice gui`, `langslice version`, and `langslice register`
- `langslice/atlas/` - BrainGlobe atlas loading, AP/index conversion, coronal slice extraction
- `langslice/vlm/` - Gemini configuration, AP estimator, offline batch helpers
- `langslice/registration/` - registration workflows, runtime wrapper, affine/TPS solving, result types
  - `agents.py` - shared utilities (retry, JSON extraction, coordinate conversion) and workflow router
  - `agents_image_gen.py` - two-shot workflow exclusively for image-gen models (gemini-3-pro-image-preview, gemini-3.1-flash-image-preview)
  - `agents_tool_loop.py` - iterative tool-loop workflow for text-centric models (default)
- `langslice/image_prep.py` - image normalization, metadata-driven pixel size detection, VLM downsampling
- `langslice/agent_trace.py` - structured trace-event helpers shared by AP and registration flows
- `langslice/export.py` - coronal anchoring math and QUINT/ABBA-compatible JSON export
- `langslice/gui/` - main window, viewers, settings dialog, trace inspector, run-metadata dialog

## Tests

- `tests/smoke_test.py` - package import and export smoke coverage
- `tests/test_atlas_features.py` and `tests/test_atlas_space.py` - atlas helpers and orientation guardrails
- `tests/test_image_prep.py` - image ingest, metadata detection, VLM resize behavior
- `tests/test_quicknii_math.py` - anchoring and coronal-frame export math
- `tests/test_registration_*.py` - registration runtime, solver, agent prompt behavior, and backends
- `tests/test_split_view_correspondences.py`, `tests/test_overlay_viewer_thread_cleanup.py`, `tests/test_main_window_manual_registration.py` - GUI behavior

## Documentation

- `README.md` - user-facing setup and current runtime behavior
- `docs/index.md` - maintained documentation index
- `docs/architecture_overview.md` - package boundaries and end-to-end runtime flow
- `docs/current_workflow.md` - current GUI behavior and operator-facing workflow
- `docs/abba_ap_coordinate_system.md` - atlas-native AP coordinate rules used by code and export
- `docs/registration_plan.md` - current registration runtime status and gaps, despite the legacy filename
- `AGENTS.md` and `langslice/**/AGENTS.md` - agent-facing guidance kept aligned with code

## Non-Active Paths

- `archive/` - preserved legacy prototypes and earlier package split attempts
- `references/` - copied external reference material

## Common Commands

- `pip install -e .`
- `python -m pytest`
- `python -m ruff check .`
- `python -m basedpyright`
- `langslice gui`
- `langslice version`
- `langslice register <image> --position <mm> [--workflow ...] [--model ...] [--out ...]`
