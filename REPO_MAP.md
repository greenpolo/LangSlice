# Repository Map

This map is intentionally short and stable so humans and agents can find the active code quickly.

## Active Code Paths

- `langslice/cli.py` -- CLI entry point for `langslice version`, `langslice register`, `langslice estimate`, and `langslice estimate-brain`
- `langslice/atlas/` -- BrainGlobe atlas loading, AP/index conversion, coronal slice extraction, colored region and smoothed boundary helpers
- `langslice/ai/` -- Gemini configuration, AP estimator (split across `estimator.py`, `estimator_tools.py`, `estimator_debug.py`), offline batch helpers
- `langslice/registration/` -- registration workflows, runtime wrapper, affine/TPS solving, result types
  - `agents.py` -- shared utilities (retry, JSON extraction, coordinate conversion) and workflow router
  - `agents_colored_segmentation.py` -- colored-segmentation workflow (default for image-gen models): model produces atlas-colored tissue segmentation, Elastix B-spline extracts deformation
  - `agents_image_gen.py` -- legacy two-shot landmark workflow for image-gen models (superseded by colored segmentation)
  - `agents_tool_loop.py` -- iterative tool-loop workflow for text-centric models (experimental, on hold)
- `langslice/brain/` -- whole-brain multi-slice AP estimation: anchor selection (coarse tool-use + nano-banana fine), interval interpolation, parallel 2-pass nano-banana for non-anchors, Huber-loss constrained monotonic fitting, checkpoint I/O, and async pipeline orchestration
- `langslice/image_prep.py` -- image normalization, metadata-driven pixel size detection, VLM downsampling
- `langslice/agent_trace.py` -- structured trace-event helpers shared by AP and registration flows
- `langslice/export.py` -- coronal anchoring math, VisuAlign markers, and QUINT/ABBA-compatible JSON export
- `tauri-gui/` -- Tauri desktop app (Rust backend, React + Three.js frontend)

## Tests

- `tests/smoke_test.py` -- package import and export smoke coverage
- `tests/test_atlas_features.py` and `tests/test_atlas_space.py` -- atlas helpers and orientation guardrails
- `tests/test_image_prep.py` -- image ingest, metadata detection, VLM resize behavior
- `tests/test_quicknii_math.py` -- anchoring and coronal-frame export math
- `tests/test_registration_*.py` -- registration runtime, solver, agent prompt behavior, and backends
- `tests/test_brain_*.py` -- brain module: types, discovery, anchor selection, interpolation, window, constraints, checkpoint, agents, pipeline

## Documentation

- `README.md` -- user-facing setup and current runtime behavior
- `docs/index.md` -- maintained documentation index
- `docs/architecture_overview.md` -- package boundaries and end-to-end runtime flow
- `docs/current_workflow.md` -- current CLI and Tauri GUI workflow
- `docs/abba_ap_coordinate_system.md` -- atlas-native AP coordinate rules used by code and export
- `docs/registration_plan.md` -- current registration runtime status and gaps, despite the legacy filename
- `AGENTS.md` and `langslice/**/AGENTS.md` -- agent-facing guidance kept aligned with code

## Non-Active Paths

- `archive/` -- preserved legacy prototypes and earlier package split attempts
- `references/` -- copied external reference material

## Common Commands

- `pip install -e .`
- `python -m pytest`
- `python -m ruff check .`
- `python -m basedpyright`
- `langslice version`
- `langslice register <image> --position <mm> [--workflow ...] [--model ...] [--out ...]`
- `langslice estimate <image> [--atlas ...] [--model ...] [--workflow ...]`
- `langslice estimate-brain <image_folder> [--atlas ...] [--anchors ...] [--ordering ...]`
- `cd tauri-gui && pnpm tauri dev`
