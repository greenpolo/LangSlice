# Repository Map

This map is intentionally short and stable so humans and agents can find the active code quickly.

## Active Code Paths

- `langslice/cli.py` -- CLI entry point for `langslice version`, `langslice register`, `langslice estimate`, `langslice estimate-group`, and `langslice estimate-brain`
- `langslice/vlm_config.py` -- Gemini client configuration, backend selection, runtime settings (shared by estimation and registration)
- `langslice/atlas/` -- BrainGlobe atlas loading, AP/index conversion, coronal slice extraction, colored region and smoothed boundary helpers
- `langslice/estimation/` -- Single-slice AP estimation
  - `google/ap_tool_use.py` -- Gemini multi-turn tool-use AP estimator
  - `google/ap_image_gen.py` -- Gemini image-gen nano-banana multi-pass zoom AP estimator
  - `google/tool_definitions.py` -- Gemini tool definitions and tool-response construction helpers
  - `google/batch_eval.py` -- Gemini offline Batch API helpers
  - `openai/` -- OpenAI stubs (imports only, not yet implemented)
  - `debug.py` -- shared debug-artifact writing helpers
- `langslice/registration/` -- registration workflows, runtime wrapper, affine/TPS solving, result types
  - `common.py` -- shared utilities (retry, JSON extraction, coordinate conversion) and workflow router
  - `google/warping_image_gen.py` -- Gemini warping workflow via colored segmentation (default): Elastix B-spline deformation
  - `google/landmarks_image_gen.py` -- Gemini legacy two-shot landmark workflow (superseded by warping)
  - `google/landmarks_tool_use.py` -- Gemini iterative landmark tool-loop (experimental, on hold)
  - `openai/` -- OpenAI stubs (imports only, not yet implemented)
- `langslice/ml/` -- non-LLM machine learning tools (GPU-accelerated target selection, etc.)
- `langslice/whole_brain/` -- whole-brain multi-slice AP estimation: anchor selection (coarse tool-use + nano-banana fine), interval interpolation, parallel 2-pass nano-banana for non-anchors, Huber-loss constrained monotonic fitting, checkpoint I/O, and async pipeline orchestration
- `langslice/image_prep.py` -- image normalization, metadata-driven pixel size detection, VLM downsampling
- `langslice/agent_trace.py` -- structured trace-event helpers shared by AP and registration flows
- `langslice/retry.py` -- shared retry with backoff and progress heartbeat infrastructure
- `langslice/export.py` -- coronal anchoring math, VisuAlign markers, and QUINT/ABBA-compatible JSON export
- `tauri-gui/` -- Tauri desktop app (Rust backend, React + Three.js frontend)

## Tests

- `tests/smoke_test.py` -- package import and export smoke coverage
- `tests/test_atlas_features.py` and `tests/test_atlas_space.py` -- atlas helpers and orientation guardrails
- `tests/test_image_prep.py` -- image ingest, metadata detection, VLM resize behavior
- `tests/test_quicknii_math.py` -- anchoring and coronal-frame export math
- `tests/test_registration_*.py` -- registration runtime, solver, agent prompt behavior, and backends
- `tests/test_brain_*.py` -- whole-brain module: types, discovery, anchor selection, interpolation, window, checkpoint, agents, pipeline

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
- `langslice estimate-brain <image_folder> [--atlas ...] [--anchors ...]`
- `cd tauri-gui && pnpm tauri dev`
