# Repository Map

Short navigation map for the active LangSlice repo.

## Active Code

- `src/langslice_harness/cli.py` -- CLI entry point for `langslice`.
- `src/langslice_harness/vlm_config.py` -- Gemini backend selection and runtime settings.
- `src/langslice_harness/atlas/` -- BrainGlobe atlas loading, AP/index conversion, and slice extraction.
- `src/langslice_harness/harness/estimation/` -- ADK single-slice and group AP estimation.
- `src/langslice_harness/harness/estimation/image_gen.py` -- image-gen AP estimation.
- `src/langslice_harness/registration/` -- registration public wrapper, runtime, solver, and result types.
- `src/langslice_harness/harness/registration/` -- image-gen registration candidate pipeline, provider adapters, and optional ADK review loop.
- `src/langslice_harness/whole_brain/` -- whole-brain multi-slice AP estimation.
- `src/langslice_harness/image_prep.py` -- image normalization, metadata detection, and VLM downsampling.
- `src/langslice_harness/export.py` -- QUINT/ABBA-compatible JSON export.
- `tauri-gui/` -- Tauri desktop app.
- `models/` -- fine-tuned model projects.

## Tests

- `tests/smoke_test.py` -- import and export smoke coverage.
- `tests/test_atlas_*.py` -- atlas helpers and orientation guardrails.
- `tests/test_harness_*.py` -- ADK estimation and registration harness coverage.
- `tests/test_registration_*.py` -- registration runtime, solver, and CLI orchestration.
- `tests/test_brain_*.py` -- whole-brain estimation.

## Docs

- `README.md` -- setup and current runtime summary.
- `docs/index.md` -- maintained docs index.
- `docs/architecture_overview.md` -- package boundaries and runtime flow.
- `docs/current_workflow.md` -- CLI and GUI workflow.
- `docs/abba_ap_coordinate_system.md` -- atlas-native AP coordinate rules.

## Local-Only

- `_local/` -- ignored scratch, archives, experiments, and development notes.
- `references/` -- ignored external reference repositories.
- `eval_outputs/` and `debug_runs/` -- ignored run artifacts.

## Common Commands

- `pip install -e .`
- `python -m pytest`
- `python -m ruff check .`
- `python -m basedpyright`
- `langslice version`
- `langslice register <image> --position <mm> [--registration-mode direct|agentic]`
- `langslice estimate <image> [--atlas ...] [--model ...] [--workflow ...]`
- `langslice estimate-brain <image_folder> [--atlas ...] [--anchors ...]`
- `cd tauri-gui && pnpm tauri dev`
