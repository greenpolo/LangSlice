# Repository Map

This map is intentionally short and stable so human and AI agents can navigate quickly.

## Active Project Paths

- `langslice/` - installable package source
- `langslice/atlas/` - BrainGlobe loading, AP indexing, atlas metadata + `brainglobe-space` adapter
- `langslice/vlm/` - Gemini client/config, AP agent loop, and Gemini affine fallback
- `langslice/registration/` - matrix-first affine and nonlinear registration runtime
- `langslice/gui/` - PySide6 application UI
- `langslice/export.py` - QUINT/ABBA-compatible export helpers
- `tests/` - verification scripts (`smoke_test.py`, `test_atlas_features.py`, `test_atlas_space.py`, `test_quicknii_math.py`, `test_registration_backends.py`)
- `pyproject.toml` - package metadata and dependencies
- `environment.yml` - conda environment definition
- `.env.example` - required environment variable template

## Context and Documentation

- `README.md` - setup, current workflow, and limitations
- `docs/index.md` - current documentation index
- `docs/architecture_overview.md` - package boundaries and runtime data flow
- `docs/current_workflow.md` - current GUI workflow, preview behavior, and limitations
- `docs/abba_ap_coordinate_system.md` - atlas-native AP coordinate notes
- `docs/legacy_ai_studio_readme.md` - preserved note from the old AI Studio prototype
- `AGENTS.md` - root agent operating guide

## Non-Active but Kept Intact

- `archive/old_ai_studio_prototype/` - original React/Vite prototype kept for provenance
- `archive/langsliceatlas/` - early empty package split attempt
- `archive/langslicegui/` - early empty package split attempt
- `archive/langslicevlm/` - early empty package split attempt
- `archive/nul_artifact.txt` - terminal artifact preserved for provenance
- `references/DeepSlice_upstream/` - copied upstream DeepSlice code for reference
- `archive/AGENTS.md` - archive edit boundary guidance
- `references/AGENTS.md` - references edit boundary guidance

## Common Commands

- `pip install -e .`
- `python tests/smoke_test.py`
- `python tests/test_atlas_features.py`
- `python tests/test_quicknii_math.py`
- `langslice gui`
- `langslice version`
