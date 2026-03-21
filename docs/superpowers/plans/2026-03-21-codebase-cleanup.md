# Codebase Cleanup Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Aggressive cleanup of the LangSlice codebase — remove dead code, delete the single_pass workflow, split oversized files, and leave the repo ready for the real priorities (AP agent quality, batch scaling, cost optimization).

**Architecture:** Module-by-module sweep in priority order: registration (remove single_pass + dead QC code), VLM (split 1966-line estimator.py), GUI (split 2087-line main_window.py, consolidate duplicates), then root cleanup (stray files, docs).

**Tech Stack:** Python 3.10+, PySide6, google-genai, brainglobe-atlasapi, pytest, ruff, basedpyright

**Constraints:**
- DO NOT modify prompts, LLM parameters, or conversation flow in `agents_tool_loop.py` or `agents_image_gen.py` — these are fickle and tuned.
- DO NOT touch `archive/` or `references/` unless the task explicitly targets them.
- Run `pytest`, `ruff check .`, and `basedpyright` after each task to verify nothing breaks.

---

### Task 1: Remove single_pass workflow and update default to multimodal_tool_loop

**Files:**
- Delete: `langslice/registration/agents_single_pass.py`
- Modify: `langslice/registration/agents.py` — remove late import and router branch for single_pass, remove token-counting preflight that calls `_build_single_pass_request`, update default `workflow` param in `estimate_registration_correspondences()` signature, fix fallback lambda at ~line 699
- Modify: `langslice/vlm/config.py` — remove `REGISTRATION_WORKFLOW_SINGLE_PASS` constant, remove single_pass entry from `REGISTRATION_WORKFLOW_LABELS` dict, update `default_registration_workflow()` return and its fallback path to return `"multimodal_tool_loop"`, remove single_pass from `get_registration_workflow_options()`
- Modify: `langslice/registration/core.py` — change default `workflow` param from `"single_pass"` to `"multimodal_tool_loop"`
- Modify: `langslice/registration/runtime.py` — change default `workflow` param from `"single_pass"` to `"multimodal_tool_loop"`
- Modify: `langslice/registration/types.py` — change default `workflow` in annotation session from `"single_pass"` to `"multimodal_tool_loop"`
- Modify: `langslice/cli.py` — remove `single_pass` from `--workflow` help text/choices
- Modify: `tests/` — update any tests that reference single_pass workflow (expect ~5 test functions in `test_registration_agents.py` to be deleted, plus string replacements in `test_registration_runtime.py`, `test_main_window_manual_registration.py`, `test_registration_backends.py`, `test_vlm_features.py`)

- [ ] **Step 1: Delete `agents_single_pass.py`**
- [ ] **Step 2: Remove single_pass imports and router branch from `agents.py`**
  - Remove the late import block for `_build_single_pass_request` and `_estimate_correspondences_single_pass`
  - Remove the elif/else branch that dispatches to single_pass
  - Remove token-counting preflight that calls `_build_single_pass_request`
  - Update default `workflow` param in `estimate_registration_correspondences()` signature (~line 661)
  - Fix fallback lambda at ~line 699: change `lambda _name: "single_pass"` → `lambda _name: "multimodal_tool_loop"` (or remove the lambda since `default_registration_workflow` definitely exists)
- [ ] **Step 3: Update `config.py` defaults**
  - Remove `REGISTRATION_WORKFLOW_SINGLE_PASS` constant
  - Remove single_pass entry from `REGISTRATION_WORKFLOW_LABELS` dict (~lines 57-61)
  - Change `default_registration_workflow()` return to `"multimodal_tool_loop"`, including the empty-options fallback path (~line 147)
  - Remove single_pass from workflow options list
- [ ] **Step 4: Update default `workflow` parameter in `core.py`, `runtime.py`, `types.py`**
  - Change `workflow: str = "single_pass"` → `workflow: str = "multimodal_tool_loop"` in all three
- [ ] **Step 5: Update `cli.py`**
  - Remove `single_pass` from `--workflow` choices/help text
- [ ] **Step 6: Update tests referencing single_pass**
  - Grep for `single_pass` across `tests/` and update or remove
  - Expect ~5 single_pass-specific test functions in `test_registration_agents.py` to be deleted
  - Update default workflow strings in `test_registration_runtime.py`, `test_main_window_manual_registration.py`, `test_registration_backends.py`, `test_vlm_features.py`
- [ ] **Step 7: Run verification**
  ```bash
  python -m pytest
  python -m ruff check .
  python -m basedpyright
  ```
- [ ] **Step 8: Commit**
  ```bash
  git add -A
  git commit -m "refactor: remove single_pass workflow, default to multimodal_tool_loop"
  ```

---

### Task 2: Remove dead QC code (vet_correspondences, rejected_correspondences, qc_state, pixel_size_um)

**Scope note:** `pixel_size_um` removal is limited to the **registration pipeline call chain** (core.py, runtime.py, workers, CLI). The image_prep.py metadata detection and GUI display/spin-box for pixel size remain — those are display/metadata features, not registration logic. A separate future task can remove them if desired.

**Files:**
- Modify: `langslice/registration/solver.py` — remove `vet_correspondences()` and `VettingResult` dataclass
- Modify: `langslice/registration/runtime.py` — remove `pixel_size_um` param, remove `rejected_correspondences` (always empty), remove hardcoded `qc_state`
- Modify: `langslice/registration/core.py` — remove `pixel_size_um` param, remove `qc_state` from log message (~line 67)
- Modify: `langslice/registration/types.py` — remove `rejected_correspondences` field from `RegistrationResult`, remove `qc_state` field
- Modify: `langslice/registration/__init__.py` — remove any exports related to removed items
- Modify: `langslice/gui/main_window.py` — remove `pixel_size_um` args passed to `AgentWorker.__init__()` and `ManualRegistrationWorker.__init__()`, and their `run()` methods
- Modify: `langslice/gui/overlay_viewer.py` — remove no-op `set_pixel_size()` method
- Modify: `langslice/cli.py` — remove `pixel_size_um` args passed to registration calls, remove `qc_state` from print output and JSON (~lines 111, 128)
- Modify: `tests/test_registration_solver.py` — remove vet_correspondences tests
- Modify: `tests/` — update any tests passing pixel_size_um or checking rejected_correspondences/qc_state

- [ ] **Step 1: Remove `vet_correspondences()` and `VettingResult` from `solver.py`**
- [ ] **Step 2: Remove `pixel_size_um` parameter from `core.py` and `runtime.py`**
  - Remove param from function signatures
  - Remove `_ = pixel_size_um` discard in runtime.py
- [ ] **Step 3: Remove `rejected_correspondences` and `qc_state` from `runtime.py`, `types.py`, `core.py`, and `cli.py`**
  - Remove fields from RegistrationResult dataclass
  - Remove population of these fields in runtime.py
  - Remove from debug artifact serialization in runtime.py
  - Remove `qc_state` from log message in core.py (~line 67)
  - Remove `qc_state` from print output and JSON in cli.py (~lines 111, 128)
- [ ] **Step 4: Remove `pixel_size_um` from worker constructors and run methods**
  - Remove from `AgentWorker.__init__()` and `AgentWorker.run()` in main_window.py (~line 180)
  - Remove from `ManualRegistrationWorker.__init__()` and `ManualRegistrationWorker.run()` (~line 263)
  - Remove from callers that construct these workers
- [ ] **Step 5: Remove `set_pixel_size()` from `overlay_viewer.py`**
- [ ] **Step 6: Update callers** — grep for `pixel_size_um`, `rejected_correspondences`, `qc_state` across the remaining codebase and remove all references in the registration call chain
- [ ] **Step 7: Update `__init__.py` exports** if any removed items were exported
- [ ] **Step 8: Update tests**
  - Remove `vet_correspondences` tests from `test_registration_solver.py`
  - Update any tests that pass `pixel_size_um` to registration calls or assert on `rejected_correspondences`/`qc_state`
- [ ] **Step 8: Run verification**
  ```bash
  python -m pytest
  python -m ruff check .
  python -m basedpyright
  ```
- [ ] **Step 9: Commit**
  ```bash
  git add -A
  git commit -m "refactor: remove dead QC code (vet_correspondences, pixel_size_um, qc_state)"
  ```

---

### Task 3: Rename `vlm/` module to `ai/`

The name "vlm" (vision-language model) is too generic and doesn't capture what the module does. Rename to `ai/` — short, clear, not vendor-specific, extensible for future non-Google models.

**Files:**
- Rename: `langslice/vlm/` → `langslice/ai/`
- Modify: Every file that imports from `langslice.vlm` — update to `langslice.ai`
- Modify: `langslice/ai/__init__.py` — update internal imports
- Modify: `tests/` — update imports
- Modify: `pyproject.toml` — if vlm is referenced in any config

- [ ] **Step 1: Rename the directory** `langslice/vlm/` → `langslice/ai/`
- [ ] **Step 2: Update all internal imports within `langslice/ai/`** — change `langslice.vlm` → `langslice.ai`
- [ ] **Step 3: Update all imports across the codebase** — grep for `langslice.vlm` and `from langslice.vlm` in all `.py` files and update
- [ ] **Step 4: Update tests** — grep for `vlm` in test imports and mock paths
- [ ] **Step 5: Run verification**
  ```bash
  python -m pytest
  python -m ruff check .
  python -m basedpyright
  ```
- [ ] **Step 6: Commit**
  ```bash
  git add -A
  git commit -m "refactor: rename vlm/ module to ai/"
  ```

---

### Task 4: Clean up ai/config.py

**Files:**
- Modify: `langslice/ai/config.py` — consolidate `_load_dotenv()` calls, remove unused predicates

- [ ] **Step 1: Move `_load_dotenv()` to module-level init** — call once at import time instead of in every getter function
- [ ] **Step 2: Remove unused predicate** — `supports_structured_image_output()` only (the `STRUCTURED_OUTPUT_IMAGE_MODELS` set is empty and no callers exist outside config.py). **DO NOT remove `supports_image_model_thinking()`** — it is actively called by `agents_image_gen.py` (~line 120-125) and tested in `test_registration_agents.py`
- [ ] **Step 3: Run verification**
  ```bash
  python -m pytest
  python -m ruff check .
  python -m basedpyright
  ```
- [ ] **Step 4: Commit**
  ```bash
  git add langslice/ai/config.py
  git commit -m "refactor: consolidate dotenv loading, remove unused config predicates"
  ```

---

### Task 5: Split estimator.py (~1966 lines → ~3 files)

This is the highest-risk refactor. The goal is to split without changing any behavior or prompt logic.

**Files:**
- Modify: `langslice/ai/estimator.py` — extract chunks into new files
- Create: `langslice/ai/estimator_tools.py` — tool handler logic (`_process_ap_function_calls` and its helpers)
- Create: `langslice/ai/estimator_debug.py` — debug artifact writing
- Modify: `langslice/ai/__init__.py` — ensure public API unchanged

**Split strategy:**
- `estimator_tools.py`: `_process_ap_function_calls()`, `_get_regions_at_position()`, `_is_broad_multi_sweep()`, `_is_narrow_multi_sweep()`, `_sorted_unique_positions()`, `_has_neighbor_bracket()`, `_build_nudge_text()`, `_extract_generate_function_calls()`, `_extract_interaction_function_calls()` (~600 lines)
- `estimator_debug.py`: Debug artifact writing block (lines ~1828-1937) only. **Note:** `_format_usage_metadata()`, `_format_count_tokens()`, and `_format_elapsed_seconds()` are used by BOTH the main loop (progress logging) and debug writing, so keep them in `estimator.py` and import them into `estimator_debug.py` as needed (~150 lines for debug-only code)
- `estimator.py` (remaining): `_APLoopState`, `APResult`, `_ImagePayload`, retry/heartbeat, `_run_interactions_ap_loop`, formatting helpers, `estimate_position`, `estimate_ap` (~1200 lines)

- [ ] **Step 1: Create `estimator_tools.py`**
  - Move `_process_ap_function_calls()` and all its helper functions
  - Add necessary imports
  - Import back into `estimator.py` so behavior is identical
- [ ] **Step 2: Create `estimator_debug.py`**
  - Move debug artifact writing logic and formatting helpers
  - Import back into `estimator.py`
- [ ] **Step 3: Remove `estimate_ap()` wrapper** — it's identical to `estimate_position()`. In `__init__.py`, add `estimate_ap = estimate_position` alias for backwards compatibility (it's in the `__all__` list)
- [ ] **Step 4: Verify public API is unchanged** — `__init__.py` still exports `estimate_ap`, `estimate_position`, `APResult`
- [ ] **Step 5: Run verification**
  ```bash
  python -m pytest
  python -m ruff check .
  python -m basedpyright
  ```
- [ ] **Step 6: Commit**
  ```bash
  git add langslice/ai/
  git commit -m "refactor: split estimator.py into tools, debug, and core modules"
  ```

---

### Task 6: GUI cleanup — consolidate duplicates, remove vestigial code

**Files:**
- Modify: `langslice/gui/main_window.py` — remove fallback AtlasViewer (lines 124-161), remove empty `resizeEvent()` override
- Modify: `langslice/gui/main_window_components.py` — make `pil_to_qpixmap()` the canonical implementation
- Modify: `langslice/gui/atlas_viewer.py` — import `pil_to_qpixmap` from main_window_components instead of redefining
- Modify: `langslice/gui/overlay_viewer.py` — import `pil_to_qpixmap` from main_window_components instead of redefining

- [ ] **Step 1: Remove fallback `AtlasViewer` class from `main_window.py`** (lines 124-161) — if the real import fails, the app should error, not silently degrade
- [ ] **Step 2: Remove empty `resizeEvent()` override** from main_window.py
- [ ] **Step 3: Consolidate `pil_to_qpixmap()`**
  - Keep the implementation in `main_window_components.py` as canonical
  - Ensure it handles RGB, RGBA, and L modes with fallback to `convert("RGBA")` (atlas_viewer falls back to RGB, overlay_viewer to RGBA — unify on RGBA since it's the safer superset)
  - Replace the duplicate in `atlas_viewer.py` with an import from `main_window_components`
  - Replace the duplicate in `overlay_viewer.py` with an import from `main_window_components`
- [ ] **Step 4: Remove unused `_ = affine_result` parameter** in `main_window_components.py:build_split_view_correspondence_points()`
- [ ] **Step 5: Run verification**
  ```bash
  python -m pytest
  python -m ruff check .
  python -m basedpyright
  ```
- [ ] **Step 6: Commit**
  ```bash
  git add langslice/gui/
  git commit -m "refactor: consolidate pil_to_qpixmap, remove vestigial GUI code"
  ```

---

### Task 7: Split main_window.py (~2087 lines → extract worker classes)

**Files:**
- Create: `langslice/gui/workers.py` — AgentWorker, ManualRegistrationWorker, _create_debug_run_dir
- Modify: `langslice/gui/main_window.py` — import workers from new module

This is the minimum useful split. The worker classes (lines ~163-340) are pure business logic with no UI dependencies beyond Qt signals, making them cleanly extractable.

- [ ] **Step 1: Create `langslice/gui/workers.py`**
  - Move `AgentWorker`, `ManualRegistrationWorker`, `_create_debug_run_dir()`
  - Add necessary imports
- [ ] **Step 2: Update `main_window.py`** — import workers from `workers.py`
- [ ] **Step 3: Run verification**
  ```bash
  python -m pytest
  python -m ruff check .
  python -m basedpyright
  ```
- [ ] **Step 4: Commit**
  ```bash
  git add langslice/gui/
  git commit -m "refactor: extract worker classes from main_window.py"
  ```

---

### Task 8: Root cleanup and docs update

**Files:**
- Delete: `nul` (empty Windows artifact)
- Modify: `REPO_MAP.md` — remove single_pass references, add new files
- Modify: `AGENTS.md` — update workflow references
- Modify: `docs/registration_plan.md` — remove stale gaps (vet_correspondences, pixel_size, qc_state)
- Modify: `docs/architecture_overview.md` — update module descriptions
- Modify: `CLAUDE.md` — update if needed

- [ ] **Step 1: Delete `nul` file**
- [ ] **Step 2: Update `REPO_MAP.md`**
  - Remove `agents_single_pass.py` reference
  - Update `vlm/` → `ai/` throughout
  - Add `estimator_tools.py`, `estimator_debug.py`, `workers.py`
  - Update descriptions as needed
- [ ] **Step 3: Update `AGENTS.md`** — remove single_pass workflow references (check root `AGENTS.md` and any module-level `AGENTS.md` files under `langslice/`)
- [ ] **Step 4: Update `docs/registration_plan.md`** — remove stale gaps that we just cleaned up
- [ ] **Step 5: Update `docs/architecture_overview.md`** — reflect new file structure
- [ ] **Step 6: Update `CLAUDE.md`** if any conventions changed
- [ ] **Step 7: Run verification**
  ```bash
  python -m pytest
  python -m ruff check .
  python -m basedpyright
  ```
- [ ] **Step 8: Commit**
  ```bash
  git add -A
  git commit -m "docs: update repo map, agents, and architecture docs for cleanup"
  ```
