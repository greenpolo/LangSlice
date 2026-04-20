# Position-estimation ADK harness — implementation plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Refactor single-slice + multi-slice-group position-estimation harnesses onto Google ADK (`google-adk`), with plane-generalized atlas helpers, orientation-agnostic symbols (AP → AP/ML/DV per plane), and a four-tool registry (`fetch_atlas`, `zoom`, `side_by_side`, `submit_*`) that will drive downstream Gemma 4 training-data generation.

**Architecture:** Two `LlmAgent`s (single-slice and group), each with the four tools. Validators gate submission via `before_tool_callback`. Tools return `types.Part` images directly (save_artifact is back-channel only). A small `runner.py` wraps `InMemoryRunner` with nudge-on-no-tool-call, retry-with-fresh-session, and max-iteration cap logic.

**Tech Stack:** Python 3.12, `google-adk`, `google-genai`, `brainglobe-atlasapi` / `brainglobe-space`, PIL, pytest, ruff, basedpyright.

**Spec:** `docs/superpowers/specs/2026-04-19-position-estimation-adk-harness-design.md`

**Branch / worktree:** `feat/harness-adk` in `.worktrees/harness-adk/`.

---

## File structure (post-refactor)

**New files:**
- `langslice/harness/__init__.py`
- `langslice/harness/estimation/__init__.py` — public API
- `langslice/harness/estimation/_types.py` — `PositionResult`, `MultiSliceResult`, `APResult` alias
- `langslice/harness/estimation/session.py` — session state dataclass and artifact-key conventions
- `langslice/harness/estimation/prompts.py` — plane-aware prompt builders
- `langslice/harness/estimation/tools.py` — the four tools
- `langslice/harness/estimation/validators.py` — before_tool_callback
- `langslice/harness/estimation/single_slice.py` — `build_single_slice_agent`
- `langslice/harness/estimation/group.py` — `build_group_agent`
- `langslice/harness/estimation/runner.py` — driver wrapper
- `langslice/harness/registration/__init__.py` — placeholder
- `tests/test_atlas_plane.py`
- `tests/test_harness_tools.py`
- `tests/test_harness_validators.py`
- `tests/test_harness_prompts.py`
- `tests/test_harness_runner.py`

**Moved (content transplanted, imports updated):**
- `langslice/estimation/google/ap_image_gen.py` → `langslice/harness/estimation/image_gen.py`
- `langslice/estimation/debug.py` → `langslice/harness/estimation/debug.py`

**Modified:**
- `langslice/atlas/space.py` — drop `require_coronal_layout`, add `Plane` type + `slice_axis_index`
- `langslice/atlas/core.py` — all slice helpers take `plane` kwarg
- `langslice/cli.py` — imports updated
- `langslice/whole_brain/estimation_agents.py` — imports updated
- `eval/eval_brain.py`, `eval/eval_group.py` — imports updated
- `slice-bench/slice_bench/adapters/gemini.py` — imports updated
- `environment.yml` — add `google-adk`
- `pyproject.toml` — add `google-adk` dep

**Deleted (or reduced to compat shim):**
- `langslice/estimation/google/ap_single_slice.py`, `ap_multi_slice.py`, `ap_tool_use.py`, `common.py`, `tool_definitions.py`, `batch_eval.py`
- `langslice/estimation/openai/ap_single_slice.py`, `ap_multi_slice.py`, `common.py`, `tool_definitions.py`, `ap_image_gen.py`
- `langslice/estimation/_shared_common.py`, `_tool_logic.py`, `_types.py`
- `langslice/estimation/__init__.py` — either delete or keep as a tiny re-export shim

---

## Phase 0 — Prep

### Task 0.1: Install google-adk and verify baseline tests pass

**Files:**
- Modify: `environment.yml`
- Modify: `pyproject.toml`

- [ ] **Step 1: Add `google-adk` to `environment.yml` under `pip:` section**

```yaml
  - pip:
      - google-adk
```

- [ ] **Step 2: Add `google-adk` to `pyproject.toml` dependencies**

Open `pyproject.toml`, find the `[project]` `dependencies` list, add:
```
    "google-adk",
```

- [ ] **Step 3: Install the new dep**

Run: `pip install google-adk`
Expected: install succeeds; `pip show google-adk` shows a version.

- [ ] **Step 4: Verify existing suite still green**

Run: `python -m pytest -q`
Expected: PASS (record the count; later tasks must not regress).

- [ ] **Step 5: Commit**

```bash
git add environment.yml pyproject.toml
git commit -m "chore(deps): add google-adk"
```

### Task 0.2: Capture pre-refactor smoke baseline

**Files:**
- Create: `eval_outputs/baseline_pre_adk_M01.json`

- [ ] **Step 1: Run the baseline eval**

Pre-filter `references/TestImages/M01/ground_truth.json` for entries with `ap_mm` in [4.0, 7.0] if you want tight mid-brain; otherwise the default eval covers the full brain — document which you ran.

Run:
```bash
python eval/eval_group.py \
  --images references/TestImages/M01 \
  --ground-truth references/TestImages/M01/ground_truth.json \
  --model gemini-3-flash-preview \
  --json > eval_outputs/baseline_pre_adk_M01.json
```
Expected: JSON written with `summary.mae_mm`, `summary.n_slices`, `summary.n_fallbacks`. Record values in the task log.

- [ ] **Step 2: Commit the baseline JSON**

```bash
git add eval_outputs/baseline_pre_adk_M01.json
git commit -m "test(eval): capture pre-ADK smoke baseline on M01"
```

---

## Phase 1 — Atlas plane generalization

Goal: every helper in `langslice/atlas/` accepts `plane: Literal["coronal","sagittal","horizontal"]`, default `"coronal"`.

### Task 1.1: Add Plane type and slice_axis_index helper

**Files:**
- Modify: `langslice/atlas/space.py`
- Create: `tests/test_atlas_plane.py`

- [ ] **Step 1: Write the failing test**

Create `tests/test_atlas_plane.py`:
```python
from langslice.atlas.space import atlas_space_context, slice_axis_index
from langslice.atlas.core import load_atlas


def test_coronal_axis_is_ap():
    atlas = load_atlas("allen_mouse_25um")
    ctx = atlas_space_context(atlas)
    assert slice_axis_index(ctx, "coronal") == ctx.ap_axis_index


def test_sagittal_axis_is_ml():
    atlas = load_atlas("allen_mouse_25um")
    ctx = atlas_space_context(atlas)
    assert slice_axis_index(ctx, "sagittal") == ctx.ml_axis_index


def test_horizontal_axis_is_dv():
    atlas = load_atlas("allen_mouse_25um")
    ctx = atlas_space_context(atlas)
    assert slice_axis_index(ctx, "horizontal") == ctx.dv_axis_index
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m pytest tests/test_atlas_plane.py::test_coronal_axis_is_ap -v`
Expected: FAIL with `ImportError: cannot import name 'slice_axis_index'`.

- [ ] **Step 3: Add the helper to `langslice/atlas/space.py`**

At the top of the file, add a type alias:
```python
from typing import Literal

Plane = Literal["coronal", "sagittal", "horizontal"]
```

At the bottom of the file, add:
```python
def slice_axis_index(context: AtlasSpaceContext, plane: Plane) -> int:
    """Return the axis index normal to the given slicing plane."""
    if plane == "coronal":
        return context.ap_axis_index
    if plane == "sagittal":
        return context.ml_axis_index
    if plane == "horizontal":
        return context.dv_axis_index
    raise ValueError(f"Unknown plane: {plane!r}")
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `python -m pytest tests/test_atlas_plane.py -v`
Expected: 3 PASS.

- [ ] **Step 5: Commit**

```bash
git add langslice/atlas/space.py tests/test_atlas_plane.py
git commit -m "feat(atlas): add Plane type and slice_axis_index helper"
```

### Task 1.2: Parameterize position↔index conversion on plane

**Files:**
- Modify: `langslice/atlas/core.py`
- Modify: `tests/test_atlas_plane.py`

- [ ] **Step 1: Write failing tests**

Append to `tests/test_atlas_plane.py`:
```python
from langslice.atlas.core import (
    get_position_range_mm,
    index_to_position_mm,
    position_mm_to_index,
)


def test_position_range_coronal_default():
    atlas = load_atlas("allen_mouse_25um")
    lo, hi = get_position_range_mm(atlas)
    assert lo == 0.0
    assert hi > 10.0  # mouse brain is ~13 mm AP


def test_position_range_sagittal():
    atlas = load_atlas("allen_mouse_25um")
    lo, hi = get_position_range_mm(atlas, plane="sagittal")
    assert lo == 0.0
    # Mouse brain ~11 mm ML total
    assert 9.0 < hi < 13.0


def test_position_roundtrip_coronal():
    atlas = load_atlas("allen_mouse_25um")
    for mm in [0.0, 2.5, 5.0, 7.5]:
        idx = position_mm_to_index(atlas, mm)
        back = index_to_position_mm(atlas, idx)
        assert abs(back - mm) < 0.025  # within one voxel
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `python -m pytest tests/test_atlas_plane.py::test_position_range_sagittal -v`
Expected: FAIL — `get_position_range_mm` doesn't accept `plane` kwarg.

- [ ] **Step 3: Modify `langslice/atlas/core.py`**

Replace `position_mm_to_index`, `index_to_position_mm`, and `get_position_range_mm` with plane-parameterized versions:

```python
from langslice.atlas.space import Plane, atlas_space_context, slice_axis_index


def _resolution_mm_for_plane(atlas: _AtlasLike, plane: Plane) -> float:
    context = atlas_space_context(atlas)
    axis = slice_axis_index(context, plane)
    return context.resolution_um[axis] / 1000.0


def _n_slices_for_plane(atlas: _AtlasLike, plane: Plane) -> int:
    context = atlas_space_context(atlas)
    axis = slice_axis_index(context, plane)
    return context.shape[axis]


def position_mm_to_index(atlas: _AtlasLike, position_mm: float, *, plane: Plane = "coronal") -> int:
    """Convert a physical position (mm) to an array index along the slice-normal axis."""
    res_mm = _resolution_mm_for_plane(atlas, plane)
    n = _n_slices_for_plane(atlas, plane)
    idx = int(round(position_mm / res_mm))
    if idx < 0 or idx >= n:
        _, max_pos = get_position_range_mm(atlas, plane=plane)
        raise ValueError(
            f"Position {position_mm:.3f}mm (plane={plane}) maps to index {idx}, "
            f"out of range [0, {n - 1}]. "
            f"Valid range for '{atlas.atlas_name}': 0.0mm to {max_pos:.3f}mm"
        )
    return idx


def index_to_position_mm(atlas: _AtlasLike, idx: int, *, plane: Plane = "coronal") -> float:
    """Convert an array index along the slice-normal axis to a physical position (mm)."""
    return idx * _resolution_mm_for_plane(atlas, plane)


def get_position_range_mm(atlas: _AtlasLike, *, plane: Plane = "coronal") -> tuple[float, float]:
    """Return (min_mm, max_mm) along the given slicing plane's normal axis."""
    res_mm = _resolution_mm_for_plane(atlas, plane)
    n = _n_slices_for_plane(atlas, plane)
    return 0.0, (n - 1) * res_mm
```

Delete the `from langslice.atlas.space import atlas_space_context, require_coronal_layout` line (require_coronal_layout is on its way out) and replace with the one above.

- [ ] **Step 4: Run tests to verify they pass**

Run: `python -m pytest tests/test_atlas_plane.py -v`
Expected: 6 PASS total.

- [ ] **Step 5: Run broader suite to catch regressions**

Run: `python -m pytest -q`
Expected: PASS — existing callers default to `plane="coronal"` so behavior is unchanged.

- [ ] **Step 6: Commit**

```bash
git add langslice/atlas/core.py tests/test_atlas_plane.py
git commit -m "feat(atlas): parameterize position/index conversions on plane"
```

### Task 1.3: Parameterize slice-extraction helpers on plane

**Files:**
- Modify: `langslice/atlas/core.py`
- Modify: `tests/test_atlas_plane.py`

- [ ] **Step 1: Write failing test**

Append to `tests/test_atlas_plane.py`:
```python
from langslice.atlas.core import get_reference_slice


def test_reference_slice_shape_differs_by_plane():
    atlas = load_atlas("allen_mouse_25um")
    coronal = get_reference_slice(atlas, 5.0)
    sagittal = get_reference_slice(atlas, 5.0, plane="sagittal")
    horizontal = get_reference_slice(atlas, 5.0, plane="horizontal")
    # All three should return PIL Images with distinct (W, H) tuples:
    # coronal is (ML, DV), sagittal is (AP, DV), horizontal is (ML, AP).
    assert coronal.size != sagittal.size
    assert coronal.size != horizontal.size
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m pytest tests/test_atlas_plane.py::test_reference_slice_shape_differs_by_plane -v`
Expected: FAIL — `get_reference_slice` still hardcodes axis 0.

- [ ] **Step 3: Modify each slice helper in `langslice/atlas/core.py`**

Replace the body of `get_reference_slice`:
```python
def get_reference_slice(atlas: _AtlasLike, position_mm: float, *, plane: Plane = "coronal") -> Image.Image:
    """Get reference slice along the specified plane as grayscale PIL image."""
    idx = position_mm_to_index(atlas, position_mm, plane=plane)
    context = atlas_space_context(atlas)
    axis = slice_axis_index(context, plane)
    ref_volume = np.asarray(atlas.reference)
    reference_slice = np.take(ref_volume, idx, axis=axis)
    normalized = _normalize_to_uint8(reference_slice)
    return Image.fromarray(normalized, mode="L")
```

Apply the same pattern (extract slice via `np.take(volume, idx, axis=axis)` using `slice_axis_index`) to:
- `get_boundary_slice`
- `get_composite_slice`
- `get_colored_region_slice`
- `get_smoothed_boundary_slice`
- `get_additional_reference_slice`
- `get_structure_mask_slice`
- `get_slice_region_metadata`

Each gains `*, plane: Plane = "coronal"` and uses `np.take(..., axis=slice_axis_index(atlas_space_context(atlas), plane))` instead of `[idx, :, :]`.

For `get_region_at_position`: keep the `dv_index` / `ml_index` kwargs (they're in-plane coords, still anatomically meaningful), but resolve the slice-normal axis via the new plane param.

- [ ] **Step 4: Run tests**

Run: `python -m pytest tests/test_atlas_plane.py -v`
Expected: all pass.

Run: `python -m pytest -q`
Expected: no regressions.

- [ ] **Step 5: Commit**

```bash
git add langslice/atlas/core.py tests/test_atlas_plane.py
git commit -m "feat(atlas): parameterize slice helpers on plane"
```

### Task 1.4: Rename `get_coronal_long_edge` → `get_in_plane_long_edge`

**Files:**
- Modify: `langslice/atlas/core.py`
- Modify: (callers) grep all `get_coronal_long_edge` references

- [ ] **Step 1: Grep callers**

Run: `git grep -n "get_coronal_long_edge"`
Record every file that uses the name.

- [ ] **Step 2: Add the new function, keep the old as a deprecated alias**

In `langslice/atlas/core.py`, replace the existing `get_coronal_long_edge`:
```python
def get_in_plane_long_edge(atlas: _AtlasLike, *, plane: Plane = "coronal") -> int:
    """Return the long-edge pixel count of an in-plane slice for the given slicing plane."""
    context = atlas_space_context(atlas)
    normal_axis = slice_axis_index(context, plane)
    in_plane_axes = [a for a in range(3) if a != normal_axis]
    shape = context.shape
    return max(shape[in_plane_axes[0]], shape[in_plane_axes[1]])


def get_coronal_long_edge(atlas: _AtlasLike) -> int:
    """Deprecated alias for get_in_plane_long_edge(atlas, plane='coronal')."""
    import warnings
    warnings.warn(
        "get_coronal_long_edge is deprecated; use get_in_plane_long_edge(atlas, plane=...)",
        DeprecationWarning,
        stacklevel=2,
    )
    return get_in_plane_long_edge(atlas, plane="coronal")
```

- [ ] **Step 3: Update each caller**

For each file from step 1:
- Replace `get_coronal_long_edge(atlas)` with `get_in_plane_long_edge(atlas, plane=...)` where the plane is known (today: always `"coronal"`).
- If the call site is inside an estimation helper that will be refactored later, leave `plane="coronal"` explicit so the rename doesn't change behavior.

- [ ] **Step 4: Run tests**

Run: `python -m pytest -q`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add langslice/atlas/core.py <all modified callers>
git commit -m "refactor(atlas): rename get_coronal_long_edge → get_in_plane_long_edge"
```

### Task 1.5: Remove `require_coronal_layout` enforcement

**Files:**
- Modify: `langslice/atlas/space.py`
- Modify: `langslice/atlas/core.py` (drop import)

- [ ] **Step 1: Grep callers**

Run: `git grep -n "require_coronal_layout"`
Expected: only `space.py` (defn) + `core.py` (import).

- [ ] **Step 2: Delete the function body, leave a stub that warns on call**

Replace `require_coronal_layout` in `langslice/atlas/space.py`:
```python
def require_coronal_layout(context: AtlasSpaceContext) -> AtlasSpaceContext:
    """Deprecated no-op. Pass `plane=...` to slice helpers instead."""
    import warnings
    warnings.warn(
        "require_coronal_layout is deprecated and no longer enforces anything; "
        "pass plane=... to slice helpers directly.",
        DeprecationWarning,
        stacklevel=2,
    )
    return context
```

- [ ] **Step 3: Drop its imports in `langslice/atlas/core.py`**

Remove `require_coronal_layout` from the imports at the top of `core.py`.

- [ ] **Step 4: Run tests**

Run: `python -m pytest -q`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add langslice/atlas/space.py langslice/atlas/core.py
git commit -m "refactor(atlas): remove require_coronal_layout enforcement"
```

---

## Phase 2 — Create harness package, move non-tool-loop files

### Task 2.1: Create harness package skeleton

**Files:**
- Create: `langslice/harness/__init__.py`
- Create: `langslice/harness/estimation/__init__.py`
- Create: `langslice/harness/registration/__init__.py`

- [ ] **Step 1: Create the three `__init__.py` files**

```bash
mkdir -p langslice/harness/estimation langslice/harness/registration
```

`langslice/harness/__init__.py`:
```python
"""LangSlice agent-harness package — ADK-based estimation and (future) registration."""
```

`langslice/harness/estimation/__init__.py`:
```python
"""Position-estimation harness: single-slice and multi-slice-group tool-use agents on Google ADK."""
# Public API is populated by later tasks.
```

`langslice/harness/registration/__init__.py`:
```python
"""Placeholder for future registration agent. Populated post-hackathon."""
```

- [ ] **Step 2: Confirm nothing imports broken**

Run: `python -c "import langslice.harness; import langslice.harness.estimation; import langslice.harness.registration"`
Expected: exits 0.

- [ ] **Step 3: Commit**

```bash
git add langslice/harness
git commit -m "feat(harness): create harness package skeleton"
```

### Task 2.2: Move `_types.py` with rename + back-compat alias

**Files:**
- Create: `langslice/harness/estimation/_types.py`
- Modify: `langslice/estimation/_types.py` (turn into a re-export shim)

- [ ] **Step 1: Create the new types module**

`langslice/harness/estimation/_types.py`:
```python
"""Provider-agnostic result types for position estimation."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass
class PositionResult:
    """Result from a single-slice position estimate."""
    position_mm: float
    reasoning: str
    debug_dir: str | None = None


@dataclass
class MultiSliceResult:
    """Result from multi-slice group estimation."""
    positions: list[PositionResult]
    group_reasoning: str
    debug_dir: str | None = None


# Back-compat alias; whole_brain/* and slice-bench still import APResult.
APResult = PositionResult
```

- [ ] **Step 2: Turn the old module into a re-export shim**

Overwrite `langslice/estimation/_types.py` with:
```python
"""Deprecated re-export shim. Import from langslice.harness.estimation instead."""
from langslice.harness.estimation._types import (  # noqa: F401
    APResult,
    MultiSliceResult,
    PositionResult,
)
```

- [ ] **Step 3: Run tests**

Run: `python -m pytest -q`
Expected: PASS.

- [ ] **Step 4: Commit**

```bash
git add langslice/harness/estimation/_types.py langslice/estimation/_types.py
git commit -m "refactor(types): move result types to harness/estimation, add PositionResult + APResult alias"
```

### Task 2.3: Move `debug.py` and `image_gen.py`

**Files:**
- Create: `langslice/harness/estimation/debug.py` (copy of `langslice/estimation/debug.py`)
- Create: `langslice/harness/estimation/image_gen.py` (copy of `langslice/estimation/google/ap_image_gen.py`)

- [ ] **Step 1: Copy the files**

```bash
cp langslice/estimation/debug.py langslice/harness/estimation/debug.py
cp langslice/estimation/google/ap_image_gen.py langslice/harness/estimation/image_gen.py
```

- [ ] **Step 2: Update imports inside the copied files**

In `langslice/harness/estimation/image_gen.py`:
- Change `from langslice.estimation._types import APResult` → `from langslice.harness.estimation._types import APResult`
- Change `from langslice.estimation.google.common import ...` → for each import, copy the source or leave a TODO referencing the equivalent in the old module; these will be rewired in Phase 3. For now, keep the import pointing at `langslice.estimation.google.common` (it still exists).
- Change `from langslice.estimation.google.tool_definitions import _build_atlas_grid` → keep pointing at the old location; will be addressed later.
- Change `from langslice.atlas.core import get_coronal_long_edge` → `from langslice.atlas.core import get_in_plane_long_edge as get_coronal_long_edge` (one-line local alias so we don't have to retouch function bodies mid-migration).

Same pattern for `debug.py`: update its `from langslice.estimation...` internal imports to point at the new location where a symbol has moved; everything else stays.

- [ ] **Step 3: Update the ONE consumer of `image_gen.py`**

`langslice/whole_brain/estimation_agents.py` imports `estimate_position_image_gen`. Change:
```python
from langslice.estimation import APResult, estimate_position_image_gen
```
to:
```python
from langslice.harness.estimation._types import APResult
from langslice.harness.estimation.image_gen import estimate_position_image_gen
```

- [ ] **Step 4: Leave a re-export shim in the old location**

Overwrite `langslice/estimation/google/ap_image_gen.py`:
```python
"""Deprecated re-export shim."""
from langslice.harness.estimation.image_gen import estimate_position_image_gen  # noqa: F401
```

Overwrite `langslice/estimation/debug.py`:
```python
"""Deprecated re-export shim."""
from langslice.harness.estimation.debug import write_debug_artifacts  # noqa: F401
```

- [ ] **Step 5: Run tests**

Run: `python -m pytest -q`
Expected: PASS.

- [ ] **Step 6: Commit**

```bash
git add langslice/harness/estimation/debug.py langslice/harness/estimation/image_gen.py \
        langslice/estimation/debug.py langslice/estimation/google/ap_image_gen.py \
        langslice/whole_brain/estimation_agents.py
git commit -m "refactor(harness): move debug and image_gen into harness/estimation"
```

---

## Phase 3 — Single-slice ADK port

### Task 3.1: Define SessionState builder and artifact-key conventions

**Files:**
- Create: `langslice/harness/estimation/session.py`
- Create: `tests/test_harness_runner.py` (will grow across later tasks)

- [ ] **Step 1: Write the failing test**

`tests/test_harness_runner.py`:
```python
from langslice.harness.estimation.session import (
    ARTIFACT_TARGET,
    build_initial_state,
)


def test_initial_state_single_slice():
    state = build_initial_state(
        atlas_name="allen_mouse_25um",
        plane="coronal",
        pos_lo=0.0, pos_hi=13.2,
        n_slices=1, interval_mm=0.0, thickness_um=50,
        max_iterations=20,
    )
    assert state["atlas"] == "allen_mouse_25um"
    assert state["plane"] == "coronal"
    assert state["axis_label"] == "AP"
    assert state["n_slices"] == 1
    assert state["fetched_positions"] == []
    assert state["saw_broad_sweep"] is False
    assert state["saw_narrow_sweep"] is False
    assert state["submit_attempts"] == 0
    assert state["result"] is None


def test_artifact_target_constant():
    assert ARTIFACT_TARGET == "target"
```

- [ ] **Step 2: Run it and watch it fail**

Run: `python -m pytest tests/test_harness_runner.py::test_initial_state_single_slice -v`
Expected: FAIL — module doesn't exist.

- [ ] **Step 3: Implement `session.py`**

```python
"""Session-state builder and artifact-key conventions for the estimation harness."""

from __future__ import annotations

from typing import Any

from langslice.atlas.space import Plane

ARTIFACT_TARGET = "target"  # single-slice target image key
ARTIFACT_TARGET_PREFIX = "target:"  # multi-slice: target:1, target:2, ...
ARTIFACT_ATLAS_PREFIX = "atlas:"  # cached atlas slices: atlas:3.20
ARTIFACT_ZOOM_PREFIX = "zoom:"
ARTIFACT_SIDE_BY_SIDE_PREFIX = "side_by_side:"


_PLANE_TO_AXIS_LABEL: dict[str, str] = {
    "coronal": "AP",
    "sagittal": "ML",
    "horizontal": "DV",
}


def axis_label_for(plane: Plane) -> str:
    return _PLANE_TO_AXIS_LABEL[plane]


def build_initial_state(
    *,
    atlas_name: str,
    plane: Plane,
    pos_lo: float,
    pos_hi: float,
    n_slices: int,
    interval_mm: float,
    thickness_um: int,
    max_iterations: int,
) -> dict[str, Any]:
    """Return the initial `tool_context.state` dict for a run."""
    return {
        "atlas": atlas_name,
        "plane": plane,
        "axis_label": axis_label_for(plane),
        "pos_lo": pos_lo,
        "pos_hi": pos_hi,
        "n_slices": n_slices,
        "interval_mm": interval_mm,
        "thickness_um": thickness_um,
        "fetched_positions": [],
        "saw_broad_sweep": False,
        "saw_narrow_sweep": False,
        "images_fetched": 0,
        "submit_attempts": 0,
        "result": None,
        "max_iterations": max_iterations,
    }


def atlas_key(position_mm: float) -> str:
    """Canonical artifact key for an atlas slice at a given position."""
    return f"{ARTIFACT_ATLAS_PREFIX}{position_mm:.2f}"


def target_key(index: int | None = None) -> str:
    """Canonical artifact key for a target image ('target' or 'target:N')."""
    if index is None:
        return ARTIFACT_TARGET
    return f"{ARTIFACT_TARGET_PREFIX}{index}"
```

- [ ] **Step 4: Tests pass**

Run: `python -m pytest tests/test_harness_runner.py -v`
Expected: 2 PASS.

- [ ] **Step 5: Commit**

```bash
git add langslice/harness/estimation/session.py tests/test_harness_runner.py
git commit -m "feat(harness): session-state builder and artifact-key conventions"
```

### Task 3.2: Plane-aware prompt builders

**Files:**
- Create: `langslice/harness/estimation/prompts.py`
- Create: `tests/test_harness_prompts.py`

- [ ] **Step 1: Write failing tests**

`tests/test_harness_prompts.py`:
```python
from langslice.harness.estimation.prompts import (
    build_group_prompt,
    build_single_slice_prompt,
)


def test_single_slice_prompt_coronal_mentions_ap_and_olfactory():
    p = build_single_slice_prompt(
        atlas_name="allen_mouse_25um", plane="coronal",
        pos_lo=0.0, pos_hi=13.2, species="mouse",
    )
    assert "AP" in p
    assert "olfactory" in p.lower()
    assert "0.00" in p or "0.0" in p
    assert "13.2" in p


def test_single_slice_prompt_sagittal_mentions_ml_not_ap():
    p = build_single_slice_prompt(
        atlas_name="allen_mouse_25um", plane="sagittal",
        pos_lo=0.0, pos_hi=11.0, species="mouse",
    )
    assert "ML" in p
    assert "AP" not in p
    assert "olfactory" not in p.lower()


def test_group_prompt_mentions_interval_and_n_slices():
    p = build_group_prompt(
        atlas_name="allen_mouse_25um", plane="coronal",
        pos_lo=0.0, pos_hi=13.2, species="mouse",
        n_slices=4, interval_mm=0.200, thickness_um=50,
    )
    assert "4" in p
    assert "0.200" in p or "200" in p  # micron or mm form
    assert "AP" in p
```

- [ ] **Step 2: Run and watch them fail**

Run: `python -m pytest tests/test_harness_prompts.py -v`
Expected: FAIL — module doesn't exist.

- [ ] **Step 3: Implement `prompts.py`**

```python
"""Plane-aware system-instruction builders for the position-estimation agents."""

from __future__ import annotations

from langslice.atlas.space import Plane

_PLANE_BOILERPLATE: dict[str, str] = {
    "coronal":
        "Coordinate system: 0.0 mm is the anterior edge (olfactory bulb); "
        "larger mm moves posterior toward the cerebellum.",
    "sagittal":
        "Coordinate system: 0.0 mm is the left hemisphere lateral edge; "
        "larger mm moves to the right.",
    "horizontal":
        "Coordinate system: 0.0 mm is dorsal (top of brain); "
        "larger mm moves ventral.",
}

_PLANE_AXIS_LABEL: dict[str, str] = {
    "coronal": "AP",
    "sagittal": "ML",
    "horizontal": "DV",
}


def build_single_slice_prompt(
    *,
    atlas_name: str,
    plane: Plane,
    pos_lo: float,
    pos_hi: float,
    species: str,
) -> str:
    axis = _PLANE_AXIS_LABEL[plane]
    boilerplate = _PLANE_BOILERPLATE[plane]
    return (
        f"You are an expert neuroanatomist. You are given a histology brain "
        f"slice image and must determine its {axis} position within a "
        f"reference atlas. {boilerplate}\n\n"
        f"Atlas: {atlas_name} ({species}). "
        f"Valid {axis} range: {pos_lo:.2f}–{pos_hi:.2f} mm.\n\n"
        f"You have tools to fetch atlas reference images, zoom into regions "
        f"of interest, view side-by-side comparisons, and submit your final "
        f"estimate.\n\n"
        f"RECOMMENDED STRATEGY:\n"
        f"1. Call `fetch_atlas` with broadly spaced positions "
        f"(e.g., [2, 4, 6, 8, 10]) to find the general region.\n"
        f"2. Call `fetch_atlas` with tighter positions around your best match.\n"
        f"3. Call `fetch_atlas` with very fine positions (~0.1–0.2mm apart) to pinpoint.\n"
        f"4. When a specific landmark is unclear, call `zoom` with a bounding box "
        f"[y1, x1, y2, x2] (0–1000) on 'target' or 'atlas:<mm>'.\n"
        f"5. To directly compare two sections, call `side_by_side` with two sources.\n"
        f"6. Verify neighbors, then call `submit_estimate`.\n\n"
        f"If atlas images don't look similar to the target, DO NOT keep narrowing "
        f"in the same area. Go back and try a different region.\n\n"
        f"Think carefully before each tool call, but always follow up with an action."
    )


def build_group_prompt(
    *,
    atlas_name: str,
    plane: Plane,
    pos_lo: float,
    pos_hi: float,
    species: str,
    n_slices: int,
    interval_mm: float,
    thickness_um: int,
) -> str:
    axis = _PLANE_AXIS_LABEL[plane]
    boilerplate = _PLANE_BOILERPLATE[plane]
    return (
        f"You are an expert neuroanatomist. You are given {n_slices} consecutive "
        f"histology brain slice images from the same brain, ordered along the "
        f"{axis} axis (Slice 1 = lowest {axis}, Slice {n_slices} = highest).\n\n"
        f"Section parameters:\n"
        f"- Slice thickness: {thickness_um} µm\n"
        f"- Section interval: {interval_mm:.3f} mm (center-to-center)\n\n"
        f"{boilerplate}\n"
        f"Atlas: {atlas_name} ({species}). "
        f"Valid {axis} range: {pos_lo:.2f}–{pos_hi:.2f} mm.\n\n"
        f"Your task: determine the {axis} position of EACH slice.\n\n"
        f"STRATEGY:\n"
        f"1. Examine all slices; describe 2–3 prominent landmarks visible in "
        f"Slice 1 and Slice {n_slices}.\n"
        f"2. Call `fetch_atlas` with broadly spaced positions to find the "
        f"general area.\n"
        f"3. Narrow down by comparing atlas slices with your input slices.\n"
        f"4. Use the known {interval_mm:.3f} mm interval as a constraint — once "
        f"you confidently match ANY slice, derive approximate positions for "
        f"the others.\n"
        f"5. Use `zoom` to examine specific features and `side_by_side` to "
        f"compare any two sources directly.\n"
        f"6. Submit all {n_slices} positions via `submit_group_estimate`.\n\n"
        f"If atlas images don't match your slices, try a different region — "
        f"restart rather than commit to the wrong neighborhood."
    )
```

- [ ] **Step 4: Tests pass**

Run: `python -m pytest tests/test_harness_prompts.py -v`
Expected: 3 PASS.

- [ ] **Step 5: Commit**

```bash
git add langslice/harness/estimation/prompts.py tests/test_harness_prompts.py
git commit -m "feat(harness): plane-aware prompt builders"
```

### Task 3.3: Implement `fetch_atlas` tool

**Files:**
- Create: `langslice/harness/estimation/tools.py`
- Create: `tests/test_harness_tools.py`

- [ ] **Step 1: Write failing tests for pure helpers first**

`tests/test_harness_tools.py`:
```python
import pytest
from PIL import Image

from langslice.harness.estimation.tools import (
    _clamp_and_dedupe_positions,
    _image_to_jpeg_bytes,
    _is_broad_sweep,
    _is_narrow_sweep,
    _parse_atlas_key,
)


def test_parse_atlas_key():
    assert _parse_atlas_key("atlas:3.20") == 3.20
    assert _parse_atlas_key("atlas:10.00") == 10.0
    with pytest.raises(ValueError):
        _parse_atlas_key("target")
    with pytest.raises(ValueError):
        _parse_atlas_key("atlas:not-a-number")


def test_is_broad_sweep_threshold():
    assert _is_broad_sweep([1.0, 4.0, 7.0]) is True
    assert _is_broad_sweep([1.0, 2.0]) is False  # too few positions


def test_is_narrow_sweep_threshold():
    assert _is_narrow_sweep([4.0, 4.3, 4.6]) is True  # span 0.6mm ≤ 1.0
    assert _is_narrow_sweep([4.0, 5.0, 6.5]) is False  # span 2.5mm > 1.0
    assert _is_narrow_sweep([4.0, 4.5]) is False  # too few


def test_clamp_and_dedupe_positions():
    out = _clamp_and_dedupe_positions(
        [1.0, 1.005, 2.0, -1.0, 99.0], pos_lo=0.0, pos_hi=10.0, dedupe_tol=0.02
    )
    # 1.0 kept, 1.005 coalesced, 2.0 kept, -1.0 clamped to 0.0, 99.0 clamped to 10.0
    assert 0.0 in out
    assert 10.0 in out
    assert out.count(1.0) == 1  # 1.005 dedupe'd into 1.0
    assert 2.0 in out


def test_image_to_jpeg_bytes_roundtrip():
    img = Image.new("RGB", (64, 64), (128, 64, 32))
    blob = _image_to_jpeg_bytes(img)
    assert isinstance(blob, bytes)
    assert len(blob) > 100  # has content
    assert blob[:2] == b"\xff\xd8"  # JPEG magic
```

- [ ] **Step 2: Run, watch fail**

Run: `python -m pytest tests/test_harness_tools.py -v`
Expected: FAIL — module doesn't exist.

- [ ] **Step 3: Implement the pure helpers in `tools.py`**

```python
"""The four position-estimation tools, wired as plain Python functions for ADK auto-wrapping.

Tools return dicts. Image outputs are placed under the key "images" as a list
of google.genai.types.Part objects — ADK surfaces these to the next model turn.
Tools also save images as artifacts via tool_context.save_artifact so later
tools (zoom, side_by_side) can retrieve them by key.
"""

from __future__ import annotations

import io
import hashlib
from typing import Any

from google.genai import types
from PIL import Image

from langslice.atlas.core import get_reference_slice, get_in_plane_long_edge, load_atlas
from langslice.harness.estimation.session import (
    ARTIFACT_ATLAS_PREFIX,
    ARTIFACT_SIDE_BY_SIDE_PREFIX,
    ARTIFACT_TARGET,
    ARTIFACT_TARGET_PREFIX,
    ARTIFACT_ZOOM_PREFIX,
    atlas_key,
)


# ---- Pure helpers -------------------------------------------------------


def _parse_atlas_key(source: str) -> float:
    if not source.startswith(ARTIFACT_ATLAS_PREFIX):
        raise ValueError(f"Not an atlas source: {source!r}")
    tail = source[len(ARTIFACT_ATLAS_PREFIX):]
    try:
        return float(tail)
    except ValueError as exc:
        raise ValueError(f"Bad atlas position in {source!r}") from exc


def _is_broad_sweep(positions: list[float]) -> bool:
    return len(positions) >= 3


def _is_narrow_sweep(positions: list[float]) -> bool:
    if len(positions) < 3:
        return False
    return (max(positions) - min(positions)) <= 1.0


def _clamp_and_dedupe_positions(
    positions: list[float], *, pos_lo: float, pos_hi: float, dedupe_tol: float = 0.02
) -> list[float]:
    clamped = [max(pos_lo, min(pos_hi, float(p))) for p in positions]
    out: list[float] = []
    for p in clamped:
        if any(abs(p - q) <= dedupe_tol for q in out):
            continue
        out.append(p)
    return out


def _image_to_jpeg_bytes(img: Image.Image, quality: int = 85) -> bytes:
    buf = io.BytesIO()
    img.convert("RGB").save(buf, format="JPEG", quality=quality)
    return buf.getvalue()


def _image_to_part(img: Image.Image) -> types.Part:
    return types.Part.from_bytes(
        mime_type="image/jpeg", data=_image_to_jpeg_bytes(img)
    )


def _short_hash(bbox: list[int]) -> str:
    return hashlib.sha1(str(bbox).encode("utf-8")).hexdigest()[:8]
```

- [ ] **Step 4: Pure-helper tests pass**

Run: `python -m pytest tests/test_harness_tools.py -v`
Expected: 5 PASS.

- [ ] **Step 5: Add failing test for fetch_atlas**

Append to `tests/test_harness_tools.py`:
```python
from unittest.mock import MagicMock

from langslice.harness.estimation.session import build_initial_state
from langslice.harness.estimation.tools import fetch_atlas


def _fake_tool_context(state: dict) -> MagicMock:
    ctx = MagicMock()
    ctx.state = state
    ctx.save_artifact = MagicMock(return_value=1)
    return ctx


def test_fetch_atlas_returns_ok_and_updates_state():
    state = build_initial_state(
        atlas_name="allen_mouse_25um", plane="coronal",
        pos_lo=0.0, pos_hi=13.2, n_slices=1,
        interval_mm=0.0, thickness_um=50, max_iterations=20,
    )
    ctx = _fake_tool_context(state)
    result = fetch_atlas(positions_mm=[2.0, 5.0, 8.0], tool_context=ctx)
    assert result["status"] == "ok"
    assert len(result["images"]) == 3
    assert state["saw_broad_sweep"] is True
    assert state["images_fetched"] == 3
    assert ctx.save_artifact.call_count == 3


def test_fetch_atlas_rejects_empty_positions():
    state = build_initial_state(
        atlas_name="allen_mouse_25um", plane="coronal",
        pos_lo=0.0, pos_hi=13.2, n_slices=1,
        interval_mm=0.0, thickness_um=50, max_iterations=20,
    )
    ctx = _fake_tool_context(state)
    result = fetch_atlas(positions_mm=[], tool_context=ctx)
    assert result["status"] == "error"
    assert result["error"] == "BAD_ARGS"
```

- [ ] **Step 6: Run, watch fail**

Run: `python -m pytest tests/test_harness_tools.py -v`
Expected: 5 PASS, 2 FAIL (the new ones).

- [ ] **Step 7: Implement `fetch_atlas` in `tools.py`**

Append to `langslice/harness/estimation/tools.py`:
```python
def fetch_atlas(
    positions_mm: list[float], tool_context: Any
) -> dict[str, Any]:
    """Fetch 1–8 atlas sections along the session's slicing plane.

    Positions outside the valid range are clamped. Duplicate positions within
    0.02 mm of an already-requested one are coalesced. Each returned slice is
    saved as an artifact keyed 'atlas:<mm:.2f>' and returned as a types.Part
    image in the "images" field of the response so the next model turn can
    see it.

    Returns:
        {"status": "ok", "positions_mm": [...], "images": [Part, ...]} or
        {"status": "error", "error": "BAD_ARGS" | "EMPTY_RESULT"}.
    """
    state = tool_context.state
    if not positions_mm:
        return {"status": "error", "error": "BAD_ARGS"}

    pos_lo = float(state["pos_lo"])
    pos_hi = float(state["pos_hi"])
    plane = state["plane"]
    atlas_name = state["atlas"]

    capped = list(positions_mm)[:8]
    positions = _clamp_and_dedupe_positions(capped, pos_lo=pos_lo, pos_hi=pos_hi)
    if not positions:
        return {"status": "error", "error": "EMPTY_RESULT"}

    atlas = load_atlas(atlas_name)
    parts: list[types.Part] = []
    descriptions: list[str] = []
    for pos in positions:
        img = get_reference_slice(atlas, pos, plane=plane)
        part = _image_to_part(img)
        parts.append(part)
        key = atlas_key(pos)
        tool_context.save_artifact(key, part)
        descriptions.append(f"{pos:.2f} mm")

    # Update session state
    state.setdefault("fetched_positions", []).extend(positions)
    state["images_fetched"] = int(state.get("images_fetched", 0)) + len(positions)
    if _is_broad_sweep(positions):
        state["saw_broad_sweep"] = True
    if _is_narrow_sweep(positions):
        state["saw_narrow_sweep"] = True

    return {
        "status": "ok",
        "positions_mm": positions,
        "description": f"{len(positions)} atlas sections at: " + ", ".join(descriptions),
        "images": parts,
    }
```

- [ ] **Step 8: Tests pass**

Run: `python -m pytest tests/test_harness_tools.py -v`
Expected: 7 PASS.

- [ ] **Step 9: Commit**

```bash
git add langslice/harness/estimation/tools.py tests/test_harness_tools.py
git commit -m "feat(harness): fetch_atlas tool with state updates"
```

### Task 3.4: Implement `submit_estimate` and its validator callback

**Files:**
- Modify: `langslice/harness/estimation/tools.py`
- Create: `langslice/harness/estimation/validators.py`
- Create: `tests/test_harness_validators.py`

- [ ] **Step 1: Write failing tests for validators**

`tests/test_harness_validators.py`:
```python
from unittest.mock import MagicMock

import pytest

from langslice.harness.estimation.session import build_initial_state
from langslice.harness.estimation.validators import gate_submit_tool


def _make_state(**overrides):
    state = build_initial_state(
        atlas_name="allen_mouse_25um", plane="coronal",
        pos_lo=0.0, pos_hi=13.2, n_slices=1,
        interval_mm=0.0, thickness_um=50, max_iterations=20,
    )
    state.update(overrides)
    return state


def _fake_ctx(state):
    ctx = MagicMock()
    ctx.state = state
    ctx.actions = MagicMock()
    return ctx


class _Tool:
    def __init__(self, name): self.name = name


def test_gate_rejects_submit_without_broad_sweep():
    state = _make_state(saw_broad_sweep=False)
    ctx = _fake_ctx(state)
    out = gate_submit_tool(
        _Tool("submit_estimate"), {"position_mm": 5.0, "reasoning": "x"}, ctx
    )
    assert out is not None and out["status"] == "error"
    assert state["submit_attempts"] == 1


def test_gate_relaxes_after_two_rejections():
    state = _make_state(saw_broad_sweep=False, submit_attempts=2)
    ctx = _fake_ctx(state)
    out = gate_submit_tool(
        _Tool("submit_estimate"), {"position_mm": 5.0, "reasoning": "x"}, ctx
    )
    # After 2+ attempts, relaxation lets the submit through.
    assert out is None


def test_gate_passes_when_all_checks_satisfied():
    state = _make_state(
        saw_broad_sweep=True, saw_narrow_sweep=True,
        fetched_positions=[4.8, 5.2],
    )
    ctx = _fake_ctx(state)
    out = gate_submit_tool(
        _Tool("submit_estimate"), {"position_mm": 5.0, "reasoning": "x"}, ctx
    )
    assert out is None


def test_gate_rejects_group_non_monotonic():
    state = _make_state(
        n_slices=3, interval_mm=0.200,
        saw_broad_sweep=True, saw_narrow_sweep=True,
    )
    ctx = _fake_ctx(state)
    out = gate_submit_tool(
        _Tool("submit_group_estimate"),
        {"positions_mm": [5.0, 4.8, 5.4], "reasoning": "x"},
        ctx,
    )
    assert out is not None and "monotonic" in out["error"].lower()


def test_gate_rejects_group_bad_interval():
    state = _make_state(
        n_slices=3, interval_mm=0.200,
        saw_broad_sweep=True, saw_narrow_sweep=True,
    )
    ctx = _fake_ctx(state)
    out = gate_submit_tool(
        _Tool("submit_group_estimate"),
        {"positions_mm": [4.0, 4.2, 5.5], "reasoning": "x"},  # 1.3mm gap
        ctx,
    )
    assert out is not None
    assert "interval" in out["error"].lower()


def test_gate_ignores_non_submit_tools():
    state = _make_state()
    ctx = _fake_ctx(state)
    out = gate_submit_tool(_Tool("fetch_atlas"), {"positions_mm": [1.0]}, ctx)
    assert out is None
```

- [ ] **Step 2: Run, watch all fail**

Run: `python -m pytest tests/test_harness_validators.py -v`
Expected: 6 FAIL.

- [ ] **Step 3: Implement `validators.py`**

```python
"""before_tool_callback: gate the submit tools on broad/narrow sweep + bracket + monotonicity."""

from __future__ import annotations

from typing import Any


_RELAXATION_AFTER_ATTEMPTS = 2


def _has_neighbor_bracket(
    fetched: list[float], center: float, *, pos_lo: float, pos_hi: float, tol: float = 0.25,
    edge_margin: float = 0.25,
) -> bool:
    needs_lower = center > pos_lo + edge_margin
    needs_upper = center < pos_hi - edge_margin
    has_lower = any(center - tol <= p < center for p in fetched)
    has_upper = any(center < p <= center + tol for p in fetched)
    return (has_lower or not needs_lower) and (has_upper or not needs_upper)


def _gate_single(args: dict[str, Any], state: dict[str, Any]) -> dict[str, Any] | None:
    relaxed = state.get("submit_attempts", 0) >= _RELAXATION_AFTER_ATTEMPTS
    if not state.get("saw_broad_sweep") and not relaxed:
        return {"status": "error", "error": "Run a broad `fetch_atlas` sweep before submitting."}
    if not state.get("saw_narrow_sweep") and not relaxed:
        return {
            "status": "error",
            "error": "Run a narrow `fetch_atlas` sweep around your best candidate before submitting.",
        }
    pos = float(args.get("position_mm", 0.0))
    if not _has_neighbor_bracket(
        state.get("fetched_positions", []), pos,
        pos_lo=float(state["pos_lo"]), pos_hi=float(state["pos_hi"]),
    ) and not relaxed:
        return {
            "status": "error",
            "error": (
                f"Verify at least one lower and one higher neighboring atlas "
                f"position around {pos:.2f} mm before submitting."
            ),
        }
    return None


def _gate_group(args: dict[str, Any], state: dict[str, Any]) -> dict[str, Any] | None:
    relaxed = state.get("submit_attempts", 0) >= _RELAXATION_AFTER_ATTEMPTS
    positions = list(args.get("positions_mm", []))
    n_expected = int(state["n_slices"])
    if len(positions) != n_expected:
        return {
            "status": "error",
            "error": f"Expected {n_expected} positions, got {len(positions)}.",
        }
    if not relaxed and not all(positions[i] <= positions[i + 1] for i in range(len(positions) - 1)):
        return {"status": "error", "error": "Positions must be monotonically increasing."}

    interval = float(state["interval_mm"])
    intervals = [positions[i + 1] - positions[i] for i in range(len(positions) - 1)]
    tolerance = max(0.5 * interval, 0.25)
    if not relaxed:
        bad = [(i, iv) for i, iv in enumerate(intervals) if abs(iv - interval) > tolerance]
        if bad:
            detail = "; ".join(f"{i + 1}->{i + 2}: {iv:.3f}mm" for i, iv in bad)
            return {
                "status": "error",
                "error": (
                    f"Intervals deviate >50% from expected {interval:.3f}mm: {detail}."
                ),
            }
    if not state.get("saw_broad_sweep") and not relaxed:
        return {"status": "error", "error": "Run a broad `fetch_atlas` sweep before submitting."}
    if not state.get("saw_narrow_sweep") and not relaxed:
        return {"status": "error", "error": "Run a narrow `fetch_atlas` sweep before submitting."}
    return None


def gate_submit_tool(tool: Any, args: dict[str, Any], tool_context: Any) -> dict[str, Any] | None:
    """ADK before_tool_callback: short-circuit submit tools that fail gating."""
    name = getattr(tool, "name", None)
    if name == "submit_estimate":
        err = _gate_single(args, tool_context.state)
    elif name == "submit_group_estimate":
        err = _gate_group(args, tool_context.state)
    else:
        return None  # Pass through all non-submit tools untouched.

    if err is not None:
        tool_context.state["submit_attempts"] = int(
            tool_context.state.get("submit_attempts", 0)
        ) + 1
    return err
```

- [ ] **Step 4: Validator tests pass**

Run: `python -m pytest tests/test_harness_validators.py -v`
Expected: 6 PASS.

- [ ] **Step 5: Add `submit_estimate` and `submit_group_estimate` tools**

Append to `langslice/harness/estimation/tools.py`:
```python
def submit_estimate(
    position_mm: float, reasoning: str, tool_context: Any
) -> dict[str, Any]:
    """Submit the final position estimate for the target slice.

    Only call this when you have completed broad + narrow atlas sweeps and
    verified at least one neighbor on each side of your candidate position.
    """
    tool_context.state["result"] = {"position_mm": float(position_mm), "reasoning": str(reasoning)}
    tool_context.actions.escalate = True
    return {"status": "ok", "position_mm": float(position_mm)}


def submit_group_estimate(
    positions_mm: list[float], reasoning: str, tool_context: Any
) -> dict[str, Any]:
    """Submit the final position estimates for all slices in the group, in order."""
    tool_context.state["result"] = {
        "positions_mm": [float(p) for p in positions_mm],
        "reasoning": str(reasoning),
    }
    tool_context.actions.escalate = True
    return {"status": "ok", "positions_mm": [float(p) for p in positions_mm]}
```

- [ ] **Step 6: Add tests for the submit tools**

Append to `tests/test_harness_tools.py`:
```python
from langslice.harness.estimation.tools import submit_estimate, submit_group_estimate


def test_submit_estimate_sets_state_and_escalates():
    state = build_initial_state(
        atlas_name="allen_mouse_25um", plane="coronal",
        pos_lo=0.0, pos_hi=13.2, n_slices=1,
        interval_mm=0.0, thickness_um=50, max_iterations=20,
    )
    ctx = _fake_tool_context(state)
    ctx.actions = MagicMock()
    out = submit_estimate(position_mm=5.0, reasoning="hippocampus visible", tool_context=ctx)
    assert out["status"] == "ok"
    assert state["result"] == {"position_mm": 5.0, "reasoning": "hippocampus visible"}
    assert ctx.actions.escalate is True


def test_submit_group_estimate_sets_state_and_escalates():
    state = build_initial_state(
        atlas_name="allen_mouse_25um", plane="coronal",
        pos_lo=0.0, pos_hi=13.2, n_slices=3,
        interval_mm=0.200, thickness_um=50, max_iterations=25,
    )
    ctx = _fake_tool_context(state)
    ctx.actions = MagicMock()
    out = submit_group_estimate(
        positions_mm=[5.0, 5.2, 5.4], reasoning="ok", tool_context=ctx,
    )
    assert out["status"] == "ok"
    assert state["result"]["positions_mm"] == [5.0, 5.2, 5.4]
    assert ctx.actions.escalate is True
```

- [ ] **Step 7: Tests pass**

Run: `python -m pytest tests/test_harness_validators.py tests/test_harness_tools.py -v`
Expected: all PASS.

- [ ] **Step 8: Commit**

```bash
git add langslice/harness/estimation/tools.py \
        langslice/harness/estimation/validators.py \
        tests/test_harness_tools.py \
        tests/test_harness_validators.py
git commit -m "feat(harness): submit tools + before_tool_callback validators"
```

### Task 3.5: Build the single-slice agent

**Files:**
- Create: `langslice/harness/estimation/single_slice.py`

- [ ] **Step 1: Write failing test**

Append to `tests/test_harness_runner.py`:
```python
from langslice.harness.estimation.single_slice import build_single_slice_agent


def test_build_single_slice_agent_registers_four_tools():
    agent = build_single_slice_agent(
        atlas_name="allen_mouse_25um", plane="coronal",
        species="mouse", pos_lo=0.0, pos_hi=13.2,
        model="gemini-3-flash-preview",
    )
    tool_names = {getattr(t, "__name__", None) or getattr(t, "name", None) for t in agent.tools}
    assert "fetch_atlas" in tool_names
    assert "zoom" in tool_names
    assert "side_by_side" in tool_names
    assert "submit_estimate" in tool_names
    assert agent.instruction  # non-empty prompt
```

- [ ] **Step 2: Watch it fail**

Run: `python -m pytest tests/test_harness_runner.py::test_build_single_slice_agent_registers_four_tools -v`
Expected: FAIL — module doesn't exist (and `zoom`/`side_by_side` aren't defined yet; they'll be stubs for now).

- [ ] **Step 3: Add stubs for zoom and side_by_side to `tools.py`**

Append to `langslice/harness/estimation/tools.py` (full implementations come in Phase 5/6):
```python
def zoom(source: str, bbox: list[int], tool_context: Any) -> dict[str, Any]:
    """STUB — full implementation in Phase 5. Returns a NOT_IMPLEMENTED error."""
    return {"status": "error", "error": "NOT_IMPLEMENTED"}


def side_by_side(left: str, right: str, tool_context: Any) -> dict[str, Any]:
    """STUB — full implementation in Phase 6. Returns a NOT_IMPLEMENTED error."""
    return {"status": "error", "error": "NOT_IMPLEMENTED"}
```

- [ ] **Step 4: Implement `single_slice.py`**

```python
"""Builder for the single-slice position-estimation LlmAgent."""

from __future__ import annotations

from google.adk.agents import LlmAgent
from google.genai import types

from langslice.atlas.space import Plane
from langslice.harness.estimation.prompts import build_single_slice_prompt
from langslice.harness.estimation.tools import (
    fetch_atlas,
    side_by_side,
    submit_estimate,
    zoom,
)
from langslice.harness.estimation.validators import gate_submit_tool


def build_single_slice_agent(
    *,
    atlas_name: str,
    plane: Plane,
    species: str,
    pos_lo: float,
    pos_hi: float,
    model: str | object = "gemini-3-flash-preview",
    temperature: float = 1.0,
    media_resolution: str = "MEDIA_RESOLUTION_MEDIUM",
    thinking_config: object | None = None,
) -> LlmAgent:
    """Construct the single-slice LlmAgent with all four tools wired."""
    config_kwargs: dict = {
        "temperature": temperature,
        "max_output_tokens": 4000,
    }
    if thinking_config is not None:
        config_kwargs["thinking_config"] = thinking_config
    # media_resolution set via a dict-style accessor to avoid stale enum mismatches.
    config_kwargs["media_resolution"] = media_resolution

    return LlmAgent(
        model=model,
        name="single_slice_position_estimator",
        instruction=build_single_slice_prompt(
            atlas_name=atlas_name, plane=plane,
            pos_lo=pos_lo, pos_hi=pos_hi, species=species,
        ),
        tools=[fetch_atlas, zoom, side_by_side, submit_estimate],
        generate_content_config=types.GenerateContentConfig(**config_kwargs),
        before_tool_callback=gate_submit_tool,
    )
```

- [ ] **Step 5: Tests pass**

Run: `python -m pytest tests/test_harness_runner.py::test_build_single_slice_agent_registers_four_tools -v`
Expected: PASS.

- [ ] **Step 6: Commit**

```bash
git add langslice/harness/estimation/tools.py \
        langslice/harness/estimation/single_slice.py \
        tests/test_harness_runner.py
git commit -m "feat(harness): build_single_slice_agent with four tools"
```

### Task 3.6: Implement the runner wrapper

**Files:**
- Create: `langslice/harness/estimation/runner.py`

- [ ] **Step 1: Add failing test with a stub LlmAgent that scripts one fetch + one submit**

Append to `tests/test_harness_runner.py`:
```python
import pytest

from langslice.harness.estimation.runner import run_single_slice_session


@pytest.mark.asyncio
async def test_run_single_slice_session_happy_path(monkeypatch):
    """Verify runner completes when the model immediately submits."""
    # Patch the agent-construction path to use a fake LlmAgent that returns
    # a canned submit_estimate call, then escalates. Details of the fake live
    # in conftest.py (Task 3.7).
    from tests.fakes import install_fake_adk_model_scripted_submit
    install_fake_adk_model_scripted_submit(monkeypatch)

    from PIL import Image
    img = Image.new("RGB", (456, 320), 128)

    result = await run_single_slice_session(
        image=img, atlas_name="allen_mouse_25um", plane="coronal",
        model="gemini-3-flash-preview", max_iterations=5,
    )
    assert result.position_mm is not None
```

- [ ] **Step 2: Create the fake-ADK helper module**

Create `tests/fakes.py`:
```python
"""Test doubles for ADK LlmAgent model invocations."""

from __future__ import annotations

from unittest.mock import MagicMock


def install_fake_adk_model_scripted_submit(monkeypatch) -> None:
    """Monkey-patch google.adk so LlmAgent instantiation returns an agent
    whose model immediately calls submit_estimate(position_mm=5.0, reasoning='x').
    The exact mechanism depends on ADK's internals — see ADK docs /
    source for the correct seam. As of this plan, the recommended approach
    is to patch LlmAgent's `_call_llm` or the LiteLlm/Gemini backend entry
    point to return a scripted LlmResponse with a function_call Part.

    This function is a TODO placeholder. On first execution, the agent
    running this plan must:
      1. Locate ADK's model-invocation seam (see src/google/adk/agents/llm_agent.py).
      2. Replace this placeholder with the actual monkeypatch.
    """
    raise NotImplementedError(
        "Implement fake ADK model seam — see docstring. See ADK issue #xxxx "
        "for the canonical test-double pattern if one ships before we need it."
    )
```

- [ ] **Step 3: Watch the fake-dependent test skip or xfail**

Run: `python -m pytest tests/test_harness_runner.py::test_run_single_slice_session_happy_path -v`
Expected: ERROR / FAIL — the fake raises NotImplementedError. **This is intentional**; the fake is a known-TODO that the implementer will fill in during Task 3.7 after inspecting ADK's source.

- [ ] **Step 4: Implement `runner.py`**

```python
"""Runner wrapper: drives nudge-on-no-tool-call, retry-with-fresh-session, and max-iteration cap."""

from __future__ import annotations

import asyncio
import logging
from typing import Any

from google.adk.runners import InMemoryRunner
from google.adk.sessions import Session
from google.genai import types
from PIL import Image

from langslice.atlas.core import load_atlas, get_position_range_mm, get_in_plane_long_edge
from langslice.atlas.space import Plane, atlas_space_context
from langslice.harness.estimation._types import PositionResult, MultiSliceResult
from langslice.harness.estimation.session import (
    ARTIFACT_TARGET,
    build_initial_state,
    target_key,
)
from langslice.harness.estimation.single_slice import build_single_slice_agent
from langslice.image_prep import normalize_image, prepare_image_for_vlm

logger = logging.getLogger(__name__)

_DEFAULT_MAX_ITERATIONS_SINGLE = 20
_DEFAULT_MAX_RETRIES = 2
_NUDGE_BROAD = (
    "Please continue. Call `fetch_atlas` with widely spaced positions "
    "(e.g., [2, 4, 6, 8, 10]) to find the correct neighborhood."
)
_NUDGE_NARROW = (
    "Please narrow down. Call `fetch_atlas` with tightly spaced positions "
    "around your best candidate (e.g., [4.0, 4.2, 4.4, 4.6, 4.8])."
)
_NUDGE_VERIFY = (
    "Please continue. Verify your candidate by checking nearby positions "
    "with `fetch_atlas`, or call `submit_estimate` if confident."
)


def _pick_nudge(state: dict[str, Any]) -> str:
    if not state.get("saw_broad_sweep"):
        return _NUDGE_BROAD
    if not state.get("saw_narrow_sweep"):
        return _NUDGE_NARROW
    return _NUDGE_VERIFY


async def run_single_slice_session(
    *,
    image: Image.Image,
    atlas_name: str,
    plane: Plane = "coronal",
    model: str | object = "gemini-3-flash-preview",
    species: str | None = None,
    max_iterations: int = _DEFAULT_MAX_ITERATIONS_SINGLE,
    max_retries: int = _DEFAULT_MAX_RETRIES,
    temperature: float = 1.0,
) -> PositionResult:
    """Drive a single-slice position-estimation session to completion."""
    atlas = load_atlas(atlas_name)
    pos_lo, pos_hi = get_position_range_mm(atlas, plane=plane)
    atlas_long_edge = get_in_plane_long_edge(atlas, plane=plane)

    # Prepare target image once.
    normalized = normalize_image(image)
    prepped = prepare_image_for_vlm(normalized, max_long_edge=atlas_long_edge).image

    # Build agent.
    species_val = species or atlas.metadata.get("species", "mouse")
    agent = build_single_slice_agent(
        atlas_name=atlas_name, plane=plane, species=species_val,
        pos_lo=pos_lo, pos_hi=pos_hi, model=model, temperature=temperature,
    )

    runner = InMemoryRunner(agent=agent)

    initial_state = build_initial_state(
        atlas_name=atlas_name, plane=plane,
        pos_lo=pos_lo, pos_hi=pos_hi,
        n_slices=1, interval_mm=0.0, thickness_um=50,
        max_iterations=max_iterations,
    )

    for attempt in range(max_retries):
        session = Session(
            id=f"single_slice_attempt_{attempt}",
            app_name="langslice",
            user_id="langslice-user",
            state=dict(initial_state),
        )
        # Put the target image into the session as an artifact.
        import io
        buf = io.BytesIO()
        prepped.convert("RGB").save(buf, format="JPEG", quality=85)
        target_part = types.Part.from_bytes(mime_type="image/jpeg", data=buf.getvalue())
        # ADK session has its own artifact store; tools expect it to be reachable via load_artifact(name).
        # Below, we inject the target via a user message that includes the image AND saves to artifact store.
        # The exact API for session-level artifact seeding will need to be verified against ADK; placeholder:
        await session.artifacts.set(ARTIFACT_TARGET, target_part)

        new_message = types.Content(
            role="user",
            parts=[
                types.Part.from_text(text=f"Target slice (artifact key: '{ARTIFACT_TARGET}'):"),
                target_part,
                types.Part.from_text(text="Determine its position in the atlas."),
            ],
        )

        tool_call_count = 0
        async for event in runner.run_async(
            user_id=session.user_id, session_id=session.id, new_message=new_message,
        ):
            # Count function-call events for the max-iteration cap.
            if getattr(event, "function_call", None) is not None:
                tool_call_count += 1
                if tool_call_count > max_iterations:
                    logger.warning("Hit max_iterations=%d; forcing end.", max_iterations)
                    break
            if session.state.get("result") is not None:
                break

        if session.state.get("result") is not None:
            result = session.state["result"]
            return PositionResult(
                position_mm=float(result["position_mm"]),
                reasoning=str(result["reasoning"]),
            )

        # No tool call exit OR model exhausted its turn without submitting.
        # Inject a nudge message and retry with a fresh session.
        logger.info("Attempt %d did not submit; nudging and retrying.", attempt + 1)
        nudge = _pick_nudge(session.state)
        # NOTE: the current simple implementation creates a fresh session per retry.
        # For a persistent-history nudge, we'd continue the same session here; leaving
        # as fresh-session retry per the design spec.
        initial_state["submit_attempts"] = 0  # reset relaxation for the new attempt

    # Fell through all retries — fall back to atlas midpoint.
    mid = (pos_lo + pos_hi) / 2.0
    logger.warning("All %d retries exhausted; falling back to %.2f mm midpoint.", max_retries, mid)
    return PositionResult(
        position_mm=mid,
        reasoning="Agent did not submit within iteration+retry budget; fell back to atlas midpoint.",
    )
```

**Implementer note:** ADK's exact API for session artifact seeding (the `session.artifacts.set` call above) needs verification against the installed `google-adk` version. If the API differs, replace with the current idiom (`BaseArtifactService.save_artifact(session_id, name, part)` or similar). The runner falls back to sending the image inline in `new_message` even if the artifact-set path fails, so `fetch_atlas` does not require it to succeed — but `zoom`/`side_by_side` targeting `"target"` will fail without it.

- [ ] **Step 5: Fix the fake in `tests/fakes.py`** (implementer task during execution)

This is where the implementer must actually look at `src/google/adk/agents/llm_agent.py` and patch the model-invocation seam to return a canned response. Target: `LlmAgent._call_llm` or the underlying `BaseLlm.generate_content_async` — whichever is the cleanest seam in the version of ADK installed.

Acceptance criterion for Task 3.6: `test_run_single_slice_session_happy_path` passes.

- [ ] **Step 6: Tests pass**

Run: `python -m pytest tests/test_harness_runner.py -v`
Expected: all PASS.

- [ ] **Step 7: Commit**

```bash
git add langslice/harness/estimation/runner.py tests/test_harness_runner.py tests/fakes.py
git commit -m "feat(harness): single-slice runner with nudge/retry/cap"
```

### Task 3.7: Wire the back-compat shim for `estimate_position`

**Files:**
- Modify: `langslice/harness/estimation/__init__.py`
- Modify: `langslice/estimation/__init__.py`

- [ ] **Step 1: Export public API from the harness package**

Overwrite `langslice/harness/estimation/__init__.py`:
```python
"""Public API of the position-estimation harness."""

from langslice.harness.estimation._types import (  # noqa: F401
    APResult,
    MultiSliceResult,
    PositionResult,
)

__all__ = ["APResult", "MultiSliceResult", "PositionResult", "estimate_position"]


def estimate_position(
    image,
    atlas_name: str,
    *,
    plane: str = "coronal",
    model: str | object = "gemini-3-flash-preview",
    max_iterations: int = 20,
    **_ignored,
) -> PositionResult:
    """Synchronous wrapper over the async runner. Sync API for CLI / eval consumers."""
    import asyncio
    from langslice.harness.estimation.runner import run_single_slice_session
    return asyncio.run(
        run_single_slice_session(
            image=image, atlas_name=atlas_name, plane=plane,
            model=model, max_iterations=max_iterations,
        )
    )
```

- [ ] **Step 2: Update the old-path shim**

Overwrite `langslice/estimation/__init__.py`:
```python
"""Deprecated re-export shim. Import from langslice.harness.estimation instead."""
from langslice.harness.estimation import (  # noqa: F401
    APResult,
    MultiSliceResult,
    PositionResult,
    estimate_position,
)

# estimate_group is added in Phase 4.
```

- [ ] **Step 3: Verify existing imports still resolve**

Run: `python -c "from langslice.estimation import APResult, estimate_position; print(APResult)"`
Expected: prints the dataclass.

- [ ] **Step 4: Run tests**

Run: `python -m pytest -q`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add langslice/harness/estimation/__init__.py langslice/estimation/__init__.py
git commit -m "feat(harness): expose estimate_position via compat shim"
```

### Task 3.8: Single-slice smoke API run

- [ ] **Step 1: Run Flash mid-brain smoke test**

```bash
python eval/eval_group.py \
  --images references/TestImages/M01 \
  --ground-truth references/TestImages/M01/ground_truth.json \
  --model gemini-3-flash-preview \
  --group-size 1 \
  --json > eval_outputs/post_step3_single_M01.json
```

- [ ] **Step 2: Inspect the JSON**

Gate: JSON parses, `summary.n_fallbacks == 0`, all per-slice `error_mm < 2.0` on mid-brain (AP 4.0–7.0) entries.

- [ ] **Step 3: Commit the eval artifact**

```bash
git add eval_outputs/post_step3_single_M01.json
git commit -m "test(eval): single-slice ADK port smoke check passes"
```

---

## Phase 4 — Multi-slice group ADK port

### Task 4.1: Build the group agent

**Files:**
- Create: `langslice/harness/estimation/group.py`

- [ ] **Step 1: Write failing test**

Append to `tests/test_harness_runner.py`:
```python
from langslice.harness.estimation.group import build_group_agent


def test_build_group_agent_registers_four_tools():
    agent = build_group_agent(
        atlas_name="allen_mouse_25um", plane="coronal", species="mouse",
        pos_lo=0.0, pos_hi=13.2, n_slices=4, interval_mm=0.200, thickness_um=50,
        model="gemini-3-flash-preview",
    )
    tool_names = {getattr(t, "__name__", None) or getattr(t, "name", None) for t in agent.tools}
    assert "fetch_atlas" in tool_names
    assert "zoom" in tool_names
    assert "side_by_side" in tool_names
    assert "submit_group_estimate" in tool_names
```

- [ ] **Step 2: Watch fail, implement `group.py`**

```python
"""Builder for the multi-slice-group position-estimation LlmAgent."""

from __future__ import annotations

from google.adk.agents import LlmAgent
from google.genai import types

from langslice.atlas.space import Plane
from langslice.harness.estimation.prompts import build_group_prompt
from langslice.harness.estimation.tools import (
    fetch_atlas,
    side_by_side,
    submit_group_estimate,
    zoom,
)
from langslice.harness.estimation.validators import gate_submit_tool


def build_group_agent(
    *,
    atlas_name: str,
    plane: Plane,
    species: str,
    pos_lo: float,
    pos_hi: float,
    n_slices: int,
    interval_mm: float,
    thickness_um: int,
    model: str | object = "gemini-3-flash-preview",
    temperature: float = 1.0,
    media_resolution: str = "MEDIA_RESOLUTION_MEDIUM",
    thinking_config: object | None = None,
) -> LlmAgent:
    """Construct the multi-slice-group LlmAgent with all four tools wired."""
    config_kwargs: dict = {
        "temperature": temperature,
        "max_output_tokens": 8000,
        "media_resolution": media_resolution,
    }
    if thinking_config is not None:
        config_kwargs["thinking_config"] = thinking_config

    return LlmAgent(
        model=model,
        name="group_position_estimator",
        instruction=build_group_prompt(
            atlas_name=atlas_name, plane=plane,
            pos_lo=pos_lo, pos_hi=pos_hi, species=species,
            n_slices=n_slices, interval_mm=interval_mm, thickness_um=thickness_um,
        ),
        tools=[fetch_atlas, zoom, side_by_side, submit_group_estimate],
        generate_content_config=types.GenerateContentConfig(**config_kwargs),
        before_tool_callback=gate_submit_tool,
    )
```

- [ ] **Step 3: Test passes**

Run: `python -m pytest tests/test_harness_runner.py::test_build_group_agent_registers_four_tools -v`

- [ ] **Step 4: Commit**

```bash
git add langslice/harness/estimation/group.py tests/test_harness_runner.py
git commit -m "feat(harness): build_group_agent with four tools"
```

### Task 4.2: Add `run_group_session` to the runner

**Files:**
- Modify: `langslice/harness/estimation/runner.py`

- [ ] **Step 1: Append `run_group_session` to `runner.py`**

```python
from langslice.harness.estimation.group import build_group_agent


async def run_group_session(
    *,
    images: list[Image.Image],
    atlas_name: str,
    interval_um: int,
    thickness_um: int = 50,
    plane: Plane = "coronal",
    model: str | object = "gemini-3-flash-preview",
    species: str | None = None,
    max_iterations: int = 25,
    max_retries: int = _DEFAULT_MAX_RETRIES,
    temperature: float = 1.0,
) -> MultiSliceResult:
    """Drive a multi-slice-group position-estimation session to completion."""
    n_slices = len(images)
    if not 2 <= n_slices <= 8:
        raise ValueError(f"Expected 2-8 slices, got {n_slices}")
    interval_mm = interval_um / 1000.0

    atlas = load_atlas(atlas_name)
    pos_lo, pos_hi = get_position_range_mm(atlas, plane=plane)
    atlas_long_edge = get_in_plane_long_edge(atlas, plane=plane)

    prepped_images: list[Image.Image] = []
    for img in images:
        normalized = normalize_image(img)
        prep = prepare_image_for_vlm(normalized, max_long_edge=atlas_long_edge).image
        prepped_images.append(prep)

    species_val = species or atlas.metadata.get("species", "mouse")
    agent = build_group_agent(
        atlas_name=atlas_name, plane=plane, species=species_val,
        pos_lo=pos_lo, pos_hi=pos_hi,
        n_slices=n_slices, interval_mm=interval_mm, thickness_um=thickness_um,
        model=model, temperature=temperature,
    )
    runner = InMemoryRunner(agent=agent)

    initial_state = build_initial_state(
        atlas_name=atlas_name, plane=plane,
        pos_lo=pos_lo, pos_hi=pos_hi,
        n_slices=n_slices, interval_mm=interval_mm, thickness_um=thickness_um,
        max_iterations=max_iterations,
    )

    for attempt in range(max_retries):
        session = Session(
            id=f"group_attempt_{attempt}",
            app_name="langslice",
            user_id="langslice-user",
            state=dict(initial_state),
        )
        import io as _io
        parts: list[types.Part] = [
            types.Part.from_text(
                text=(
                    f"Here are {n_slices} consecutive brain slices along the "
                    f"{plane} plane. Interval: {interval_um} µm "
                    f"({interval_mm:.3f} mm). Determine the position of each slice."
                ),
            ),
        ]
        for i, img in enumerate(prepped_images):
            buf = _io.BytesIO()
            img.convert("RGB").save(buf, format="JPEG", quality=85)
            blob = buf.getvalue()
            part = types.Part.from_bytes(mime_type="image/jpeg", data=blob)
            await session.artifacts.set(target_key(i + 1), part)
            parts.append(types.Part.from_text(text=f"Slice {i + 1}:"))
            parts.append(part)

        new_message = types.Content(role="user", parts=parts)

        tool_call_count = 0
        async for event in runner.run_async(
            user_id=session.user_id, session_id=session.id, new_message=new_message,
        ):
            if getattr(event, "function_call", None) is not None:
                tool_call_count += 1
                if tool_call_count > max_iterations:
                    logger.warning("Hit max_iterations=%d; forcing end.", max_iterations)
                    break
            if session.state.get("result") is not None:
                break

        if session.state.get("result") is not None:
            result = session.state["result"]
            positions = [float(p) for p in result["positions_mm"]]
            reasoning = str(result["reasoning"])
            return MultiSliceResult(
                positions=[PositionResult(position_mm=p, reasoning=reasoning) for p in positions],
                group_reasoning=reasoning,
            )

        initial_state["submit_attempts"] = 0

    # Fallback: center the group around the atlas midpoint.
    mid = (pos_lo + pos_hi) / 2.0
    span = (n_slices - 1) * interval_mm
    start = mid - span / 2
    positions = [max(pos_lo, min(pos_hi, start + i * interval_mm)) for i in range(n_slices)]
    fallback_reasoning = f"Fallback: centered midpoint after {max_retries} attempts."
    return MultiSliceResult(
        positions=[PositionResult(position_mm=p, reasoning=fallback_reasoning) for p in positions],
        group_reasoning=fallback_reasoning,
    )
```

- [ ] **Step 2: Expose via the public shim**

Update `langslice/harness/estimation/__init__.py` (append to existing file):
```python
def estimate_group(
    images,
    atlas_name: str,
    interval_um: int,
    *,
    thickness_um: int = 50,
    plane: str = "coronal",
    model: str | object = "gemini-3-flash-preview",
    max_iterations: int = 25,
    **_ignored,
) -> MultiSliceResult:
    import asyncio
    from langslice.harness.estimation.runner import run_group_session
    return asyncio.run(
        run_group_session(
            images=images, atlas_name=atlas_name, interval_um=interval_um,
            thickness_um=thickness_um, plane=plane, model=model,
            max_iterations=max_iterations,
        )
    )


__all__.append("estimate_group")
```

Update `langslice/estimation/__init__.py`:
```python
from langslice.harness.estimation import estimate_group  # noqa: F401
```

- [ ] **Step 3: Run existing group tests**

Run: `python -m pytest -q`
Expected: PASS.

- [ ] **Step 4: Commit**

```bash
git add langslice/harness/estimation/runner.py langslice/harness/estimation/__init__.py langslice/estimation/__init__.py
git commit -m "feat(harness): run_group_session + estimate_group shim"
```

### Task 4.3: Update `eval_group.py` to use the harness directly

**Files:**
- Modify: `eval/eval_group.py`

- [ ] **Step 1: Update the import**

In `eval/eval_group.py`, replace:
```python
if args.provider == "openai":
    from langslice.estimation.openai.ap_multi_slice import estimate_group
else:
    import langslice.vlm_config as vlm_config
    from langslice.estimation import estimate_group
    ...
```
with:
```python
from langslice.harness.estimation import estimate_group
```

The `--provider openai` code path can keep its old import OR be dropped as part of the "Delete old langslice/estimation" cleanup. For now, **leave the OpenAI branch in place** (it uses the legacy non-ADK implementation); only the `google` branch goes through the new harness.

Make the call:
```python
result = estimate_group(
    images=images, atlas_name=atlas_name, interval_um=interval_um,
    thickness_um=_THICKNESS_UM, model=args.model or "gemini-3-flash-preview",
    max_iterations=args.max_iterations,
)
```

- [ ] **Step 2: Commit**

```bash
git add eval/eval_group.py
git commit -m "refactor(eval): route google-provider eval through harness"
```

### Task 4.4: Multi-slice smoke API run

- [ ] **Step 1: Run Flash mid-brain smoke test, default group size**

```bash
python eval/eval_group.py \
  --images references/TestImages/M01 \
  --ground-truth references/TestImages/M01/ground_truth.json \
  --model gemini-3-flash-preview \
  --json > eval_outputs/post_step4_group_M01.json
```

- [ ] **Step 2: Inspect JSON**

Gate: `summary.n_failures == 0`, `summary.n_fallbacks == 0`, all mid-brain `error_mm < 2.0`.

- [ ] **Step 3: Commit**

```bash
git add eval_outputs/post_step4_group_M01.json
git commit -m "test(eval): multi-slice group ADK port smoke check passes"
```

---

## Phase 5 — `zoom` tool

### Task 5.1: Implement `zoom` with full image processing

**Files:**
- Modify: `langslice/harness/estimation/tools.py`
- Modify: `tests/test_harness_tools.py`

- [ ] **Step 1: Write failing test**

Append to `tests/test_harness_tools.py`:
```python
def test_zoom_rejects_bad_source():
    state = build_initial_state(
        atlas_name="allen_mouse_25um", plane="coronal",
        pos_lo=0.0, pos_hi=13.2, n_slices=1,
        interval_mm=0.0, thickness_um=50, max_iterations=20,
    )
    ctx = _fake_tool_context(state)
    ctx.load_artifact = MagicMock(return_value=None)
    out = zoom(source="atlas:99.99", bbox=[100, 200, 300, 400], tool_context=ctx)
    assert out["status"] == "error"
    assert out["error"] == "BAD_SOURCE"


def test_zoom_rejects_bad_bbox():
    state = build_initial_state(
        atlas_name="allen_mouse_25um", plane="coronal",
        pos_lo=0.0, pos_hi=13.2, n_slices=1,
        interval_mm=0.0, thickness_um=50, max_iterations=20,
    )
    ctx = _fake_tool_context(state)
    # Non-null artifact so source passes, but bbox is degenerate
    ctx.load_artifact = MagicMock(return_value=_image_to_part(Image.new("RGB", (100, 100))))
    out = zoom(source="target", bbox=[500, 500, 400, 400], tool_context=ctx)
    assert out["status"] == "error"
    assert out["error"] == "BAD_BBOX"


def test_zoom_returns_cropped_part_and_saves_artifact():
    state = build_initial_state(
        atlas_name="allen_mouse_25um", plane="coronal",
        pos_lo=0.0, pos_hi=13.2, n_slices=1,
        interval_mm=0.0, thickness_um=50, max_iterations=20,
    )
    src_img = Image.new("RGB", (1000, 1000), (128, 64, 32))
    ctx = _fake_tool_context(state)
    ctx.load_artifact = MagicMock(return_value=_image_to_part(src_img))
    out = zoom(source="target", bbox=[100, 100, 500, 500], tool_context=ctx)
    assert out["status"] == "ok"
    assert "image" in out
    assert ctx.save_artifact.called
```

- [ ] **Step 2: Add `zoom` implementation in `tools.py`** (replacing the Phase-3 stub)

```python
def zoom(source: str, bbox: list[int], tool_context: Any) -> dict[str, Any]:
    """Return a zoomed crop of a previously-fetched image.

    Args:
        source: 'target', 'target:N', or 'atlas:<mm>'.
        bbox: [y1, x1, y2, x2] integers 0–1000 (Gemini/Gemma native format).
    """
    # Validate bbox
    if (
        len(bbox) != 4
        or any(not (0 <= v <= 1000) for v in bbox)
        or bbox[0] >= bbox[2]
        or bbox[1] >= bbox[3]
    ):
        return {"status": "error", "error": "BAD_BBOX"}

    # Load source image
    part = tool_context.load_artifact(source)
    if part is None:
        return {"status": "error", "error": "BAD_SOURCE"}

    src_img = _part_to_pil(part)
    w, h = src_img.size
    y1, x1, y2, x2 = bbox
    px = (
        int(round(x1 / 1000 * w)),
        int(round(y1 / 1000 * h)),
        int(round(x2 / 1000 * w)),
        int(round(y2 / 1000 * h)),
    )
    if px[2] <= px[0] or px[3] <= px[1]:
        return {"status": "error", "error": "EMPTY_CROP"}

    crop = src_img.crop(px)

    # Upscale to match in-plane long-edge for the session's atlas.
    atlas = load_atlas(tool_context.state["atlas"])
    target_long = get_in_plane_long_edge(atlas, plane=tool_context.state["plane"])
    cw, ch = crop.size
    scale = target_long / max(cw, ch)
    new_size = (int(cw * scale), int(ch * scale))
    resized = crop.resize(new_size, Image.Resampling.LANCZOS)

    # Save + return
    part_out = _image_to_part(resized)
    key = f"{ARTIFACT_ZOOM_PREFIX}{source}:{_short_hash(bbox)}"
    tool_context.save_artifact(key, part_out)
    return {
        "status": "ok",
        "source": source,
        "bbox": list(bbox),
        "artifact_key": key,
        "image": part_out,
    }


def _part_to_pil(part: Any) -> Image.Image:
    """Extract PIL from a genai types.Part (best effort)."""
    data = getattr(part, "inline_data", None) or getattr(part, "data", None)
    if hasattr(part, "inline_data") and getattr(part.inline_data, "data", None):
        data = part.inline_data.data
    if data is None:
        raise ValueError("Part has no image data")
    import io
    return Image.open(io.BytesIO(data))
```

- [ ] **Step 3: Update prompts** to mention `zoom` (already mentioned in Task 3.2). Verify by re-running `tests/test_harness_prompts.py` — prompts should already reference zoom.

- [ ] **Step 4: Tests pass**

Run: `python -m pytest tests/test_harness_tools.py -v`
Expected: all PASS.

- [ ] **Step 5: Commit**

```bash
git add langslice/harness/estimation/tools.py tests/test_harness_tools.py
git commit -m "feat(harness): zoom tool with bbox crop and artifact save"
```

### Task 5.2: Smoke API run with zoom

- [ ] **Step 1: Run**

```bash
python eval/eval_group.py \
  --images references/TestImages/M01 \
  --ground-truth references/TestImages/M01/ground_truth.json \
  --model gemini-3-flash-preview \
  --json > eval_outputs/post_step5_zoom_M01.json
```

- [ ] **Step 2: Inspect artifacts**

Enable `LANGSLICE_VLM_DEBUG_DIR=debug_runs/post_step5` on a smaller targeted run to record traces. Verify at least one `zoom` call appears in the trace JSON.

- [ ] **Step 3: Commit**

```bash
git add eval_outputs/post_step5_zoom_M01.json
git commit -m "test(eval): zoom tool integrated, smoke check passes"
```

---

## Phase 6 — `side_by_side` tool

### Task 6.1: Implement `side_by_side`

**Files:**
- Modify: `langslice/harness/estimation/tools.py`
- Modify: `tests/test_harness_tools.py`

- [ ] **Step 1: Write failing test**

Append to `tests/test_harness_tools.py`:
```python
def test_side_by_side_happy_path():
    state = build_initial_state(
        atlas_name="allen_mouse_25um", plane="coronal",
        pos_lo=0.0, pos_hi=13.2, n_slices=1,
        interval_mm=0.0, thickness_um=50, max_iterations=20,
    )
    ctx = _fake_tool_context(state)
    left = _image_to_part(Image.new("RGB", (456, 320), (128, 128, 128)))
    right = _image_to_part(Image.new("RGB", (228, 160), (64, 64, 64)))
    ctx.load_artifact = MagicMock(side_effect=lambda k: left if k == "target" else right)

    out = side_by_side(left="target", right="atlas:5.00", tool_context=ctx)
    assert out["status"] == "ok"
    assert "image" in out
    assert ctx.save_artifact.called


def test_side_by_side_rejects_missing_source():
    state = build_initial_state(
        atlas_name="allen_mouse_25um", plane="coronal",
        pos_lo=0.0, pos_hi=13.2, n_slices=1,
        interval_mm=0.0, thickness_um=50, max_iterations=20,
    )
    ctx = _fake_tool_context(state)
    ctx.load_artifact = MagicMock(return_value=None)
    out = side_by_side(left="target", right="atlas:99.99", tool_context=ctx)
    assert out["status"] == "error" and out["error"] == "BAD_SOURCE"
```

- [ ] **Step 2: Replace the Phase-3 stub in `tools.py`**

```python
def side_by_side(left: str, right: str, tool_context: Any) -> dict[str, Any]:
    """Build an aspect-ratio-matched horizontal composite of two images.

    Args:
        left, right: one of 'target', 'target:N', 'atlas:<mm>', 'zoom:...'.

    Both sources rescaled to a common height (aspect-ratio preserved), placed
    side-by-side with a thin gap, labeled with the source string.
    """
    from PIL import ImageDraw, ImageFont

    l_part = tool_context.load_artifact(left)
    r_part = tool_context.load_artifact(right)
    if l_part is None or r_part is None:
        return {"status": "error", "error": "BAD_SOURCE"}

    l_img = _part_to_pil(l_part).convert("RGB")
    r_img = _part_to_pil(r_part).convert("RGB")

    common_h = min(l_img.height, r_img.height)
    # Scale both to common_h, keep each panel's aspect ratio.
    l_scaled = l_img.resize(
        (int(l_img.width * common_h / l_img.height), common_h),
        Image.Resampling.LANCZOS,
    )
    r_scaled = r_img.resize(
        (int(r_img.width * common_h / r_img.height), common_h),
        Image.Resampling.LANCZOS,
    )

    gap = 12
    label_h = 40
    total_w = l_scaled.width + gap + r_scaled.width
    total_h = common_h + label_h
    canvas = Image.new("RGB", (total_w, total_h), (0, 0, 0))
    canvas.paste(l_scaled, (0, 0))
    canvas.paste(r_scaled, (l_scaled.width + gap, 0))

    draw = ImageDraw.Draw(canvas)
    try:
        font = ImageFont.truetype("arial.ttf", 24)
    except OSError:
        font = ImageFont.load_default()
    draw.text((8, common_h + 8), left, fill=(255, 255, 0), font=font)
    draw.text((l_scaled.width + gap + 8, common_h + 8), right, fill=(255, 255, 0), font=font)

    part_out = _image_to_part(canvas)
    key = f"{ARTIFACT_SIDE_BY_SIDE_PREFIX}{left}:{right}"
    tool_context.save_artifact(key, part_out)
    return {"status": "ok", "left": left, "right": right, "image": part_out, "artifact_key": key}
```

- [ ] **Step 3: Tests pass**

Run: `python -m pytest tests/test_harness_tools.py -v`
Expected: all PASS.

- [ ] **Step 4: Commit**

```bash
git add langslice/harness/estimation/tools.py tests/test_harness_tools.py
git commit -m "feat(harness): side_by_side tool with aspect-ratio composite"
```

### Task 6.2: Smoke API run with side_by_side

- [ ] **Step 1: Run**

```bash
python eval/eval_group.py \
  --images references/TestImages/M01 \
  --ground-truth references/TestImages/M01/ground_truth.json \
  --model gemini-3-flash-preview \
  --json > eval_outputs/post_step6_sbs_M01.json
```

- [ ] **Step 2: Verify traces show at least one `side_by_side` invocation**

(Again, enable `LANGSLICE_VLM_DEBUG_DIR` for a targeted run.)

- [ ] **Step 3: Commit**

```bash
git add eval_outputs/post_step6_sbs_M01.json
git commit -m "test(eval): side_by_side integrated, smoke check passes"
```

---

## Phase 7 — Final cleanup and acceptance

### Task 7.1: Delete dead legacy files or reduce to import shims

**Files:**
- Delete/shim: `langslice/estimation/google/ap_single_slice.py`, `ap_multi_slice.py`, `ap_tool_use.py`, `common.py`, `tool_definitions.py`, `batch_eval.py`
- Delete/shim: `langslice/estimation/openai/*`
- Delete: `langslice/estimation/_shared_common.py`, `_tool_logic.py`

- [ ] **Step 1: Grep for remaining imports from these modules**

```bash
git grep -nE "from langslice\.estimation\.(google|openai|_shared_common|_tool_logic)"
```
Record the list.

- [ ] **Step 2: Update each remaining caller**

For each match: replace with the harness equivalent (`from langslice.harness.estimation import ...`). If the OpenAI-provider path is still wanted (for Ollama/Gemma local), leave `langslice/estimation/openai/*` in place for now — those run the legacy (non-ADK) OpenAI Responses API path.

If a file is now unreferenced (grep count 0), delete it.

- [ ] **Step 3: Run pytest**

Run: `python -m pytest -q`
Expected: PASS.

- [ ] **Step 4: Run linting**

Run: `python -m ruff check .` and `python -m basedpyright`
Expected: clean (or equivalent to pre-refactor baseline).

- [ ] **Step 5: Commit**

```bash
git add -A
git commit -m "refactor(estimation): delete legacy tool-use modules replaced by harness/"
```

### Task 7.2: Final acceptance smoke run

- [ ] **Step 1: Full mid-brain smoke run, all tools live**

```bash
python eval/eval_group.py \
  --images references/TestImages/M01 \
  --ground-truth references/TestImages/M01/ground_truth.json \
  --model gemini-3-flash-preview \
  --json > eval_outputs/final_acceptance_M01.json
```

- [ ] **Step 2: Gate**

- `summary.n_failures == 0`
- `summary.n_fallbacks == 0`
- All mid-brain `error_mm < 2.0`
- Trace files (set `LANGSLICE_VLM_DEBUG_DIR` on a targeted re-run) contain at least one invocation each of `fetch_atlas`, `zoom`, `side_by_side`, `submit_group_estimate`.

- [ ] **Step 3: Commit**

```bash
git add eval_outputs/final_acceptance_M01.json
git commit -m "test(eval): final acceptance smoke check passes"
```

### Task 7.3: Update CLAUDE.md and REPO_MAP.md

**Files:**
- Modify: `CLAUDE.md`
- Modify: `REPO_MAP.md`

- [ ] **Step 1: Update the Architecture section of CLAUDE.md**

Replace the `langslice/estimation/` and related entries with a single `langslice/harness/estimation/` entry describing the new layout. Note:
- Tool-use is now on Google ADK
- Four tools: `fetch_atlas`, `zoom`, `side_by_side`, `submit_*`
- Image-gen (`image_gen.py`) is still raw `google-genai`, used only by `whole_brain/`
- Session state + artifacts documented in `harness/estimation/session.py`
- Position estimation is plane-generalized — AP/ML/DV per plane

Remove or flag as stale the old guidance about `ap_multi_slice.py` being the default. The new default is the ADK harness through `estimate_group`.

- [ ] **Step 2: Update REPO_MAP.md** similarly.

- [ ] **Step 3: Commit**

```bash
git add CLAUDE.md REPO_MAP.md
git commit -m "docs: update architecture for ADK harness refactor"
```

---

## Self-review checklist for the implementer

Before declaring the plan done:

- [ ] All 5 smoke eval JSONs exist in `eval_outputs/` and satisfy their gates.
- [ ] `git grep -n "require_coronal_layout"` shows only the deprecated-stub definition.
- [ ] `git grep -n "get_coronal_long_edge"` shows only the deprecated alias + callers that explicitly pass `plane="coronal"`.
- [ ] `git grep -nE "from langslice\.estimation\.(google|openai)"` is empty (except OpenAI-provider path if kept).
- [ ] All tools appear in trace files when the agent runs end-to-end.
- [ ] `python -m pytest`, `python -m ruff check .`, `python -m basedpyright` all clean.
- [ ] CLAUDE.md and REPO_MAP.md reflect the new architecture.

---

## Known implementer TODOs embedded in tasks

These are acceptable TODOs because they require inspecting the installed ADK version, but they MUST be resolved before Phase 7 closes:

1. **Task 3.6 Step 5** — Wire the fake ADK model seam in `tests/fakes.py` (needs inspection of `src/google/adk/agents/llm_agent.py` in the installed version).
2. **Task 3.6 implementer note** — Verify `session.artifacts.set(...)` is the correct ADK API for seeding session artifacts. If the API differs, use the current idiom (likely `BaseArtifactService.save_artifact(...)` or the analogous session-method on whichever ADK release is installed).
3. **Task 4.2** — Same artifact-seeding verification applies to the multi-slice runner.

Each TODO has a concrete verification path (read the installed ADK source) and a fallback (inline the image in `new_message` if artifact-seeding fails — tools targeting `"target"` will then fail, which is a visible failure mode in traces, not a silent one).
