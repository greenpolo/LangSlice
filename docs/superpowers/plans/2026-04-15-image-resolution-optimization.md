# Image Resolution Optimization Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Remove the `atlas_resolution` parameter and instead derive image sizes from the atlas's native coronal slice dimensions, downscaling tissue slices to match.

**Architecture:** Add a `get_coronal_long_edge(atlas)` helper that returns `max(shape[1], shape[2])` from the atlas volume. Use this as the target for both atlas slice extraction (no resize — already at native) and tissue slice downscaling (cap at native long edge). Remove `atlas_resolution` from all estimation function signatures, tool handlers, CLI flags, and eval harness. Fix the `media_resolution` parameter in `ap_multi_slice.py` which is accepted but never passed to `GenerateContentConfig`.

**Tech Stack:** Python, PIL, BrainGlobe atlas API, google-genai SDK

**Important context:** `atlas_resolution` in `export.py` and `tests/test_quicknii_math.py` means **voxel resolution in microns** (e.g., `(25.0, 25.0, 25.0)`), not image pixel resolution. Those are completely unrelated and must NOT be touched.

---

### Task 1: Add `get_coronal_long_edge` helper to atlas/core.py

**Files:**
- Modify: `langslice/atlas/core.py`
- Test: `tests/test_atlas_core.py` (create if needed, or add to existing test file)

- [ ] **Step 1: Write the failing test**

Create `tests/test_coronal_long_edge.py`:

```python
"""Test the get_coronal_long_edge helper."""
from unittest.mock import MagicMock

import numpy as np

from langslice.atlas.core import get_coronal_long_edge


def test_coronal_long_edge_returns_max_of_dv_ml():
    atlas = MagicMock()
    atlas.reference = np.zeros((100, 320, 528), dtype=np.uint8)  # AP=100, DV=320, ML=528
    assert get_coronal_long_edge(atlas) == 528


def test_coronal_long_edge_tall_atlas():
    atlas = MagicMock()
    atlas.reference = np.zeros((200, 600, 400), dtype=np.uint8)  # DV > ML
    assert get_coronal_long_edge(atlas) == 600
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m pytest tests/test_coronal_long_edge.py -v`
Expected: FAIL with `ImportError` (function doesn't exist yet)

- [ ] **Step 3: Implement `get_coronal_long_edge`**

In `langslice/atlas/core.py`, add after the `get_atlas_info` function (around line 540):

```python
def get_coronal_long_edge(atlas: _AtlasLike) -> int:
    """Return the long-edge pixel count of a native coronal slice.

    For a coronal layout (AP/DV/ML on axes 0/1/2), this is
    ``max(shape[1], shape[2])`` — the larger of the DV and ML dimensions.
    """
    _, dv, ml = _shape3d(atlas.reference)
    return max(dv, ml)
```

- [ ] **Step 4: Run test to verify it passes**

Run: `python -m pytest tests/test_coronal_long_edge.py -v`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add langslice/atlas/core.py tests/test_coronal_long_edge.py
git commit -m "feat: add get_coronal_long_edge helper for native atlas resolution"
```

---

### Task 2: Remove `atlas_resolution` from `_shared_common.py` and `_tool_logic.py`

These are the provider-agnostic core files. The atlas fetch function currently accepts `max_long_edge` and upscales atlas slices. Change it to use the atlas's native resolution (no resize at all for atlas slices), and remove the `atlas_resolution` parameter from the tool handler.

**Files:**
- Modify: `langslice/estimation/_shared_common.py:73-98`
- Modify: `langslice/estimation/_tool_logic.py:471-598`

- [ ] **Step 1: Remove `max_long_edge` from `_fetch_atlas_slice_bytes`**

In `langslice/estimation/_shared_common.py`, change `_fetch_atlas_slice_bytes` to remove the resize logic entirely. Atlas slices are already at native resolution:

```python
def _fetch_atlas_slice_bytes(
    atlas: Any,
    position_mm: float,
    *,
    show_borders: bool = False,
) -> bytes:
    """Fetch a single atlas slice, normalize, and return JPEG bytes.

    Atlas slices are sent at their native resolution — no resize.
    """
    if show_borders:
        from langslice.atlas.core import get_composite_slice

        img = get_composite_slice(atlas, position_mm)
    else:
        from langslice.atlas.core import get_reference_slice

        img = get_reference_slice(atlas, position_mm)
    img = normalize_image(img)
    return _image_to_bytes(img)
```

- [ ] **Step 2: Remove `atlas_resolution` from `_handle_fetch_atlas_core`**

In `langslice/estimation/_tool_logic.py`, remove the `atlas_resolution: int` parameter from `_handle_fetch_atlas_core` (line 482). Update the call to `_fetch_atlas_slice_bytes` on line 518 to drop `max_long_edge=atlas_resolution`. Update the call to `_build_atlas_grid` on line 551 to drop `cell_width=atlas_resolution`.

The updated signature:

```python
def _handle_fetch_atlas_core(
    *,
    args: dict[str, object],
    pos_lo: float,
    pos_hi: float,
    atlas: object,
    state: _APLoopState,
    iteration: int,
    run_dir: str | None,
    show_borders: bool,
    send_individually: bool,
    target_image: Image.Image | None,
    stage: str,
    on_trace: Callable[[dict[str, object]], None] | None,
) -> FetchAtlasResult:
```

Inside the function, the `_fetch_atlas_slice_bytes` call (line 514-519) becomes:

```python
ref_bytes = _fetch_atlas_slice_bytes(
    atlas_obj,
    pos,
    show_borders=show_borders,
)
```

And the grid-mode `_build_atlas_grid` call (line 547-552) drops `cell_width`:

```python
grid_img = _build_atlas_grid(
    atlas,
    positions,
    target_image=target_image,
    show_borders=show_borders,
)
```

- [ ] **Step 3: Run tests**

Run: `python -m pytest tests/ -v -x`
Expected: Existing tests pass (some estimation tests may fail if they pass `atlas_resolution` — those get fixed in later tasks)

- [ ] **Step 4: Commit**

```bash
git add langslice/estimation/_shared_common.py langslice/estimation/_tool_logic.py
git commit -m "refactor: remove atlas_resolution from shared estimation internals

Atlas slices are now sent at native resolution without upscaling."
```

---

### Task 3: Remove `atlas_resolution` from Gemini estimation modules

**Files:**
- Modify: `langslice/estimation/google/tool_definitions.py`
- Modify: `langslice/estimation/google/ap_multi_slice.py`
- Modify: `langslice/estimation/google/ap_single_slice.py`
- Modify: `langslice/estimation/google/ap_image_gen.py`

- [ ] **Step 1: Update `google/tool_definitions.py`**

Remove `atlas_resolution` from both `_handle_fetch_atlas` (line 91) and `_process_ap_function_calls` (line 149). Both functions pass it through to `_handle_fetch_atlas_core` which no longer accepts it.

In `_handle_fetch_atlas`, remove the `atlas_resolution: int` parameter and drop it from the `_handle_fetch_atlas_core` call.

In `_process_ap_function_calls`, remove `atlas_resolution: int = 1024` from the signature (line 149) and drop it from the `_handle_fetch_atlas` call (line 201).

- [ ] **Step 2: Update `google/ap_multi_slice.py`**

Remove `atlas_resolution` from:
- `_process_group_function_calls_gc` signature (line 120) and its call to `_handle_fetch_atlas` (line 176)
- `estimate_group` signature (line 279) and its call to `_process_group_function_calls_gc` (line 656)

- [ ] **Step 3: Update `google/ap_single_slice.py`**

Remove `atlas_resolution` from the `estimate_position` signature (line 75) and its call to `_process_ap_function_calls` (line 414).

- [ ] **Step 4: Update `google/ap_image_gen.py`**

Remove `atlas_resolution` from the `estimate_position_image_gen` signature (line 230) and the `_fetch_atlas_slice_bytes` call inside (line 428). The call becomes:

```python
ref_bytes = _fetch_atlas_slice_bytes(
    atlas_obj,
    pos,
    show_borders=show_borders,
)
```

- [ ] **Step 5: Run linter and tests**

Run: `python -m ruff check langslice/estimation/google/ && python -m pytest tests/ -v -x`
Expected: PASS (unused imports may need cleanup)

- [ ] **Step 6: Commit**

```bash
git add langslice/estimation/google/
git commit -m "refactor: remove atlas_resolution from Gemini estimation modules"
```

---

### Task 4: Remove `atlas_resolution` from OpenAI estimation modules

**Files:**
- Modify: `langslice/estimation/openai/tool_definitions.py`
- Modify: `langslice/estimation/openai/ap_multi_slice.py`
- Modify: `langslice/estimation/openai/ap_single_slice.py`
- Modify: `langslice/estimation/openai/ap_image_gen.py`

- [ ] **Step 1: Update `openai/tool_definitions.py`**

Remove `atlas_resolution: int` from both `_handle_fetch_atlas_openai` (line 132) and `_process_ap_function_calls_openai` (line 197). Drop it from all inner calls to `_handle_fetch_atlas_core`.

- [ ] **Step 2: Update `openai/ap_multi_slice.py`**

Remove `atlas_resolution` from both `_process_group_function_calls_openai` (line 74) and `estimate_group` (line 253). Drop from all inner calls.

- [ ] **Step 3: Update `openai/ap_single_slice.py`**

Remove `atlas_resolution` from `estimate_position` (line 66). Drop from inner call.

- [ ] **Step 4: Update `openai/ap_image_gen.py`**

Remove `atlas_resolution` from `estimate_position_image_gen` (line 167). Update `_fetch_atlas_slice_bytes` call (line 369) to drop `max_long_edge`.

- [ ] **Step 5: Run linter and tests**

Run: `python -m ruff check langslice/estimation/openai/ && python -m pytest tests/ -v -x`
Expected: PASS

- [ ] **Step 6: Commit**

```bash
git add langslice/estimation/openai/
git commit -m "refactor: remove atlas_resolution from OpenAI estimation modules"
```

---

### Task 5: Remove `atlas_resolution` from whole-brain estimation agents

**Files:**
- Modify: `langslice/whole_brain/estimation_agents.py`

- [ ] **Step 1: Remove all `atlas_resolution=1024` keyword arguments**

There are 4 calls to `estimate_position_image_gen` in this file (lines 80, 112, 180, 202). Remove `atlas_resolution=1024` from each.

- [ ] **Step 2: Run linter**

Run: `python -m ruff check langslice/whole_brain/estimation_agents.py`
Expected: PASS

- [ ] **Step 3: Commit**

```bash
git add langslice/whole_brain/estimation_agents.py
git commit -m "refactor: remove atlas_resolution from whole-brain estimation agents"
```

---

### Task 6: Downscale tissue slices to atlas native resolution

Currently, `prepare_image_for_vlm` uses a hardcoded max of 4096px / 12M pixels. For estimation, tissue slices should be capped at the atlas's native long edge instead.

**Files:**
- Modify: `langslice/estimation/google/ap_multi_slice.py:319-333`
- Modify: `langslice/estimation/google/ap_single_slice.py` (find the `prepare_image_for_vlm` call)
- Modify: `langslice/estimation/google/ap_image_gen.py` (find the `prepare_image_for_vlm` call)
- Modify: `langslice/estimation/openai/ap_multi_slice.py` (find the `prepare_image_for_vlm` call)
- Modify: `langslice/estimation/openai/ap_single_slice.py` (find the `prepare_image_for_vlm` call)
- Modify: `langslice/estimation/openai/ap_image_gen.py` (find the `prepare_image_for_vlm` call)
- Modify: `langslice/whole_brain/estimation_agents.py:32-37`

- [ ] **Step 1: Update `google/ap_multi_slice.py`**

In `estimate_group`, after loading the atlas (line 316-317), compute the native long edge and use it to cap slice images:

```python
from langslice.atlas.core import get_coronal_long_edge

# ...existing atlas load...
atlas = _load_atlas_lazy(atlas_name)
pos_lo, pos_hi = _get_position_range_lazy(atlas)
atlas_long_edge = get_coronal_long_edge(atlas)
```

Then change the `prepare_image_for_vlm` call (line 324) from:

```python
prep = prepare_image_for_vlm(normalized)
```

to:

```python
prep = prepare_image_for_vlm(normalized, max_long_edge=atlas_long_edge)
```

- [ ] **Step 2: Update `google/ap_single_slice.py`**

Same pattern — after loading the atlas, get `atlas_long_edge` and pass it to `prepare_image_for_vlm`.

- [ ] **Step 3: Update `google/ap_image_gen.py`**

Same pattern.

- [ ] **Step 4: Update OpenAI estimation modules**

Apply the same pattern to:
- `openai/ap_multi_slice.py`
- `openai/ap_single_slice.py`
- `openai/ap_image_gen.py`

- [ ] **Step 5: Update `whole_brain/estimation_agents.py`**

The `_prepare_slice` function (line 32-37) currently uses `_VLM_MAX_LONG_EDGE = 2048`. It doesn't have access to the atlas. Change the function to accept a `max_long_edge` parameter:

```python
def _prepare_slice(image_path: str, *, max_long_edge: int = 2048) -> Image.Image:
    """Load and preprocess a slice image, matching the estimation pipeline."""
    raw = Image.open(image_path).convert("RGB")
    canonical = normalize_image(raw)
    downscaled = prepare_image_for_vlm(canonical, max_long_edge=max_long_edge).image
    return adaptive_preprocess(downscaled)
```

Then in `run_anchor_estimation` and `run_slice_estimation`, load the atlas first and pass `get_coronal_long_edge(atlas)`:

```python
from langslice.atlas.core import get_coronal_long_edge, load_atlas

atlas = load_atlas(atlas_name)
atlas_long_edge = get_coronal_long_edge(atlas)
image = _prepare_slice(image_path, max_long_edge=atlas_long_edge)
```

Note: `load_atlas` is already imported and called in the fallback path of `run_anchor_estimation`. Move the load earlier so it's available for `_prepare_slice`.

- [ ] **Step 6: Remove the `_VLM_MAX_LONG_EDGE = 2048` constant**

It's no longer used after the above changes. Delete line 22.

- [ ] **Step 7: Run tests**

Run: `python -m pytest tests/ -v -x`
Expected: PASS

- [ ] **Step 8: Commit**

```bash
git add langslice/estimation/ langslice/whole_brain/estimation_agents.py
git commit -m "feat: downscale tissue slices to atlas native resolution

Instead of hardcoded 2048/4096px caps, estimation now scales tissue
slices to match the atlas coronal long edge (~528px for allen_mouse_25um).
This avoids wasting tokens on resolution the atlas can't match."
```

---

### Task 7: Fix `media_resolution` bug in `ap_multi_slice.py`

The `media_resolution` parameter is accepted but never passed to `GenerateContentConfig`. Fix this for both `google/ap_multi_slice.py` and `google/ap_single_slice.py`.

**Files:**
- Modify: `langslice/estimation/google/ap_multi_slice.py:430-508`
- Modify: `langslice/estimation/google/ap_single_slice.py` (check for same bug)

- [ ] **Step 1: Check `ap_single_slice.py` for the same bug**

Read the file and verify whether `media_resolution` is passed to the config there. If it has the same bug, fix it too.

- [ ] **Step 2: Fix `ap_multi_slice.py`**

Change the `GenerateContentConfig` construction (line 502-508) to include `media_resolution`:

```python
_MEDIA_RES_MAP = {
    "low": "MEDIA_RESOLUTION_LOW",
    "medium": "MEDIA_RESOLUTION_MEDIUM",
    "high": "MEDIA_RESOLUTION_HIGH",
    "ultra_high": "MEDIA_RESOLUTION_ULTRA_HIGH",
}

# ...inside estimate_group, after line 431...
media_res_enum = _MEDIA_RES_MAP.get(media_resolution, "MEDIA_RESOLUTION_LOW")

# ...the config becomes:
config = types.GenerateContentConfig(
    system_instruction=system_instruction,
    temperature=temperature,
    max_output_tokens=8000,
    thinking_config=cast(Any, thinking_cfg),
    tools=cast(Any, tools),
    media_resolution=media_res_enum,
)
```

**Important:** Check the google-genai SDK docs via context7 to confirm the exact field name and enum values before implementing. The field may be `media_resolution` on `GenerateContentConfig` or it may need to be set per-part. Verify first.

- [ ] **Step 3: Change the default from `"ultra_high"` to `"low"`**

The research proved LOW is best for estimation. Update the CLI default in `cli.py` (line 311):

```python
"--media-resolution",
default="low",
```

And update the fallback in `ap_multi_slice.py` (line 430-431):

```python
if media_resolution is None:
    media_resolution = "low"
```

- [ ] **Step 4: Run the full check suite**

Run: `python -m ruff check . && python -m basedpyright && python -m pytest tests/ -v`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add langslice/estimation/google/ langslice/cli.py
git commit -m "fix: wire media_resolution into GenerateContentConfig

Previously the parameter was accepted but silently ignored — the model
always used the default (1120 tokens/image). Research showed LOW (280
tokens) is optimal for estimation, so that's now the default."
```

---

### Task 8: Remove `--atlas-resolution` and `--vlm-resolution` CLI flags for estimation

**Files:**
- Modify: `langslice/cli.py`

- [ ] **Step 1: Remove `--atlas-resolution` from estimate parser (line 271-276)**

Delete these lines from `_add_estimate_parser`.

- [ ] **Step 2: Remove `--atlas-resolution` from estimate-group parser (line 337-342)**

Delete these lines from `_add_estimate_group_parser`.

- [ ] **Step 3: Remove `--vlm-resolution` from estimate-group parser (line 343-348)**

Delete these lines. Tissue slices are now auto-capped to atlas native resolution.

- [ ] **Step 4: Remove `atlas_resolution=args.atlas_resolution` from all estimation call sites**

In `_run_estimate` and `_run_estimate_group`, remove the `atlas_resolution=` keyword argument from all calls to estimation functions (lines 426, 456, 638, 650, 691, 702).

- [ ] **Step 5: Remove `vlm_resolution` usage from `_run_estimate_group`**

The `prepare_image_for_vlm(canonical, max_long_edge=args.vlm_resolution)` call (line 374) should use a large default since the estimation functions themselves now handle the atlas-matched downscale. Change to:

```python
prep = prepare_image_for_vlm(canonical)
```

This uses the existing defaults (4096px / 12M pixels) as a safety cap. The estimation functions will do the atlas-matched downscale internally.

- [ ] **Step 6: Run the CLI to verify it still works**

Run: `langslice estimate --help` and `langslice estimate-group --help`
Expected: No `--atlas-resolution` flag. No `--vlm-resolution` flag on estimate-group.

- [ ] **Step 7: Commit**

```bash
git add langslice/cli.py
git commit -m "refactor: remove --atlas-resolution and --vlm-resolution CLI flags

Image resolution is now derived automatically from the atlas's native
coronal slice dimensions."
```

---

### Task 9: Update eval harness

**Files:**
- Modify: `eval/eval_group.py`

- [ ] **Step 1: Remove `atlas_resolution` from eval_group.py**

Remove `atlas_resolution=args.atlas_resolution` from the estimation call (line 339). Remove the `"atlas_resolution"` key from the results metadata dict (line 462). Remove the `--atlas-resolution` argument definition if it exists in the argparser.

- [ ] **Step 2: Update the `prepare_image_for_vlm` call (line 241)**

Currently uses `max_long_edge=2048`. Change to use the default:

```python
prep = prepare_image_for_vlm(canonical)
```

The estimation functions handle atlas-matched downscaling internally.

- [ ] **Step 3: Run the eval harness help**

Run: `python eval/eval_group.py --help`
Expected: No `--atlas-resolution` flag.

- [ ] **Step 4: Commit**

```bash
git add eval/eval_group.py
git commit -m "refactor: remove atlas_resolution from eval harness"
```

---

### Task 10: Run full verification

- [ ] **Step 1: Run all three checks**

```bash
python -m pytest tests/ -v
python -m ruff check .
python -m basedpyright
```

Expected: All pass.

- [ ] **Step 2: Smoke test the CLI**

```bash
langslice estimate-group --help
```

Verify: no `--atlas-resolution`, no `--vlm-resolution`, `--media-resolution` default is `low`.

- [ ] **Step 3: Commit any remaining fixes**

If any check fails, fix and commit.

---

## Notes

- The `--vlm-resolution` flag on the `register` command is **not** touched by this plan. Registration has different resolution requirements (2048px for Elastix B-spline detail) and should keep its own controls.
- The `atlas_resolution` parameter in `export.py` and `tests/test_quicknii_math.py` means voxel resolution in microns — completely unrelated. Do not modify.
- Eval hypothesis JSON files in `eval/hypotheses/` contain `"atlas_resolution": 1024` in their recorded configs. These are historical records. Do not modify them — they document what settings were used for past experiments.
