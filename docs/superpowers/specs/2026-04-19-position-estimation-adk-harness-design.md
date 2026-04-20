# Position-estimation ADK harness — design spec

**Date:** 2026-04-19
**Branch:** `feat/harness-adk`
**Target:** Gemma 4 hackathon submission (deadline 2026-05-18)

## Goal

Finalize the tool-use agent harness for position-estimation (AP / ML / DV, multi-orientation) so that:

1. The two current tool-use estimators (single-slice, multi-slice group) share one loop and one tool registry, implemented on top of **Google ADK** (`google-adk`).
2. The tools exposed to the agent are the ones we want to appear in Gemma training data: `fetch_atlas`, `zoom`, `side_by_side`, and the submission tools.
3. The codebase is provider-neutral — Gemini, Gemma (via Ollama), OpenRouter, OpenAI-compatible, all reachable with a one-line model swap via `LiteLlm`.
4. The module tree is ready for a future `langslice/harness/registration/` agent using ADK multi-agent orchestration (estimator passes slices to warper via `AgentTool`).

## Non-goals (this round)

- Registration agent — reserved as empty namespace; built post-hackathon.
- Whole-brain pipeline — keeps using `estimation/image_gen.py` (not tool-use). Out of scope.
- SFT training-data generation pipeline (`build_triplets.py`, `distill_cot.py`) — separate work downstream.
- Plane support in whole-brain (still coronal-only). This round generalizes the atlas helpers and single/group harness; whole-brain's image-gen estimator stays coronal-only for now.
- New providers beyond what ADK+LiteLlm already supports.

## Task framing

Task is **position estimation along the slice-normal axis** — AP for coronal, ML for sagittal, DV for horizontal. Existing `ap_` symbols and coronal-hardcoded helpers get generalized. Public result types rename (`APResult` → `PositionResult`) with a back-compat alias so the whole-brain pipeline and slice-bench keep working during the transition.

## Baselines to protect

| Baseline | Config | Value |
|---|---|---|
| Single-slice | Flash, 25um, medium media, MEDIUM thinking, M01+M09 weighted | 0.252 mm MAE |
| Multi-slice group | Pro, 25um, medium media, M03 | 0.186 mm MAE |
| Whole-brain | Pro anchors + Flash non-anchors, M01 | 0.173 mm MAE (NOT touched by this refactor) |

Post-refactor Flash single-slice and Flash multi-slice group performance on mid-brain M01 slices must not regress by more than 0.05 mm MAE vs a fresh pre-refactor baseline on the same slices.

---

## Architecture

### Module layout (post-refactor)

```
langslice/
├── harness/
│   ├── __init__.py
│   ├── estimation/
│   │   ├── __init__.py          # public API (compat with langslice.estimation symbols)
│   │   ├── _types.py            # PositionResult, MultiSliceResult; APResult = PositionResult
│   │   ├── session.py           # ADK session state + artifact-key conventions
│   │   ├── tools.py             # fetch_atlas, zoom, side_by_side, submit_estimate, submit_group_estimate
│   │   ├── validators.py        # sweep / neighbor-bracket / monotonicity / interval checks
│   │   ├── prompts.py           # plane-aware instruction builders
│   │   ├── single_slice.py      # build_single_slice_agent(...) -> LlmAgent
│   │   ├── group.py             # build_group_agent(...) -> LlmAgent
│   │   ├── runner.py            # run_agent(...): wires InMemoryRunner, streams events, returns result
│   │   ├── image_gen.py         # moved from estimation/google/ap_image_gen.py (raw google-genai, unchanged behavior)
│   │   └── debug.py             # moved from estimation/debug.py
│   └── registration/
│       └── __init__.py          # placeholder; populated post-hackathon
└── atlas/                       # plane-generalized helpers (see Atlas section below)
```

### Files deleted in this refactor

- `langslice/estimation/google/ap_single_slice.py`
- `langslice/estimation/google/ap_multi_slice.py`
- `langslice/estimation/google/ap_tool_use.py` (the existing back-compat shim — subsumed)
- `langslice/estimation/google/common.py`
- `langslice/estimation/google/tool_definitions.py`
- `langslice/estimation/google/batch_eval.py` (moved to `langslice/harness/estimation/batch_eval.py` if still used; deleted if not)
- `langslice/estimation/openai/ap_single_slice.py`
- `langslice/estimation/openai/ap_multi_slice.py`
- `langslice/estimation/openai/common.py`
- `langslice/estimation/openai/tool_definitions.py`
- `langslice/estimation/openai/ap_image_gen.py` (folded into unified `image_gen.py` if provider split is still needed, otherwise deleted)
- `langslice/estimation/_shared_common.py`
- `langslice/estimation/_tool_logic.py`
- `langslice/estimation/_types.py` (content moved to `harness/estimation/_types.py`)
- `langslice/estimation/__init__.py` — replaced by a thin re-export shim that forwards `estimate_position`, `estimate_group`, `APResult`, `MultiSliceResult` to `langslice.harness.estimation`. Marked deprecated; removed after all consumers are updated. (Alternative: update all imports and delete outright; preferred if grep count is small.)

### Import sweep

- `langslice/cli.py`
- `langslice/whole_brain/*.py`
- `eval/eval_brain.py`, `eval/eval_group.py`
- `slice-bench/slice_bench/adapters/gemini.py`
- `tauri-gui/src-tauri/src/*.rs` — if any Rust sidecar code imports Python symbols by name, check
- `tests/`

All updated to `from langslice.harness.estimation import ...`. The compat shim at `langslice/estimation/__init__.py` is kept during transition but removed once greps are clean.

---

## Atlas: plane generalization

### `langslice/atlas/space.py`

- **Delete** `require_coronal_layout()`. Callers no longer assert; they pass `plane`.
- Keep `atlas_space_context(atlas)` — already exposes `ap_axis_index`, `dv_axis_index`, `ml_axis_index`.
- Add `slice_axis_index(context, plane)` helper that maps `plane ∈ {"coronal", "sagittal", "horizontal"}` to the correct axis via the existing per-atlas context.

### `langslice/atlas/core.py`

Every slice helper takes a `plane: Plane` argument (default `"coronal"` during the transition):

- `position_mm_to_index(atlas, position_mm, *, plane)`
- `index_to_position_mm(atlas, idx, *, plane)`
- `get_position_range_mm(atlas, *, plane)` — returns `(min_mm, max_mm)` along the slice-normal axis.
- `get_reference_slice(atlas, position_mm, *, plane)`
- `get_boundary_slice(atlas, position_mm, *, plane)`
- `get_composite_slice(atlas, position_mm, *, plane, opacity=0.4)`
- `get_colored_region_slice(atlas, position_mm, *, plane)`
- `get_smoothed_boundary_slice(atlas, position_mm, *, plane, target_size=None)`
- `get_additional_reference_slice(atlas, reference_name, position_mm, *, plane)`
- `get_structure_mask_slice(atlas, structure, position_mm, *, plane)`
- `get_slice_region_metadata(atlas, position_mm, *, plane)`
- `get_region_at_position(atlas, position_mm, *, plane, dv_index=None, ml_index=None, ...)` — in-plane axis args are no longer AP-specific; rename to two generic `in_plane_a`/`in_plane_b` args OR keep DV/ML args but pivot their meaning based on plane. Keep the DV/ML names — still anatomically correct regardless of slicing plane.

### Axis naming

- `get_coronal_long_edge(atlas)` → `get_in_plane_long_edge(atlas, plane)` — returns `max(shape[a], shape[b])` where `a, b` are the two non-slice-normal axes.

### Prompts

- System instructions are plane-aware, via `build_single_slice_prompt(plane, atlas, ...)` and `build_group_prompt(plane, atlas, ..., interval_mm, thickness_um)`.
- Plane-specific boilerplate:
  - Coronal: `"0.0 mm = anterior (olfactory bulb); higher = posterior."`
  - Sagittal: `"0.0 mm = left hemisphere lateral edge; higher = right."` (exact wording finalized during implementation)
  - Horizontal: `"0.0 mm = dorsal (top of brain); higher = ventral."`
- Axis-name substitution: `"AP"` → `"AP" | "ML" | "DV"` per plane.

---

## Agent design

### The four tools

#### `fetch_atlas`

```python
def fetch_atlas(positions_mm: list[float], tool_context: ToolContext) -> dict:
    """Fetch atlas slices at specified positions along the session's slice-normal axis.

    The agent can request 1 to 8 positions per call. Each slice is saved as an
    artifact keyed 'atlas:<mm:.2f>' and returned as image Parts for the next turn.
    Positions outside the valid range are clamped. Duplicate positions within
    0.02 mm of a previous request are coalesced.
    """
```

- Updates `tool_context.state["fetched_positions"]`.
- Sets `tool_context.state["saw_broad_sweep"] = True` if ≥3 positions are provided.
- Sets `tool_context.state["saw_narrow_sweep"] = True` if ≥3 positions span ≤1.0 mm.
- Errors: `BAD_ARGS` (no positions), `EMPTY_RESULT` (all positions clamped outside atlas).
- Returns structured dict `{"status": "ok", "positions_mm": [...], "description": "..."}`; attaches `types.Part` images for the new-turn turn's content.

#### `zoom`

```python
def zoom(source: str, bbox: list[int], tool_context: ToolContext) -> dict:
    """Return a zoomed crop of a previously-fetched image.

    Args:
        source: "target" (or "target:N" for multi-slice group), or "atlas:<mm>"
            — must match an existing artifact.
        bbox: [y1, x1, y2, x2] integers 0–1000 (Gemini/Gemma native format).

    The crop is resized to `get_in_plane_long_edge(atlas, plane)` and saved as
    'zoom:<source>:<hash>'.
    """
```

- Errors: `BAD_SOURCE` (no matching artifact), `BAD_BBOX` (y1≥y2 or x1≥x2, or outside 0–1000), `EMPTY_CROP` (zero-area).
- Bbox is interpreted against the source artifact's own pixel dimensions.

#### `side_by_side`

```python
def side_by_side(left: str, right: str, tool_context: ToolContext) -> dict:
    """Build an aspect-ratio-matched horizontal composite of two images.

    Both source images are rescaled to a common height (aspect-ratio preserved
    per panel), placed side-by-side with a thin gap, and stamped with short
    labels. Saved as 'side_by_side:<left>:<right>'.
    """
```

- `left` and `right` each accept `"target"`, `"target:N"`, `"atlas:<mm>"`, `"zoom:..."`.
- Errors: `BAD_SOURCE`.

#### `submit_estimate` (single-slice)

```python
def submit_estimate(position_mm: float, reasoning: str, tool_context: ToolContext) -> dict:
    """Submit the final position estimate for the target slice."""
```

- Stores `{"position_mm": ..., "reasoning": ...}` in `tool_context.state["result"]`.
- Sets `tool_context.actions.escalate = True` → Runner exits the loop.

#### `submit_group_estimate` (multi-slice)

```python
def submit_group_estimate(positions_mm: list[float], reasoning: str, tool_context: ToolContext) -> dict:
    """Submit final position estimates for all slices in the group, in order."""
```

- Same pattern; validates list length matches `state["n_slices"]`.

### Validator callbacks

`before_tool_callback` intercepts `submit_estimate` / `submit_group_estimate` calls:

- **Single-slice**:
  - Reject if `not state["saw_broad_sweep"]` (unless iteration ≥ `max_iterations - 2`).
  - Reject if `not state["saw_narrow_sweep"]` (same unless clause).
  - Reject if no neighbor bracket: at least one `fetched_position` below and one above `position_mm` within 0.25 mm (unless at atlas edge or near iteration limit).
- **Multi-slice group**:
  - Reject if `len(positions_mm) != state["n_slices"]`.
  - Reject if positions not monotonically increasing.
  - Reject if any interval deviates >50% from `state["interval_mm"]`.
  - Reject if no broad sweep, no narrow sweep.

Rejections return a dict response to the model: `{"status": "error", "error": "<actionable text>"}`. ADK feeds this back to the LLM as a tool output.

Non-submit tools (`fetch_atlas`, `zoom`, `side_by_side`) are never gated.

### Loop semantics

- `LlmAgent(..., include_contents="default")` preserves conversation history.
- No tool call on a turn → `after_model_callback` detects, injects a nudge prompt as a synthetic user turn, loop continues. Nudge text matches current `_build_nudge_text`.
- `submit_*` setting `escalate = True` terminates.
- `max_iterations` enforced externally: the runner caps iterations by counting events; on cap hit, extract the best result if any, else fall back to midpoint (matches current behavior).
- One retry with fresh history on first-attempt failure (matches current `for attempt in range(2)` in both loops).

### Session state shape

Stored in `tool_context.state`:

```python
{
    "atlas": "allen_mouse_25um",
    "plane": "coronal",
    "axis_label": "AP",           # for prompt substitution
    "pos_lo": 0.0,
    "pos_hi": 13.2,
    "n_slices": 1 | N,            # 1 for single-slice, N for group
    "interval_mm": 0.0 | 0.200,   # 0 for single-slice
    "thickness_um": 50,
    "fetched_positions": [],
    "saw_broad_sweep": False,
    "saw_narrow_sweep": False,
    "images_fetched": 0,
    "result": None,               # set by submit_*
    "max_iterations": 20 | 25,
}
```

Artifacts (`tool_context.save_artifact` / `load_artifact`):

- `target` — single-slice query image.
- `target:1`, ..., `target:N` — multi-slice group images.
- `atlas:<mm:.2f>` — cached atlas slice at position `mm`.
- `zoom:<source>:<hash>` — zoom results.
- `side_by_side:<left>:<right>` — composites.

### Model configuration

Model is passed directly to `LlmAgent`:

```python
from google.adk.agents import LlmAgent
from google.genai import types

# Direct Gemini
model = "gemini-3.1-pro-preview"

# OpenRouter / Gemma / any LiteLLM-supported model:
# from google.adk.models.lite_llm import LiteLlm
# model = LiteLlm(model="openrouter/google/gemma-3-27b-it")

agent = LlmAgent(
    model=model,
    name="group_position_estimator",
    instruction=build_group_prompt(atlas, plane, n_slices, interval_mm, thickness_um),
    tools=[fetch_atlas, zoom, side_by_side, submit_group_estimate],
    generate_content_config=types.GenerateContentConfig(
        temperature=1.0,
        max_output_tokens=8000,
        thinking_config=thinking_cfg,
        media_resolution=types.MediaResolution.MEDIA_RESOLUTION_MEDIUM,
    ),
    before_tool_callback=gate_submit_tool,
    after_model_callback=nudge_on_no_tool_call,
)
```

- `thinking_cfg` derived from model via existing `vlm_config.build_thinking_config(...)` logic, lifted to `harness/estimation/prompts.py` or a new `harness/estimation/model_config.py`.
- Temperature stays at 1.0 (Gemini low-temp policy).

### Future multi-agent hook

Registration agent lives at `langslice/harness/registration/`. Once built:

```python
# In harness/estimation/group.py (post-hackathon)
from langslice.harness.registration import build_warper_agent
from google.adk.tools.agent_tool import AgentTool

warper = build_warper_agent(...)
group_agent = LlmAgent(
    ...,
    tools=[fetch_atlas, zoom, side_by_side, submit_group_estimate,
           AgentTool(agent=warper)],   # delegate warping per slice
)
```

This refactor's file boundaries and session-state shape are chosen so that hook is a simple addition, not a restructure.

---

## Implementation order and eval gates

All work on `feat/harness-adk`. Each commit is verifiable independently.

### 0. Baseline snapshot (1 API run, **Flash**)

Before any code change:

```bash
python eval/eval_group.py \
  --images references/TestImages/M01 \
  --ground-truth references/TestImages/M01/ground_truth.json \
  --model gemini-3-flash-preview \
  --json > eval_outputs/baseline_pre_adk_M01.json
```

Filter to mid-brain slices only (AP 4.0–7.0 mm range) either by pre-selecting those images or post-filtering the JSON. Target: 5–10 slices, ~2 groups.

This is the MAE to beat on the final check.

### 1. Plane generalization of `langslice/atlas/`

- Drop `require_coronal_layout()`.
- Add `plane` parameter to all slice helpers (default `"coronal"`).
- Rename `get_coronal_long_edge` → `get_in_plane_long_edge`.
- Unit tests for each plane on an atlas stub.

No API eval gate — pure refactor.

### 2. Rename symbols, create `harness/estimation/` package

- Move `_types.py` into `harness/estimation/_types.py`; rename `APResult` → `PositionResult` with `APResult` alias.
- Stub empty modules `session.py`, `tools.py`, `validators.py`, `prompts.py`, `single_slice.py`, `group.py`, `runner.py`.
- Move `image_gen.py` and `debug.py` (no behavior change, import-update only).
- Compat shim `langslice/estimation/__init__.py` re-exports.
- Run existing `pytest` — must pass.

### 3. Single-slice ADK port

- Implement `fetch_atlas` + `submit_estimate` tools in `tools.py`.
- Implement validator callback for `submit_estimate` in `validators.py`.
- Implement `build_single_slice_agent()` in `single_slice.py`.
- Implement `runner.py` with event-stream tap writing the existing `agent_trace` JSON.
- Wire the compat shim so `estimate_position(...)` calls the new agent.

**Eval gate (1 API run, Flash):**

```bash
python eval/eval_group.py \
  --images references/TestImages/M01 \
  --ground-truth references/TestImages/M01/ground_truth.json \
  --model gemini-3-flash-preview \
  --group-size 1 \
  --json > eval_outputs/post_step3_single_M01.json
```

(Single-slice evaluation via group eval with group-size=1, or a dedicated flag path — decide during implementation.)

Gate: MAE within **0.05 mm** of baseline, on the same mid-brain slices.

### 4. Multi-slice group ADK port

- Implement `submit_group_estimate` and its validator.
- Implement `build_group_agent()` in `group.py`.
- Wire the compat shim so `estimate_group(...)` calls the new agent.

**Eval gate (1 API run, Flash, mid-brain):** same `eval_group.py` command, default group size. Same 0.05 mm gate.

### 5. Add `zoom` tool

- Implement in `tools.py`; no gating callback.
- Update both system prompts to mention the tool and when it might help.

**Eval gate (1 API run, Flash, mid-brain):** same command. MAE must not regress past baseline + 0.05 mm; improvement is welcome but not required.

### 6. Add `side_by_side` tool

- Implement in `tools.py`; no gating callback.
- Update both system prompts to mention the tool.

**Eval gate (1 API run, Flash, mid-brain):** same command. Same gate.

### 7. Final acceptance run

- One consolidated run with all tools enabled on mid-brain M01. Archive JSON. Done.

### Total API budget

5 Flash multi-slice runs on mid-brain M01 (~5–10 slices, 2 groups per run). Rough cost with current Flash pricing: under $5 total.

Zero Pro runs for this refactor.

### Dev-loop smoke tests (pennies)

Between commits where a full eval isn't warranted:

```bash
python -c "
from langslice.harness.estimation import estimate_position
from PIL import Image
import logging; logging.basicConfig(level=logging.INFO)
img = Image.open('references/TestImages/M01/<one_image>.png')
r = estimate_position(img, 'allen_mouse_25um', model_name='gemini-3-flash-preview')
print(r.position_mm, r.reasoning[:100])
"
```

Single image, Flash, one fetch_atlas cycle + submit. ~$0.01 per invocation. Used liberally during development to catch wiring bugs before spending full-eval budget.

---

## Non-API tests

### Unit tests

- `tests/test_atlas_plane.py` — slice helpers work on coronal, sagittal, horizontal with a stub atlas.
- `tests/test_harness_tools.py` — `zoom` bbox parsing, `atlas:<mm>` key parsing, artifact round-trip.
- `tests/test_harness_validators.py` — broad/narrow sweep detection, neighbor-bracket logic, monotonicity, interval checks, all edge cases (near-edge slices, iteration-limit escape valve).
- `tests/test_harness_prompts.py` — plane-aware instruction substitution (no `"AP"` leaks on sagittal, etc.).

### Integration tests

- `tests/test_harness_loop.py` — single-slice agent run with a fake model that scripts `fetch_atlas` → `submit_estimate`; verifies the Runner+state+escalate path without API calls.
- Equivalent multi-slice integration test with `submit_group_estimate`.

If ADK doesn't provide a mock-model hook, inject a fake `LlmAgent.model` that is a Python function returning canned `LlmResponse` objects. Determine during implementation.

---

## Risks and mitigations

| Risk | Mitigation |
|---|---|
| ADK's default retry/nudge/history semantics differ subtly from hand-rolled and regress MAE | Incremental eval gates (steps 3–6); revert any step that crosses the 0.05 mm threshold. |
| `before_tool_callback` can't return a synthetic tool response the way the current `_validate_submit_*` does | Verify during step 3 implementation; if unsupported, move validators inside the tool functions themselves (trivial port). |
| `save_artifact` → next-turn image visibility isn't automatic in ADK | Verify during step 3 implementation; if not automatic, tools return `types.Part` image objects directly in their result dict. |
| Dev-loop smoke-test script relies on CLI-not-yet-updated compat shim | Keep compat shim working through all steps; only remove at the very end. |
| `AgentTool` orchestration imposes shape constraints we haven't accounted for (future registration hook) | Not blocking for this round; flag as a spike during registration design. |

## Success criteria

- **Functional:** single-slice and multi-slice group both produce `PositionResult` / `MultiSliceResult` identical in shape to today's `APResult` / `MultiSliceResult`.
- **Performance:** Flash mid-brain M01 MAE within 0.05 mm of pre-refactor baseline.
- **Tool availability:** all four tools (`fetch_atlas`, `zoom`, `side_by_side`, `submit_*`) callable from both agents and visible in `eval_outputs/` trace files.
- **Cleanup:** `langslice/estimation/` either deleted outright or reduced to a back-compat shim.
- **Multi-provider:** swapping to `LiteLlm(model="openrouter/...")` is a one-line change in the CLI; verified with one Flash-equivalent via OpenRouter smoke test.
- **Training-data readiness:** `build_triplets.py` / `distill_cot.py` can introspect `tools.py` to emit the canonical tool declaration block.

## Open questions (to resolve during implementation)

1. Does `tool_context.save_artifact` auto-surface the saved image as multimodal content to the next turn, or must tools return `types.Part` objects directly? → verify against ADK docs / source during step 3.
2. Does `before_tool_callback` support returning a synthetic tool-response dict, or does rejection require raising? → verify against ADK source.
3. Does ADK ship a first-class fake-model harness for integration tests? → verify during step 3; if not, roll a minimal one.
4. How does `LiteLlm(model="openrouter/...")` handle multi-image message parts? → smoke test during the multi-provider verification step.
5. Final rename decision: should `langslice/estimation/` be a back-compat shim or deleted outright? → depends on import-grep count; decide after step 2.

---
