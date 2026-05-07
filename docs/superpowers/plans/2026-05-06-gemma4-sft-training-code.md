# Gemma 4 E4B SFT Training Code Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build the supervised fine-tuning training code that consumes a single-slice langslice-native trace JSONL and produces a Gemma 4 E4B LoRA adapter the RLVR phase loads.

**Architecture:** Mirrors the existing `models/langslice-gemma-4/training/rlvr/` layout, but v1 SFT is single-slice only. Renderer translates single-slice langslice trace shape → HF chat-template `messages` + `tools`. Collator runs `processor.apply_chat_template`, builds `labels` with `-100` on every non-assistant token (image-placeholder tokens included). TRL `SFTTrainer` consumes the LoRA-wrapped Unsloth `FastVisionModel`. Eval callback reuses `LangSliceEstimateEnv` from RLVR with `kind="single"` for the agent-loop runs.

**Tech Stack:** Python 3.11+, Unsloth `FastVisionModel`, TRL `SFTTrainer` + `SFTConfig`, HuggingFace transformers + datasets, PEFT LoRA, `processor.apply_chat_template(..., return_assistant_tokens_mask=True)`, trackio for metrics, pytest + ruff + basedpyright for verification.

**Reference docs:**
- Spec: `docs/superpowers/specs/2026-05-05-gemma4-sft-training-design.md`
- RLVR scaffolding (parallel pattern): `models/langslice-gemma-4/training/rlvr/`
- Production agent code: `src/langslice_harness/harness/estimation/`
- Test images + ground truth: `references/TestImages/M0[1-9]/ground_truth.json`

---

## Review Corrections Baked Into This Plan

These points supersede any stale group/multi-slice snippets from older drafts:

- v1 SFT is **single-slice only**: `system_prompt_kind == "single_slice"`, exactly one query image, final tool is always `submit_estimate`.
- Terminal submit args must include both `position_mm` and non-empty `reasoning`; production and RLVR both require the reasoning parameter.
- The exact default model ID is `unsloth/gemma-4-E4B-it`.
- `max_seq_length` is for Unsloth model loading and the custom collator's explicit overflow rejection. TRL `SFTConfig` must use `max_length=None` for VLM safety.
- Dataset validation must resolve every query image and atlas tool-result image on load, relative to the JSONL parent directory.
- Agent-loop eval must call RLVR env as it exists today: `LangSliceEstimateEnv.reset(..., kind="single", valid_range_mm=(pos_lo, pos_hi), ground_truth_positions_mm=(truth,))`, and append `result["content"]` for `fetch_atlas` tool messages.

## File Structure

**Files to create:**

```
models/langslice-gemma-4/training/
  sft/
    __init__.py             — package marker, public exports
    dataset.py              — Example dataclass, JSONL load + validation, subject-aware split
    render.py               — langslice trace → HF chat-template messages
    collate.py              — processor.apply_chat_template + labels masking
    eval.py                 — agent-loop eval callbacks (baseline + periodic) + metric utils
    train_sft.py            — driver script (CLI entry point)
  configs/
    sft_default.toml        — hyperparameters (mirrors grpo_default.toml structure)

tests/
  test_sft_dataset.py
  test_sft_render.py
  test_sft_collate.py
  test_sft_eval.py
  fixtures/
      sft_traces/
      single_slice_minimal.jsonl   — canned 1-row fixture for renderer/collator tests
      malformed_examples.jsonl     — bad rows for validation tests
      query.png, a3.png, a5.png, a7.png — tiny PNGs used by validation/render tests

docs/superpowers/notes/
  2026-05-06-unsloth-trl-api-verification.md  — output of Task 1
```

**Files to modify:**

- `requirements-rlvr.txt` at repo root — confirm versions cover SFT needs (probably no change; SFT uses same libs as RLVR)

**Files to delete (cleanup, Task 17):**

- `models/langslice-gemma-4/training/finetune.py`
- `models/langslice-gemma-4/data/build_triplets.py`
- `models/langslice-gemma-4/data/distill_cot.py`
- `models/langslice-gemma-4/data/generate_atlas_slices.py`

**Files NOT to change** (reuse, don't duplicate):

- `src/langslice_harness/harness/estimation/prompts.py` — production system prompts; renderer imports
- `src/langslice_harness/harness/estimation/tools.py` — production tool signatures; renderer imports
- `src/langslice_harness/atlas/core.py` — atlas loaders for renderer's metadata cache
- `models/langslice-gemma-4/training/rlvr/env.py` — `LangSliceEstimateEnv` reused by eval callback
- `models/langslice-gemma-4/training/rlvr/atlas_grid.py` — atlas-grid pre-render used by eval

---

## Task 1: Verify Unsloth + TRL VLM SFT API surface (research, no code)

**Files:**
- Create: `docs/superpowers/notes/2026-05-06-unsloth-trl-api-verification.md`

The spec assumed several Unsloth + TRL API names. Verify them before writing code per memory note `feedback_verify_third_party_docs`.

- [ ] **Step 1: Look up Unsloth FastVisionModel signatures via Context7**

Query Context7 for:
- `unsloth/unsloth` library
- Topics: `FastVisionModel.from_pretrained`, `FastVisionModel.get_peft_model`, `FastVisionModel.for_inference`, `FastVisionModel.for_training`

Record the exact current method signatures and any kwargs that have changed.

- [ ] **Step 2: Look up TRL SFTTrainer + SFTConfig signatures via Context7**

Query Context7 for:
- `huggingface/trl` library
- Topics: `SFTTrainer`, `SFTConfig`, `assistant_only_loss`, multimodal SFT, `processing_class`, `data_collator`

Confirm:
- `assistant_only_loss=True` is text-only / unsupported for VLM data (per Codex review).
- `SFTTrainer` accepts `model=<PeftModel>` without `peft_config` (or vice versa, never both).
- `SFTConfig` field names (`max_length`, `chat_template_kwargs`, etc.). For VLM SFT, confirm `max_length=None` is the safe default so TRL does not truncate image tokens.

- [ ] **Step 3: Verify processor.apply_chat_template return_assistant_tokens_mask for Gemma 4**

Query Context7 for:
- `huggingface/transformers` library
- Topic: `apply_chat_template return_assistant_tokens_mask` and Gemma 4 chat template

Confirm:
- `return_assistant_tokens_mask=True` is supported for VLM processors.
- Gemma 4's chat template uses `{% generation %}{% endgeneration %}` markers correctly around tool-call output.
- If unconfirmed, the manual-span fallback (Task 9) becomes the primary path.

- [ ] **Step 4: Verify the Gemma 4 E4B model ID and Unsloth quantization**

Query Context7 / Hugging Face for the exact model ID. Current expected ID is `unsloth/gemma-4-E4B-it`; verify before committing.

- [ ] **Step 5: Write the verification notes file**

Create `docs/superpowers/notes/2026-05-06-unsloth-trl-api-verification.md` with:
- Verified API signatures (one short code block per call)
- Any deltas vs the spec's assumed signatures
- The exact verified Gemma 4 E4B model ID
- Decision: trust `return_assistant_tokens_mask=True` (primary path) or use manual-span fallback (primary path) — based on Step 3 findings

- [ ] **Step 6: Commit**

```bash
git add docs/superpowers/notes/2026-05-06-unsloth-trl-api-verification.md
git commit -m "docs(gemma-4): verify unsloth + trl VLM SFT API surface"
```

---

## Task 2: Module skeletons + test fixtures

**Files:**
- Create: `models/langslice-gemma-4/training/sft/__init__.py`
- Create: `models/langslice-gemma-4/training/sft/dataset.py`
- Create: `models/langslice-gemma-4/training/sft/render.py`
- Create: `models/langslice-gemma-4/training/sft/collate.py`
- Create: `models/langslice-gemma-4/training/sft/eval.py`
- Create: `models/langslice-gemma-4/training/sft/train_sft.py`
- Create: `tests/test_sft_dataset.py`
- Create: `tests/test_sft_render.py`
- Create: `tests/test_sft_collate.py`
- Create: `tests/test_sft_eval.py`
- Create: `tests/fixtures/sft_traces/single_slice_minimal.jsonl`
- Create: `tests/fixtures/sft_traces/malformed_examples.jsonl`
- Create: `tests/fixtures/sft_traces/query.png`
- Create: `tests/fixtures/sft_traces/a3.png`
- Create: `tests/fixtures/sft_traces/a5.png`
- Create: `tests/fixtures/sft_traces/a7.png`

- [ ] **Step 1: Create the sft/ package marker**

Write `models/langslice-gemma-4/training/sft/__init__.py`:

```python
"""Supervised fine-tuning code for the langslice-gemma-4 model.

See docs/superpowers/specs/2026-05-05-gemma4-sft-training-design.md.
"""
```

- [ ] **Step 2: Create empty module stubs with docstrings**

Each of `dataset.py`, `render.py`, `collate.py`, `eval.py`, `train_sft.py` gets a one-line docstring describing its responsibility (per the file-structure section above) and `from __future__ import annotations` at the top. No implementation yet.

`models/langslice-gemma-4/training/sft/dataset.py`:
```python
"""Load + validate langslice-native SFT trace JSONL; subject-aware train/eval split."""

from __future__ import annotations
```

`models/langslice-gemma-4/training/sft/render.py`:
```python
"""Translate langslice-native trace examples to HF chat-template messages + tools."""

from __future__ import annotations
```

`models/langslice-gemma-4/training/sft/collate.py`:
```python
"""Apply processor.apply_chat_template and build labels with -100 outside assistant turns."""

from __future__ import annotations
```

`models/langslice-gemma-4/training/sft/eval.py`:
```python
"""SFT-time evaluation: agent-loop callbacks (baseline + periodic) and metric utilities."""

from __future__ import annotations
```

`models/langslice-gemma-4/training/sft/train_sft.py`:
```python
"""Driver script: CLI entry point for SFT training of Gemma 4 E4B via Unsloth + TRL."""

from __future__ import annotations
```

- [ ] **Step 3: Create empty test files**

For each test file, write a one-line module docstring and `from __future__ import annotations`. Tests will be added in subsequent tasks.

`tests/test_sft_dataset.py`:
```python
"""Tests for models/langslice-gemma-4/training/sft/dataset.py."""

from __future__ import annotations
```

Same shape for `test_sft_render.py`, `test_sft_collate.py`, `test_sft_eval.py`.

- [ ] **Step 4: Create canned fixture JSONL files**

`tests/fixtures/sft_traces/single_slice_minimal.jsonl` (one line, JSON below pretty-printed for readability — write it as a single line):

```json
{"bucket": 1, "atlas_name": "allen_mouse_25um", "atlas_version": "CCFv3", "plane": "coronal", "subject_id": "test_subj_01", "system_prompt_kind": "single_slice", "query_image_paths": ["query.png"], "user_prompt_text": "Estimate the AP position of this slice.", "trace": [{"tool_call": {"name": "fetch_atlas", "args": {"positions_mm": [3.0, 5.0, 7.0]}}, "tool_result": {"image_paths": ["a3.png", "a5.png", "a7.png"], "text": "Atlas at 3.00 mm | 5.00 mm | 7.00 mm"}}, {"submit": {"name": "submit_estimate", "args": {"position_mm": 5.2, "reasoning": "Best match after broad and narrow atlas comparison."}}}]}
```

Create the tiny PNG files referenced by the fixture:

```bash
python -c "from pathlib import Path; from PIL import Image; root=Path('tests/fixtures/sft_traces'); root.mkdir(parents=True, exist_ok=True); [Image.new('RGB',(32,32),c).save(root/n) for n,c in {'query.png':(128,128,128),'a3.png':(90,90,90),'a5.png':(130,130,130),'a7.png':(170,170,170)}.items()]"
```

Expected: `query.png`, `a3.png`, `a5.png`, and `a7.png` exist under `tests/fixtures/sft_traces/`.

`tests/fixtures/sft_traces/malformed_examples.jsonl` (6 lines, each a different defect):

```jsonl
{"bucket": 2, "atlas_name": "allen_mouse_25um", "atlas_version": "CCFv3", "plane": "coronal", "subject_id": "x", "system_prompt_kind": "single_slice", "query_image_paths": ["query.png"], "user_prompt_text": "x", "trace": [{"submit": {"name": "submit_estimate", "args": {"position_mm": 5.0, "reasoning": "x"}}}]}
{"bucket": 1, "atlas_name": "allen_mouse_25um", "atlas_version": "CCFv3", "plane": "coronal", "subject_id": "", "system_prompt_kind": "single_slice", "query_image_paths": ["query.png"], "user_prompt_text": "x", "trace": [{"submit": {"name": "submit_estimate", "args": {"position_mm": 5.0, "reasoning": "x"}}}]}
{"bucket": 1, "atlas_name": "allen_mouse_25um", "atlas_version": "CCFv3", "plane": "coronal", "subject_id": "x", "system_prompt_kind": "single_slice", "query_image_paths": [], "user_prompt_text": "x", "trace": [{"submit": {"name": "submit_estimate", "args": {"position_mm": 5.0, "reasoning": "x"}}}]}
{"bucket": 1, "atlas_name": "allen_mouse_25um", "atlas_version": "CCFv3", "plane": "coronal", "subject_id": "x", "system_prompt_kind": "single_slice", "query_image_paths": ["q.png"], "user_prompt_text": "x", "trace": [{"tool_call": {"name": "fetch_atlas", "args": {"positions_mm": [3.0]}}, "tool_result": {"image_paths": ["a.png"], "text": "x"}}]}
{"bucket": 1, "atlas_name": "allen_mouse_25um", "atlas_version": "CCFv3", "plane": "coronal", "subject_id": "x", "system_prompt_kind": "group", "query_image_paths": ["query.png"], "user_prompt_text": "x", "interval_mm": 0.3, "thickness_um": 30, "trace": [{"submit": {"name": "submit_estimate", "args": {"position_mm": 5.0, "reasoning": "x"}}}]}
{"bucket": 1, "atlas_name": "allen_mouse_25um", "atlas_version": "CCFv3", "plane": "coronal", "subject_id": "x", "system_prompt_kind": "single_slice", "query_image_paths": ["missing.png"], "user_prompt_text": "x", "trace": [{"submit": {"name": "submit_estimate", "args": {"position_mm": 5.0, "reasoning": "x"}}}]}
```

Defects: bucket≠1, empty subject_id, empty query_image_paths, trace doesn't end with submit, group kind rejected in v1, missing image path.

- [ ] **Step 5: Verify imports work**

Run:
```bash
python -c "import models.langslice_gemma_4.training.sft.dataset"
```

Wait — Python module names can't have hyphens. The directory `langslice-gemma-4` has hyphens; importing it requires path manipulation (already done elsewhere in the codebase since `models/langslice-gemma-4/training/rlvr/` is imported). Confirm by checking how `rlvr/train_grpo.py` is invoked:

Run:
```bash
ls models/langslice-gemma-4/training/rlvr/
```

Expected: lists existing files. The RLVR pattern uses `python -m rlvr.train_grpo` from inside the package, or invokes the script directly. Mirror that pattern; do not try to `import models.langslice_gemma_4...` from arbitrary places.

For unit tests (which run from the repo root), the test files add the package path manually:

Add to the top of each `tests/test_sft_*.py` file (after `from __future__ import annotations`):

```python
import sys
import types
import json
from pathlib import Path

_SFT_ROOT = Path(__file__).resolve().parents[1] / "models" / "langslice-gemma-4" / "training"
if str(_SFT_ROOT) not in sys.path:
    sys.path.insert(0, str(_SFT_ROOT))
```

This lets tests `from sft.dataset import ...` etc. Same trick `tests/test_rlvr_*.py` already uses.

- [ ] **Step 6: Run pytest to confirm test files at least collect**

Run:
```bash
python -m pytest tests/test_sft_dataset.py tests/test_sft_render.py tests/test_sft_collate.py tests/test_sft_eval.py --collect-only
```

Expected: collects 0 tests in 4 files, no errors. Tests don't exist yet.

- [ ] **Step 7: Commit**

```bash
git add models/langslice-gemma-4/training/sft/ tests/test_sft_*.py tests/fixtures/sft_traces/
git commit -m "feat(gemma-4): SFT module skeletons + test fixtures"
```

---

## Task 3: Dataset — Example dataclass, JSONL load, row validation

**Files:**
- Modify: `models/langslice-gemma-4/training/sft/dataset.py`
- Modify: `tests/test_sft_dataset.py`

- [ ] **Step 1: Write failing tests for valid-row loading**

Add to `tests/test_sft_dataset.py`:

```python
from pathlib import Path

import sys

import pytest

from sft.dataset import Example, load_examples, DatasetValidationError

FIXTURES = Path(__file__).resolve().parent / "fixtures" / "sft_traces"


def test_load_single_slice_minimal_returns_one_example() -> None:
    examples = load_examples(FIXTURES / "single_slice_minimal.jsonl")
    assert len(examples) == 1
    ex = examples[0]
    assert isinstance(ex, Example)
    assert ex.bucket == 1
    assert ex.atlas_name == "allen_mouse_25um"
    assert ex.system_prompt_kind == "single_slice"
    assert ex.query_image_paths == ["query.png"]
    assert ex.subject_id == "test_subj_01"
    assert len(ex.trace) == 2
    assert ex.trace[0]["tool_call"]["name"] == "fetch_atlas"
    assert ex.trace[1]["submit"]["name"] == "submit_estimate"
    assert ex.trace[1]["submit"]["args"]["reasoning"]


def test_malformed_examples_raise_validation_errors() -> None:
    with pytest.raises(DatasetValidationError) as exc:
        load_examples(FIXTURES / "malformed_examples.jsonl")
    msg = str(exc.value)
    assert "line 1" in msg or "1:" in msg  # first defect should be reported


def test_missing_image_path_is_rejected(tmp_path: Path) -> None:
    row = {
        "bucket": 1,
        "atlas_name": "allen_mouse_25um",
        "atlas_version": "CCFv3",
        "plane": "coronal",
        "subject_id": "x",
        "system_prompt_kind": "single_slice",
        "query_image_paths": ["missing.png"],
        "user_prompt_text": "x",
        "trace": [{"submit": {"name": "submit_estimate", "args": {"position_mm": 5.0, "reasoning": "x"}}}],
    }
    path = tmp_path / "missing_image.jsonl"
    path.write_text(json.dumps(row) + "\n", encoding="utf-8")
    with pytest.raises(DatasetValidationError, match="image not found"):
        load_examples(path)
```

- [ ] **Step 2: Run tests to verify they fail**

Run:
```bash
python -m pytest tests/test_sft_dataset.py -v
```

Expected: ImportError (Example, load_examples, DatasetValidationError don't exist yet).

- [ ] **Step 3: Implement Example dataclass + JSONL loader + validation**

Write to `models/langslice-gemma-4/training/sft/dataset.py`:

```python
"""Load + validate langslice-native SFT trace JSONL; subject-aware train/eval split."""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any


class DatasetValidationError(ValueError):
    """Raised when a JSONL row fails schema validation."""


@dataclass(frozen=True)
class Example:
    bucket: int
    atlas_name: str
    atlas_version: str
    plane: str
    subject_id: str
    system_prompt_kind: str            # always "single_slice" in v1
    query_image_paths: list[str]
    user_prompt_text: str
    trace: list[dict[str, Any]]        # langslice-native trace; see spec §6.1
    gemini_reasoning: str | None = None  # ignored by trainer in v1

    @property
    def dataset_root(self) -> Path | None:
        """Set after load by load_examples; used by renderer to resolve image paths."""
        return self._dataset_root  # type: ignore[attr-defined]


_REQUIRED_FIELDS = (
    "bucket", "atlas_name", "atlas_version", "plane", "subject_id",
    "system_prompt_kind", "query_image_paths", "user_prompt_text", "trace",
)
_VALID_KIND = "single_slice"
_SUBMIT_NAME = "submit_estimate"


def load_examples(jsonl_path: str | Path) -> list[Example]:
    """Load and validate every row in *jsonl_path*. Raise on the first defect."""
    path = Path(jsonl_path)
    root = path.parent
    examples: list[Example] = []
    with path.open("r", encoding="utf-8") as f:
        for lineno, raw in enumerate(f, start=1):
            line = raw.strip()
            if not line:
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError as e:
                raise DatasetValidationError(f"line {lineno}: invalid JSON ({e.msg})") from e
            row["_dataset_root"] = root
            _validate_row(row, lineno)
            ex = Example(
                bucket=row["bucket"],
                atlas_name=row["atlas_name"],
                atlas_version=row["atlas_version"],
                plane=row["plane"],
                subject_id=row["subject_id"],
                system_prompt_kind=row["system_prompt_kind"],
                query_image_paths=list(row["query_image_paths"]),
                user_prompt_text=row["user_prompt_text"],
                trace=list(row["trace"]),
                gemini_reasoning=row.get("gemini_reasoning"),
            )
            object.__setattr__(ex, "_dataset_root", root)
            examples.append(ex)
    if not examples:
        raise DatasetValidationError(f"{path}: no examples loaded (file is empty)")
    return examples


def _validate_row(row: dict[str, Any], lineno: int) -> None:
    for required in _REQUIRED_FIELDS:
        if required not in row:
            raise DatasetValidationError(f"line {lineno}: missing required field '{required}'")
    if row["bucket"] != 1:
        raise DatasetValidationError(
            f"line {lineno}: bucket must be 1 in v1 (got {row['bucket']})"
        )
    if not isinstance(row["subject_id"], str) or not row["subject_id"].strip():
        raise DatasetValidationError(f"line {lineno}: subject_id must be non-empty string")
    kind = row["system_prompt_kind"]
    if kind != _VALID_KIND:
        raise DatasetValidationError(
            f"line {lineno}: system_prompt_kind must be {_VALID_KIND!r} in v1 (got {kind!r})"
        )
    qips = row["query_image_paths"]
    if not isinstance(qips, list) or not qips:
        raise DatasetValidationError(f"line {lineno}: query_image_paths must be non-empty list")
    if len(qips) != 1:
        raise DatasetValidationError(
            f"line {lineno}: v1 single_slice requires exactly 1 query image (got {len(qips)})"
        )
    if "interval_mm" in row or "thickness_um" in row:
        raise DatasetValidationError(
            f"line {lineno}: group-only interval_mm/thickness_um are not accepted in v1"
        )
    trace = row["trace"]
    if not isinstance(trace, list) or not trace:
        raise DatasetValidationError(f"line {lineno}: trace must be non-empty list")
    last = trace[-1]
    if "submit" not in last:
        raise DatasetValidationError(f"line {lineno}: trace must end with a submit step")
    submit = last["submit"]
    submit_name = submit.get("name")
    if submit_name != _SUBMIT_NAME:
        raise DatasetValidationError(
            f"line {lineno}: submit.name must be {_SUBMIT_NAME!r} for v1 "
            f"(got {submit_name!r})"
        )
    args = submit.get("args", {})
    if not isinstance(args.get("position_mm"), (int, float)):
        raise DatasetValidationError(f"line {lineno}: submit.args.position_mm must be numeric")
    if not isinstance(args.get("reasoning"), str) or not args["reasoning"].strip():
        raise DatasetValidationError(f"line {lineno}: submit.args.reasoning must be non-empty string")
    for i, step in enumerate(trace[:-1]):
        if "tool_call" not in step or "tool_result" not in step:
            raise DatasetValidationError(
                f"line {lineno}: trace[{i}] must have both tool_call and tool_result"
            )
    _validate_image_paths(row, lineno)


def _validate_image_paths(row: dict[str, Any], lineno: int) -> None:
    root = row["_dataset_root"]
    for rel in row["query_image_paths"]:
        _require_existing_image(root, rel, lineno, "query_image_paths")
    for i, step in enumerate(row["trace"][:-1]):
        tool_result = step["tool_result"]
        image_paths = tool_result.get("image_paths", [])
        if not isinstance(image_paths, list) or not image_paths:
            raise DatasetValidationError(
                f"line {lineno}: trace[{i}].tool_result.image_paths must be non-empty list"
            )
        for rel in image_paths:
            _require_existing_image(root, rel, lineno, f"trace[{i}].tool_result.image_paths")


def _require_existing_image(root: Path, rel: Any, lineno: int, field: str) -> None:
    if not isinstance(rel, str) or not rel:
        raise DatasetValidationError(f"line {lineno}: {field} entries must be non-empty strings")
    path = (root / rel).resolve()
    if not path.is_file():
        raise DatasetValidationError(f"line {lineno}: {field} image not found: {path}")
```

- [ ] **Step 4: Run tests to verify they pass**

Run:
```bash
python -m pytest tests/test_sft_dataset.py -v
```

Expected: all four tests PASS.

- [ ] **Step 5: Commit**

```bash
git add models/langslice-gemma-4/training/sft/dataset.py tests/test_sft_dataset.py
git commit -m "feat(gemma-4): SFT dataset loader + row validation"
```

---

## Task 4: Dataset — subject-aware train/eval split

**Files:**
- Modify: `models/langslice-gemma-4/training/sft/dataset.py`
- Modify: `tests/test_sft_dataset.py`

- [ ] **Step 1: Write failing test for subject-aware split**

Append to `tests/test_sft_dataset.py`:

```python
from sft.dataset import split_subject_aware


def _make_examples(subject_ids: list[str]) -> list[Example]:
    """Tiny helper for split tests — uses the single_slice fixture's shape."""
    template_path = FIXTURES / "single_slice_minimal.jsonl"
    base = load_examples(template_path)[0]
    return [
        Example(
            bucket=base.bucket,
            atlas_name=base.atlas_name,
            atlas_version=base.atlas_version,
            plane=base.plane,
            subject_id=sid,
            system_prompt_kind=base.system_prompt_kind,
            query_image_paths=base.query_image_paths,
            user_prompt_text=base.user_prompt_text,
            trace=base.trace,
        )
        for sid in subject_ids
    ]


def test_split_subject_aware_no_subject_in_both_partitions() -> None:
    # 10 subjects, each contributing 3 examples. Holdout fraction 0.3.
    subjects = [f"subj_{i:02d}" for i in range(10)]
    examples = []
    for sid in subjects:
        examples.extend(_make_examples([sid] * 3))

    train, eval_ = split_subject_aware(examples, holdout_fraction=0.3, seed=0)

    train_subjects = {ex.subject_id for ex in train}
    eval_subjects = {ex.subject_id for ex in eval_}
    assert train_subjects.isdisjoint(eval_subjects), (
        f"subject leakage between partitions: "
        f"{train_subjects & eval_subjects}"
    )
    # Holdout fraction is approximate (subject-level, not row-level)
    assert 2 <= len(eval_subjects) <= 4
    # No examples lost
    assert len(train) + len(eval_) == len(examples)


def test_split_subject_aware_deterministic_with_seed() -> None:
    examples = _make_examples([f"subj_{i:02d}" for i in range(10)])
    train_a, eval_a = split_subject_aware(examples, holdout_fraction=0.3, seed=42)
    train_b, eval_b = split_subject_aware(examples, holdout_fraction=0.3, seed=42)
    assert [ex.subject_id for ex in train_a] == [ex.subject_id for ex in train_b]
    assert [ex.subject_id for ex in eval_a] == [ex.subject_id for ex in eval_b]
```

- [ ] **Step 2: Run tests to verify they fail**

Run:
```bash
python -m pytest tests/test_sft_dataset.py::test_split_subject_aware_no_subject_in_both_partitions tests/test_sft_dataset.py::test_split_subject_aware_deterministic_with_seed -v
```

Expected: ImportError (`split_subject_aware` doesn't exist).

- [ ] **Step 3: Implement subject-aware split**

Append to `models/langslice-gemma-4/training/sft/dataset.py`:

```python
import random


def split_subject_aware(
    examples: list[Example],
    *,
    holdout_fraction: float,
    seed: int,
) -> tuple[list[Example], list[Example]]:
    """Split *examples* into (train, eval) so no subject_id appears in both."""
    if not 0.0 < holdout_fraction < 1.0:
        raise ValueError(f"holdout_fraction must be in (0, 1), got {holdout_fraction}")

    by_subject: dict[str, list[Example]] = {}
    for ex in examples:
        by_subject.setdefault(ex.subject_id, []).append(ex)

    subjects = sorted(by_subject.keys())  # deterministic order before shuffle
    rng = random.Random(seed)
    rng.shuffle(subjects)

    n_holdout = max(1, round(len(subjects) * holdout_fraction))
    if n_holdout >= len(subjects):
        raise ValueError(
            f"holdout_fraction={holdout_fraction} would consume all "
            f"{len(subjects)} subjects; need at least one for train"
        )
    eval_subjects = set(subjects[:n_holdout])

    train: list[Example] = []
    eval_: list[Example] = []
    for ex in examples:
        if ex.subject_id in eval_subjects:
            eval_.append(ex)
        else:
            train.append(ex)
    return train, eval_
```

- [ ] **Step 4: Run tests to verify they pass**

Run:
```bash
python -m pytest tests/test_sft_dataset.py -v
```

Expected: all five tests PASS.

- [ ] **Step 5: Commit**

```bash
git add models/langslice-gemma-4/training/sft/dataset.py tests/test_sft_dataset.py
git commit -m "feat(gemma-4): SFT subject-aware train/eval split"
```

---

## Task 5: Renderer — atlas metadata cache + tools schema + system prompt

**Files:**
- Modify: `models/langslice-gemma-4/training/sft/render.py`
- Modify: `tests/test_sft_render.py`

- [ ] **Step 1: Write failing tests for atlas metadata cache + tools schema**

Add to `tests/test_sft_render.py`:

```python
import pytest

from sft.render import (
    AtlasMetaCache,
    build_system_prompt,
    build_tools_schema,
)


def test_tools_schema_single_slice_has_fetch_atlas_and_submit_estimate() -> None:
    tools = build_tools_schema("single_slice")
    names = [t["function"]["name"] for t in tools]
    assert "fetch_atlas" in names
    assert "submit_estimate" in names
    assert "submit_group_estimate" not in names


def test_tools_schema_function_shape_matches_hf_format() -> None:
    tools = build_tools_schema("single_slice")
    fetch = next(t for t in tools if t["function"]["name"] == "fetch_atlas")
    assert fetch["type"] == "function"
    assert "description" in fetch["function"]
    assert "parameters" in fetch["function"]
    params = fetch["function"]["parameters"]
    assert params["type"] == "object"
    assert "positions_mm" in params["properties"]


def test_submit_estimate_schema_requires_position_and_reasoning() -> None:
    tools = build_tools_schema("single_slice")
    submit = next(t for t in tools if t["function"]["name"] == "submit_estimate")
    params = submit["function"]["parameters"]
    assert set(params["required"]) == {"position_mm", "reasoning"}
    assert "reasoning" in params["properties"]


def test_atlas_meta_cache_returns_same_instance_for_same_atlas() -> None:
    cache = AtlasMetaCache()
    a = cache.get("allen_mouse_25um", "coronal")
    b = cache.get("allen_mouse_25um", "coronal")
    assert a is b  # identity, not just equality


def test_build_system_prompt_single_slice_uses_production_prompt() -> None:
    cache = AtlasMetaCache()
    prompt = build_system_prompt(
        kind="single_slice",
        atlas_name="allen_mouse_25um",
        plane="coronal",
        atlas_meta_cache=cache,
    )
    # Production prompt mentions "AP" axis label for coronal
    assert "AP" in prompt
    assert "allen_mouse_25um" in prompt
```

- [ ] **Step 2: Run tests to verify they fail**

Run:
```bash
python -m pytest tests/test_sft_render.py -v
```

Expected: ImportError on `AtlasMetaCache`, `build_system_prompt`, `build_tools_schema`.

- [ ] **Step 3: Implement atlas meta cache + tools + prompt builders**

Write to `models/langslice-gemma-4/training/sft/render.py`:

```python
"""Translate langslice-native trace examples to HF chat-template messages + tools."""

from __future__ import annotations

import json
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

# langslice_harness lives at <repo>/src — make it importable when this module
# is run from inside the gemma-4 training directory.
_REPO_SRC = Path(__file__).resolve().parents[4] / "src"
if str(_REPO_SRC) not in sys.path:
    sys.path.insert(0, str(_REPO_SRC))

from langslice_harness.atlas.core import get_position_range_mm, load_atlas
from langslice_harness.atlas.space import Plane
from langslice_harness.harness.estimation.prompts import build_single_slice_prompt


@dataclass(frozen=True)
class AtlasMeta:
    pos_lo: float
    pos_hi: float
    species: str  # human-readable, e.g. "mouse" / "rat" / "developmental mouse"


_SPECIES_BY_ATLAS_PREFIX: dict[str, str] = {
    "allen_mouse_": "mouse",
    "whs_sd_rat_": "rat",
    "admba_": "developmental mouse",
}


def _infer_species(atlas_name: str) -> str:
    for prefix, species in _SPECIES_BY_ATLAS_PREFIX.items():
        if atlas_name.startswith(prefix):
            return species
    return "unknown"


class AtlasMetaCache:
    """Memoized (atlas_name, plane) -> AtlasMeta lookup. Avoids reloading volumes."""

    def __init__(self) -> None:
        self._cache: dict[tuple[str, str], AtlasMeta] = {}

    def get(self, atlas_name: str, plane: str) -> AtlasMeta:
        key = (atlas_name, plane)
        if key in self._cache:
            return self._cache[key]
        atlas = load_atlas(atlas_name)
        pos_lo, pos_hi = get_position_range_mm(atlas, plane)
        meta = AtlasMeta(
            pos_lo=float(pos_lo),
            pos_hi=float(pos_hi),
            species=_infer_species(atlas_name),
        )
        self._cache[key] = meta
        return meta


def build_system_prompt(
    *,
    kind: str,
    atlas_name: str,
    plane: str,
    atlas_meta_cache: AtlasMetaCache,
) -> str:
    """Build the system prompt by delegating to the production builders."""
    meta = atlas_meta_cache.get(atlas_name, plane)
    plane_typed: Plane = plane  # type: ignore[assignment]  # Plane is a Literal alias
    if kind == "single_slice":
        return build_single_slice_prompt(
            atlas_name=atlas_name,
            plane=plane_typed,
            pos_lo=meta.pos_lo,
            pos_hi=meta.pos_hi,
            species=meta.species,
        )
    raise ValueError(f"unknown system_prompt_kind: {kind!r}")


_FETCH_ATLAS_TOOL: dict[str, Any] = {
    "type": "function",
    "function": {
        "name": "fetch_atlas",
        "description": (
            "Fetch atlas reference slices at the given positions (in mm). "
            "Returns up to 8 atlas images per call."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "positions_mm": {
                    "type": "array",
                    "items": {"type": "number"},
                    "description": "Positions in mm at which to render atlas slices.",
                }
            },
            "required": ["positions_mm"],
        },
    },
}

_SUBMIT_ESTIMATE_TOOL: dict[str, Any] = {
    "type": "function",
    "function": {
        "name": "submit_estimate",
        "description": "Submit the final position estimate for the query slice.",
        "parameters": {
            "type": "object",
            "properties": {
                "position_mm": {"type": "number"},
                "reasoning": {"type": "string"},
            },
            "required": ["position_mm", "reasoning"],
        },
    },
}


def build_tools_schema(kind: str) -> list[dict[str, Any]]:
    """Return the HF-format function-schema list for the given kind."""
    if kind == "single_slice":
        return [_FETCH_ATLAS_TOOL, _SUBMIT_ESTIMATE_TOOL]
    raise ValueError(f"unknown system_prompt_kind: {kind!r}")
```

- [ ] **Step 4: Run tests to verify they pass**

Run:
```bash
python -m pytest tests/test_sft_render.py -v
```

Expected: all five tests PASS. **If `test_atlas_meta_cache_returns_same_instance_for_same_atlas` fails** because `load_atlas` requires an actual atlas download, mark that test with `@pytest.mark.skipif(not Path("~/.brainglobe/allen_mouse_25um").expanduser().exists(), reason="atlas not downloaded locally")` and document the skip in the task notes — it'll be re-enabled in CI when atlases are available.

- [ ] **Step 5: Commit**

```bash
git add models/langslice-gemma-4/training/sft/render.py tests/test_sft_render.py
git commit -m "feat(gemma-4): SFT renderer atlas-meta cache + tools/prompt builders"
```

---

## Task 6: Renderer — image hydration helper + trace translation

**Files:**
- Modify: `models/langslice-gemma-4/training/sft/render.py`
- Modify: `tests/test_sft_render.py`

- [ ] **Step 1: Write failing test for full render translation**

Append to `tests/test_sft_render.py`:

```python
from PIL import Image

from sft.dataset import load_examples
from sft.render import render_example, RenderedExample


def _write_dummy_png(path: Path, color: tuple[int, int, int] = (128, 128, 128)) -> None:
    Image.new("RGB", (32, 32), color=color).save(path)


def test_render_single_slice_minimal(tmp_path: Path) -> None:
    # Stage the fixture trace alongside dummy images in tmp_path
    src = FIXTURES / "single_slice_minimal.jsonl"
    dest_jsonl = tmp_path / "single_slice_minimal.jsonl"
    dest_jsonl.write_text(src.read_text(encoding="utf-8"), encoding="utf-8")
    for name in ("query.png", "a3.png", "a5.png", "a7.png"):
        _write_dummy_png(tmp_path / name)

    examples = load_examples(dest_jsonl)
    cache = AtlasMetaCache()
    rendered = render_example(examples[0], atlas_meta_cache=cache)

    assert isinstance(rendered, RenderedExample)
    msgs = rendered.messages
    # system + user + (assistant + tool) + (assistant final submit)
    assert msgs[0]["role"] == "system"
    assert msgs[1]["role"] == "user"
    # User turn has 1 image + 1 text content block
    user_content = msgs[1]["content"]
    assert sum(1 for c in user_content if c.get("type") == "image") == 1
    assert sum(1 for c in user_content if c.get("type") == "text") == 1
    # Find the first assistant turn — must carry tool_calls with fetch_atlas
    first_assistant = next(m for m in msgs[2:] if m["role"] == "assistant")
    assert first_assistant["tool_calls"][0]["function"]["name"] == "fetch_atlas"
    args = json.loads(first_assistant["tool_calls"][0]["function"]["arguments"])
    assert args["positions_mm"] == [3.0, 5.0, 7.0]
    # Tool message must have matching tool_call_id and 3 images + 1 text
    first_tool = msgs[msgs.index(first_assistant) + 1]
    assert first_tool["role"] == "tool"
    assert first_tool["tool_call_id"] == first_assistant["tool_calls"][0]["id"]
    assert sum(1 for c in first_tool["content"] if c.get("type") == "image") == 3
    # Final assistant turn has the submit
    final_assistant = msgs[-1]
    assert final_assistant["role"] == "assistant"
    fn = final_assistant["tool_calls"][0]["function"]
    assert fn["name"] == "submit_estimate"
    submit_args = json.loads(fn["arguments"])
    assert submit_args["position_mm"] == pytest.approx(5.2)
    assert submit_args["reasoning"]


def test_render_unique_tool_call_ids(tmp_path: Path) -> None:
    """Every assistant tool_call.id must be unique within the trace."""
    src = FIXTURES / "single_slice_minimal.jsonl"
    dest_jsonl = tmp_path / "single_slice_minimal.jsonl"
    dest_jsonl.write_text(src.read_text(encoding="utf-8"), encoding="utf-8")
    for name in ("query.png", "a3.png", "a5.png", "a7.png"):
        _write_dummy_png(tmp_path / name)

    rendered = render_example(load_examples(dest_jsonl)[0], atlas_meta_cache=AtlasMetaCache())
    ids = []
    for m in rendered.messages:
        if m["role"] == "assistant":
            for tc in m.get("tool_calls", []):
                ids.append(tc["id"])
    assert len(ids) == len(set(ids))
```

- [ ] **Step 2: Run tests to verify they fail**

Run:
```bash
python -m pytest tests/test_sft_render.py::test_render_single_slice_minimal tests/test_sft_render.py::test_render_unique_tool_call_ids -v
```

Expected: ImportError on `render_example`, `RenderedExample`.

- [ ] **Step 3: Implement render_example**

Append to `models/langslice-gemma-4/training/sft/render.py`:

```python
from PIL import Image

from .dataset import Example


@dataclass
class RenderedExample:
    """Output of the renderer — ready for processor.apply_chat_template(...)."""
    messages: list[dict[str, Any]]
    tools: list[dict[str, Any]]
    metadata: dict[str, Any]


def _hydrate_image(rel_path: str, root: Path) -> Image.Image:
    abs_path = (root / rel_path).resolve()
    if not abs_path.is_file():
        raise FileNotFoundError(f"image not found: {abs_path}")
    return Image.open(abs_path).convert("RGB")


def _user_turn(query_image_paths: list[str], user_text: str, root: Path) -> dict[str, Any]:
    """Image-before-text per Gemma 4 chat-template rule."""
    content: list[dict[str, Any]] = []
    for p in query_image_paths:
        content.append({"type": "image", "image": _hydrate_image(p, root)})
    content.append({"type": "text", "text": user_text})
    return {"role": "user", "content": content}


def _assistant_tool_call(call_id: str, name: str, args: dict[str, Any]) -> dict[str, Any]:
    return {
        "role": "assistant",
        "tool_calls": [
            {
                "id": call_id,
                "type": "function",
                "function": {
                    "name": name,
                    "arguments": json.dumps(args, separators=(",", ":")),
                },
            }
        ],
    }


def _tool_response(call_id: str, image_paths: list[str], text: str, root: Path) -> dict[str, Any]:
    content: list[dict[str, Any]] = []
    for p in image_paths:
        content.append({"type": "image", "image": _hydrate_image(p, root)})
    content.append({"type": "text", "text": text})
    return {"role": "tool", "tool_call_id": call_id, "content": content}


def render_example(
    example: Example,
    *,
    atlas_meta_cache: AtlasMetaCache,
) -> RenderedExample:
    """Translate a langslice-native Example to HF chat-template messages + tools."""
    root = example.dataset_root
    if root is None:
        raise ValueError("Example.dataset_root not set; load via load_examples()")

    system_prompt = build_system_prompt(
        kind=example.system_prompt_kind,
        atlas_name=example.atlas_name,
        plane=example.plane,
        atlas_meta_cache=atlas_meta_cache,
    )
    tools = build_tools_schema(example.system_prompt_kind)

    messages: list[dict[str, Any]] = [
        {"role": "system", "content": system_prompt},
        _user_turn(example.query_image_paths, example.user_prompt_text, root),
    ]

    seen_ids: set[str] = set()
    for i, step in enumerate(example.trace):
        if "submit" in step:
            call_id = f"call_final_{i}"
            assert call_id not in seen_ids
            seen_ids.add(call_id)
            messages.append(_assistant_tool_call(
                call_id, step["submit"]["name"], step["submit"]["args"]
            ))
            # No matching tool message for the terminal submit.
            continue
        # tool_call + tool_result pair
        call_id = f"call_{i}"
        assert call_id not in seen_ids
        seen_ids.add(call_id)
        tc = step["tool_call"]
        tr = step["tool_result"]
        messages.append(_assistant_tool_call(call_id, tc["name"], tc["args"]))
        messages.append(_tool_response(call_id, tr["image_paths"], tr["text"], root))

    metadata = {
        "atlas_name": example.atlas_name,
        "atlas_version": example.atlas_version,
        "plane": example.plane,
        "subject_id": example.subject_id,
        "system_prompt_kind": example.system_prompt_kind,
    }
    return RenderedExample(messages=messages, tools=tools, metadata=metadata)
```

- [ ] **Step 4: Run tests to verify they pass**

Run:
```bash
python -m pytest tests/test_sft_render.py -v
```

Expected: all seven tests PASS (five from Task 5 + two new).

- [ ] **Step 5: Commit**

```bash
git add models/langslice-gemma-4/training/sft/render.py tests/test_sft_render.py
git commit -m "feat(gemma-4): SFT renderer trace-to-messages translation"
```

---

## Task 7: Collator — processor invocation + assistant-only labels mask

**Files:**
- Modify: `models/langslice-gemma-4/training/sft/collate.py`
- Modify: `tests/test_sft_collate.py`

- [ ] **Step 1: Write failing test using a real Gemma 4 processor**

Add to `tests/test_sft_collate.py`:

```python
from pathlib import Path

import pytest
import torch

from sft.collate import LangSliceCollator
from sft.dataset import load_examples
from sft.render import AtlasMetaCache, render_example

FIXTURES = Path(__file__).resolve().parent / "fixtures" / "sft_traces"


def _has_gemma4() -> bool:
    """Check whether the Gemma 4 processor is locally available."""
    try:
        from transformers import AutoProcessor
        AutoProcessor.from_pretrained("unsloth/gemma-4-E4B-it", trust_remote_code=False)
        return True
    except Exception:
        return False


pytestmark = pytest.mark.skipif(
    not _has_gemma4(),
    reason="Gemma 4 processor not available locally; run after Task 1 verification "
            "downloads it via Unsloth.",
)


@pytest.fixture(scope="module")
def processor():
    from transformers import AutoProcessor
    return AutoProcessor.from_pretrained("unsloth/gemma-4-E4B-it", trust_remote_code=False)


@pytest.fixture
def rendered_single_slice(tmp_path):
    """Stage the single-slice fixture with dummy images for the renderer."""
    from PIL import Image
    src = FIXTURES / "single_slice_minimal.jsonl"
    dest = tmp_path / "single_slice_minimal.jsonl"
    dest.write_text(src.read_text(encoding="utf-8"), encoding="utf-8")
    for name in ("query.png", "a3.png", "a5.png", "a7.png"):
        Image.new("RGB", (224, 224), color=(128, 128, 128)).save(tmp_path / name)
    return render_example(load_examples(dest)[0], atlas_meta_cache=AtlasMetaCache())


def test_collate_labels_match_input_ids_on_assistant_tokens(processor, rendered_single_slice):
    collator = LangSliceCollator(processor=processor, max_seq_length=4096)
    batch = collator([rendered_single_slice])
    input_ids = batch["input_ids"]
    labels = batch["labels"]
    assert input_ids.shape == labels.shape
    # Where labels != -100, they must equal input_ids
    keep_mask = labels != -100
    assert torch.equal(labels[keep_mask], input_ids[keep_mask])


def test_collate_labels_minus_100_outside_assistant(processor, rendered_single_slice):
    collator = LangSliceCollator(processor=processor, max_seq_length=4096)
    batch = collator([rendered_single_slice])
    labels = batch["labels"][0]
    # At least one token should be -100 (system + user content)
    assert (labels == -100).sum().item() > 0
    # At least one token should not be -100 (assistant content)
    assert (labels != -100).sum().item() > 0
    # The fraction of kept tokens should be small relative to total
    keep_fraction = (labels != -100).float().mean().item()
    assert keep_fraction < 0.5, f"unexpectedly high keep fraction: {keep_fraction}"
```

- [ ] **Step 2: Run tests to verify they fail or skip**

Run:
```bash
python -m pytest tests/test_sft_collate.py -v
```

Expected: tests skip if processor unavailable, otherwise fail with `ImportError: LangSliceCollator`.

- [ ] **Step 3: Implement LangSliceCollator (happy path)**

Write to `models/langslice-gemma-4/training/sft/collate.py`:

```python
"""Apply processor.apply_chat_template and build labels with -100 outside assistant turns."""

from __future__ import annotations

from typing import Any

import torch

from .render import RenderedExample


class LangSliceCollator:
    """Builds a TRL-compatible batch from RenderedExample objects.

    The processor's chat template is the source of truth for the assistant-token
    mask. Labels are constructed by cloning input_ids and zeroing (with -100)
    every position where the assistant_mask is False.
    """

    def __init__(self, *, processor: Any, max_seq_length: int) -> None:
        self.processor = processor
        self.max_seq_length = max_seq_length

    def __call__(self, examples: list[RenderedExample]) -> dict[str, torch.Tensor]:
        # Apply chat template per-example (not as a batch) so the per-example
        # assistant_mask aligns 1:1 with that example's input_ids.
        per_example: list[dict[str, torch.Tensor]] = []
        for ex in examples:
            out = self.processor.apply_chat_template(
                ex.messages,
                tools=ex.tools,
                chat_template_kwargs={"enable_thinking": False},
                add_generation_prompt=False,
                tokenize=True,
                return_assistant_tokens_mask=True,
                return_dict=True,
                return_tensors="pt",
            )
            ids = out["input_ids"]
            if ids.shape[1] > self.max_seq_length:
                raise ValueError(
                    f"rendered example exceeds max_seq_length="
                    f"{self.max_seq_length} (got {ids.shape[1]} tokens). "
                    f"subject_id={ex.metadata.get('subject_id')!r}"
                )
            assistant_mask = out["assistant_masks"]  # 1 where assistant, 0 elsewhere
            labels = ids.clone()
            labels[assistant_mask == 0] = -100
            per_example.append({
                "input_ids": ids[0],
                "attention_mask": out["attention_mask"][0],
                "labels": labels[0],
                # Pixel values + image grid passed through verbatim
                **{k: v for k, v in out.items()
                   if k not in ("input_ids", "attention_mask", "assistant_masks")},
            })

        # Pad to the longest example in the batch
        return _pad_batch(per_example, pad_token_id=self.processor.tokenizer.pad_token_id)


def _pad_batch(
    per_example: list[dict[str, Any]],
    *,
    pad_token_id: int,
) -> dict[str, torch.Tensor]:
    max_len = max(ex["input_ids"].shape[0] for ex in per_example)
    out: dict[str, torch.Tensor] = {}
    keys_with_seq_dim = ("input_ids", "attention_mask", "labels")
    for k in keys_with_seq_dim:
        padded = []
        for ex in per_example:
            t = ex[k]
            pad_len = max_len - t.shape[0]
            if pad_len > 0:
                pad_value = pad_token_id if k == "input_ids" else (0 if k == "attention_mask" else -100)
                padding = torch.full((pad_len,), pad_value, dtype=t.dtype)
                t = torch.cat([t, padding], dim=0)
            padded.append(t)
        out[k] = torch.stack(padded, dim=0)

    # Image-related tensors: stack along batch dim if present in all examples
    image_keys = set()
    for ex in per_example:
        image_keys.update(k for k in ex if k not in keys_with_seq_dim and isinstance(ex[k], torch.Tensor))
    for k in image_keys:
        try:
            out[k] = torch.stack([ex[k] if ex[k].dim() == per_example[0][k].dim() else ex[k].squeeze(0)
                                  for ex in per_example], dim=0)
        except RuntimeError:
            # If image tensors have variable shapes, leave them as a list — TRL
            # vision collators handle this case downstream.
            out[k] = [ex[k] for ex in per_example]  # type: ignore[assignment]
    return out
```

- [ ] **Step 4: Run tests to verify they pass**

Run:
```bash
python -m pytest tests/test_sft_collate.py -v
```

Expected: PASS (or SKIP if processor unavailable). If FAIL, the most likely cause is that `assistant_masks` is not returned by the Gemma 4 chat template — Task 8 builds the manual-span fallback for that case.

- [ ] **Step 5: Commit**

```bash
git add models/langslice-gemma-4/training/sft/collate.py tests/test_sft_collate.py
git commit -m "feat(gemma-4): SFT collator with assistant-only labels mask"
```

---

## Task 8: Collator — image-token sanity check + manual-span fallback

**Files:**
- Modify: `models/langslice-gemma-4/training/sft/collate.py`
- Modify: `tests/test_sft_collate.py`

- [ ] **Step 1: Write failing test for image-token sanity check**

Append to `tests/test_sft_collate.py`:

```python
def test_collate_image_token_sanity_check(processor, rendered_single_slice):
    """No labels position should fall on an image-placeholder token ID."""
    collator = LangSliceCollator(processor=processor, max_seq_length=4096)
    batch = collator([rendered_single_slice])
    labels = batch["labels"][0]
    input_ids = batch["input_ids"][0]
    image_token_id = processor.tokenizer.convert_tokens_to_ids("<image_soft_token>")
    if image_token_id is None or image_token_id < 0:
        pytest.skip("Gemma 4 image-placeholder token not exposed under that name")
    keep_positions = (labels != -100).nonzero(as_tuple=True)[0]
    for pos in keep_positions:
        assert input_ids[pos].item() != image_token_id, (
            f"labels[{pos}] is {input_ids[pos].item()} which is the image token id"
        )


def test_collate_falls_back_to_manual_span_when_no_assistant_mask(monkeypatch, processor, rendered_single_slice):
    """If processor doesn't return assistant_masks, collator builds it manually."""
    # Force the processor's apply_chat_template to omit assistant_masks
    real = processor.apply_chat_template

    def fake_apply(*args, **kwargs):
        out = real(*args, **kwargs)
        if "assistant_masks" in out:
            del out["assistant_masks"]
        return out

    monkeypatch.setattr(processor, "apply_chat_template", fake_apply)
    collator = LangSliceCollator(processor=processor, max_seq_length=4096)
    batch = collator([rendered_single_slice])
    # Fallback should still produce some non-masked positions
    assert (batch["labels"] != -100).sum().item() > 0
```

- [ ] **Step 2: Run tests to verify they fail**

Run:
```bash
python -m pytest tests/test_sft_collate.py::test_collate_image_token_sanity_check tests/test_sft_collate.py::test_collate_falls_back_to_manual_span_when_no_assistant_mask -v
```

Expected: PASS for the sanity test (already implemented). FAIL for the fallback test — `LangSliceCollator` raises `KeyError` on `assistant_masks`.

- [ ] **Step 3: Add fallback logic to collator**

Modify `models/langslice-gemma-4/training/sft/collate.py` — replace the `__call__` body around the `apply_chat_template` call:

```python
    def __call__(self, examples: list[RenderedExample]) -> dict[str, torch.Tensor]:
        per_example: list[dict[str, torch.Tensor]] = []
        for ex in examples:
            out = self.processor.apply_chat_template(
                ex.messages,
                tools=ex.tools,
                chat_template_kwargs={"enable_thinking": False},
                add_generation_prompt=False,
                tokenize=True,
                return_assistant_tokens_mask=True,
                return_dict=True,
                return_tensors="pt",
            )
            ids = out["input_ids"]
            if ids.shape[1] > self.max_seq_length:
                raise ValueError(
                    f"rendered example exceeds max_seq_length="
                    f"{self.max_seq_length} (got {ids.shape[1]} tokens). "
                    f"subject_id={ex.metadata.get('subject_id')!r}"
                )

            if "assistant_masks" in out:
                assistant_mask = out["assistant_masks"]
            else:
                # Fallback: re-tokenize each assistant turn separately and
                # find their token spans in the full sequence.
                assistant_mask = self._manual_span_mask(ex, ids[0])

            labels = ids.clone()
            labels[assistant_mask == 0] = -100

            # Sanity check: assistant tokens never overlap image tokens.
            self._sanity_check_no_image_tokens_in_labels(ids[0], labels[0])

            per_example.append({
                "input_ids": ids[0],
                "attention_mask": out["attention_mask"][0],
                "labels": labels[0],
                **{k: v for k, v in out.items()
                   if k not in ("input_ids", "attention_mask", "assistant_masks")},
            })

        return _pad_batch(per_example, pad_token_id=self.processor.tokenizer.pad_token_id)

    def _manual_span_mask(self, example: RenderedExample, input_ids: torch.Tensor) -> torch.Tensor:
        """Build a per-token assistant mask by re-tokenizing each assistant turn."""
        mask = torch.zeros_like(input_ids)
        for msg in example.messages:
            if msg["role"] != "assistant":
                continue
            # Render just this assistant turn and find its span in input_ids
            sub = self.processor.apply_chat_template(
                [msg],
                tools=example.tools,
                chat_template_kwargs={"enable_thinking": False},
                add_generation_prompt=False,
                tokenize=True,
                return_dict=True,
                return_tensors="pt",
            )
            sub_ids = sub["input_ids"][0]
            # Linear search for the sub-sequence (cheap; assistant turns are short)
            n = sub_ids.shape[0]
            for start in range(0, input_ids.shape[0] - n + 1):
                if torch.equal(input_ids[start:start + n], sub_ids):
                    mask[start:start + n] = 1
                    break
        return mask.unsqueeze(0)

    def _sanity_check_no_image_tokens_in_labels(self, ids: torch.Tensor, labels: torch.Tensor) -> None:
        """Optional safety net — disabled if the processor doesn't expose image-token IDs."""
        candidate_names = ("<image_soft_token>", "<image>", "<|image|>")
        for name in candidate_names:
            tok_id = self.processor.tokenizer.convert_tokens_to_ids(name)
            if tok_id is not None and tok_id >= 0:
                bad = ((labels != -100) & (ids == tok_id)).any().item()
                if bad:
                    raise RuntimeError(
                        f"labels mask is keeping image-token positions (token id={tok_id}, "
                        f"name={name}); the chat template's assistant-mask logic is wrong "
                        f"for this trace shape — investigate before training"
                    )
                return
```

- [ ] **Step 4: Run tests to verify they pass**

Run:
```bash
python -m pytest tests/test_sft_collate.py -v
```

Expected: all four tests PASS (or SKIP if processor unavailable).

- [ ] **Step 5: Commit**

```bash
git add models/langslice-gemma-4/training/sft/collate.py tests/test_sft_collate.py
git commit -m "feat(gemma-4): SFT collator manual-span fallback + image-token sanity"
```

---

## Task 9: Eval — metric utilities

**Files:**
- Modify: `models/langslice-gemma-4/training/sft/eval.py`
- Modify: `tests/test_sft_eval.py`

- [ ] **Step 1: Write failing tests for metric computation**

Add to `tests/test_sft_eval.py`:

```python
import pytest

from sft.eval import (
    compute_position_mae_mm,
    parse_submit_call,
    summarize_eval_runs,
    EvalRun,
)


def test_parse_submit_call_extracts_position_from_valid_json():
    raw = '{"position_mm": 5.2, "reasoning": "looks like AC level"}'
    parsed = parse_submit_call(raw, expected_kind="single_slice")
    assert parsed.position_mm == pytest.approx(5.2)
    assert parsed.is_parseable is True


def test_parse_submit_call_handles_malformed_json():
    parsed = parse_submit_call("{not json}", expected_kind="single_slice")
    assert parsed.is_parseable is False
    assert parsed.position_mm is None


def test_compute_position_mae_mm_simple():
    pred = [1.0, 2.0, 3.0]
    truth = [1.5, 2.0, 4.0]
    mae = compute_position_mae_mm(pred, truth)
    # Mean of |0.5|, |0|, |1.0| = 0.5
    assert mae == pytest.approx(0.5)


def test_summarize_eval_runs():
    runs = [
        EvalRun(subject_id="M01", predicted_mm=[5.0], truth_mm=[5.5], parseable=True, n_turns=4),
        EvalRun(subject_id="M02", predicted_mm=[7.2], truth_mm=[7.0], parseable=True, n_turns=6),
        EvalRun(subject_id="M03", predicted_mm=None, truth_mm=[3.0], parseable=False, n_turns=12),
    ]
    summary = summarize_eval_runs(runs)
    # Only the 2 parseable runs contribute to MAE
    assert summary["position_mae_mm"] == pytest.approx((0.5 + 0.2) / 2)
    assert summary["tool_call_parseability_rate"] == pytest.approx(2 / 3)
    assert summary["no_submit_rate"] == pytest.approx(1 / 3)
    assert summary["mean_trace_length"] == pytest.approx((4 + 6 + 12) / 3)
```

- [ ] **Step 2: Run tests to verify they fail**

Run:
```bash
python -m pytest tests/test_sft_eval.py -v
```

Expected: ImportError on `compute_position_mae_mm`, `parse_submit_call`, `summarize_eval_runs`, `EvalRun`.

- [ ] **Step 3: Implement metric utilities**

Write to `models/langslice-gemma-4/training/sft/eval.py`:

```python
"""SFT-time evaluation: agent-loop callbacks (baseline + periodic) and metric utilities."""

from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True)
class ParsedSubmit:
    is_parseable: bool
    position_mm: float | None = None


def parse_submit_call(arguments_json: str, *, expected_kind: str) -> ParsedSubmit:
    """Parse a submit_*_estimate call's arguments string. Returns is_parseable=False on any failure."""
    try:
        args = json.loads(arguments_json)
    except (json.JSONDecodeError, TypeError):
        return ParsedSubmit(is_parseable=False)
    if expected_kind == "single_slice":
        pos = args.get("position_mm")
        reasoning = args.get("reasoning")
        if not isinstance(pos, (int, float)):
            return ParsedSubmit(is_parseable=False)
        if not isinstance(reasoning, str) or not reasoning.strip():
            return ParsedSubmit(is_parseable=False)
        return ParsedSubmit(is_parseable=True, position_mm=float(pos))
    raise ValueError(f"unknown expected_kind: {expected_kind!r}")


def compute_position_mae_mm(predicted: list[float], truth: list[float]) -> float:
    """Mean absolute error in mm. predicted and truth must be the same length."""
    if len(predicted) != len(truth):
        raise ValueError(f"length mismatch: {len(predicted)} vs {len(truth)}")
    if not predicted:
        raise ValueError("empty prediction list")
    errors = [abs(p - t) for p, t in zip(predicted, truth)]
    return sum(errors) / len(errors)


@dataclass
class EvalRun:
    subject_id: str
    predicted_mm: list[float] | None  # None if not parseable / no submit
    truth_mm: list[float]
    parseable: bool
    n_turns: int


def summarize_eval_runs(runs: list[EvalRun]) -> dict[str, float]:
    """Aggregate metrics over a held-out eval run."""
    if not runs:
        raise ValueError("no eval runs to summarize")
    parseable_runs = [r for r in runs if r.parseable and r.predicted_mm is not None]
    submit_runs = [r for r in runs if r.predicted_mm is not None]
    if parseable_runs:
        per_run_maes = [
            compute_position_mae_mm(r.predicted_mm, r.truth_mm)  # type: ignore[arg-type]
            for r in parseable_runs
        ]
        position_mae = sum(per_run_maes) / len(per_run_maes)
    else:
        position_mae = float("nan")
    return {
        "position_mae_mm": position_mae,
        "tool_call_parseability_rate": len(parseable_runs) / len(runs),
        "no_submit_rate": 1.0 - (len(submit_runs) / len(runs)),
        "mean_trace_length": sum(r.n_turns for r in runs) / len(runs),
    }
```

- [ ] **Step 4: Run tests to verify they pass**

Run:
```bash
python -m pytest tests/test_sft_eval.py -v
```

Expected: all four tests PASS.

- [ ] **Step 5: Commit**

```bash
git add models/langslice-gemma-4/training/sft/eval.py tests/test_sft_eval.py
git commit -m "feat(gemma-4): SFT eval metric utilities"
```

---

## Task 10: Eval — agent-loop callbacks (baseline + periodic)

**Files:**
- Modify: `models/langslice-gemma-4/training/sft/eval.py`
- Modify: `tests/test_sft_eval.py`

The agent-loop callbacks reuse `models/langslice-gemma-4/training/rlvr/env.py`'s `LangSliceEstimateEnv` to execute tool calls — same env that RLVR uses for rollouts. This avoids duplicating the tool-call loop.

- [ ] **Step 1: Inspect LangSliceEstimateEnv interface**

Run:
```bash
python -c "from pathlib import Path; print(Path('models/langslice-gemma-4/training/rlvr/env.py').read_text()[:2000])"
```

Confirm:
- Constructor signature for instantiation in eval.
- Methods used by the eval loop: `reset(...)`, the public tool methods, and however the env signals `done`.

The current env accepts extra row kwargs but requires `atlas_name`, `plane`, `valid_range_mm`, `ground_truth_positions_mm`, and `kind`. For v1 SFT eval, call it with `kind="single"` and `valid_range_mm=(pos_lo, pos_hi)`.

- [ ] **Step 2: Write a callback class skeleton + skipped integration test**

Append to `tests/test_sft_eval.py`:

```python
def test_agent_loop_eval_callback_signature_compiles():
    """Smoke check that the callback class is importable + has the expected attrs."""
    from sft.eval import AgentLoopEvalCallback, BaselineEvalCallback
    assert hasattr(AgentLoopEvalCallback, "on_step_end")
    assert hasattr(BaselineEvalCallback, "on_train_begin")


def test_run_agent_loop_for_one_uses_rlvr_env_single_slice(monkeypatch, tmp_path):
    """Stub model/processor path verifies env.reset/fetch wiring without loading Gemma."""
    from PIL import Image
    from sft import eval as eval_mod

    image_path = tmp_path / "query.png"
    Image.new("RGB", (32, 32), (128, 128, 128)).save(image_path)

    class StubCache:
        def get(self, atlas_name, plane):  # noqa: ANN001
            return type("Meta", (), {"pos_lo": 0.0, "pos_hi": 13.2, "species": "mouse"})()

    class StubEnv:
        reset_kwargs = None

        def __init__(self, atlas_grid):  # noqa: ANN001
            self._state = type("State", (), {"turns": 0})()

        def reset(self, **kwargs):  # noqa: ANN003
            StubEnv.reset_kwargs = kwargs

        def fetch_atlas(self, positions_mm):  # noqa: ANN001
            self._state.turns += 1
            return {"content": [{"type": "text", "text": "Atlas at 5.00 mm."}]}

    calls = iter([
        {"id": "call_0", "type": "function", "function": {"name": "fetch_atlas", "arguments": '{"positions_mm":[5.0]}'}},
        {"id": "call_1", "type": "function", "function": {"name": "submit_estimate", "arguments": '{"position_mm":5.2,"reasoning":"matched"}'}},
    ])
    monkeypatch.setattr(eval_mod, "AtlasMetaCache", StubCache)
    monkeypatch.setattr(eval_mod, "_extract_tool_call_from_decoded", lambda text: next(calls, None))
    rlvr_pkg = types.ModuleType("rlvr")
    rlvr_env_mod = types.ModuleType("rlvr.env")
    rlvr_env_mod.LangSliceEstimateEnv = StubEnv
    monkeypatch.setitem(sys.modules, "rlvr", rlvr_pkg)
    monkeypatch.setitem(sys.modules, "rlvr.env", rlvr_env_mod)

    class StubProcessor:
        def apply_chat_template(self, *args, **kwargs):  # noqa: ANN002, ANN003
            class Batch(dict):
                def to(self, device):  # noqa: ANN001
                    return self
            return Batch()
        def decode(self, *args, **kwargs):  # noqa: ANN002, ANN003
            return "<tool_call>{}</tool_call>"

    class StubModel:
        device = "cpu"
        def generate(self, **kwargs):  # noqa: ANN003
            return [[0]]

    run = eval_mod._run_agent_loop_for_one(
        model=StubModel(),
        processor=StubProcessor(),
        eval_row={"subject_id": "M01", "image_path": image_path, "atlas_name": "allen_mouse_25um", "plane": "coronal", "ground_truth_position_mm": 5.0},
        atlas_grid=object(),
    )
    assert run.predicted_mm == [5.2]
    assert run.parseable is True
    assert StubEnv.reset_kwargs["kind"] == "single"
    assert StubEnv.reset_kwargs["valid_range_mm"] == (0.0, 13.2)
```

- [ ] **Step 3: Implement BaselineEvalCallback + AgentLoopEvalCallback**

Append to `models/langslice-gemma-4/training/sft/eval.py`:

```python
import logging
import sys
from pathlib import Path
from typing import TYPE_CHECKING

from transformers import TrainerCallback

# Make the rlvr package importable from the SFT package
_TRAINING_ROOT = Path(__file__).resolve().parents[1]
if str(_TRAINING_ROOT) not in sys.path:
    sys.path.insert(0, str(_TRAINING_ROOT))

if TYPE_CHECKING:
    from rlvr.env import LangSliceEstimateEnv  # noqa: F401

logger = logging.getLogger(__name__)


def _load_test_images_with_truth(test_images_root: Path) -> list[dict[str, Any]]:
    """Read references/TestImages/M0[1-9]/ground_truth.json into a list of eval rows."""
    rows: list[dict[str, Any]] = []
    for sub in sorted(test_images_root.glob("M0[1-9]")):
        gt_path = sub / "ground_truth.json"
        if not gt_path.is_file():
            continue
        gt = json.loads(gt_path.read_text(encoding="utf-8"))
        rows.append({
            "subject_id": sub.name,
            "image_path": sub / gt["image_filename"],
            "atlas_name": gt["atlas_name"],
            "plane": gt["plane"],
            "ground_truth_position_mm": gt["position_mm"],
        })
    return rows


def _run_agent_loop_for_one(
    *,
    model: Any,
    processor: Any,
    eval_row: dict[str, Any],
    atlas_grid: Any,
    max_turns: int = 16,
) -> EvalRun:
    """Run the SFT model through the agent loop on one eval row, return EvalRun.

    Reuses LangSliceEstimateEnv (RLVR scaffolding) for tool execution.
    """
    from PIL import Image
    from rlvr.env import LangSliceEstimateEnv

    env = LangSliceEstimateEnv(atlas_grid=atlas_grid)
    query_image = Image.open(eval_row["image_path"]).convert("RGB")
    system_prompt_kind = "single_slice"
    truth = [eval_row["ground_truth_position_mm"]]
    # Build a renderer with no Example overhead — direct prompt construction
    cache = AtlasMetaCache()
    system_prompt = build_system_prompt(
        kind=system_prompt_kind,
        atlas_name=eval_row["atlas_name"],
        plane=eval_row["plane"],
        atlas_meta_cache=cache,
    )
    tools = build_tools_schema(system_prompt_kind)
    meta = cache.get(eval_row["atlas_name"], eval_row["plane"])
    messages: list[dict[str, Any]] = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": [
            {"type": "image", "image": query_image},
            {"type": "text", "text": "Estimate the position of this slice."},
        ]},
    ]
    env.reset(
        atlas_name=eval_row["atlas_name"],
        plane=eval_row["plane"],
        valid_range_mm=(meta.pos_lo, meta.pos_hi),
        ground_truth_positions_mm=truth,
        kind="single",
        prompt=messages,
        image=query_image,
        subject_id=eval_row["subject_id"],
    )

    n_turns = 0
    parseable = False
    predicted: list[float] | None = None
    for _ in range(max_turns):
        n_turns += 1
        gen_out = model.generate(
            **processor.apply_chat_template(
                messages, tools=tools, chat_template_kwargs={"enable_thinking": False},
                add_generation_prompt=True, tokenize=True, return_dict=True, return_tensors="pt",
            ).to(model.device),
            max_new_tokens=512,
            do_sample=False,
        )
        text = processor.decode(gen_out[0], skip_special_tokens=False)
        # Extract the assistant's tool_calls from `text`. The exact parsing depends
        # on Gemma 4's tool-call output format. This module function is the place
        # that may need to change if the format differs.
        tool_call = _extract_tool_call_from_decoded(text)
        messages.append({"role": "assistant", "tool_calls": [tool_call] if tool_call else []})
        if tool_call is None:
            break
        if tool_call["function"]["name"].startswith("submit"):
            parsed = parse_submit_call(tool_call["function"]["arguments"], expected_kind=system_prompt_kind)
            parseable = parsed.is_parseable
            if parsed.is_parseable and parsed.position_mm is not None:
                predicted = [parsed.position_mm]
            break
        # Execute fetch_atlas via env, append a tool message
        result = env.fetch_atlas(**json.loads(tool_call["function"]["arguments"]))
        tool_msg = {
            "role": "tool",
            "tool_call_id": tool_call["id"],
            "content": result.get("content", [{"type": "text", "text": str(result)}]),
        }
        messages.append(tool_msg)

    return EvalRun(
        subject_id=eval_row["subject_id"],
        predicted_mm=predicted,
        truth_mm=truth,
        parseable=parseable,
        n_turns=n_turns,
    )


def _extract_tool_call_from_decoded(text: str) -> dict[str, Any] | None:
    """Pull the first tool_call out of Gemma 4's decoded assistant turn.

    Gemma 4's tool-call format wraps function calls in a known marker. The exact
    pattern is verified in Task 1 (API verification) and adjusted here.
    Returns a dict shaped like HF tool_calls items, or None if no parseable call.
    """
    # Placeholder pattern — replace with the verified pattern from Task 1.
    import re
    m = re.search(r"<tool_call>(.*?)</tool_call>", text, flags=re.DOTALL)
    if not m:
        return None
    try:
        payload = json.loads(m.group(1).strip())
    except json.JSONDecodeError:
        return None
    return {
        "id": f"call_{abs(hash(text)) % 100000}",
        "type": "function",
        "function": {
            "name": payload.get("name", ""),
            "arguments": json.dumps(payload.get("arguments", {})),
        },
    }


class _AgentLoopEvalBase(TrainerCallback):
    """Shared logic for baseline + periodic eval callbacks."""

    def __init__(
        self,
        *,
        processor: Any,
        atlas_grid: Any,
        test_images_root: Path,
        log_prefix: str,
    ) -> None:
        self.processor = processor
        self.atlas_grid = atlas_grid
        self.test_rows = _load_test_images_with_truth(test_images_root)
        self.log_prefix = log_prefix

    def _run(self, model: Any, step: int) -> dict[str, float]:
        runs: list[EvalRun] = []
        for row in self.test_rows:
            try:
                runs.append(_run_agent_loop_for_one(
                    model=model, processor=self.processor,
                    eval_row=row, atlas_grid=self.atlas_grid,
                ))
            except Exception:
                logger.exception("agent-loop eval failed for %s", row["subject_id"])
                runs.append(EvalRun(
                    subject_id=row["subject_id"],
                    predicted_mm=None,
                    truth_mm=[row["ground_truth_position_mm"]],
                    parseable=False,
                    n_turns=0,
                ))
        summary = summarize_eval_runs(runs)
        prefixed = {f"{self.log_prefix}/{k}": v for k, v in summary.items()}
        logger.info("eval@step=%d: %s", step, prefixed)
        return prefixed


class BaselineEvalCallback(_AgentLoopEvalBase):
    """Run the agent-loop eval once before training starts."""

    def __init__(self, **kwargs: Any) -> None:
        super().__init__(log_prefix="baseline", **kwargs)
        self._has_run = False

    def on_train_begin(self, args, state, control, model=None, **kwargs):
        if self._has_run or model is None:
            return
        self._has_run = True
        metrics = self._run(model, step=0)
        if hasattr(state, "log_history"):
            state.log_history.append({"step": 0, **metrics})


class AgentLoopEvalCallback(_AgentLoopEvalBase):
    """Run the agent-loop eval every `agent_eval_steps` optimizer steps."""

    def __init__(self, *, agent_eval_steps: int, **kwargs: Any) -> None:
        super().__init__(log_prefix="eval", **kwargs)
        self.agent_eval_steps = agent_eval_steps

    def on_step_end(self, args, state, control, model=None, **kwargs):
        if model is None or state.global_step == 0:
            return
        if state.global_step % self.agent_eval_steps != 0:
            return
        # Switch model to inference mode if Unsloth requires it
        try:
            from unsloth import FastVisionModel
            FastVisionModel.for_inference(model)
        except Exception:
            pass
        try:
            metrics = self._run(model, step=state.global_step)
            state.log_history.append({"step": state.global_step, **metrics})
        finally:
            try:
                FastVisionModel.for_training(model)
            except Exception:
                pass
```

- [ ] **Step 4: Run tests to verify they pass**

Run:
```bash
python -m pytest tests/test_sft_eval.py -v
```

Expected: 6 PASS.

- [ ] **Step 5: Cross-check against rlvr/env.py for arg consistency**

Open `models/langslice-gemma-4/training/rlvr/env.py` and verify:
- The constructor `LangSliceEstimateEnv(atlas_grid=...)` matches the call in `_run_agent_loop_for_one`.
- The `reset(...)` arg names match. **If they don't, edit the `env.reset(...)` call site to match what env.py actually accepts** — the spec's RLVR §5 sketch may have drifted from the real implementation.
- The return shape of `env.fetch_atlas(...)` is the list-of-content-blocks the eval code assumes; if not, adapt the tool message construction.

If env.py uses different names, update `_run_agent_loop_for_one` accordingly. The eval code is the consumer; env.py is authoritative.

- [ ] **Step 6: Commit**

```bash
git add models/langslice-gemma-4/training/sft/eval.py tests/test_sft_eval.py
git commit -m "feat(gemma-4): SFT agent-loop eval callbacks (baseline + periodic)"
```

---

## Task 11: Config — sft_default.toml

**Files:**
- Create: `models/langslice-gemma-4/training/configs/sft_default.toml`

- [ ] **Step 1: Write the config file**

Create `models/langslice-gemma-4/training/configs/sft_default.toml`:

```toml
# Default SFT configuration for the LangSlice Gemma 4 E4B run.
# See docs/superpowers/specs/2026-05-05-gemma4-sft-training-design.md.

[sft]
base_model = "unsloth/gemma-4-E4B-it"
load_in_4bit = true
max_seq_length = 16384                  # multi-turn + multi-image traces
num_train_epochs = 3
per_device_train_batch_size = 1
gradient_accumulation_steps = 8
learning_rate = 2e-4
lr_scheduler_type = "cosine"
warmup_ratio = 0.03
weight_decay = 0.01
optim = "adamw_8bit"
chat_template_kwargs = { enable_thinking = false }
logging_steps = 5
eval_steps = 50
agent_eval_steps = 200
save_steps = 100
report_to = "trackio"
seed = 0

[lora]
r = 16
lora_alpha = 32
finetune_vision_layers = false
finetune_language_layers = true
finetune_attention_modules = true
finetune_mlp_modules = true
use_gradient_checkpointing = "unsloth"

[data]
holdout_fraction = 0.10
include_rationale = false
```

- [ ] **Step 2: Commit**

```bash
git add models/langslice-gemma-4/training/configs/sft_default.toml
git commit -m "feat(gemma-4): SFT default hyperparameters config"
```

---

## Task 12: Driver — argument parsing, config loading, dataset loading

**Files:**
- Modify: `models/langslice-gemma-4/training/sft/train_sft.py`

- [ ] **Step 1: Implement the CLI + config + dataset loading skeleton**

Write to `models/langslice-gemma-4/training/sft/train_sft.py`:

```python
"""Driver script: CLI entry point for SFT training of Gemma 4 E4B via Unsloth + TRL.

Usage:
    python -m sft.train_sft \\
        --config models/langslice-gemma-4/training/configs/sft_default.toml \\
        --dataset models/langslice-gemma-4/data/sft_examples.jsonl \\
        --output-dir out/sft/run0

Heavy deps (unsloth, trl) are imported lazily inside main() so unit tests can
import sibling modules without a runtime install.
"""

from __future__ import annotations

import argparse
import logging
import sys
import tomllib
from pathlib import Path
from typing import Any

from .dataset import load_examples, split_subject_aware
from .render import AtlasMetaCache, render_example
from .collate import LangSliceCollator
from .eval import AgentLoopEvalCallback, BaselineEvalCallback

logger = logging.getLogger(__name__)


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="SFT for Gemma 4 E4B on langslice-native trace JSONL."
    )
    p.add_argument("--config", type=Path, required=True,
                   help="TOML config with [sft], [lora], [data] tables.")
    p.add_argument("--dataset", type=Path, required=True,
                   help="JSONL of langslice-native trace examples.")
    p.add_argument("--output-dir", type=Path, required=True,
                   help="Where to save the LoRA adapter + tokenizer/processor.")
    p.add_argument("--test-images-root", type=Path,
                   default=Path("references/TestImages"),
                   help="Root containing M01-M09 ground-truth-labeled test images.")
    p.add_argument("--seed", type=int, default=None,
                   help="Override config's seed.")
    p.add_argument("--dry-run", action="store_true",
                   help="Load everything but do not train (smoke for wiring).")
    return p.parse_args(argv)


def _load_config(path: Path) -> dict[str, dict[str, Any]]:
    with path.open("rb") as f:
        return tomllib.load(f)


def _build_datasets(args: argparse.Namespace, data_cfg: dict[str, Any]) -> tuple[list, list]:
    examples = load_examples(args.dataset)
    train, eval_ = split_subject_aware(
        examples,
        holdout_fraction=float(data_cfg["holdout_fraction"]),
        seed=args.seed if args.seed is not None else 0,
    )
    logger.info("Loaded %d examples (%d train, %d eval) from %s",
                len(examples), len(train), len(eval_), args.dataset)
    return train, eval_


class _RenderedDataset:
    """torch.utils.data.Dataset shim: lazily renders examples on __getitem__."""

    def __init__(self, examples: list, atlas_meta_cache: AtlasMetaCache) -> None:
        self.examples = examples
        self.cache = atlas_meta_cache

    def __len__(self) -> int:
        return len(self.examples)

    def __getitem__(self, idx: int):
        return render_example(self.examples[idx], atlas_meta_cache=self.cache)


def main(argv: list[str] | None = None) -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )
    args = _parse_args(argv)
    config = _load_config(args.config)
    seed = args.seed if args.seed is not None else int(config["sft"].get("seed", 0))

    train_examples, eval_examples = _build_datasets(args, config["data"])

    cache = AtlasMetaCache()
    train_ds = _RenderedDataset(train_examples, cache)
    eval_ds = _RenderedDataset(eval_examples, cache)

    if args.dry_run:
        logger.info("--dry-run: skipping model load + training")
        return

    _train(args, config, train_ds, eval_ds, cache, seed)


def _train(args, config, train_ds, eval_ds, cache, seed: int) -> None:
    """Heavy-import path. Defined separately so dry-run never reaches it."""
    raise NotImplementedError("filled in by Task 13")


if __name__ == "__main__":
    main(sys.argv[1:])
```

- [ ] **Step 2: Verify dry-run works end-to-end**

This requires staging a real-looking JSONL with images. Use the fixture path:

Create `tests/fixtures/sft_traces/dryrun_dataset.jsonl` with the same content as `single_slice_minimal.jsonl` (one row), and the dummy images alongside it (`query.png`, `a3.png`, `a5.png`, `a7.png` as 32x32 grey PNGs — generate via a one-off shell command):

```bash
python -c "
from PIL import Image
import shutil
from pathlib import Path
src = Path('tests/fixtures/sft_traces/single_slice_minimal.jsonl')
dst = src.parent / 'dryrun_dataset.jsonl'
shutil.copy(src, dst)
for name in ('query.png', 'a3.png', 'a5.png', 'a7.png'):
    Image.new('RGB', (32, 32), (128,128,128)).save(src.parent / name)
"
```

Then dry-run:

```bash
cd models/langslice-gemma-4/training && python -m sft.train_sft \
    --config configs/sft_default.toml \
    --dataset ../../../tests/fixtures/sft_traces/dryrun_dataset.jsonl \
    --output-dir ../../../out/sft/dryrun \
    --dry-run
```

This mirrors the RLVR module invocation pattern (`python -m rlvr.train_grpo`) from the training directory. Check by running:

```bash
ls models/langslice-gemma-4/training/rlvr/train_grpo.py
```

Expected: file exists; copy its run pattern into the `train_sft.py` docstring header.

The dry-run command should print:
```
INFO sft.train_sft: Loaded 1 examples (... train, ... eval) from ...
INFO sft.train_sft: --dry-run: skipping model load + training
```

If `split_subject_aware` raises because there's only one subject, that's expected — for dry-run the user should provide a JSONL with at least 2 subjects, or the driver should accept single-subject for dry-run:

Patch `_build_datasets` to allow single-subject in dry-run:

```python
def _build_datasets(args, data_cfg):
    examples = load_examples(args.dataset)
    if args.dry_run and len({e.subject_id for e in examples}) < 2:
        logger.warning("--dry-run with <2 subjects: skipping subject-aware split")
        return examples, []
    train, eval_ = split_subject_aware(
        examples,
        holdout_fraction=float(data_cfg["holdout_fraction"]),
        seed=args.seed if args.seed is not None else 0,
    )
    logger.info("Loaded %d examples (%d train, %d eval) from %s",
                len(examples), len(train), len(eval_), args.dataset)
    return train, eval_
```

- [ ] **Step 3: Commit**

```bash
git add models/langslice-gemma-4/training/sft/train_sft.py tests/fixtures/sft_traces/dryrun_dataset.jsonl tests/fixtures/sft_traces/*.png
git commit -m "feat(gemma-4): SFT driver CLI + config + dataset loading (dry-run)"
```

---

## Task 13: Driver — model loading, LoRA wrap, trainer construction

**Files:**
- Modify: `models/langslice-gemma-4/training/sft/train_sft.py`

- [ ] **Step 1: Implement _train (heavy import path)**

Replace the placeholder `_train` in `train_sft.py` with:

```python
def _train(args, config, train_ds, eval_ds, cache, seed: int) -> None:
    """Heavy-import path. Loads model, builds trainer, runs trainer.train()."""
    sft_cfg = dict(config["sft"])
    lora_cfg = dict(config["lora"])

    # Lazy imports — keep dataset/render/collate unit tests cheap
    from trl import SFTConfig, SFTTrainer
    from unsloth import FastVisionModel
    from rlvr.atlas_grid import build_atlas_grid

    # Pre-render the atlas grid once for the eval callbacks
    pairs = {(ex.atlas_name, ex.plane) for ex in train_ds.examples + eval_ds.examples}
    atlas_grid = build_atlas_grid(pairs)

    # Load base model + processor
    model, processor = FastVisionModel.from_pretrained(
        sft_cfg["base_model"],
        load_in_4bit=bool(sft_cfg.get("load_in_4bit", True)),
        max_seq_length=int(sft_cfg.get("max_seq_length", 16384)),
    )
    model = FastVisionModel.get_peft_model(
        model,
        finetune_vision_layers=bool(lora_cfg.get("finetune_vision_layers", False)),
        finetune_language_layers=bool(lora_cfg.get("finetune_language_layers", True)),
        finetune_attention_modules=bool(lora_cfg.get("finetune_attention_modules", True)),
        finetune_mlp_modules=bool(lora_cfg.get("finetune_mlp_modules", True)),
        r=int(lora_cfg.get("r", 16)),
        lora_alpha=int(lora_cfg.get("lora_alpha", 32)),
        use_gradient_checkpointing=lora_cfg.get("use_gradient_checkpointing", "unsloth"),
        random_state=seed,
    )
    FastVisionModel.for_training(model)

    collator = LangSliceCollator(
        processor=processor,
        max_seq_length=int(sft_cfg.get("max_seq_length", 16384)),
    )

    sft_config_kwargs = {
        k: v for k, v in sft_cfg.items()
        if k not in ("base_model", "load_in_4bit", "max_seq_length", "agent_eval_steps")
    }
    training_args = SFTConfig(
        output_dir=str(args.output_dir),
        seed=seed,
        # Critical: do not let TRL try to apply text-only assistant-only loss
        assistant_only_loss=False,
        # Critical for VLMs: no TRL truncation. The custom collator rejects
        # examples beyond max_seq_length before they reach the trainer.
        max_length=None,
        **sft_config_kwargs,
    )

    callbacks = [
        BaselineEvalCallback(
            processor=processor,
            atlas_grid=atlas_grid,
            test_images_root=args.test_images_root,
        ),
        AgentLoopEvalCallback(
            processor=processor,
            atlas_grid=atlas_grid,
            test_images_root=args.test_images_root,
            agent_eval_steps=int(sft_cfg.get("agent_eval_steps", 200)),
        ),
    ]

    trainer = SFTTrainer(
        model=model,                  # already a PeftModel — do NOT pass peft_config
        processing_class=processor,
        args=training_args,
        train_dataset=train_ds,
        eval_dataset=eval_ds,
        data_collator=collator,
        callbacks=callbacks,
    )
    trainer.train()
    trainer.save_model(str(args.output_dir))
    processor.save_pretrained(str(args.output_dir))
    logger.info("Saved adapter + processor to %s", args.output_dir)
```

- [ ] **Step 2: Verify imports (no execution)**

Run:
```bash
python -c "
import sys; sys.path.insert(0, 'models/langslice-gemma-4/training')
from sft import train_sft
print('imports OK; main:', train_sft.main)
"
```

Expected: prints `imports OK; main: <function main at ...>`. No ImportError. (Heavy imports happen inside `_train`, not at module load.)

- [ ] **Step 3: Commit**

```bash
git add models/langslice-gemma-4/training/sft/train_sft.py
git commit -m "feat(gemma-4): SFT driver model loading + trainer construction"
```

---

## Task 14: Smoke run (manual)

**Files:**
- No code changes — manual exercise.

- [ ] **Step 1: Generate a synthetic 100-row JSONL**

Create a one-off script `_local/scripts/gen_sft_smoke_corpus.py` (in `_local/` so it's gitignored):

```python
"""Generate 100-row synthetic SFT corpus for the smoke run."""
from __future__ import annotations
import json
from pathlib import Path
from PIL import Image

OUT = Path("_local/sft_smoke")
OUT.mkdir(parents=True, exist_ok=True)
JSONL = OUT / "smoke.jsonl"

# Generate 12 dummy subject ids so subject-aware split works
SUBJECTS = [f"smoke_{i:02d}" for i in range(12)]
ROWS = []
for i in range(100):
    sid = SUBJECTS[i % len(SUBJECTS)]
    qpath = OUT / f"q_{i:03d}.png"
    apath3 = OUT / f"a3_{i:03d}.png"
    apath5 = OUT / f"a5_{i:03d}.png"
    apath7 = OUT / f"a7_{i:03d}.png"
    for p in (qpath, apath3, apath5, apath7):
        Image.new("RGB", (224, 224), (i % 255, (i*3) % 255, (i*7) % 255)).save(p)
    ROWS.append({
        "bucket": 1,
        "atlas_name": "allen_mouse_25um",
        "atlas_version": "CCFv3",
        "plane": "coronal",
        "subject_id": sid,
        "system_prompt_kind": "single_slice",
        "query_image_paths": [qpath.name],
        "user_prompt_text": "Estimate the AP position of this slice.",
        "trace": [
            {
                "tool_call": {"name": "fetch_atlas", "args": {"positions_mm": [3.0, 5.0, 7.0]}},
                "tool_result": {
                    "image_paths": [apath3.name, apath5.name, apath7.name],
                    "text": "Atlas at 3.00 mm | 5.00 mm | 7.00 mm",
                },
            },
            {
                "submit": {
                    "name": "submit_estimate",
                    "args": {
                        "position_mm": 5.0 + (i % 5) * 0.1,
                        "reasoning": "Synthetic smoke example with known target position.",
                    },
                }
            },
        ],
    })

with JSONL.open("w", encoding="utf-8") as f:
    for r in ROWS:
        f.write(json.dumps(r) + "\n")
print("wrote", JSONL, "rows:", len(ROWS))
```

Run:
```bash
python _local/scripts/gen_sft_smoke_corpus.py
```

Expected: `wrote _local/sft_smoke/smoke.jsonl rows: 100`.

- [ ] **Step 2: Run a 50-step smoke training**

Make a smoke config — copy `sft_default.toml` to `_local/sft_smoke/sft_smoke.toml` and override:

```toml
[sft]
# inherit base_model, load_in_4bit, etc. from default — copy them in
base_model = "unsloth/gemma-4-E4B-it"
load_in_4bit = true
max_seq_length = 8192
num_train_epochs = 1
per_device_train_batch_size = 1
gradient_accumulation_steps = 4
learning_rate = 2e-4
lr_scheduler_type = "linear"
warmup_ratio = 0.03
weight_decay = 0.01
optim = "adamw_8bit"
chat_template_kwargs = { enable_thinking = false }
logging_steps = 5
eval_steps = 25
agent_eval_steps = 200    # won't fire in 50 steps; keeps smoke fast
save_steps = 25
report_to = "none"
seed = 0
max_steps = 50            # caps the run regardless of epochs

[lora]
r = 16
lora_alpha = 32
finetune_vision_layers = false
finetune_language_layers = true
finetune_attention_modules = true
finetune_mlp_modules = true
use_gradient_checkpointing = "unsloth"

[data]
holdout_fraction = 0.10
include_rationale = false
```

Run from the repo root:
```bash
cd models/langslice-gemma-4/training && python -m sft.train_sft \
    --config configs/sft_default.toml \
    --dataset ../../../_local/sft_smoke/smoke.jsonl \
    --output-dir ../../../_local/sft_smoke/out
```

(Adjust the config path to point at the smoke variant if you saved it under `_local/`.)

Expected:
- Loads 100 examples, splits into train/eval.
- Loads Gemma 4 E4B in 4-bit (~5–10 minutes first time, downloads weights).
- Trains 50 steps. Loss visible in stdout, decreases roughly monotonically.
- Saves an adapter directory at `_local/sft_smoke/out/`.

If loss is flat or NaN, **stop and debug** before continuing. Likely causes: wrong assistant_mask (all tokens masked → no gradient → flat loss), wrong base model ID, OOM.

- [ ] **Step 3: Verify the saved checkpoint loads**

Run:
```bash
python -c "
import json
from pathlib import Path
from unsloth import FastVisionModel
from peft import PeftModel
adapter_dir = Path('_local/sft_smoke/out')
base_model = json.loads((adapter_dir / 'adapter_config.json').read_text())['base_model_name_or_path']
m, p = FastVisionModel.from_pretrained(base_model, load_in_4bit=True)
m = PeftModel.from_pretrained(m, str(adapter_dir))
print('loaded OK; type:', type(m).__name__)
"
```

Expected: prints `loaded OK; type: ...`. No exception.

- [ ] **Step 4: Smoke-run summary note**

Capture a short note in `_local/sft_smoke/SMOKE_NOTES.md` with:
- Loss trajectory (5 sample values)
- Final adapter directory size
- Wall-clock time
- Any anomalies

This file stays in `_local/` (gitignored) — not committed.

- [ ] **Step 5: Commit (no code changes — but mark progress)**

No commit needed for this task; it's a manual verification.

---

## Task 15: Inference smoke (manual)

**Files:**
- No code changes — manual exercise.

- [ ] **Step 1: Run inference on M01_001_001.tif**

```bash
python -c "
import json
from pathlib import Path
from PIL import Image
from unsloth import FastVisionModel
from peft import PeftModel
import sys
sys.path.insert(0, 'models/langslice-gemma-4/training')
from sft.render import build_system_prompt, build_tools_schema, AtlasMetaCache

adapter_dir = Path('_local/sft_smoke/out')
base_model = json.loads((adapter_dir / 'adapter_config.json').read_text())['base_model_name_or_path']
m, p = FastVisionModel.from_pretrained(base_model, load_in_4bit=True)
m = PeftModel.from_pretrained(m, str(adapter_dir))
FastVisionModel.for_inference(m)
img = Image.open('references/TestImages/M01/M01_001_001.tif').convert('RGB')
gt = json.loads(Path('references/TestImages/M01/ground_truth.json').read_text())
cache = AtlasMetaCache()
system = build_system_prompt(kind='single_slice', atlas_name=gt['atlas_name'], plane=gt['plane'], atlas_meta_cache=cache)
tools = build_tools_schema('single_slice')
messages = [
    {'role':'system','content':system},
    {'role':'user','content':[{'type':'image','image':img},{'type':'text','text':'Estimate the position of this slice.'}]},
]
inputs = p.apply_chat_template(messages, tools=tools, chat_template_kwargs={'enable_thinking':False}, add_generation_prompt=True, tokenize=True, return_dict=True, return_tensors='pt').to(m.device)
out = m.generate(**inputs, max_new_tokens=512, do_sample=False)
print(p.decode(out[0], skip_special_tokens=False))
"
```

Expected: prints a generation that contains a recognizable tool-call (e.g. `<tool_call>{"name": "fetch_atlas", ...}</tool_call>`) — the exact wrapping is whatever Gemma 4's tool-call format renders. **Pass criterion:** the output contains a parseable function call. Accuracy is *not* asserted; this is wire-compatibility only.

If the output is blank or contains only system tokens, the smoke run didn't train enough — bump `max_steps` or check that loss decreased in Task 14.

---

## Task 16: RLVR handoff smoke (manual)

**Files:**
- No code changes expected; `models/langslice-gemma-4/training/rlvr/train_grpo.py` already detects SFT adapter directories via `adapter_config.json`.

- [ ] **Step 1: Run the existing RLVR loader against the SFT adapter**

```bash
cd models/langslice-gemma-4/training && python -m rlvr.train_grpo \
    --config configs/grpo_pilot.toml \
    --sft-model ../../../_local/sft_smoke/out \
    --output-dir ../../../_local/rlvr_smoke \
    --test-images-root ../../../references/TestImages \
    --seed 0
```

Expected: training begins, prints "Pre-rendering atlas grid...", reads `_local/sft_smoke/out/adapter_config.json`, loads the recorded base model, attaches the SFT adapter, and starts rolling out.

- [ ] **Step 2: Re-run the smoke with a no-optimizer config**

```bash
cd models/langslice-gemma-4/training && python -m rlvr.train_grpo \
    --config ../../../_local/rlvr_smoke/rlvr_smoke.toml \
    --sft-model ../../../_local/sft_smoke/out \
    --output-dir ../../../_local/rlvr_smoke \
    --test-images-root ../../../references/TestImages \
    --seed 0
```

Override the GRPO config to do a no-optimizer 10-rollout smoke (set `max_steps = 0` in a `_local/rlvr_smoke.toml` copy and pass that with `--config`).

Expected: env unit tests pass, 10 rollouts complete, no exception loading the adapter.

- [ ] **Step 3: Run RLVR's existing test suite to confirm no regressions**

```bash
python -m pytest tests/test_rlvr_env.py tests/test_rlvr_rewards.py -v
```

Expected: clean.

- [ ] **Step 4: Commit only if the manual smoke revealed a new issue**

```bash
git add <changed-files>
git commit -m "fix(gemma-4): stabilize SFT-to-RLVR handoff smoke"
```

---

## Task 17: Cleanup — delete dead scaffolding from the older plan

**Files:**
- Delete: `models/langslice-gemma-4/training/finetune.py`
- Delete: `models/langslice-gemma-4/data/build_triplets.py`
- Delete: `models/langslice-gemma-4/data/distill_cot.py`
- Delete: `models/langslice-gemma-4/data/generate_atlas_slices.py`

Per memory note `feedback_confirm_spec_deletions`: confirm with the user before deleting.

- [ ] **Step 1: Confirm with the user that the four files are safe to delete**

These are stubs from the abandoned triplet-based plan, flagged in the RLVR spec §13 and SFT spec §14. Before `git rm`-ing them, confirm by reading each and verifying:

```bash
head -30 models/langslice-gemma-4/training/finetune.py
head -30 models/langslice-gemma-4/data/build_triplets.py
head -30 models/langslice-gemma-4/data/distill_cot.py
head -30 models/langslice-gemma-4/data/generate_atlas_slices.py
```

Expected: each is a stub or unused scaffold. If any is actively imported elsewhere, **do not delete**:

```bash
python -m grep --include="*.py" -rn "build_triplets\|distill_cot\|generate_atlas_slices\|finetune" models/ src/ tests/
```

(Replace with actual ripgrep / Grep tool calls.) Expected: only references are inside the four files themselves and in the RLVR spec §13 / SFT spec §14 markdown.

If any unexpected reference shows up (e.g. an import in `models/langslice-gemma-4/__init__.py`), update or skip the deletion.

- [ ] **Step 2: Delete the four stub files**

```bash
git rm models/langslice-gemma-4/training/finetune.py
git rm models/langslice-gemma-4/data/build_triplets.py
git rm models/langslice-gemma-4/data/distill_cot.py
git rm models/langslice-gemma-4/data/generate_atlas_slices.py
```

- [ ] **Step 3: Commit**

```bash
git commit -m "chore(gemma-4): remove dead scaffolding from older triplet plan

Per docs/superpowers/specs/2026-05-04-gemma4-rlvr-training-design.md §13 and
docs/superpowers/specs/2026-05-05-gemma4-sft-training-design.md §14."
```

---

## Task 18: Final regression — pytest + ruff + basedpyright

**Files:**
- No new code; verification only.

- [ ] **Step 1: Run full test suite**

```bash
python -m pytest tests/ -v
```

Expected: all tests PASS or SKIP. No FAILs. No new test failures introduced by SFT work.

If any test fails because the LangSliceEstimateEnv interface differs from what eval.py expected, fix it (Task 10 step 5 already cross-checks but a final pass catches drift).

- [ ] **Step 2: Run ruff**

```bash
python -m ruff check models/langslice-gemma-4/training/sft/ tests/test_sft_*.py
```

Expected: no errors. If errors, fix them inline (most likely unused imports or line-length).

- [ ] **Step 3: Run basedpyright**

```bash
python -m basedpyright models/langslice-gemma-4/training/sft/ tests/test_sft_*.py
```

Expected: no errors. If type errors, fix them — common ones: PIL.Image.Image type hints, dict[str, Any] vs more specific TypedDicts.

- [ ] **Step 4: Run the project's standard verify suite per AGENTS.md**

```bash
python -m langslice_harness version
langslice version
```

Expected: both print the same version string. The new SFT module shouldn't have broken the harness CLI (sanity check only).

- [ ] **Step 5: Commit any final cleanups**

If any lint/type fixes were needed:

```bash
git add -u
git commit -m "style(gemma-4): SFT lint + type cleanup"
```

If no changes, no commit.

---

## Task 19: Pre-RLVR parseability gate (one-time check after real training)

**Files:**
- No code changes — gating exercise per spec §10.3.

- [ ] **Step 1: Run a real SFT training pass on the actual corpus**

Once the real SFT JSONL is delivered by the data sessions:

```bash
cd models/langslice-gemma-4/training && python -m sft.train_sft \
    --config configs/sft_default.toml \
    --dataset <path-to-real-corpus.jsonl> \
    --output-dir ../../../out/sft/run0
```

This is the actual SFT run — duration depends on corpus size. Per spec §9, num_train_epochs=3 is the default.

- [ ] **Step 2: Inspect the post-training agent-loop eval metrics**

The `AgentLoopEvalCallback` logs to trackio at `agent_eval_steps` intervals. The final periodic eval should report:

- `eval/position_mae_mm` — primary quality metric
- `eval/tool_call_parseability_rate` — **gate threshold: ≥ 0.80**
- `eval/no_submit_rate` — should be near 0
- `eval/mean_trace_length` — should be reasonable (under the spec's RLVR turn budget of 8–12)

- [ ] **Step 3: Decision point**

- **If parseability rate ≥ 0.80:** SFT is good to hand off to RLVR. Proceed to running `train_grpo.py --sft-model out/sft/run0`.
- **If parseability rate < 0.80:** SFT is not RLVR-ready. **Stop**. Per spec §10.3 + §12, the cheapest remediation is to add the deferred programmatic-skeletons bucket (§5 of the 2026-04-25 SFT data spec) — synthetic traces that are format-correct by construction. This requires a separate data-side workstream; do not attempt to "fix" via training-side knobs alone.

---

## Self-Review Notes

### Spec coverage check

Walking through `docs/superpowers/specs/2026-05-05-gemma4-sft-training-design.md` v1.3:

| Spec section | Implemented in task |
|---|---|
| §1 Goal | All tasks |
| §2 Scope | All tasks |
| §3 Bucket revision | Task 3 (validates bucket==1) |
| §4 Vision frozen | Task 13 (LoRA config) |
| §5 File layout | Task 2 |
| §6 Data contract | Task 3 (loader + validation) |
| §7.1 Renderer | Tasks 5, 6 |
| §7.2 Collator happy path | Task 7 |
| §7.3 Off-the-shelf doesn't work / fallback | Task 8 |
| §8 Driver | Tasks 12, 13 |
| §9 Config | Task 11 |
| §10.1 Pre-training baseline | Task 10 (BaselineEvalCallback) |
| §10.2 During training | Task 10 (AgentLoopEvalCallback) |
| §10.3 Pre-RLVR parseability gate | Task 19 |
| §10.4 Post-training (SliceBench) | Out of scope (separate work) |
| §11 Verification | Tasks 3-10 (unit), 14-16 (smoke), 18 (lint/type) |
| §12 Risks | Task 1 (API verification) addresses most; rest are runtime concerns |
| §13 Reuse pointers | Tasks 5, 6, 10 (renderer/eval consumers) |
| §14 Cleanup | Task 17 |
| §15 Dependencies | Tasks 1 (verification), 16 (RLVR handoff smoke) |

All sections accounted for.

### Placeholder scan

- "TBD" / "TODO" / "implement later" / "fill in details" — none found in tasks.
- "Add appropriate error handling" — none.
- Task 10's `_extract_tool_call_from_decoded` has a "Placeholder pattern — replace with the verified pattern from Task 1" comment. This is intentional — the pattern depends on Task 1's findings. Acceptable because Task 1 is gated to run first; the comment links the dependency.
- Task 16 has conditional commits ("if patched"). Acceptable because the patch may not be needed.

### Type consistency

- `Example` dataclass fields are referenced consistently across tasks (`bucket`, `atlas_name`, `system_prompt_kind`, `query_image_paths`, `trace`, `subject_id`, `gemini_reasoning`).
- `RenderedExample` fields (`messages`, `tools`, `metadata`) consistent across renderer + collator + driver.
- `EvalRun` fields consistent across `summarize_eval_runs` + agent-loop callbacks.
- Function names: `load_examples`, `split_subject_aware`, `render_example`, `build_system_prompt`, `build_tools_schema`, `parse_submit_call`, `compute_position_mae_mm`, `summarize_eval_runs` — all referenced consistently.

No inconsistencies found.
