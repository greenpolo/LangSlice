# Gemma 4 BBox Training Data — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build the assembly + Gemini-Batch-API pipeline that produces the BBox-grounded multi-section morphology caption SFT corpus for the langslice-gemma-4 model.

**Architecture:** Layered modules under `_local/eval/lib/` (utilities tied to the local data tree) and `models/langslice-gemma-4/data/` (the gemma-4 pipeline). Two-stage CLI on the orchestrator: `--stage sample` produces a draft manifest with bbox overlays, the existing QC app gains a `--mode bbox` switch for the user to verify/reject, then `--stage submit` packs verified examples into AI Studio Batch API JSONL, polls for completion, and writes final SFT records.

**Tech Stack:** Python 3.11+, `google-genai` SDK (AI Studio Batch API), BrainGlobe atlases, scipy (Delaunay for VisuAlign warp — already used in `_local/eval/lib/registration.py`), Pillow (overlay rendering), `synth_dataset.py` augmentation pipeline (existing).

**Authoritative spec:** `docs/superpowers/specs/2026-05-04-gemma4-bbox-training-design.md`. Read it before starting.

---

## Pre-flight

These should already be in place before any task starts:

- `models/langslice-gemma-4/data/landmarks.json` — user-curated landmark list (committed).
- `_local/eval/lib/registration.py` — VisuAlign-marker triangle warp + OUV fallback (built by the audit agent; smoke-tested).
- `_local/eval/data_inventory.md` — per-dataset Tier A / Tier B classification (gitignored).
- `_local/eval/data/manifest.jsonl` — built by `_local/eval/build_manifest.py`; the canonical source-of-truth for what real-histology sections exist on disk and where.

Verify the registration helper works before Task 4:

```bash
cd C:/LabSoftware/LangSlice
python -c "import sys; sys.path.insert(0, '_local/eval/lib'); from registration import section_to_atlas_voxel; print('ok')"
```

Expected output: `ok`

---

## File Structure

| Path | Status | Responsibility |
|---|---|---|
| `models/langslice-gemma-4/data/landmarks.json` | Exists | User-curated landmarks per orientation |
| `models/langslice-gemma-4/data/landmark_atlas_map.json` | Create | Landmark name → atlas-acronyms with descendant policy |
| `models/langslice-gemma-4/data/resolve_landmarks.py` | Create | One-shot helper that drafts the atlas map via fuzzy match against BrainGlobe |
| `models/langslice-gemma-4/data/landmarks.py` | Create | Loader + name resolver (descendant walk via atlas structure tree) |
| `models/langslice-gemma-4/data/region_bbox.py` | Create | Hemisphere-split bbox computation (atlas slice + real-section paths) |
| `models/langslice-gemma-4/data/build_bbox_data.py` | Create | Orchestrator with `--stage sample` and `--stage submit` |
| `models/langslice-gemma-4/data/bbox_io.py` | Create | Helpers: draft manifest schema, batch JSONL packing, response parsing, mm-strip filter |
| `_local/eval/lib/landmark_coverage.py` | Create | One-time coarse coverage bake; emits `landmark_coverage.json` |
| `_local/eval/lib/registration.py` | Exists | `section_to_atlas_voxel` — used by both new modules |
| `src/langslice_harness/vlm_config.py` | Modify | Widen `supports_batch_api()` to allow AI Studio backend |
| `_local/qc_app/app.py` | Modify | Add `--mode bbox` switch + bbox-overlay rendering |
| `_local/qc_app/static/bbox.html` | Create | Bucket-bbox view template (N section strip + metadata + hotkeys) |
| `tests/test_landmarks.py` | Create | Loader + resolver tests |
| `tests/test_region_bbox.py` | Create | Atlas-path + realhist-path tests with synthetic fixtures |
| `tests/test_bbox_io.py` | Create | Manifest schema + mm-strip filter tests |
| `tests/test_build_bbox_data.py` | Create | End-to-end with stubbed batch client |
| `tests/test_vlm_config_batch.py` | Modify or Create | Verify AI Studio backend now passes the gate |

---

## Task 1: Landmark loader + resolver

**Files:**
- Create: `models/langslice-gemma-4/data/landmarks.py`
- Create: `tests/test_landmarks.py`
- Reference: `models/langslice-gemma-4/data/landmarks.json` (exists)

**Spec sections covered:** §5 (Region resolution).

- [ ] **Step 0: Ensure `tests/conftest.py` adds the gemma-4 data dir to `sys.path`**

The directory `models/langslice-gemma-4/` has a hyphen and can't be a normal
Python package. `pyproject.toml`'s `tool.pytest.ini_options.pythonpath`
already lists `models/langslice-gemma-4/data`, and `tests/conftest.py`
duplicates the insertion as a defensive guard so a clean `import landmarks`
(and later `import region_bbox`, etc.) works in every test file without
per-file `sys.path` hacks. Verify the conftest contains:

```python
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
GEMMA4_DATA = REPO_ROOT / "models" / "langslice-gemma-4" / "data"
if str(GEMMA4_DATA) not in sys.path:
    sys.path.insert(0, str(GEMMA4_DATA))
```

- [ ] **Step 1: Write the loader test**

Create `tests/test_landmarks.py` (no `sys.path` hack — the conftest from
Step 0 handles it):

```python
"""Tests for the landmark loader and resolver."""
from __future__ import annotations

import json
from pathlib import Path
from unittest.mock import MagicMock

import landmarks
import pytest


@pytest.fixture
def tmp_landmarks_json(tmp_path: Path) -> Path:
    payload = {
        "version": "0.1.0",
        "landmarks_by_orientation": {
            "coronal": [
                {"name": "Hippocampal Formation"},
                {"name": "Anterior Commissure"},
            ],
            "sagittal": [
                {"name": "Dentate Gyrus"},
            ],
            "horizontal": [],
        },
    }
    p = tmp_path / "landmarks.json"
    p.write_text(json.dumps(payload), encoding="utf-8")
    return p


def test_landmarks_for_orientation_returns_list(tmp_landmarks_json: Path):
    loader = landmarks.LandmarkLoader(landmarks_path=tmp_landmarks_json)
    coronal = loader.landmarks_for_orientation("coronal")
    assert coronal == ["Hippocampal Formation", "Anterior Commissure"]
    assert loader.landmarks_for_orientation("sagittal") == ["Dentate Gyrus"]
    assert loader.landmarks_for_orientation("horizontal") == []


def test_landmarks_for_unknown_orientation_raises(tmp_landmarks_json: Path):
    loader = landmarks.LandmarkLoader(landmarks_path=tmp_landmarks_json)
    with pytest.raises(KeyError):
        loader.landmarks_for_orientation("axial")
```

- [ ] **Step 2: Run test to verify it fails**

```bash
pytest tests/test_landmarks.py::test_landmarks_for_orientation_returns_list -v
```

Expected: FAIL — `ModuleNotFoundError: No module named 'landmarks'` or similar.

- [ ] **Step 3: Implement the loader**

Create `models/langslice-gemma-4/data/landmarks.py`:

```python
"""Landmark loader and atlas-region resolver for the BBox training data pipeline.

Loads the curated landmark list (`landmarks.json`) and the per-atlas mapping
(`landmark_atlas_map.json`); resolves each landmark to a set of BrainGlobe
region IDs, walking the structure-tree descendants when the mapping flags
`include_descendants: true`.
"""
from __future__ import annotations

import json
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any

_DEFAULT_LANDMARKS_PATH = Path(__file__).resolve().parent / "landmarks.json"
_DEFAULT_ATLAS_MAP_PATH = Path(__file__).resolve().parent / "landmark_atlas_map.json"


@dataclass(frozen=True)
class LandmarkLoader:
    landmarks_path: Path = _DEFAULT_LANDMARKS_PATH
    atlas_map_path: Path = _DEFAULT_ATLAS_MAP_PATH

    def _load_landmarks(self) -> dict:
        return json.loads(self.landmarks_path.read_text(encoding="utf-8"))

    def _load_atlas_map(self) -> dict:
        if not self.atlas_map_path.exists():
            return {}
        return json.loads(self.atlas_map_path.read_text(encoding="utf-8"))

    def landmarks_for_orientation(self, orientation: str) -> list[str]:
        payload = self._load_landmarks()
        try:
            entries = payload["landmarks_by_orientation"][orientation]
        except KeyError as exc:
            raise KeyError(
                f"Unknown orientation {orientation!r}; expected one of "
                f"{list(payload['landmarks_by_orientation'].keys())!r}."
            ) from exc
        return [entry["name"] for entry in entries]
```

- [ ] **Step 4: Run test to verify it passes**

```bash
pytest tests/test_landmarks.py::test_landmarks_for_orientation_returns_list -v tests/test_landmarks.py::test_landmarks_for_unknown_orientation_raises
```

Expected: 2 passed.

- [ ] **Step 5: Add the resolver test**

Append to `tests/test_landmarks.py`:

```python
@pytest.fixture
def tmp_atlas_map_json(tmp_path: Path) -> Path:
    payload = {
        "Hippocampal Formation": {
            "allen_mouse_25um": {
                "acronym": "HPF",
                "include_descendants": True,
            },
        },
        "Anterior Commissure": {
            "allen_mouse_25um": {
                "acronym": "act",
                "include_descendants": False,
            },
        },
    }
    p = tmp_path / "landmark_atlas_map.json"
    p.write_text(json.dumps(payload), encoding="utf-8")
    return p


def _fake_atlas_with_tree() -> object:
    """Build a tiny fake BrainGlobe atlas exposing the real API surface:
    structures / lookup_df / get_structure_descendants(acronym)."""
    atlas = MagicMock()
    atlas.structures = {
        "HPF": {"acronym": "HPF", "id": 1089},
        "CA1": {"acronym": "CA1", "id": 382},
        "DG": {"acronym": "DG", "id": 726},
        "act": {"acronym": "act", "id": 908},
    }
    atlas.lookup_df = None

    def descendants(acronym: str) -> list[str]:
        return ["CA1", "DG"] if acronym == "HPF" else []

    atlas.get_structure_descendants.side_effect = descendants
    return atlas


def test_resolve_landmark_with_descendants(
    tmp_landmarks_json: Path, tmp_atlas_map_json: Path
):
    loader = landmarks.LandmarkLoader(
        landmarks_path=tmp_landmarks_json, atlas_map_path=tmp_atlas_map_json
    )
    atlas = _fake_atlas_with_tree()
    ids = loader.resolve("Hippocampal Formation", atlas, atlas_name="allen_mouse_25um")
    assert ids == {1089, 382, 726}


def test_resolve_landmark_without_descendants(
    tmp_landmarks_json: Path, tmp_atlas_map_json: Path
):
    loader = landmarks.LandmarkLoader(
        landmarks_path=tmp_landmarks_json, atlas_map_path=tmp_atlas_map_json
    )
    atlas = _fake_atlas_with_tree()
    ids = loader.resolve("Anterior Commissure", atlas, atlas_name="allen_mouse_25um")
    assert ids == {908}


def test_resolve_unmapped_landmark_returns_empty(
    tmp_landmarks_json: Path, tmp_atlas_map_json: Path
):
    loader = landmarks.LandmarkLoader(
        landmarks_path=tmp_landmarks_json, atlas_map_path=tmp_atlas_map_json
    )
    atlas = _fake_atlas_with_tree()
    ids = loader.resolve("Dentate Gyrus", atlas, atlas_name="allen_mouse_25um")
    assert ids == set()
```

- [ ] **Step 6: Run resolver tests, verify they fail**

```bash
pytest tests/test_landmarks.py -v
```

Expected: 3 new tests FAIL with `AttributeError: 'LandmarkLoader' object has no attribute 'resolve'`.

- [ ] **Step 7: Implement the resolver**

Append to `models/langslice-gemma-4/data/landmarks.py`:

```python
    def resolve(
        self, landmark_name: str, atlas: object, atlas_name: str
    ) -> set[int]:
        """Resolve a landmark name to BrainGlobe region IDs for the given atlas.

        Walks the atlas structure-tree descendants when the mapping flags
        `include_descendants: true`. Returns an empty set if the landmark is
        unmapped for this atlas — the orchestrator drops unmapped tuples.
        """
        atlas_map = self._load_atlas_map()
        entry = atlas_map.get(landmark_name)
        if entry is None:
            return set()
        per_atlas = entry.get(atlas_name)
        if per_atlas is None:
            return set()

        acronym = per_atlas["acronym"]
        include_descendants = bool(per_atlas.get("include_descendants", False))

        structures = getattr(atlas, "structures", {})
        lookup_df = getattr(atlas, "lookup_df", None)

        def _structure_for(ac: str) -> Mapping[str, Any] | None:
            try:
                return structures[ac]
            except KeyError:
                pass
            if lookup_df is not None:
                matches = lookup_df.loc[lookup_df["acronym"] == ac]
                if not getattr(matches, "empty", True):
                    row = matches.iloc[0]
                    return {"acronym": str(row["acronym"]), "id": int(row["id"])}
            return None

        ids: set[int] = set()
        root_structure = _structure_for(acronym)
        if root_structure is not None:
            ids.add(int(root_structure["id"]))
        if include_descendants and root_structure is not None:
            # BrainGlobe's get_structure_descendants takes an acronym (or int id),
            # not a Structure object.
            for descendant_acronym in atlas.get_structure_descendants(acronym):
                descendant = _structure_for(descendant_acronym)
                if descendant is not None:
                    ids.add(int(descendant["id"]))
        return ids
```

Note: the imports at the top of `landmarks.py` must now also include
`from collections.abc import Mapping` and `from typing import Any` (used by
the `_structure_for` annotation above).

- [ ] **Step 8: Run all landmarks tests, verify pass**

```bash
pytest tests/test_landmarks.py -v
```

Expected: 5 passed.

- [ ] **Step 9: Commit**

```bash
git add models/langslice-gemma-4/data/landmarks.py tests/test_landmarks.py
git commit -m "feat(gemma-4): landmark loader and atlas-region resolver"
```

---

## Task 2: Landmark atlas-map drafting helper

**Files:**
- Create: `models/langslice-gemma-4/data/resolve_landmarks.py`

**Spec sections covered:** §5 (Region resolution; semi-automated map seeding).

This is a **one-shot setup helper**, not a library. It runs once to draft `landmark_atlas_map.json`, the curator hand-resolves ambiguous entries, and the JSON is then maintained by hand. No tests required — its output is validated by Task 1's resolver running against the produced JSON.

- [ ] **Step 1: Implement the helper**

Create `models/langslice-gemma-4/data/resolve_landmarks.py`:

```python
"""One-shot helper to draft `landmark_atlas_map.json` via fuzzy match against
BrainGlobe structure trees.

Run once after editing `landmarks.json`; review the output, hand-resolve any
entries flagged as ambiguous (`status: "ambiguous"`) or missing
(`status: "no_match"`), then save as `landmark_atlas_map.json`.

Usage:
    python models/langslice-gemma-4/data/resolve_landmarks.py \
        --atlases allen_mouse_25um whs_sd_rat_39um \
        --out models/langslice-gemma-4/data/landmark_atlas_map.draft.json

Developmental-mouse coverage is left to the synth_dataset augmentation
pipeline; this map only enumerates the two atlases we curate by hand.
"""
from __future__ import annotations

import argparse
import difflib
import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO_ROOT / "src"))

from langslice_harness.atlas.core import load_atlas  # noqa: E402

_LANDMARKS_PATH = Path(__file__).resolve().parent / "landmarks.json"


def _flatten_landmarks(landmarks: dict) -> list[str]:
    out: list[str] = []
    seen: set[str] = set()
    for entries in landmarks["landmarks_by_orientation"].values():
        for entry in entries:
            name = entry["name"]
            if name not in seen:
                out.append(name)
                seen.add(name)
    return out


def _fuzzy_match(name: str, atlas) -> tuple[str | None, str]:
    """Return (acronym, status). Status: 'ok' | 'ambiguous' | 'no_match'."""
    candidates = []
    name_lower = name.lower()
    for s in atlas.structures.values():
        # Match against full name and acronym; both are useful signals.
        full = s.get("name", "").lower()
        ac = s.get("acronym", "")
        score = max(
            difflib.SequenceMatcher(None, name_lower, full).ratio(),
            difflib.SequenceMatcher(None, name_lower, ac.lower()).ratio(),
        )
        if score >= 0.55:
            candidates.append((score, ac))
    candidates.sort(key=lambda x: x[0], reverse=True)
    if not candidates:
        return (None, "no_match")
    if len(candidates) >= 2 and candidates[0][0] - candidates[1][0] < 0.05:
        return (candidates[0][1], "ambiguous")
    return (candidates[0][1], "ok")


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--atlases", nargs="+", required=True)
    p.add_argument("--out", required=True)
    args = p.parse_args(argv)

    landmarks = json.loads(_LANDMARKS_PATH.read_text(encoding="utf-8"))
    names = _flatten_landmarks(landmarks)

    out: dict[str, dict] = {}
    for name in names:
        out[name] = {}
        for atlas_name in args.atlases:
            atlas = load_atlas(atlas_name)
            acronym, status = _fuzzy_match(name, atlas)
            out[name][atlas_name] = {
                "acronym": acronym,
                "include_descendants": False,  # curator decides
                "status": status,
            }

    Path(args.out).write_text(json.dumps(out, indent=2), encoding="utf-8")
    print(f"Wrote draft map for {len(names)} landmarks to {args.out}")
    print("Review entries with status='ambiguous' or 'no_match' before "
          "saving as landmark_atlas_map.json.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
```

- [ ] **Step 2: Run the helper to draft the map**

```bash
python models/langslice-gemma-4/data/resolve_landmarks.py \
  --atlases allen_mouse_25um whs_sd_rat_39um \
  --out models/langslice-gemma-4/data/landmark_atlas_map.draft.json
```

Expected: prints "Wrote draft map for ~30 landmarks to ...".

Developmental-mouse atlases (BrainGlobe `admba_3d_*`) are intentionally not
curated here — that coverage is provided by the `synth_dataset` augmentation
pipeline (Task 9 / spec §4.2), not by hand-mapped atlas IDs.

- [ ] **Step 3: Curator review**

Open `models/langslice-gemma-4/data/landmark_atlas_map.draft.json`. For each entry:
- `status: "ok"` — accept the acronym; set `include_descendants: true` for known parent regions (e.g., "Hippocampal Formation", "Thalamus", "Hypothalamus", "Cerebellar Cortex - Hemispheric Regions").
- `status: "ambiguous"` — pick the right acronym manually from the atlas's structure tree.
- `status: "no_match"` — either pick an acronym manually or set the per-atlas entry to `null` (this landmark will be skipped for this atlas).

Strip the `status` field after review. Save the cleaned file as `models/langslice-gemma-4/data/landmark_atlas_map.json`.

- [ ] **Step 4: Verify with the resolver**

```bash
python -c "
import sys
from pathlib import Path
sys.path.insert(0, 'models/langslice-gemma-4/data')
sys.path.insert(0, 'src')
from langslice_harness.atlas.core import load_atlas
from landmarks import LandmarkLoader
loader = LandmarkLoader()
atlas = load_atlas('allen_mouse_25um')
for name in loader.landmarks_for_orientation('coronal'):
    ids = loader.resolve(name, atlas, 'allen_mouse_25um')
    print(f'{name!r}: {len(ids)} region IDs')
"
```

Expected: prints each coronal landmark and its resolved region-ID count. Zero counts indicate landmarks unmapped for this atlas — verify intentional.

- [ ] **Step 5: Commit**

```bash
git add models/langslice-gemma-4/data/resolve_landmarks.py models/langslice-gemma-4/data/landmark_atlas_map.json
git commit -m "feat(gemma-4): atlas-map seeding helper and curated map"
```

---

## Task 3: Region-bbox atlas-slice path

**Files:**
- Create: `models/langslice-gemma-4/data/region_bbox.py`
- Create: `tests/test_region_bbox.py`

**Spec sections covered:** §6.1 (Atlas-slice path), §3.1 (Bbox structure).

- [ ] **Step 1: Write the atlas-slice test**

Create `tests/test_region_bbox.py`:

```python
"""Tests for region_bbox: hemisphere-split bbox computation."""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

import region_bbox


def test_atlas_slice_whole_brain_returns_left_and_right():
    # 100-px-wide annotation: region ID {1} appears in both hemispheres.
    annotation = np.zeros((100, 100), dtype=np.int32)
    annotation[20:50, 10:40] = 1   # left side, region 1
    annotation[20:50, 60:90] = 1   # right side, region 1
    hemispheres = np.zeros((100, 100), dtype=np.int32)
    hemispheres[:, :50] = 1
    hemispheres[:, 50:] = 2
    bbox = region_bbox.bbox_from_atlas_slice(
        annotation_slice=annotation,
        hemisphere_slice=hemispheres,
        region_ids={1},
        is_hemisphere=False,
    )
    assert bbox == {
        "left": [10, 20, 39, 49],
        "right": [60, 20, 89, 49],
    }


def test_atlas_slice_whole_brain_drops_when_one_side_empty():
    annotation = np.zeros((100, 100), dtype=np.int32)
    annotation[20:50, 10:40] = 1   # left only
    hemispheres = np.zeros((100, 100), dtype=np.int32)
    hemispheres[:, :50] = 1
    hemispheres[:, 50:] = 2
    bbox = region_bbox.bbox_from_atlas_slice(
        annotation_slice=annotation,
        hemisphere_slice=hemispheres,
        region_ids={1},
        is_hemisphere=False,
    )
    assert bbox is None  # whole-brain example dropped if either side empty


def test_atlas_slice_hemisphere_returns_single_bbox():
    annotation = np.zeros((100, 100), dtype=np.int32)
    annotation[20:50, 10:40] = 1
    bbox = region_bbox.bbox_from_atlas_slice(
        annotation_slice=annotation,
        region_ids={1},
        is_hemisphere=True,
    )
    assert bbox == [10, 20, 39, 49]


def test_atlas_slice_coverage_gate_too_small():
    annotation = np.zeros((100, 100), dtype=np.int32)
    annotation[0, 0] = 1  # 1 pixel out of 10000 = 0.01%, below 1% gate
    bbox = region_bbox.bbox_from_atlas_slice(
        annotation_slice=annotation,
        region_ids={1},
        is_hemisphere=True,
    )
    assert bbox is None


def test_atlas_slice_coverage_gate_too_large():
    annotation = np.full((100, 100), 1, dtype=np.int32)  # 100% of image
    bbox = region_bbox.bbox_from_atlas_slice(
        annotation_slice=annotation,
        region_ids={1},
        is_hemisphere=True,
    )
    assert bbox is None
```

- [ ] **Step 2: Run tests, verify they fail**

```bash
pytest tests/test_region_bbox.py -v
```

Expected: FAIL — `ModuleNotFoundError: No module named 'region_bbox'`.

- [ ] **Step 3: Implement the atlas-slice path**

Create `models/langslice-gemma-4/data/region_bbox.py`:

```python
"""Hemisphere-split bbox computation.

Two backends:
- `bbox_from_atlas_slice`: consumes a 2D annotation array (atlas-aligned).
- `bbox_from_real_section`: projects probes through a per-section pixel→voxel
  registration helper (see `_local/eval/lib/registration.py`).

Coverage gate: the qualifying pixel/probe area must be ≥1% and ≤40% of total
image area, else the bbox fails. Whole-brain coronal/horizontal returns
`{left, right}`; either side empty causes the example to fail (returns None).
Sagittal / hemisphere returns a single bbox.
"""
from __future__ import annotations

from typing import Callable

import numpy as np

_COVERAGE_MIN = 0.01
_COVERAGE_MAX = 0.40


def _bbox_of_mask(mask: np.ndarray) -> list[int] | None:
    """Return [x1, y1, x2, y2] or None if mask is empty."""
    if not mask.any():
        return None
    ys, xs = np.where(mask)
    return [int(xs.min()), int(ys.min()), int(xs.max()), int(ys.max())]


def bbox_from_atlas_slice(
    annotation_slice: np.ndarray,
    region_ids: set[int],
    is_hemisphere: bool,
    hemisphere_slice: np.ndarray | None = None,
) -> dict | list | None:
    """Compute hemisphere-split bbox from a 2D atlas annotation.

    Returns:
        - {"left": [...], "right": [...]} for whole-brain coronal/horizontal.
        - [...] for sagittal or hemisphere sections.
        - None if coverage gate fails or whole-brain has empty side.
    """
    if annotation_slice.ndim != 2:
        raise ValueError(
            f"annotation_slice must be 2D, got shape {annotation_slice.shape}"
        )
    h, w = annotation_slice.shape
    total = h * w

    mask = np.isin(annotation_slice, list(region_ids))
    coverage = float(mask.sum()) / float(total)
    if coverage < _COVERAGE_MIN or coverage > _COVERAGE_MAX:
        return None

    if is_hemisphere:
        return _bbox_of_mask(mask)

    if hemisphere_slice is None:
        raise ValueError("hemisphere_slice is required for whole-brain atlas bboxes")
    if hemisphere_slice.shape != annotation_slice.shape:
        raise ValueError(
            "hemisphere_slice must match annotation_slice shape, got "
            f"{hemisphere_slice.shape} vs {annotation_slice.shape}"
        )

    left_mask = mask & (hemisphere_slice == 1)
    right_mask = mask & (hemisphere_slice == 2)

    left_bbox = _bbox_of_mask(left_mask)
    right_bbox = _bbox_of_mask(right_mask)
    if left_bbox is None or right_bbox is None:
        return None
    return {"left": left_bbox, "right": right_bbox}
```

- [ ] **Step 4: Run tests, verify pass**

```bash
pytest tests/test_region_bbox.py -v
```

Expected: 5 passed.

- [ ] **Step 5: Commit**

```bash
git add models/langslice-gemma-4/data/region_bbox.py tests/test_region_bbox.py
git commit -m "feat(gemma-4): atlas-slice region bbox with hemisphere split"
```

---

## Task 4: Region-bbox real-histology path

**Files:**
- Modify: `models/langslice-gemma-4/data/region_bbox.py`
- Modify: `tests/test_region_bbox.py`

**Spec sections covered:** §6.2 (Real-histology path), §6.3 (Two-pass projection).

- [ ] **Step 1: Add the realhist test**

Append to `tests/test_region_bbox.py`:

```python
def test_real_section_identity_transform_matches_atlas_path():
    """Identity pixel→voxel: realhist path should match atlas-slice path."""
    H, W = 96, 96
    # Tiny BrainGlobe volume is the same as a single annotation slice. The fake
    # QuickNII transform returns (x_ml, y_ap, z_dv), so x=col, AP=0, z=row.
    annotation_volume = np.zeros((1, H, W), dtype=np.int32)
    annotation_volume[0, 20:50, 10:40] = 1   # left
    annotation_volume[0, 20:50, 60:90] = 1   # right

    def pixel_to_voxel(i: float, j: float) -> np.ndarray:
        return np.array([i, 0.0, j], dtype=np.float64)  # QuickNII: (x_ml, y_ap, z_dv)

    def midline_x(i: float) -> float:  # vertical midline at x=W/2 in pixel coords
        return W / 2.0

    bbox = region_bbox.bbox_from_real_section(
        section_image_shape=(H, W),
        pixel_to_voxel=pixel_to_voxel,
        annotation_volume=annotation_volume,
        region_ids={1},
        midline_x_at_row=midline_x,
        is_hemisphere=False,
        grid_step=4,
    )
    assert bbox is not None
    assert "left" in bbox and "right" in bbox
    # With grid_step=4 and ~30x30 region, expect coords near the atlas truth
    # (10,20)-(39,49) and (60,20)-(89,49) within one grid step of padding.
    assert bbox["left"][0] <= 12
    assert bbox["left"][1] <= 22
    assert bbox["left"][2] >= 36
    assert bbox["left"][3] >= 47
    assert bbox["right"][0] <= 62
    assert bbox["right"][2] >= 86


def test_real_section_hemisphere_returns_single_bbox():
    H, W = 96, 96
    annotation_volume = np.zeros((1, H, W), dtype=np.int32)
    annotation_volume[0, 20:50, 10:40] = 1

    def pixel_to_voxel(i: float, j: float) -> np.ndarray:
        return np.array([i, 0.0, j], dtype=np.float64)

    bbox = region_bbox.bbox_from_real_section(
        section_image_shape=(H, W),
        pixel_to_voxel=pixel_to_voxel,
        annotation_volume=annotation_volume,
        region_ids={1},
        midline_x_at_row=lambda i: W / 2.0,
        is_hemisphere=True,
        grid_step=4,
    )
    assert isinstance(bbox, list)
    assert len(bbox) == 4


def test_real_section_coverage_gate_drops_too_small():
    H, W = 96, 96
    annotation_volume = np.zeros((1, H, W), dtype=np.int32)
    annotation_volume[0, 0, 0] = 1  # 1 voxel only

    def pixel_to_voxel(i: float, j: float) -> np.ndarray:
        return np.array([i, 0.0, j], dtype=np.float64)

    bbox = region_bbox.bbox_from_real_section(
        section_image_shape=(H, W),
        pixel_to_voxel=pixel_to_voxel,
        annotation_volume=annotation_volume,
        region_ids={1},
        midline_x_at_row=lambda i: W / 2.0,
        is_hemisphere=True,
        grid_step=4,
    )
    assert bbox is None


def test_quicknii_to_brainglobe_indices_converts_axis_order():
    # QuickNII pixel_to_voxel returns (x_ml, y_ap, z_dv); BrainGlobe annotation
    # arrays are indexed as (AP, DV, ML).
    assert region_bbox.quicknii_to_brainglobe_indices(
        np.array([7.2, 3.0, 5.9], dtype=np.float64)
    ) == (3, 6, 7)


def test_scale_bbox_handles_split_and_single_boxes():
    split = {"left": [10, 20, 30, 40], "right": [60, 20, 80, 40]}
    assert region_bbox.scale_bbox(split, scale_x=0.5, scale_y=0.25) == {
        "left": [5, 5, 15, 10],
        "right": [30, 5, 40, 10],
    }
    assert region_bbox.scale_bbox([10, 20, 30, 40], scale_x=0.5, scale_y=0.25) == [
        5, 5, 15, 10,
    ]
```

- [ ] **Step 2: Run new tests, verify they fail**

```bash
pytest tests/test_region_bbox.py -v -k real_section
```

Expected: FAIL — `AttributeError: module 'region_bbox' has no attribute 'bbox_from_real_section'`.

- [ ] **Step 3: Implement the realhist path**

Append to `models/langslice-gemma-4/data/region_bbox.py`:

```python
def bbox_from_real_section(
    section_image_shape: tuple[int, int],
    pixel_to_voxel: Callable[[float, float], np.ndarray],
    annotation_volume: np.ndarray,
    region_ids: set[int],
    midline_x_at_row: Callable[[float], float],
    is_hemisphere: bool,
    grid_step: int = 8,
) -> dict | list | None:
    """Compute hemisphere-split bbox by projecting probes into atlas voxel space.

    Args:
        section_image_shape: (H, W) of the section image (post-resize).
        pixel_to_voxel: returns QuickNII voxel coords as (x_ml, y_ap, z_dv)
            for a given section pixel (i, j). This matches
            `_local/eval/lib/registration.py::pixel_to_voxel`.
        annotation_volume: 3D BrainGlobe annotation, shape (AP, DV, ML).
        region_ids: BrainGlobe region IDs to match (use `LandmarkLoader.resolve`).
        midline_x_at_row: returns the pixel x-coord of the projected atlas
            midline at a given row i. For atlas-symmetric sections, a simple
            constant `lambda _: W/2`.
        is_hemisphere: if True, return a single bbox; otherwise hemisphere-split.
        grid_step: probe stride in pixels.

    Returns:
        Same shape as `bbox_from_atlas_slice`. None on coverage-gate failure
        or empty hemisphere on whole-brain.
    """
    H, W = section_image_shape
    AP, DV, ML = annotation_volume.shape

    is_left = np.zeros((H, W), dtype=bool)
    is_match = np.zeros((H, W), dtype=bool)

    for i in range(0, H, grid_step):
        midline_at_i = float(midline_x_at_row(float(i)))
        for j in range(0, W, grid_step):
            voxel = pixel_to_voxel(float(j), float(i))
            apv, dvv, mlv = quicknii_to_brainglobe_indices(voxel)
            if not (0 <= apv < AP and 0 <= dvv < DV and 0 <= mlv < ML):
                continue
            ann = int(annotation_volume[apv, dvv, mlv])
            if ann in region_ids:
                is_match[i, j] = True
                if j < midline_at_i:
                    is_left[i, j] = True

    n_match = int(is_match.sum())
    n_probes = (H // grid_step) * (W // grid_step)
    if n_probes == 0:
        return None
    coverage = n_match / float(n_probes)
    if coverage < _COVERAGE_MIN or coverage > _COVERAGE_MAX:
        return None

    def _padded_bbox(mask: np.ndarray) -> list[int] | None:
        bbox = _bbox_of_mask(mask)
        if bbox is None:
            return None
        x1, y1, x2, y2 = bbox
        x1 = max(0, x1 - grid_step)
        y1 = max(0, y1 - grid_step)
        x2 = min(W - 1, x2 + grid_step)
        y2 = min(H - 1, y2 + grid_step)
        return [x1, y1, x2, y2]

    if is_hemisphere:
        return _padded_bbox(is_match)

    left_mask = is_match & is_left
    right_mask = is_match & ~is_left

    left_bbox = _padded_bbox(left_mask)
    right_bbox = _padded_bbox(right_mask)
    if left_bbox is None or right_bbox is None:
        return None
    return {"left": left_bbox, "right": right_bbox}


def quicknii_to_brainglobe_indices(voxel: np.ndarray) -> tuple[int, int, int]:
    """Convert QuickNII (x_ml, y_ap, z_dv) to BrainGlobe (AP, DV, ML)."""
    x_ml, y_ap, z_dv = float(voxel[0]), float(voxel[1]), float(voxel[2])
    return (int(round(y_ap)), int(round(z_dv)), int(round(x_ml)))


def scale_bbox(bbox: dict | list, scale_x: float, scale_y: float) -> dict | list:
    """Scale bbox coordinates into a resized image frame."""
    if isinstance(bbox, dict):
        return {
            side: scale_bbox(coords, scale_x=scale_x, scale_y=scale_y)
            for side, coords in bbox.items()
            if coords is not None
        }
    x1, y1, x2, y2 = bbox
    return [
        int(round(x1 * scale_x)),
        int(round(y1 * scale_y)),
        int(round(x2 * scale_x)),
        int(round(y2 * scale_y)),
    ]
```

- [ ] **Step 4: Run all region_bbox tests, verify pass**

```bash
pytest tests/test_region_bbox.py -v
```

Expected: 10 passed (5 atlas + 3 realhist + axis conversion + bbox scaling).

- [ ] **Step 5: Commit**

```bash
git add models/langslice-gemma-4/data/region_bbox.py tests/test_region_bbox.py
git commit -m "feat(gemma-4): real-histology bbox via probe-grid projection"
```

---

## Task 5: Landmark coverage bake

**Files:**
- Create: `_local/eval/lib/landmark_coverage.py`

**Spec sections covered:** §6.3 (Two-pass projection — coarse coverage scan).

This module lives under `_local/` (gitignored) per the spec. Use a local pytest smoke test first so the coverage-bake logic still follows the plan's TDD pattern; do not commit the local test.

- [ ] **Step 1: Write the local failing test**

Create `_local/eval/lib/test_landmark_coverage.py`:

```python
"""Local smoke tests for landmark_coverage.py; gitignored with _local/."""
from __future__ import annotations

import numpy as np

import landmark_coverage


def test_coarse_coverage_converts_quicknii_to_brainglobe_order():
    annotation = np.zeros((1, 8, 8), dtype=np.int32)  # BrainGlobe (AP, DV, ML)
    annotation[0, 2:6, 2:6] = 7

    def pixel_to_voxel(i: float, j: float) -> np.ndarray:
        return np.array([i, 0.0, j], dtype=np.float64)  # QuickNII (x_ml, y_ap, z_dv)

    assert landmark_coverage._coarse_coverage(
        pixel_to_voxel=pixel_to_voxel,
        image_shape=(8, 8),
        annotation_volume=annotation,
        region_ids={7},
        grid_step=1,
    ) > 0.0
```

Run it and verify it fails because `landmark_coverage.py` does not exist yet:

```bash
pytest _local/eval/lib/test_landmark_coverage.py -v
```

- [ ] **Step 2: Implement the bake**

Create `_local/eval/lib/landmark_coverage.py`:

```python
"""One-time coarse coverage scan: for each (real-histology brain × landmark),
record which sections contain the landmark above a threshold.

Reads the manifest at `_local/eval/data/manifest.jsonl`, projects a coarse probe
grid through every Tier A real-histology section's `pixel_to_voxel`, counts
region-matching probes, and emits `_local/eval/data/landmark_coverage.json`.

Output shape:

```json
{
  "<brain_id>": {
    "<landmark>": {
      "atlas": "allen_mouse_25um",
      "orientation": "coronal",
      "section_ids": ["<section_id>", ...]
    }
  }
}
```

Re-bake when manifest, landmark map, or registration helper changes.
"""
from __future__ import annotations

import json
import logging
import sys
from collections import defaultdict
from pathlib import Path
from typing import Iterable

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO_ROOT / "_local" / "eval" / "lib"))
sys.path.insert(0, str(REPO_ROOT / "models" / "langslice-gemma-4" / "data"))
sys.path.insert(0, str(REPO_ROOT / "src"))

from registration import section_to_atlas_voxel  # noqa: E402
from langslice_harness.atlas.core import load_atlas  # noqa: E402
from landmarks import LandmarkLoader  # noqa: E402
import region_bbox  # noqa: E402

log = logging.getLogger("landmark_coverage")

_DEFAULT_MANIFEST = REPO_ROOT / "_local" / "eval" / "data" / "manifest.jsonl"
_DEFAULT_OUT = REPO_ROOT / "_local" / "eval" / "data" / "landmark_coverage.json"
_COARSE_GRID = 32          # one probe per 32×32 pixel block — fast pass
_COVERAGE_THRESHOLD = 0.01  # ≥1% probes match → section is "covered"

# Tier A datasets (from the registration audit). Other datasets are skipped
# because their per-section nonlinear registration is not recoverable.
_TIER_A_DATASETS = {
    "zenodo_pnnpv",
    "rat_tract_eval",
    "rat_tract_tracing_rlvr",
    "timm_nissl",
}


def _iter_tier_a_records(manifest_path: Path) -> Iterable[dict]:
    with manifest_path.open("r", encoding="utf-8") as f:
        for line in f:
            rec = json.loads(line)
            if rec.get("dataset") in _TIER_A_DATASETS:
                yield rec


def _coarse_coverage(
    pixel_to_voxel,
    image_shape: tuple[int, int],
    annotation_volume: np.ndarray,
    region_ids: set[int],
    grid_step: int = _COARSE_GRID,
) -> float:
    """Return fraction of probes whose annotation matches region_ids."""
    H, W = image_shape
    AP, DV, ML = annotation_volume.shape
    n_match = 0
    n_total = 0
    for i in range(0, H, grid_step):
        for j in range(0, W, grid_step):
            voxel = pixel_to_voxel(float(j), float(i))
            apv, dvv, mlv = region_bbox.quicknii_to_brainglobe_indices(voxel)
            n_total += 1
            if not (0 <= apv < AP and 0 <= dvv < DV and 0 <= mlv < ML):
                continue
            if int(annotation_volume[apv, dvv, mlv]) in region_ids:
                n_match += 1
    return float(n_match) / float(n_total) if n_total else 0.0


def bake(
    manifest_path: Path = _DEFAULT_MANIFEST,
    out_path: Path = _DEFAULT_OUT,
) -> None:
    loader = LandmarkLoader()
    atlas_cache: dict[str, object] = {}
    annotation_cache: dict[str, np.ndarray] = {}

    out: dict[str, dict] = defaultdict(dict)

    for rec in _iter_tier_a_records(manifest_path):
        atlas_name = rec["atlas"]
        if atlas_name not in atlas_cache:
            atlas = load_atlas(atlas_name)
            atlas_cache[atlas_name] = atlas
            annotation_cache[atlas_name] = atlas.annotation
        atlas = atlas_cache[atlas_name]
        annotation_volume = annotation_cache[atlas_name]

        slice_record = rec.get("slice_record")
        if slice_record is None:
            log.debug("skipping %s: no slice_record (anchoring missing)", rec.get("section_id"))
            continue
        try:
            pixel_to_voxel = section_to_atlas_voxel(slice_record, atlas)
        except Exception as exc:
            log.warning("registration failed for %s: %s", rec.get("section_id"), exc)
            continue

        # Image shape: prefer the actual on-disk image shape over manifest hint.
        H = int(slice_record.get("height") or 0)
        W = int(slice_record.get("width") or 0)
        if H == 0 or W == 0:
            log.warning("skipping %s: missing height/width", rec.get("section_id"))
            continue

        orientation = rec.get("orientation", "coronal")
        landmarks_for_orient = loader.landmarks_for_orientation(orientation)

        for landmark_name in landmarks_for_orient:
            region_ids = loader.resolve(landmark_name, atlas, atlas_name)
            if not region_ids:
                continue
            coverage = _coarse_coverage(
                pixel_to_voxel=pixel_to_voxel,
                image_shape=(H, W),
                annotation_volume=annotation_volume,
                region_ids=region_ids,
            )
            if coverage < _COVERAGE_THRESHOLD:
                continue
            brain_id = rec.get("subject_id") or rec.get("brain_id") or "unknown"
            entry = out[brain_id].setdefault(
                landmark_name,
                {"atlas": atlas_name, "orientation": orientation, "section_ids": []},
            )
            entry["section_ids"].append(rec["section_id"])

    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(out, indent=2), encoding="utf-8")
    log.info("Wrote coverage index for %d brains to %s", len(out), out_path)


def load_coverage_index(path: Path = _DEFAULT_OUT) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    bake()
```

- [ ] **Step 3: Run the local test, verify pass**

```bash
pytest _local/eval/lib/test_landmark_coverage.py -v
```

Expected: 1 passed.

- [ ] **Step 4: Smoke-run the bake**

```bash
python _local/eval/lib/landmark_coverage.py
```

Expected: prints "Wrote coverage index for N brains to ...", where N is approximately 17 (the Tier A brain count). Inspect `_local/eval/data/landmark_coverage.json`; spot-check that "Hippocampal Formation" lists section IDs for at least one mouse-coronal brain.

- [ ] **Step 5: No commit**

This module is `_local/`, gitignored.

---

## Task 6: Widen `vlm_config.supports_batch_api()` for AI Studio

**Files:**
- Modify: `src/langslice_harness/vlm_config.py:335-336` (the `supports_batch_api` function)
- Create or Modify: `tests/test_vlm_config_batch.py`
- Modify: `tests/test_vlm_features.py` (existing AI Studio assertion)

**Spec sections covered:** §9 (Gemini Batch API path; AI Studio backend confirmed).

- [ ] **Step 1: Write the test**

Create `tests/test_vlm_config_batch.py`:

```python
"""Verify supports_batch_api() now allows AI Studio in addition to Vertex."""
from __future__ import annotations

import os
from unittest.mock import patch

from langslice_harness import vlm_config


def test_supports_batch_api_ai_studio():
    with patch.dict(os.environ, {"LANGSLICE_GENAI_BACKEND": "ai_studio"}, clear=False):
        assert vlm_config.supports_batch_api() is True


def test_supports_batch_api_vertex_adc():
    with patch.dict(os.environ, {"LANGSLICE_GENAI_BACKEND": "vertex_adc"}, clear=False):
        assert vlm_config.supports_batch_api() is True


def test_supports_batch_api_vertex_api_key_unsupported():
    # Vertex API-key mode does not currently route batch - keep false.
    with patch.dict(os.environ, {"LANGSLICE_GENAI_BACKEND": "vertex_api_key"}, clear=False):
        assert vlm_config.supports_batch_api() is False
```

- [ ] **Step 2: Update the existing feature assertion**

In `tests/test_vlm_features.py`, find the AI Studio case that currently asserts `supports_batch_api` is false. Flip it to true, or relax the assertion to accept AI Studio batch support explicitly while keeping `vertex_api_key` unsupported.

- [ ] **Step 3: Run test, verify the AI Studio case fails**

```bash
pytest tests/test_vlm_config_batch.py -v
pytest tests/test_vlm_features.py -v -k batch
```

Expected: `test_supports_batch_api_ai_studio` FAILS (current gate is Vertex-only).

- [ ] **Step 4: Widen the gate**

Edit `src/langslice_harness/vlm_config.py`. Find this block (around line 335):

```python
def supports_batch_api() -> bool:
    return get_backend() == _BACKEND_VERTEX_ADC
```

Replace with:

```python
def supports_batch_api() -> bool:
    backend = get_backend()
    return backend in (_BACKEND_AI_STUDIO, _BACKEND_VERTEX_ADC)
```

- [ ] **Step 5: Run tests, verify pass**

```bash
pytest tests/test_vlm_config_batch.py -v
pytest tests/test_vlm_features.py -v -k batch
```

Expected: 3 passed.

- [ ] **Step 6: Commit**

```bash
git add src/langslice_harness/vlm_config.py tests/test_vlm_config_batch.py tests/test_vlm_features.py
git commit -m "feat(vlm_config): allow AI Studio backend for Batch API"
```

---

## Task 7: bbox_io — schemas, packing, mm-strip filter

**Files:**
- Create: `models/langslice-gemma-4/data/bbox_io.py`
- Create: `tests/test_bbox_io.py`

**Spec sections covered:** §3 (Example shape), §9.1 (Wire format), §10 (Validation gates), §11 (Outputs).

- [ ] **Step 1: Write the schema + filter tests**

Create `tests/test_bbox_io.py`:

```python
"""Tests for bbox_io: manifest schema, batch packing, mm-strip filter."""
from __future__ import annotations

from pathlib import Path

import pytest

import bbox_io


def test_mm_strip_passes_clean_caption():
    text = "The hippocampus appears as a thin crescent and broadens caudally."
    assert bbox_io.mm_strip_filter(text) == text


def test_mm_strip_rejects_mm_value():
    text = "At 5.4 mm the hippocampus appears..."
    assert bbox_io.mm_strip_filter(text) is None


def test_mm_strip_rejects_micron_value():
    text = "The dentate gyrus expands by 200 μm caudally."
    assert bbox_io.mm_strip_filter(text) is None


def test_mm_strip_rejects_section_index():
    text = "In Section 3, the region begins to broaden."
    assert bbox_io.mm_strip_filter(text) is None


def test_mm_strip_rejects_bregma():
    text = "Posterior to bregma, the region narrows."
    assert bbox_io.mm_strip_filter(text) is None


def test_length_gate_too_short():
    text = "It broadens."
    assert bbox_io.length_gate(text) is False


def test_length_gate_ok():
    text = (
        "The hippocampus appears as a thin crescent. It broadens caudally "
        "and reveals dentate gyrus subdivisions."
    )
    assert bbox_io.length_gate(text) is True


def test_length_gate_too_long():
    sentences = ["This is a sentence."] * 7
    text = " ".join(sentences)
    assert bbox_io.length_gate(text) is False


def test_pack_batch_request_whole_brain():
    example = {
        "id": "bbox_000001",
        "atlas": "allen_mouse_25um",
        "orientation": "coronal",
        "region": "Hippocampal Formation",
        "is_hemisphere": False,
        "section_image_paths": [],   # the test stubs PNG bytes via inject
        "bboxes": [
            {"left": [10, 20, 30, 40], "right": [60, 20, 80, 40]},
            {"left": [12, 22, 32, 42], "right": [62, 22, 82, 42]},
            {"left": [14, 24, 34, 44], "right": [64, 24, 84, 44]},
            {"left": [16, 26, 36, 46], "right": [66, 26, 86, 46]},
        ],
    }
    images_b64 = ["AAAA", "BBBB", "CCCC", "DDDD"]
    line = bbox_io.pack_batch_request(example, images_b64=images_b64)
    assert line["key"] == "bbox_000001"
    contents = line["request"]["contents"]
    assert len(contents) == 1
    parts = contents[0]["parts"]
    # 4 images + 1 text part.
    assert sum(1 for p in parts if "inline_data" in p) == 4
    text_parts = [p for p in parts if "text" in p]
    assert len(text_parts) == 1
    text = text_parts[0]["text"]
    assert "Hippocampal Formation" in text
    assert "left=[10, 20, 30, 40]" in text
    assert "right=[60, 20, 80, 40]" in text
    assert "anterior to posterior" in text
    assert "millimeter" in text.lower() or "Do NOT mention" in text
    config = line["request"]["generation_config"]
    assert config["thinking_config"]["thinking_level"] == "LOW"
    assert config["temperature"] == 1.0
    assert "max_output_tokens" not in config


def test_pack_batch_request_sagittal_single_bbox():
    example = {
        "id": "bbox_000002",
        "atlas": "allen_mouse_25um",
        "orientation": "sagittal",
        "region": "Dentate Gyrus",
        "is_hemisphere": True,
        "section_image_paths": [],
        "bboxes": [
            [220, 140, 410, 320],
            [218, 138, 408, 318],
            [216, 136, 406, 316],
            [214, 134, 404, 314],
        ],
    }
    images_b64 = ["A", "B", "C", "D"]
    line = bbox_io.pack_batch_request(example, images_b64=images_b64)
    text = next(p["text"] for p in line["request"]["contents"][0]["parts"] if "text" in p)
    assert "medial to lateral" in text
    assert "[220, 140, 410, 320]" in text
    assert "left=" not in text
    assert "right=" not in text
```

- [ ] **Step 2: Run tests, verify they fail**

```bash
pytest tests/test_bbox_io.py -v
```

Expected: FAIL — `ModuleNotFoundError: No module named 'bbox_io'`.

- [ ] **Step 3: Implement bbox_io**

Create `models/langslice-gemma-4/data/bbox_io.py`:

```python
"""I/O helpers for the BBox training data pipeline.

Responsibilities:
- Manifest schema (draft + final SFT).
- Pack a draft example into a Gemini Batch API request line.
- mm-strip and length-gate filters on Gemini responses.
"""
from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Iterable

# ---------------------------------------------------------------------------
# Filters

_MM_PATTERNS = [
    re.compile(r"\b\d+(\.\d+)?\s*(mm|μm|um|microns?)\b", re.IGNORECASE),
    re.compile(r"\b(section|slice)\s*\d+\b", re.IGNORECASE),
    re.compile(r"\bbregma\b", re.IGNORECASE),
]


def mm_strip_filter(text: str) -> str | None:
    """Return text if no mm/coordinate/section-index leakage; else None."""
    for pat in _MM_PATTERNS:
        if pat.search(text):
            return None
    return text


def length_gate(text: str, min_sentences: int = 2, max_sentences: int = 6) -> bool:
    """Approximate sentence count via terminal punctuation."""
    n = sum(1 for ch in text if ch in ".!?")
    return min_sentences <= n <= max_sentences


# ---------------------------------------------------------------------------
# Prompt building

_SYSTEM_PROMPT = (
    "You are an expert neuroanatomist describing brain section morphology to "
    "a student. You describe how anatomical regions transform in shape, size, "
    "and boundary characteristics as the brain is sectioned along its "
    "anatomical axis. You use precise but accessible language. You never "
    "reference millimeter values, atlas coordinates, position numbers, or "
    "section indices."
)

_AXIS_BY_ORIENTATION = {
    "coronal": "anterior to posterior",
    "sagittal": "medial to lateral",
    "horizontal": "dorsal to ventral",
}

_USER_TEMPLATE_INSTRUCTIONS = (
    "Describe how this region transforms across the sections — its shape, "
    "size, boundary characteristics, and any sub-divisions visible. The "
    "bounding boxes are provided so you can locate the region in each "
    "section; you do not need to reference them in your description.\n\n"
    "Do NOT mention millimeter values, section indices, atlas coordinates, "
    "or position numbers. 2-4 sentences."
)


def _format_bboxes(example: dict) -> str:
    lines = []
    if example["is_hemisphere"]:
        lines.append("Per-section bounding boxes in section pixel coords [x1, y1, x2, y2]:")
        for idx, bbox in enumerate(example["bboxes"], start=1):
            lines.append(f"  Section {idx}: {bbox}")
    else:
        lines.append(
            "Per-section bounding boxes (left, right hemisphere) in section "
            "pixel coords [x1, y1, x2, y2]:"
        )
        for idx, pair in enumerate(example["bboxes"], start=1):
            lines.append(
                f"  Section {idx}: left={pair['left']}, right={pair['right']}"
            )
    return "\n".join(lines)


def build_user_text(example: dict) -> str:
    axis = _AXIS_BY_ORIENTATION[example["orientation"]]
    bboxes = _format_bboxes(example)
    return (
        f"Atlas: {example['atlas']}\n"
        f"Plane: {example['orientation']}\n"
        f"Region: {example['region']}\n"
        f"Section ordering: {axis}\n\n"
        f"{bboxes}\n\n"
        f"{_USER_TEMPLATE_INSTRUCTIONS}"
    )


def pack_batch_request(example: dict, images_b64: list[str]) -> dict:
    """Convert one draft example + N base64 PNGs into one Batch API request line."""
    image_parts = [
        {"inline_data": {"mime_type": "image/png", "data": b64}}
        for b64 in images_b64
    ]
    text_part = {"text": build_user_text(example)}
    return {
        "key": example["id"],
        "request": {
            "system_instruction": {"parts": [{"text": _SYSTEM_PROMPT}]},
            "contents": [
                {"role": "user", "parts": image_parts + [text_part]}
            ],
            "generation_config": {
                "response_modalities": ["TEXT"],
                "temperature": 1.0,
                "thinking_config": {"thinking_level": "LOW"},
            },
        },
    }


# ---------------------------------------------------------------------------
# Manifest IO

def write_jsonl(path: Path, records: Iterable[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for rec in records:
            f.write(json.dumps(rec, ensure_ascii=True) + "\n")


def read_jsonl(path: Path) -> Iterable[dict]:
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                yield json.loads(line)


# ---------------------------------------------------------------------------
# Response extraction

def extract_caption_from_response(response: dict) -> str | None:
    """Walk parts, drop thought parts, join text.

    Thought parts (`thought: true`) should only appear when
    `include_thoughts=True` is set, but this defensive filter prevents any
    accidental thought-part exposure from entering the SFT assistant content.
    """
    candidates = response.get("candidates", [])
    if not candidates:
        return None
    parts = candidates[0].get("content", {}).get("parts", [])
    answer_parts = [p.get("text", "") for p in parts if not p.get("thought")]
    text = "".join(answer_parts).strip()
    return text or None
```

- [ ] **Step 4: Run tests, verify pass**

```bash
pytest tests/test_bbox_io.py -v
```

Expected: 9 passed.

- [ ] **Step 5: Commit**

```bash
git add models/langslice-gemma-4/data/bbox_io.py tests/test_bbox_io.py
git commit -m "feat(gemma-4): bbox_io schema, batch packing, mm-strip filter"
```

---

## Task 8: build_bbox_data — `--stage sample` core

**Files:**
- Create: `models/langslice-gemma-4/data/build_bbox_data.py`
- Create: `tests/test_build_bbox_data.py`

**Spec sections covered:** §3.2 (variability), §3.3 (single source per example), §4 (source corpus), §4.4 (priority logic), §8.1 stage 1.

- [ ] **Step 1: Write the source-priority test**

Create `tests/test_build_bbox_data.py`:

```python
"""Tests for build_bbox_data orchestrator (stage 'sample')."""
from __future__ import annotations

import json
from pathlib import Path

import build_bbox_data
import numpy as np
import pytest
from PIL import Image

REPO_ROOT = Path(__file__).resolve().parents[1]


def test_pick_source_prefers_real_when_eligible():
    coverage_index = {
        "AL1A": {
            "Hippocampal Formation": {
                "atlas": "allen_mouse_25um",
                "orientation": "coronal",
                "section_ids": ["AL1A:s001", "AL1A:s005", "AL1A:s010", "AL1A:s015"],
            }
        }
    }
    rng = np.random.default_rng(42)
    decision = build_bbox_data.pick_source(
        atlas="allen_mouse_25um",
        orientation="coronal",
        landmark="Hippocampal Formation",
        coverage_index=coverage_index,
        rng=rng,
    )
    assert decision["source_type"] == "real_histology"
    assert decision["source_brain"] == "AL1A"


def test_pick_source_falls_through_when_real_thin():
    coverage_index = {
        "AL1A": {
            "Hippocampal Formation": {
                "atlas": "allen_mouse_25um",
                "orientation": "coronal",
                "section_ids": ["AL1A:s001"],  # only 1 — below ≥4 gate
            }
        }
    }
    rng = np.random.default_rng(42)
    decision = build_bbox_data.pick_source(
        atlas="allen_mouse_25um",
        orientation="coronal",
        landmark="Hippocampal Formation",
        coverage_index=coverage_index,
        rng=rng,
    )
    assert decision["source_type"] in {"augmented_atlas", "reference_atlas"}
    assert decision["source_brain"] is None


def test_pick_source_respects_per_brain_region_cap():
    coverage_index = {
        "AL1A": {
            "Hippocampal Formation": {
                "atlas": "allen_mouse_25um",
                "orientation": "coronal",
                "section_ids": ["AL1A:s001", "AL1A:s005", "AL1A:s010", "AL1A:s015"],
            }
        }
    }
    rng = np.random.default_rng(42)
    decision = build_bbox_data.pick_source(
        atlas="allen_mouse_25um",
        orientation="coronal",
        landmark="Hippocampal Formation",
        coverage_index=coverage_index,
        rng=rng,
        source_counts={("AL1A", "Hippocampal Formation"): 3},
    )
    assert decision["source_type"] in {"augmented_atlas", "reference_atlas"}
    assert decision["source_brain"] is None


def test_sample_section_count_is_in_range():
    rng = np.random.default_rng(0)
    counts = [build_bbox_data.sample_section_count(rng) for _ in range(2000)]
    assert min(counts) == 4
    assert max(counts) == 8
    assert 4 in counts and 8 in counts and 5 in counts and 6 in counts and 7 in counts


def test_sample_spacings_mm_are_independent_and_in_range():
    rng = np.random.default_rng(0)
    samples = [build_bbox_data.sample_spacings_mm(rng, n_gaps=5) for _ in range(2000)]
    flat = [gap for gaps in samples for gap in gaps]
    assert all(len(gaps) == 5 for gaps in samples)
    assert min(flat) >= 0.2
    assert max(flat) <= 0.8
    assert any(len({round(gap, 6) for gap in gaps}) > 1 for gaps in samples)


def test_sample_anchor_returns_none_when_extent_too_narrow():
    """If region's mm extent can't fit minimum span ((4-1)×0.2=0.6 mm), reject."""
    rng = np.random.default_rng(0)
    anchor = build_bbox_data.sample_anchor_mm(
        rng=rng, region_mm_min=2.0, region_mm_max=2.5,
        spacings_mm=[0.2, 0.2, 0.2],
    )
    assert anchor is None


def test_sample_anchor_returns_value_when_extent_fits():
    rng = np.random.default_rng(0)
    anchor = build_bbox_data.sample_anchor_mm(
        rng=rng, region_mm_min=2.0, region_mm_max=6.0,
        spacings_mm=[0.2, 0.3, 0.4],
    )
    # sum([0.2, 0.3, 0.4]) = 0.9 mm span; valid anchor range is [2.0, 6.0-0.9=5.1]
    assert anchor is not None
    assert 2.0 <= anchor <= 5.1
```

- [ ] **Step 2: Run tests, verify they fail**

```bash
pytest tests/test_build_bbox_data.py -v
```

Expected: FAIL — `ModuleNotFoundError: No module named 'build_bbox_data'`.

- [ ] **Step 3: Implement the orchestrator core**

Create `models/langslice-gemma-4/data/build_bbox_data.py`:

```python
"""Orchestrator for the BBox training data pipeline.

Two stages, both via this script's CLI:
- `--stage sample` writes a draft manifest + bbox-overlay PNGs (no API calls).
- `--stage submit --verdicts <path>` filters by QC verdicts, packs batch JSONL,
   submits to AI Studio Batch API, polls, retrieves, mm-strips, writes final SFT.

Spec: docs/superpowers/specs/2026-05-04-gemma4-bbox-training-design.md
"""
from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path
from typing import Any, Sequence

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO_ROOT / "src"))
sys.path.insert(0, str(REPO_ROOT / "models" / "langslice-gemma-4" / "data"))
sys.path.insert(0, str(REPO_ROOT / "_local" / "eval" / "lib"))

log = logging.getLogger("build_bbox_data")

_MIN_REAL_SECTIONS_FOR_ELIGIBILITY = 4
_MAX_PER_BRAIN_REGION = 3
_SECTION_COUNTS = (4, 5, 6, 7, 8)
_SPACING_MIN_MM = 0.2
_SPACING_MAX_MM = 0.8


def pick_source(
    *,
    atlas: str,
    orientation: str,
    landmark: str,
    coverage_index: dict,
    rng: np.random.Generator,
    source_counts: dict[tuple[str, str], int] | None = None,
) -> dict:
    """Decide source type for one example.

    Spec §4.4: real Tier A if coverage index reports any brain with ≥4 covered
    sections for this (atlas, orientation, landmark) tuple; else split between
    augmented and reference atlas with equal probability.
    """
    eligible_brains: list[tuple[str, list[str]]] = []
    for brain_id, by_landmark in coverage_index.items():
        entry = by_landmark.get(landmark)
        if entry is None:
            continue
        if entry["atlas"] != atlas or entry["orientation"] != orientation:
            continue
        section_ids = entry.get("section_ids", [])
        if source_counts is not None:
            if source_counts.get((brain_id, landmark), 0) >= _MAX_PER_BRAIN_REGION:
                continue
        if len(section_ids) >= _MIN_REAL_SECTIONS_FOR_ELIGIBILITY:
            eligible_brains.append((brain_id, section_ids))

    if eligible_brains:
        weights = np.array([len(secs) for _, secs in eligible_brains], dtype=np.float64)
        weights /= weights.sum()
        idx = int(rng.choice(len(eligible_brains), p=weights))
        brain_id, section_ids = eligible_brains[idx]
        return {
            "source_type": "real_histology",
            "source_brain": brain_id,
            "available_section_ids": section_ids,
        }

    # Atlas fallback — equal split between augmented and reference.
    if rng.random() < 0.5:
        return {"source_type": "augmented_atlas", "source_brain": None}
    return {"source_type": "reference_atlas", "source_brain": None}


def sample_section_count(rng: np.random.Generator) -> int:
    return int(rng.choice(np.array(_SECTION_COUNTS, dtype=np.int64)))


def sample_spacings_mm(rng: np.random.Generator, n_gaps: int) -> list[float]:
    """Sample one independent continuous spacing for each adjacent-section gap."""
    return [
        float(rng.uniform(_SPACING_MIN_MM, _SPACING_MAX_MM))
        for _ in range(n_gaps)
    ]


def sample_anchor_mm(
    *,
    rng: np.random.Generator,
    region_mm_min: float,
    region_mm_max: float,
    spacings_mm: Sequence[float],
) -> float | None:
    """Anchor for the first-section position. Returns None if span doesn't fit."""
    span = sum(spacings_mm)
    max_anchor = region_mm_max - span
    if max_anchor < region_mm_min:
        return None
    return float(rng.uniform(region_mm_min, max_anchor))


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description="BBox training data orchestrator")
    p.add_argument("--stage", required=True, choices=("sample", "submit"))
    p.add_argument("--target-total", type=int, default=300)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument(
        "--out-dir",
        type=Path,
        default=REPO_ROOT / "_local" / "bbox_data",
    )
    p.add_argument(
        "--coverage-index",
        type=Path,
        default=REPO_ROOT / "_local" / "eval" / "data" / "landmark_coverage.json",
    )
    p.add_argument("--verdicts", type=Path, default=None,
                   help="QC verdicts JSONL (required for --stage submit)")
    p.add_argument("--model", default="gemini-3.1-pro-preview")
    p.add_argument("--dry-run", action="store_true",
                   help="Build batch JSONL but don't submit")
    return p.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    logging.basicConfig(level=logging.INFO)
    if args.stage == "sample":
        from _stage_sample import run_stage_sample  # type: ignore
        return run_stage_sample(args)
    if args.stage == "submit":
        if args.verdicts is None:
            log.error("--verdicts is required for --stage submit")
            return 2
        from _stage_submit import run_stage_submit  # type: ignore
        return run_stage_submit(args)
    return 2


if __name__ == "__main__":
    sys.exit(main())
```

- [ ] **Step 4: Run tests, verify pass**

```bash
pytest tests/test_build_bbox_data.py -v
```

Expected: 7 passed.

- [ ] **Step 5: Commit**

```bash
git add models/langslice-gemma-4/data/build_bbox_data.py tests/test_build_bbox_data.py
git commit -m "feat(gemma-4): build_bbox_data orchestrator core (source priority + sampling)"
```

---

## Task 9: build_bbox_data — `--stage sample` end-to-end

**Files:**
- Create: `models/langslice-gemma-4/data/_stage_sample.py`
- Modify: `tests/test_build_bbox_data.py`

**Spec sections covered:** §8.1 (Two-stage flow), §11 (Outputs), §6 (bbox computation invocation).

- [ ] **Step 1: Add the end-to-end test**

Append to `tests/test_build_bbox_data.py`:

```python
def test_stage_sample_writes_draft_manifest(tmp_path: Path, monkeypatch):
    """End-to-end: stage 1 produces a valid draft manifest from a stubbed coverage index."""
    coverage_index_path = tmp_path / "coverage.json"
    coverage_index_path.write_text(json.dumps({}), encoding="utf-8")

    out_dir = tmp_path / "bbox_data"

    # Minimal stub: monkeypatch the heavy renderers to return a tiny atlas slice
    # and a fixed bbox. The orchestrator's real I/O is too heavy for unit tests.
    import _stage_sample as stage_sample  # type: ignore

    monkeypatch.setattr(
        stage_sample, "_iter_viable_tuples",
        lambda *a, **kw: iter([
            ("allen_mouse_25um", "coronal", "Hippocampal Formation"),
        ]),
    )
    monkeypatch.setattr(
        stage_sample, "_render_example_atlas",
        lambda *a, **kw: {
            "id": "bbox_000001",
            "atlas": "allen_mouse_25um",
            "atlas_version": "CCFv3",
            "orientation": "coronal",
            "region": "Hippocampal Formation",
            "source_type": "augmented_atlas",
            "source_brain": None,
            "modality": "dapi",
            "is_hemisphere": False,
            "section_image_paths": [
                str(tmp_path / f"sec_{i}.png") for i in range(4)
            ],
            "section_positions_mm": [3.0, 3.4, 3.8, 4.2],
            "bboxes": [
                {"left": [10, 20, 30, 40], "right": [60, 20, 80, 40]},
            ] * 4,
        },
    )
    monkeypatch.setattr(
        stage_sample, "_render_overlay_strip",
        lambda *a, **kw: tmp_path / "overlay.png",
    )

    args = build_bbox_data.parse_args([
        "--stage", "sample",
        "--target-total", "1",
        "--seed", "0",
        "--out-dir", str(out_dir),
        "--coverage-index", str(coverage_index_path),
    ])
    rc = stage_sample.run_stage_sample(args)
    assert rc == 0

    manifest_path = out_dir / "draft_manifest.jsonl"
    assert manifest_path.exists()
    lines = manifest_path.read_text(encoding="utf-8").strip().splitlines()
    assert len(lines) == 1
    record = json.loads(lines[0])
    assert record["region"] == "Hippocampal Formation"
    assert len(record["bboxes"]) == 4
```

- [ ] **Step 2: Run new test, verify it fails**

```bash
pytest tests/test_build_bbox_data.py::test_stage_sample_writes_draft_manifest -v
```

Expected: FAIL — `ModuleNotFoundError: No module named '_stage_sample'`.

- [ ] **Step 3: Implement stage_sample**

Create `models/langslice-gemma-4/data/_stage_sample.py`:

```python
"""--stage sample: enumerate viable tuples, render examples, write draft manifest."""
from __future__ import annotations

import argparse
import base64
import json
import logging
import sys
from pathlib import Path
from typing import Iterable

import numpy as np
from PIL import Image, ImageDraw

REPO_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO_ROOT / "src"))
sys.path.insert(0, str(REPO_ROOT / "models" / "langslice-gemma-4" / "data"))
sys.path.insert(0, str(REPO_ROOT / "_local" / "eval" / "lib"))

import build_bbox_data
import bbox_io
import region_bbox
from landmarks import LandmarkLoader

from langslice_harness.atlas.core import (
    get_position_range_mm, load_atlas,
)
from augmentation.oblique import get_oblique_slice

log = logging.getLogger("stage_sample")


def _iter_viable_tuples(
    *,
    atlases: list[str],
    orientations: list[str],
    loader: LandmarkLoader,
    atlas_objs: dict[str, object],
) -> Iterable[tuple[str, str, str]]:
    for atlas_name in atlases:
        atlas = atlas_objs[atlas_name]
        for orientation in orientations:
            for landmark in loader.landmarks_for_orientation(orientation):
                ids = loader.resolve(landmark, atlas, atlas_name)
                if not ids:
                    continue
                yield (atlas_name, orientation, landmark)


def _region_mm_extent(
    atlas, orientation: str, region_ids: set[int]
) -> tuple[float, float] | None:
    """Walk the annotation volume along the position axis; return (min, max) mm
    where any region voxel exists, else None."""
    annotation = atlas.annotation
    mask = np.isin(annotation, list(region_ids))
    axis = {"coronal": 0, "horizontal": 1, "sagittal": 2}[orientation]
    along = mask.any(axis=tuple(a for a in (0, 1, 2) if a != axis))
    indices = np.where(along)[0]
    if len(indices) == 0:
        return None
    res_um = float(atlas.resolution[axis])
    return (
        float(indices.min()) * res_um / 1000.0,
        float(indices.max()) * res_um / 1000.0,
    )


def _render_example_atlas(
    *,
    atlas_name: str,
    orientation: str,
    landmark: str,
    region_ids: set[int],
    n_sections: int,
    spacings_mm: Sequence[float],
    anchor_mm: float,
    source_type: str,  # "augmented_atlas" or "reference_atlas"
    rng: np.random.Generator,
    example_id: str,
    out_dir: Path,
) -> dict | None:
    """Render N atlas sections at the chosen positions; compute bboxes; return
    a draft-manifest record or None if any section's bbox fails."""
    from synth_dataset import sample_spec, render  # type: ignore

    atlas = load_atlas(atlas_name)

    section_paths: list[str] = []
    bboxes: list[Any] = []
    is_hemisphere = False  # whole-brain by default for atlas rendering

    positions_mm = [
        anchor_mm + sum(spacings_mm[:k])
        for k in range(n_sections)
    ]

    for k, pos_mm in enumerate(positions_mm):
        if source_type == "augmented_atlas":
            spec = sample_spec(
                rng=rng, atlases=[atlas_name],
                position_strata="uniform", oblique_prob=0.0,
            )
            object.__setattr__(spec, "plane", orientation)
            object.__setattr__(spec, "position_mm", pos_mm)
            image_f32, _ = render(spec)
            modality = spec.modality
        else:
            ref_u8, _ann = get_oblique_slice(
                atlas, base_position_mm=pos_mm, plane=orientation,
                yaw_deg=0.0, pitch_deg=0.0, roll_deg=0.0,
            )
            image_f32 = (np.repeat(ref_u8[..., None], 3, axis=2) / 255.0).astype(np.float32)
            modality = None

        # Annotation slice for bbox.
        _ref, ann_slice = get_oblique_slice(
            atlas, base_position_mm=pos_mm, plane=orientation,
            yaw_deg=0.0, pitch_deg=0.0, roll_deg=0.0,
        )
        hemi_slice = None if is_hemisphere else _slice_hemispheres(
            atlas, base_position_mm=pos_mm, plane=orientation,
        )
        bbox = region_bbox.bbox_from_atlas_slice(
            annotation_slice=ann_slice,
            hemisphere_slice=hemi_slice,
            region_ids=region_ids,
            is_hemisphere=is_hemisphere,
        )
        if bbox is None:
            return None
        bboxes.append(bbox)

        # Save the section PNG.
        section_dir = out_dir / "section_images" / example_id
        section_dir.mkdir(parents=True, exist_ok=True)
        section_path = section_dir / f"section_{k:02d}.png"
        Image.fromarray(np.clip(image_f32 * 255.0, 0, 255).astype(np.uint8)).save(section_path)
        section_paths.append(str(section_path.relative_to(REPO_ROOT)))

    atlas_version = (atlas.metadata or {}).get("version") or "unknown"
    return {
        "id": example_id,
        "atlas": atlas_name,
        "atlas_version": atlas_version,
        "orientation": orientation,
        "region": landmark,
        "source_type": source_type,
        "source_brain": None,
        "modality": modality,
        "is_hemisphere": is_hemisphere,
        "section_image_paths": section_paths,
        "section_positions_mm": positions_mm,
        "bboxes": bboxes,
    }


def _slice_hemispheres(atlas, *, base_position_mm: float, plane: str) -> np.ndarray:
    """Project atlas.hemispheres through get_oblique_slice into the section frame."""
    hemi_atlas = type("HemisphereAtlas", (), {})()
    hemi_atlas.reference = atlas.reference
    hemi_atlas.annotation = atlas.hemispheres
    hemi_atlas.resolution = atlas.resolution
    hemi_atlas.orientation = atlas.orientation
    hemi_atlas.atlas_name = getattr(atlas, "atlas_name", "hemisphere_proxy")
    _ref, hemi_slice = get_oblique_slice(
        hemi_atlas, base_position_mm=base_position_mm, plane=plane,
        yaw_deg=0.0, pitch_deg=0.0, roll_deg=0.0,
    )
    return hemi_slice


def _render_overlay_strip(record: dict, out_dir: Path) -> Path:
    """Compose N section PNGs into a horizontal strip with bbox rectangles drawn.
    Cyan = left, magenta = right (or single bbox in cyan for sagittal/hemisphere)."""
    images = [Image.open(REPO_ROOT / p) for p in record["section_image_paths"]]
    h = max(im.height for im in images)
    w = sum(im.width for im in images)
    strip = Image.new("RGB", (w, h), (24, 24, 28))

    cursor_x = 0
    for im, bbox in zip(images, record["bboxes"]):
        strip.paste(im, (cursor_x, 0))
        draw = ImageDraw.Draw(strip)
        if record["is_hemisphere"]:
            x1, y1, x2, y2 = bbox
            draw.rectangle(
                [cursor_x + x1, y1, cursor_x + x2, y2], outline="cyan", width=3
            )
        else:
            for side, color in (("left", "cyan"), ("right", "magenta")):
                if bbox.get(side) is not None:
                    x1, y1, x2, y2 = bbox[side]
                    draw.rectangle(
                        [cursor_x + x1, y1, cursor_x + x2, y2], outline=color, width=3
                    )
        cursor_x += im.width

    overlays_dir = out_dir / "draft_overlays"
    overlays_dir.mkdir(parents=True, exist_ok=True)
    out_path = overlays_dir / f"{record['id']}_strip.png"
    strip.save(out_path)
    return out_path


def run_stage_sample(args: argparse.Namespace) -> int:
    rng = np.random.default_rng(args.seed)
    source_counts: dict[tuple[str, str], int] = {}
    loader = LandmarkLoader()

    atlases = ["allen_mouse_25um", "whs_sd_rat_39um"]
    orientations = ["coronal", "sagittal", "horizontal"]
    atlas_objs = {name: load_atlas(name) for name in atlases}

    coverage_index = json.loads(args.coverage_index.read_text(encoding="utf-8")) if args.coverage_index.exists() else {}

    viable = list(_iter_viable_tuples(
        atlases=atlases, orientations=orientations,
        loader=loader, atlas_objs=atlas_objs,
    ))
    log.info("Viable (atlas, orientation, region) tuples: %d", len(viable))
    if not viable:
        log.error("No viable tuples — check landmark_atlas_map.json")
        return 1

    target_per_tuple = max(1, args.target_total // len(viable))

    records: list[dict] = []
    next_id = 0
    for atlas_name, orientation, landmark in viable:
        atlas = atlas_objs[atlas_name]
        region_ids = loader.resolve(landmark, atlas, atlas_name)
        extent = _region_mm_extent(atlas, orientation, region_ids)
        if extent is None:
            continue
        rmin, rmax = extent
        produced = 0
        attempts = 0
        while produced < target_per_tuple and attempts < target_per_tuple * 5:
            attempts += 1
            decision = build_bbox_data.pick_source(
                atlas=atlas_name, orientation=orientation, landmark=landmark,
                coverage_index=coverage_index, rng=rng,
                source_counts=source_counts,
            )
            if decision["source_type"] == "real_histology":
                # Real-histology rendering is implemented separately; for the
                # first iteration, fall back to atlas if real path is not yet
                # ready. The orchestrator's hook is `_render_example_real`.
                log.debug("real-histology path not yet implemented for %s/%s/%s",
                          atlas_name, orientation, landmark)
                continue

            n = build_bbox_data.sample_section_count(rng)
            spacings = build_bbox_data.sample_spacings_mm(rng, n_gaps=n - 1)
            anchor = build_bbox_data.sample_anchor_mm(
                rng=rng, region_mm_min=rmin, region_mm_max=rmax,
                spacings_mm=spacings,
            )
            if anchor is None:
                continue

            example_id = f"bbox_{next_id:06d}"
            next_id += 1
            rec = _render_example_atlas(
                atlas_name=atlas_name, orientation=orientation,
                landmark=landmark, region_ids=region_ids,
                n_sections=n, spacings_mm=spacings, anchor_mm=anchor,
                source_type=decision["source_type"], rng=rng,
                example_id=example_id, out_dir=args.out_dir,
            )
            if rec is None:
                continue
            _render_overlay_strip(rec, args.out_dir)
            records.append(rec)
            produced += 1

    args.out_dir.mkdir(parents=True, exist_ok=True)
    bbox_io.write_jsonl(args.out_dir / "draft_manifest.jsonl", records)
    log.info("Wrote %d records to draft_manifest.jsonl", len(records))
    return 0
```

- [ ] **Step 4: Run end-to-end test, verify pass**

```bash
pytest tests/test_build_bbox_data.py::test_stage_sample_writes_draft_manifest -v
```

Expected: 1 passed.

- [ ] **Step 5: Smoke-run `--stage sample` (atlas-only path)**

```bash
python models/langslice-gemma-4/data/build_bbox_data.py --stage sample --target-total 5 --seed 0
```

Expected: prints "Viable (atlas, orientation, region) tuples: N" and "Wrote M records to draft_manifest.jsonl" with M ≤ 5. Inspect `_local/bbox_data/draft_manifest.jsonl` and `_local/bbox_data/draft_overlays/`.

- [ ] **Step 6: Commit**

```bash
git add models/langslice-gemma-4/data/_stage_sample.py tests/test_build_bbox_data.py
git commit -m "feat(gemma-4): --stage sample end-to-end (atlas sources)"
```

---

## Task 10: build_bbox_data — real-histology source path

**Files:**
- Modify: `models/langslice-gemma-4/data/_stage_sample.py`
- Modify: `tests/test_build_bbox_data.py`

**Spec sections covered:** §4.1 (Tier A real), §4.4 (priority logic), §6.2 (real-section bbox), §3.2 (snap to actual section spacing).

- [ ] **Step 1: Add a real-histology integration test (stubbed)**

Append to `tests/test_build_bbox_data.py`:

```python
def test_render_example_real_returns_record(tmp_path: Path, monkeypatch):
    import _stage_sample as stage_sample  # type: ignore

    monkeypatch.setattr(build_bbox_data, "sample_section_count", lambda rng: 4)
    monkeypatch.setattr(build_bbox_data, "sample_spacings_mm", lambda rng, n_gaps: [0.2, 0.3, 0.4][:n_gaps])

    # Stub the registration helper + atlas annotation so we don't need a real
    # brain on disk.
    fake_voxel = lambda i, j: np.array([i, 0.0, j], dtype=np.float64)
    fake_records = []
    for idx, pos in enumerate([3.0, 3.2, 3.5, 3.9]):
        fake_records.append({
            "section_id": f"FAKE:s{idx:03d}",
            "image_path": tmp_path / "fake_real_section.png",
            "image_shape": (96, 96),
            "pixel_to_voxel": fake_voxel,
            "midline_x_at_row": (lambda _i: 48.0),
            "is_hemisphere": False,
            "position_mm": pos,
        })
    by_id = {r["section_id"]: r for r in fake_records}
    monkeypatch.setattr(stage_sample, "_load_real_section_record", lambda **kw: by_id[kw["section_id"]])
    # Ensure the test image exists.
    Image.new("RGB", (96, 96), (50, 50, 50)).save(tmp_path / "fake_real_section.png")

    fake_annotation = np.zeros((1, 96, 96), dtype=np.int32)
    fake_annotation[0, 20:50, 10:40] = 1   # left
    fake_annotation[0, 20:50, 60:90] = 1   # right
    fake_atlas = type("FA", (), {"annotation": fake_annotation, "resolution": (25.0, 25.0, 25.0), "metadata": {"version": "CCFv3"}})()
    monkeypatch.setattr(stage_sample, "_load_atlas_cached", lambda name: fake_atlas)

    rec = stage_sample._render_example_real(
        atlas_name="allen_mouse_25um", orientation="coronal",
        landmark="Hippocampal Formation", region_ids={1},
        brain_id="FAKE", section_ids=list(by_id),
        out_dir=tmp_path, example_id="bbox_test",
        rng=np.random.default_rng(0),
    )
    assert rec is not None
    assert rec["source_type"] == "real_histology"
    assert rec["source_brain"] == "FAKE"
    assert len(rec["bboxes"]) == 4
    for bbox in rec["bboxes"]:
        assert "left" in bbox and "right" in bbox


def test_render_example_real_scales_bbox_after_thumbnail(tmp_path: Path, monkeypatch):
    import _stage_sample as stage_sample  # type: ignore

    monkeypatch.setattr(build_bbox_data, "sample_section_count", lambda rng: 4)
    monkeypatch.setattr(build_bbox_data, "sample_spacings_mm", lambda rng, n_gaps: [0.2, 0.3, 0.4][:n_gaps])

    Image.new("RGB", (1600, 800), (50, 50, 50)).save(tmp_path / "wide_section.png")
    fake_records = []
    for idx, pos in enumerate([0.0, 0.2, 0.5, 0.9]):
        fake_records.append({
            "section_id": f"FAKE:s{idx:03d}",
            "image_path": tmp_path / "wide_section.png",
            "image_shape": (800, 1600),
            "pixel_to_voxel": lambda i, j: np.array([i / 10.0, 0.0, j / 10.0]),
            "midline_x_at_row": (lambda _i: 800.0),
            "is_hemisphere": True,
            "position_mm": pos,
        })
    by_id = {r["section_id"]: r for r in fake_records}
    monkeypatch.setattr(stage_sample, "_load_real_section_record", lambda **kw: by_id[kw["section_id"]])
    fake_annotation = np.zeros((1, 100, 200), dtype=np.int32)
    fake_annotation[0, 20:60, 20:80] = 1
    monkeypatch.setattr(stage_sample, "_load_atlas_cached", lambda name: type(
        "FA", (), {"annotation": fake_annotation, "metadata": {"version": "CCFv3"}}
    )())

    rec = stage_sample._render_example_real(
        atlas_name="allen_mouse_25um", orientation="coronal",
        landmark="Hippocampal Formation", region_ids={1},
        brain_id="FAKE", section_ids=list(by_id),
        out_dir=tmp_path, example_id="bbox_scale",
        rng=np.random.default_rng(0),
    )
    assert rec is not None
    # 1600x800 thumbnails to 768x384; bbox coordinates must be in that frame.
    assert all(0 <= coord <= 768 for bbox in rec["bboxes"] for coord in bbox)


def test_choose_real_section_subset_samples_n_and_per_gap_spacings(monkeypatch):
    import _stage_sample as stage_sample  # type: ignore

    monkeypatch.setattr(build_bbox_data, "sample_section_count", lambda rng: 4)
    monkeypatch.setattr(build_bbox_data, "sample_spacings_mm", lambda rng, n_gaps: [0.2, 0.4, 0.6][:n_gaps])
    monkeypatch.setattr(build_bbox_data, "sample_anchor_mm", lambda **kw: 0.0)

    records = [
        {"section_id": f"s{i}", "position_mm": i * 0.2}
        for i in range(20)
    ]
    chosen = stage_sample._choose_real_section_subset(records, rng=np.random.default_rng(4))
    assert chosen is not None
    assert 4 <= len(chosen) <= 8
    gaps = [
        b["position_mm"] - a["position_mm"]
        for a, b in zip(chosen, chosen[1:])
    ]
    assert all(0.2 <= gap <= 0.8 for gap in gaps)
    assert len({round(gap, 6) for gap in gaps}) > 1
```

- [ ] **Step 2: Run test, verify it fails**

```bash
pytest tests/test_build_bbox_data.py::test_render_example_real_returns_record -v
```

Expected: FAIL — `_render_example_real` not defined.

- [ ] **Step 3: Add the real-section helpers and renderer**

Append to `models/langslice-gemma-4/data/_stage_sample.py`:

```python
def _load_atlas_cached(name: str) -> object:
    """Cache atlas loads; called from _render_example_real."""
    if not hasattr(_load_atlas_cached, "_cache"):
        _load_atlas_cached._cache = {}
    cache = _load_atlas_cached._cache
    if name not in cache:
        cache[name] = load_atlas(name)
    return cache[name]


def _load_real_section_record(*, section_id: str, manifest_path: Path) -> dict:
    """Load a section's image path, anchoring/markers, and registration helper.

    Reads the canonical manifest, picks the section's record, calls
    `section_to_atlas_voxel` to build the pixel→voxel function, and computes
    the projected midline-line function.
    """
    from registration import section_to_atlas_voxel  # type: ignore

    with manifest_path.open("r", encoding="utf-8") as f:
        for line in f:
            rec = json.loads(line)
            if rec.get("section_id") == section_id:
                break
        else:
            raise KeyError(f"section {section_id!r} not in manifest {manifest_path}")

    atlas = _load_atlas_cached(rec["atlas"])
    slice_record = rec["slice_record"]
    pixel_to_voxel = section_to_atlas_voxel(slice_record, atlas)

    H = int(slice_record["height"])
    W = int(slice_record["width"])

    # Project the atlas ML-midline plane to a line in section pixel space.
    # BrainGlobe annotation order is (AP, DV, ML), while pixel_to_voxel returns
    # QuickNII order (x_ml, y_ap, z_dv). Midline is therefore voxel[0].
    midline_voxel_x = float(atlas.annotation.shape[2]) / 2.0

    def midline_x_at_row(i: float) -> float:
        # Bisection along the row to find the (i, j) where x_ml crosses midline.
        lo, hi = 0.0, float(W - 1)
        for _ in range(20):
            mid = (lo + hi) / 2.0
            voxel = pixel_to_voxel(mid, i)
            if voxel[0] < midline_voxel_x:
                lo = mid
            else:
                hi = mid
        return (lo + hi) / 2.0

    return {
        "section_id": section_id,
        "image_path": REPO_ROOT / rec["image_path"],
        "image_shape": (H, W),
        "pixel_to_voxel": pixel_to_voxel,
        "midline_x_at_row": midline_x_at_row,
        "is_hemisphere": bool(rec.get("is_hemisphere", False)),
        "position_mm": float(rec["position_mm"]),
        "atlas": rec["atlas"],
    }


def _render_example_real(
    *,
    atlas_name: str,
    orientation: str,
    landmark: str,
    region_ids: set[int],
    brain_id: str,
    section_ids: list[str],
    out_dir: Path,
    example_id: str,
    manifest_path: Path | None = None,
    rng: np.random.Generator | None = None,
) -> dict | None:
    """Render a real-histology example: sample N in {4..8}, choose spacings and
    anchor to fit available sections, compute bboxes."""
    if manifest_path is None:
        manifest_path = REPO_ROOT / "_local" / "eval" / "data" / "manifest.jsonl"
    if rng is None:
        rng = np.random.default_rng()

    section_records = [
        _load_real_section_record(section_id=sid, manifest_path=manifest_path)
        for sid in section_ids
    ]
    section_records.sort(key=lambda r: r["position_mm"])

    chosen = _choose_real_section_subset(section_records, rng=rng)
    if chosen is None:
        return None

    atlas = _load_atlas_cached(atlas_name)
    is_hemisphere = chosen[0]["is_hemisphere"]
    section_paths: list[str] = []
    bboxes: list[Any] = []
    positions_mm: list[float] = []

    for k, rec in enumerate(chosen):
        bbox = region_bbox.bbox_from_real_section(
            section_image_shape=rec["image_shape"],
            pixel_to_voxel=rec["pixel_to_voxel"],
            annotation_volume=atlas.annotation,
            region_ids=region_ids,
            midline_x_at_row=rec["midline_x_at_row"],
            is_hemisphere=is_hemisphere,
            grid_step=8,
        )
        if bbox is None:
            return None

        # Resize and save into the example's output dir for QC + submission.
        from PIL import Image as _Image
        section_dir = out_dir / "section_images" / example_id
        section_dir.mkdir(parents=True, exist_ok=True)
        section_path = section_dir / f"section_{k:02d}.png"
        im = _Image.open(rec["image_path"])
        original_w, original_h = im.size
        im.thumbnail((768, 768))
        scale_x = im.width / float(original_w)
        scale_y = im.height / float(original_h)
        bboxes.append(region_bbox.scale_bbox(bbox, scale_x=scale_x, scale_y=scale_y))
        im.save(section_path)
        section_paths.append(str(section_path.relative_to(REPO_ROOT)))
        positions_mm.append(rec["position_mm"])

    atlas_version = (atlas.metadata or {}).get("version") or "unknown"
    return {
        "id": example_id,
        "atlas": atlas_name,
        "atlas_version": atlas_version,
        "orientation": orientation,
        "region": landmark,
        "source_type": "real_histology",
        "source_brain": brain_id,
        "modality": None,
        "is_hemisphere": is_hemisphere,
        "section_image_paths": section_paths,
        "section_positions_mm": positions_mm,
        "bboxes": bboxes,
    }


def _choose_real_section_subset(
    section_records: list[dict],
    rng: np.random.Generator,
) -> list[dict] | None:
    """Sample N uniformly from {4..8}, then choose per-gap spacings/anchor to fit records."""
    if len(section_records) < 4:
        return None
    by_pos = sorted(section_records, key=lambda r: r["position_mm"])
    for _attempt in range(100):
        n = build_bbox_data.sample_section_count(rng)
        if len(by_pos) < n:
            continue
        spacings = build_bbox_data.sample_spacings_mm(rng, n_gaps=n - 1)
        positions = np.array([float(r["position_mm"]) for r in by_pos])
        min_pos = float(positions.min())
        max_pos = float(positions.max())
        anchor = build_bbox_data.sample_anchor_mm(
            rng=rng, region_mm_min=min_pos, region_mm_max=max_pos,
            spacings_mm=spacings,
        )
        if anchor is None:
            continue
        targets = np.array([anchor + sum(spacings[:k]) for k in range(n)])
        chosen: list[dict] = []
        used: set[int] = set()
        for target in targets:
            order = np.argsort(np.abs(positions - target))
            idx = next((int(i) for i in order if int(i) not in used), None)
            if idx is None:
                break
            used.add(idx)
            chosen.append(by_pos[idx])
        chosen.sort(key=lambda r: r["position_mm"])
        if len(chosen) != n:
            continue
        gaps = [b["position_mm"] - a["position_mm"] for a, b in zip(chosen, chosen[1:])]
        if all(0.2 <= gap <= 0.8 for gap in gaps):
            return chosen
    return None
```

Then update the `run_stage_sample` loop in the same file: replace the `if decision["source_type"] == "real_histology":` block with the real call:

```python
            if decision["source_type"] == "real_histology":
                example_id = f"bbox_{next_id:06d}"
                next_id += 1
                rec = _render_example_real(
                    atlas_name=atlas_name, orientation=orientation,
                    landmark=landmark, region_ids=region_ids,
                    brain_id=decision["source_brain"],
                    section_ids=decision["available_section_ids"],
                    out_dir=args.out_dir, example_id=example_id,
                    rng=rng,
                )
                if rec is None:
                    continue
                _render_overlay_strip(rec, args.out_dir)
                records.append(rec)
                source_counts[(decision["source_brain"], landmark)] = (
                    source_counts.get((decision["source_brain"], landmark), 0) + 1
                )
                produced += 1
                continue
```

- [ ] **Step 4: Run all build_bbox_data tests, verify pass**

```bash
pytest tests/test_build_bbox_data.py -v
```

Expected: 11 passed.

- [ ] **Step 5: Smoke-run with real coverage**

```bash
# Bake the coverage index first (Task 5 already done).
python models/langslice-gemma-4/data/build_bbox_data.py --stage sample --target-total 30 --seed 0
```

Expected: a mix of real_histology / augmented_atlas / reference_atlas records in the draft manifest. Spot-check overlays for a real-histology example: bboxes should land on the named region.

- [ ] **Step 6: Commit**

```bash
git add models/langslice-gemma-4/data/_stage_sample.py tests/test_build_bbox_data.py
git commit -m "feat(gemma-4): real-histology source path with marker-warp bbox"
```

---

## Task 11: QC app `--mode bbox` extension

**Files:**
- Modify: `_local/qc_app/app.py`
- Create: `_local/qc_app/static/bbox.html`

**Spec sections covered:** §8.2 (QC app extension).

This module is `_local/`, gitignored - use a local route-level pytest before implementation, then manually verify by loading the draft manifest and clicking through.

- [ ] **Step 1: Add a route-level failing test**

Create `_local/qc_app/test_bbox_routes.py`:

```python
"""Local route tests for bbox QC mode; gitignored with _local/."""
from __future__ import annotations

import json
from pathlib import Path

import pytest

import app


def test_bbox_manifest_loader_reads_jsonl(tmp_path: Path):
    path = tmp_path / "draft_manifest.jsonl"
    path.write_text(json.dumps({"id": "bbox_000001"}) + "\n", encoding="utf-8")
    assert app.load_bbox_manifest(path) == [{"id": "bbox_000001"}]


def test_bbox_overlay_route_serves_png(tmp_path: Path):
    handler = app.make_test_handler(
        mode="bbox",
        bbox_data_dir=tmp_path,
        bbox_records=[{"id": "bbox_000001"}],
    )
    overlay_dir = tmp_path / "draft_overlays"
    overlay_dir.mkdir(parents=True)
    (overlay_dir / "bbox_000001_strip.png").write_bytes(b"\x89PNG\r\n\x1a\n")

    response = handler.get("/bbox/overlay/bbox_000001")
    assert response.status_code == 200
    assert response.headers["Content-Type"] == "image/png"
```

Run it and verify it fails because the bbox loader/routes are not wired yet:

```bash
pytest _local/qc_app/test_bbox_routes.py -v
```

- [ ] **Step 2: Add the `--mode` argparse switch**

Edit `_local/qc_app/app.py`. Find the existing argparse setup (search for `argparse.ArgumentParser`). Add:

```python
parser.add_argument(
    "--mode",
    choices=("trace", "bbox"),
    default="trace",
    help="Manifest type to QC. 'trace' is the default trace-collection format; "
         "'bbox' loads a BBox-data draft manifest and renders bbox overlays.",
)
parser.add_argument(
    "--bbox-manifest",
    type=Path,
    default=REPO_ROOT / "_local" / "bbox_data" / "draft_manifest.jsonl",
    help="Draft manifest path for --mode bbox.",
)
parser.add_argument(
    "--bbox-verdicts",
    type=Path,
    default=REPO_ROOT / "_local" / "bbox_data" / "qc_verdicts.jsonl",
    help="Verdict output path for --mode bbox.",
)
```

- [ ] **Step 3: Add the bbox manifest loader**

Add near the existing trace-manifest loading code:

```python
def load_bbox_manifest(path: Path) -> list[dict]:
    if not path.exists():
        return []
    records = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                records.append(json.loads(line))
    return records
```

- [ ] **Step 4: Add the overlay endpoint**

In the HTTP handler, branch on mode. Add a new endpoint `/bbox/overlay/<id>` that streams the pre-rendered overlay PNG (already produced by `--stage sample`):

```python
def _handle_bbox_overlay(self, parsed):
    parts = parsed.path.split("/")
    if len(parts) < 4:
        self.send_response(HTTPStatus.NOT_FOUND); self.end_headers(); return
    example_id = parts[3]
    overlay_path = (
        REPO_ROOT / "_local" / "bbox_data" / "draft_overlays"
        / f"{example_id}_strip.png"
    )
    if not overlay_path.exists():
        self.send_response(HTTPStatus.NOT_FOUND); self.end_headers(); return
    data = overlay_path.read_bytes()
    self.send_response(HTTPStatus.OK)
    self.send_header("Content-Type", "image/png")
    self.send_header("Content-Length", str(len(data)))
    self.end_headers()
    self.wfile.write(data)
```

- [ ] **Step 5: Add the bbox HTML template**

Create `_local/qc_app/static/bbox.html`:

```html
<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<title>BBox QC</title>
<style>
  body { background: #1a1a1f; color: #eee; font: 14px/1.4 system-ui, sans-serif; margin: 0; padding: 16px; }
  .meta { display: grid; grid-template-columns: max-content 1fr; gap: 4px 16px; margin-bottom: 12px; }
  .meta b { color: #aaa; font-weight: 600; }
  .strip { background: #000; border: 1px solid #333; max-width: 100%; }
  .footer { margin-top: 12px; color: #888; }
  kbd { background: #333; padding: 2px 6px; border-radius: 3px; }
  .verdict { margin-top: 16px; }
  .verdict button { padding: 8px 16px; margin-right: 8px; cursor: pointer; }
  #note { width: 100%; padding: 6px; box-sizing: border-box; background: #222; color: #eee; border: 1px solid #444; }
</style>
</head>
<body>
<div id="meta" class="meta"></div>
<img id="strip" class="strip" alt="bbox overlay strip">
<div class="verdict">
  <button onclick="verdict('verify')">Verify (y)</button>
  <button onclick="verdict('reject')">Reject (n)</button>
  <button onclick="verdict('skip')">Skip (s)</button>
  <input id="note" placeholder="optional note">
</div>
<div class="footer">
  <kbd>j</kbd>/<kbd>k</kbd> next/prev · <kbd>y</kbd> verify · <kbd>n</kbd> reject · <kbd>s</kbd> skip · <kbd>g</kbd> next un-verdicted
</div>
<script>
let records = [];
let idx = 0;
let verdicts = {};

async function load() {
  records = await (await fetch("/bbox/records")).json();
  verdicts = await (await fetch("/bbox/verdicts")).json();
  render();
}
function render() {
  const r = records[idx];
  if (!r) return;
  const meta = document.getElementById("meta");
  meta.innerHTML = `
    <b>ID</b><span>${r.id} (${idx + 1}/${records.length})</span>
    <b>Atlas</b><span>${r.atlas}</span>
    <b>Plane</b><span>${r.orientation}</span>
    <b>Region</b><span>${r.region}</span>
    <b>Source</b><span>${r.source_type}${r.source_brain ? ' / ' + r.source_brain : ''}${r.modality ? ' / ' + r.modality : ''}</span>
    <b>Sections</b><span>N=${r.bboxes.length}, positions ${r.section_positions_mm.map(p => p.toFixed(2)).join(', ')} mm</span>
    <b>Verdict</b><span>${verdicts[r.id] || '—'}</span>
  `;
  document.getElementById("strip").src = `/bbox/overlay/${r.id}?t=${Date.now()}`;
}
async function verdict(v) {
  const r = records[idx];
  const note = document.getElementById("note").value;
  await fetch("/bbox/verdict", {
    method: "POST", headers: {"Content-Type": "application/json"},
    body: JSON.stringify({id: r.id, verdict: v, note}),
  });
  verdicts[r.id] = v;
  document.getElementById("note").value = "";
  next();
}
function next() { idx = Math.min(idx + 1, records.length - 1); render(); }
function prev() { idx = Math.max(idx - 1, 0); render(); }
function nextUnverdicted() {
  for (let i = idx + 1; i < records.length; i++) {
    if (!verdicts[records[i].id]) { idx = i; render(); return; }
  }
}
window.addEventListener("keydown", (ev) => {
  if (ev.target.tagName === "INPUT") return;
  if (ev.key === "y" || ev.key === " ") verdict("verify");
  else if (ev.key === "n" || ev.key === "x") verdict("reject");
  else if (ev.key === "s") verdict("skip");
  else if (ev.key === "j" || ev.key === "ArrowRight") next();
  else if (ev.key === "k" || ev.key === "ArrowLeft") prev();
  else if (ev.key === "g") nextUnverdicted();
});
load();
</script>
</body>
</html>
```

- [ ] **Step 6: Wire the bbox endpoints in the existing handler**

In `_local/qc_app/app.py`, find the existing `do_GET` / `do_POST` dispatch. Add cases for the bbox routes:

```python
def do_GET(self):
    parsed = urlparse(self.path)
    if parsed.path == "/" and getattr(self.server, "mode", "trace") == "bbox":
        self._serve_static("bbox.html"); return
    if parsed.path == "/bbox/records":
        self._serve_json(self.server.bbox_records); return
    if parsed.path == "/bbox/verdicts":
        verdicts = self._read_bbox_verdicts()
        self._serve_json(verdicts); return
    if parsed.path.startswith("/bbox/overlay/"):
        self._handle_bbox_overlay(parsed); return
    # fall through to existing trace-mode handlers...

def do_POST(self):
    parsed = urlparse(self.path)
    if parsed.path == "/bbox/verdict":
        self._handle_bbox_verdict(); return
    # fall through...

def _handle_bbox_verdict(self):
    length = int(self.headers.get("Content-Length", "0"))
    body = json.loads(self.rfile.read(length))
    line = json.dumps({
        "id": body["id"], "verdict": body["verdict"],
        "ts": time.time(), "note": body.get("note", ""),
    })
    self.server.bbox_verdicts_path.parent.mkdir(parents=True, exist_ok=True)
    with self.server.bbox_verdicts_path.open("a", encoding="utf-8") as f:
        f.write(line + "\n")
    self.send_response(HTTPStatus.NO_CONTENT); self.end_headers()

def _read_bbox_verdicts(self) -> dict:
    out = {}
    if self.server.bbox_verdicts_path.exists():
        with self.server.bbox_verdicts_path.open("r", encoding="utf-8") as f:
            for line in f:
                rec = json.loads(line)
                out[rec["id"]] = rec["verdict"]
    return out
```

Add a tiny `make_test_handler(...)` helper only for the local route test if the app does not already expose a testable handler factory. In server setup: when `args.mode == "bbox"`, attach `server.mode = "bbox"`, `server.bbox_records = load_bbox_manifest(args.bbox_manifest)`, `server.bbox_verdicts_path = args.bbox_verdicts`, and `server.bbox_data_dir = args.bbox_manifest.parent`.

- [ ] **Step 7: Run the local route test, verify pass**

```bash
pytest _local/qc_app/test_bbox_routes.py -v
```

Expected: 2 passed.

- [ ] **Step 8: Smoke-run the QC app in bbox mode**

```bash
python _local/qc_app/app.py --mode bbox
```

Open `http://localhost:8765`. Expected: the first draft-manifest record renders with its bbox-overlay strip, hotkeys work, verdicts append to `_local/bbox_data/qc_verdicts.jsonl`.

- [ ] **Step 9: No commit**

`_local/` is gitignored.

---

## Task 12: build_bbox_data — `--stage submit`

**Files:**
- Create: `models/langslice-gemma-4/data/_stage_submit.py`
- Modify: `tests/test_build_bbox_data.py`

**Spec sections covered:** §8.1 stage 2, §9 (Batch API), §10 (validation gates), §11 (final SFT shape).

- [ ] **Step 1: Write the submit-stage test (with stubbed batch client)**

Append to `tests/test_build_bbox_data.py`:

```python
def test_stage_submit_writes_sft_jsonl(tmp_path: Path, monkeypatch):
    import _stage_submit as stage_submit  # type: ignore

    # Build a tiny draft manifest + a verdicts file.
    draft = [
        {
            "id": "bbox_000001",
            "atlas": "allen_mouse_25um",
            "atlas_version": "CCFv3",
            "orientation": "coronal",
            "region": "Hippocampal Formation",
            "source_type": "augmented_atlas",
            "source_brain": None,
            "is_hemisphere": False,
            "modality": "dapi",
            "section_image_paths": [str(tmp_path / f"sec_{i}.png") for i in range(4)],
            "section_positions_mm": [3.0, 3.4, 3.8, 4.2],
            "bboxes": [{"left": [10, 20, 30, 40], "right": [60, 20, 80, 40]}] * 4,
        },
    ]
    for p in draft[0]["section_image_paths"]:
        Image.new("RGB", (96, 96), (50, 50, 50)).save(p)

    out_dir = tmp_path / "bbox_data"
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "draft_manifest.jsonl").write_text(
        "\n".join(json.dumps(r) for r in draft) + "\n", encoding="utf-8"
    )
    verdicts_path = tmp_path / "verdicts.jsonl"
    verdicts_path.write_text(
        json.dumps({"id": "bbox_000001", "verdict": "verify",
                    "ts": 0.0, "note": ""}) + "\n", encoding="utf-8"
    )

    # Stub the current Batch API shape: upload JSONL, create batch with the
    # uploaded file resource name, then download batch_job.dest.file_name.
    class FakeFileRef:
        name = "files/request-jsonl"

    class FakeFiles:
        def __init__(self):
            self.uploaded_file = None
            self.downloaded_file = None
        def upload(self, **kw):
            self.uploaded_file = kw["file"]
            return FakeFileRef()
        def download(self, *, file):
            self.downloaded_file = file
            rows = [
                {
                    "key": "bbox_000001",
                    "response": {
                        "candidates": [{
                            "content": {"parts": [
                                {"thought": True, "text": "[thinking]"},
                                {"text": "The hippocampus appears as a thin crescent and broadens caudally. Dentate gyrus folding becomes more distinct across the strip."},
                            ]}
                        }]
                    }
                }
            ]
            return ("\n".join(json.dumps(r) for r in rows) + "\n").encode("utf-8")

    class FakeBatches:
        def __init__(self):
            self.created_src = None
        def create(self, **kw):
            self.created_src = kw["src"]
            return {"name": "fake-batch-id", "dest": {"file_name": "files/results-jsonl"}}
        def get(self, **kw):
            return {"state": "JOB_STATE_SUCCEEDED", "dest": {"file_name": "files/results-jsonl"}}

    class FakeClient:
        def __init__(self):
            self.files = FakeFiles()
            self.batches = FakeBatches()

    fake = FakeClient()
    monkeypatch.setattr(stage_submit, "_get_genai_client", lambda: fake)

    args = build_bbox_data.parse_args([
        "--stage", "submit",
        "--out-dir", str(out_dir),
        "--verdicts", str(verdicts_path),
        "--coverage-index", str(tmp_path / "coverage.json"),
    ])
    rc = stage_submit.run_stage_submit(args)
    assert rc == 0
    assert fake.batches.created_src == "files/request-jsonl"
    assert fake.files.downloaded_file == "files/results-jsonl"

    sft_path = out_dir / "sft.jsonl"
    assert sft_path.exists()
    sft = [json.loads(l) for l in sft_path.read_text(encoding="utf-8").splitlines() if l.strip()]
    assert len(sft) == 1
    rec = sft[0]
    assert rec["id"] == "bbox_000001"
    # Caption should be present and not contain the dropped thinking text.
    assistant_msg = [m for m in rec["messages"] if m["role"] == "assistant"][0]
    assert "thin crescent" in assistant_msg["content"]
    assert "[thinking]" not in assistant_msg["content"]
```

- [ ] **Step 2: Run test, verify it fails**

```bash
pytest tests/test_build_bbox_data.py::test_stage_submit_writes_sft_jsonl -v
```

Expected: FAIL — `_stage_submit` not found.

- [ ] **Step 3: Implement stage_submit**

Create `models/langslice-gemma-4/data/_stage_submit.py`:

```python
"""--stage submit: pack verified examples, submit to AI Studio Batch API,
poll, retrieve, mm-strip, write final SFT."""
from __future__ import annotations

import argparse
import base64
import json
import logging
import sys
import time
from pathlib import Path
from typing import Iterable

REPO_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO_ROOT / "src"))
sys.path.insert(0, str(REPO_ROOT / "models" / "langslice-gemma-4" / "data"))

import bbox_io

log = logging.getLogger("stage_submit")

_POLL_INTERVAL_S = 30.0


def _get_genai_client():
    """Return the configured google-genai client for Batch API calls.

    Test stubs expose `.files.upload`, `.batches.create`, `.batches.get`, and
    `.files.download`, matching the current Gemini Batch API flow.
    """
    from langslice_harness import vlm_config
    if not vlm_config.supports_batch_api():
        raise RuntimeError(
            f"Backend {vlm_config.get_backend()!r} does not support Batch API"
        )
    return vlm_config.get_client()


def _encode_b64(path: Path) -> str:
    return base64.b64encode(path.read_bytes()).decode("ascii")


def _build_request_jsonl(
    draft: list[dict], verdicts: dict[str, str], out_path: Path
) -> int:
    n = 0
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", encoding="utf-8") as f:
        for record in draft:
            if verdicts.get(record["id"]) != "verify":
                continue
            images_b64 = [
                _encode_b64(REPO_ROOT / p) for p in record["section_image_paths"]
            ]
            line = bbox_io.pack_batch_request(record, images_b64=images_b64)
            f.write(json.dumps(line) + "\n")
            n += 1
    return n


def _read_verdicts(path: Path) -> dict[str, str]:
    out = {}
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rec = json.loads(line)
                out[rec["id"]] = rec["verdict"]
    return out


def _poll_until_done(client, batch_name: str) -> object:
    while True:
        info = client.batches.get(name=batch_name)
        state = info.get("state") if isinstance(info, dict) else getattr(info, "state", None)
        if state in ("JOB_STATE_SUCCEEDED", "JOB_STATE_FAILED", "JOB_STATE_CANCELLED"):
            log.info("Batch %s reached state %s", batch_name, state)
            if state != "JOB_STATE_SUCCEEDED":
                raise RuntimeError(f"Batch {batch_name} ended in state {state}")
            return info
        log.info("Batch %s state=%s; sleeping %.0fs", batch_name, state, _POLL_INTERVAL_S)
        time.sleep(_POLL_INTERVAL_S)


def _batch_dest_file_name(batch_job: object) -> str:
    if isinstance(batch_job, dict):
        return str(batch_job["dest"]["file_name"])
    dest = getattr(batch_job, "dest")
    return str(getattr(dest, "file_name"))


def _build_sft_record(draft_record: dict, caption: str) -> dict:
    user_text = bbox_io.build_user_text(draft_record)
    image_parts = [
        {"type": "image", "path": p} for p in draft_record["section_image_paths"]
    ]
    return {
        "id": draft_record["id"],
        "atlas": draft_record["atlas"],
        "atlas_version": draft_record["atlas_version"],
        "orientation": draft_record["orientation"],
        "region": draft_record["region"],
        "source_type": draft_record["source_type"],
        "source_brain": draft_record["source_brain"],
        "is_hemisphere": draft_record["is_hemisphere"],
        "messages": [
            {"role": "system", "content": bbox_io._SYSTEM_PROMPT},
            {"role": "user", "content": image_parts + [{"type": "text", "text": user_text}]},
            {"role": "assistant", "content": caption},
        ],
    }


def run_stage_submit(args: argparse.Namespace) -> int:
    draft_path = args.out_dir / "draft_manifest.jsonl"
    if not draft_path.exists():
        log.error("Draft manifest missing: %s", draft_path)
        return 1
    draft = list(bbox_io.read_jsonl(draft_path))

    verdicts = _read_verdicts(args.verdicts)
    log.info("Read %d verdicts; %d verified", len(verdicts),
             sum(1 for v in verdicts.values() if v == "verify"))

    request_path = args.out_dir / "request.jsonl"
    n = _build_request_jsonl(draft, verdicts, request_path)
    log.info("Built %d batch requests at %s", n, request_path)
    if n == 0:
        log.error("No verified examples — nothing to submit")
        return 1

    if args.dry_run:
        log.info("--dry-run: skipping API submission")
        return 0

    client = _get_genai_client()

    # Upload the JSONL file.
    uploaded_file = client.files.upload(
        file=str(request_path),
        config={"display_name": f"bbox-{int(time.time())}"},
    )
    uploaded_name = uploaded_file.name
    log.info("Uploaded file %s", uploaded_name)

    batch = client.batches.create(model=args.model, src=uploaded_file.name)
    batch_name = batch.get("name") if isinstance(batch, dict) else getattr(batch, "name", None)
    log.info("Created batch %s", batch_name)

    metadata_path = args.out_dir / "batch_metadata.json"
    metadata_path.write_text(
        json.dumps({"batch_name": batch_name, "model": args.model,
                    "submitted_at": time.time(), "n_requests": n}, indent=2),
        encoding="utf-8",
    )

    completed_batch = _poll_until_done(client, batch_name)

    # Pull responses from the result file named by batch_job.dest.file_name.
    result_file_name = _batch_dest_file_name(completed_batch)
    downloaded = client.files.download(file=result_file_name)
    response_text = downloaded.decode("utf-8") if isinstance(downloaded, bytes) else str(downloaded)
    responses_path = args.out_dir / "responses.jsonl"
    by_id: dict[str, dict] = {}
    with responses_path.open("w", encoding="utf-8") as f:
        for line in response_text.splitlines():
            if not line.strip():
                continue
            resp = json.loads(line)
            f.write(json.dumps(resp) + "\n")
            by_id[resp["key"]] = resp

    # Build SFT records (mm-strip + length gate).
    sft_records: list[dict] = []
    n_dropped = 0
    for record in draft:
        if verdicts.get(record["id"]) != "verify":
            continue
        resp = by_id.get(record["id"])
        if resp is None:
            n_dropped += 1
            continue
        caption = bbox_io.extract_caption_from_response(resp.get("response", {}))
        if caption is None:
            n_dropped += 1
            continue
        if not bbox_io.length_gate(caption):
            n_dropped += 1
            continue
        if bbox_io.mm_strip_filter(caption) is None:
            n_dropped += 1
            continue
        sft_records.append(_build_sft_record(record, caption))

    bbox_io.write_jsonl(args.out_dir / "sft.jsonl", sft_records)
    log.info("Wrote %d SFT records (%d dropped) to sft.jsonl", len(sft_records), n_dropped)
    return 0
```

- [ ] **Step 4: Run all tests, verify pass**

```bash
pytest tests/test_build_bbox_data.py tests/test_bbox_io.py tests/test_landmarks.py tests/test_region_bbox.py tests/test_vlm_config_batch.py -v
```

Expected: all pass (12 build + 9 io + 5 landmarks + 10 region_bbox + 3 vlm_config = 39 passed).

- [ ] **Step 5: Commit**

```bash
git add models/langslice-gemma-4/data/_stage_submit.py tests/test_build_bbox_data.py
git commit -m "feat(gemma-4): --stage submit with AI Studio Batch API + mm-strip filter"
```

---

## Task 13: SliceBench subject-leakage gate

**Files:**
- Modify: `models/langslice-gemma-4/data/build_bbox_data.py`
- Modify: `models/langslice-gemma-4/data/_stage_submit.py`
- Modify: `tests/test_build_bbox_data.py`
- Reference: SliceBench eval-holdout subject list (path supplied by user before submission)

**Spec sections covered:** §10 (pre-training validation; subject-level leakage).

- [ ] **Step 1: Write the failing leakage test**

Append to `tests/test_build_bbox_data.py`:

```python
def test_stage_submit_rejects_slicebench_subject_leakage(tmp_path: Path, monkeypatch):
    import _stage_submit as stage_submit  # type: ignore

    out_dir = tmp_path / "bbox_data"
    out_dir.mkdir(parents=True)
    draft = {
        "id": "bbox_000001",
        "atlas": "allen_mouse_25um",
        "atlas_version": "CCFv3",
        "orientation": "coronal",
        "region": "Hippocampal Formation",
        "source_type": "real_histology",
        "source_brain": "SLICEBENCH_HELDOUT_001",
        "is_hemisphere": False,
        "modality": None,
        "section_image_paths": [],
        "section_positions_mm": [3.0, 3.4, 3.8, 4.2],
        "bboxes": [{"left": [1, 1, 2, 2], "right": [3, 3, 4, 4]}] * 4,
    }
    (out_dir / "draft_manifest.jsonl").write_text(json.dumps(draft) + "\n", encoding="utf-8")
    verdicts = tmp_path / "verdicts.jsonl"
    verdicts.write_text(json.dumps({"id": "bbox_000001", "verdict": "verify"}) + "\n", encoding="utf-8")
    holdouts = tmp_path / "slicebench_holdout_subjects.txt"
    holdouts.write_text("SLICEBENCH_HELDOUT_001\n", encoding="utf-8")

    args = build_bbox_data.parse_args([
        "--stage", "submit",
        "--out-dir", str(out_dir),
        "--verdicts", str(verdicts),
        "--slicebench-holdout-subjects", str(holdouts),
        "--dry-run",
    ])
    assert stage_submit.run_stage_submit(args) == 1
```

Run it and verify it fails because the CLI flag and leakage check are missing:

```bash
pytest tests/test_build_bbox_data.py::test_stage_submit_rejects_slicebench_subject_leakage -v
```

- [ ] **Step 2: Add the CLI flag**

In `build_bbox_data.parse_args`, add:

```python
p.add_argument(
    "--slicebench-holdout-subjects",
    type=Path,
    default=None,
    help="Text or JSON list of SliceBench held-out subject IDs; submit fails on overlap.",
)
```

- [ ] **Step 3: Implement the leakage check**

In `_stage_submit.py`, add:

```python
def _load_holdout_subjects(path: Path | None) -> set[str]:
    if path is None:
        return set()
    text = path.read_text(encoding="utf-8").strip()
    if not text:
        return set()
    if text.startswith("["):
        return {str(item) for item in json.loads(text)}
    return {line.strip() for line in text.splitlines() if line.strip()}


def _assert_no_slicebench_leakage(draft: list[dict], holdouts: set[str]) -> bool:
    if not holdouts:
        return True
    leaked = sorted({
        str(record.get("source_brain"))
        for record in draft
        if record.get("source_type") == "real_histology"
        and record.get("source_brain") in holdouts
    })
    if leaked:
        log.error("SliceBench subject leakage detected: %s", ", ".join(leaked))
        return False
    return True
```

Call it in `run_stage_submit` immediately after loading `draft` and before building request JSONL:

```python
holdouts = _load_holdout_subjects(args.slicebench_holdout_subjects)
if not _assert_no_slicebench_leakage(draft, holdouts):
    return 1
```

- [ ] **Step 4: Run the leakage test, verify pass**

```bash
pytest tests/test_build_bbox_data.py::test_stage_submit_rejects_slicebench_subject_leakage -v
```

Expected: 1 passed.

- [ ] **Step 5: Commit**

```bash
git add models/langslice-gemma-4/data/build_bbox_data.py models/langslice-gemma-4/data/_stage_submit.py tests/test_build_bbox_data.py
git commit -m "feat(gemma-4): block SliceBench subject leakage before bbox submit"
```

---

## Task 14: Pilot run + final smoke

**Files:** none — operational task.

**Spec sections covered:** §12 (Volume + cost; pilot batch).

- [ ] **Step 1: Bake the coverage index**

```bash
python _local/eval/lib/landmark_coverage.py
```

Expected: prints brain count and writes `_local/eval/data/landmark_coverage.json`.

- [ ] **Step 2: Stage 1 — sample 100 examples**

```bash
python models/langslice-gemma-4/data/build_bbox_data.py --stage sample --target-total 100 --seed 0
```

Expected: ~100 records in `_local/bbox_data/draft_manifest.jsonl`, ~100 strips in `_local/bbox_data/draft_overlays/`.

- [ ] **Step 3: QC pass**

```bash
python _local/qc_app/app.py --mode bbox
```

Click through all examples; verify or reject each. Verdicts accumulate in `_local/bbox_data/qc_verdicts.jsonl`.

- [ ] **Step 4: Stage 2 — dry run**

```bash
python models/langslice-gemma-4/data/build_bbox_data.py --stage submit \
  --verdicts _local/bbox_data/qc_verdicts.jsonl --dry-run
```

Expected: writes `_local/bbox_data/request.jsonl` with one line per verified example. Spot-check a line — it should match the wire format in the spec §9.1.

- [ ] **Step 5: Stage 2 — live submit**

```bash
python models/langslice-gemma-4/data/build_bbox_data.py --stage submit \
  --verdicts _local/bbox_data/qc_verdicts.jsonl
```

Expected: uploads JSONL, creates batch, polls until succeeded (likely several minutes), writes `_local/bbox_data/sft.jsonl`. Spot-check a few captions — they should be 2-4 sentences, mention the region, no mm/section-index/bregma references.

- [ ] **Step 6: Inspect distribution**

```bash
python -c "
import json
from collections import Counter
recs = [json.loads(l) for l in open('_local/bbox_data/sft.jsonl', encoding='utf-8')]
print('Total:', len(recs))
print('By source:', Counter(r['source_type'] for r in recs))
print('By orientation:', Counter(r['orientation'] for r in recs))
print('By atlas:', Counter(r['atlas'] for r in recs))
"
```

Expected: print shows the per-(source, orientation, atlas) breakdown.

- [ ] **Step 7: Decide on full run**

If pilot quality is good (mm-strip drop rate <5%, captions descriptive, bboxes accurate), scale up:

```bash
python models/langslice-gemma-4/data/build_bbox_data.py --stage sample --target-total 400 --seed 1
# QC pass
python models/langslice-gemma-4/data/build_bbox_data.py --stage submit \
  --verdicts _local/bbox_data/qc_verdicts.jsonl
```

If quality is poor: bump `thinking_level` to MEDIUM in `bbox_io.pack_batch_request`, re-submit rejected examples only.

---

## Self-review checklist

**Spec coverage** — each spec section maps to at least one task:

- §3 (Example shape) → Task 7 (bbox_io.pack_batch_request)
- §3.1 (Bbox structure) → Task 3, Task 4, Task 7
- §3.2 (Variability) → Task 8 (sample_section_count, sample_spacings_mm, sample_anchor_mm)
- §3.3 (Single source per example) → Task 9 (loop in run_stage_sample picks one source per example)
- §4 (Source corpus) → Task 5, Task 8 (pick_source), Task 10
- §4.4 (Priority logic) → Task 8 (pick_source) + tests
- §5 (Region resolution) → Task 1, Task 2
- §6.1 (Atlas-slice path) → Task 3
- §6.2 (Real-section path) → Task 4
- §6.3 (Two-pass projection) → Task 5 (coarse) + Task 4 (fine)
- §7 (Module architecture) → File-structure table + Tasks 1-13
- §8.1 (Two-stage flow) → Task 9, Task 12
- §8.2 (QC app extension) → Task 11
- §9 (Batch API path + wire format) → Task 6, Task 7, Task 12
- §9.4 (Response extraction with thought-filter) → Task 7 (extract_caption_from_response) + Task 12 test
- §10 (Validation gates) → Task 7 (filters) + Task 12 (caption gating in run_stage_submit) + Task 13 (SliceBench subject-leakage gate)
- §11 (Outputs) → Tasks 9, 11, 12
- §12 (Volume + cost) → Task 14
- §13 (Risks) → Documentation only; no implementation
- §14 (Dependencies) → Pre-flight + Task 2
- §15 (Out of scope) → Documentation only
- §16 (References) → Documentation only

**Placeholder scan:** No "TBD", "implement later", "similar to Task N", or "add error handling" in any task body. Each step has the actual code or command.

**Type consistency:**
- `LandmarkLoader.resolve(name, atlas, atlas_name)` signature matches across Task 1, Task 5, Task 9, Task 10.
- `bbox_from_atlas_slice(annotation_slice, region_ids, is_hemisphere, hemisphere_slice=None)` and `bbox_from_real_section(section_image_shape, pixel_to_voxel, annotation_volume, region_ids, midline_x_at_row, is_hemisphere, grid_step)` are stable across the tests and the orchestrator's calls.
- `pick_source` returns `{source_type, source_brain, available_section_ids?}` consistently and accepts optional `source_counts` for the per-(brain, region) cap.
- `pack_batch_request(example, images_b64)` and `extract_caption_from_response(response)` match across tests and `_stage_submit`.
- Manifest record fields (`id`, `atlas`, `atlas_version`, `orientation`, `region`, `source_type`, `source_brain`, `modality`, `is_hemisphere`, `section_image_paths`, `section_positions_mm`, `bboxes`) appear identically in `_render_example_atlas`, `_render_example_real`, the QC app, and `_build_sft_record`.

**Open assumption:** The `_local/eval/data/manifest.jsonl` records carry a per-section `slice_record` field with `anchoring`, `markers`, `height`, and `width`. Verify this exists during pre-flight; if not, the manifest builder needs a small extension (out of scope here, but call it out before starting Task 5).
