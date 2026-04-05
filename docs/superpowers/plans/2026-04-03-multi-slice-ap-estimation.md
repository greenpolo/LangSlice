# Multi-Slice AP Estimation Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Implement whole-brain AP estimation (20-60 slices) using Google ADK for multi-agent orchestration with anchor estimation, interval interpolation, wave-based nano-banana refinement, and constraint enforcement.

**Architecture:** New `langslice/brain/` module containing pure-function algorithms (anchor selection, interpolation, windowing, constraints), ADK agent wrappers around existing estimators, and a pipeline orchestrator that composes them via SequentialAgent/ParallelAgent. The existing `estimate_position()` and `estimate_position_image_gen()` are called unmodified via `asyncio.to_thread()`.

**Tech Stack:** Google ADK (`google-adk`), asyncio, existing google-genai SDK, BrainGlobe atlas API.

**Spec:** `docs/superpowers/specs/2026-04-02-multi-slice-ap-estimation-design.md`

---

## File Structure

### New files

| File | Responsibility |
|------|---------------|
| `langslice/brain/__init__.py` | Public API: re-exports `run_brain_estimation`, `BrainEstimationConfig`, `BrainEstimationResult` |
| `langslice/brain/types.py` | Data classes: config, per-slice position, result, summary |
| `langslice/brain/discovery.py` | Image folder discovery + natural sort |
| `langslice/brain/anchor_selection.py` | Center-out anchor index selection |
| `langslice/brain/interpolation.py` | Interval-based interpolation + extrapolation + atlas bounds clamping |
| `langslice/brain/window.py` | Nano-banana search window bounds + dynamic image count |
| `langslice/brain/constraints.py` | Ordering enforcement (strict/loose/none) + minimum spacing |
| `langslice/brain/checkpoint.py` | Incremental JSON checkpoint read/write/resume |
| `langslice/brain/agents.py` | ADK `BaseAgent` subclasses: `AnchorAgent`, `RefinementAgent` |
| `langslice/brain/pipeline.py` | Wave computation, ADK pipeline composition, main async entry point |
| `tests/test_brain_discovery.py` | Tests for image discovery |
| `tests/test_brain_anchor_selection.py` | Tests for anchor selection |
| `tests/test_brain_interpolation.py` | Tests for interpolation + extrapolation |
| `tests/test_brain_window.py` | Tests for window construction |
| `tests/test_brain_constraints.py` | Tests for constraint enforcement |
| `tests/test_brain_checkpoint.py` | Tests for checkpoint I/O |
| `tests/test_brain_agents.py` | Tests for ADK agent wrappers (mocked estimators) |
| `tests/test_brain_pipeline.py` | Integration tests for wave computation and pipeline |

### Modified files

| File | Change |
|------|--------|
| `pyproject.toml` | Add `google-adk` dependency |
| `langslice/cli.py` | Add `estimate-brain` subcommand |

---

## Task 1: Add google-adk dependency

**Files:**
- Modify: `pyproject.toml`

- [ ] **Step 1: Add google-adk to dependencies**

In `pyproject.toml`, add `google-adk` to the `dependencies` list:

```toml
dependencies = [
    "google-genai",
    "google-adk",
    "brainglobe-atlasapi",
    "brainglobe-space",
    "numpy",
    "scipy",
    "Pillow",
    "tifffile",
    "python-dotenv",
    "itk-elastix",
]
```

- [ ] **Step 2: Install and verify**

Run:
```bash
pip install -e .
python -c "from google.adk.agents import BaseAgent, SequentialAgent, ParallelAgent; print('ADK OK')"
```
Expected: `ADK OK`

- [ ] **Step 3: Commit**

```bash
git add pyproject.toml
git commit -m "deps: add google-adk for multi-slice orchestration"
```

---

## Task 2: Data types

**Files:**
- Create: `langslice/brain/__init__.py`
- Create: `langslice/brain/types.py`
- Test: `tests/test_brain_types.py`

- [ ] **Step 1: Create the brain package**

Create `langslice/brain/__init__.py`:

```python
"""Whole-brain multi-slice AP estimation."""

from langslice.brain.types import (
    BrainEstimationConfig,
    BrainEstimationResult,
    BrainEstimationSummary,
    SlicePosition,
)

__all__ = [
    "BrainEstimationConfig",
    "BrainEstimationResult",
    "BrainEstimationSummary",
    "SlicePosition",
]
```

- [ ] **Step 2: Write the failing test**

Create `tests/test_brain_types.py`:

```python
from langslice.brain.types import (
    BrainEstimationConfig,
    BrainEstimationResult,
    BrainEstimationSummary,
    SlicePosition,
)


def test_slice_position_creation():
    sp = SlicePosition(
        filename="slice_001.tif",
        index=0,
        position_mm=1.30,
        source="anchor",
        locked=True,
    )
    assert sp.filename == "slice_001.tif"
    assert sp.position_mm == 1.30
    assert sp.source == "anchor"
    assert sp.locked is True


def test_config_interval_mm():
    cfg = BrainEstimationConfig(
        image_folder="/tmp/slices",
        atlas_name="allen_mouse_25um",
        thickness_um=50,
        interval_um=200,
        n_anchors=4,
        ordering="strict",
        refinement=True,
        max_parallel=4,
        z_axis="AP",
    )
    assert cfg.interval_mm == 0.200
    assert cfg.thickness_mm == 0.050


def test_result_to_dict():
    slices = [
        SlicePosition("a.tif", 0, 1.0, "anchor", True),
        SlicePosition("b.tif", 1, 1.2, "interpolated+refined", True),
    ]
    summary = BrainEstimationSummary(
        mean_interval_mm=0.2,
        std_interval_mm=0.01,
        n_slices=2,
        n_anchors=1,
        n_refined=1,
        n_skipped=0,
    )
    result = BrainEstimationResult(
        config=BrainEstimationConfig(
            image_folder="/tmp",
            atlas_name="allen_mouse_25um",
            thickness_um=50,
            interval_um=200,
            n_anchors=1,
            ordering="strict",
            refinement=True,
            max_parallel=4,
            z_axis="AP",
        ),
        slices=slices,
        summary=summary,
    )
    d = result.to_dict()
    assert d["atlas"] == "allen_mouse_25um"
    assert d["thickness_um"] == 50
    assert len(d["slices"]) == 2
    assert d["slices"][0]["filename"] == "a.tif"
    assert d["slices"][0]["source"] == "anchor"
```

- [ ] **Step 3: Run test to verify it fails**

Run: `python -m pytest tests/test_brain_types.py -v`
Expected: FAIL (module not found)

- [ ] **Step 4: Implement types.py**

Create `langslice/brain/types.py`:

```python
"""Data classes for whole-brain AP estimation."""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class BrainEstimationConfig:
    """User-provided configuration for a whole-brain estimation run."""

    image_folder: str
    atlas_name: str
    thickness_um: int
    interval_um: int
    n_anchors: int
    ordering: str  # "strict" | "loose" | "none"
    refinement: bool
    max_parallel: int
    z_axis: str  # "AP" | "PA"

    @property
    def thickness_mm(self) -> float:
        return self.thickness_um / 1000.0

    @property
    def interval_mm(self) -> float:
        return self.interval_um / 1000.0


@dataclass
class SlicePosition:
    """Position state for a single slice."""

    filename: str
    index: int
    position_mm: float
    source: str  # "anchor", "interpolated", "extrapolated", "*+refined"
    locked: bool


@dataclass
class BrainEstimationSummary:
    """Summary statistics for a completed run."""

    mean_interval_mm: float
    std_interval_mm: float
    n_slices: int
    n_anchors: int
    n_refined: int
    n_skipped: int


@dataclass
class BrainEstimationResult:
    """Complete result of a whole-brain estimation run."""

    config: BrainEstimationConfig
    slices: list[SlicePosition]
    summary: BrainEstimationSummary

    def to_dict(self) -> dict:
        """Serialize to a JSON-compatible dict."""
        return {
            "atlas": self.config.atlas_name,
            "thickness_um": self.config.thickness_um,
            "interval_um": self.config.interval_um,
            "ordering_mode": self.config.ordering,
            "z_axis": self.config.z_axis,
            "slices": [
                {
                    "filename": s.filename,
                    "position_mm": round(s.position_mm, 4),
                    "source": s.source,
                }
                for s in self.slices
            ],
        }
```

- [ ] **Step 5: Run test to verify it passes**

Run: `python -m pytest tests/test_brain_types.py -v`
Expected: all 3 tests PASS

- [ ] **Step 6: Run linting**

Run: `python -m ruff check langslice/brain/ tests/test_brain_types.py`
Expected: clean

- [ ] **Step 7: Commit**

```bash
git add langslice/brain/__init__.py langslice/brain/types.py tests/test_brain_types.py
git commit -m "feat(brain): add data types for whole-brain AP estimation"
```

---

## Task 3: Image discovery

**Files:**
- Create: `langslice/brain/discovery.py`
- Test: `tests/test_brain_discovery.py`

- [ ] **Step 1: Write the failing test**

Create `tests/test_brain_discovery.py`:

```python
import os
import tempfile
from pathlib import Path

from langslice.brain.discovery import discover_slices


def test_discover_natural_sort(tmp_path: Path):
    """Filenames with numbers sort naturally, not lexicographically."""
    for name in ["slice_2.tif", "slice_10.tif", "slice_1.tif"]:
        (tmp_path / name).write_bytes(b"fake")
    result = discover_slices(str(tmp_path))
    assert [os.path.basename(r) for r in result] == [
        "slice_1.tif",
        "slice_2.tif",
        "slice_10.tif",
    ]


def test_discover_filters_extensions(tmp_path: Path):
    """Only image files are returned; other files are ignored."""
    (tmp_path / "slice_01.tif").write_bytes(b"fake")
    (tmp_path / "slice_02.png").write_bytes(b"fake")
    (tmp_path / "notes.txt").write_bytes(b"fake")
    (tmp_path / "data.csv").write_bytes(b"fake")
    result = discover_slices(str(tmp_path))
    names = [os.path.basename(r) for r in result]
    assert len(names) == 2
    assert "notes.txt" not in names
    assert "data.csv" not in names


def test_discover_empty_folder(tmp_path: Path):
    """Empty folder returns empty list."""
    result = discover_slices(str(tmp_path))
    assert result == []


def test_discover_mixed_extensions(tmp_path: Path):
    """All supported image extensions are found."""
    for name in ["a.png", "b.jpg", "c.jpeg", "d.tif", "e.tiff"]:
        (tmp_path / name).write_bytes(b"fake")
    result = discover_slices(str(tmp_path))
    assert len(result) == 5
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m pytest tests/test_brain_discovery.py -v`
Expected: FAIL (module not found)

- [ ] **Step 3: Implement discovery.py**

Create `langslice/brain/discovery.py`:

```python
"""Discover and naturally sort slice images in a folder."""

from __future__ import annotations

import os
import re

_IMAGE_EXTENSIONS = {".png", ".jpg", ".jpeg", ".tif", ".tiff"}

_NATURAL_SORT_RE = re.compile(r"(\d+)")


def _natural_sort_key(path: str) -> list[str | int]:
    """Sort key that orders embedded numbers numerically."""
    basename = os.path.basename(path).lower()
    parts: list[str | int] = []
    for piece in _NATURAL_SORT_RE.split(basename):
        if piece.isdigit():
            parts.append(int(piece))
        else:
            parts.append(piece)
    return parts


def discover_slices(folder: str) -> list[str]:
    """Return absolute paths to slice images in *folder*, naturally sorted.

    Scans for files with extensions: .png, .jpg, .jpeg, .tif, .tiff.
    Non-image files are silently skipped.
    """
    hits: list[str] = []
    for entry in os.listdir(folder):
        ext = os.path.splitext(entry)[1].lower()
        if ext in _IMAGE_EXTENSIONS:
            hits.append(os.path.join(folder, entry))
    hits.sort(key=_natural_sort_key)
    return hits
```

- [ ] **Step 4: Run test to verify it passes**

Run: `python -m pytest tests/test_brain_discovery.py -v`
Expected: all 4 tests PASS

- [ ] **Step 5: Commit**

```bash
git add langslice/brain/discovery.py tests/test_brain_discovery.py
git commit -m "feat(brain): image discovery with natural sort"
```

---

## Task 4: Anchor selection

**Files:**
- Create: `langslice/brain/anchor_selection.py`
- Test: `tests/test_brain_anchor_selection.py`

- [ ] **Step 1: Write the failing test**

Create `tests/test_brain_anchor_selection.py`:

```python
from langslice.brain.anchor_selection import select_anchor_indices


def test_single_anchor_picks_midpoint():
    # 20 slices (indices 0..19), 1 anchor -> midpoint index 9
    result = select_anchor_indices(n_slices=20, n_anchors=1)
    assert result == [9]


def test_two_anchors_trisect():
    # 40 slices (0..39), 2 anchors -> indices ~13, 26
    result = select_anchor_indices(n_slices=40, n_anchors=2)
    assert len(result) == 2
    assert result == sorted(result)
    # Both should be in the middle third, not at extremes
    assert result[0] > 5
    assert result[1] < 35


def test_four_anchors_center_weighted():
    # 20 slices, 4 anchors -> should be spread with center priority
    result = select_anchor_indices(n_slices=20, n_anchors=4)
    assert len(result) == 4
    assert result == sorted(result)
    # No duplicates
    assert len(set(result)) == 4
    # All within bounds
    assert all(0 <= i < 20 for i in result)


def test_anchors_equal_slices():
    # 4 slices, 4 anchors -> every slice is an anchor
    result = select_anchor_indices(n_slices=4, n_anchors=4)
    assert result == [0, 1, 2, 3]


def test_one_slice_one_anchor():
    result = select_anchor_indices(n_slices=1, n_anchors=1)
    assert result == [0]


def test_anchors_never_exceed_slices():
    # More anchors than slices -> clamp
    result = select_anchor_indices(n_slices=3, n_anchors=10)
    assert len(result) == 3
    assert result == [0, 1, 2]


def test_center_out_avoids_extremes():
    # With few anchors on a large set, none should be at index 0 or n-1
    result = select_anchor_indices(n_slices=60, n_anchors=3)
    assert 0 not in result
    assert 59 not in result
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m pytest tests/test_brain_anchor_selection.py -v`
Expected: FAIL (module not found)

- [ ] **Step 3: Implement anchor_selection.py**

Create `langslice/brain/anchor_selection.py`:

```python
"""Center-out anchor index selection.

Anchors are placed starting from the center of the slice list and expanding
outward.  Anterior brain slices (especially in mouse) lack visually distinct
tissue and are unreliable for AP estimation, so we avoid placing anchors at
the extremes unless the user requests enough to cover the full range.
"""

from __future__ import annotations


def select_anchor_indices(n_slices: int, n_anchors: int) -> list[int]:
    """Return sorted 0-based indices for anchor slices.

    Places anchors center-out: the midpoint is chosen first, then positions
    expand symmetrically.  When *n_anchors* >= *n_slices*, every slice is an
    anchor.
    """
    if n_slices <= 0:
        return []
    n_anchors = min(n_anchors, n_slices)
    if n_anchors == n_slices:
        return list(range(n_slices))

    # Evenly space n_anchors points across the range, offset inward from edges.
    # gap = total_range / (n_anchors + 1) keeps anchors away from index 0 and n-1.
    gap = n_slices / (n_anchors + 1)
    indices: list[int] = []
    for k in range(1, n_anchors + 1):
        idx = int(round(k * gap)) - 1  # -1 for 0-based
        idx = max(0, min(idx, n_slices - 1))
        if idx not in indices:
            indices.append(idx)

    # If rounding caused duplicates or we're short, fill from center outward
    if len(indices) < n_anchors:
        mid = n_slices // 2
        for offset in range(n_slices):
            for candidate in [mid + offset, mid - offset]:
                if 0 <= candidate < n_slices and candidate not in indices:
                    indices.append(candidate)
                    if len(indices) == n_anchors:
                        break
            if len(indices) == n_anchors:
                break

    indices.sort()
    return indices
```

- [ ] **Step 4: Run test to verify it passes**

Run: `python -m pytest tests/test_brain_anchor_selection.py -v`
Expected: all 7 tests PASS

- [ ] **Step 5: Commit**

```bash
git add langslice/brain/anchor_selection.py tests/test_brain_anchor_selection.py
git commit -m "feat(brain): center-out anchor selection algorithm"
```

---

## Task 5: Interpolation + extrapolation

**Files:**
- Create: `langslice/brain/interpolation.py`
- Test: `tests/test_brain_interpolation.py`

- [ ] **Step 1: Write the failing test**

Create `tests/test_brain_interpolation.py`:

```python
import pytest

from langslice.brain.interpolation import interpolate_positions
from langslice.brain.types import SlicePosition


def _locked(filename: str, index: int, position_mm: float) -> SlicePosition:
    return SlicePosition(filename, index, position_mm, "anchor", locked=True)


def _unlocked(filename: str, index: int) -> SlicePosition:
    return SlicePosition(filename, index, 0.0, "", locked=False)


def test_interpolate_between_two_anchors():
    """Even spacing between two anchors."""
    slices = [
        _locked("s0.tif", 0, 1.0),
        _unlocked("s1.tif", 1),
        _unlocked("s2.tif", 2),
        _locked("s3.tif", 3, 1.6),
    ]
    result = interpolate_positions(slices, interval_mm=0.200, atlas_range=(0.0, 13.0), z_axis="AP")
    assert result[1].position_mm == pytest.approx(1.2, abs=0.001)
    assert result[2].position_mm == pytest.approx(1.4, abs=0.001)
    assert result[1].source == "interpolated"
    assert result[1].locked is False


def test_extrapolate_before_first_anchor():
    """Slices before the first anchor use the average interval."""
    slices = [
        _unlocked("s0.tif", 0),
        _unlocked("s1.tif", 1),
        _locked("s2.tif", 2, 2.0),
        _locked("s5.tif", 5, 2.6),
    ]
    result = interpolate_positions(slices, interval_mm=0.200, atlas_range=(0.0, 13.0), z_axis="AP")
    assert result[1].position_mm == pytest.approx(1.8, abs=0.001)
    assert result[0].position_mm == pytest.approx(1.6, abs=0.001)
    assert result[0].source == "extrapolated"


def test_extrapolate_clamped_to_atlas_bounds():
    """Extrapolation does not go below 0.0mm."""
    slices = [
        _unlocked("s0.tif", 0),
        _locked("s1.tif", 1, 0.1),
        _locked("s3.tif", 3, 0.5),
    ]
    result = interpolate_positions(slices, interval_mm=0.200, atlas_range=(0.0, 13.0), z_axis="AP")
    assert result[0].position_mm >= 0.0


def test_pa_axis_extrapolates_correctly():
    """PA z-axis: first slice has highest AP, last has lowest."""
    slices = [
        _locked("s0.tif", 0, 8.0),
        _unlocked("s1.tif", 1),
        _locked("s2.tif", 2, 7.6),
    ]
    result = interpolate_positions(slices, interval_mm=0.200, atlas_range=(0.0, 13.0), z_axis="PA")
    assert result[1].position_mm == pytest.approx(7.8, abs=0.001)


def test_anchors_unchanged():
    """Locked anchor positions are never modified."""
    slices = [
        _locked("s0.tif", 0, 1.0),
        _unlocked("s1.tif", 1),
        _locked("s2.tif", 2, 1.5),
    ]
    result = interpolate_positions(slices, interval_mm=0.200, atlas_range=(0.0, 13.0), z_axis="AP")
    assert result[0].position_mm == 1.0
    assert result[2].position_mm == 1.5
    assert result[0].locked is True
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m pytest tests/test_brain_interpolation.py -v`
Expected: FAIL (module not found)

- [ ] **Step 3: Implement interpolation.py**

Create `langslice/brain/interpolation.py`:

```python
"""Interval-based interpolation and extrapolation of slice positions."""

from __future__ import annotations

from langslice.brain.types import SlicePosition


def interpolate_positions(
    slices: list[SlicePosition],
    *,
    interval_mm: float,
    atlas_range: tuple[float, float],
    z_axis: str,
) -> list[SlicePosition]:
    """Assign positions to unlocked slices based on locked anchors.

    Between adjacent anchors: distribute evenly (interval as baseline, residual
    spread across gaps).  Beyond outermost anchors: step outward using
    *interval_mm*, clamped to *atlas_range*.

    Returns a new list — locked slices are copied unchanged.
    """
    lo, hi = atlas_range
    n = len(slices)
    if n == 0:
        return []

    # Copy so we don't mutate the input
    out = [
        SlicePosition(s.filename, s.index, s.position_mm, s.source, s.locked)
        for s in slices
    ]

    # Direction multiplier: AP means increasing index → increasing mm
    direction = 1.0 if z_axis == "AP" else -1.0
    step = interval_mm * direction

    # Collect locked anchor indices
    anchor_idxs = [i for i, s in enumerate(out) if s.locked]
    if not anchor_idxs:
        return out

    # --- Interpolate between each pair of adjacent anchors ---
    for a_idx, b_idx in zip(anchor_idxs, anchor_idxs[1:]):
        a_pos = out[a_idx].position_mm
        b_pos = out[b_idx].position_mm
        n_gaps = b_idx - a_idx
        if n_gaps <= 1:
            continue
        gap_step = (b_pos - a_pos) / n_gaps
        for k in range(1, n_gaps):
            i = a_idx + k
            out[i] = SlicePosition(
                out[i].filename,
                out[i].index,
                a_pos + k * gap_step,
                "interpolated",
                locked=False,
            )

    # --- Extrapolate before the first anchor ---
    first_anchor = anchor_idxs[0]
    for k in range(1, first_anchor + 1):
        i = first_anchor - k
        pos = out[first_anchor].position_mm - k * step
        pos = max(lo, min(hi, pos))
        out[i] = SlicePosition(
            out[i].filename, out[i].index, pos, "extrapolated", locked=False
        )

    # --- Extrapolate after the last anchor ---
    last_anchor = anchor_idxs[-1]
    for k in range(1, n - last_anchor):
        i = last_anchor + k
        pos = out[last_anchor].position_mm + k * step
        pos = max(lo, min(hi, pos))
        out[i] = SlicePosition(
            out[i].filename, out[i].index, pos, "extrapolated", locked=False
        )

    return out
```

- [ ] **Step 4: Run test to verify it passes**

Run: `python -m pytest tests/test_brain_interpolation.py -v`
Expected: all 5 tests PASS

- [ ] **Step 5: Commit**

```bash
git add langslice/brain/interpolation.py tests/test_brain_interpolation.py
git commit -m "feat(brain): interval-based interpolation and extrapolation"
```

---

## Task 6: Window construction

**Files:**
- Create: `langslice/brain/window.py`
- Test: `tests/test_brain_window.py`

- [ ] **Step 1: Write the failing test**

Create `tests/test_brain_window.py`:

```python
import pytest

from langslice.brain.window import compute_refinement_window, RefinementWindow


def test_window_locked_both_sides():
    """Window bounded by two locked neighbors."""
    win = compute_refinement_window(
        position_mm=2.5,
        left_locked_mm=2.3,
        right_locked_mm=2.7,
        thickness_mm=0.050,
        interval_mm=0.200,
    )
    assert win.lo == pytest.approx(2.35, abs=0.001)  # 2.3 + 0.05
    assert win.hi == pytest.approx(2.65, abs=0.001)  # 2.7 - 0.05
    assert win.center == pytest.approx(2.5)
    assert win.n_images >= 5


def test_window_locked_left_only():
    """Only left neighbor locked; right uses interpolated + interval."""
    win = compute_refinement_window(
        position_mm=2.5,
        left_locked_mm=2.3,
        right_locked_mm=None,
        thickness_mm=0.050,
        interval_mm=0.200,
    )
    assert win.lo == pytest.approx(2.35, abs=0.001)
    assert win.hi == pytest.approx(2.7, abs=0.001)  # center + interval
    assert win.n_images >= 5


def test_window_locked_right_only():
    """Only right neighbor locked."""
    win = compute_refinement_window(
        position_mm=2.5,
        left_locked_mm=None,
        right_locked_mm=2.7,
        thickness_mm=0.050,
        interval_mm=0.200,
    )
    assert win.lo == pytest.approx(2.3, abs=0.001)  # center - interval
    assert win.hi == pytest.approx(2.65, abs=0.001)


def test_window_skip_when_too_narrow():
    """Window smaller than thickness -> skip (n_images=0)."""
    win = compute_refinement_window(
        position_mm=2.5,
        left_locked_mm=2.48,
        right_locked_mm=2.52,
        thickness_mm=0.050,
        interval_mm=0.200,
    )
    assert win.skip is True
    assert win.n_images == 0


def test_window_image_count_scales():
    """Wide window gets more images than narrow window."""
    wide = compute_refinement_window(
        position_mm=5.0,
        left_locked_mm=4.7,
        right_locked_mm=5.3,
        thickness_mm=0.050,
        interval_mm=0.200,
    )
    narrow = compute_refinement_window(
        position_mm=5.0,
        left_locked_mm=4.9,
        right_locked_mm=5.1,
        thickness_mm=0.050,
        interval_mm=0.200,
    )
    assert wide.n_images > narrow.n_images


def test_window_max_images_capped():
    """Even a very wide window caps at 13 images."""
    win = compute_refinement_window(
        position_mm=5.0,
        left_locked_mm=3.0,
        right_locked_mm=7.0,
        thickness_mm=0.050,
        interval_mm=0.200,
    )
    assert win.n_images <= 13
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m pytest tests/test_brain_window.py -v`
Expected: FAIL (module not found)

- [ ] **Step 3: Implement window.py**

Create `langslice/brain/window.py`:

```python
"""Nano-banana search window construction."""

from __future__ import annotations

import math
from dataclasses import dataclass

_IMAGE_SPACING_MM = 0.025
_MIN_IMAGES = 5
_MAX_IMAGES = 13


@dataclass(frozen=True)
class RefinementWindow:
    """Search window for a single slice's nano-banana refinement."""

    lo: float
    hi: float
    center: float
    n_images: int
    skip: bool


def compute_refinement_window(
    *,
    position_mm: float,
    left_locked_mm: float | None,
    right_locked_mm: float | None,
    thickness_mm: float,
    interval_mm: float,
) -> RefinementWindow:
    """Compute the atlas search window for one slice's nano-banana pass.

    Bounds are derived from locked neighbors.  If no locked neighbor exists on
    a side, fall back to ``position_mm +/- interval_mm``.  Returns ``skip=True``
    when the window is narrower than *thickness_mm*.
    """
    if left_locked_mm is not None:
        lo = left_locked_mm + thickness_mm
    else:
        lo = position_mm - interval_mm

    if right_locked_mm is not None:
        hi = right_locked_mm - thickness_mm
    else:
        hi = position_mm + interval_mm

    # Ensure lo <= hi (can happen with tight neighbors)
    if lo > hi:
        lo, hi = hi, lo

    width = hi - lo

    if width < thickness_mm:
        return RefinementWindow(lo=lo, hi=hi, center=position_mm, n_images=0, skip=True)

    n_images = max(_MIN_IMAGES, min(_MAX_IMAGES, math.floor(width / _IMAGE_SPACING_MM)))

    return RefinementWindow(lo=lo, hi=hi, center=position_mm, n_images=n_images, skip=False)
```

- [ ] **Step 4: Run test to verify it passes**

Run: `python -m pytest tests/test_brain_window.py -v`
Expected: all 6 tests PASS

- [ ] **Step 5: Commit**

```bash
git add langslice/brain/window.py tests/test_brain_window.py
git commit -m "feat(brain): nano-banana search window construction"
```

---

## Task 7: Constraint enforcement

**Files:**
- Create: `langslice/brain/constraints.py`
- Test: `tests/test_brain_constraints.py`

- [ ] **Step 1: Write the failing test**

Create `tests/test_brain_constraints.py`:

```python
import pytest

from langslice.brain.constraints import enforce_constraints
from langslice.brain.types import SlicePosition


def _sp(index: int, pos: float) -> SlicePosition:
    return SlicePosition(f"s{index}.tif", index, pos, "refined", locked=True)


def test_strict_already_monotonic():
    """No changes when positions are already in order."""
    slices = [_sp(0, 1.0), _sp(1, 1.2), _sp(2, 1.4)]
    result = enforce_constraints(slices, ordering="strict", thickness_mm=0.050, z_axis="AP")
    assert [s.position_mm for s in result] == [1.0, 1.2, 1.4]


def test_strict_clamps_violation():
    """Strict mode clamps a non-monotonic slice to midpoint of neighbors."""
    slices = [_sp(0, 1.0), _sp(1, 1.5), _sp(2, 1.3)]  # index 2 violates
    result = enforce_constraints(slices, ordering="strict", thickness_mm=0.050, z_axis="AP")
    positions = [s.position_mm for s in result]
    # Slice 2 should be clamped to > slice 1
    assert positions[2] > positions[1]


def test_strict_enforces_minimum_spacing():
    """Two slices closer than thickness get nudged apart."""
    slices = [_sp(0, 1.0), _sp(1, 1.02), _sp(2, 1.3)]  # 0.02 < 0.05 thickness
    result = enforce_constraints(slices, ordering="strict", thickness_mm=0.050, z_axis="AP")
    positions = [s.position_mm for s in result]
    assert positions[1] - positions[0] >= 0.050 - 0.001


def test_loose_swaps_reversed_pair():
    """Loose mode swaps two adjacent slices that are reversed."""
    slices = [_sp(0, 1.0), _sp(1, 1.4), _sp(2, 1.2), _sp(3, 1.6)]
    result = enforce_constraints(slices, ordering="loose", thickness_mm=0.050, z_axis="AP")
    positions = [s.position_mm for s in result]
    # After swap, should be monotonic
    assert positions == sorted(positions)


def test_none_no_reordering():
    """None mode does not enforce monotonicity."""
    slices = [_sp(0, 1.0), _sp(1, 1.5), _sp(2, 1.3)]
    result = enforce_constraints(slices, ordering="none", thickness_mm=0.050, z_axis="AP")
    positions = [s.position_mm for s in result]
    assert positions == [1.0, 1.5, 1.3]


def test_none_still_enforces_minimum_spacing():
    """Even in none mode, minimum spacing is enforced."""
    slices = [_sp(0, 1.0), _sp(1, 1.02)]
    result = enforce_constraints(slices, ordering="none", thickness_mm=0.050, z_axis="AP")
    positions = [s.position_mm for s in result]
    assert abs(positions[1] - positions[0]) >= 0.050 - 0.001


def test_pa_axis_strict():
    """PA axis: monotonically decreasing is valid."""
    slices = [_sp(0, 8.0), _sp(1, 7.8), _sp(2, 7.6)]
    result = enforce_constraints(slices, ordering="strict", thickness_mm=0.050, z_axis="PA")
    positions = [s.position_mm for s in result]
    assert positions == [8.0, 7.8, 7.6]
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m pytest tests/test_brain_constraints.py -v`
Expected: FAIL (module not found)

- [ ] **Step 3: Implement constraints.py**

Create `langslice/brain/constraints.py`:

```python
"""Constraint enforcement for slice positions."""

from __future__ import annotations

from langslice.brain.types import SlicePosition


def enforce_constraints(
    slices: list[SlicePosition],
    *,
    ordering: str,
    thickness_mm: float,
    z_axis: str,
) -> list[SlicePosition]:
    """Validate and fix ordering/spacing violations.

    Returns a new list.  Locked positions may be nudged to satisfy hard
    constraints (minimum spacing).
    """
    if len(slices) <= 1:
        return list(slices)

    out = [
        SlicePosition(s.filename, s.index, s.position_mm, s.source, s.locked)
        for s in slices
    ]

    # For PA axis, we work in negated space so "increasing" logic applies,
    # then negate back.
    if z_axis == "PA":
        for s in out:
            s.position_mm = -s.position_mm

    if ordering == "loose":
        _apply_swaps(out)

    if ordering in ("strict", "loose"):
        _enforce_monotonic(out, thickness_mm)

    _enforce_min_spacing(out, thickness_mm)

    if z_axis == "PA":
        for s in out:
            s.position_mm = -s.position_mm

    return out


def _apply_swaps(slices: list[SlicePosition]) -> None:
    """Scan for adjacent pairs that are reversed and swap them (one pass)."""
    i = 0
    while i < len(slices) - 1:
        if slices[i].position_mm > slices[i + 1].position_mm:
            # Swap positions (keep filenames/indices in place)
            slices[i].position_mm, slices[i + 1].position_mm = (
                slices[i + 1].position_mm,
                slices[i].position_mm,
            )
            slices[i].source, slices[i + 1].source = (
                slices[i + 1].source,
                slices[i].source,
            )
            i += 2  # skip past swapped pair to avoid cascading
        else:
            i += 1


def _enforce_monotonic(slices: list[SlicePosition], thickness_mm: float) -> None:
    """Clamp any non-monotonic slice to midpoint between neighbors."""
    for i in range(1, len(slices)):
        if slices[i].position_mm <= slices[i - 1].position_mm:
            if i < len(slices) - 1:
                mid = (slices[i - 1].position_mm + slices[i + 1].position_mm) / 2
                slices[i].position_mm = max(
                    slices[i - 1].position_mm + thickness_mm, mid
                )
            else:
                slices[i].position_mm = slices[i - 1].position_mm + thickness_mm


def _enforce_min_spacing(slices: list[SlicePosition], thickness_mm: float) -> None:
    """Nudge slices that are closer than thickness_mm apart."""
    for i in range(1, len(slices)):
        gap = abs(slices[i].position_mm - slices[i - 1].position_mm)
        if gap < thickness_mm:
            if slices[i].position_mm >= slices[i - 1].position_mm:
                slices[i].position_mm = slices[i - 1].position_mm + thickness_mm
            else:
                slices[i].position_mm = slices[i - 1].position_mm - thickness_mm
```

- [ ] **Step 4: Run test to verify it passes**

Run: `python -m pytest tests/test_brain_constraints.py -v`
Expected: all 7 tests PASS

- [ ] **Step 5: Commit**

```bash
git add langslice/brain/constraints.py tests/test_brain_constraints.py
git commit -m "feat(brain): constraint enforcement (strict/loose/none ordering)"
```

---

## Task 8: Checkpoint I/O

**Files:**
- Create: `langslice/brain/checkpoint.py`
- Test: `tests/test_brain_checkpoint.py`

- [ ] **Step 1: Write the failing test**

Create `tests/test_brain_checkpoint.py`:

```python
import json
from pathlib import Path

from langslice.brain.checkpoint import save_checkpoint, load_checkpoint
from langslice.brain.types import BrainEstimationConfig, SlicePosition


def _cfg() -> BrainEstimationConfig:
    return BrainEstimationConfig(
        image_folder="/tmp",
        atlas_name="allen_mouse_25um",
        thickness_um=50,
        interval_um=200,
        n_anchors=2,
        ordering="strict",
        refinement=True,
        max_parallel=4,
        z_axis="AP",
    )


def test_save_and_load_roundtrip(tmp_path: Path):
    path = str(tmp_path / "checkpoint.json")
    slices = [
        SlicePosition("a.tif", 0, 1.0, "anchor", True),
        SlicePosition("b.tif", 1, 1.2, "interpolated", False),
    ]
    save_checkpoint(path, _cfg(), slices)
    loaded_slices = load_checkpoint(path)
    assert len(loaded_slices) == 2
    assert loaded_slices[0].filename == "a.tif"
    assert loaded_slices[0].position_mm == 1.0
    assert loaded_slices[0].locked is True
    assert loaded_slices[1].locked is False


def test_incremental_save(tmp_path: Path):
    """Saving again with updated slices overwrites the file."""
    path = str(tmp_path / "checkpoint.json")
    slices = [SlicePosition("a.tif", 0, 1.0, "anchor", True)]
    save_checkpoint(path, _cfg(), slices)

    slices.append(SlicePosition("b.tif", 1, 1.2, "refined", True))
    save_checkpoint(path, _cfg(), slices)

    loaded = load_checkpoint(path)
    assert len(loaded) == 2


def test_load_nonexistent_returns_empty(tmp_path: Path):
    path = str(tmp_path / "nonexistent.json")
    loaded = load_checkpoint(path)
    assert loaded == []


def test_checkpoint_json_is_human_readable(tmp_path: Path):
    path = str(tmp_path / "checkpoint.json")
    slices = [SlicePosition("a.tif", 0, 1.0, "anchor", True)]
    save_checkpoint(path, _cfg(), slices)
    with open(path) as f:
        data = json.load(f)
    assert "slices" in data
    assert data["slices"][0]["filename"] == "a.tif"
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m pytest tests/test_brain_checkpoint.py -v`
Expected: FAIL (module not found)

- [ ] **Step 3: Implement checkpoint.py**

Create `langslice/brain/checkpoint.py`:

```python
"""Incremental JSON checkpoint for whole-brain estimation."""

from __future__ import annotations

import json
import os

from langslice.brain.types import BrainEstimationConfig, SlicePosition


def save_checkpoint(
    path: str,
    config: BrainEstimationConfig,
    slices: list[SlicePosition],
) -> None:
    """Write current state to a JSON file (overwrites)."""
    data = {
        "atlas": config.atlas_name,
        "thickness_um": config.thickness_um,
        "interval_um": config.interval_um,
        "ordering_mode": config.ordering,
        "z_axis": config.z_axis,
        "slices": [
            {
                "filename": s.filename,
                "index": s.index,
                "position_mm": round(s.position_mm, 4),
                "source": s.source,
                "locked": s.locked,
            }
            for s in slices
        ],
    }
    with open(path, "w") as f:
        json.dump(data, f, indent=2)


def load_checkpoint(path: str) -> list[SlicePosition]:
    """Load slice positions from a checkpoint file.

    Returns an empty list if the file does not exist.
    """
    if not os.path.exists(path):
        return []
    with open(path) as f:
        data = json.load(f)
    return [
        SlicePosition(
            filename=s["filename"],
            index=s["index"],
            position_mm=s["position_mm"],
            source=s["source"],
            locked=s["locked"],
        )
        for s in data["slices"]
    ]
```

- [ ] **Step 4: Run test to verify it passes**

Run: `python -m pytest tests/test_brain_checkpoint.py -v`
Expected: all 4 tests PASS

- [ ] **Step 5: Commit**

```bash
git add langslice/brain/checkpoint.py tests/test_brain_checkpoint.py
git commit -m "feat(brain): JSON checkpoint read/write for resumability"
```

---

## Task 9: ADK agent wrappers

**Files:**
- Create: `langslice/brain/agents.py`
- Test: `tests/test_brain_agents.py`

- [ ] **Step 1: Write the failing test**

Create `tests/test_brain_agents.py`:

```python
"""Tests for ADK agent wrappers with mocked estimators."""

import asyncio
from types import SimpleNamespace
from unittest.mock import patch

from langslice.brain.agents import run_anchor_estimation, run_refinement
from langslice.ai.estimator import APResult


def test_run_anchor_estimation():
    """Anchor estimation calls estimate_position then nano-banana."""
    coarse_result = APResult(position_mm=3.45, reasoning="coarse", debug_dir=None)
    fine_result = APResult(position_mm=3.42, reasoning="fine", debug_dir=None)

    call_log = []

    def fake_estimate(image, atlas_name, **kwargs):
        call_log.append("coarse")
        return coarse_result

    def fake_image_gen(image, atlas_name, **kwargs):
        call_log.append("fine")
        return fine_result

    with (
        patch("langslice.brain.agents.estimate_position", fake_estimate),
        patch("langslice.brain.agents.estimate_position_image_gen", fake_image_gen),
    ):
        result = asyncio.run(
            run_anchor_estimation(
                image_path="/fake/slice.tif",
                atlas_name="allen_mouse_25um",
            )
        )

    assert result.position_mm == 3.42
    assert call_log == ["coarse", "fine"]


def test_run_refinement():
    """Refinement calls nano-banana with window-constrained positions."""
    fine_result = APResult(position_mm=2.55, reasoning="refined", debug_dir=None)

    captured_kwargs: dict = {}

    def fake_image_gen(image, atlas_name, **kwargs):
        captured_kwargs.update(kwargs)
        return fine_result

    with (
        patch("langslice.brain.agents.estimate_position_image_gen", fake_image_gen),
        patch("langslice.brain.agents.Image") as mock_pil,
    ):
        mock_pil.open.return_value = mock_pil
        mock_pil.convert.return_value = mock_pil
        result = asyncio.run(
            run_refinement(
                image_path="/fake/slice.tif",
                atlas_name="allen_mouse_25um",
                window_lo=2.3,
                window_hi=2.7,
                window_center=2.5,
                n_images=8,
            )
        )

    assert result.position_mm == 2.55


def test_run_refinement_returns_none_on_skip():
    """When skip=True is signalled via n_images=0, returns None."""
    result = asyncio.run(
        run_refinement(
            image_path="/fake/slice.tif",
            atlas_name="allen_mouse_25um",
            window_lo=2.48,
            window_hi=2.52,
            window_center=2.5,
            n_images=0,
        )
    )
    assert result is None
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m pytest tests/test_brain_agents.py -v`
Expected: FAIL (module not found)

- [ ] **Step 3: Implement agents.py**

Create `langslice/brain/agents.py`:

```python
"""ADK-compatible async wrappers around existing AP estimators.

These functions are designed to be called from ADK BaseAgent subclasses
or directly from the pipeline orchestrator via asyncio.  They delegate
to the existing synchronous estimator code via ``asyncio.to_thread()``.
"""

from __future__ import annotations

import asyncio
import logging

from PIL import Image

from langslice.ai.estimator import APResult, estimate_position
from langslice.ai.estimator_image_gen import estimate_position_image_gen
from langslice.image_prep import normalize_image

logger = logging.getLogger(__name__)


async def run_anchor_estimation(
    *,
    image_path: str,
    atlas_name: str,
    on_progress: object | None = None,
    debug_dir: str | None = None,
) -> APResult:
    """Run full AP estimation + nano-banana refinement for an anchor slice.

    Stage A: multi-turn tool-use estimation (coarse).
    Stage B: nano-banana fine pass centered on Stage A result.
    """
    image = Image.open(image_path).convert("RGB")
    image = normalize_image(image)

    # Stage A: coarse estimation
    coarse = await asyncio.to_thread(
        estimate_position,
        image,
        atlas_name,
        on_progress=on_progress,
        debug_dir=debug_dir,
    )
    logger.info("Anchor coarse: %.3fmm (%s)", coarse.position_mm, image_path)

    # Stage B: nano-banana fine pass centered on coarse result
    fine = await asyncio.to_thread(
        estimate_position_image_gen,
        image,
        atlas_name,
        on_progress=on_progress,
        debug_dir=debug_dir,
        send_individually=True,
        atlas_resolution=1024,
    )
    logger.info("Anchor fine: %.3fmm (%s)", fine.position_mm, image_path)

    return fine


async def run_refinement(
    *,
    image_path: str,
    atlas_name: str,
    window_lo: float,
    window_hi: float,
    window_center: float,
    n_images: int,
    on_progress: object | None = None,
    debug_dir: str | None = None,
) -> APResult | None:
    """Run nano-banana refinement for a single slice within a bounded window.

    Returns ``None`` if *n_images* is 0 (window too narrow, skip).
    """
    if n_images == 0:
        return None

    image = Image.open(image_path).convert("RGB")
    image = normalize_image(image)

    # TODO: pass window bounds to nano-banana to constrain atlas image range.
    # For now, use the standard nano-banana call.  The window parameters will
    # be wired into estimate_position_image_gen once the API is extended to
    # accept explicit position bounds.
    result = await asyncio.to_thread(
        estimate_position_image_gen,
        image,
        atlas_name,
        on_progress=on_progress,
        debug_dir=debug_dir,
        send_individually=True,
        atlas_resolution=1024,
    )

    # Clamp result to window bounds
    clamped_mm = max(window_lo, min(window_hi, result.position_mm))
    return APResult(
        position_mm=clamped_mm,
        reasoning=result.reasoning,
        debug_dir=result.debug_dir,
    )
```

- [ ] **Step 4: Run test to verify it passes**

Run: `python -m pytest tests/test_brain_agents.py -v`
Expected: all 3 tests PASS

- [ ] **Step 5: Commit**

```bash
git add langslice/brain/agents.py tests/test_brain_agents.py
git commit -m "feat(brain): async agent wrappers for anchor and refinement estimation"
```

---

## Task 10: Pipeline orchestration

**Files:**
- Create: `langslice/brain/pipeline.py`
- Test: `tests/test_brain_pipeline.py`

- [ ] **Step 1: Write the failing test**

Create `tests/test_brain_pipeline.py`:

```python
"""Tests for wave computation and pipeline orchestration."""

from langslice.brain.pipeline import compute_waves


def test_compute_waves_simple():
    """4 anchors among 12 slices, indices 0-11."""
    anchor_indices = {2, 5, 8, 11}
    n_slices = 12
    waves = compute_waves(n_slices, anchor_indices)
    # Wave 1: distance 1 from any anchor -> {1,3, 4,6, 7,9, 10}
    assert 1 in waves[0]
    assert 3 in waves[0]
    # All non-anchor indices should appear exactly once across all waves
    all_assigned = set()
    for wave in waves:
        for idx in wave:
            assert idx not in all_assigned, f"index {idx} in multiple waves"
            assert idx not in anchor_indices, f"anchor {idx} in wave"
            all_assigned.add(idx)
    expected = set(range(n_slices)) - anchor_indices
    assert all_assigned == expected


def test_compute_waves_all_anchors():
    """When every slice is an anchor, no waves needed."""
    waves = compute_waves(4, {0, 1, 2, 3})
    assert waves == []


def test_compute_waves_single_anchor():
    """Single anchor at midpoint, waves radiate outward."""
    waves = compute_waves(7, {3})
    # Wave 1: {2, 4}
    assert set(waves[0]) == {2, 4}
    # Wave 2: {1, 5}
    assert set(waves[1]) == {1, 5}
    # Wave 3: {0, 6}
    assert set(waves[2]) == {0, 6}


def test_compute_waves_adjacent_anchors():
    """Two adjacent anchors: only outer slices need waves."""
    waves = compute_waves(5, {2, 3})
    all_in_waves = set()
    for w in waves:
        all_in_waves.update(w)
    assert all_in_waves == {0, 1, 4}
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m pytest tests/test_brain_pipeline.py -v`
Expected: FAIL (module not found)

- [ ] **Step 3: Implement pipeline.py (wave computation)**

Create `langslice/brain/pipeline.py`:

```python
"""Wave computation and pipeline orchestration for whole-brain estimation."""

from __future__ import annotations

import asyncio
import json
import logging
import os
from typing import Callable

from PIL import Image

from langslice.ai.estimator import APResult
from langslice.atlas.core import get_position_range_mm, load_atlas
from langslice.brain.agents import run_anchor_estimation, run_refinement
from langslice.brain.anchor_selection import select_anchor_indices
from langslice.brain.checkpoint import load_checkpoint, save_checkpoint
from langslice.brain.constraints import enforce_constraints
from langslice.brain.discovery import discover_slices
from langslice.brain.interpolation import interpolate_positions
from langslice.brain.types import (
    BrainEstimationConfig,
    BrainEstimationResult,
    BrainEstimationSummary,
    SlicePosition,
)
from langslice.brain.window import compute_refinement_window

logger = logging.getLogger(__name__)


def compute_waves(n_slices: int, anchor_indices: set[int]) -> list[list[int]]:
    """Compute refinement waves radiating outward from anchors.

    Returns a list of waves, where each wave is a list of slice indices that
    can be processed in parallel.  Slices at distance 1 from any anchor are
    in wave 0, distance 2 in wave 1, etc.
    """
    remaining = set(range(n_slices)) - anchor_indices
    locked = set(anchor_indices)
    waves: list[list[int]] = []

    while remaining:
        wave: list[int] = []
        for idx in sorted(remaining):
            if (idx - 1) in locked or (idx + 1) in locked:
                wave.append(idx)
        if not wave:
            # Unreachable if anchors exist, but safety fallback
            wave = sorted(remaining)
        for idx in wave:
            remaining.discard(idx)
        locked.update(wave)
        waves.append(wave)

    return waves


async def run_brain_estimation(
    config: BrainEstimationConfig,
    *,
    checkpoint_path: str | None = None,
    on_progress: Callable[[str], None] | None = None,
) -> BrainEstimationResult:
    """Main entry point: run the full whole-brain AP estimation pipeline.

    Phases:
      1. Parallel anchor estimation (coarse + nano-banana).
      2. Deterministic interpolation.
      3. Wave-based nano-banana refinement (optional).
      4. Constraint enforcement.
    """

    def _progress(msg: str) -> None:
        if on_progress:
            on_progress(msg)
        logger.info(msg)

    # --- Discover images ---
    image_paths = discover_slices(config.image_folder)
    n_slices = len(image_paths)
    if n_slices == 0:
        raise ValueError(f"No images found in {config.image_folder}")
    _progress(f"Found {n_slices} slices")

    # --- Load atlas for bounds ---
    atlas = load_atlas(config.atlas_name)
    atlas_range = get_position_range_mm(atlas)

    # --- Check for checkpoint ---
    cp_path = checkpoint_path or os.path.join(config.image_folder, "brain_estimate.json")
    existing = load_checkpoint(cp_path)
    existing_locked = {s.filename: s for s in existing if s.locked}

    # --- Build initial slice list ---
    slices = [
        existing_locked.get(
            os.path.basename(p),
            SlicePosition(os.path.basename(p), i, 0.0, "", locked=False),
        )
        for i, p in enumerate(image_paths)
    ]

    # --- Phase 1: Anchor estimation ---
    anchor_indices = select_anchor_indices(n_slices, config.n_anchors)
    anchor_set = set(anchor_indices)

    # Skip anchors that are already locked from a checkpoint
    anchors_to_run = [
        i for i in anchor_indices if not slices[i].locked
    ]

    if anchors_to_run:
        _progress(f"Phase 1: estimating {len(anchors_to_run)} anchors")
        sem = asyncio.Semaphore(config.max_parallel)

        async def _run_anchor(idx: int) -> tuple[int, APResult]:
            async with sem:
                _progress(f"  Anchor slice {idx} ({os.path.basename(image_paths[idx])})")
                result = await run_anchor_estimation(
                    image_path=image_paths[idx],
                    atlas_name=config.atlas_name,
                )
                return idx, result

        tasks = [_run_anchor(i) for i in anchors_to_run]
        results = await asyncio.gather(*tasks, return_exceptions=True)

        for r in results:
            if isinstance(r, Exception):
                raise RuntimeError(f"Anchor estimation failed: {r}") from r
            idx, ap_result = r
            slices[idx] = SlicePosition(
                slices[idx].filename,
                idx,
                ap_result.position_mm,
                "anchor",
                locked=True,
            )

        # Sanity check: monotonic order
        anchor_positions = [(i, slices[i].position_mm) for i in anchor_indices]
        _validate_anchor_order(anchor_positions, config.z_axis)

        save_checkpoint(cp_path, config, slices)
        _progress("Phase 1 complete, checkpoint saved")

    # --- Phase 2: Interpolation ---
    _progress("Phase 2: interpolating positions")
    slices = interpolate_positions(
        slices,
        interval_mm=config.interval_mm,
        atlas_range=atlas_range,
        z_axis=config.z_axis,
    )
    save_checkpoint(cp_path, config, slices)

    # --- Phase 3: Nano-banana refinement ---
    if config.refinement:
        waves = compute_waves(n_slices, anchor_set)
        _progress(f"Phase 3: {len(waves)} refinement waves")
        sem = asyncio.Semaphore(config.max_parallel)
        n_refined = 0
        n_skipped = 0

        for wave_num, wave in enumerate(waves):
            _progress(f"  Wave {wave_num + 1}/{len(waves)}: {len(wave)} slices")

            async def _run_refine(idx: int) -> tuple[int, APResult | None]:
                async with sem:
                    left_locked = _find_locked_neighbor(slices, idx, direction=-1)
                    right_locked = _find_locked_neighbor(slices, idx, direction=1)
                    win = compute_refinement_window(
                        position_mm=slices[idx].position_mm,
                        left_locked_mm=left_locked,
                        right_locked_mm=right_locked,
                        thickness_mm=config.thickness_mm,
                        interval_mm=config.interval_mm,
                    )
                    if win.skip:
                        return idx, None
                    return idx, await run_refinement(
                        image_path=image_paths[idx],
                        atlas_name=config.atlas_name,
                        window_lo=win.lo,
                        window_hi=win.hi,
                        window_center=win.center,
                        n_images=win.n_images,
                    )

            tasks = [_run_refine(i) for i in wave]
            results = await asyncio.gather(*tasks, return_exceptions=True)

            for r in results:
                if isinstance(r, Exception):
                    logger.warning("Refinement failed for a slice: %s", r)
                    continue
                idx, ap_result = r
                if ap_result is None:
                    slices[idx] = SlicePosition(
                        slices[idx].filename,
                        idx,
                        slices[idx].position_mm,
                        slices[idx].source,
                        locked=True,
                    )
                    n_skipped += 1
                else:
                    slices[idx] = SlicePosition(
                        slices[idx].filename,
                        idx,
                        ap_result.position_mm,
                        slices[idx].source + "+refined",
                        locked=True,
                    )
                    n_refined += 1

            save_checkpoint(cp_path, config, slices)

        _progress(f"Phase 3 complete: {n_refined} refined, {n_skipped} skipped")
    else:
        n_refined = 0
        n_skipped = n_slices - len(anchor_indices)

    # --- Phase 4: Constraint enforcement ---
    _progress("Phase 4: enforcing constraints")
    slices = enforce_constraints(
        slices,
        ordering=config.ordering,
        thickness_mm=config.thickness_mm,
        z_axis=config.z_axis,
    )
    save_checkpoint(cp_path, config, slices)

    # --- Summary ---
    positions = [s.position_mm for s in slices]
    intervals = [abs(positions[i + 1] - positions[i]) for i in range(len(positions) - 1)]
    import statistics

    summary = BrainEstimationSummary(
        mean_interval_mm=statistics.mean(intervals) if intervals else 0.0,
        std_interval_mm=statistics.stdev(intervals) if len(intervals) > 1 else 0.0,
        n_slices=n_slices,
        n_anchors=len(anchor_indices),
        n_refined=n_refined,
        n_skipped=n_skipped,
    )

    return BrainEstimationResult(config=config, slices=slices, summary=summary)


def _validate_anchor_order(
    anchors: list[tuple[int, float]], z_axis: str
) -> None:
    """Raise if anchors are not in expected monotonic order."""
    if len(anchors) < 2:
        return
    for i in range(len(anchors) - 1):
        idx_a, pos_a = anchors[i]
        idx_b, pos_b = anchors[i + 1]
        if z_axis == "AP" and pos_b <= pos_a:
            raise ValueError(
                f"Anchor at slice {idx_b} ({pos_b:.3f}mm) is not posterior to "
                f"slice {idx_a} ({pos_a:.3f}mm). Check images or re-run."
            )
        if z_axis == "PA" and pos_b >= pos_a:
            raise ValueError(
                f"Anchor at slice {idx_b} ({pos_b:.3f}mm) is not anterior to "
                f"slice {idx_a} ({pos_a:.3f}mm). Check images or re-run."
            )


def _find_locked_neighbor(
    slices: list[SlicePosition], idx: int, direction: int
) -> float | None:
    """Walk in *direction* (-1 or +1) from *idx* to find nearest locked slice."""
    i = idx + direction
    while 0 <= i < len(slices):
        if slices[i].locked:
            return slices[i].position_mm
        i += direction
    return None
```

- [ ] **Step 4: Run test to verify it passes**

Run: `python -m pytest tests/test_brain_pipeline.py -v`
Expected: all 4 tests PASS

- [ ] **Step 5: Run full test suite**

Run: `python -m pytest tests/ -v`
Expected: all tests PASS (including existing tests)

- [ ] **Step 6: Commit**

```bash
git add langslice/brain/pipeline.py tests/test_brain_pipeline.py
git commit -m "feat(brain): pipeline orchestration with wave computation"
```

---

## Task 11: CLI integration

**Files:**
- Modify: `langslice/cli.py`

- [ ] **Step 1: Read the current cli.py to identify insertion point**

Read `langslice/cli.py` to find where subcommands are added (look for `_add_estimate_parser` and how `subparsers` is created).

- [ ] **Step 2: Add estimate-brain subcommand**

Add a new function `_add_estimate_brain_parser` after the existing `_add_estimate_parser` function:

```python
def _add_estimate_brain_parser(subparsers: argparse._SubParsersAction) -> None:
    p = subparsers.add_parser(
        "estimate-brain",
        help="Estimate AP positions for a folder of brain slices",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("image_folder", help="Folder containing slice images")
    p.add_argument("--atlas", default="allen_mouse_25um", help="BrainGlobe atlas name")
    p.add_argument("--thickness", type=int, default=50, help="Slice thickness in microns")
    p.add_argument("--interval", type=int, default=200, help="Average slice interval in microns")
    p.add_argument("--anchors", type=int, default=4, help="Number of anchor agents")
    p.add_argument(
        "--ordering",
        choices=["strict", "loose", "none"],
        default="strict",
        help="Ordering enforcement mode",
    )
    p.add_argument(
        "--refinement",
        choices=["on", "off"],
        default="on",
        help="Enable nano-banana refinement",
    )
    p.add_argument("--parallel", type=int, default=4, help="Max concurrent Gemini calls")
    p.add_argument(
        "--z-axis",
        choices=["AP", "PA"],
        default="AP",
        help="Z-axis orientation of the slice series",
    )
    p.add_argument("--out", help="Output JSON path (default: <folder>/brain_estimate.json)")
    p.set_defaults(func=_run_estimate_brain)
```

- [ ] **Step 3: Add the handler function**

Add `_run_estimate_brain` function:

```python
def _run_estimate_brain(args: argparse.Namespace) -> None:
    import asyncio

    from langslice.brain.pipeline import run_brain_estimation
    from langslice.brain.types import BrainEstimationConfig

    config = BrainEstimationConfig(
        image_folder=args.image_folder,
        atlas_name=args.atlas,
        thickness_um=args.thickness,
        interval_um=args.interval,
        n_anchors=args.anchors,
        ordering=args.ordering,
        refinement=args.refinement == "on",
        max_parallel=args.parallel,
        z_axis=args.z_axis,
    )

    def on_progress(msg: str) -> None:
        print(msg)

    result = asyncio.run(
        run_brain_estimation(
            config,
            checkpoint_path=args.out,
            on_progress=on_progress,
        )
    )

    out_path = args.out or os.path.join(args.image_folder, "brain_estimate.json")
    import json

    with open(out_path, "w") as f:
        json.dump(result.to_dict(), f, indent=2)

    print(f"\nResults saved to {out_path}")
    print(f"  {result.summary.n_slices} slices, {result.summary.n_anchors} anchors")
    print(f"  Mean interval: {result.summary.mean_interval_mm:.3f}mm")
    print(f"  Std interval:  {result.summary.std_interval_mm:.3f}mm")
```

- [ ] **Step 4: Wire the subparser into main()**

In the `main()` function, add after the existing `_add_estimate_parser(subparsers)` call:

```python
_add_estimate_brain_parser(subparsers)
```

- [ ] **Step 5: Add cost estimation display**

In `_run_estimate_brain`, before calling `run_brain_estimation`, print a plan summary:

```python
    n_images = len(discover_slices(config.image_folder))
    n_refinements = n_images - config.n_anchors if config.refinement else 0
    print(f"\nBrain estimation plan:")
    print(f"  {n_images} slices, {config.n_anchors} anchors, {config.ordering} ordering")
    print(f"  Refinement: {'ON' if config.refinement else 'OFF'}")
    print(f"")
    print(f"  Phase 1:  {config.n_anchors} anchor estimations          ~${config.n_anchors * 0.05:.2f}")
    print(f"  Phase 1b: {config.n_anchors} anchor nano-banana passes   cost TBD")
    if config.refinement:
        print(f"  Phase 3:  {n_refinements} nano-banana refinements    cost TBD")
    print(f"  --parallel {config.max_parallel}")
    print()
```

This requires importing `discover_slices` at the top of the handler.

- [ ] **Step 6: Verify CLI parses correctly**

Run:
```bash
langslice estimate-brain --help
```
Expected: help text showing all arguments

- [ ] **Step 6: Update __init__.py exports**

Update `langslice/brain/__init__.py` to also export `run_brain_estimation`:

```python
"""Whole-brain multi-slice AP estimation."""

from langslice.brain.pipeline import run_brain_estimation
from langslice.brain.types import (
    BrainEstimationConfig,
    BrainEstimationResult,
    BrainEstimationSummary,
    SlicePosition,
)

__all__ = [
    "run_brain_estimation",
    "BrainEstimationConfig",
    "BrainEstimationResult",
    "BrainEstimationSummary",
    "SlicePosition",
]
```

- [ ] **Step 7: Run linting and type checking**

Run:
```bash
python -m ruff check langslice/brain/ langslice/cli.py
python -m basedpyright
```
Expected: clean (or fix any issues)

- [ ] **Step 8: Commit**

```bash
git add langslice/cli.py langslice/brain/__init__.py
git commit -m "feat: add estimate-brain CLI subcommand for whole-brain AP estimation"
```

---

## Task 12: Update ruff/pyright config and documentation

**Files:**
- Modify: `pyproject.toml`
- Modify: `REPO_MAP.md`
- Modify: `docs/architecture_overview.md`

- [ ] **Step 1: Update ruff include to cover brain module**

In `pyproject.toml`, verify that the ruff `include` pattern already covers `langslice/**/*.py`. If not, add `langslice/brain/**/*.py`.

- [ ] **Step 2: Update REPO_MAP.md**

Add `langslice/brain/` to the active code section:

```markdown
- **`langslice/brain/`** — Whole-brain multi-slice AP estimation. Anchor selection, interval interpolation, wave-based nano-banana refinement, constraint enforcement, checkpoint I/O, and ADK pipeline orchestration.
```

- [ ] **Step 3: Update architecture_overview.md**

Add a section for the brain module describing the four-phase pipeline and how it composes the existing estimator modules.

- [ ] **Step 4: Run full verification**

Run:
```bash
python -m pytest
python -m ruff check .
python -m basedpyright
```
Expected: all pass

- [ ] **Step 5: Commit**

```bash
git add pyproject.toml REPO_MAP.md docs/architecture_overview.md
git commit -m "docs: add brain module to repo map and architecture overview"
```
