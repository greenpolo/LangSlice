# ruff: noqa: E402
"""Unit tests for iSFT/synth_corpus.py — synthetic terminal-trace corpus generator.

Tests run entirely without Docker, GPU, SigLIP, or real atlas files. Fake
images are written into tmp_path so _validate_row's image-path checks pass.
"""

from __future__ import annotations

import json
import random
import sys
import types
from pathlib import Path
from typing import Any

import pytest
from PIL import Image

# Make the training package importable.
_REPO = Path(__file__).resolve().parents[1]
_TRAINING = _REPO / "models" / "langslice-gemma-4" / "training"
for _p in (str(_REPO), str(_REPO / "src"), str(_TRAINING)):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from iSFT.synth_corpus import (
    _FALLBACK_REASONING,
    SectionSpec,
    _cap_canonical_tool_step_positions,
    _fallback_grid,
    generate_synthetic_rows,
    weighted_sample_without_replacement,
    write_synthetic_jsonl,
)  # noqa: E402
from sft.dataset import DatasetValidationError, _validate_row  # noqa: E402

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _tiny_grid(center: float = 5.0, span: float = 6.0, step: float = 0.5) -> list[float]:
    """Return a small but non-trivial sorted grid centred around ``center``."""
    lo = max(0.0, center - span / 2.0)
    hi = center + span / 2.0
    n = max(4, round((hi - lo) / step) + 1)
    return [round(lo + i * (hi - lo) / (n - 1), 4) for i in range(n)]


def _make_spec(
    *,
    subject_id: str = "sub01",
    section_id: str = "sec001",
    gt_mm: float = 5.0,
    plane: str = "coronal",
    atlas_name: str = "allen_mouse_25um",
    atlas_version: str = "CCFv3",
    image_path: str = "queries/test_img.jpg",
    dataset: str = "test_dataset",
) -> SectionSpec:
    return SectionSpec(
        plane=plane,
        dataset=dataset,
        subject_id=subject_id,
        section_id=section_id,
        image_path=image_path,
        ground_truth_mm=gt_mm,
        atlas_name=atlas_name,
        atlas_version=atlas_version,
    )


def _write_tiny_png(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.new("RGB", (8, 8), (100, 150, 200)).save(path, format="PNG")


def _stage_images_for_row(row: dict[str, Any], root: Path) -> None:
    """Create stub PNG files for every image path referenced in a JSONL row."""
    for rel in row["query_image_paths"]:
        _write_tiny_png(root / rel)
    for step in row["trace"][:-1]:
        for rel in step["tool_result"]["image_paths"]:
            _write_tiny_png(root / rel)


def _make_specs_with_unique_ids(n: int, gt_mm: float = 5.0) -> list[SectionSpec]:
    return [
        _make_spec(
            subject_id=f"sub{i:03d}",
            section_id=f"sec{i:03d}",
            gt_mm=gt_mm,
        )
        for i in range(n)
    ]


# ---------------------------------------------------------------------------
# Monkeypatch fixture: stub generate_trace to avoid atlas grid cache on disk
# ---------------------------------------------------------------------------

class _FakeToolStep:
    """Minimal duck-type for langslice_traces.schema.ToolStep."""
    def __init__(self, positions: list[float], atlas_name: str, plane: str) -> None:
        self.call_name = "fetch_atlas"
        self.call_args = {"positions_mm": positions}
        image_paths = [
            f"models/langslice-gemma-4/data/atlas/{atlas_name}/{plane}/{p:.2f}mm.jpg"
            for p in positions
        ]
        self.result_image_paths = image_paths
        n = len(positions)
        noun = "atlas section" if n == 1 else "atlas sections"
        formatted = ", ".join(f"{p:.2f} mm" for p in positions)
        self.result_text = f"Fetched {n} {noun}: {formatted}"


class _FakeCanonicalTrace:
    """Minimal duck-type for langslice_traces.schema.CanonicalTrace."""
    def __init__(
        self,
        *,
        image_path: str,
        ground_truth_mm: float,
        plane: str,
        atlas_name: str,
        atlas_version: str,
        subject_id: str,
        grid: list[float],
        rng: random.Random,
    ) -> None:
        self.atlas_name = atlas_name
        self.atlas_version = atlas_version
        self.plane = plane
        self.subject_id = subject_id
        self.system_prompt_kind = "single_slice"
        self.bucket = 1
        self.query_image_paths = [image_path]
        axis = {"coronal": "AP", "sagittal": "ML", "horizontal": "DV"}.get(plane, "AP")
        self.user_prompt_text = (
            f"Determine this {plane} slice's {axis} position in the {atlas_name} atlas."
        )
        # Build 1 synthetic tool step (2 positions bracketing GT).
        lo = max(grid[0], ground_truth_mm - 0.5)
        hi = min(grid[-1], ground_truth_mm + 0.5)
        positions = sorted({round(lo, 2), round(hi, 2)})
        if len(positions) < 2:
            positions = [round(grid[0], 2), round(grid[-1], 2)]
        self.tool_steps = [_FakeToolStep(positions, atlas_name, plane)]
        self.final_answer = None
        self.quality: dict = {}
        self.gemini_reasoning = None
        self.dataset_root = None


class _FakeCanonicalTraceTwoSteps(_FakeCanonicalTrace):
    """Same as _FakeCanonicalTrace, but with two fetch turns."""

    def __init__(
        self,
        *,
        image_path: str,
        ground_truth_mm: float,
        plane: str,
        atlas_name: str,
        atlas_version: str,
        subject_id: str,
        grid: list[float],
        rng: random.Random,
    ) -> None:
        super().__init__(
            image_path=image_path,
            ground_truth_mm=ground_truth_mm,
            plane=plane,
            atlas_name=atlas_name,
            atlas_version=atlas_version,
            subject_id=subject_id,
            grid=grid,
            rng=rng,
        )
        lo = max(grid[0], ground_truth_mm - 0.5)
        mid = round(ground_truth_mm, 2)
        hi = min(grid[-1], ground_truth_mm + 0.5)
        self.tool_steps = [
            _FakeToolStep([round(lo, 2)], atlas_name, plane),
            _FakeToolStep([mid, round(hi, 2)], atlas_name, plane),
        ]


@pytest.fixture
def patched_generate_trace(monkeypatch: pytest.MonkeyPatch):
    """Replace generate_trace with a stub that returns _FakeCanonicalTrace.

    Also stubs load_atlas_grid so tests don't need real .pt files.
    """
    def fake_generate_trace(
        *,
        image_path: str,
        ground_truth_mm: float,
        plane: str,
        atlas_name: str,
        atlas_version: str,
        subject_id: str,
        grid: list[float],
        strategy: str,
        rng: random.Random,
        bucket: int = 1,
    ) -> _FakeCanonicalTrace:
        _ = strategy, bucket
        return _FakeCanonicalTrace(
            image_path=image_path,
            ground_truth_mm=ground_truth_mm,
            plane=plane,
            atlas_name=atlas_name,
            atlas_version=atlas_version,
            subject_id=subject_id,
            grid=grid,
            rng=rng,
        )

    # Patch inside synth_corpus's imported name space.
    import langslice_traces.generator as gen_mod
    monkeypatch.setattr(gen_mod, "generate_trace", fake_generate_trace)

    # Also patch the FinalAnswer import so the injected answer is a proper
    # dataclass (uses the real class — just ensures the import path works).
    yield


@pytest.fixture
def patched_generate_trace_two_steps(monkeypatch: pytest.MonkeyPatch):
    """Patch generate_trace to synthesize two fetch turns per canonical trace."""
    def fake_generate_trace(
        *,
        image_path: str,
        ground_truth_mm: float,
        plane: str,
        atlas_name: str,
        atlas_version: str,
        subject_id: str,
        grid: list[float],
        strategy: str,
        rng: random.Random,
        bucket: int = 1,
    ) -> _FakeCanonicalTraceTwoSteps:
        _ = strategy, bucket
        return _FakeCanonicalTraceTwoSteps(
            image_path=image_path,
            ground_truth_mm=ground_truth_mm,
            plane=plane,
            atlas_name=atlas_name,
            atlas_version=atlas_version,
            subject_id=subject_id,
            grid=grid,
            rng=rng,
        )

    import langslice_traces.generator as gen_mod
    monkeypatch.setattr(gen_mod, "generate_trace", fake_generate_trace)
    yield


@pytest.fixture
def fake_visible_clues_module(monkeypatch: pytest.MonkeyPatch):
    """Install a deterministic iSFT.atlas_signature.compose_visible_clues."""

    def compose_visible_clues(
        atlas_name: str,
        plane: str,
        position_mm: float,
        *,
        comparison_positions: list[float] | None = None,
        top_k: int = 8,
    ) -> str:
        _ = top_k
        if comparison_positions:
            comps = ",".join(f"{p:.2f}" for p in comparison_positions)
            return (
                f"Visible clues: target={atlas_name}/{plane}@{position_mm:.2f}; "
                f"compare={comps}"
            )
        return f"Visible clues: target={atlas_name}/{plane}@{position_mm:.2f}"

    mod = types.ModuleType("iSFT.atlas_signature")
    mod.compose_visible_clues = compose_visible_clues
    monkeypatch.setitem(sys.modules, "iSFT.atlas_signature", mod)
    yield


# ---------------------------------------------------------------------------
# Test A: emits valid JSONL rows
# ---------------------------------------------------------------------------

def test_generate_synthetic_rows_emits_valid_jsonl(
    patched_generate_trace,  # noqa: ANN001, ARG001
    tmp_path: Path,
) -> None:
    """generate_synthetic_rows + write_synthetic_jsonl → rows pass _validate_row."""
    rng = random.Random(42)
    grid = _tiny_grid(center=5.0)
    specs = _make_specs_with_unique_ids(5, gt_mm=5.0)

    rows = generate_synthetic_rows(specs, multi_turn_floor=0.5, rng=rng, grid=grid)
    assert len(rows) == 5, "one row per spec"

    jsonl_path = tmp_path / "synth.jsonl"
    write_synthetic_jsonl(rows, jsonl_path)
    assert jsonl_path.is_file()

    # Stage stub images so _validate_row's file-existence checks pass.
    root = tmp_path
    for row in rows:
        _stage_images_for_row(row, root)

    # Validate every row through the official validator.
    with jsonl_path.open("r", encoding="utf-8") as fh:
        for lineno, raw in enumerate(fh, start=1):
            stripped = raw.strip()
            if not stripped:
                continue
            row_dict = json.loads(stripped)
            try:
                _validate_row(row_dict, lineno, root)
            except DatasetValidationError as exc:
                pytest.fail(f"line {lineno}: _validate_row rejected: {exc}\nrow={row_dict}")


# ---------------------------------------------------------------------------
# Test B: multi_turn_floor proportion
# ---------------------------------------------------------------------------

def test_multi_turn_floor_proportion(
    patched_generate_trace,  # noqa: ANN001, ARG001
) -> None:
    """With multi_turn_floor=0.5 and 100 specs, ~50 rows should be multi-turn."""
    rng = random.Random(0)
    grid = _tiny_grid(center=5.0)
    specs = _make_specs_with_unique_ids(100, gt_mm=5.0)

    rows = generate_synthetic_rows(specs, multi_turn_floor=0.5, rng=rng, grid=grid)
    assert len(rows) == 100

    n_multi = sum(
        1 for r in rows
        # Multi-turn rows have at least one tool_call step before the submit.
        if len(r["trace"]) > 1
    )
    n_answer_only = 100 - n_multi

    # Allow ±15 tolerance around the expected ~50/50 split.
    assert abs(n_multi - 50) <= 15, (
        f"expected ~50 multi-turn rows, got {n_multi} (answer_only={n_answer_only})"
    )
    assert abs(n_answer_only - 50) <= 15, (
        f"expected ~50 answer-only rows, got {n_answer_only}"
    )


# ---------------------------------------------------------------------------
# Test C: injected final answer has non-empty reasoning
# ---------------------------------------------------------------------------

def test_injected_final_answer_has_nonempty_reasoning(
    patched_generate_trace,  # noqa: ANN001, ARG001
) -> None:
    """Every generated row's submit step must have non-empty reasoning."""
    rng = random.Random(7)
    grid = _tiny_grid(center=5.0)
    specs = _make_specs_with_unique_ids(10, gt_mm=5.0)

    rows = generate_synthetic_rows(specs, multi_turn_floor=1.0, rng=rng, grid=grid)

    for i, row in enumerate(rows):
        submit_step = row["trace"][-1]
        assert "submit" in submit_step, f"row {i}: last step lacks 'submit'"
        args = submit_step["submit"]["args"]
        reasoning = args.get("reasoning", "")
        assert isinstance(reasoning, str) and reasoning.strip(), (
            f"row {i}: reasoning is empty — got {reasoning!r}"
        )
        # Reasoning is either the brainglobe-derived "Target ..." form (when
        # atlas is loadable in the test env) or the canned fallback.
        assert reasoning == _FALLBACK_REASONING or reasoning.startswith(
            "Target "
        ), f"row {i}: unexpected reasoning prefix {reasoning[:80]!r}"
        # position_mm should equal GT.
        assert isinstance(args.get("position_mm"), (int, float)), (
            f"row {i}: position_mm missing or non-numeric"
        )


# ---------------------------------------------------------------------------
# Test D: --synthetic-per-round 0 does not generate rows (byte-identical path)
# ---------------------------------------------------------------------------

def test_synthetic_disabled_byte_identical(tmp_path: Path) -> None:
    """When --synthetic-per-round=0, _phase_synthetic is never called.

    We verify this by calling _parse_args with the new flags and confirming:
    (a) the default value is 0, and
    (b) with 0 specs + 0 per_round, generate_synthetic_rows returns empty list.
    """
    # (a) Default arg value.
    import iSFT.iterate as iterate_mod  # noqa: PLC0415

    args = iterate_mod._parse_args([
        "--base-checkpoint", str(tmp_path / "base"),
        "--base-corpus", str(tmp_path / "corpus.jsonl"),
        "--iterative-corpus-dir", str(tmp_path / "iter"),
        "--allocation-root", str(tmp_path / "alloc"),
        "--output-dir", str(tmp_path / "out"),
    ])
    assert args.synthetic_per_round == 0, (
        f"default --synthetic-per-round should be 0, got {args.synthetic_per_round}"
    )
    assert args.multi_turn_floor == 0.10, (
        f"default --multi-turn-floor should be 0.10, got {args.multi_turn_floor}"
    )
    assert args.synthetic_reasoning_mode == "region_dump"

    # (b) With zero specs, generate_synthetic_rows returns empty list
    #     (regardless of multi_turn_floor).
    rng = random.Random(99)
    rows = generate_synthetic_rows([], multi_turn_floor=0.25, rng=rng)
    assert rows == [], "empty specs → empty rows"


# ---------------------------------------------------------------------------
# Test E: _fallback_grid always contains GT
# ---------------------------------------------------------------------------

def test_fallback_grid_contains_gt() -> None:
    """_fallback_grid should always produce a grid that brackets the GT."""
    for gt in (0.0, 1.5, 5.23, 9.99):
        g = _fallback_grid(gt)
        assert len(g) >= 2, f"grid too short for gt={gt}"
        assert min(g) <= gt <= max(g), (
            f"GT {gt} not bracketed by fallback grid {g[:3]}…{g[-3:]}"
        )
        # Grid should be sorted.
        assert g == sorted(g), "fallback grid not sorted"


# ---------------------------------------------------------------------------
# Test F: answer-only rows have exactly one trace step (the submit)
# ---------------------------------------------------------------------------

def test_answer_only_rows_have_single_trace_step(
    patched_generate_trace,  # noqa: ANN001, ARG001
) -> None:
    """Answer-only rows (multi_turn_floor=0.0) must have trace=[submit_step]."""
    rng = random.Random(11)
    grid = _tiny_grid(center=5.0)
    specs = _make_specs_with_unique_ids(10, gt_mm=5.0)

    rows = generate_synthetic_rows(specs, multi_turn_floor=0.0, rng=rng, grid=grid)
    for i, row in enumerate(rows):
        trace = row["trace"]
        assert len(trace) == 1, (
            f"row {i}: answer-only should have trace=[submit], got {len(trace)} steps"
        )
        assert "submit" in trace[0], f"row {i}: single step must be a submit step"


# ---------------------------------------------------------------------------
# Test G: multi-turn rows have ≥1 tool_call step before submit
# ---------------------------------------------------------------------------

def test_multi_turn_rows_have_tool_steps(
    patched_generate_trace,  # noqa: ANN001, ARG001
) -> None:
    """Multi-turn rows (multi_turn_floor=1.0) must have ≥1 tool_call steps."""
    rng = random.Random(22)
    grid = _tiny_grid(center=5.0)
    specs = _make_specs_with_unique_ids(10, gt_mm=5.0)

    rows = generate_synthetic_rows(specs, multi_turn_floor=1.0, rng=rng, grid=grid)
    for i, row in enumerate(rows):
        trace = row["trace"]
        assert len(trace) >= 2, (
            f"row {i}: multi-turn should have ≥2 trace steps (tool_call + submit)"
        )
        assert "tool_call" in trace[0], (
            f"row {i}: first step must be tool_call in multi-turn mode"
        )
        assert "submit" in trace[-1], (
            f"row {i}: last step must be submit in multi-turn mode"
        )


# ---------------------------------------------------------------------------
# Test H: write_synthetic_jsonl appends (does not overwrite)
# ---------------------------------------------------------------------------

def test_write_synthetic_jsonl_appends(
    patched_generate_trace,  # noqa: ANN001, ARG001
    tmp_path: Path,
) -> None:
    """write_synthetic_jsonl appends — calling it twice yields 2× rows."""
    rng = random.Random(33)
    grid = _tiny_grid(center=5.0)
    specs = _make_specs_with_unique_ids(3, gt_mm=5.0)
    rows = generate_synthetic_rows(specs, multi_turn_floor=0.0, rng=rng, grid=grid)

    out = tmp_path / "synth.jsonl"
    write_synthetic_jsonl(rows, out)
    write_synthetic_jsonl(rows, out)  # append again

    lines = [line for line in out.read_text(encoding="utf-8").splitlines() if line.strip()]
    assert len(lines) == 6, f"expected 6 lines after two appends, got {len(lines)}"


# ---------------------------------------------------------------------------
# Test I: weighted_sample_without_replacement helper — bias + no duplicates
# ---------------------------------------------------------------------------

def test_weighted_sample_returns_full_pool_when_n_at_least_size() -> None:
    """n >= len(specs) returns the pool as-is (no surprise duplicates)."""
    specs = _make_specs_with_unique_ids(5)
    rng = random.Random(0)
    out = weighted_sample_without_replacement(specs, n=10, rng=rng, section_weights=None)
    assert {s.section_id for s in out} == {s.section_id for s in specs}


def test_weighted_sample_uniform_when_weights_none() -> None:
    """``section_weights=None`` falls back to ``rng.sample`` and picks ``n`` distinct."""
    specs = _make_specs_with_unique_ids(20)
    rng = random.Random(1)
    out = weighted_sample_without_replacement(specs, n=8, rng=rng, section_weights=None)
    sids = [s.section_id for s in out]
    assert len(out) == 8
    assert len(set(sids)) == 8, "weighted sample must not produce duplicates"


def test_weighted_sample_biases_toward_high_weights() -> None:
    """Sections with weight >> 1 should be picked vastly more often than weight 1.

    Sets up 100 specs, gives the first 10 weight 100 and the rest weight 1.
    Across 200 independent samples of size 10, the high-weight group should
    dominate.  A loose threshold (>=8/10 average picks from high-weight bin)
    leaves headroom for RNG variance while still failing if A-Res keying breaks.
    """
    specs = _make_specs_with_unique_ids(100)
    high_ids = {s.section_id for s in specs[:10]}
    weights = {s.section_id: (100.0 if s.section_id in high_ids else 1.0) for s in specs}
    rng = random.Random(2)

    high_hits = 0
    trials = 200
    for _ in range(trials):
        sample = weighted_sample_without_replacement(
            specs, n=10, rng=rng, section_weights=weights,
        )
        high_hits += sum(1 for s in sample if s.section_id in high_ids)

    avg = high_hits / trials
    assert avg >= 8.0, (
        f"expected >=8/10 high-weight picks per sample on average; got {avg:.2f}. "
        "A-Res keying may be broken."
    )


def test_weighted_sample_no_duplicates_with_weights() -> None:
    """Each sample must contain distinct section_ids even with weighted draws."""
    specs = _make_specs_with_unique_ids(30)
    weights = {specs[0].section_id: 1e6}  # one absurdly heavy item — still picked once
    rng = random.Random(3)
    out = weighted_sample_without_replacement(specs, n=10, rng=rng, section_weights=weights)
    sids = [s.section_id for s in out]
    assert len(sids) == len(set(sids)), "duplicates emitted by A-Res sampler"
    assert specs[0].section_id in sids, "heavy-weight item should be picked once"


# ---------------------------------------------------------------------------
# Test J: load_specs_from_allocation honors seen_ids + n=None
# ---------------------------------------------------------------------------

def test_load_specs_seen_ids_excludes(tmp_path: Path) -> None:
    """Sections present in ``seen_ids`` must be dropped from the returned pool.

    Uses minimal allocation + shard JSONLs (no atlas grid required since this
    function only reads on-disk JSONL metadata).  ``n=None`` returns the full
    eligible pool so the assertion is exact (no sampling noise).
    """
    from iSFT.synth_corpus import load_specs_from_allocation

    plane = "coronal"
    alloc_dir = tmp_path / "allocations" / plane
    shards_dir = tmp_path / "shards" / plane
    alloc_dir.mkdir(parents=True)
    shards_dir.mkdir(parents=True)

    # 4 sections in the allocation; 2 will be marked seen.
    section_ids = [f"sec{i:03d}" for i in range(4)]
    with (alloc_dir / "rlvr.jsonl").open("w", encoding="utf-8") as fh:
        for sid in section_ids:
            fh.write(json.dumps({"section_id": sid, "dataset": "ds"}) + "\n")
    # Shard rows — image_path doesn't need to exist on disk for spec construction.
    with (shards_dir / "ds.jsonl").open("w", encoding="utf-8") as fh:
        for sid in section_ids:
            fh.write(json.dumps({
                "section_id": sid,
                "subject_id": sid,
                "image_path": f"queries/{sid}.jpg",
                "position_mm": 5.0,
                "atlas": "allen_mouse_25um",
                "atlas_version": "CCFv3",
            }) + "\n")

    rng = random.Random(0)
    seen = {section_ids[0], section_ids[2]}
    specs = load_specs_from_allocation(
        tmp_path, plane, n=None, rng=rng, seen_ids=seen,
    )
    returned = {s.section_id for s in specs}
    assert returned == {section_ids[1], section_ids[3]}, (
        f"expected only unseen ids, got {returned}"
    )


# ---------------------------------------------------------------------------
# Tests K-N: _cap_canonical_tool_step_positions (atlas-image cap for synth rows)
# ---------------------------------------------------------------------------

def test_cap_canonical_tool_step_positions_noop_when_under_cap() -> None:
    """No change when total positions already fits the cap."""
    positions = [[1.0, 2.0, 3.0], [4.0, 5.0]]  # total = 5
    out = _cap_canonical_tool_step_positions(
        positions, max_total_images=8, gt_position_mm=3.0,
    )
    assert out == positions


def test_cap_canonical_tool_step_positions_drops_farthest_first() -> None:
    """Cap of 4 on a 6-image trace drops the 2 farthest-from-GT positions."""
    positions = [[1.0, 2.0, 9.0], [4.0, 5.0, 8.0]]  # total = 6, gt = 5.0
    # distances from 5: [4, 3, 4] step0, [1, 0, 3] step1
    # Farthest first: 9.0 (4), 1.0 (4), 8.0 (3), 2.0 (3), 4.0 (1), 5.0 (0)
    # Need to drop 2 → drop 9.0 and 1.0.
    out = _cap_canonical_tool_step_positions(
        positions, max_total_images=4, gt_position_mm=5.0,
    )
    assert sum(len(s) for s in out) == 4
    assert 9.0 not in out[0], f"9.0 (farthest) should be dropped; got {out[0]}"
    assert 1.0 not in out[0], f"1.0 (farthest) should be dropped; got {out[0]}"
    assert out[0] == [2.0]
    assert out[1] == [4.0, 5.0, 8.0]


def test_cap_canonical_tool_step_positions_preserves_step_existence() -> None:
    """Never drops the last surviving position in a step."""
    # Step 0 has 1 close + 1 far position; step 1 has 1 far position only.
    # If we naively dropped by distance, the single step-1 position would be
    # the farthest and would be dropped. The cap must refuse that drop.
    positions = [[5.0, 9.0], [10.0]]  # total = 3, gt = 5
    # distances: [0, 4] step0, [5] step1
    # Cap to 1 → must drop 2.  Farthest: 10.0 (5), 9.0 (4), 5.0 (0).
    # 10.0 is step1's only position → skip.
    # 9.0 is step0's farther position → drop. step0 now has [5.0].
    # 5.0 is step0's only remaining → skip.
    # Then 10.0 again (already considered).  We can only drop 1, so total stays at 2.
    out = _cap_canonical_tool_step_positions(
        positions, max_total_images=1, gt_position_mm=5.0,
    )
    assert len(out) == 2, "both steps must survive"
    assert all(len(s) >= 1 for s in out), f"every step must keep >=1 position; got {out}"


def test_generate_synthetic_rows_max_total_images_caps_multi_turn_only(
    patched_generate_trace,  # noqa: ANN001, ARG001
) -> None:
    """max_total_images caps multi-turn synth rows; answer-only unaffected."""
    rng = random.Random(33)
    grid = _tiny_grid(center=5.0)
    specs = _make_specs_with_unique_ids(20, gt_mm=5.0)

    # multi_turn_floor=1.0 → all multi-turn; the stub generates 2 positions
    # per row, so cap=1 should reduce every row to ≤1 atlas image total.
    # But _cap preserves step existence → with 1 step having 2 positions and
    # cap=1, we drop 1 (the farther), leaving 1 image.
    rows = generate_synthetic_rows(
        specs, multi_turn_floor=1.0, rng=rng, grid=grid, max_total_images=1,
    )
    for i, row in enumerate(rows):
        n_atlas = sum(
            len(s.get("tool_result", {}).get("image_paths") or [])
            for s in row["trace"] if "tool_call" in s
        )
        assert n_atlas <= 1, (
            f"row {i}: max_total_images=1 should cap atlas images at 1, got {n_atlas}"
        )

    # multi_turn_floor=0.0 + max_total_images=1 → all answer-only; no atlas
    # images at all (no tool_call steps), so cap is a no-op for them.
    rng2 = random.Random(34)
    rows2 = generate_synthetic_rows(
        _make_specs_with_unique_ids(10, gt_mm=5.0),
        multi_turn_floor=0.0, rng=rng2, grid=grid, max_total_images=1,
    )
    for i, row in enumerate(rows2):
        assert len(row["trace"]) == 1, f"row {i}: answer-only should be single submit step"
        assert "submit" in row["trace"][0]


# ---------------------------------------------------------------------------
# Tests O-R: synthetic reasoning modes + thinking-signature turn splitting
# ---------------------------------------------------------------------------


def test_region_dump_default_matches_explicit_mode(
    patched_generate_trace,  # noqa: ANN001, ARG001
) -> None:
    """Default mode must stay byte-compatible with explicit region_dump."""
    specs = _make_specs_with_unique_ids(4, gt_mm=5.0)
    grid = _tiny_grid(center=5.0)

    rows_default = generate_synthetic_rows(
        specs,
        multi_turn_floor=0.5,
        rng=random.Random(123),
        grid=grid,
    )
    rows_explicit = generate_synthetic_rows(
        specs,
        multi_turn_floor=0.5,
        rng=random.Random(123),
        grid=grid,
        synthetic_reasoning_mode="region_dump",
    )
    assert rows_default == rows_explicit


def test_thinking_signature_answer_only_row_shape(
    patched_generate_trace,  # noqa: ANN001, ARG001
    fake_visible_clues_module,  # noqa: ANN001, ARG001
) -> None:
    """Answer-only thinking mode emits one submit turn with sibling thinking."""
    spec = _make_spec(gt_mm=5.0)
    rows = generate_synthetic_rows(
        [spec],
        multi_turn_floor=0.0,
        rng=random.Random(9),
        grid=_tiny_grid(center=5.0),
        synthetic_reasoning_mode="thinking_signature",
    )

    assert len(rows) == 1
    trace = rows[0]["trace"]
    assert len(trace) == 1
    terminal = trace[0]
    assert "submit" in terminal
    assert "thinking" in terminal
    assert terminal["thinking"].startswith("Visible clues:")
    assert terminal["submit"]["args"] == {"position_mm": spec.ground_truth_mm}
    assert rows[0]["thinking_mode"] is True


def test_thinking_signature_multiturn_two_fetches_splits_into_three_rows(
    patched_generate_trace_two_steps,  # noqa: ANN001, ARG001
    fake_visible_clues_module,  # noqa: ANN001, ARG001
) -> None:
    """Two fetches + submit should become three turn-split rows."""
    spec = _make_spec(gt_mm=5.0)
    rows = generate_synthetic_rows(
        [spec],
        multi_turn_floor=1.0,
        rng=random.Random(10),
        grid=_tiny_grid(center=5.0),
        synthetic_reasoning_mode="thinking_signature",
    )
    assert len(rows) == 3
    assert all(row["thinking_mode"] is True for row in rows)

    # Row 0: terminal first fetch call, no tool_result.
    row0_trace = rows[0]["trace"]
    assert len(row0_trace) == 1
    assert "tool_call" in row0_trace[-1]
    assert "tool_result" not in row0_trace[-1]
    assert "thinking" in row0_trace[-1]
    assert "compare=" not in row0_trace[-1]["thinking"]

    # Row 1: prior fetch is complete history, terminal second fetch call.
    row1_trace = rows[1]["trace"]
    assert len(row1_trace) == 2
    assert "tool_call" in row1_trace[0] and "tool_result" in row1_trace[0]
    assert "thinking" not in row1_trace[0]
    assert "tool_call" in row1_trace[-1]
    assert "tool_result" not in row1_trace[-1]
    assert "thinking" in row1_trace[-1]
    assert "compare=" in row1_trace[-1]["thinking"]

    # Row 2: full history + terminal submit.
    row2_trace = rows[2]["trace"]
    assert len(row2_trace) == 3
    assert "tool_call" in row2_trace[0] and "tool_result" in row2_trace[0]
    assert "tool_call" in row2_trace[1] and "tool_result" in row2_trace[1]
    assert "thinking" not in row2_trace[0]
    assert "thinking" not in row2_trace[1]
    assert "submit" in row2_trace[-1]
    assert "thinking" in row2_trace[-1]
    assert "compare=" in row2_trace[-1]["thinking"]
    assert row2_trace[-1]["submit"]["args"] == {"position_mm": spec.ground_truth_mm}
    assert "reasoning" not in row2_trace[-1]["submit"]["args"]


def test_thinking_signature_comparison_turn_uses_one_capped_signature(
    patched_generate_trace_two_steps,  # noqa: ANN001, ARG001
    fake_visible_clues_module,  # noqa: ANN001, ARG001
) -> None:
    spec = _make_spec(gt_mm=5.0)
    rows = generate_synthetic_rows(
        [spec],
        multi_turn_floor=1.0,
        rng=random.Random(10),
        grid=_tiny_grid(center=5.0),
        synthetic_reasoning_mode="thinking_signature",
    )

    final_thinking = rows[-1]["trace"][-1]["thinking"]
    assert "compare=" in final_thinking
    assert final_thinking.count("Visible clues:") == 1
    assert "@5.00" in final_thinking
