from __future__ import annotations

from pathlib import Path


def test_fixture_allocation_split_lookup() -> None:
    from langslice_data.manifest import allocations

    fixture_root = Path(__file__).resolve().parents[2] / "models" / "data" / "fixtures" / "manifest"
    allocations.ALLOCATIONS_ROOT = fixture_root / "allocations"
    assert allocations.compute_split_for("coronal", "demo_section_001") == "eval"
    assert allocations.compute_split_for("coronal", "missing_section") is None
