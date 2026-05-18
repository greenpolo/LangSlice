from __future__ import annotations

from pathlib import Path


def test_default_manifest_root_prefers_models_data() -> None:
    from langslice_data.manifest.paths import default_manifest_root

    repo_root = Path("C:/repo")
    assert default_manifest_root(repo_root) == repo_root / "models" / "data" / "manifest"


def test_resolve_manifest_root_falls_back_to_data_manifest(tmp_path: Path) -> None:
    from langslice_data.manifest.paths import resolve_manifest_root

    (tmp_path / "data" / "manifest").mkdir(parents=True)
    resolved = resolve_manifest_root(tmp_path)
    assert resolved == (tmp_path / "data" / "manifest")
