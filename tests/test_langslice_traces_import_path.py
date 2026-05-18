from __future__ import annotations

from pathlib import Path

import langslice_traces
from langslice_traces.generator import canonical_atlas_repo_path


def test_langslice_traces_imports_from_new_package_root() -> None:
    module_file = Path(langslice_traces.__file__).resolve()
    expected_root = Path("models/langslice-traces/langslice_traces")
    assert expected_root.as_posix() in module_file.as_posix()


def test_canonical_atlas_repo_path_env_override(monkeypatch) -> None:
    monkeypatch.setenv("LANGSLICE_CANONICAL_ATLAS_ROOT", "models/custom/atlas")
    out = canonical_atlas_repo_path("atlas/demo/coronal/1.00mm.jpg")
    assert out == "models/custom/atlas/demo/coronal/1.00mm.jpg"
