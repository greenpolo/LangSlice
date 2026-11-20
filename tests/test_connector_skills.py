"""The connectors' committed skill copies are what the build script writes."""

from __future__ import annotations

import importlib.util
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def test_committed_connector_skills_match_the_canonical_source() -> None:
    script = ROOT / "packaging" / "build_connector_skills.py"
    spec = importlib.util.spec_from_file_location("build_connector_skills", script)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    drift = [path.relative_to(ROOT).as_posix() for path in module.build(check=True)]
    assert not drift, (
        f"connector skills differ from src/langslice/skills: {drift}. "
        "Run python packaging/build_connector_skills.py."
    )
