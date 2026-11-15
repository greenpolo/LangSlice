"""The app plugins carry the package's version.

Claude Code and Codex update an installed plugin only when its manifest's
version changes, so a manifest left behind keeps users on old skills and
agents however often the connectors change.
"""

from __future__ import annotations

import json
import tomllib
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
MANIFESTS = (
    "connectors/claude-code/.claude-plugin/plugin.json",
    "connectors/codex/.codex-plugin/plugin.json",
)


def test_plugin_manifests_carry_the_package_version():
    pyproject = tomllib.loads((REPO / "pyproject.toml").read_text(encoding="utf-8"))
    version = pyproject["project"]["version"]
    found = {name: json.loads((REPO / name).read_text(encoding="utf-8"))["version"]
             for name in MANIFESTS}
    assert found == dict.fromkeys(MANIFESTS, version), (
        f"set each manifest's version to the package's {version}: {found}")
