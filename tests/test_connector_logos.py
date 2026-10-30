"""Each AI-app plugin carries LangSlice's logo as a copy inside its own folder
(an app installs only that folder): the copies are the brand files in
assets/logo, and each manifest names its copy."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
LOGO = ROOT / "assets" / "logo"
CONNECTORS = ROOT / "connectors"

COPIES = [
    ("codex/assets/logo.svg", "logomark_light.svg"),
    ("codex/assets/logo-dark.svg", "logomark_dark.svg"),
    ("codex/assets/icon.svg", "logomark.svg"),
    ("antigravity/assets/logo.svg", "logomark.svg"),
    ("claude-code/.claude-plugin/icon.png", "logomark_light_512.png"),
]


@pytest.mark.parametrize(("copy", "brand"), COPIES,
                         ids=[copy for copy, _ in COPIES])
def test_plugin_logo_is_the_brand_file(copy: str, brand: str) -> None:
    copied = (CONNECTORS / copy).read_bytes()
    assert copied == (LOGO / brand).read_bytes(), copy


def _manifest(path: str) -> dict[str, object]:
    data: dict[str, object] = json.loads(
        (CONNECTORS / path).read_text(encoding="utf-8"))
    return data


def test_manifests_name_their_logo_copies() -> None:
    codex = _manifest("codex/.codex-plugin/plugin.json")["interface"]
    assert isinstance(codex, dict)
    assert codex["logo"] == "./assets/logo.svg"
    assert codex["logoDark"] == "./assets/logo-dark.svg"
    assert codex["composerIcon"] == "./assets/icon.svg"
    assert _manifest("antigravity/plugin.json")["logo"] == "assets/logo.svg"
