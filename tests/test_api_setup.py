"""Desktop setup keeps credentials private and works without a model connection."""

from __future__ import annotations

import json
import os
from pathlib import Path

import pytest

from langslice.api import setup


@pytest.fixture
def isolated_home(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    monkeypatch.setattr(Path, "home", classmethod(lambda cls: tmp_path))
    for name in ("OPENAI_API_KEY", "OPENAI_BASE_URL", "GEMINI_API_KEY", "GOOGLE_API_KEY"):
        monkeypatch.delenv(name, raising=False)
    return tmp_path


def test_status_without_credentials(isolated_home: Path) -> None:
    status = setup.setup_status()
    assert status["protocol_version"] == 1
    assert status["python_executable"]
    assert not status["providers"]["openai-oauth"]["configured"]
    assert not status["providers"]["gemini-api"]["configured"]
    assert not (isolated_home / ".langslice").exists()


def test_save_both_keys_preserves_them_and_never_returns_secrets(isolated_home: Path) -> None:
    result = setup.save_api_key("openai-api", "test-private-openai")
    assert result == {"configured": True, "source": "saved", "validated": False}
    setup.save_api_key("gemini-api", "test-private-gemini")
    path = isolated_home / ".langslice" / "provider_credentials.json"
    assert json.loads(path.read_text()) == {
        "openai-api": "test-private-openai", "gemini-api": "test-private-gemini"
    }
    if os.name == "posix":
        assert path.stat().st_mode & 0o777 == 0o600
    assert "test-private" not in json.dumps(setup.setup_status())
    assert list(path.parent.glob(".provider_credentials-*")) == []


def test_apply_respects_existing_credentials(
    isolated_home: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    setup.save_api_key("openai-api", "saved-openai")
    setup.save_api_key("gemini-api", "saved-gemini")
    monkeypatch.setenv("GOOGLE_API_KEY", "existing-google")
    # Register this environment change with monkeypatch before the helper writes it.
    monkeypatch.setenv("OPENAI_API_KEY", "")
    monkeypatch.setenv("OPENAI_BASE_URL", "")
    setup.apply_saved_credentials()
    assert os.environ["OPENAI_API_KEY"] == "saved-openai"
    assert os.environ["OPENAI_BASE_URL"] == "https://api.openai.com/v1"
    assert os.environ["GOOGLE_API_KEY"] == "existing-google"
    assert "GEMINI_API_KEY" not in os.environ


@pytest.mark.parametrize("provider,key", [
    ("openai-oauth", "secret"), ("other", "secret"),
    ("openai-api", " "), ("gemini-api", "bad\nkey"),
])
def test_invalid_save_does_not_write(isolated_home: Path, provider: str, key: str) -> None:
    with pytest.raises(ValueError):
        setup.save_api_key(provider, key)
    assert not (isolated_home / ".langslice").exists()


def test_corrupt_settings_report_generic_error_and_are_not_overwritten(isolated_home: Path) -> None:
    path = isolated_home / ".langslice" / "provider_credentials.json"
    path.parent.mkdir()
    path.write_text("sensitive malformed input")
    status = setup.setup_status()
    assert status["credentials_error"]
    assert "sensitive" not in json.dumps(status)
    with pytest.raises(ValueError):
        setup.save_api_key("openai-api", "new-key")
    assert path.read_text() == "sensitive malformed input"


def test_oauth_presence_does_not_refresh_or_expose_token(isolated_home: Path) -> None:
    path = isolated_home / ".codex" / "auth.json"
    path.parent.mkdir()
    path.write_text(json.dumps({"tokens": {"access_token": "private-oauth-token"}}))
    original = path.read_text()
    status = setup.setup_status()
    assert status["providers"]["openai-oauth"] == {
        "configured": True, "source": "codex", "validated": False,
    }
    assert "private-oauth-token" not in json.dumps(status)
    assert path.read_text() == original


def test_login_forwards_url_callback(isolated_home: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    from langslice.providers import openai_oauth

    urls: list[str] = []

    def fake_login(*, timeout_s, on_url):
        assert timeout_s == 60
        on_url("https://example.test/login")
        path = isolated_home / ".langslice" / "openai_auth.json"
        path.parent.mkdir()
        path.write_text(json.dumps({"access_token": "private-token"}))

    monkeypatch.setattr(openai_oauth, "login", fake_login)
    result = setup.login_oauth(urls.append, timeout_s=60)
    assert urls == ["https://example.test/login"]
    assert result["configured"]
    assert "private-token" not in json.dumps(result)
