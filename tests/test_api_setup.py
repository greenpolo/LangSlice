"""Desktop setup keeps credentials private and works without a model connection."""

from __future__ import annotations

import json
import os
from pathlib import Path

import pytest

from langslice.doors.api import setup


@pytest.fixture
def isolated_home(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    monkeypatch.setattr(Path, "home", classmethod(lambda cls: tmp_path))
    monkeypatch.setenv("HOME", str(tmp_path))
    monkeypatch.setenv("USERPROFILE", str(tmp_path))
    for name in ("OPENAI_API_KEY", "OPENAI_BASE_URL", "GEMINI_API_KEY", "GOOGLE_API_KEY",
                 "LANGSLICE_OPENAI_AUTH"):
        # set then delete, so the variable is restored (or removed) after the test
        # even when the code under test writes it to os.environ
        monkeypatch.setenv(name, "")
        monkeypatch.delenv(name)
    return tmp_path


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
    status = setup.setup_status()
    assert status["protocol_version"] == 2
    assert "test-private" not in json.dumps(status)
    assert list(path.parent.glob(".provider_credentials-*")) == []


def test_apply_respects_existing_credentials(
    isolated_home: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    setup.save_api_key("openai-api", "saved-openai")
    setup.save_api_key("gemini-api", "saved-gemini")
    monkeypatch.setenv("GOOGLE_API_KEY", "existing-google")
    # Register this environment change with monkeypatch before the helper writes it.
    monkeypatch.setenv("OPENAI_API_KEY", "")
    setup.apply_saved_credentials()
    assert os.environ["OPENAI_API_KEY"] == "saved-openai"
    assert os.environ["GOOGLE_API_KEY"] == "existing-google"
    assert "GEMINI_API_KEY" not in os.environ


def test_unknown_or_unreadable_saved_entries_are_skipped(
    isolated_home: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A settings file naming a provider this version does not know (a newer
    version's) loads the rest and keeps that entry; an unreadable one loads
    nothing and raises nowhere."""
    path = isolated_home / ".langslice" / "provider_credentials.json"
    path.parent.mkdir()
    path.write_text(json.dumps({"future-api": "f", "gemini-api": "saved-gemini"}))
    monkeypatch.setenv("GEMINI_API_KEY", "")
    setup.apply_saved_credentials()
    assert os.environ["GEMINI_API_KEY"] == "saved-gemini"
    assert setup.setup_status()["providers"]["gemini-api"]["configured"]
    setup.save_api_key("openai-api", "new")
    assert json.loads(path.read_text()) == {"future-api": "f", "gemini-api": "saved-gemini",
                                            "openai-api": "new"}
    path.write_text("not json")
    setup.apply_saved_credentials()  # warns, raises nothing
    setup.load_credentials()


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
    path = isolated_home / ".langslice" / "openai_auth.json"
    path.parent.mkdir(exist_ok=True)
    path.write_text(json.dumps({"tokens": {"access_token": "private-oauth-token"}}))
    original = path.read_text()
    status = setup.setup_status()
    oauth = status["providers"]["openai-oauth"]
    assert {key: oauth[key] for key in ("configured", "source", "validated")} == {
        "configured": True, "source": "saved", "validated": False,
    }
    assert "private-oauth-token" not in json.dumps(status)
    assert path.read_text() == original


def test_a_codex_login_is_not_a_langslice_login(isolated_home: Path) -> None:
    path = isolated_home / ".codex" / "auth.json"
    path.parent.mkdir()
    path.write_text(json.dumps({"tokens": {"access_token": "codex-token"}}))
    assert setup.setup_status()["providers"]["openai-oauth"]["configured"] is False


def test_the_library_loads_saved_keys_as_the_cli_does(isolated_home: Path,
                                                      monkeypatch: pytest.MonkeyPatch) -> None:
    """A script's ``langslice.open_job`` gets the keys saved by the setup
    dialog (``setup.api_key``) the way every CLI command does, through one
    shared loader (review finding 13); an explicit environment key wins."""
    import dataclasses

    import langslice
    from langslice.core.spec import NonlinearSpec
    from langslice.doors.jobs import create
    from tests.synthetic_stack import atlas_loader, full_spec, write_sections

    monkeypatch.setattr(setup, "load_dotenv", lambda: None)  # no developer .env here
    setup.save_api_key("gemini-api", "saved-gemini")
    setup.save_api_key("openai-api", "saved-openai")
    monkeypatch.setenv("OPENAI_API_KEY", "from-the-environment")
    folder = isolated_home / "stack"
    write_sections(folder)
    spec = dataclasses.replace(full_spec(folder), nonlinear=NonlinearSpec(provider="none"))
    create(spec, atlas_loader=atlas_loader()).close()
    job = langslice.open_job(str(folder), atlas_loader=atlas_loader())
    job.close()
    assert os.environ["GEMINI_API_KEY"] == "saved-gemini"
    assert os.environ["OPENAI_API_KEY"] == "from-the-environment"


def test_status_lists_every_image_model_choice_with_its_connection(
    isolated_home: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The dialog's image-model choices, connected or not, from offline presence checks."""
    choices = setup.setup_status()["image_models"]
    assert [c["provider"] for c in choices] == ["openai-oauth", "gemini-api", "openai-api",
                                                "none"]
    connected = {c["provider"]: c["connected"] for c in choices}
    assert connected == {"openai-oauth": False, "gemini-api": False, "openai-api": False,
                         "none": True}
    monkeypatch.setenv("GEMINI_API_KEY", "x")
    path = isolated_home / ".langslice" / "openai_auth.json"
    path.parent.mkdir(exist_ok=True)
    path.write_text(json.dumps({"tokens": {"access_token": "token"}}))
    connected = {c["provider"]: c["connected"] for c in setup.setup_status()["image_models"]}
    assert connected["gemini-api"] and connected["openai-oauth"]
    assert not connected["openai-api"]
