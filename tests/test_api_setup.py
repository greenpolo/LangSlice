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


def test_status_lists_the_chatgpt_agent_and_image_models(isolated_home: Path) -> None:
    from langslice.providers.registry import OPENAI_OAUTH_DEFAULT_AGENT_MODEL

    oauth = setup.setup_status()["providers"]["openai-oauth"]
    assert oauth["agent_models"] == [
        "openai-oauth/gpt-6-astra", "openai-oauth/gpt-6-sol", "openai-oauth/gpt-6-luna",
        "openai-oauth/gpt-5.6-sol", "openai-oauth/gpt-5.6-terra", "openai-oauth/gpt-5.6-luna",
    ]
    assert oauth["default_agent_model"] == OPENAI_OAUTH_DEFAULT_AGENT_MODEL
    assert oauth["default_agent_model"] in oauth["agent_models"]
    assert oauth["image_models"] == ["gpt-image-2"]
    assert oauth["default_image_model"] == "gpt-image-2"
    # One definition: the transport's defaults are the listed ones.
    from langslice.providers import openai_oauth

    assert openai_oauth.DEFAULT_REVIEW_MODEL == oauth["default_agent_model"]
    assert openai_oauth.DEFAULT_IMAGE_MODEL == oauth["default_image_model"]


def test_status_models_do_not_import_the_transport() -> None:
    import subprocess
    import sys

    script = (
        "import sys; from langslice.doors.api import setup; setup.setup_status(); "
        "print('langslice.providers.openai_oauth' in sys.modules, 'google.adk' in sys.modules)"
    )
    out = subprocess.run([sys.executable, "-c", script], capture_output=True, text=True,
                         timeout=60, check=True)
    assert out.stdout.split() == ["False", "False"]


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


def test_the_library_loads_saved_keys_as_the_cli_does(isolated_home: Path,
                                                      monkeypatch: pytest.MonkeyPatch) -> None:
    """A script's ``langslice.open_job`` gets the keys saved by the setup
    dialog (``setup.api_key``) the way every CLI command does, through one
    shared loader (review finding 13); an explicit environment key wins."""
    import dataclasses

    import langslice
    from langslice.core.spec import NonlinearSpec
    from langslice.doors.jobs import create
    from tests.golden.record import atlas_loader, full_spec, write_sections

    monkeypatch.setenv("HOME", str(isolated_home))
    monkeypatch.setattr(setup, "load_dotenv", lambda: None)  # no developer .env here
    setup.save_api_key("gemini-api", "saved-gemini")
    setup.save_api_key("openai-api", "saved-openai")
    monkeypatch.setenv("OPENAI_API_KEY", "from-the-environment")
    folder = isolated_home / "stack"
    write_sections(folder)
    spec = dataclasses.replace(full_spec(folder), nonlinear=NonlinearSpec(provider="none"),
                               agent_preprocessing=False)
    create(spec, atlas_loader=atlas_loader()).close()
    monkeypatch.delenv("GEMINI_API_KEY", raising=False)
    job = langslice.open_job(str(folder), atlas_loader=atlas_loader())
    job.close()
    assert os.environ["GEMINI_API_KEY"] == "saved-gemini"
    assert os.environ["OPENAI_API_KEY"] == "from-the-environment"


def test_status_lists_every_image_model_choice_with_its_connection(
    isolated_home: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The dialog's image-model choices, in its order, connected or not, from
    offline presence checks only."""
    for name in ("GEMINI_API_KEY", "GOOGLE_API_KEY", "OPENAI_API_KEY", "OPENAI_IMAGE_API_KEY",
                 "OPENAI_IMAGE_BASE_URL", "OPENAI_BASE_URL", "LANGSLICE_OPENAI_AUTH"):
        monkeypatch.delenv(name, raising=False)
    choices = setup.setup_status()["image_models"]
    assert [c["provider"] for c in choices] == ["openai-oauth", "gemini-api", "openai-api",
                                                "none"]
    connected = {c["provider"]: c["connected"] for c in choices}
    assert connected == {"openai-oauth": False, "gemini-api": False, "openai-api": False,
                         "none": True}
    oauth = choices[0]
    assert oauth["models"] == ["gpt-image-2"] and oauth["default_model"] == "gpt-image-2"
    gemini = choices[1]
    assert gemini["models"] and all("-image" in m for m in gemini["models"])
    assert gemini["default_model"] == "gemini-3.1-flash-image"
    assert choices[3]["models"] == [] and choices[3]["default_model"] is None
    monkeypatch.setenv("GEMINI_API_KEY", "x")
    path = isolated_home / ".langslice" / "openai_auth.json"
    path.parent.mkdir(exist_ok=True)
    path.write_text(json.dumps({"tokens": {"access_token": "token"}}))
    connected = {c["provider"]: c["connected"] for c in setup.setup_status()["image_models"]}
    assert connected["gemini-api"] and connected["openai-oauth"]
    assert not connected["openai-api"]
