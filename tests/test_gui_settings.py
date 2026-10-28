"""The Settings tab: the settings file, the POST protections, keys never sent
back, the ChatGPT sign-in, and ``linear run``'s model unchanged without the file."""

from __future__ import annotations

import argparse
import http.client
import json
import os
import time
from collections.abc import Iterator
from pathlib import Path
from typing import Any

import pytest

from langslice.doors.api import settings as store
from langslice.doors.api import setup
from langslice.hosts.gui.server import GuiServer
from langslice.hosts.gui.settings import Settings
from langslice.hosts.gui.watch import JobList

SECRET = "sk-test-NEVER-ECHO-0123456789"


@pytest.fixture
def home(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    monkeypatch.setattr(Path, "home", classmethod(lambda cls: tmp_path))
    monkeypatch.setenv("HOME", str(tmp_path))
    monkeypatch.delenv(store.SETTINGS_ENV, raising=False)
    for name in ("OPENAI_API_KEY", "OPENAI_IMAGE_API_KEY", "GEMINI_API_KEY", "GOOGLE_API_KEY",
                 "ANTHROPIC_API_KEY", "LANGSLICE_OPENAI_AUTH"):
        monkeypatch.setenv(name, "")  # restored after the test even if the code sets it
        monkeypatch.delenv(name)
    return tmp_path


def _fake_login(home: Path) -> Any:
    def login(on_url: Any = None, timeout_s: float = 0) -> dict[str, object]:
        if on_url:
            on_url("https://auth.example/authorize?x=1")
        time.sleep(0.2)
        path = home / ".langslice" / "openai_auth.json"
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps({"tokens": {"access_token": "fake-token-XYZ"}}))
        return {"configured": True}
    return login


@pytest.fixture
def server(home: Path) -> Iterator[GuiServer]:
    gui = GuiServer(JobList([], recent=None), settings=Settings(login=_fake_login(home)))
    gui.start()
    yield gui
    gui.close()


def _request(gui: GuiServer, resource: str, body: Any = None, *, method: str = "POST",
             headers: dict[str, str | None] | None = None, token: str | None = None,
             raw: bytes | None = None) -> tuple[int, bytes]:
    connection = http.client.HTTPConnection("127.0.0.1", gui.port, timeout=5)
    path = f"/{token if token is not None else gui.token}/{resource}"
    data = raw if raw is not None else (json.dumps(body).encode() if body is not None else b"")
    sent = {"Host": f"127.0.0.1:{gui.port}", "Origin": f"http://127.0.0.1:{gui.port}",
            "Content-Type": "application/json", "Content-Length": str(len(data))}
    for key, value in (headers or {}).items():
        if value is None:
            sent.pop(key, None)
        else:
            sent[key] = value
    connection.putrequest(method, path, skip_host=True, skip_accept_encoding=True)
    for key, value in sent.items():
        connection.putheader(key, value)
    connection.endheaders(data if method == "POST" else None)
    response = connection.getresponse()
    out = response.read()
    connection.close()
    return response.status, out


def _get(gui: GuiServer) -> dict[str, Any]:
    status, body = _request(gui, "settings", method="GET")
    assert status == 200
    return json.loads(body)


# --- the file ---------------------------------------------------------------


def test_the_file_round_trips_in_the_home_and_cleans_what_it_reads(home: Path) -> None:
    assert store.settings_file() == home / ".langslice" / "settings.json"
    assert store.read() == store.empty()
    saved = store.write({"connection": "anthropic-api", "runs_in": "app", "agent": {
        "anthropic-api": {"model": "anthropic-api/claude-opus-5-5", "reasoning": "high"},
        "nobody": {"model": "x"}}})
    assert json.loads((home / ".langslice" / "settings.json").read_text()) == saved
    assert store.read() == {"version": 1, "connection": "anthropic-api", "runs_in": "app",
                            "agent": {"anthropic-api": {"model": "anthropic-api/claude-opus-5-5",
                                                        "reasoning": "high"}}}
    (home / ".langslice" / "settings.json").write_text(json.dumps(
        {"connection": "skynet", "runs_in": "moon",
         "agent": {"gemini-api": {"reasoning": "huge"}}}))
    assert store.read() == {**store.empty(),
                            "agent": {"gemini-api": {"model": None, "reasoning": None}}}
    (home / ".langslice" / "settings.json").write_text("{not json")
    assert store.read() == store.empty()


def test_the_environment_names_another_file_or_none(home: Path,
                                                    monkeypatch: pytest.MonkeyPatch) -> None:
    other = home / "elsewhere.json"
    monkeypatch.setenv(store.SETTINGS_ENV, str(other))
    store.write({"connection": "gemini-api"})
    assert json.loads(other.read_text())["connection"] == "gemini-api"
    monkeypatch.setenv(store.SETTINGS_ENV, "off")
    assert store.settings_file() is None and store.read() == store.empty()
    with pytest.raises(ValueError):
        store.write({"connection": "gemini-api"})


def test_the_test_suite_never_reads_the_real_home() -> None:
    assert os.environ.get(store.SETTINGS_ENV) == "off"


# --- the harness --------------------------------------------------------------


def _args(**given: Any) -> argparse.Namespace:
    return argparse.Namespace(**{"model": None, "reasoning": None, **given})


def test_linear_run_keeps_its_default_without_the_file(home: Path) -> None:
    from langslice.doors.cli.linear import apply_saved_agent_choice

    args = _args()
    assert apply_saved_agent_choice(args) is False
    assert (args.model, args.reasoning) == (None, None)
    # A connection chosen with no model saved changes nothing either.
    store.write({"connection": "gemini-api"})
    assert apply_saved_agent_choice(args) is False and args.model is None


def test_linear_run_keeps_its_default_with_settings_off(home: Path,
                                                        monkeypatch: pytest.MonkeyPatch) -> None:
    from langslice.doors.cli.linear import apply_saved_agent_choice

    store.write({"connection": "anthropic-api",
                 "agent": {"anthropic-api": {"model": "anthropic-api/claude-opus-5-5"}}})
    monkeypatch.setenv(store.SETTINGS_ENV, "off")
    args = _args()
    assert apply_saved_agent_choice(args) is False and args.model is None


def test_linear_run_reads_the_chosen_model_only_without_flags(home: Path) -> None:
    from langslice.doors.cli.linear import apply_saved_agent_choice

    store.write({"connection": "anthropic-api", "agent": {
        "anthropic-api": {"model": "anthropic-api/claude-opus-5-5", "reasoning": "high"},
        "openai-oauth": {"model": "openai-oauth/gpt-6-luna"}}})
    args = _args()
    assert apply_saved_agent_choice(args) is True
    assert (args.model, args.reasoning) == ("anthropic-api/claude-opus-5-5", "high")
    args = _args(reasoning="low")
    apply_saved_agent_choice(args)
    assert (args.model, args.reasoning) == ("anthropic-api/claude-opus-5-5", "low")
    args = _args(model="openai-oauth/gpt-6.1-sol")
    assert apply_saved_agent_choice(args) is False
    assert (args.model, args.reasoning) == ("openai-oauth/gpt-6.1-sol", None)


# --- the server ---------------------------------------------------------------


def test_posts_need_the_token_host_and_this_origin(server: GuiServer, home: Path) -> None:
    body = {"connection": "openai-api"}
    port = server.port
    assert _request(server, "settings", body, token="wrong")[0] == 404
    assert _request(server, "settings", body, headers={"Host": "evil.example"})[0] == 404
    assert _request(server, "settings", body, headers={"Origin": None})[0] == 403
    assert _request(server, "settings", body, headers={"Origin": "http://evil.example"})[0] == 403
    assert _request(server, "settings", body,
                    headers={"Origin": f"http://localhost:{port}"})[0] == 403
    assert _request(server, "settings", body, headers={"Sec-Fetch-Site": "cross-site"})[0] == 403
    assert _request(server, "settings", body, headers={"Content-Type": "text/plain"})[0] == 415
    assert _request(server, "settings", raw=b"{" * (20 * 1024))[0] == 413
    assert _request(server, "settings", raw=b"{nope")[0] == 400
    assert not (home / ".langslice" / "settings.json").exists()
    status, answer = _request(server, "settings", body)
    assert status == 200 and json.loads(answer)["connection"] == "openai-api"
    assert store.read()["connection"] == "openai-api"


def test_viewer_endpoints_stay_get_only(server: GuiServer) -> None:
    for resource in ("", "jobs", "jobs/0/snapshot", "app.js", "settings.js", "settings/other"):
        assert _request(server, resource, {})[0] == 405


def test_a_saved_key_is_never_sent_back(server: GuiServer, home: Path) -> None:
    assert _get(server)["connections"][1]["state"] == "No key"
    status, answer = _request(server, "settings/key", {"provider": "anthropic-api",
                                                       "key": SECRET})
    assert status == 200 and SECRET.encode() not in answer
    rows = {row["id"]: row for row in json.loads(answer)["connections"]}
    assert rows["anthropic-api"]["state"] == "Key saved" and rows["anthropic-api"]["present"]
    status, raw = _request(server, "settings", method="GET")
    assert SECRET.encode() not in raw
    saved = json.loads((home / ".langslice" / "provider_credentials.json").read_text())
    assert saved == {"anthropic-api": SECRET}
    status, answer = _request(server, "settings/key", {"provider": "openai-api",
                                                       "key": "has space " + SECRET})
    assert status == 400 and SECRET.encode() not in answer
    assert _request(server, "settings/key", {"provider": "none", "key": SECRET})[0] == 400


def test_a_saved_claude_key_loads_for_the_agent_and_runs_no_image_model(
        home: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    setup.save_api_key("anthropic-api", SECRET)
    assert setup.connections()["providers"]["anthropic-api"]["configured"] is True
    assert setup.image_model_connected("anthropic-api") is False
    setup.apply_saved_credentials()
    assert os.environ["ANTHROPIC_API_KEY"] == SECRET


def test_the_old_claude_key_file_counts_as_saved(home: Path) -> None:
    (home / ".langslice").mkdir()
    (home / ".langslice" / "anthropic_api_key").write_text(SECRET + "\n")
    status = setup.connections()["providers"]["anthropic-api"]
    assert status == {"configured": True, "source": "saved", "validated": False}


def test_run_location_waits_for_a_working_connection(server: GuiServer) -> None:
    state = _get(server)
    assert state["connection"] is None
    assert state["runs_in"] == {"saved": None, "enabled": False, "value": None}
    _, answer = _request(server, "settings", {"connection": "gemini-api"})
    assert json.loads(answer)["runs_in"]["enabled"] is False  # chosen, no key
    _request(server, "settings/key", {"provider": "gemini-api", "key": SECRET})
    state = _get(server)
    assert state["runs_in"] == {"saved": None, "enabled": True, "value": "langslice"}
    _, answer = _request(server, "settings", {"runs_in": "app"})
    assert json.loads(answer)["runs_in"]["value"] == "app"
    assert store.read()["runs_in"] == "app"
    assert _request(server, "settings", {"runs_in": "cloud"})[0] == 400


def test_models_default_to_langslices_and_save_per_connection(server: GuiServer) -> None:
    from langslice.providers import registry

    _, answer = _request(server, "settings", {"connection": "openai-oauth"})
    oauth = json.loads(answer)["agent"]["openai-oauth"]
    assert oauth["model"] == oauth["default"] == store.model_name(
        "openai-oauth", registry.OPENAI_OAUTH_DEFAULT_AGENT_MODEL)
    # Choosing the ChatGPT account saves LangSlice's own default: linear run is unchanged.
    assert store.agent_choice() == (registry.OPENAI_OAUTH_DEFAULT_AGENT_MODEL, None)
    _request(server, "settings", {"model": "gpt-6-luna", "reasoning": "high"})
    assert store.agent_choice() == ("openai-oauth/gpt-6-luna", "high")
    _, answer = _request(server, "settings", {"connection": "anthropic-api"})
    claude = json.loads(answer)["agent"]["anthropic-api"]
    assert claude["default"] is None and claude["model"] is None and claude["models"]
    assert store.agent_choice() is None
    _request(server, "settings", {"model": "claude-opus-5-5"})
    assert store.agent_choice() == ("anthropic-api/claude-opus-5-5", None)
    _request(server, "settings", {"connection": "openai-oauth"})
    assert store.agent_choice() == ("openai-oauth/gpt-6-luna", "high")
    assert _request(server, "settings", {"reasoning": "extreme"})[0] == 400
    assert _request(server, "settings", {"model": "two words"})[0] == 400


def test_chatgpt_sign_in_runs_in_the_window_and_shows_no_token(server: GuiServer) -> None:
    rows = {row["id"]: row for row in _get(server)["connections"]}
    assert rows["openai-oauth"]["state"] == "Not signed in"
    status, answer = _request(server, "settings/signin", {})
    assert status == 200 and json.loads(answer)["signin"]["running"] is True
    deadline = time.monotonic() + 5
    while (state := _get(server))["signin"]["running"] and time.monotonic() < deadline:
        time.sleep(0.05)
    rows = {row["id"]: row for row in state["connections"]}
    assert rows["openai-oauth"]["state"] == "Signed in" and state["signin"]["error"] is None
    assert b"fake-token-XYZ" not in _request(server, "settings", method="GET")[1]


def test_a_failed_sign_in_says_so_without_the_providers_words(home: Path) -> None:
    def login(on_url: Any = None, timeout_s: float = 0) -> None:
        raise RuntimeError("OAuth error: secret provider detail")

    settings = Settings(login=login)
    settings.post("settings/signin", {})
    deadline = time.monotonic() + 5
    while settings.state()["signin"]["running"] and time.monotonic() < deadline:
        time.sleep(0.02)
    error = settings.state()["signin"]["error"]
    assert error and "secret" not in error
