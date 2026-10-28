"""The Settings tab's server side: connections, the agent model, where
registration runs.

Read by ``GET settings``; changed by three ``POST`` endpoints the server
checks first (token, ``Host``, same-site ``Origin``, a JSON body):

- ``settings``: ``{connection?, runs_in?, model?, reasoning?}``, saved to the
  user's settings file (:mod:`langslice.doors.api.settings`); ``model`` and
  ``reasoning`` belong to the chosen connection (or ``connection`` in the
  same body);
- ``settings/key``: ``{provider, key}``, saved where LangSlice keeps keys
  (:func:`langslice.doors.api.setup.save_api_key`,
  ``~/.langslice/provider_credentials.json``); the key is never sent back;
- ``settings/signin``: starts the ChatGPT browser sign-in
  (:func:`langslice.doors.api.setup.login_oauth`) in a thread of this
  process; ``GET settings`` reports its progress.

Every answer is :meth:`Settings.state`: presence words, never a key or a
token. Only the person at this window sets these; no agent tool calls here.
"""

from __future__ import annotations

import threading
from collections.abc import Callable
from typing import Any

from langslice.doors.api import settings as store
from langslice.doors.api import setup

#: The connections as the window names them, with how each is set up.
LABELS = {
    "openai-oauth": ("ChatGPT account", "signin"),
    "openai-api": ("OpenAI API key", "key"),
    "anthropic-api": ("Claude (Anthropic) API key", "key"),
    "gemini-api": ("Gemini API key", "key"),
}
#: The POST resources (under the server's token).
POSTS = ("settings", "settings/key", "settings/signin")
#: How long the browser sign-in waits for the person.
SIGNIN_TIMEOUT_S = 300.0


class SettingsError(ValueError):
    """A request the settings cannot take; its text is safe to show."""


class Settings:
    """The Settings tab's state and changes. *login* runs the browser
    sign-in (``setup.login_oauth``'s signature; a test passes its own)."""

    def __init__(self, *, login: Callable[..., Any] | None = None) -> None:
        self._login = login or setup.login_oauth
        self._lock = threading.Lock()
        self._signin: dict[str, Any] = {"running": False, "url": None, "error": None}

    # --- reading -------------------------------------------------------------

    def state(self) -> dict[str, Any]:
        """What the tab shows: each connection's presence, the saved choices,
        the effective run location, the model choices, the sign-in's progress."""
        saved = store.read()
        found = setup.connections()
        rows = []
        for name in store.CONNECTIONS:
            label, kind = LABELS[name]
            presence = found["providers"].get(name, {})
            configured = bool(presence.get("configured"))
            source = presence.get("source")
            if kind == "signin":
                word = "Signed in" if configured else "Not signed in"
            elif source == "environment":
                word = "Key set in the environment"
            else:
                word = "Key saved" if configured else "No key"
            rows.append({"id": name, "label": label, "kind": kind, "present": configured,
                         "state": word, "source": source})
        chosen = saved["connection"]
        present = any(row["present"] for row in rows if row["id"] == chosen)
        models = store.agent_models()
        agent = {}
        for name in store.CONNECTIONS:
            entry = saved["agent"].get(name) or {}
            model = entry.get("model")
            agent[name] = {
                "models": models[name]["models"],
                "default": models[name]["default"],
                "model": store.model_name(name, model) if model else models[name]["default"],
                "saved": bool(model),
                "reasoning": entry.get("reasoning"),
            }
        with self._lock:
            signin = dict(self._signin)
        return {
            "connections": rows,
            "connection": chosen,
            "runs_in": {
                "saved": saved["runs_in"],
                "enabled": bool(chosen and present),
                "value": (saved["runs_in"] or "langslice") if chosen and present else None,
            },
            "agent": agent,
            "reasoning": list(store.REASONING),
            "signin": signin,
            "settings_file": str(store.settings_file() or ""),
            "credentials_error": found["credentials_error"],
        }

    # --- changing ------------------------------------------------------------

    def post(self, resource: str, body: Any) -> dict[str, Any]:
        """Apply one POST; :class:`SettingsError` for a request it refuses."""
        if not isinstance(body, dict):
            raise SettingsError("Expected a JSON object.")
        if resource == "settings":
            self._save_choices(body)
        elif resource == "settings/key":
            provider = body.get("provider")
            if provider not in ("openai-api", "anthropic-api", "gemini-api"):
                raise SettingsError("Unknown API-key provider.")
            key = body.get("key")
            if not isinstance(key, str):
                raise SettingsError("Enter a key.")
            try:
                setup.save_api_key(provider, key)
            except ValueError:
                # The message names no key, but a reply never repeats the input.
                raise SettingsError("Enter a key without spaces, or check that "
                                    "~/.langslice/provider_credentials.json is readable."
                                    ) from None
        elif resource == "settings/signin":
            self._start_signin()
        else:
            raise SettingsError("Unknown settings request.")
        return self.state()

    def _save_choices(self, body: dict[str, Any]) -> None:
        if store.settings_file() is None:
            raise SettingsError("Settings are off in this environment (LANGSLICE_SETTINGS).")
        saved = store.read()
        if "connection" in body:
            if body["connection"] not in store.CONNECTIONS:
                raise SettingsError("Unknown connection.")
            saved["connection"] = body["connection"]
        if "runs_in" in body:
            if body["runs_in"] not in store.RUNS_IN:
                raise SettingsError("Unknown run location.")
            saved["runs_in"] = body["runs_in"]
        if "model" in body or "reasoning" in body:
            connection = saved["connection"]
            if connection is None:
                raise SettingsError("Choose a connection first.")
            entry = dict(saved["agent"].get(connection) or {})
            if "model" in body:
                model = body["model"]
                if not isinstance(model, str) or not model.strip() or any(
                        c.isspace() or ord(c) < 32 for c in model.strip()):
                    raise SettingsError("Enter a model name without spaces.")
                entry["model"] = store.model_string(connection, model)
            if "reasoning" in body:
                reasoning = body["reasoning"] or None
                if reasoning is not None and reasoning not in store.REASONING:
                    raise SettingsError("Unknown reasoning effort.")
                entry["reasoning"] = reasoning
            saved["agent"][connection] = entry
        # A connection chosen with no model of its own saved gets LangSlice's default.
        connection = saved["connection"]
        if connection and not (saved["agent"].get(connection) or {}).get("model"):
            default = store.agent_models()[connection]["default"]
            if default:
                saved["agent"][connection] = {
                    **(saved["agent"].get(connection) or {"reasoning": None}),
                    "model": store.model_string(connection, default)}
        store.write(saved)

    def _start_signin(self) -> None:
        with self._lock:
            if self._signin["running"]:
                return
            self._signin = {"running": True, "url": None, "error": None}

        def on_url(url: str) -> None:
            with self._lock:
                self._signin["url"] = url

        def run() -> None:
            error = None
            try:
                self._login(on_url=on_url, timeout_s=SIGNIN_TIMEOUT_S)
            except Exception as exc:  # a provider's reply may hold anything: say nothing of it
                error = ("The sign-in timed out. Try again." if "Timed out" in str(exc)
                         else "The sign-in did not finish. Try again.")
            with self._lock:
                self._signin = {"running": False, "url": None, "error": error}

        threading.Thread(target=run, daemon=True, name="langslice-gui-signin").start()
