"""The user's settings file: the chosen connection, its agent model and where
registration runs (``~/.langslice/settings.json``).

Only a person sets it, in the LangSlice window's Settings tab
(``hosts/gui/settings.py``); no agent tool reads or writes it. Keys and
logins are not in it: they stay where :mod:`langslice.doors.api.setup` keeps
them. ``LANGSLICE_SETTINGS`` (:data:`SETTINGS_ENV`) names another file, or
``off`` (also ``0``, ``false``, ``no``, ``none``, empty) for none; the test
suite sets it off. The format (``version`` 1)::

    {"version": 1,
     "connection": "openai-oauth" | "openai-api" | "anthropic-api" | "gemini-api" | null,
     "runs_in": "langslice" | "app" | null,
     "agent": {"<connection>": {"model": "<--model string>", "reasoning": "<effort>" | null}}}

``agent`` keeps a choice per connection, so switching back finds it again.
``langslice linear run`` reads the chosen connection's ``agent`` entry when it
is given neither ``--model`` nor ``--reasoning`` (:func:`agent_choice`); with
no file, or no model saved for the chosen connection, nothing changes.
``runs_in`` is saved but read by nothing yet.
"""

from __future__ import annotations

import json
import logging
import os
import tempfile
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)

#: Another file for the settings, or ``off`` for none.
SETTINGS_ENV = "LANGSLICE_SETTINGS"
SETTINGS_FILE = "settings.json"
VERSION = 1
_OFF = frozenset({"", "0", "off", "false", "no", "none"})
#: The connections, in the order the window lists them.
CONNECTIONS: tuple[str, ...] = ("openai-oauth", "openai-api", "anthropic-api", "gemini-api")
#: Where registration runs: the user's agent app, or LangSlice's own agent.
RUNS_IN: tuple[str, ...] = ("app", "langslice")
#: The reasoning efforts ``langslice linear run --reasoning`` takes.
REASONING: tuple[str, ...] = ("low", "medium", "high", "xhigh", "max")
#: How each connection spells a model for ``--model``.
_MODEL_PREFIX = {"openai-oauth": "openai-oauth/", "openai-api": "",
                 "anthropic-api": "anthropic-api/", "gemini-api": "gemini-api/"}


def settings_file() -> Path | None:
    """The settings file, or None when settings are off."""
    value = os.environ.get(SETTINGS_ENV)
    if value is None:
        return Path.home() / ".langslice" / SETTINGS_FILE
    if value.strip().lower() in _OFF:
        return None
    return Path(value).expanduser()


def empty() -> dict[str, Any]:
    return {"version": VERSION, "connection": None, "runs_in": None, "agent": {}}


def read() -> dict[str, Any]:
    """The settings, cleaned: an unknown connection or effort, or a file that
    cannot be read, reads as unset (with a warning), never as an error."""
    out = empty()
    path = settings_file()
    if path is None or not path.exists():
        return out
    try:
        doc = json.loads(path.read_text(encoding="utf-8"))
        if not isinstance(doc, dict):
            raise ValueError("not a JSON object")
    except (OSError, ValueError) as exc:
        logger.warning("The settings file %s could not be read (%s); using none.", path, exc)
        return out
    if doc.get("connection") in CONNECTIONS:
        out["connection"] = doc["connection"]
    if doc.get("runs_in") in RUNS_IN:
        out["runs_in"] = doc["runs_in"]
    agent = doc.get("agent")
    for connection, entry in (agent.items() if isinstance(agent, dict) else ()):
        if connection not in CONNECTIONS or not isinstance(entry, dict):
            continue
        model = entry.get("model")
        reasoning = entry.get("reasoning")
        out["agent"][connection] = {
            "model": model.strip() if isinstance(model, str) and model.strip() else None,
            "reasoning": reasoning if reasoning in REASONING else None,
        }
    return out


def write(settings: dict[str, Any]) -> dict[str, Any]:
    """Save *settings* (cleaned as :func:`read` cleans them) whole, by a
    temporary file and a rename; returns what was saved. ``ValueError`` when
    settings are off."""
    path = settings_file()
    if path is None:
        raise ValueError(f"Settings are off ({SETTINGS_ENV}).")
    clean = empty()
    if settings.get("connection") in CONNECTIONS:
        clean["connection"] = settings["connection"]
    if settings.get("runs_in") in RUNS_IN:
        clean["runs_in"] = settings["runs_in"]
    for connection, entry in (settings.get("agent") or {}).items():
        if connection in CONNECTIONS and isinstance(entry, dict):
            model = entry.get("model")
            clean["agent"][connection] = {
                "model": model.strip() if isinstance(model, str) and model.strip() else None,
                "reasoning": entry.get("reasoning") if entry.get("reasoning") in REASONING
                else None,
            }
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary_name = tempfile.mkstemp(prefix=".settings-", dir=path.parent)
    temporary = Path(temporary_name)
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as stream:
            json.dump(clean, stream, indent=1)
            stream.write("\n")
        temporary.replace(path)
    finally:
        temporary.unlink(missing_ok=True)
    return clean


def model_string(connection: str, name: str) -> str:
    """*name* (a model id, or a ``--model`` string already) as
    ``langslice linear run --model`` takes it for *connection*."""
    prefix = _MODEL_PREFIX[connection]
    name = name.strip()
    return name if not prefix or name.startswith(prefix) else prefix + name


def model_name(connection: str, model: str) -> str:
    """A ``--model`` string of *connection* as its bare model id."""
    prefix = _MODEL_PREFIX[connection]
    return model[len(prefix):] if prefix and model.startswith(prefix) else model


def agent_models() -> dict[str, dict[str, Any]]:
    """Per connection, the agent models to offer (bare ids) and LangSlice's
    default (None where LangSlice names none). The ChatGPT account's are
    :mod:`langslice.providers.registry`'s; an API key's are the models
    LangSlice prices (:data:`langslice.agent.spend.PRICES`), and the OpenAI
    API's default is the ChatGPT account's default model."""
    from langslice.agent.spend import PRICES
    from langslice.providers import registry

    oauth = [model_name("openai-oauth", m) for m in registry.OPENAI_OAUTH_AGENT_MODELS]
    oauth_default = model_name("openai-oauth", registry.OPENAI_OAUTH_DEFAULT_AGENT_MODEL)
    out: dict[str, dict[str, Any]] = {
        "openai-oauth": {"models": oauth, "default": oauth_default}}
    for connection in ("openai-api", "anthropic-api", "gemini-api"):
        models = list(PRICES.get(connection, {}))
        default = oauth_default if connection == "openai-api" and oauth_default in models else None
        out[connection] = {"models": models, "default": default}
    return out


def agent_choice(settings: dict[str, Any] | None = None) -> tuple[str, str | None] | None:
    """``(model, reasoning)`` for ``langslice linear run`` from the chosen
    connection's saved agent entry, or None (no file, no connection chosen,
    or no model saved for it)."""
    settings = read() if settings is None else settings
    connection = settings.get("connection")
    if connection not in CONNECTIONS:
        return None
    entry = settings.get("agent", {}).get(connection) or {}
    if not entry.get("model"):
        return None
    return model_string(connection, entry["model"]), entry.get("reasoning")
