"""Local setup operations for desktop hosts; responses never contain credentials.

Configuration status is deliberately offline and is not an account validation.
Saved API keys are loaded explicitly by the worker before running model tasks.
"""

from __future__ import annotations

import json
import os
import sys
import tempfile
from collections.abc import Callable
from pathlib import Path
from typing import Any

from langslice import __version__

PROTOCOL_VERSION = 1
_KEY_ENV = {
    "openai-api": ("OPENAI_API_KEY",),
    "gemini-api": ("GEMINI_API_KEY", "GOOGLE_API_KEY"),
}


def _credentials_path() -> Path:
    return Path.home() / ".langslice" / "provider_credentials.json"


def _read_keys() -> dict[str, str]:
    path = _credentials_path()
    if not path.exists():
        return {}
    try:
        doc = json.loads(path.read_text(encoding="utf-8"))
        if not isinstance(doc, dict) or any(
            provider not in _KEY_ENV or not isinstance(key, str) or not key.strip()
            for provider, key in doc.items()
        ):
            raise ValueError
        return doc
    except (OSError, ValueError) as exc:
        raise ValueError(
            "Saved API-key settings could not be read. Check the settings file."
        ) from exc


def _api_status(provider: str, keys: dict[str, str]) -> dict[str, object]:
    source = next((name for name in _KEY_ENV[provider] if os.getenv(name, "").strip()), None)
    return {
        "configured": bool(source or keys.get(provider)),
        "source": "environment" if source else "saved" if keys.get(provider) else None,
        "validated": False,
    }


def _oauth_path() -> Path:
    """The login file ``providers.openai_oauth`` reads: ``LANGSLICE_OPENAI_AUTH``
    when set (another account for one process), else LangSlice's own."""
    override = os.environ.get("LANGSLICE_OPENAI_AUTH", "").strip()
    return Path(override).expanduser() if override else (
        Path.home() / ".langslice" / "openai_auth.json")


def _oauth_status() -> dict[str, object]:
    # Do not call load_credentials: status must never refresh tokens or contact a provider.
    try:
        path = _oauth_path()
        doc = json.loads(path.read_text(encoding="utf-8"))
        tokens = doc.get("tokens", doc)
        if isinstance(tokens, dict) and isinstance(tokens.get("access_token"), str):
            if tokens["access_token"].strip():
                return {"configured": True, "source": "saved", "validated": False}
    except (OSError, ValueError, AttributeError):
        pass
    return {"configured": False, "source": None, "validated": False}


def _oauth_models() -> dict[str, object]:
    """The agent and image models a host may offer for the ChatGPT account."""
    from langslice.providers import registry

    return {
        "agent_models": list(registry.OPENAI_OAUTH_AGENT_MODELS),
        "default_agent_model": registry.OPENAI_OAUTH_DEFAULT_AGENT_MODEL,
        "image_models": list(registry.OPENAI_OAUTH_IMAGE_MODELS),
        "default_image_model": registry.OPENAI_OAUTH_DEFAULT_IMAGE_MODEL,
    }


def setup_status() -> dict[str, Any]:
    """Return installation and credential presence, without network access or secrets."""
    error = None
    try:
        keys = _read_keys()
    except ValueError as exc:
        keys = {}
        error = str(exc)
    providers = {provider: _api_status(provider, keys) for provider in _KEY_ENV}
    providers["openai-oauth"] = {**_oauth_status(), **_oauth_models()}
    providers["none"] = {"configured": True, "source": None, "validated": False}
    return {
        "version": __version__,
        "protocol_version": PROTOCOL_VERSION,
        "python_executable": sys.executable,
        "environment_prefix": sys.prefix,
        "providers": providers,
        "credentials_error": error,
    }


def image_model_connected(provider: str) -> bool:
    """Whether LangSlice can reach *provider*'s image model now: the provider
    (any accepted spelling) is not ``none`` and its key or login is present.

    The offline presence check of :func:`setup_status` (a key in the
    environment or saved by setup, the ChatGPT login file; for
    ``openai-api`` also a custom endpoint, ``OPENAI_IMAGE_BASE_URL`` or
    ``OPENAI_BASE_URL``): nothing is validated, no token refreshed, no
    provider contacted. The MCP door asks it before offering ``trace_borders``.
    """
    from langslice.core.provider_names import canonical_provider

    canonical = canonical_provider(str(provider or ""))
    if canonical == "openai-oauth":
        return bool(_oauth_status()["configured"])
    if canonical not in _KEY_ENV:
        return False
    if canonical == "openai-api" and any(
            os.getenv(name, "").strip() for name in (
                "OPENAI_IMAGE_API_KEY", "OPENAI_IMAGE_BASE_URL", "OPENAI_BASE_URL")):
        return True
    try:
        keys = _read_keys()
    except ValueError:
        keys = {}
    return bool(_api_status(canonical, keys)["configured"])


def save_api_key(provider: str, api_key: str) -> dict[str, object]:
    """Atomically save a supported provider key with owner-only file permissions."""
    if provider not in _KEY_ENV:
        raise ValueError("API-key setup supports openai-api and gemini-api only.")
    key = api_key.strip()
    if not key or any(character.isspace() or ord(character) < 32 for character in key):
        raise ValueError("Enter a nonempty API key without whitespace or control characters.")
    keys = _read_keys()
    keys[provider] = key
    path = _credentials_path()
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary_name = tempfile.mkstemp(prefix=".provider_credentials-", dir=path.parent)
    temporary = Path(temporary_name)
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as stream:
            json.dump(keys, stream)
            stream.write("\n")
            stream.flush()
            os.fsync(stream.fileno())
        # mkstemp creates mode 600 on POSIX, before writing any secret.
        temporary.replace(path)
    finally:
        temporary.unlink(missing_ok=True)
    return _api_status(provider, keys)


def apply_saved_credentials() -> None:
    """Load saved keys into this worker, respecting explicit environment settings."""
    for provider, key in _read_keys().items():
        names = _KEY_ENV[provider]
        if not any(os.getenv(name, "").strip() for name in names):
            os.environ[names[0]] = key
            if provider == "openai-api" and not os.getenv("OPENAI_BASE_URL", "").strip():
                # GUI setup names the OpenAI provider. The legacy configuration
                # defaults to a local compatible server, which is inappropriate
                # for an OpenAI key saved by this dialog.
                os.environ["OPENAI_BASE_URL"] = "https://api.openai.com/v1"


def load_dotenv() -> None:
    """Read a ``.env`` file's keys (GEMINI_API_KEY, OPENAI_API_KEY, ...) into
    the environment when python-dotenv is installed; every lane reads them."""
    import importlib

    try:
        importlib.import_module("dotenv").load_dotenv()
    except ImportError:
        pass


def load_credentials(*, saved: bool = True) -> None:
    """The one place a door loads the model keys: ``.env``, then with
    *saved* the keys saved by setup (:func:`apply_saved_credentials`; an
    explicit environment setting wins). The CLI calls it per command, the
    library on :func:`langslice.open_job`."""
    load_dotenv()
    if saved:
        apply_saved_credentials()


def login_oauth(
    on_url: Callable[[str], None] | None = None, timeout_s: float = 300.0
) -> dict[str, object]:
    """Run the existing browser login; hosts receive a URL without receiving tokens."""
    from langslice.providers.openai_oauth import login

    login(timeout_s=timeout_s, on_url=on_url)
    return _oauth_status()
