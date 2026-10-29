"""The browser sign-in to the ChatGPT subscription and its token file."""

from __future__ import annotations

import base64
import hashlib
import http.server
import json
import secrets
import threading
import time
import urllib.parse
import webbrowser
from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import requests

from langslice.providers.connections import (
    openai_oauth_credentials_path,
)

# --- constants ---------------------------------------------------------------
CLIENT_ID = "app_EMoamEEZ73f0CkXaXp7hrann"
AUTHORIZE_URL = "https://auth.openai.com/oauth/authorize"
TOKEN_URL = "https://auth.openai.com/oauth/token"
ORIGINATOR = "langslice"
REFRESH_SKEW_S = 5 * 60  # refresh when the access token expires within 5 min

# OAuth callback (the Codex client_id only whitelists this redirect).
_REDIRECT_URI = "http://localhost:1455/auth/callback"
_LOGIN_PORT = 1455
_LOGIN_SCOPE = "openid profile email offline_access"


# --- credentials -------------------------------------------------------------
@dataclass
class Creds:
    """A ChatGPT-subscription access token and the metadata needed to use it."""

    access_token: str
    refresh_token: str | None = None
    id_token: str | None = None
    account_id: str | None = None
    source: str = "?"
    raw: dict[str, Any] = field(default_factory=dict, repr=False)


def _decode_jwt_payload(token: str) -> dict[str, Any]:
    """Decode a JWT payload without verifying the signature (client-side only)."""
    payload_b64 = token.split(".")[1]
    payload_b64 += "=" * (-len(payload_b64) % 4)  # fix padding
    decoded = json.loads(base64.urlsafe_b64decode(payload_b64))
    if not isinstance(decoded, dict):
        raise ValueError("JWT payload is not an object")
    return decoded


def _auth_claims(token: str | None) -> dict[str, Any]:
    if not token:
        return {}
    try:
        claims = _decode_jwt_payload(token).get("https://api.openai.com/auth", {})
    except Exception:
        return {}
    return claims if isinstance(claims, dict) else {}


def account_id_from_id_token(id_token: str | None) -> str | None:
    """Return the ChatGPT account id embedded in an OAuth id_token, if present."""
    value = _auth_claims(id_token).get("chatgpt_account_id")
    return value if isinstance(value, str) else None


def _token_expiry(access_token: str) -> float | None:
    try:
        return float(_decode_jwt_payload(access_token)["exp"])
    except Exception:
        return None


def creds_from_doc(doc: dict[str, Any], source: str) -> Creds:
    """Build :class:`Creds` from an ``auth.json``-shaped document.

    Accepts both the nested Codex shape (``{"tokens": {...}}``) and a flat one.
    """
    tokens = doc.get("tokens", doc)
    access = tokens.get("access_token")
    if not access:
        raise ValueError(f"{source}: no access_token in credential document")
    id_token = tokens.get("id_token")
    return Creds(
        access_token=access,
        refresh_token=tokens.get("refresh_token"),
        id_token=id_token,
        account_id=tokens.get("account_id") or account_id_from_id_token(id_token),
        source=source,
        raw=doc,
    )


def _write_creds(creds: Creds) -> None:
    """Persist tokens to LangSlice's login file with owner-only permissions."""
    path = openai_oauth_credentials_path()
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(
            {
                "tokens": {
                    "access_token": creds.access_token,
                    "refresh_token": creds.refresh_token,
                    "id_token": creds.id_token,
                    "account_id": creds.account_id,
                },
                "last_refresh": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            },
            indent=2,
        )
    )
    path.chmod(0o600)


def refresh(creds: Creds) -> Creds:
    """Exchange the refresh token for a fresh access token (JSON body)."""
    if not creds.refresh_token:
        return creds
    resp = requests.post(
        TOKEN_URL,
        json={
            "client_id": CLIENT_ID,
            "grant_type": "refresh_token",
            "refresh_token": creds.refresh_token,
        },
        timeout=30,
    )
    resp.raise_for_status()
    data = resp.json()
    id_token = data.get("id_token") or creds.id_token
    new = Creds(
        access_token=data.get("access_token") or creds.access_token,
        refresh_token=data.get("refresh_token") or creds.refresh_token,
        id_token=id_token,
        account_id=creds.account_id or account_id_from_id_token(id_token),
        source=creds.source,
        raw=creds.raw,
    )
    # Persist only to our own file; never rewrite the Codex CLI's auth.json.
    if creds.source == str(openai_oauth_credentials_path()):
        _write_creds(new)
    return new


def load_credentials() -> Creds:
    """Load subscription credentials, refreshing when near expiry.

    Reads only LangSlice's login file
    (:func:`~langslice.providers.connections.openai_oauth_credentials_path`).
    """
    path = openai_oauth_credentials_path()
    try:
        creds = creds_from_doc(json.loads(path.read_text()), source=str(path))
    except (OSError, ValueError) as exc:
        raise RuntimeError(
            "No ChatGPT-subscription credentials found. Run `langslice login` "
            f"(writes {path}), then retry."
        ) from exc

    expiry = _token_expiry(creds.access_token)
    if expiry is not None and expiry <= time.time() + REFRESH_SKEW_S:
        creds = refresh(creds)
    return creds


# --- "Sign in with ChatGPT" (PKCE OAuth) -------------------------------------
def _pkce() -> tuple[str, str]:
    verifier = base64.urlsafe_b64encode(secrets.token_bytes(64)).rstrip(b"=").decode()
    digest = hashlib.sha256(verifier.encode()).digest()
    challenge = base64.urlsafe_b64encode(digest).rstrip(b"=").decode()
    return verifier, challenge


def _callback_handler(result: dict[str, str]) -> type[http.server.BaseHTTPRequestHandler]:
    class Handler(http.server.BaseHTTPRequestHandler):
        def do_GET(self) -> None:  # noqa: N802 - BaseHTTPRequestHandler API
            parsed = urllib.parse.urlparse(self.path)
            if not parsed.path.startswith("/auth/callback"):
                self.send_response(404)
                self.end_headers()
                return
            result.update(
                {k: v[0] for k, v in urllib.parse.parse_qs(parsed.query).items()}
            )
            self.send_response(200)
            self.send_header("Content-Type", "text/html")
            self.end_headers()
            message = (
                "Signed in. You can close this tab."
                if "code" in result
                else "Sign-in failed."
            )
            self.wfile.write(f"<html><body><h2>{message}</h2></body></html>".encode())

        def log_message(self, *args: object) -> None:
            pass

    return Handler


def login(
    timeout_s: float = 300.0,
    open_browser: bool = True,
    *,
    on_url: Callable[[str], None] | None = None,
) -> Path:
    """Run the "Sign in with ChatGPT" PKCE flow and store the token.

    Returns the path the credentials were written to. Desktop hosts can supply
    ``on_url`` to show the browser URL instead of printing it to stdout.
    """
    verifier, challenge = _pkce()
    state = secrets.token_urlsafe(16)
    url = AUTHORIZE_URL + "?" + urllib.parse.urlencode(
        {
            "response_type": "code",
            "client_id": CLIENT_ID,
            "redirect_uri": _REDIRECT_URI,
            "scope": _LOGIN_SCOPE,
            "code_challenge": challenge,
            "code_challenge_method": "S256",
            "id_token_add_organizations": "true",
            "codex_cli_simplified_flow": "true",
            "state": state,
            "originator": ORIGINATOR,
        }
    )

    result: dict[str, str] = {}
    server = http.server.HTTPServer(("127.0.0.1", _LOGIN_PORT), _callback_handler(result))
    threading.Thread(target=server.serve_forever, daemon=True).start()
    try:
        if on_url is not None:
            on_url(url)
        else:
            print("Opening browser to sign in with ChatGPT...")
            print(f"If it doesn't open, visit:\n  {url}")
        if open_browser:
            try:
                webbrowser.open(url)
            except Exception:
                pass

        deadline = time.time() + timeout_s
        while "code" not in result and time.time() < deadline:
            if "error" in result:
                raise RuntimeError(f"OAuth error: {result.get('error')}")
            time.sleep(0.5)
    finally:
        server.shutdown()
        server.server_close()

    if "code" not in result:
        raise RuntimeError("Timed out waiting for sign-in.")
    if result.get("state") != state:
        raise RuntimeError("OAuth state mismatch - aborting.")

    response = requests.post(
        TOKEN_URL,
        data={
            "grant_type": "authorization_code",
            "code": result["code"],
            "redirect_uri": _REDIRECT_URI,
            "client_id": CLIENT_ID,
            "code_verifier": verifier,
        },
        headers={"Content-Type": "application/x-www-form-urlencoded"},
        timeout=30,
    )
    response.raise_for_status()
    data = response.json()
    id_token = data.get("id_token")
    _write_creds(
        Creds(
            access_token=data["access_token"],
            refresh_token=data.get("refresh_token"),
            id_token=id_token,
            account_id=account_id_from_id_token(id_token),
            source=str(openai_oauth_credentials_path()),
        )
    )
    return openai_oauth_credentials_path()
