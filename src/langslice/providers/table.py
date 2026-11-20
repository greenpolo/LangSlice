"""The provider table: what each provider is, in one place.

A provider is an ACCESS METHOD, a vendor plus how you authenticate, never a
product name. :data:`PROVIDERS` holds one :class:`Provider` per access method
with everything any reader needs: its label in the window, the sign-in or the
environment variables and key file that make it connected, the prefix of its
``--model`` strings, the agent and image models it offers with their defaults,
and the list price of each model. Every other place (the window's Settings,
the saved-settings reader, the key setup, the model resolver, the usage meter,
the image-model resolver) reads this table and keeps no list of its own.

A new provider is one entry here, its name in :mod:`langslice.core.connections`
if it can make images (checked when this module is imported), and its
transport in this package.

``none`` (no image model) is not a provider and is not in the table.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from datetime import date
from pathlib import Path

from langslice.core.connections import CANONICAL_PROVIDERS, canonical_provider


@dataclass(frozen=True)
class Rate:
    """USD per million tokens. ``cache_write`` is what an input token that is
    not a cache read costs when the provider bills it as a cache write
    (None: it is plain input)."""

    input: float
    cached: float
    output: float
    cache_write: float | None = None


@dataclass(frozen=True)
class ModelPrice:
    """The list price of one model: its rates by the date they start
    (``since``, oldest first; the first row applies before any date), and
    optionally another rate for a request whose prompt passes ``long_over``
    tokens."""

    rates: tuple[tuple[date, Rate], ...]
    long_over: int | None = None
    long_rate: Rate | None = None

    def rate(self, *, today: date, prompt_tokens: int = 0) -> Rate:
        """The rate on *today* for a request of *prompt_tokens*."""
        if self.long_rate is not None and self.long_over is not None:
            if prompt_tokens > self.long_over:
                return self.long_rate
        current = [
            rate for since, rate in sorted(self.rates, key=lambda row: row[0]) if since <= today
        ]
        return current[-1] if current else self.rates[0][1]


@dataclass(frozen=True)
class Provider:
    """One access method and everything the program says about it.

    Attributes
    ----------
    name
        The canonical name (``openai-oauth``, ``openai-api``,
        ``anthropic-api``, ``gemini-api``).
    label
        What the window's Settings calls it.
    signin
        True when a login makes it connected (``langslice login``); False when
        an API key does.
    key_env
        The environment variables that hold its API key, the first being
        the one saved keys are loaded into.
    image_key_env
        Environment variables that connect its image model beyond ``key_env``.
    key_file
        A file under ``~/.langslice`` that also holds its key, or None.
    model_prefix
        What its ``--model`` strings start with (``""``: none).
    thoughts_separate
        Whether its usage reports thinking tokens apart from the answer's, so
        they are billed as output on top of it.
    agent_models, agent_default
        The agent models offered (bare ids) and the default, None for none.
    images
        Whether it can make images for the border trace.
    image_models, image_default
        The image models offered and the one used when none is named.
    image_env
        An environment variable that names the image model to use when none
        is named, ahead of ``image_default``; None for none.
    prices
        The list price of each priced model (bare id); empty when the provider
        is not billed per token here.
    """

    name: str
    label: str
    signin: bool = False
    key_env: tuple[str, ...] = ()
    image_key_env: tuple[str, ...] = ()
    key_file: str | None = None
    model_prefix: str = ""
    thoughts_separate: bool = False
    agent_models: tuple[str, ...] = ()
    agent_default: str | None = None
    images: bool = False
    image_models: tuple[str, ...] = ()
    image_default: str | None = None
    image_env: str | None = None
    prices: dict[str, ModelPrice] = field(default_factory=dict)

    @property
    def kind(self) -> str:
        """``signin`` or ``key``: how the window asks for its credential."""
        return "signin" if self.signin else "key"

    def key_file_path(self) -> Path | None:
        """Where its key file is, read at call time."""
        return None if self.key_file is None else Path.home() / ".langslice" / self.key_file

    def model_string(self, name: str) -> str:
        """*name* (a model id, or a ``--model`` string already) as
        ``langslice linear run --model`` takes it."""
        name = name.strip()
        prefix = self.model_prefix
        return name if not prefix or name.startswith(prefix) else prefix + name

    def model_name(self, model: str) -> str:
        """A ``--model`` string of this provider as its bare model id."""
        prefix = self.model_prefix
        return model[len(prefix) :] if prefix and model.startswith(prefix) else model

    def default_image_model(self) -> str | None:
        """The image model to name when none is given: None when an
        environment variable (``image_env``) decides it at call time."""
        return None if self.image_env else self.image_default

    def image_model_or_default(self, named: str | None = None) -> str:
        """The image model to call: *named*, else the one its environment
        variable names, else ``image_default``. ``ValueError`` for a provider
        with no image model."""
        env = os.environ.get(self.image_env, "").strip() if self.image_env else ""
        chosen = named or env or self.image_default
        if not chosen:
            raise ValueError(f"{self.name} has no image model")
        return chosen

    def price(self, model: str) -> ModelPrice | None:
        """The price of *model*, or None when the table has none."""
        return self.prices.get(model)


_ALWAYS = date.min


def _rate(input: float, cached: float, output: float, write: float | None = None) -> ModelPrice:
    return ModelPrice(((_ALWAYS, Rate(input, cached, output, write)),))


def _dated(*rows: tuple[date, float, float, float]) -> ModelPrice:
    return ModelPrice(tuple((since, Rate(i, c, o)) for since, i, c, o in rows))


_OAUTH_AGENT_MODELS = (
    "gpt-6.1-sol",
    "gpt-6-astra",
    "gpt-6-sol",
    "gpt-6-luna",
    "gpt-5.6-sol",
    "gpt-5.6-terra",
    "gpt-5.6-luna",
)
_OPENAI_IMAGE_MODELS = ("gpt-image-2.5-sunburst", "gpt-image-2.5-flare")

_ANTHROPIC_PRICES = {
    "claude-fable-5-1": _rate(10, 0.25, 50, 12.5),
    "claude-fable-5": _rate(10, 1, 50, 12.5),
    "claude-opus-5-5": _rate(4, 0.20, 20, 5),
    "claude-opus-5": _rate(5, 0.50, 25, 6.25),
    "claude-opus-4-8": _rate(5, 0.50, 25, 6.25),
    "claude-opus-4-7": _rate(5, 0.50, 25, 6.25),
    "claude-opus-4-6": _rate(5, 0.50, 25, 6.25),
    "claude-sonnet-5-5": _rate(2, 0.10, 10, 2.5),
    "claude-sonnet-5": _rate(2, 0.20, 10, 2.5),
    "claude-sonnet-4-6": _rate(3, 0.30, 15, 3.75),
    "claude-haiku-5-5": ModelPrice(
        ((_ALWAYS, Rate(0.10, 0.01, 0.50, 0.125)),),
        long_over=100_000,
        long_rate=Rate(0.50, 0.05, 2.5, 0.625),
    ),
    "claude-haiku-4-5": _rate(1, 0.10, 5, 1.25),
}
_OPENAI_PRICES = {
    "gpt-6-astra": _rate(10, 1, 50),
    "gpt-6.1-sol": _rate(2, 0.10, 10),
    "gpt-6-sol": _rate(2, 0.20, 10),
    "gpt-6-luna": _rate(0.10, 0.01, 0.50),
    "gpt-5.6-sol": _rate(4, 0.40, 20),
    "gpt-5.6-terra": _rate(2, 0.20, 12),
    "gpt-5.6-luna": _rate(0.20, 0.02, 1.20),
    "gpt-5.5": _rate(5, 0.50, 30),
    "gpt-5.4": _rate(2.5, 0.25, 15),
    "gpt-5.4-mini": _rate(0.75, 0.075, 4.5),
    "gpt-5.2": _rate(1.75, 0.175, 14),
    "gpt-5.1": _rate(1.25, 0.125, 10),
    "gpt-5": _rate(1.25, 0.125, 10),
}
_GEMINI_PRICES = {
    "gemini-4-argon": _dated(
        (_ALWAYS, 2, 0.10, 10),
        (date(2026, 12, 1), 4, 0.20, 20),
    ),
    "gemini-3.8-flash": _dated(
        (_ALWAYS, 0.75, 0.075, 3.75),
        (date(2027, 1, 1), 1.5, 0.15, 7.5),
    ),
    "gemini-3.6-flash": _dated(
        (_ALWAYS, 0.75, 0.075, 3.75),
        (date(2027, 1, 1), 1.5, 0.15, 7.5),
    ),
    "gemini-3.5-flash-lite": _rate(0.30, 0.03, 2.5),
    "gemini-3.1-flash-lite": _rate(0.25, 0.025, 1.5),
    "gemini-3.1-pro-preview": _rate(2, 0.20, 12),
    "gemini-3-flash-preview": _rate(0.50, 0.05, 3),
    "gemini-2.5-pro": _rate(1.25, 0.125, 10),
    "gemini-2.5-flash": _rate(0.30, 0.03, 2.5),
    "gemini-2.5-flash-lite": _rate(0.10, 0.01, 0.4),
}

PROVIDERS: tuple[Provider, ...] = (
    Provider(
        name="openai-oauth",
        label="ChatGPT account",
        signin=True,
        model_prefix="openai-oauth/",
        agent_models=_OAUTH_AGENT_MODELS,
        agent_default=_OAUTH_AGENT_MODELS[0],
        images=True,
        image_models=_OPENAI_IMAGE_MODELS,
        image_default=_OPENAI_IMAGE_MODELS[0],
    ),
    Provider(
        name="openai-api",
        label="OpenAI API key",
        key_env=("OPENAI_API_KEY",),
        image_key_env=("OPENAI_IMAGE_API_KEY",),
        agent_models=tuple(_OPENAI_PRICES),
        agent_default=_OAUTH_AGENT_MODELS[0],
        images=True,
        image_models=_OPENAI_IMAGE_MODELS,
        image_default=_OPENAI_IMAGE_MODELS[0],
        image_env="OPENAI_IMAGE_MODEL",
        prices=_OPENAI_PRICES,
    ),
    Provider(
        name="anthropic-api",
        label="Claude (Anthropic) API key",
        key_env=("ANTHROPIC_API_KEY",),
        key_file="anthropic_api_key",
        model_prefix="anthropic-api/",
        thoughts_separate=True,
        agent_models=tuple(_ANTHROPIC_PRICES),
        prices=_ANTHROPIC_PRICES,
    ),
    Provider(
        name="gemini-api",
        label="Gemini API key",
        key_env=("GEMINI_API_KEY", "GOOGLE_API_KEY"),
        model_prefix="gemini-api/",
        thoughts_separate=True,
        agent_models=tuple(_GEMINI_PRICES),
        images=True,
        image_models=(
            "gemini-3.1-flash-image",
            "gemini-3.1-flash-lite-image",
            "gemini-3-pro-image",
        ),
        image_default="gemini-3.1-flash-image",
        prices=_GEMINI_PRICES,
    ),
)

if {p.name for p in PROVIDERS if p.images} | {"none"} != set(CANONICAL_PROVIDERS):
    raise RuntimeError(
        "langslice.core.connections.CANONICAL_PROVIDERS and the image providers of "
        "langslice.providers.table.PROVIDERS name different providers"
    )

IMAGE_PROVIDERS: tuple[Provider, ...] = tuple(p for p in PROVIDERS if p.images)
KEY_PROVIDERS: tuple[Provider, ...] = tuple(p for p in PROVIDERS if p.key_env)


def find(name: str) -> Provider | None:
    """The provider called *name* (trimmed, lower-cased), or None."""
    wanted = canonical_provider(name)
    return next((p for p in PROVIDERS if p.name == wanted), None)


def provider(name: str) -> Provider:
    """The provider called *name*; ``ValueError`` when there is none."""
    found = find(name)
    if found is None:
        raise ValueError(f"Unknown provider: {name}")
    return found


__all__ = [
    "IMAGE_PROVIDERS",
    "KEY_PROVIDERS",
    "PROVIDERS",
    "ModelPrice",
    "Provider",
    "Rate",
    "find",
    "provider",
]
