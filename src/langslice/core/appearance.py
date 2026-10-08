"""What a section looks like: its preprocessed channel.

Every section has exactly ONE preprocessed channel, derived from its raw
channels, and it is what every algorithm and the image model read: the
Elastix affine and every deformable fit (``core.deformation.stain_image``)
and the image-model routes (``core.nonlinear.registration_tool``, through
:func:`langslice.core.handoff.prepare_linear_registration` with
``preprocessed=True``). Its DEFAULT is the tested appearance: ``preprocess``
auto (:func:`langslice.core.image_prep.adaptive_preprocess`) for a plain
image file, or the host's channel blend
(:func:`langslice.core.image_prep.host_preprocess`, the image
``preprocess.preview`` shows) for snapshots exported one page per channel. A
recipe (:func:`validate_settings`: channel weights, CLAHE, N4, denoising)
replaces the default, stack-wide or per section; ``None`` is the default.
The raw channels stay readable and are never edited
(:meth:`langslice.core.workspace.Workspace.section_channels`); how a raw
channel is DISPLAYED is :mod:`langslice.core.channels`.

``StackState.appearance`` holds, so a change is undone and checkpointed with
everything else:

- ``"preprocessed"``: ``{"stack": recipe or None, "sections": {id: recipe}}``;
- ``"channels"``: the raw channels' display properties, by channel name
  (:mod:`langslice.core.channels`).

A state saved before the preprocessed channel held its recipe under
``"fit"``; :func:`migrated` moves it to ``"preprocessed"``, and every reader
here reads it there too. An older saved ``"view"`` look is dropped.

Calibration and the tissue pivot measure geometry on the default render
whatever the recipe, so neither a recipe nor a display property can move
them.
"""

from __future__ import annotations

import json
import math
from typing import Any

from PIL import Image

from langslice.core.state import SliceState, StackState
from langslice.core.workspace import Workspace

#: The preprocessed channel's key on ``StackState.appearance``.
PREPROCESSED = "preprocessed"
#: The key a state saved before the preprocessed channel held it under (also
#: its name as a fit reads it, ``ops.inputs``).
FIT = "fit"
#: A view look an older saved state may hold; dropped on reading.
_OLD_VIEW = "view"
#: Where each name is held on ``StackState.appearance``.
HELD_UNDER: dict[str, str] = {FIT: PREPROCESSED, PREPROCESSED: PREPROCESSED}
#: CLAHE clip limit and tile grid of the automatic path.
DEFAULT_CLAHE_CLIP = 4.0
DEFAULT_CLAHE_TILES = 8
#: Accepted CLAHE ranges (0 clip = no CLAHE).
MAX_CLAHE_CLIP = 40.0
MAX_CLAHE_TILES = 32

#: A look: ``None`` (the default appearance), ``{"channel": name}`` (one raw
#: channel, unenhanced: the `channels` strip), ``{"overlay": [names]}`` (raw
#: channels, each stretched by percentile; one in gray, several added in their
#: colours, :func:`channel_colors`), either with ``"properties"`` (their
#: display properties, :func:`langslice.core.channels.with_properties`), or
#: a settings dict from :func:`validate_settings`.
Look = dict[str, Any] | None

#: Colours an overlay gives its channels, in order. A channel NAMED after a
#: colour (an RGB file's ``red``/``green``/``blue``) keeps that colour, so
#: the three planes of a colour image overlay back into it.
OVERLAY_PALETTE: tuple[tuple[str, tuple[int, int, int]], ...] = (
    ("green", (0, 255, 0)),
    ("magenta", (255, 0, 255)),
    ("cyan", (0, 255, 255)),
    ("yellow", (255, 255, 0)),
    ("red", (255, 0, 0)),
    ("blue", (0, 96, 255)),
)
NAMED_COLORS: dict[str, tuple[int, int, int]] = {
    "red": (255, 0, 0), "green": (0, 255, 0), "blue": (0, 0, 255),
}
#: Percentiles an overlay maps to black and white, per channel.
OVERLAY_STRETCH = (1.0, 99.5)


def channel_colors(names: Any) -> list[tuple[str, str, tuple[int, int, int]]]:
    """``(name, colour word, rgb)`` per channel of an overlay, in order."""
    out: list[tuple[str, str, tuple[int, int, int]]] = []
    taken = {str(name).lower() for name in names if str(name).lower() in NAMED_COLORS}
    palette = [entry for entry in OVERLAY_PALETTE if entry[0] not in taken]
    for name in names:
        key = str(name).lower()
        if key in NAMED_COLORS:
            out.append((str(name), key, NAMED_COLORS[key]))
        else:
            word, rgb = palette.pop(0) if palette else ("white", (255, 255, 255))
            out.append((str(name), word, rgb))
    return out


def validate_settings(
    *,
    channel_weights: list[float] | None,
    clahe_clip: float,
    clahe_tiles: int,
    n4: bool,
    denoise: bool,
) -> dict[str, Any]:
    """One recipe as stored, or ``ValueError`` naming what is wrong.

    *channel_weights* None (or empty) means automatic weights (squared tissue
    coverage, as the default appearance weighs channels).
    """
    weights: list[float] | None = None
    if channel_weights:
        try:
            weights = [float(value) for value in channel_weights]
        except (TypeError, ValueError):
            raise ValueError("channel_weights must be numbers") from None
        if any(not math.isfinite(value) or value < 0 for value in weights):
            raise ValueError("channel_weights must be finite and non-negative")
        if sum(weights) <= 0:
            raise ValueError("at least one channel weight must be above zero")
    try:
        clip = float(clahe_clip)
        tiles = int(clahe_tiles)
    except (TypeError, ValueError):
        raise ValueError("clahe_clip must be a number and clahe_tiles an integer") from None
    if not math.isfinite(clip) or not 0.0 <= clip <= MAX_CLAHE_CLIP:
        raise ValueError(f"clahe_clip must be from 0 (off) to {MAX_CLAHE_CLIP:g}")
    if not 1 <= tiles <= MAX_CLAHE_TILES:
        raise ValueError(f"clahe_tiles must be from 1 to {MAX_CLAHE_TILES}")
    if not isinstance(n4, bool) or not isinstance(denoise, bool):
        raise ValueError("n4 and denoise must be true or false")
    return {
        "channel_weights": weights,
        "clahe_clip": clip,
        "clahe_tiles": tiles,
        "n4": n4,
        "denoise": denoise,
    }


def migrated(appearance: Any) -> dict[str, Any]:
    """A saved state's ``appearance`` as this version holds it.

    ``"fit"`` (a state saved before the preprocessed channel) becomes
    ``"preprocessed"`` unless that is there already; ``"view"`` is dropped;
    everything else is kept. The input is not changed.
    """
    if not isinstance(appearance, dict):
        return {}
    out = {key: value for key, value in appearance.items() if key not in (_OLD_VIEW, FIT)}
    if PREPROCESSED not in out and appearance.get(FIT) is not None:
        out[PREPROCESSED] = appearance[FIT]
    return out


def _held(state: StackState, target: str) -> dict[str, Any]:
    if target not in HELD_UNDER:
        raise ValueError(f"target must be one of {tuple(HELD_UNDER)}")
    key = HELD_UNDER[target]
    held = state.appearance.get(key)
    if held is None and key == PREPROCESSED:
        held = state.appearance.get(FIT)
    return held or {}


def section_settings(state: StackState, target: str, section_id: str) -> Look:
    """The recipe *target* (``"preprocessed"``, or its older name ``"fit"``)
    uses for one section: its override, else the stack's."""
    held = _held(state, target)
    override = (held.get("sections") or {}).get(section_id)
    if override is not None:
        return dict(override)
    stack = held.get("stack")
    return dict(stack) if stack is not None else None


def preprocessed_settings(state: StackState, section_id: str) -> Look:
    """The section's preprocessed-channel recipe (None: the default)."""
    return section_settings(state, PREPROCESSED, section_id)


def set_settings(
    state: StackState, target: str, section_ids: list[str] | None, settings: Look,
) -> None:
    """Write *settings* (None = back to the default) for the stack or sections.

    Stack-wide writes leave per-section overrides in place; a section reset
    removes its override, so it follows the stack again.
    """
    held = {**_held(state, target)}
    key = HELD_UNDER[target]
    if key == PREPROCESSED:
        state.appearance.pop(FIT, None)  # now held under its own name
    sections = dict(held.get("sections") or {})
    if section_ids is None:
        held["stack"] = settings
    else:
        for name in section_ids:
            if settings is None:
                sections.pop(name, None)
            else:
                sections[name] = dict(settings)
    held["sections"] = sections
    if held.get("stack") is None and not sections:
        state.appearance.pop(key, None)
    else:
        state.appearance[key] = held


def look_token(ctx: Workspace, look: Look) -> str:
    """The render-cache slot of a look; the default keeps ``spec.preprocess``."""
    if look is None:
        return str(ctx.spec.preprocess)
    return json.dumps(look, sort_keys=True)


def preprocessed_image(
    ctx: Workspace, state: StackState, record: SliceState, *, long_edge: int,
) -> Image.Image:
    """The section's preprocessed channel, unframed: what the fits read.

    Same frame as :func:`langslice.core.handoff.prepare_linear_registration`
    (oriented, not tissue-framed) at *long_edge*; with the default recipe it
    is that function's render exactly.
    """
    from langslice.core.sections import render_slice

    return render_slice(ctx, record, long_edge=long_edge, frame=False,
                        look=preprocessed_settings(state, record.id))


def describe(look: Look) -> str:
    """A short caption fragment for a look."""
    if look is None:
        return "default appearance"
    if "channel" in look:
        return f"raw {look['channel']}"
    if "overlay" in look:
        if len(look["overlay"]) == 1:
            return f"raw {look['overlay'][0]}, stretched"
        return "raw " + " + ".join(name if name.lower() == word else f"{name} {word}"
                                   for name, word, _rgb in channel_colors(look["overlay"]))
    weights = look.get("channel_weights")
    parts = [
        "weights " + ("auto" if not weights else "/".join(f"{w:g}" for w in weights)),
        f"CLAHE {look['clahe_clip']:g}x{look['clahe_tiles']}" if look["clahe_clip"] else "no CLAHE",
    ]
    if look.get("n4"):
        parts.append("N4")
    if look.get("denoise"):
        parts.append("denoise")
    return ", ".join(parts)
