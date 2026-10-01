"""What a section looks like: the default appearance, the agent's settings, raw channels.

Every section has a DEFAULT appearance, the tested one: ``preprocess`` auto
(:func:`langslice.image_prep.adaptive_preprocess`) for a plain image file, or
the host's channel blend (:func:`langslice.image_prep.host_preprocess`, the
image ``preprocess.preview`` shows) for snapshots exported one page per
channel. The blend is an appearance, not a lossy step: the raw channels stay
readable (:meth:`langslice.linear.engine.EngineContext.section_channels`).

With ``JobSpec.agent_preprocessing`` the ``preprocess`` tool may replace the
default per TARGET, independently:

- ``"view"`` — every picture the agent is shown (seed, views, write pictures);
- ``"fit"`` — the image a deformable fit reads (:func:`fit_image`).

Each target holds a stack-wide setting and per-section overrides on
``StackState.appearance``, so a change is undone and checkpointed with
everything else. A look that is ``None`` is the default appearance.

Computation never follows the view target: the silhouette fit, calibration,
the tissue pivot and the image model's input (``trace_borders``) all read the
default appearance, so what the agent chooses to look at cannot move a fit.
"""

from __future__ import annotations

import json
import math
from typing import TYPE_CHECKING, Any

from PIL import Image

from langslice.linear.state import SliceState, StackState

if TYPE_CHECKING:  # ponytail: import cycle — engine builds the toolbox
    from langslice.linear.engine import EngineContext

#: The images an appearance can be set for.
TARGETS: tuple[str, ...] = ("view", "fit")
#: ``preprocess``'s ``target`` values.
TARGET_CHOICES: tuple[str, ...] = (*TARGETS, "both")
#: CLAHE clip limit and tile grid of the automatic path.
DEFAULT_CLAHE_CLIP = 4.0
DEFAULT_CLAHE_TILES = 8
#: Accepted CLAHE ranges (0 clip = no CLAHE).
MAX_CLAHE_CLIP = 40.0
MAX_CLAHE_TILES = 32

#: A look: ``None`` (the default appearance), ``{"channel": name}`` (one raw
#: channel, unenhanced) or a settings dict from :func:`validate_settings`.
Look = dict[str, Any] | None


def validate_settings(
    *,
    channel_weights: list[float] | None,
    clahe_clip: float,
    clahe_tiles: int,
    n4: bool,
    denoise: bool,
) -> dict[str, Any]:
    """One appearance as stored, or ``ValueError`` naming what is wrong.

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


def _target(state: StackState, target: str) -> dict[str, Any]:
    return state.appearance.get(target) or {}


def section_settings(state: StackState, target: str, section_id: str) -> Look:
    """The look *target* uses for one section: its override, else the stack's."""
    held = _target(state, target)
    override = (held.get("sections") or {}).get(section_id)
    if override is not None:
        return dict(override)
    stack = held.get("stack")
    return dict(stack) if stack is not None else None


def set_settings(
    state: StackState, target: str, section_ids: list[str] | None, settings: Look,
) -> None:
    """Write *settings* (None = back to the default) for the stack or sections.

    Stack-wide writes leave per-section overrides in place; a section reset
    removes its override, so it follows the stack again.
    """
    if target not in TARGETS:
        raise ValueError(f"target must be one of {TARGETS}")
    held = {**_target(state, target)}
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
        state.appearance.pop(target, None)
    else:
        state.appearance[target] = held


def look_token(ctx: EngineContext, look: Look) -> str:
    """The render-cache slot of a look; the default keeps ``spec.preprocess``."""
    if look is None:
        return str(ctx.spec.preprocess)
    return json.dumps(look, sort_keys=True)


def view_look(state: StackState, record: SliceState, section_image: str = "current") -> Look:
    """What a picture of *record* shows: its view appearance or one raw channel."""
    if section_image and section_image != "current":
        return {"channel": section_image}
    return section_settings(state, "view", record.id)


def fit_image(
    ctx: EngineContext, state: StackState, record: SliceState, *, long_edge: int,
) -> Image.Image:
    """The section as a deformable fit reads it: the fit appearance, unframed.

    Same frame as :func:`langslice.registration_handoff.linear_registration_input`
    (oriented, not tissue-framed) at *long_edge*.
    """
    from langslice.linear.render import render_slice

    return render_slice(
        ctx, record, long_edge=long_edge, frame=False,
        look=section_settings(state, "fit", record.id),
    )


def describe(look: Look) -> str:
    """A short caption fragment for a look."""
    if look is None:
        return "default appearance"
    if "channel" in look:
        return f"raw {look['channel']}"
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
