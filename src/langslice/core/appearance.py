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

- ``"derived"``: ``{name: {"stack": recipe or None, "sections": {id: recipe}}}``,
  the derived channels by name (``preprocessed`` is one: LangSlice's default
  recipe when its recipe is None; any other name is a channel the agent made).
  A recipe is ``{"weights": {raw name: weight} or None, "clahe_clip",
  "clahe_tiles", "n4", "denoise", "contrast_limits": [lo, hi] or None (fractions
  of the blended image, 0 to 1), "gamma": float or None, "invert": bool}``
  (:func:`validate_recipe`); a recipe an older job wrote, with
  ``"channel_weights"`` (one weight per raw channel, by position) and no
  tone steps, is read as it is;
- ``"preprocessed"``: the older home of ``derived["preprocessed"]``, read by
  :func:`migrate` and never written again;
- ``"channels"``: the raw channels' display properties, by channel name
  (:mod:`langslice.core.channels`).

Calibration measures geometry on the default render whatever the recipe,
so neither a recipe nor a display property can move it.
"""

from __future__ import annotations

import json
import math
from typing import Any

import numpy as np
from PIL import Image

from langslice.core.image_prep import HOST_CLAHE_CLIP, HOST_CLAHE_TILE
from langslice.core.state import SliceState, StackState
from langslice.core.workspace import Workspace

#: The preprocessed channel's name (and its older key on ``StackState.appearance``).
PREPROCESSED = "preprocessed"
#: The derived channels' key on ``StackState.appearance``.
DERIVED = "derived"
#: Longest name a derived channel may have.
MAX_DERIVED_NAME = 32
#: Accepted gamma range of a recipe (as :mod:`langslice.core.channels`).
MIN_RECIPE_GAMMA, MAX_RECIPE_GAMMA = 0.1, 10.0
#: CLAHE clip limit and tile grid of the automatic path.
DEFAULT_CLAHE_CLIP = HOST_CLAHE_CLIP["medium"]
DEFAULT_CLAHE_TILES = HOST_CLAHE_TILE[0]
#: Accepted CLAHE ranges (0 clip = no CLAHE).
MAX_CLAHE_CLIP = 40.0
MAX_CLAHE_TILES = 32

#: A look: ``None`` (the default appearance), ``{"channel": name}`` (one raw
#: channel, unenhanced: the `channels` strip), ``{"overlay": [names]}`` (raw
#: channels, each between its automatic contrast limits; one in gray, several
#: added in their colours, :func:`channel_colors`), either with ``"properties"`` (their
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
#: Percentiles an overlay maps to black and white, per channel, on a light
#: background (:func:`langslice.core.channels.auto_limits`).
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


def validate_recipe(
    *,
    weights: dict[str, float] | None = None,
    clahe_clip: float = DEFAULT_CLAHE_CLIP,
    clahe_tiles: int = DEFAULT_CLAHE_TILES,
    n4: bool = False,
    denoise: bool = False,
    contrast_limits: list[float] | tuple[float, ...] | None = None,
    gamma: float | None = None,
    invert: bool = False,
) -> dict[str, Any]:
    """One derived-channel recipe as stored, or ``ValueError`` naming what is wrong.

    *weights* ``{raw channel name: weight}``; None (or empty) means automatic
    weights (squared tissue coverage, as the default appearance weighs
    channels). *contrast_limits* ``[lo, hi]`` and *gamma* stretch the blended
    image (fractions of its range, 0 to 1); None leaves it as blended.
    """
    held: dict[str, float] | None = None
    if weights:
        if not isinstance(weights, dict):
            raise ValueError("weights must be {channel name: number}")
        try:
            held = {str(name): float(value) for name, value in weights.items()}
        except (TypeError, ValueError):
            raise ValueError("weights must be numbers") from None
        if any(not math.isfinite(value) or value < 0 for value in held.values()):
            raise ValueError("weights must be finite and non-negative")
        if sum(held.values()) <= 0:
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
    if not isinstance(n4, bool) or not isinstance(denoise, bool) or not isinstance(invert, bool):
        raise ValueError("n4, denoise and invert must be true or false")
    limits: list[float] | None = None
    if contrast_limits:
        try:
            limits = [float(value) for value in contrast_limits]
        except (TypeError, ValueError):
            raise ValueError("contrast_limits must be two numbers") from None
        if (len(limits) != 2 or not all(math.isfinite(v) for v in limits)
                or not 0.0 <= limits[0] < limits[1] <= 1.0):
            raise ValueError("contrast_limits are [low, high] with 0 <= low < high <= 1 "
                             "(fractions of the blended image)")
    power: float | None = None
    if gamma:
        try:
            power = float(gamma)
        except (TypeError, ValueError):
            raise ValueError("gamma must be a number") from None
        if not math.isfinite(power) or not MIN_RECIPE_GAMMA <= power <= MAX_RECIPE_GAMMA:
            raise ValueError(f"gamma must be from {MIN_RECIPE_GAMMA:g} to {MAX_RECIPE_GAMMA:g}")
        if power == 1.0:
            power = None
    return {
        "weights": held,
        "clahe_clip": clip,
        "clahe_tiles": tiles,
        "n4": n4,
        "denoise": denoise,
        "contrast_limits": limits,
        "gamma": power,
        "invert": invert,
    }


def validate_settings(
    *,
    channel_weights: list[float] | None,
    clahe_clip: float,
    clahe_tiles: int,
    n4: bool,
    denoise: bool,
) -> dict[str, Any]:
    """The positional recipe older callers write (weights one per raw channel, by
    position), or ``ValueError``. Read-compatible with :func:`validate_recipe`'s."""
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
    base = validate_recipe(clahe_clip=clahe_clip, clahe_tiles=clahe_tiles, n4=n4,
                           denoise=denoise)
    return {"channel_weights": weights, "clahe_clip": base["clahe_clip"],
            "clahe_tiles": base["clahe_tiles"], "n4": n4, "denoise": denoise}


def weights_for(recipe: dict[str, Any], names: list[str]) -> list[float] | None:
    """The blend weights of *recipe* over a section's raw channel *names*, one per
    name (a name the recipe does not weigh counts zero); None for automatic
    weights. ``ValueError`` when none of the weighed channels exists."""
    held = recipe.get("weights")
    if held is None:
        positional = recipe.get("channel_weights")
        return None if not positional else [float(v) for v in positional]
    out = [float(held.get(name, 0.0)) for name in names]
    if sum(out) <= 0:
        raise ValueError(f"none of the channels the recipe weighs ({', '.join(held)}) is "
                         f"in this section; its channels: {', '.join(names)}")
    return out


def tone(image: Image.Image, recipe: dict[str, Any]) -> Image.Image:
    """The recipe's ``stretch`` (contrast limits, gamma) and ``invert`` steps on the
    blended gray *image*; the image itself when the recipe has neither."""
    limits, gamma, invert = recipe.get("contrast_limits"), recipe.get("gamma"), recipe.get("invert")
    if not limits and not gamma and not invert:
        return image
    values = np.asarray(image, dtype=np.float32) / 255.0
    if limits:
        low, high = float(limits[0]), float(limits[1])
        values = np.clip((values - low) / (high - low), 0.0, 1.0)
    if gamma:
        values = np.power(values, float(gamma))
    if invert:
        values = 1.0 - values
    return Image.fromarray(np.rint(values * 255.0).astype(np.uint8))


# --- the derived registry ----------------------------------------------------


def migrate(appearance: dict[str, Any]) -> dict[str, Any]:
    """*appearance* with the older ``"preprocessed"`` entry read as
    ``derived["preprocessed"]`` (a copy; nothing is mutated, a recipe is kept as it
    was written)."""
    out = dict(appearance)
    old = out.pop(PREPROCESSED, None)
    derived = {name: entry for name, entry in (out.get(DERIVED) or {}).items()}
    if old and PREPROCESSED not in derived:
        derived[PREPROCESSED] = old
    if derived:
        out[DERIVED] = derived
    return out


def _derived(state: StackState) -> dict[str, Any]:
    return migrate(state.appearance).get(DERIVED) or {}


def derived_names(state: StackState) -> list[str]:
    """The derived channels now, ``preprocessed`` first (it always exists)."""
    held = _derived(state)
    return [PREPROCESSED, *(name for name in held if name != PREPROCESSED)]


def derived_settings(state: StackState, name: str, section_id: str) -> Look:
    """The recipe of derived channel *name* for the section (None: the default for
    ``preprocessed``, not defined for another name): its override, else the stack's."""
    held = _derived(state).get(name) or {}
    override = (held.get("sections") or {}).get(section_id)
    if override is not None:
        return dict(override)
    stack = held.get("stack")
    return dict(stack) if stack is not None else None


def set_derived(
    state: StackState, name: str, section_ids: list[str] | None, recipe: Look,
) -> None:
    """Write derived channel *name*'s *recipe* (None = back to the default for
    ``preprocessed``; removes any other name) for the stack or for sections.

    Stack-wide writes leave per-section overrides in place (a stack-wide delete of
    another name takes them with it); a section reset removes its override, so it
    follows the stack again.
    """
    appearance = migrate(state.appearance)
    state.appearance.clear()
    state.appearance.update(appearance)
    derived = dict(state.appearance.get(DERIVED) or {})
    held = {**(derived.get(name) or {})}
    sections = dict(held.get("sections") or {})
    if section_ids is None:
        held["stack"] = recipe
        if recipe is None and name != PREPROCESSED:
            sections = {}
    else:
        for section in section_ids:
            if recipe is None:
                sections.pop(section, None)
            else:
                sections[section] = dict(recipe)
    held["sections"] = sections
    if held.get("stack") is None and not sections:
        derived.pop(name, None)
    else:
        derived[name] = held
    if derived:
        state.appearance[DERIVED] = derived
    else:
        state.appearance.pop(DERIVED, None)


def channel_names(state: StackState, ws: Workspace, record: SliceState) -> list[str]:
    """Every channel name the section can be read in: its raw names, then the
    derived ones (``preprocessed`` first)."""
    raw = list(ws.section_channels(record.id)[0])
    return [*raw, *(name for name in derived_names(state) if name not in raw)]


def preprocessed_settings(state: StackState, section_id: str) -> Look:
    """The section's preprocessed-channel recipe (None: the default)."""
    return derived_settings(state, PREPROCESSED, section_id)


def set_settings(state: StackState, section_ids: list[str] | None, settings: Look) -> None:
    """Write the preprocessed channel's *settings* (None = the default)."""
    set_derived(state, PREPROCESSED, section_ids, settings)


def look_token(ctx: Workspace, look: Look) -> str:
    """The render-cache slot of a look; the default keeps ``spec.preprocess``."""
    if look is None:
        return str(ctx.spec.preprocess)
    return json.dumps(look, sort_keys=True)


def preprocessed_image(
    ctx: Workspace, state: StackState, record: SliceState, *, long_edge: int,
) -> Image.Image:
    """The section's preprocessed channel: what the fits read.

    Same frame as :func:`langslice.core.handoff.prepare_linear_registration`
    (oriented) at *long_edge*; with the default recipe it
    is that function's render exactly.
    """
    from langslice.core.sections import render_slice

    return render_slice(ctx, record, long_edge=long_edge,
                        look=preprocessed_settings(state, record.id))


def channel_image(
    ws: Workspace, state: StackState, record: SliceState, name: str, *, long_edge: int,
) -> Image.Image:
    """The section's channel *name* on the oriented working grid at *long_edge*: a
    raw channel as the file holds it (unenhanced, display properties not applied),
    or a derived one (``preprocessed`` included) as its recipe draws it.
    ``ValueError`` naming the channels for any other name."""
    from langslice.core.sections import render_slice

    raw = list(ws.section_channels(record.id)[0])
    if name in raw:
        return render_slice(ws, record, long_edge=long_edge, look={"channel": name})
    if name in derived_names(state):
        recipe = derived_settings(state, name, record.id)
        if recipe is None and name != PREPROCESSED:
            raise ValueError(f"channel {name!r} has no recipe for {record.id}")
        return render_slice(ws, record, long_edge=long_edge, look=recipe)
    raise ValueError(f"{record.id} has no channel {name!r}; channels: "
                     f"{', '.join(channel_names(state, ws, record))}")


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
    held = look.get("weights")
    if held:
        weights_word = ", ".join(f"{name} {value:g}" for name, value in held.items())
    else:
        positional = look.get("channel_weights")
        weights_word = "auto" if not positional else "/".join(f"{w:g}" for w in positional)
    parts = [
        "weights " + weights_word,
        f"CLAHE {look['clahe_clip']:g}x{look['clahe_tiles']}" if look["clahe_clip"] else "no CLAHE",
    ]
    if look.get("n4"):
        parts.append("N4")
    if look.get("denoise"):
        parts.append("denoise")
    if look.get("contrast_limits"):
        low, high = look["contrast_limits"]
        parts.append(f"limits {low:g}-{high:g}")
    if look.get("gamma"):
        parts.append(f"gamma {look['gamma']:g}")
    if look.get("invert"):
        parts.append("inverted")
    return ", ".join(parts)
