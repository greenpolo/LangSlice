"""Section images: the named grayscale pictures of a section that ``look``
draws and every fit reads.

One noun, two kinds, one namespace per section:

- **raw**: a plane of the image file (``red``/``green``/``blue``, ``gray``,
  ``DAPI``, ``ch2``...), never edited
  (:meth:`langslice.core.workspace.Workspace.section_channels`);
- **made**: an image produced from the section's other images by a recipe
  (``make_image``, :func:`validate_recipe` and :func:`make`) or handed in by a
  script (``job.context.add_image``), saved as a file
  ``sections/<stem>/images/<name>.png`` (8-bit gray, the working copy's size,
  unoriented: the quarter turn and flip are applied where it is drawn) with an
  entry in ``SliceState.images``.

An entry is ``{"source": {...}, "file": <relative to the job folder>,
"digest": <8 hex of its pixels>, "made": <ISO time>}`` and, for a recipe whose
weights are automatic, ``"weights_used"``. ``source`` is ``{"recipe": recipe}``,
``{"script": note}`` or ``{"opening": ...}``. Files are never deleted: undo
hides the entry, and a name made again with other pixels gets a new file
(``<name>.2.png``, ...), so an undone state still finds its own.

``preprocessed`` is the made image LangSlice writes for every section when the
job is created (:func:`opening_entry`, ``{"opening": "adaptive"}``: the
automatic path, :func:`langslice.core.image_prep.adaptive_preprocess`;
``{"opening": "host", "settings": ...}``: a host's channel blend;
``{"opening": "plain"}``: the working copy as read, in gray, for a job that
preprocesses nothing), computed once at the working copy's size. It is the
image every picture that names no image draws and every fit's default. Every
made image, the opening one included, is read from its file by every reader
(fits, the image model, ``look``, zooms, masks), resized to the size that
reader works at, so the file and its digest are exactly what a reader read.

Geometry never depends on an image's pixels: every image is held at the
working copy's size and drawn at the same render size, and calibration comes
from the file, the host or the atlas width (:func:`langslice.core.transform.calibrate`).
"""

from __future__ import annotations

import hashlib
import json
import logging
import math
import re
from collections.abc import Callable, Iterable, Mapping
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Protocol

import numpy as np
from PIL import Image

from langslice.core.image_prep import (
    HOST_CLAHE_CLIP,
    HOST_CLAHE_TILE,
    adaptive_preprocess,
    coverage_weights,
    custom_appearance,
)
from langslice.core.state import SliceState, StackState
from langslice.core.workspace import Workspace

logger = logging.getLogger(__name__)

#: The image every fit reads by default and every picture without an image draws.
PREPROCESSED = "preprocessed"
#: What a made image's name may be (it is also its file's name).
NAME_PATTERN = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_-]{0,31}$")
#: Longest name a made image may have.
MAX_NAME = 32
#: The kinds of section image.
RAW, MADE = "raw", "made"
#: CLAHE clip limit and tile grid of the automatic path (the opening recipe's).
DEFAULT_CLAHE_CLIP = HOST_CLAHE_CLIP["medium"]
DEFAULT_CLAHE_TILES = HOST_CLAHE_TILE[0]
#: Accepted CLAHE ranges (0 clip = no CLAHE).
MAX_CLAHE_CLIP = 40.0
MAX_CLAHE_TILES = 32
#: Accepted gamma range of a recipe.
MIN_GAMMA, MAX_GAMMA = 0.1, 10.0
#: A recipe's automatic weights (each raw channel by its squared tissue coverage).
AUTO = "auto"
#: zlib level of the image files (fast; the files are a few megabytes).
PNG_COMPRESS_LEVEL = 3
#: Threads the opening images of a stack are made on.
OPENING_THREADS = 8
#: Working-size planes kept on ``Workspace.image_cache`` (a few megabytes each).
IMAGE_CACHE_SIZE = 64

#: A look, as :func:`langslice.core.sections.render_slice` takes it: ``None``
#: (the section's ``preprocessed`` image), ``{"image": name}`` (one section
#: image alone, in gray), ``{"overlay": [raw names]}`` (raw channels together
#: in their colours, :func:`langslice.core.channels.composite`) or
#: ``{"file": True}`` (the working copy as read, in colour: a plugin's raw
#: raster).
Look = dict[str, Any] | None


class _Layout(Protocol):
    """The part of the job's layout an image file needs (``job.layout.JobLayout``)."""

    folder: Path

    def images_dir(self, section_id: str) -> Path: ...

    def relative(self, path: str | Path) -> str: ...


# --- recipes ---------------------------------------------------------------------------


def validate_recipe(
    *,
    weights: Mapping[str, float] | str | None = None,
    clahe_clip: float = DEFAULT_CLAHE_CLIP,
    clahe_tiles: int = DEFAULT_CLAHE_TILES,
    n4: bool = False,
    denoise: bool = False,
    contrast: Iterable[float] | None = None,
    gamma: float | None = None,
    invert: bool = False,
) -> dict[str, Any]:
    """One recipe as stored, or ``ValueError`` naming what is wrong.

    *weights* ``{image name: weight}``; empty, None or ``"auto"`` is automatic
    (each raw channel by its squared tissue coverage). *contrast* ``[lo, hi]``
    (fractions 0 to 1 of the blended image) and *gamma* (1 or 0 linear) and
    *invert* run after the blend.
    """
    held: dict[str, float] | str = AUTO
    if weights and weights != AUTO:
        if not isinstance(weights, Mapping):
            raise ValueError("weights must be {image name: number}")
        try:
            held = {str(name): float(value) for name, value in weights.items()}
        except (TypeError, ValueError):
            raise ValueError("weights must be numbers") from None
        if any(not math.isfinite(value) or value < 0 for value in held.values()):
            raise ValueError("weights must be finite and non-negative")
        if sum(held.values()) <= 0:
            raise ValueError("at least one weight must be above zero")
    try:
        clip = float(clahe_clip)
        tiles = int(clahe_tiles)
    except (TypeError, ValueError):
        raise ValueError("clahe_clip must be a number and clahe_tiles an integer") from None
    if not math.isfinite(clip) or not 0.0 <= clip <= MAX_CLAHE_CLIP:
        raise ValueError(f"clahe_clip must be from 0 (none) to {MAX_CLAHE_CLIP:g}")
    if not 1 <= tiles <= MAX_CLAHE_TILES:
        raise ValueError(f"clahe_tiles must be from 1 to {MAX_CLAHE_TILES}")
    if not isinstance(n4, bool) or not isinstance(denoise, bool) or not isinstance(invert, bool):
        raise ValueError("n4, denoise and invert must be true or false")
    limits: list[float] | None = None
    given = list(contrast) if contrast is not None else []
    if given:
        try:
            limits = [float(value) for value in given]
        except (TypeError, ValueError):
            raise ValueError("contrast must be two numbers") from None
        if (len(limits) != 2 or not all(math.isfinite(v) for v in limits)
                or not 0.0 <= limits[0] < limits[1] <= 1.0):
            raise ValueError("contrast is [low, high] with 0 <= low < high <= 1 (fractions "
                             "of the blended image)")
    power: float | None = None
    if gamma:
        try:
            power = float(gamma)
        except (TypeError, ValueError):
            raise ValueError("gamma must be a number") from None
        if not math.isfinite(power) or not MIN_GAMMA <= power <= MAX_GAMMA:
            raise ValueError(f"gamma must be from {MIN_GAMMA:g} to {MAX_GAMMA:g}")
        if power == 1.0:
            power = None
    return {"weights": held, "clahe_clip": clip, "clahe_tiles": tiles, "n4": n4,
            "denoise": denoise, "contrast": limits, "gamma": power, "invert": invert}


#: The recipe the arguments' defaults give: the opening image's own.
OPENING_RECIPE: dict[str, Any] = validate_recipe()


def recipe_words(recipe: Mapping[str, Any]) -> str:
    """A recipe in a caption's words: ``weights auto, CLAHE 4 on 8x8 tiles, denoise``."""
    held = recipe.get("weights")
    weights = (AUTO if not isinstance(held, Mapping) else
               ", ".join(f"{name} {value:g}" for name, value in held.items()))
    clip = float(recipe.get("clahe_clip", 0.0))
    tiles = int(recipe.get("clahe_tiles", DEFAULT_CLAHE_TILES))
    parts = [f"weights {weights}",
             f"CLAHE {clip:g} on {tiles}x{tiles} tiles" if clip else "no CLAHE"]
    if recipe.get("n4"):
        parts.append("N4")
    if recipe.get("denoise"):
        parts.append("denoise")
    if recipe.get("contrast"):
        low, high = recipe["contrast"]
        parts.append(f"contrast {low:g}-{high:g}")
    if recipe.get("gamma"):
        parts.append(f"gamma {recipe['gamma']:g}")
    if recipe.get("invert"):
        parts.append("inverted")
    return ", ".join(parts)


def compact_recipe(recipe: Mapping[str, Any]) -> dict[str, Any]:
    """*recipe* without its steps that do nothing (``status``'s form)."""
    out: dict[str, Any] = {"weights": recipe.get("weights", AUTO),
                           "clahe_clip": recipe.get("clahe_clip"),
                           "clahe_tiles": recipe.get("clahe_tiles")}
    for key in ("n4", "denoise", "contrast", "gamma", "invert"):
        if recipe.get(key):
            out[key] = recipe[key]
    return out


# --- the opening image -----------------------------------------------------------------


def opening_source(spec: Any) -> dict[str, Any]:
    """How *spec*'s jobs make ``preprocessed`` when they open (an entry's ``source``)."""
    if spec.host_preprocessing is not None:
        return {"opening": "host", "settings": dict(spec.host_preprocessing)}
    if spec.preprocess == "auto":
        return {"opening": "adaptive"}
    return {"opening": "plain"}


def opening_words(source: Mapping[str, Any]) -> str:
    """What an opening image is, in the job statement's words."""
    kind = source.get("opening")
    if kind == "adaptive":
        return "automatic channel weights by tissue coverage, CLAHE clip 4 on 8x8 tiles"
    if kind == "host":
        settings = dict(source.get("settings") or {})
        if settings.get("mode") != "custom":
            return ("the host's automatic blend: channel weights by tissue coverage, "
                    "CLAHE clip 4 on 8x8 tiles")
        weights = ", ".join(f"{float(v):g}" for v in settings.get("channel_weights") or [])
        clahe = (f"CLAHE clip {HOST_CLAHE_CLIP[str(settings.get('clahe_strength'))]:g} on "
                 "8x8 tiles" if settings.get("clahe", True) else "no CLAHE")
        return f"the host's blend: channel weights {weights} (one per page), {clahe}"
    return "the file as read, in gray, unenhanced"


def opening_render(ws: Workspace, section_id: str) -> Image.Image:
    """The opening ``preprocessed`` image, unoriented, 8-bit gray, at the working
    copy's size: the automatic path (:func:`~langslice.core.image_prep.adaptive_preprocess`)
    on the working copy (a host's blend is the working copy itself; a job that
    preprocesses nothing has the copy in gray)."""
    source, _factor = ws.working_source(section_id)
    image = source
    if ws.spec.preprocess == "auto" and ws.spec.host_preprocessing is None:
        image = adaptive_preprocess(source)
    return image.convert("L")


def is_opening(entry: Mapping[str, Any] | None) -> bool:
    """Whether *entry* (None: no entry, the opening is implied) is an opening image."""
    return entry is None or "opening" in (entry.get("source") or {})


# --- digests, files ---------------------------------------------------------------------


def digest_of(array: np.ndarray) -> str:
    """Eight hex digits of an image's pixels (and its shape)."""
    held = np.ascontiguousarray(np.asarray(array, dtype=np.uint8))
    sha = hashlib.sha256(json.dumps(list(held.shape)).encode())
    sha.update(held.tobytes())
    return sha.hexdigest()[:8]


def _now() -> str:
    return datetime.now(UTC).isoformat()


def _read_png(path: Path) -> np.ndarray:
    with Image.open(path) as opened:
        return np.asarray(opened.convert("L"), dtype=np.uint8)


def _file_for(layout: _Layout, section_id: str, name: str, digest: str) -> tuple[Path, bool]:
    """Where *name* with pixels of *digest* goes: an existing file of the name
    holding those pixels (True: nothing to write), else the next free one."""
    folder = layout.images_dir(section_id)
    path = folder / f"{name}.png"
    version = 1
    while path.exists():
        try:
            if digest_of(_read_png(path)) == digest:
                return path, True
        except (OSError, ValueError):
            pass
        version += 1
        path = folder / f"{name}.{version}.png"
    return path, False


def write_file(
    layout: _Layout, section_id: str, name: str, array: np.ndarray, source: Mapping[str, Any],
    **extra: Any,
) -> dict[str, Any]:
    """Save *array* (``(H, W)`` uint8) as image *name* of the section and return
    its entry (not set anywhere). A file of the name with the same pixels is
    reused; another is never overwritten."""
    held = np.ascontiguousarray(np.asarray(array))
    if held.ndim != 2 or held.dtype != np.uint8:
        raise ValueError("a section image is an (H, W) uint8 array")
    digest = digest_of(held)
    path, exists = _file_for(layout, section_id, name, digest)
    if not exists:
        path.parent.mkdir(parents=True, exist_ok=True)
        temporary = path.with_name(f"{path.stem}.partial.png")
        try:
            Image.fromarray(held, mode="L").save(temporary, format="PNG",
                                                 compress_level=PNG_COMPRESS_LEVEL)
            temporary.replace(path)
        finally:
            temporary.unlink(missing_ok=True)
    return {"source": dict(source), "file": layout.relative(path), "digest": digest,
            "made": _now(), **extra}


def write_image(
    layout: _Layout, record: SliceState, name: str, array: np.ndarray,
    source: Mapping[str, Any], **extra: Any,
) -> dict[str, Any]:
    """:func:`write_file`, the entry set as ``record.images[name]``."""
    entry = write_file(layout, record.id, name, array, source, **extra)
    record.images[name] = entry
    return entry


# --- names and planes -------------------------------------------------------------------


def raw_names(ws: Workspace, section_id: str) -> list[str]:
    """The section's raw channels, in file order."""
    return [str(name) for name in ws.section_channels(section_id)[0]]


def image_names(ws: Workspace, record: SliceState) -> list[str]:
    """Every section image of *record*: its raw channels, then ``preprocessed``,
    then the images made since (in the order they were made)."""
    raw = raw_names(ws, record.id)
    made = [PREPROCESSED, *(name for name in record.images if name != PREPROCESSED)]
    return [*raw, *(name for name in made if name not in raw)]


def kind_of(ws: Workspace, record: SliceState, name: str) -> str | None:
    """``raw``, ``made`` or None (no such image on the section)."""
    if name in raw_names(ws, record.id):
        return RAW
    if name == PREPROCESSED or name in record.images:
        return MADE
    return None


def _working_size(ws: Workspace, section_id: str) -> tuple[int, int]:
    return ws.working_source(section_id)[0].size


def _resolved_file(ws: Workspace, entry: Mapping[str, Any]) -> Path:
    path = Path(str(entry["file"]))
    if path.is_absolute():
        return path
    folder = getattr(ws, "job_folder", None)
    if not folder:
        raise ValueError("a made section image is read from the job folder, and this "
                         "workspace has none")
    return Path(folder) / path


def _cached(ws: Workspace, key: tuple[str, str], make: Callable[[], np.ndarray]) -> np.ndarray:
    """*key*'s plane from the workspace's cache, least recently used out past
    :data:`IMAGE_CACHE_SIZE` planes."""
    cache = ws.image_cache
    held = cache.get(key)
    if held is None:
        held = make()
        cache[key] = held
        while len(cache) > IMAGE_CACHE_SIZE:
            cache.popitem(last=False)
    else:
        cache.move_to_end(key)
    return held


def raw_plane(ws: Workspace, section_id: str, name: str) -> np.ndarray:
    """Raw channel *name* at the working copy's size (uint8, unoriented)."""
    names, planes = ws.section_channels(section_id)
    if name not in names:
        raise ValueError(f"{section_id} has no raw channel {name!r}; its channels: "
                         f"{', '.join(map(str, names))}")
    size = _working_size(ws, section_id)

    def plane() -> np.ndarray:
        image = Image.fromarray(np.asarray(planes[list(names).index(name)], dtype=np.uint8))
        if image.size != size:
            image = image.resize(size, Image.Resampling.LANCZOS)
        return np.asarray(image, dtype=np.uint8)

    return _cached(ws, (section_id, f"raw:{name}"), plane)


def entry_of(record: SliceState, name: str) -> dict[str, Any] | None:
    """The made image *name*'s entry (None: none, which for ``preprocessed`` is
    the opening image itself)."""
    return record.images.get(name)


def plane_of(
    ws: Workspace, section_id: str, images: Mapping[str, Any], name: str,
) -> np.ndarray:
    """Section image *name* at the working copy's size (uint8 ``(H, W)``,
    unoriented): a raw plane, or a made image's file (``preprocessed`` without
    an entry, a workspace with no job: the opening image made in memory).
    ``ValueError`` naming the
    images for any other name."""
    if name in raw_names(ws, section_id):
        return raw_plane(ws, section_id, name)
    entry = images.get(name)
    if entry is None and name == PREPROCESSED:
        return _cached(ws, (section_id, "opening"),
                       lambda: np.asarray(opening_render(ws, section_id), dtype=np.uint8))
    if entry is None:
        raise ValueError(f"{section_id} has no section image {name!r}; its images: "
                         f"{', '.join([*raw_names(ws, section_id), PREPROCESSED, *images])}")
    path = _resolved_file(ws, entry)
    return _cached(ws, (section_id, f"file:{entry['file']}:{entry.get('digest')}"),
                   lambda: _read_png(path))


def image_plane(ws: Workspace, record: SliceState, name: str) -> np.ndarray:
    """:func:`plane_of` for a section record."""
    return plane_of(ws, record.id, record.images, name)


@dataclass(frozen=True, eq=False)
class ImageView:
    """One section image as a fit reads it: its pixels at the working copy's
    size (uint8, unoriented) and what names it."""

    name: str
    kind: str
    array: np.ndarray
    digest: str | None
    #: The made image's file, relative to the job folder; for a raw channel the
    #: section's image file (in the image folder).
    file: str | None
    source: dict[str, Any] | None

    def payload(self) -> dict[str, Any]:
        """``{name, kind, file, digest}``: a fit reply's ``inputs.section_image``."""
        return {"name": self.name, "kind": self.kind, "file": self.file, "digest": self.digest}


def read_image(ws: Workspace, record: SliceState, name: str) -> ImageView:
    """Section image *name* of *record* (``ValueError`` for a name it lacks)."""
    kind = kind_of(ws, record, name)
    if kind is None:
        raise ValueError(f"{record.id} has no section image {name!r}; its images: "
                         f"{', '.join(image_names(ws, record))}")
    array = image_plane(ws, record, name)
    if kind == RAW:
        return ImageView(name=name, kind=RAW, array=array, digest=None, file=record.id,
                         source=None)
    entry = entry_of(record, name)
    if entry is None:  # the opening image, implied
        return ImageView(name=name, kind=MADE, array=array, digest=digest_of(array), file=None,
                         source=opening_source(ws.spec))
    return ImageView(name=name, kind=MADE, array=array, digest=str(entry.get("digest")),
                     file=str(entry.get("file")), source=dict(entry.get("source") or {}))


def identity(ws: Workspace, record: SliceState, name: str) -> dict[str, Any]:
    """What names the image a fit reads, for its cache key and record:
    ``{name, kind, digest}`` (a raw channel's digest is the file's name)."""
    view = read_image(ws, record, name)
    return {"name": name, "kind": view.kind, "digest": view.digest}


def images_digest(record: SliceState) -> dict[str, str]:
    """``{name: digest}`` of the section's made images: what a long verb
    compares, so remaking an image while it runs makes the section stale."""
    return {name: str(entry.get("digest")) for name, entry in sorted(record.images.items())}


# --- making an image from a recipe --------------------------------------------------------


def _tone(values: np.ndarray, recipe: Mapping[str, Any]) -> np.ndarray:
    """The recipe's contrast, gamma and invert steps on a blended uint8 plane."""
    limits, gamma, invert = recipe.get("contrast"), recipe.get("gamma"), recipe.get("invert")
    if not limits and not gamma and not invert:
        return values
    out = values.astype(np.float32) / 255.0
    if limits:
        low, high = float(limits[0]), float(limits[1])
        out = np.clip((out - low) / (high - low), 0.0, 1.0)
    if gamma:
        out = np.power(out, float(gamma))
    if invert:
        out = 1.0 - out
    return np.rint(out * 255.0).astype(np.uint8)


def sources_of(
    ws: Workspace, section_id: str, images: Mapping[str, Any], recipe: Mapping[str, Any],
) -> tuple[list[str], list[float] | None]:
    """``(source names, weights)`` *recipe* blends on the section: automatic
    weights are every raw channel (weights None, worked out by :func:`make`);
    explicit ones the images weighted above zero. ``ValueError`` when a source
    is no image of the section."""
    held = recipe.get("weights")
    if not isinstance(held, Mapping):
        return raw_names(ws, section_id), None
    known = [*raw_names(ws, section_id), PREPROCESSED, *images]
    missing = [name for name, value in held.items() if value > 0 and name not in known]
    if missing:
        raise ValueError(f"{section_id} has no section image {', '.join(map(repr, missing))}; "
                         f"its images: {', '.join(dict.fromkeys(known))}")
    used = [(str(name), float(value)) for name, value in held.items() if value > 0]
    return [name for name, _v in used], [value for _n, value in used]


def make(
    ws: Workspace, section_id: str, images: Mapping[str, Any], recipe: Mapping[str, Any],
) -> tuple[np.ndarray, dict[str, float]]:
    """*recipe* drawn on the section's working copy: ``(uint8 (H, W), the weights
    used by name)``. The steps on each source: N4, denoise, CLAHE; then the
    weighted blend, brightened to a mid-gray tissue mean; then contrast, gamma,
    invert (:func:`langslice.core.image_prep.custom_appearance`)."""
    names, weights = sources_of(ws, section_id, images, recipe)
    planes = [plane_of(ws, section_id, images, name) for name in names]
    used = list(weights) if weights is not None else coverage_weights(planes)
    blended = custom_appearance(
        planes, channel_weights=used, clahe_clip=float(recipe["clahe_clip"]),
        clahe_tiles=int(recipe["clahe_tiles"]), n4=bool(recipe["n4"]),
        denoise=bool(recipe["denoise"]))
    values = np.asarray(blended, dtype=np.uint8)[..., 0]
    total = float(sum(used)) or 1.0
    return _tone(values, recipe), {name: round(value / total, 4)
                                   for name, value in zip(names, used, strict=True)}


# --- the opening image of a job ---------------------------------------------------------


def opening_entry(ws: Workspace, layout: _Layout, section_id: str) -> dict[str, Any]:
    """The opening ``preprocessed`` image of one section, written: its entry."""
    array = np.asarray(opening_render(ws, section_id), dtype=np.uint8)
    return write_file(layout, section_id, PREPROCESSED, array, opening_source(ws.spec))


def ensure_opening(ws: Workspace, layout: _Layout, state: StackState) -> list[str]:
    """Give every section of *state* without a ``preprocessed`` entry its opening
    image (in place, outside undo); the sections given one. A section whose
    file cannot be read is left without (logged): its pictures say so."""
    missing = [record for record in state.slices if PREPROCESSED not in record.images]
    if not missing:
        return []

    def one(record: SliceState) -> tuple[SliceState, dict[str, Any] | None]:
        try:
            return record, opening_entry(ws, layout, record.id)
        except Exception:  # noqa: BLE001 - an unreadable file is the section's own problem
            logger.warning("the opening image of %s could not be made", record.id,
                           exc_info=True)
            return record, None

    with ThreadPoolExecutor(max_workers=max(1, min(OPENING_THREADS, len(missing)))) as pool:
        done = list(pool.map(one, missing))
    written: list[str] = []
    for record, entry in done:
        if entry is not None:
            record.images = {PREPROCESSED: entry, **record.images}
            written.append(record.id)
    return written


# --- status --------------------------------------------------------------------------------


def describe_source(entry: Mapping[str, Any] | None, spec: Any = None) -> dict[str, Any]:
    """An entry's source as ``status`` gives it: a recipe (its steps), a
    script's note, or the opening image's recipe with when it was made."""
    source = dict((entry or {}).get("source") or ({} if spec is None else opening_source(spec)))
    if "recipe" in source:
        return compact_recipe(source["recipe"])
    if "script" in source:
        return {"script": str(source["script"])}
    if source.get("opening") == "adaptive":
        return {**compact_recipe(OPENING_RECIPE), "made": "when the job opened"}
    return {"opening": opening_words(source), "made": "when the job opened"}


def summary(state: StackState, spec: Any) -> dict[str, Any]:
    """The made images of the stack for ``status``: each name with its source
    when every section holding it shares one (else the most common one and
    ``varies``: the sections that differ, with theirs), and ``only`` (the
    sections holding it, when fewer than half do) or ``except`` (the sections
    without it)."""
    records = sorted(state.slices, key=lambda record: record.index_original)
    names = list(dict.fromkeys(
        [PREPROCESSED, *(name for record in records for name in record.images)]))
    out: dict[str, Any] = {}
    for name in names:
        held = {record.id: describe_source(record.images.get(name), spec)
                for record in records if name in record.images or name == PREPROCESSED}
        if not held:
            continue
        texts = [json.dumps(value, sort_keys=True) for value in held.values()]
        common = max(texts, key=texts.count)  # ties: the first section's
        row: dict[str, Any] = dict(json.loads(common))
        differ = {section: value for (section, value), text in zip(held.items(), texts,
                                                                    strict=True)
                  if text != common}
        if differ:
            row["varies"] = differ
        lacking = [record.id for record in records if record.id not in held]
        if lacking:
            if len(held) < len(records) / 2:
                row["only"] = list(held)
            else:
                row["except"] = lacking
        out[name] = row
    return out


# --- migration from state format 3 ---------------------------------------------------------


class Migration:
    """State format 3 to 4, for one job: every recipe the old ``appearance``
    held (stack-wide, or a section's override) made into a file and an entry
    per section; the opening ``preprocessed`` where the old one had no recipe;
    the raw channels' display properties dropped. A memo keeps one file per
    (section, name, source) across the checkpoint and its history steps."""

    def __init__(self, ws: Workspace, layout: _Layout) -> None:
        self.ws = ws
        self.layout = layout
        self.memo: dict[tuple[str, str, str], dict[str, Any] | None] = {}
        #: Whether a state of the older format has been read through it.
        self.used = False

    def seed(self, data: Mapping[str, Any]) -> None:
        """Remember the entries a format-4 state already holds (made by an
        earlier migration), so the history's steps reuse their files."""
        for row in data.get("slices") or []:
            for name, entry in (row.get("images") or {}).items():
                key = (str(row.get("id")), str(name),
                       json.dumps(entry.get("source"), sort_keys=True))
                self.memo.setdefault(key, dict(entry))

    def _entry(self, section_id: str, name: str, source: dict[str, Any],
               make_array: Callable[[], tuple[np.ndarray, dict[str, Any]]],
               ) -> dict[str, Any] | None:
        key = (section_id, name, json.dumps(source, sort_keys=True))
        if key not in self.memo:
            try:
                array, extra = make_array()
                self.memo[key] = write_file(self.layout, section_id, name, array, source,
                                            **extra)
            except Exception:  # noqa: BLE001 - one section's recipe that cannot be made is skipped
                logger.warning("section image %s of %s could not be made from its recipe",
                               name, section_id, exc_info=True)
                self.memo[key] = None
        held = self.memo[key]
        return None if held is None else dict(held)

    def __call__(self, data: dict[str, Any]) -> dict[str, Any]:
        """*data* (a format-3 state's fields) as format 4 (a copy)."""
        self.used = True
        out = dict(data)
        legacy = dict(out.pop("appearance", None) or {})  # format 3's recipes
        derived = dict(legacy.get("derived") or {})
        if legacy.get(PREPROCESSED) and PREPROCESSED not in derived:
            derived[PREPROCESSED] = legacy[PREPROCESSED]
        rows = []
        for row in out.get("slices") or []:
            row = dict(row)
            section_id = str(row.get("id"))
            images: dict[str, Any] = {}
            names = [PREPROCESSED, *(name for name in derived if name != PREPROCESSED)]
            for name in names:
                held = dict(derived.get(name) or {})
                old = (held.get("sections") or {}).get(section_id, held.get("stack"))
                if old is None:
                    if name != PREPROCESSED:
                        continue
                    source = opening_source(self.ws.spec)
                    entry = self._entry(section_id, name, source, lambda sid=section_id: (
                        np.asarray(opening_render(self.ws, sid), dtype=np.uint8), {}))
                else:
                    try:
                        recipe = _old_recipe(old, raw_names(self.ws, section_id))
                    except (ValueError, OSError):
                        logger.warning("recipe %s of %s cannot be read; left out", name,
                                       section_id, exc_info=True)
                        continue
                    source = {"recipe": recipe}
                    made = dict(images)

                    def drawn(sid: str = section_id, rec: dict[str, Any] = recipe,
                              held_images: dict[str, Any] = made,
                              ) -> tuple[np.ndarray, dict[str, Any]]:
                        array, used = make(self.ws, sid, held_images, rec)
                        return array, ({"weights_used": used}
                                       if not isinstance(rec["weights"], Mapping) else {})

                    entry = self._entry(section_id, name, source, drawn)
                if entry is not None:
                    images[name] = entry
            masks = row.get("masks")
            if isinstance(masks, dict):
                row["masks"] = {name: _mask_entry(entry) for name, entry in masks.items()}
            if images:
                row["images"] = images
            rows.append(row)
        if "slices" in out:
            out["slices"] = rows
        return out


def _mask_entry(entry: Any) -> Any:
    """A format-3 mask entry with its ``channel`` named ``section_image``."""
    if not isinstance(entry, dict) or "channel" not in entry:
        return entry
    held = dict(entry)
    held["section_image"] = held.pop("channel")
    return held


def _old_recipe(old: Mapping[str, Any], raw: list[str]) -> dict[str, Any]:
    """A format-3 recipe (weights by raw name or None, or positional
    ``channel_weights``; ``contrast_limits``) as a format-4 one."""
    weights = old.get("weights")
    if not weights and old.get("channel_weights"):
        weights = {raw[index]: float(value)
                   for index, value in enumerate(old["channel_weights"]) if index < len(raw)}
    return validate_recipe(
        weights=weights or None, clahe_clip=float(old.get("clahe_clip", DEFAULT_CLAHE_CLIP)),
        clahe_tiles=int(old.get("clahe_tiles", DEFAULT_CLAHE_TILES)),
        n4=bool(old.get("n4", False)), denoise=bool(old.get("denoise", False)),
        contrast=old.get("contrast_limits") or None, gamma=old.get("gamma") or None,
        invert=bool(old.get("invert", False)))


__all__ = [
    "AUTO", "DEFAULT_CLAHE_CLIP", "DEFAULT_CLAHE_TILES", "ImageView", "Look", "MADE",
    "Migration", "OPENING_RECIPE", "PREPROCESSED", "RAW", "compact_recipe", "describe_source",
    "digest_of", "ensure_opening", "entry_of", "identity", "image_names", "image_plane",
    "images_digest", "is_opening", "kind_of", "make", "opening_entry", "opening_render",
    "opening_source", "opening_words", "plane_of", "raw_names", "raw_plane", "read_image",
    "recipe_words", "sources_of", "summary", "validate_recipe", "write_file", "write_image",
]
