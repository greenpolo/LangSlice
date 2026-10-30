"""Old job folders, read as the current format: the table of migrations.

Each row of :data:`STATE_MIGRATIONS` is a named function from one format
of ``state.json`` to the next. The state-file reader hands a checkpoint
or an undo step of an older format to the job's :class:`FolderMigration`,
which applies the rows in order from the file's format to the current
one. A job opened to write saves what it read at the current format once,
so each migration runs once per folder.

================  =====  =============================================
migration         reads  what it does, to the next format
================  =====  =============================================
``state_3_to_4``  3      the stack's ``appearance`` recipes become
                         section image files with an entry per section
                         (the opening ``preprocessed`` where there was
                         no recipe); the raw channels' display
                         properties are dropped; a mask's ``channel``
                         is named ``section_image``
``state_4_to_5``  4      a section's ``image_correction`` is named
                         ``border_trace``; the paths it stores, and the
                         trace a deformation step records, name the
                         section's ``border_trace/`` folder once that
                         folder has moved there
================  =====  =============================================

The folder itself, ``sections/<name>/image_correction/``, moves to
``sections/<name>/border_trace/`` when the job is opened to write, under
its write lock and before the state is read (``trace_folders``, which
also writes the paths naming it again in the trace folder's own requests
and replies, the pictures' records and the sections' registrations). An
open that writes nothing (a dry run, a read-only folder) moves nothing
and reads the paths as the folder stands, so its traces are still found.

An older format is refused: a pre-release LangSlice made it.

What a file of any format may still hold of what LangSlice no longer has
is dropped as it is read, with one logged warning per name (the
``langslice.job.migrations`` logger; the job's ``logs/events.jsonl``
gets a ``migrated`` line when it writes the folder again):

- ``state.json`` and the undo steps: the stack's
  ``REMOVED_STACK_KEYS`` and a section's ``REMOVED_SECTION_KEYS``
  (``drop_removed_keys``, which the job's migration runs after the
  table, at every format);
- ``job.json``'s settings: ``REMOVED_SETTINGS`` and the tasks of
  ``REMOVED_TASKS`` (``without_removed_settings``), which the
  doors read the saved settings through.

Two old shapes are read in the core, because the job never sees them
first: a deformation's ``record.json`` is read by the core's record store
when a fit, a picture or an export needs it, and record folders are
content-addressed and never rewritten. Its reader refuses a record of
format 1 (``arrays.npz``), so such a deformation is held but cannot be
drawn, and reads a record's fit settings with the former ``stiffness``
level as their ``smoothness_um``.
"""

from __future__ import annotations

import json
import logging
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from functools import partial

import numpy as np

from langslice.core.jsonable import Json, JsonObject, JsonValue, to_jsonable
from langslice.core.sections.images import (
    DEFAULT_CLAHE_CLIP,
    DEFAULT_CLAHE_TILES,
    PREPROCESSED,
    make,
    opening_render,
    opening_source,
    raw_names,
    validate_recipe,
    write_file,
)
from langslice.core.stack.codec import image_from_json, image_to_json
from langslice.core.stack.json_reads import (
    as_number,
    as_object,
    number_at,
    numbers_at,
    object_at,
)
from langslice.core.workspace import Workspace
from langslice.job.layout import JobLayout
from langslice.job.state_file import STATE_FORMAT_VERSION, older_format
from langslice.job.trace_folders import moved_path

logger = logging.getLogger(__name__)

#: A state's fields as a migration returns them.
Fields = dict[str, JsonValue]
#: Makes a section image's pixels and what else its entry records.
MakePixels = Callable[[], tuple[np.ndarray, JsonObject]]

#: The settings an earlier LangSlice wrote to ``job.json`` that this one
#: does not have (a task section's as ``section.key``).
REMOVED_SETTINGS = (
    "max_input_tokens",
    "max_quota_percent",
    "agent_preprocessing",
    "agent_damage",
    "enabled",
    "position.gated",
    "position.bayesian",
    "transform.flip",
    "transform.interactive",
    "transform.automatic",
    "nonlinear.engine",
)
#: The tasks an earlier LangSlice's ``job.json`` names that this one lacks.
REMOVED_TASKS = ("reorder",)
#: The keys of the stack a state may hold that no reader uses.
REMOVED_STACK_KEYS = ("appearance",)
#: The keys of a section row a state may hold that no reader uses.
REMOVED_SECTION_KEYS = ("position_source", "damaged")

#: A section row's key for its border trace, before format 5 and since.
OLD_TRACE_KEY, TRACE_KEY = "image_correction", "border_trace"
#: The keys that hold such a path: a trace's folder and files, and the
#: trace a deformation step records.
PATH_KEYS = ("artifact_dir", "artifact_paths", "trace")


class FolderMigration:
    """Read one job folder's state files of older formats as the current one.

    The state-file reader calls it with a file's format and fields. It
    keeps each section image it made by section, name and source, across
    the checkpoint and its history steps, so each is written once.

    Parameters
    ----------
    workspace
        The workspace the job's sections are read through.
    layout
        The job folder the made images are written to.
    """

    def __init__(self, workspace: Workspace, layout: JobLayout) -> None:
        self.workspace = workspace
        self.layout = layout
        #: The entry of each section image made, by (section, name,
        #: source); None for one that could not be made.
        self.made: dict[tuple[str, str, str], Fields | None] = {}
        #: Whether it changed a state it read: one of an older format, or
        #: one holding a key it dropped.
        self.used = False
        #: The migrations of the table it ran, by name, in order.
        self.applied: list[str] = []
        #: Each removed key it dropped (a section's as ``slices.key``),
        #: with the number of sections that held it where it first was.
        self.dropped: dict[str, int] = {}

    def seed(self, data: JsonObject) -> None:
        """Remember the section images a current state already holds.

        An earlier migration made them, so the history's steps reuse
        their files.

        Parameters
        ----------
        data
            A state's fields at the current format.
        """
        for row in _rows(data):
            for name, entry in _object_or_empty(row, "images").items():
                fields = to_jsonable(as_object(name, entry))
                key = (
                    str(row.get("id")),
                    name,
                    _sorted_text(fields.get("source")),
                )
                self.made.setdefault(key, fields)

    def __call__(self, version: int, data: JsonObject) -> JsonObject:
        """Return a state's fields as the current format.

        The table's rows run from the state's format; then the keys
        LangSlice no longer has are dropped, each logged the first time.

        Parameters
        ----------
        version
            The format the fields were written at.
        data
            The state's fields, without the format version; not changed.

        Returns
        -------
        JsonObject
            The fields at the current state format: *data* itself when
            nothing changed.

        Raises
        ------
        ValueError
            For an older format no row of :data:`STATE_MIGRATIONS` reads.
        """
        fields = data
        if version < STATE_FORMAT_VERSION:
            fields = self._run_table(version, data)
        dropped = drop_removed_keys(fields)
        for name, count in dropped.counts.items():
            if name not in self.dropped:
                self.dropped[name] = count
                _warn_dropped_key(name, count)
        if dropped.counts:
            self.used = True
        return dropped.fields

    def _run_table(self, version: int, data: JsonObject) -> Fields:
        """Run the table's rows from *version*; refuse a format it lacks."""
        if version not in {row.reads for row in STATE_MIGRATIONS}:
            raise older_format(version)
        fields = to_jsonable(data)
        for row in STATE_MIGRATIONS:
            if row.reads >= version:
                fields = row.migrate(fields, self)
                if row.name not in self.applied:
                    self.applied.append(row.name)
        self.used = True
        return fields

    def section_image(
        self,
        section_id: str,
        name: str,
        source: JsonObject,
        make_pixels: MakePixels,
    ) -> Fields | None:
        """Return the entry of a section image made from a source.

        The image is written the first time it is asked for; later asks
        return the same entry.

        Parameters
        ----------
        section_id
            The section the image is made for.
        name
            The image's name among the section's images.
        source
            The entry's ``source``: a recipe, or the opening image's.
        make_pixels
            Makes the image's pixels and what else its entry records.

        Returns
        -------
        dict or None
            The entry; None when the image cannot be made (logged), so
            the section goes without it.
        """
        key = (section_id, name, _sorted_text(source))
        if key not in self.made:
            self.made[key] = self._write(section_id, name, source, make_pixels)
        entry = self.made[key]
        if entry is None:
            return None
        return dict(entry)

    def _write(
        self,
        section_id: str,
        name: str,
        source: JsonObject,
        make_pixels: MakePixels,
    ) -> Fields | None:
        """Write a section image and return its entry, or None (logged)."""
        try:
            array, extra = make_pixels()
            image = write_file(
                self.layout, section_id, name, array, source, extra=extra
            )
        except Exception:  # noqa: BLE001 - a recipe that cannot be made is left out
            logger.warning(
                "section image %s of %s could not be made from its recipe",
                name,
                section_id,
                exc_info=True,
            )
            return None
        return image_to_json(image)


def state_3_to_4(data: Fields, folder: FolderMigration) -> Fields:
    """Return a format-3 state's fields as format 4.

    Parameters
    ----------
    data
        The fields at format 3; not changed.
    folder
        The job folder's migration, which writes the section images.

    Returns
    -------
    dict
        The fields at format 4.
    """
    fields = dict(data)
    recipes = _old_recipes(fields.pop("appearance", None))
    if "slices" in fields:
        rows = [_section_3_to_4(row, recipes, folder) for row in _rows(fields)]
        fields["slices"] = list[JsonValue](rows)
    return fields


def _old_recipes(appearance: JsonValue) -> dict[str, Json]:
    """Return a format-3 ``appearance``'s recipes, by section image name.

    Each is ``{"stack": recipe, "sections": {id: recipe}}``; the stack's
    ``preprocessed`` recipes may stand outside ``derived``.
    """
    if not appearance:
        return {}
    scopes = as_object("appearance", appearance)
    recipes = dict(_object_or_empty(scopes, "derived"))
    preprocessed = scopes.get(PREPROCESSED)
    if preprocessed and PREPROCESSED not in recipes:
        recipes[PREPROCESSED] = preprocessed
    return recipes


def _section_3_to_4(
    row: JsonObject, recipes: Mapping[str, Json], folder: FolderMigration
) -> Fields:
    """Return a format-3 section row as format 4."""
    section_id = str(row.get("id"))
    names = [PREPROCESSED, *(name for name in recipes if name != PREPROCESSED)]
    images: Fields = {}
    for name in names:
        entry = _image_3_to_4(
            section_id, name, recipes.get(name), images, folder
        )
        if entry is not None:
            images[name] = entry

    section = to_jsonable(row)
    masks = row.get("masks")
    if isinstance(masks, Mapping):
        section["masks"] = {
            name: _mask_3_to_4(entry) for name, entry in masks.items()
        }
    if images:
        section["images"] = images
    return section


def _image_3_to_4(
    section_id: str,
    name: str,
    recipes: Json,
    made: Fields,
    folder: FolderMigration,
) -> Fields | None:
    """Return the entry a format-3 recipe of a section image becomes.

    The section's own recipe, else the stack's; ``preprocessed`` without
    either is the opening image. None for a recipe that cannot be read
    (logged) or made, and for another image without a recipe.
    """
    workspace = folder.workspace
    by_scope: JsonObject = {}
    if recipes:
        by_scope = as_object(name, recipes)
    by_section = _object_or_empty(by_scope, "sections")
    old = by_section.get(section_id, by_scope.get("stack"))
    if old is None:
        if name != PREPROCESSED:
            return None
        source = opening_source(workspace.settings)
        pixels = partial(_opening_pixels, workspace, section_id)
        return folder.section_image(section_id, name, source, pixels)

    try:
        recipe = _recipe_3_to_4(
            as_object(name, old), raw_names(workspace, section_id)
        )
    except (ValueError, OSError):
        logger.warning(
            "recipe %s of %s cannot be read; left out",
            name,
            section_id,
            exc_info=True,
        )
        return None
    source: JsonObject = {"recipe": recipe}
    pixels = partial(_recipe_pixels, workspace, section_id, dict(made), recipe)
    return folder.section_image(section_id, name, source, pixels)


def _opening_pixels(
    workspace: Workspace, section_id: str
) -> tuple[np.ndarray, JsonObject]:
    """Return the opening image's pixels; its entry records nothing else."""
    array = np.asarray(opening_render(workspace, section_id), dtype=np.uint8)
    return array, {}


def _recipe_pixels(
    workspace: Workspace,
    section_id: str,
    made: Fields,
    recipe: JsonObject,
) -> tuple[np.ndarray, JsonObject]:
    """Return a recipe's pixels and, for automatic weights, those it used.

    The recipe may read the section images made before it (*made*).
    """
    images = {
        name: image_from_json(as_object(name, entry))
        for name, entry in made.items()
    }
    array, weights_used = make(workspace, section_id, images, recipe)
    if isinstance(recipe["weights"], Mapping):
        return array, {}
    return array, {"weights_used": weights_used}


def _mask_3_to_4(entry: Json) -> JsonValue:
    """Return a format-3 mask entry, its ``channel`` as ``section_image``."""
    if not isinstance(entry, Mapping) or "channel" not in entry:
        return to_jsonable(entry)
    mask = to_jsonable(entry)
    mask["section_image"] = mask.pop("channel")
    return mask


def _recipe_3_to_4(old: JsonObject, raw: Sequence[str]) -> JsonObject:
    """Return a format-3 recipe as a format-4 one.

    A format-3 recipe holds its weights by raw channel name (or None), or
    by position as ``channel_weights``, and its contrast as
    ``contrast_limits``.
    """
    weights: dict[str, float] | None = None
    if old.get("weights"):
        weights = {
            name: as_number(name, value)
            for name, value in object_at(old, "weights").items()
        }
    elif old.get("channel_weights"):
        values = numbers_at(old, "channel_weights")[: len(raw)]
        weights = dict(zip(raw, values, strict=False))
    contrast: tuple[float, ...] | None = None
    if old.get("contrast_limits"):
        contrast = numbers_at(old, "contrast_limits")
    gamma: float | None = None
    if old.get("gamma"):
        gamma = number_at(old, "gamma")
    clahe_clip = old.get("clahe_clip", DEFAULT_CLAHE_CLIP)
    clahe_tiles = old.get("clahe_tiles", DEFAULT_CLAHE_TILES)
    return validate_recipe(
        weights=weights,
        clahe_clip=as_number("clahe_clip", clahe_clip),
        clahe_tiles=int(as_number("clahe_tiles", clahe_tiles)),
        n4=bool(old.get("n4", False)),
        denoise=bool(old.get("denoise", False)),
        contrast=contrast,
        gamma=gamma,
        invert=bool(old.get("invert", False)),
    )


def state_4_to_5(data: Fields, folder: FolderMigration) -> Fields:
    """Return a format-4 state's fields as format 5.

    Parameters
    ----------
    data
        The fields at format 4; not changed.
    folder
        The job folder's migration; the paths are checked in its folder.

    Returns
    -------
    dict
        The fields at format 5: each section's border trace under
        ``border_trace``, its paths and its deformation steps' traces in
        the moved folder once it has moved
        (:func:`~langslice.job.trace_folders.moved_path`).
    """
    fields = dict(data)
    if "slices" in fields:
        layout = folder.layout
        rows = [_section_4_to_5(row, layout) for row in _rows(fields)]
        fields["slices"] = list[JsonValue](rows)
    return fields


def _section_4_to_5(row: JsonObject, layout: JobLayout) -> Fields:
    """Return a format-4 section row as format 5."""
    section: Fields = {}
    for key, value in row.items():
        name = TRACE_KEY if key == OLD_TRACE_KEY else key
        section[name] = to_jsonable(value)
    trace = section.get(TRACE_KEY)
    if isinstance(trace, dict):
        section[TRACE_KEY] = {
            key: _moved_value(value, layout, key)
            for key, value in trace.items()
        }
    deformation = section.get("deformation")
    steps = None
    if isinstance(deformation, dict):
        steps = deformation.get("steps")
    if isinstance(deformation, dict) and isinstance(steps, list):
        section["deformation"] = {
            **deformation,
            "steps": [_step_moved(step, layout) for step in steps],
        }
    return section


def _step_moved(step: JsonValue, layout: JobLayout) -> JsonValue:
    """Return a deformation step with its trace's folder moved."""
    if not isinstance(step, dict):
        return step
    return {
        key: _moved_value(value, layout, key) for key, value in step.items()
    }


def _moved_value(value: JsonValue, layout: JobLayout, key: str) -> JsonValue:
    """Return a stored path, or an object of paths, in the moved folder.

    The keys that hold one are :data:`PATH_KEYS`.
    """
    if key not in PATH_KEYS:
        return value
    if isinstance(value, str):
        return moved_path(value, layout)
    if isinstance(value, dict):
        return {
            name: _moved_value(path, layout, key)
            for name, path in value.items()
        }
    return value


@dataclass(frozen=True)
class Dropped:
    """Hold a state's fields without its removed keys, and what was dropped.

    Parameters
    ----------
    fields
        The fields; the state's own when nothing was dropped.
    counts
        Each key dropped (a section's as ``slices.key``) with the number
        of sections that held it (1 for a stack key).
    """

    fields: JsonObject
    counts: Mapping[str, int]


def drop_removed_keys(data: JsonObject) -> Dropped:
    """Drop the stack's and the sections' keys that LangSlice no longer has.

    Parameters
    ----------
    data
        A state's fields at the current format; not changed.

    Returns
    -------
    Dropped
        The fields without :data:`REMOVED_STACK_KEYS` and
        :data:`REMOVED_SECTION_KEYS`, and the keys dropped.
    """
    counts: dict[str, int] = {}
    for key in REMOVED_STACK_KEYS:
        if key in data:
            counts[key] = 1
    rows = _rows(data)
    for key in REMOVED_SECTION_KEYS:
        holders = sum(key in row for row in rows)
        if holders:
            counts[f"slices.{key}"] = holders
    if not counts:
        return Dropped(fields=data, counts={})

    fields = {k: v for k, v in data.items() if k not in REMOVED_STACK_KEYS}
    if "slices" in fields:
        fields["slices"] = [
            _without(row, REMOVED_SECTION_KEYS) for row in rows
        ]
    return Dropped(fields=fields, counts=counts)


def _without(row: JsonObject, keys: Sequence[str]) -> JsonObject:
    """Return a row without some keys."""
    return {key: value for key, value in row.items() if key not in keys}


def without_removed_settings(
    settings: JsonObject,
) -> tuple[Fields, tuple[str, ...]]:
    """Return saved job settings without those LangSlice no longer has.

    Parameters
    ----------
    settings
        The settings as ``job.json`` holds them (its ``spec``); not
        changed.

    Returns
    -------
    kept : dict
        The settings without :data:`REMOVED_SETTINGS`, and without the
        tasks of :data:`REMOVED_TASKS` in ``tasks``.
    removed : tuple of str
        What was left out: each setting's name, and ``tasks: <task>``.
    """
    kept = to_jsonable(settings)
    removed: list[str] = []
    for name in REMOVED_SETTINGS:
        section, _, key = name.rpartition(".")
        holder = kept.get(section) if section else kept
        if isinstance(holder, dict) and key in holder:
            del holder[key]
            removed.append(name)
    tasks = kept.get("tasks")
    if isinstance(tasks, list):
        for task in REMOVED_TASKS:
            if task in tasks:
                tasks.remove(task)
                removed.append(f"tasks: {task}")
    return kept, tuple(removed)


def warn_removed_settings(removed: Sequence[str]) -> None:
    """Log one warning per setting :func:`without_removed_settings` left out.

    Parameters
    ----------
    removed
        The names it returned.
    """
    for name in removed:
        logger.warning(
            "job.json holds %s, which this LangSlice no longer has; it is "
            "left out of the job's settings.",
            name,
        )


def _warn_dropped_key(name: str, count: int) -> None:
    """Log the warning for one removed state key, on its first reading."""
    where = "the stack"
    if name.startswith("slices."):
        where = f"{count} section(s)"
    logger.warning(
        "The job's state holds %r (in %s), which this LangSlice no longer "
        "reads; it is dropped.",
        name.removeprefix("slices."),
        where,
    )


def _rows(data: JsonObject) -> list[JsonObject]:
    """Return a state's section rows; an absent or null list is none."""
    rows = data.get("slices") or []
    if isinstance(rows, str) or not isinstance(rows, Sequence):
        raise ValueError(f"slices must be a list, not {rows!r}")
    return [as_object("slices", row) for row in rows]


def _object_or_empty(data: JsonObject, key: str) -> JsonObject:
    """Return the object under a key; an absent, null or empty one is {}."""
    if not data.get(key):
        return {}
    return object_at(data, key)


def _sorted_text(value: Json) -> str:
    """Return a JSON value's text with its keys sorted, a memo's key."""
    return json.dumps(value, sort_keys=True)


@dataclass(frozen=True)
class Migration:
    """Hold one row of the table: a named step from one format to the next.

    Parameters
    ----------
    name
        The migration's name, as the table gives it.
    reads
        The state format the step reads; it returns the next one.
    migrate
        The step: a state's fields at *reads* and the folder's
        migration, to the fields at the next format.
    """

    name: str
    reads: int
    migrate: Callable[[Fields, FolderMigration], Fields]


#: Every state migration, oldest first; the last one's next format is
#: the current one, ``state_file.STATE_FORMAT_VERSION``.
STATE_MIGRATIONS = (
    Migration("state_3_to_4", 3, state_3_to_4),
    Migration("state_4_to_5", 4, state_4_to_5),
)
