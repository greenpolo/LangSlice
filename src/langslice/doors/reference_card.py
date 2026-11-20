"""The job folder's reference card: ``AGENTS.md`` and ``CLAUDE.md``, identical.

A coding agent (Claude Code reads ``CLAUDE.md``, Codex ``AGENTS.md``) that
opens a job folder reads this first: what the folder holds, which file is the
truth and which are derived, how to get atlas coordinates for a picture, the
CLI verbs, their per-call limits, the Python import path and what a method
returns, and where the agent keeps its own files. Generated from code (the
verbs the job's run has, :func:`langslice.tools.catalog.enabled` with the
scripting verbs, their declarations and ``Verb.limits``, the job's own
transform cap), so it cannot drift from what the doors offer; written when a
job folder is created and rewritten whenever a LangSlice that would word it
differently opens the folder (:func:`write_card`). One screen: the details are
one command away (``langslice job FOLDER ops``, ``langslice job FOLDER schema VERB``).
"""

from __future__ import annotations

import logging
from collections.abc import Mapping
from pathlib import Path
from typing import Any, cast

import langslice
from langslice.core.jsonable import JsonObject
from langslice.job.folder.layout import JobLayout, read_job_file
from langslice.job.settings.spec import JobSpec
from langslice.job.settings.tools_on import ToolsOn
from langslice.tools.catalog import VERBS, enabled, verbs_for
from langslice.tools.descriptions.render import summary

logger = logging.getLogger(__name__)

CARD_FILES = ("AGENTS.md", "CLAUDE.md")
TOOLS_FILE = "TOOLS.md"


LIMIT_WORDS = {"fits": "section x candidate fits"}


def limits_note(limits: Mapping[str, int]) -> str:
    """A verb's per-call limits (``Verb.limits``) as the card words them."""
    if not limits:
        return ""
    return (
        "; at most "
        + ", ".join(f"{most} {LIMIT_WORDS.get(what, what)}" for what, most in limits.items())
        + " per call"
    )


def job_spec(layout: JobLayout) -> JobSpec:
    """The job's settings as ``job.json`` holds them (the tool plugins it
    names loaded)."""
    record = read_job_file(layout) or {}
    saved = cast("JsonObject", record.get("spec") or {})
    return JobSpec.from_dict(dict(saved))


def off_tools(spec: Any) -> ToolsOn:
    """What the card words by for a job of *spec*: its switched-off names and
    the verbs its run does not have (a task that is off takes its tools out of
    the run), so the card never names a verb the job refuses."""
    run = set(enabled(spec, scripting=True))
    return ToolsOn(
        frozenset([*spec.switched_off, *(name for name in verbs_for(spec) if name not in run)])
    )


def transform_cap_line(layout: JobLayout) -> str:
    """The job's own lower cap on the transform verbs
    (``transform.max_parallel``), when its settings set one."""
    from langslice.job.settings.spec import MAX_PARALLEL_TRANSFORMS

    held = (read_job_file(layout) or {}).get("spec") or {}
    cap = int((held.get("transform") or {}).get("max_parallel") or MAX_PARALLEL_TRANSFORMS)
    if cap >= MAX_PARALLEL_TRANSFORMS:
        return ""
    fits = off_tools(job_spec(layout)).listed(
        [
            ("elastix_affine", "`elastix_affine`"),
            ("interactive_transform", "`interactive_transform`"),
        ],
        last="or",
    )
    return f"\nThis job: at most {cap} sections per {fits} call."


def card_text(layout: JobLayout) -> str:
    """The card for the job folder *layout* (see the module text)."""
    images = str(layout.image_folder) if layout.image_folder is not None else "the images"
    spec = job_spec(layout)
    on = off_tools(spec)
    run = set(enabled(spec, scripting=True))
    from langslice.tools.descriptions.description import DescriptionFacts

    variant = DescriptionFacts.from_spec(spec, auto=True, door="cli")
    verbs = "\n".join(
        f"- `{name}` ({verb.kind}, {verb.group}{', long' if verb.long else ''}"
        f"{limits_note(verb.limits)}): "
        f"{summary(name) if name in VERBS else summary(name, variant)}"
        for name, verb in verbs_for(spec).items()
        if name in run
    ) + transform_cap_line(layout)
    example = (
        "(`job.status()`"
        + on.say(
            "position_sections",
            ',\n`job.position_sections(sections=[{"id": "<file>", "position_mm": 5.2}])`',
        )
        + ")"
    )
    history = on.say("redo", "undo and redo", "undo")
    histories = (
        f"`history/undo.json`: the {history} steps; "
        "`sections/<name>/history.jsonl`: every registration a section held."
    )
    exported = on.say(
        "export_maps", " written by `export_maps`", ", not written by this job's verbs"
    )
    export_command = on.say(
        "export_maps",
        "`export_maps` (answers within 30 s: its result, or its run id; `wait [id]`, `runs`), ",
    )
    background = on.say(
        "trace_borders",
        "`trace_borders` lands in the background: `job.close()`\n"
        "(or a `with` block) waits for it. ",
    )
    return f"""# LangSlice job folder

One registration job: the sections in `{images}` placed in a BrainGlobe atlas.
Written by LangSlice {langslice.__version__} from its code; regenerated on open, so do not edit it.

A registering agent's job is the message that launched it, made by
`langslice job {layout.folder} launch --role register` (or `register-scripting`).

## Files: settings, state and derived outputs
- `state.json`: THE TRUTH. Order, positions, flips, transforms and the applied
  deformation of every section. Change it through the verbs below; a direct
  edit by a script is picked up by a running agent as one undo step.
- `registration.json`: the same registrations in public units (read only,
  rewritten on every change): per section its parameters and `pixel_to_atlas_um`
  (image file pixel [row, col, 1] -> atlas um).
- `job.json`: settings (tasks, atlas, plane); {histories}
- `sections/<name>/`: per section `deformable/<key>/` (applied deformation
  records), `border_trace/<key>/` (image-model traces), `views/`, and the
  maps{exported} (on the image's working copy; submit does not write them):
  - `coords.tif`: float32 (3, rows, cols) atlas um per pixel, NaN outside the atlas.
  - `labels.tif`: uint32 atlas ids per pixel (0 outside the atlas); `labels_fiji.tif`
    + `labels.csv`: a uint16 index Fiji opens, and index -> id, acronym, name, RGB.
  - `residual.tif`: float32 (2, rows, cols) (drow, dcol) px of an applied deformation;
    `maps.json`: the grid, its pixel_to_atlas_um, the parameters they came from.
- `views/`, `views.jsonl` (pictures, their index); `exports/` (results, `quicknii.json`,
  `visualign.json`); `logs/` (events, background runs, CLI calls, each reply in `calls/`).
Maps and pictures are derived, never read back: editing one changes no registration.

## Pictures and coordinates
Each picture folder holds `view.jpg` (what was shown), `view.json` (its frame) and, for a section
on its atlas, `labels.tif` (uint32 atlas ids per pixel) and `borders.png` (the drawn borders).
`langslice.coordinate_map("<picture>/view.json")`: every
pixel's atlas position, (rows, cols, 3) float32. Coordinates are BrainGlobe atlas micrometres in
the atlas's own axis order (its orientation string; packaged atlases are `asr`: 0 anterior to
posterior, 1 superior to inferior, 2 right to left), voxel i's centre at i * resolution. Pixels
are (row, col), centres at integers. Section positions in `state.json` and the verbs are
millimetres along the slicing axis.

## Command line (JSON on stdout: ok, result, artifacts, warnings, next)
For the session that sets a job up and checks it, not for registering:
`langslice job {layout.folder} COMMAND`: `init` (`schema init`: its flags), `launch --role
register|register-scripting`, `status`, {export_command}`env`, `python SCRIPT`.
Exit codes: 0 ok, 2 bad arguments, 3 refused by the job (`error.code`, `error.message`), 4 internal.

## Tools
A registration agent has these as MCP tools; a script has them as methods (Python, below).
`{TOOLS_FILE}` (from `init`) describes each and its arguments;
`langslice job {layout.folder} schema TOOL [TOOL ...]` as JSON (every tool without one).
{verbs}

## Python
`import langslice; job = langslice.open_job("{layout.folder}")`; the verbs are methods
with the same arguments {example}. Each returns, once its pictures are on disk, a JSON-safe
dict: the tool's reply in full (status rows with every field, null where unset) plus
`artifacts`, the files of its pictures; `reply.images` holds the pictures as PIL images.
A script's writes and a registration agent's tools share the folder's lock; a long fit
re-checks each section before it applies (`STALE_INPUT`: run it again for that section).
{background}Atlases: `langslice.load_atlas("allen_mouse_25um")`.
Keep your own files here, never in /tmp: scripts in `scripts/`, the rest in `scratch/`.
Run a script with the job's own Python, `env/` here (made by `launch --role register-scripting`,
whose message names its interpreter, or by `langslice job {layout.folder} python
scripts/NAME.py [ARGS]`; `env install PACKAGE ...` adds packages).
Your own method (an affine, a warp field) goes in through `job.context`: `help(job.context)`.
Judge and choose fits by looking at their pictures, never by image-similarity scores (mutual
information, outline overlap, edge correlation): the fits optimise those, they track border
placement poorly, the field has turned away from them and so have we (they chose worse fits).
"""


def write_card(layout: JobLayout) -> list[Path]:
    """Write the card into the job folder where it is missing or worded
    differently; return the files written. Never raises: a card that cannot
    be written is logged."""
    text = card_text(layout)
    written: list[Path] = []
    for name in CARD_FILES:
        path = layout.folder / name
        try:
            if path.exists() and path.read_text(encoding="utf-8") == text:
                continue
            path.write_text(text, encoding="utf-8", newline="\n")
            written.append(path)
        except OSError:
            logger.warning("Could not write the reference card %s", path, exc_info=True)
    return written
