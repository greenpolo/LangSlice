"""The job folder's reference card: ``AGENTS.md`` and ``CLAUDE.md``, identical.

A coding agent (Claude Code reads ``CLAUDE.md``, Codex ``AGENTS.md``) that
opens a job folder reads this first: what the folder holds, which file is
the truth and which are derived, how to get atlas coordinates for a
picture, the CLI verbs, their per-call limits, the Python import path and
what a method returns, and where the agent keeps its own files. Generated
from code (the verbs the job's run has, :func:`langslice.ops.registry.enabled`
with the scripting verbs, their declarations and ``Verb.limits``, the job's
own transform cap), so it
cannot drift from what the doors offer; written when a job folder is
created and rewritten whenever a LangSlice that would word it differently
opens the folder (:func:`write_card`). One screen: the details are one
command away (``langslice-job ops``, ``langslice-job schema VERB``).
"""

from __future__ import annotations

import logging
from collections.abc import Mapping
from pathlib import Path
from typing import Any

import langslice
from langslice.core.needs import render
from langslice.core.toolset import TOOLS
from langslice.doors.declarations import summary
from langslice.job.layout import JobLayout, read_job_file
from langslice.ops.registry import enabled, verbs_for

logger = logging.getLogger(__name__)

#: The card's two names (Codex reads the first, Claude Code the second).
CARD_FILES = ("AGENTS.md", "CLAUDE.md")
#: The brief the agent CLI writes beside the card (``langslice-job FOLDER
#: brief``, :mod:`langslice.doors.cli.brief`), which the card names first.
BRIEF_FILE = "BRIEF.md"


#: How the card words what a limit counts (``Verb.limits``' keys).
LIMIT_WORDS = {"fits": "section x candidate fits"}


def limits_note(limits: Mapping[str, int]) -> str:
    """A verb's per-call limits (``Verb.limits``) as the card words them."""
    if not limits:
        return ""
    return "; at most " + ", ".join(f"{most} {LIMIT_WORDS.get(what, what)}"
                                    for what, most in limits.items()) + " per call"


def job_spec(layout: JobLayout) -> Any:
    """The job's settings as ``job.json`` holds them (the tool plugins it
    names loaded)."""
    from langslice.core.spec import JobSpec

    return JobSpec.from_dict((read_job_file(layout) or {}).get("spec") or {})


def off_names(spec: Any) -> list[str]:
    """What the card renders as off for a job of *spec*: its switched-off
    names and the verbs its run does not have (a task that is off takes its
    tools out of the run), so the card never names a verb the job refuses."""
    run = set(enabled(spec, scripting=True))
    return [*spec.switched_off, *(name for name in verbs_for(spec) if name not in run)]


def disabled_names(layout: JobLayout) -> list[str]:
    """The tools and options that are off for the job (:func:`off_names`)."""
    return off_names(job_spec(layout))


def transform_cap_line(layout: JobLayout) -> str:
    """The job's own lower cap on the transform verbs
    (``transform.max_parallel``), when its settings set one."""
    from langslice.core.spec import MAX_PARALLEL_TRANSFORMS

    held = (read_job_file(layout) or {}).get("spec") or {}
    cap = int((held.get("transform") or {}).get("max_parallel") or MAX_PARALLEL_TRANSFORMS)
    if cap >= MAX_PARALLEL_TRANSFORMS:
        return ""
    return render(f"\nThis job: at most {cap} sections per [[?elastix_affine `elastix_affine`"
                  "[[?interactive_transform  or ]]]][[?interactive_transform "
                  "`interactive_transform`]] call.", disabled_names(layout))


def card_text(layout: JobLayout) -> str:
    """The card for the job folder *layout* (see the module text)."""
    images = str(layout.image_folder) if layout.image_folder is not None else "the images"
    spec = job_spec(layout)
    disabled = off_names(spec)
    run = set(enabled(spec, scripting=True))
    variant = None
    if spec.plugins:  # the tools plugins add are declared in its run
        from langslice.doors.declarations import Variant

        variant = Variant.of(spec, auto=True, door="cli")
    verbs = "\n".join(
        f"- `{name}` ({verb.kind}, {verb.group}{', long' if verb.long else ''}"
        f"{limits_note(verb.limits)}): "
        f"{summary(name, variant) if variant is not None and name not in TOOLS else summary(name)}"
        for name, verb in verbs_for(spec).items() if name in run
    ) + transform_cap_line(layout)
    example = render("(`job.status()`[[?position_sections ,\n`job.position_sections("
                     'sections=[{"id": "<file>", "position_mm": 5.2}])`]])', disabled)
    history = render("[[?redo undo/redo||undo]]", disabled)
    exported = render("[[?export_maps  written by `export_maps`||, not written by this job's "
                      "verbs]]", disabled)
    background = render("[[?trace_borders `trace_borders` lands in the background: "
                        "`job.close()`\n(or a `with` block) waits for it. ]]", disabled)
    return f"""# LangSlice job folder

One registration job: the sections in `{images}` placed in a BrainGlobe atlas.
Written by LangSlice {langslice.__version__} from its code; regenerated on open, so do not edit it.

Registering it? Start with `langslice-job {layout.folder} brief`: the job statement
LangSlice's own agent gets, the user's notes and the opening pictures as files, also
written to `{BRIEF_FILE}` here. Open every picture, then work as it says.

## Files: settings, state and derived outputs
- `state.json`: THE TRUTH. Order, positions, flips, transforms and the applied
  deformation of every section. Change it through the verbs below; a direct
  edit by a script is picked up by a running agent as one undo step.
- `registration.json`: the same registrations in public units (read only,
  rewritten on every change): per section its parameters and `pixel_to_atlas_um`
  (image file pixel [row, col, 1] -> atlas um).
- `job.json`: settings (tasks, atlas, plane); `history/`: {history}, a file per step.
- `sections/<name>/`: per section `deformable/<key>/` (applied deformation
  records), `image_correction/<key>/` (image-model traces), `views/`, and the
  maps{exported} (on the image's working copy; submit does not write them):
  - `coords.tif`: float32 (3, rows, cols) atlas um per pixel, NaN outside the atlas.
  - `labels.tif`: uint32 atlas ids per pixel (0 outside the atlas); `labels_fiji.tif`
    + `labels.csv`: a uint16 index Fiji opens, and index -> id, acronym, name, RGB.
  - `residual.tif`: float32 (2, rows, cols) (drow, dcol) px of an applied deformation;
    `maps.json`: the grid, its pixel_to_atlas_um, the parameters they came from.
- `views/`, `views.jsonl` (pictures, their index); `exports/` (results, `quicknii.json`,
  `visualign.json`); `logs/` (events, CLI calls, background runs).
Maps and pictures are derived, never read back: editing one changes no registration.

## Pictures and coordinates
Each picture folder holds `view.jpg` (what was shown), `view.json` (its frame) and, for a section
on its atlas, `labels.tif` (uint32 atlas ids per pixel) and `borders.png` (the drawn borders); a
deformable fit's adds `residual.tif`. `langslice.coordinate_map("<picture>/view.json")`: every
pixel's atlas position, (rows, cols, 3) float32. Coordinates are BrainGlobe atlas micrometres in
the atlas's own axis order (its orientation string; packaged atlases are `asr`: 0 anterior to
posterior, 1 superior to inferior, 2 right to left), voxel i's centre at i * resolution. Pixels
are (row, col), centres at integers. Section positions in `state.json` and the verbs are
millimetres along the slicing axis.

## CLI (JSON on stdout: ok, picture_files, result, artifacts, warnings, next)
`langslice-job {layout.folder} VERB --args '{{...}}'` (or `--key value`); `--dry-run`
(writes nothing), `--verbose`. A long verb's call answers within 30 s: its result, or its run id
and state `running` (the run goes on, also if your shell ends the command where a process may
outlive it; `wait [id]` waits 30 s more, `runs` lists runs, `--background` answers at once).
Pictures: `picture_files` (paths) first, then `artifacts`, each with its `index` in the reply's
`pictures`. Exit codes: 0 ok, 2 bad arguments, 3 refused by the job (`error.code`,
`error.fix`), 4 internal. Calls may run in parallel (each write holds the folder's lock; a long
verb re-checks each section first: `STALE_INPUT`, run it again for that section).
`langslice-job ops` lists the verbs; `langslice-job {layout.folder} schema VERB [VERB ...]` gives
their arguments, several at once (every verb without one); `schema init` the job flags.
{verbs}

## Python
`import langslice; job = langslice.open_job("{layout.folder}")`; the verbs are methods
with the same arguments {example}. Each returns, once its pictures are on disk, a JSON-safe
dict: the CLI's `result` in full (status rows with every field, null where unset) plus
`artifacts` as the CLI lists them; `reply.images` holds the pictures as PIL images.
{background}Atlases: `langslice.load_atlas("allen_mouse_25um")`.
Keep your own files here, never in /tmp: scripts in `scripts/`, the rest (montages, tables)
in `scratch/`; LangSlice never reads them. Every picture a verb shows is saved here.
Run a script with `langslice-job {layout.folder} python scripts/NAME.py [ARGS]`: the job's
`env/` python, LangSlice usually loaded already. `env install PACKAGE ...` adds packages there.
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
            path.write_text(text, encoding="utf-8")
            written.append(path)
        except OSError:
            logger.warning("Could not write the reference card %s", path, exc_info=True)
    return written
