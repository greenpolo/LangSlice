"""The job folder's reference card: ``AGENTS.md`` and ``CLAUDE.md``, identical.

A coding agent (Claude Code reads ``CLAUDE.md``, Codex ``AGENTS.md``) that
opens a job folder reads this first: what the folder holds, which file is
the truth and which are derived, how to get atlas coordinates for a
picture, the CLI verbs, their per-call limits, the Python import path and
what a method returns, and where the agent keeps its own files. Generated
from code (the listed verbs, :func:`langslice.ops.registry.listed`, their
declarations and ``Verb.limits``, the job's own transform cap), so it
cannot drift from what the doors offer; written when a job folder is
created and rewritten whenever a LangSlice that would word it differently
opens the folder (:func:`write_card`). One screen: the details are one
command away (``langslice-job ops``, ``langslice-job schema VERB``).
"""

from __future__ import annotations

import logging
from collections.abc import Mapping
from pathlib import Path

import langslice
from langslice.doors.declarations import summary
from langslice.job.layout import JobLayout, read_job_file
from langslice.ops.registry import listed

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


def transform_cap_line(layout: JobLayout) -> str:
    """The job's own lower cap on the transform verbs
    (``transform.max_parallel``), when its settings set one."""
    from langslice.core.spec import MAX_PARALLEL_TRANSFORMS

    try:
        held = (read_job_file(layout) or {}).get("spec") or {}
        cap = int((held.get("transform") or {}).get("max_parallel") or MAX_PARALLEL_TRANSFORMS)
    except (OSError, ValueError, TypeError, AttributeError):
        return ""
    if cap >= MAX_PARALLEL_TRANSFORMS:
        return ""
    return (f"\nThis job: at most {cap} sections per `elastix_affine` or `interactive_transform` "
            "call.")


def card_text(layout: JobLayout) -> str:
    """The card for the job folder *layout* (see the module text)."""
    images = str(layout.image_folder) if layout.image_folder is not None else "the images"
    verbs = "\n".join(
        f"- `{name}` ({verb.kind}, {verb.group}{', long' if verb.long else ''}"
        f"{limits_note(verb.limits)}): {summary(name)}"
        for name, verb in listed().items()
    ) + transform_cap_line(layout)
    return f"""# LangSlice job folder

One registration job: the sections in `{images}` placed in a BrainGlobe atlas.
Written by LangSlice {langslice.__version__} from its code; regenerated on open, so do not edit it.

Registering it? Start with `langslice-job {layout.folder} brief`: the job statement
LangSlice's own agent gets, the user's notes and the opening pictures as files, also
written to `{BRIEF_FILE}` here. Open every picture, then work as it says.

## Files: state is truth, everything else is derived
- `state.json`: THE TRUTH. Order, positions, flips, transforms and the applied
  deformation of every section. Change it through the verbs below; a direct
  edit by a script is picked up by a running agent as one undo step.
- `registration.json`: the same registrations in public units (read only,
  rewritten on every change): per section its parameters and `pixel_to_atlas_um`
  (image file pixel [row, col, 1] -> atlas um).
- `job.json`: settings (tasks, atlas, plane); `history/`: undo/redo, a file per step.
- `sections/<name>/`: per section `deformable/<key>/` (applied deformation
  records), `image_correction/<key>/` (image-model traces), `views/`, and the
  maps written at submit or by `export_maps` (on the image's working copy):
  - `coords.tif`: float32 (3, rows, cols) atlas um per pixel over the section's filled
    outline, NaN outside it; `tissue.png`: the tissue estimate (0/255), to mask with.
  - `labels.tif`: uint32 atlas ids per pixel (0 outside the outline); `labels_fiji.tif`
    + `labels.csv`: a uint16 index Fiji opens, and index -> id, acronym, name, RGB.
  - `residual.tif`: float32 (2, rows, cols) (drow, dcol) px of an applied deformation;
    `maps.json`: the grid, its pixel_to_atlas_um, the parameters they came from.
- `views/`, `views.jsonl` (pictures, their index); `exports/` (results, `quicknii.json`,
  `visualign.json`); `logs/` (events, CLI calls, background runs).
Maps and pictures are derived, never read back: editing one changes no registration.

## Pictures and coordinates
Each picture folder holds `view.jpg` (what was shown), `view.json` (its frame)
and, for a section on its atlas, `labels.tif` (uint32 atlas ids per pixel) and
`borders.png` (the drawn borders); a deformable fit's adds `residual.tif`.
`langslice.coordinate_map("<picture>/view.json")`: every pixel's atlas position,
(rows, cols, 3) float32. Coordinates are BrainGlobe atlas micrometres in the atlas's own
axis order (its orientation string; packaged atlases are `asr`: 0 anterior to
posterior, 1 superior to inferior, 2 right to left), voxel i's centre at
i * resolution. Pixels are (row, col), centres at integers. Section positions
in `state.json` and the verbs are millimetres along the slicing axis.

## CLI (JSON on stdout: ok, result, artifacts, warnings, next)
`langslice-job {layout.folder} VERB --args '{{...}}'` (or `--key value`); `--dry-run`
(writes nothing), `--background` then `wait [id]` (`runs` lists background runs), `--verbose`.
Pictures come back as file paths under `artifacts`, each with its `index` in the
reply's `pictures` (their numbers). Exit codes: 0 ok, 2 bad arguments, 3 refused by the job
(`error.code`, `error.fix`), 4 internal. Calls may run in parallel (each write holds the
folder's lock; a long verb re-checks each section first: `STALE_INPUT`, run it again for
that section). `langslice-job ops` lists the verbs; `langslice-job schema VERB` (or
`VERB --help`) gives the arguments, `langslice-job schema init` the job flags.
{verbs}

## Python
`import langslice; job = langslice.open_job("{layout.folder}")`; the verbs are methods
with the same arguments (`job.status()`,
`job.position_sections(sections=[{{"id": "<file>", "position_mm": 5.2}}])`). Each returns, once
its pictures are on disk, a JSON-safe dict: the CLI's `result` in full (status rows with
every field, null where unset) plus `artifacts` as the CLI lists them; `reply.images`
holds the pictures as PIL images. `trace_borders` lands in the background: `job.close()`
(or a `with` block) waits for it. Atlases: `langslice.load_atlas("allen_mouse_25um")`.
Keep your own files here, never in /tmp: scripts in `scripts/`, the rest (montages,
tables) in `scratch/`; LangSlice never reads them. Every picture a verb shows is saved here.
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
