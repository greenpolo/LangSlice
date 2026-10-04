"""The job folder's reference card: ``AGENTS.md`` and ``CLAUDE.md``, identical.

A coding agent (Claude Code reads ``CLAUDE.md``, Codex ``AGENTS.md``) that
opens a job folder reads this first: what the folder holds, which file is
the truth and which are derived, how to get atlas coordinates for a
picture, the CLI verbs and the Python import path. Generated from code (the
verbs from :data:`langslice.ops.registry.VERBS` and their declarations), so
it cannot drift from what the doors offer; written when a job folder is
created and rewritten whenever a LangSlice that would word it differently
opens the folder (:func:`write_card`). One screen: the details are one
command away (``langslice ops``, ``langslice schema VERB``).
"""

from __future__ import annotations

import logging
from pathlib import Path

import langslice
from langslice.doors.declarations import summary
from langslice.job.layout import JobLayout
from langslice.ops.registry import VERBS

logger = logging.getLogger(__name__)

#: The card's two names (Codex reads the first, Claude Code the second).
CARD_FILES = ("AGENTS.md", "CLAUDE.md")


def card_text(layout: JobLayout) -> str:
    """The card for the job folder *layout* (see the module text)."""
    images = str(layout.image_folder) if layout.image_folder is not None else "the images"
    verbs = "\n".join(
        f"- `{name}` ({verb.kind}, {verb.group}): {summary(name)}"
        for name, verb in VERBS.items()
    )
    return f"""# LangSlice job folder

One registration job: the sections in `{images}` placed in a BrainGlobe atlas.
Written by LangSlice {langslice.__version__} from its code; regenerated on open,
so do not edit it.

## Files: state is truth, pictures are derived
- `state.json`: THE TRUTH. Order, positions, flips, transforms and the applied
  deformation of every section. Change it through the verbs below; a direct
  edit by a script is picked up by a running agent as one undo step.
- `job.json`: settings (tasks, atlas, plane) and format version.
- `history/`: undo and redo, one file per step.
- `sections/<name>/`: per section, `deformable/<key>/` (applied deformation
  records), `image_correction/<key>/` (image-model traces), `views/`.
- `views/` (several sections or none), `views.jsonl` (index of every picture).
- `exports/` results, `logs/` events and background runs (`logs/runs/`).
Pictures, layers and maps are derived: regenerate them, never edit them as truth.

## Pictures and coordinates
Each picture folder holds `view.jpg` (what was shown), `view.json` (its frame)
and, for a section on its atlas, `labels.tif` (uint32 atlas ids per pixel) and
`borders.png` (the drawn borders). Atlas coordinates of every pixel:
`langslice.coordinate_map("<picture folder>/view.json")` gives a (rows, cols, 3)
float32 array. Coordinates are BrainGlobe atlas micrometres in the atlas's own
axis order (its orientation string; packaged atlases are `asr`: 0 anterior to
posterior, 1 superior to inferior, 2 right to left), voxel i's centre at
i * resolution. Pixels are (row, col), centres at integers. Section positions
in `state.json` and the verbs are millimetres along the slicing axis.

## CLI (JSON on stdout: ok, result, artifacts, warnings, next)
`langslice job {layout.folder} VERB --args '{{...}}'` (or `--key value`);
`--dry-run` (writes nothing), `--background` then `wait [id]` (`runs` lists
background runs), `--verbose`.
Pictures come back as file paths under `artifacts`. Exit codes: 0 ok,
2 bad arguments, 3 refused by the job (`error.code`, `error.fix`), 4 internal.
`langslice ops` lists the verbs; `langslice schema VERB` gives the arguments.
{verbs}

## Python
`import langslice; job = langslice.open_job("{layout.folder}")`, then the
verbs as methods with the same arguments: `job.status()`,
`job.set_positions(entries=[{{"id": "<file>", "position_mm": 5.2}}])`.
Atlases: `langslice.load_atlas("allen_mouse_25um")` (BrainGlobe).
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
