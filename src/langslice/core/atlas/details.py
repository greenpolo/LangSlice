"""BrainGlobe's own description of each atlas, for the job statement.

BrainGlobe's documentation describes every atlas it packages in a paragraph
(its "Atlas details" page): what its reference images are (Nissl, two-photon,
light-sheet, MRI), and warnings such as an additional reference that does not
line up with the annotation. Its packaged metadata carries none of this, so
LangSlice keeps the page's paragraphs in ``brainglobe_atlas_details.json``,
keyed by atlas name (``ccfv2_mouse_25um``), and the job statement quotes the
one for the job's atlas (:func:`atlas_description`). The page is CC BY 4.0
(BrainGlobe); the file records the commit it was read at.

Refresh the file from the current page with
``python -m langslice.core.atlas.details``.
"""

from __future__ import annotations

import json
import re
from functools import cache
from pathlib import Path

DETAILS_FILE = Path(__file__).with_name("brainglobe_atlas_details.json")
REPOSITORY = "brainglobe/brainglobe.github.io"
PAGE = "docs/source/documentation/brainglobe-atlasapi/usage/atlas-details.md"

_LINK = re.compile(r"!?\[([^[\]]*)\]\([^)]*\)")
_KEY = re.compile(r"`([A-Za-z0-9_.\-]+)`")


def _plain(text: str) -> str:
    """Markdown as plain text: links and badges become their words, spaces collapse."""
    while _LINK.search(text):  # badges (images) go; a link keeps its words
        text = _LINK.sub(lambda match: "" if match.group(0).startswith("!") else match.group(1),
                         text)
    return re.sub(r"\s+", " ", text.replace(" (see below)", "")).strip()


def parse(markdown: str) -> dict[str, str]:
    """Each atlas name the page lists in a section's bullets, with the
    section's paragraph (and the version's own note, such as "LSFM template
    at 20 um resolution", where the page gives one). Names are lower case."""
    out: dict[str, str] = {}
    for section in re.split(r"^#{2,3} ", markdown, flags=re.M)[1:]:
        title, _, body = section.partition("\n")
        lines = body.splitlines()
        first = next((i for i, line in enumerate(lines)
                      if line.lstrip().startswith("*") and _KEY.search(line)), None)
        if first is None:
            continue
        about = [line for line in lines[:first] if not line.rstrip().endswith(":")
                 or "available" not in line.lower()]
        description = f"{_plain(title)}: {_plain(' '.join(about))}"
        for line in lines[first:]:
            keys = _KEY.findall(line)
            if not keys or not line.lstrip().startswith("*"):
                continue
            _, _, note = _plain(line).partition(" - ")
            out[keys[0].lower()] = description + (
                f" (This version: {note.strip()}.)" if note.strip() else "")
    return out


@cache
def _details() -> dict[str, str]:
    try:
        return dict(json.loads(DETAILS_FILE.read_text(encoding="utf-8"))["atlases"])
    except (OSError, ValueError, KeyError):
        return {}


def atlas_description(atlas_name: str) -> str | None:
    """BrainGlobe's description of *atlas_name*, or None for an atlas its page
    does not list."""
    return _details().get(atlas_name.lower())


def _refresh() -> None:  # pragma: no cover - reads the network
    from urllib.request import urlopen

    latest = f"https://api.github.com/repos/{REPOSITORY}/commits?path={PAGE}&per_page=1"
    with urlopen(latest) as reply:
        commit = json.load(reply)[0]["sha"]
    with urlopen(f"https://raw.githubusercontent.com/{REPOSITORY}/{commit}/{PAGE}") as reply:
        atlases = parse(reply.read().decode("utf-8"))
    DETAILS_FILE.write_text(json.dumps({
        "source": f"https://github.com/{REPOSITORY}/blob/{commit}/{PAGE}",
        "licence": "CC BY 4.0, BrainGlobe",
        "atlases": dict(sorted(atlases.items())),
    }, indent=1, ensure_ascii=False) + "\n", encoding="utf-8")
    print(f"{len(atlases)} atlases from {REPOSITORY}@{commit[:10]} -> {DETAILS_FILE}")


if __name__ == "__main__":  # pragma: no cover
    _refresh()
