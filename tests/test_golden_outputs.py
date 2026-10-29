"""Pin what LangSlice produces at each door: pictures, files, maps, replies.

The golden scenario registers three real sections to ``allen_mouse_50um``
with fixed arguments and no model, once through each door, each door a test
of its own so a failure names the door that changed:

- ``library`` (``tests/golden/library_door.py``): the public library places
  the sections, sets transforms by hand, refines them with Elastix, makes
  masks and section images, marks damage, fits FireANTs SyN (coarse, on the
  CPU), applies landmark pairs, undoes and redoes, exports the maps, looks
  regions up, reads the job folder, makes calls the tools refuse and
  submits, drawing every ``look`` mode, a zoom and the status after each
  write: every tool that needs no image model.
- ``cli`` (``tests/golden/cli_door.py``): the installed ``langslice-job``
  command, one process per call: the catalog, ``init``, reads, writes, a
  dry run, a fit, refusals and a launch prompt; each call's exit code and
  printed envelope.
- ``mcp`` (``tests/golden/mcp_door.py``): the installed ``langslice mcp``
  server over stdio with the ``mcp`` library's client: the greeting, the
  tool lists before and after ``start_job`` (names, descriptions, input
  schemas, annotations), every ``show_stack`` page and a run of reads,
  writes and refusals; each result's blocks, text whole and images as
  decoded-pixel digests.

Each door's scenario runs in a child interpreter on the checkout's own
code (``PYTHONPATH``), through public entry points only, and this test reads
only the job folder's public files, so the test survives any reorganisation
of the package. The child gets a fresh home folder (``HOME``, the XDG
folders, ``TMPDIR``) and only the caller's ``PATH`` and locale, so no
setting, saved job, credential or ``.env`` key of the user's reaches it
even where the switches below are renamed; BrainGlobe's configuration in
that home points at the caller's BrainGlobe folder, where the atlases are.

What is compared with the goldens (``expected/<door>/``), exactly (no
tolerance: every engine here gave identical output run to run):

- every reply, with paths, clock times and run ids replaced by tokens
  (``calls.json``), with a digest of each picture's pixels before JPEG;
- every JSON and text file of the job folder (``state.json``,
  ``registration.json``, ``job.json``, each picture's ``view.json``,
  ``views.jsonl``, each section's ``registrations.jsonl`` and
  ``maps.json``, the deformation records, the exports, the reference cards,
  ``logs/events.jsonl``, the agent CLI's ``logs/calls.jsonl`` and saved
  replies, the launch prompts, ``logs/engine.log``), normalised the same
  way (``files.json``);
- every picture's decoded pixels (``view.jpg``, compared as pixels, never as
  JPEG bytes), each picture's ``labels.tif`` and ``borders.png``, every
  section image, mask and map (``coords.tif``, ``labels.tif``,
  ``residual.tif``...), and every array of each ``warp.ome.zarr``, as
  digests of their decoded arrays;
- the pictures the writing tools returned, kept whole in
  ``expected/<door>/pictures/`` for a reader to look at and for an exact
  pixel diff.

Not compared: the undo steps (``history/``), the locks and counters, and
what the children print on stderr (progress lines and library notices).

The three sections and the goldens are unpublished data, so they are kept
out of the repository, in ``tests/golden/data/`` (git-ignored) or the folder
``LANGSLICE_GOLDEN_DATA`` names (``sections/``, ``expected/``); the test is
skipped where they are missing.

On a mismatch the run's outputs go to ``actual/<door>/`` beside them (every
picture included) and the failure, headed by the door and the commit the
goldens were recorded at, lists what changed: the files added or removed,
each changed JSON key with its old and new value (a changed text around its
first difference), each changed array's shape and statistics, and per
picture the pixels that differ and the largest difference (whole pictures),
or the tiles of an 8 x 8 grid that changed (pictures kept only as digests).

Run it with ``python -m pytest -m golden`` (under a minute, the doors in
parallel; it is left out of the default run because it needs the cached
BrainGlobe atlases and its pixels are those of this machine's libraries).
An intended change is recorded on ``main`` with ``LANGSLICE_UPDATE_GOLDEN=1
python -m pytest -m golden``, which rewrites ``expected/``, records the
commit in each door's ``provenance.json`` and fails once, printing what
changed, so a regeneration never passes unreviewed. The goldens are not
under version control, so goldens made on a branch would let that branch's
own changes pass: the update refuses to run unless the checkout is on
``main``, or ``LANGSLICE_GOLDEN_FORCE=1`` is set as well.
"""

from __future__ import annotations

import hashlib
import json
import os
import re
import shutil
import subprocess
import sys
from collections.abc import Iterator
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path

import numpy as np
import pytest
import tifffile
import zarr
from brainglobe_atlasapi.config import get_brainglobe_dir
from PIL import Image

pytestmark = pytest.mark.golden

ROOT = Path(__file__).resolve().parent.parent
GOLDEN = ROOT / "tests" / "golden"
#: The sections and goldens are kept out of the repository (the sections
#: are unpublished data): ``tests/golden/data/`` (git-ignored), or the
#: folder this variable names.
DATA_ENV = "LANGSLICE_GOLDEN_DATA"
DATA = Path(os.environ.get(DATA_ENV) or GOLDEN / "data")
SECTIONS = DATA / "sections"
EXPECTED = DATA / "expected"
ACTUAL = DATA / "actual"
UPDATE_ENV = "LANGSLICE_UPDATE_GOLDEN"
#: Set with ``UPDATE_ENV`` to regenerate on a branch other than ``main``.
FORCE_ENV = "LANGSLICE_GOLDEN_FORCE"
#: The branch whose goldens every branch is compared with.
MAIN_BRANCH = "main"
#: The doors, each with its scenario ``tests/golden/<door>_door.py``.
DOORS = ("library", "cli", "mcp")
#: The caller's environment names a child keeps: what finds programs and
#: decodes text. Every other name (the user's LangSlice settings, model
#: keys, display) stays behind.
KEPT_ENV = ("PATH", "LANG", "LC_ALL", "LC_CTYPE", "USER", "LOGNAME")
#: Model keys set empty: python-dotenv reads a ``.env`` above the checkout
#: into a child and never replaces a name that is set.
EMPTY_ENV = (
    "ANTHROPIC_API_KEY",
    "GEMINI_API_KEY",
    "GOOGLE_API_KEY",
    "OPENAI_API_KEY",
)
#: A child has no GPU, so FireANTs runs its deterministic CPU path; no
#: atlas cache, recent-jobs list or user settings; BrainGlobe asks no
#: server for a newer atlas; the agent CLI runs each call cold and a long
#: verb in the calling process, so no process outlives the scenario.
CHILD_ENV = {
    "CUDA_VISIBLE_DEVICES": "",
    "LANGSLICE_ATLAS_CACHE": "off",
    "LANGSLICE_ATLAS_SKIP_LATEST_CHECK": "1",
    "LANGSLICE_DETACH": "0",
    "LANGSLICE_RECENT_JOBS": "off",
    "LANGSLICE_SETTINGS": "off",
    "LANGSLICE_WARM": "0",
    "PYTHONPATH": str(ROOT / "src"),
}
#: A fixed mtime for the copied sections (2026-01-01 UTC).
FIXED_MTIME = 1767225600
#: Job folder parts that are not outputs: undo steps, locks and counters.
SKIPPED_PARTS = frozenset({"history"})
SKIPPED_SUFFIXES = frozenset({".lock", ".seq"})
#: Folder names whose zarr store is read as arrays, not as files.
ZARR_SUFFIX = ".zarr"
#: Tokens for what changes between runs of identical code.
ROOT_TOKEN = "<ROOT>"
TIME_TOKEN = "<time>"
SECONDS_TOKEN = "<seconds>"
ID_TOKEN = "<id>"
#: ISO times (2026-10-29T01:17:00.81+00:00) and local times (2026-10-28
#: 21:17:01), as the job files write them.
TIME_PATTERN = re.compile(
    r"\d{4}-\d{2}-\d{2}[T ]\d{2}:\d{2}:\d{2}(\.\d+)?([+-]\d{2}:\d{2}|Z)?"
)
#: Keys whose numbers are wall-clock durations.
SECONDS_KEYS = frozenset(
    {"runtime_s", "elapsed_s", "seconds", "wall_s", "gpu_wait_s"}
)
#: Keys whose values are random per run: each tool call's id in the event
#: log.
ID_KEYS = frozenset({"execution_id"})
#: The ``source`` fields that name the checkout's commit, not the code's
#: outputs.
SOURCE_KEY = "source"
SOURCE_FIELDS = frozenset({"commit", "uncommitted_changes"})
SOURCE_TOKEN = "<source>"
#: Tiles per side of the grid a picture's digest is cut into.
TILES = 8
#: Changed keys listed per file or call before the rest are counted.
MAX_LINES = 25
#: Lines of the diff the failure prints; the whole diff is written to
#: ``actual/<door>/diff.txt``.
MAX_PRINTED = 150
#: Characters of a changed text shown before and after its first
#: difference.
CONTEXT = 50
#: Tools whose pictures are kept whole: the writes.
KEPT_TOOLS = (
    "position_sections",
    "interactive_transform",
    "elastix_affine",
    "make_mask",
    "make_image",
    "mark_damage",
    "fireants_syn",
    "set_landmarks",
    "landmark_triangulation",
)


# --- the run ---------------------------------------------------------------


@dataclass(frozen=True)
class Run:
    """One run of a door's scenario: where its job is and what it recorded."""

    root: Path
    job: Path
    calls: list[object]


def child_environment(root: Path) -> dict[str, str]:
    """Return a child's environment, its home a fresh folder under *root*."""
    home = root / "home"
    brainglobe_config = home / ".config" / "brainglobe"
    brainglobe_config.mkdir(parents=True)
    atlases = get_brainglobe_dir()
    (brainglobe_config / "bg_config.conf").write_text(
        "[default_dirs]\n"
        f"brainglobe_dir = {atlases}\n"
        f"interm_download_dir = {atlases}\n"
    )
    runtime = home / "runtime"
    runtime.mkdir(mode=0o700)
    (root / "tmp").mkdir()
    environment = {
        name: os.environ[name] for name in KEPT_ENV if name in os.environ
    }
    environment.update(dict.fromkeys(EMPTY_ENV, ""))
    environment.update(CHILD_ENV)
    environment.update(
        HOME=str(home),
        XDG_CONFIG_HOME=str(home / ".config"),
        XDG_CACHE_HOME=str(home / ".cache"),
        XDG_DATA_HOME=str(home / ".local" / "share"),
        XDG_STATE_HOME=str(home / ".local" / "state"),
        XDG_RUNTIME_DIR=str(runtime),
        BRAINGLOBE_CONFIG_DIR=str(brainglobe_config),
        TMPDIR=str(root / "tmp"),
    )
    return environment


def run_door(door: str, root: Path) -> Run:
    """Copy the sections under *root*, run *door*'s scenario, read calls."""
    images = root / "images"
    images.mkdir(parents=True)
    for source in sorted(SECTIONS.glob("*.png")):
        target = images / source.name
        shutil.copyfile(source, target)
        os.utime(target, (FIXED_MTIME, FIXED_MTIME))
    calls_path = root / "calls.json"
    command = [
        sys.executable,
        str(GOLDEN / f"{door}_door.py"),
        str(images),
        str(calls_path),
    ]
    finished = subprocess.run(
        command,
        cwd=root,
        env=child_environment(root),
        capture_output=True,
        text=True,
        timeout=900,
        check=False,
    )
    if finished.returncode != 0:
        pytest.fail(
            f"the {door} door's scenario failed:\n" + finished.stderr[-4000:]
        )
    calls = json.loads(calls_path.read_text())
    return Run(root=root, job=images / "langslice", calls=calls)


# --- normalising -------------------------------------------------------------


class Normaliser:
    """Replace a run's paths, clock times, durations and ids with tokens."""

    def __init__(self, root: Path) -> None:
        spellings = {str(root), os.path.realpath(root)}
        self.roots = sorted(spellings, key=len, reverse=True)

    def text(self, value: str) -> str:
        """Return *value* with its root and times replaced."""
        for root in self.roots:
            value = value.replace(root, ROOT_TOKEN)
        return TIME_PATTERN.sub(TIME_TOKEN, value)

    def data(self, value: object, key: str = "") -> object:
        """Return the JSON value *value* with every string normalised."""
        if key in SECONDS_KEYS and isinstance(value, (int, float)):
            return SECONDS_TOKEN
        if key in ID_KEYS and isinstance(value, str):
            return ID_TOKEN
        if key == SOURCE_KEY and isinstance(value, dict):
            value = {
                name: SOURCE_TOKEN if name in SOURCE_FIELDS else item
                for name, item in value.items()
            }
        if isinstance(value, dict):
            return {
                self.text(str(name)): self.data(item, str(name))
                for name, item in value.items()
            }
        if isinstance(value, list):
            return [self.data(item) for item in value]
        if isinstance(value, str):
            return self.text(value)
        return value


# --- reading the job folder ---------------------------------------------


def array_entry(array: np.ndarray) -> dict[str, object]:
    """Return the digest and statistics of one decoded array."""
    contiguous = np.ascontiguousarray(array)
    entry: dict[str, object] = {
        "shape": list(contiguous.shape),
        "dtype": str(contiguous.dtype),
        "sha256": hashlib.sha256(contiguous.tobytes()).hexdigest(),
    }
    values = contiguous.astype(np.float64)
    finite = values[np.isfinite(values)]
    entry["nan"] = int(values.size - finite.size)
    if finite.size:
        entry["min"] = float(finite.min())
        entry["max"] = float(finite.max())
        entry["mean"] = round(float(finite.mean()), 9)
    return entry


def tile_digests(pixels: np.ndarray) -> list[str]:
    """Return a short digest of each tile of a TILES x TILES grid."""
    rows = np.array_split(np.arange(pixels.shape[0]), TILES)
    cols = np.array_split(np.arange(pixels.shape[1]), TILES)
    digests = []
    for row in rows:
        for col in cols:
            tile = pixels[row[0] : row[-1] + 1, col[0] : col[-1] + 1]
            digest = hashlib.sha256(np.ascontiguousarray(tile).tobytes())
            digests.append(digest.hexdigest()[:8])
    return digests


def read_rgb(path: Path) -> np.ndarray:
    """Return a picture's decoded RGB pixels."""
    with Image.open(path) as image:
        return np.asarray(image.convert("RGB"))


def picture_entry(path: Path) -> dict[str, object]:
    """Return the size, pixel digest and tile digests of a ``view.jpg``."""
    pixels = read_rgb(path)
    return {
        "size": [pixels.shape[1], pixels.shape[0]],
        "sha256": hashlib.sha256(pixels.tobytes()).hexdigest(),
        "tiles": tile_digests(pixels),
    }


def zarr_entries(store: Path, normaliser: Normaliser) -> dict[str, object]:
    """Return an entry per array and group metadata of a zarr store."""
    group = zarr.open_group(str(store), mode="r")
    entries: dict[str, object] = {
        "attributes": normaliser.data(group.attrs.asdict())
    }
    for name, array in group.arrays():
        entries[name] = array_entry(np.asarray(array[...]))
    for name, _group in group.groups():
        inner_entries = zarr_entries(store / name, normaliser)
        for inner, entry in inner_entries.items():
            entries[f"{name}/{inner}"] = entry
    return entries


def npz_entry(path: Path, normaliser: Normaliser) -> dict[str, object]:
    """Return an entry per array of an ``.npz``; text arrays as text."""
    entries: dict[str, object] = {}
    with np.load(path) as arrays:
        for name in arrays.files:
            array = arrays[name]
            if array.dtype.kind in "USO":
                entries[name] = normaliser.data(array.tolist())
            else:
                entries[name] = array_entry(array)
    return entries


def file_entry(path: Path, normaliser: Normaliser) -> object:
    """Return what is pinned of one job-folder file."""
    suffix = path.suffix
    if suffix == ".json":
        return normaliser.data(json.loads(path.read_text()))
    if suffix == ".jsonl":
        lines = path.read_text().splitlines()
        return [normaliser.data(json.loads(line)) for line in lines if line]
    if path.name == "view.jpg":
        return picture_entry(path)
    if suffix == ".png":
        with Image.open(path) as image:
            return array_entry(np.asarray(image))
    if suffix == ".tif":
        return array_entry(tifffile.imread(path))
    if suffix == ".npz":
        return npz_entry(path, normaliser)
    return normaliser.text(path.read_text()).splitlines()


def job_files(job: Path) -> Iterator[Path]:
    """Yield the job folder's output files and zarr stores, sorted."""
    for path in sorted(job.rglob("*")):
        parts = path.relative_to(job).parts
        if parts[0] in SKIPPED_PARTS or path.suffix in SKIPPED_SUFFIXES:
            continue
        inside_store = any(part.endswith(ZARR_SUFFIX) for part in parts[:-1])
        if path.name.endswith(ZARR_SUFFIX) or (
            path.is_file() and not inside_store
        ):
            yield path


@dataclass(frozen=True)
class Outputs:
    """What one run produced, normalised: replies, files and pictures."""

    calls: object
    files: dict[str, object]
    #: Every ``view.jpg`` by its flat name (``000012_elastix_affine...``).
    pictures: dict[str, Path]


def collect(run: Run) -> Outputs:
    """Read a run's replies and job folder into comparable values."""
    normaliser = Normaliser(run.root)
    files: dict[str, object] = {}
    pictures: dict[str, Path] = {}
    for path in job_files(run.job):
        name = path.relative_to(run.job).as_posix()
        if path.name.endswith(ZARR_SUFFIX):
            files[name] = zarr_entries(path, normaliser)
            continue
        files[name] = file_entry(path, normaliser)
        if path.name == "view.jpg":
            pictures[path.parent.name] = path
    return Outputs(
        calls=normaliser.data(run.calls), files=files, pictures=pictures
    )


def is_kept(picture: str) -> bool:
    """Return whether a picture is kept whole in the goldens."""
    tool = picture.split("_", 1)[1]
    return picture.startswith("000001_") or tool.startswith(KEPT_TOOLS)


# --- writing and reading the goldens ------------------------------------


def dump(value: object) -> str:
    """Return the one JSON spelling goldens are written and compared in."""
    return (
        json.dumps(value, indent=1, sort_keys=True, ensure_ascii=False) + "\n"
    )


def write_outputs(
    outputs: Outputs, folder: Path, *, every_picture: bool
) -> None:
    """Write *outputs* as goldens (kept pictures) or as a failed run's."""
    if folder.exists():
        shutil.rmtree(folder)
    (folder / "pictures").mkdir(parents=True)
    (folder / "calls.json").write_text(dump(outputs.calls))
    (folder / "files.json").write_text(dump(outputs.files))
    for name, path in outputs.pictures.items():
        if every_picture or is_kept(name):
            shutil.copyfile(path, folder / "pictures" / f"{name}.jpg")


def git(*arguments: str) -> str:
    """Return what ``git`` prints for *arguments* in this checkout."""
    finished = subprocess.run(
        ["git", "-C", str(ROOT), *arguments],
        capture_output=True,
        text=True,
        check=True,
    )
    return finished.stdout.strip()


def provenance() -> dict[str, object]:
    """Return the checkout the goldens are being generated from."""
    return {
        "commit": git("rev-parse", "HEAD"),
        "branch": git("rev-parse", "--abbrev-ref", "HEAD"),
        "uncommitted_changes": git("diff", "--name-only", "HEAD").split(),
        "generated_at": datetime.now(UTC).isoformat(timespec="seconds"),
    }


def recorded_at(folder: Path) -> str:
    """Return the commit and date a door's goldens were generated at."""
    path = folder / "provenance.json"
    if not path.exists():
        return "an unknown commit (no provenance.json)"
    record = json.loads(path.read_text())
    n_changed = len(record["uncommitted_changes"])
    changes = f" with {n_changed} files uncommitted" * bool(n_changed)
    return (
        f"commit {record['commit'][:12]} ({record['branch']}{changes}, "
        f"{record['generated_at']})"
    )


def check_update_allowed() -> None:
    """Refuse a regeneration on a branch other than main, unless forced."""
    branch = git("rev-parse", "--abbrev-ref", "HEAD")
    if branch == MAIN_BRANCH or os.environ.get(FORCE_ENV):
        return
    pytest.fail(
        f"{UPDATE_ENV}=1 regenerates the goldens only on {MAIN_BRANCH} "
        f"(this checkout is on {branch!r}). The goldens are not under "
        "version control, so goldens made on a branch would let that "
        "branch's own changes pass. Compare the branch with goldens made "
        f"on {MAIN_BRANCH}, or set {FORCE_ENV}=1 as well to regenerate "
        "here on purpose."
    )


# --- the readable diff -------------------------------------------------------


def flatten(value: object, prefix: str = "") -> dict[str, object]:
    """Return the leaves of a JSON value keyed by their dotted path."""
    if isinstance(value, dict) and value:
        leaves: dict[str, object] = {}
        for key, item in value.items():
            leaves.update(flatten(item, f"{prefix}.{key}" if prefix else key))
        return leaves
    if isinstance(value, list) and value:
        leaves = {}
        for index, item in enumerate(value):
            leaves.update(flatten(item, f"{prefix}[{index}]"))
        return leaves
    return {prefix: value}


def short(value: object) -> str:
    """Return a value's JSON, cut to one readable line."""
    text = json.dumps(value, ensure_ascii=False)
    if len(text) > 100:
        return text[:97] + "..."
    return text


def text_change(old: str, new: str) -> str:
    """Return two texts around their first difference, on one line."""
    first = next(
        (i for i, (a, b) in enumerate(zip(old, new, strict=False)) if a != b),
        min(len(old), len(new)),
    )
    start = max(first - CONTEXT, 0)
    old_part = json.dumps(old[start : first + CONTEXT], ensure_ascii=False)
    new_part = json.dumps(new[start : first + CONTEXT], ensure_ascii=False)
    return f"at character {first}: ...{old_part}... -> ...{new_part}..."


def json_diff(old: object, new: object) -> list[str]:
    """Return one line per changed, added or removed key."""
    old_leaves, new_leaves = flatten(old), flatten(new)
    lines = []
    for key in sorted(old_leaves.keys() | new_leaves.keys()):
        if key not in new_leaves:
            lines.append(f"  - {key}: {short(old_leaves[key])}")
        elif key not in old_leaves:
            lines.append(f"  + {key}: {short(new_leaves[key])}")
        elif old_leaves[key] == new_leaves[key]:
            continue
        elif isinstance(old_leaves[key], str) and isinstance(
            new_leaves[key], str
        ):
            change = text_change(old_leaves[key], new_leaves[key])
            lines.append(f"  ~ {key}: {change}")
        else:
            lines.append(
                f"  ~ {key}: {short(old_leaves[key])} -> "
                f"{short(new_leaves[key])}"
            )
    if len(lines) > MAX_LINES:
        lines = lines[:MAX_LINES] + [f"  ... {len(lines) - MAX_LINES} more"]
    return lines


def pixel_diff(old_path: Path, new_path: Path) -> str:
    """Return how many pixels of a kept picture differ, and by how much."""
    old, new = read_rgb(old_path), read_rgb(new_path)
    if old.shape != new.shape:
        return (
            f"size {old.shape[1]}x{old.shape[0]} -> "
            f"{new.shape[1]}x{new.shape[0]}"
        )
    difference = np.abs(old.astype(np.int16) - new.astype(np.int16))
    changed = difference.max(axis=-1) > 0
    if not changed.any():
        return "decoded pixels equal to the kept picture's"
    rows, cols = np.nonzero(changed)
    return (
        f"{int(changed.sum())} of {changed.size} pixels differ, largest "
        f"difference {int(difference.max())}, within x {cols.min()}-"
        f"{cols.max()} y {rows.min()}-{rows.max()}"
    )


def tile_diff(
    old: dict[str, object], new: dict[str, object], door: str
) -> str:
    """Return how many tiles of a digest-only picture changed, and where."""
    if old["size"] != new["size"]:
        return f"size {old['size']} -> {new['size']}"
    width, height = new["size"]
    pairs = zip(old["tiles"], new["tiles"], strict=True)
    changed = [index for index, (a, b) in enumerate(pairs) if a != b]
    col_edges = [part[0] for part in np.array_split(np.arange(width), TILES)]
    row_edges = [part[0] for part in np.array_split(np.arange(height), TILES)]
    col_edges.append(width)
    row_edges.append(height)
    rows = [index // TILES for index in changed]
    cols = [index % TILES for index in changed]
    return (
        f"{len(changed)} of {TILES * TILES} tiles differ, within x "
        f"{col_edges[min(cols)]}-{col_edges[max(cols) + 1] - 1} y "
        f"{row_edges[min(rows)]}-{row_edges[max(rows) + 1] - 1} (only "
        f"its digest is kept; the new picture is in "
        f"{ACTUAL / door}/pictures)"
    )


def picture_line(
    name: str, old: object, new: object, outputs: Outputs, door: str
) -> str:
    """Return the diff line of one changed ``view.jpg``."""
    stem = name.split("/")[-2]
    kept = EXPECTED / door / "pictures" / f"{stem}.jpg"
    if kept.exists():
        detail = pixel_diff(kept, outputs.pictures[stem])
    else:
        detail = tile_diff(old, new, door)
    return f"{name}: {detail}"


def files_diff(
    old: dict[str, object], outputs: Outputs, door: str
) -> list[str]:
    """Return the readable diff of a ``files.json`` and a run's files."""
    new = outputs.files
    lines = [f"removed: {name}" for name in sorted(old.keys() - new.keys())]
    lines += [f"added: {name}" for name in sorted(new.keys() - old.keys())]
    for name in sorted(old.keys() & new.keys()):
        if old[name] == new[name]:
            continue
        if name.endswith("view.jpg"):
            line = picture_line(name, old[name], new[name], outputs, door)
            lines.append(line)
        else:
            lines.append(f"{name}:")
            lines += json_diff(old[name], new[name])
    return lines


def calls_diff(old: object, new: object) -> list[str]:
    """Return the readable diff of two ``calls.json`` contents."""
    lines = []
    old_calls = old if isinstance(old, list) else []
    new_calls = new if isinstance(new, list) else []
    if len(old_calls) != len(new_calls):
        lines.append(f"calls: {len(old_calls)} -> {len(new_calls)}")
    for index, (a, b) in enumerate(zip(old_calls, new_calls, strict=False)):
        if a != b:
            lines.append(f"call {index} ({b.get('verb')}):")
            lines += json_diff(a, b)
    return lines


# --- the test ----------------------------------------------------------------


def differences(outputs: Outputs, door: str) -> list[str]:
    """List what *outputs* change against the goldens; empty when nothing."""
    expected = EXPECTED / door
    expected_calls = (expected / "calls.json").read_text()
    expected_files = (expected / "files.json").read_text()
    actual_calls, actual_files = dump(outputs.calls), dump(outputs.files)
    # compared as text: a NaN in a file never equals itself as a value
    if actual_calls == expected_calls and actual_files == expected_files:
        return []

    call_lines = calls_diff(
        json.loads(expected_calls), json.loads(actual_calls)
    )
    file_lines = files_diff(json.loads(expected_files), outputs, door)
    n_calls = sum(line.startswith("call ") for line in call_lines)
    n_pictures = sum("view.jpg:" in line for line in file_lines)
    n_files = sum(not line.startswith(" ") for line in file_lines)
    return [
        f"{n_calls} replies, {n_files} files ({n_pictures} pictures) differ",
        *call_lines,
        *file_lines,
    ]


def printed(lines: list[str], door: str) -> str:
    """Join *lines* for a failure message, cut to ``MAX_PRINTED``."""
    if len(lines) > MAX_PRINTED:
        lines = lines[:MAX_PRINTED] + [
            f"... {len(lines) - MAX_PRINTED} more lines in "
            f"{ACTUAL / door}/diff.txt"
        ]
    return "\n".join(lines)


@pytest.mark.parametrize("door", DOORS)
def test_golden_outputs(door: str, tmp_path: Path) -> None:
    """Run one door's scenario and compare all it produced with the goldens."""
    if not SECTIONS.is_dir():
        pytest.skip(f"no golden data in {DATA} (set {DATA_ENV})")
    updating = bool(os.environ.get(UPDATE_ENV))
    if updating:
        check_update_allowed()
    expected, actual = EXPECTED / door, ACTUAL / door
    outputs = collect(run_door(door, tmp_path))
    lines = differences(outputs, door) if expected.is_dir() else ["no goldens"]
    if actual.exists():
        shutil.rmtree(actual)
    if not lines:
        return

    write_outputs(outputs, actual, every_picture=True)
    (actual / "diff.txt").write_text("\n".join(lines) + "\n")
    if updating:
        # the goldens are not under version control: the failure carries
        # the diff, so a regeneration is always reviewed
        write_outputs(outputs, expected, every_picture=False)
        (expected / "provenance.json").write_text(dump(provenance()))
        pytest.fail(
            f"the {door} door: regenerated {expected} at "
            f"{recorded_at(expected)}; it changed:\n" + printed(lines, door)
        )
    pytest.fail(
        f"the {door} door's outputs differ from {expected}, recorded at "
        f"{recorded_at(expected)} (this run's are in {actual}):\n"
        + printed(lines, door)
    )
