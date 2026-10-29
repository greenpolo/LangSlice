"""Pin what LangSlice produces: pictures, registrations, maps and replies.

The golden scenario (``tests/golden/scenario.py``) registers three real
sections to ``allen_mouse_50um`` with fixed arguments and no model: it places
them, sets transforms by hand, refines them with Elastix, makes a mask, marks
damage, fits FireANTs SyN (coarse, on the CPU), applies landmark pairs, undoes
and redoes, and exports the maps, drawing every ``look`` mode, a zoom and the
status after each write. It runs in a child interpreter through the public
library only, and this test reads only the job folder's public files, so the
test survives any reorganisation of the package.

What is compared with the goldens (``expected/``), exactly (no tolerance:
every engine here gave identical output run to run):

- every tool reply, paths and clock times replaced by tokens
  (``calls.json``), with a digest of each picture's pixels before JPEG;
- every JSON file of the job folder (``state.json``, ``registration.json``,
  ``job.json``, each picture's ``view.json``, ``views.jsonl``, each
  section's ``registrations.jsonl`` and ``maps.json``, the deformation
  records, the exports), normalised the same way (``files.json``);
- every picture's decoded pixels (``view.jpg``, compared as pixels, never as
  JPEG bytes), each picture's ``labels.tif`` and ``borders.png``, every
  section image, mask and map (``coords.tif``, ``labels.tif``,
  ``residual.tif``...), and every array of each ``warp.ome.zarr``, as
  digests of their decoded arrays;
- the pictures the writing tools returned, kept whole in
  ``expected/pictures/`` for a reader to look at and for an exact pixel
  diff.

The three sections and the goldens are unpublished data, so they are kept
out of the repository, in ``tests/golden/data/`` (git-ignored) or the folder
``LANGSLICE_GOLDEN_DATA`` names (``sections/``, ``expected/``); the test is
skipped where they are missing.

On a mismatch the run's outputs go to ``actual/`` beside them (every
picture included) and the failure lists what changed: the files added or
removed, each changed JSON key with its old and new value, each changed
array's shape and statistics, and per picture the pixels that differ and the
largest difference (whole pictures), or the tiles of an 8 x 8 grid that
changed (pictures kept only as digests).

Run it with ``python -m pytest -m golden`` (about 30 s; it is left out of
the default run because it needs the cached BrainGlobe atlases and its
pixels are those of this machine's libraries). An intended change is
recorded with ``LANGSLICE_UPDATE_GOLDEN=1 python -m pytest -m golden``,
which rewrites ``expected/`` and fails once, printing what changed, so a
regeneration never passes unreviewed.
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
from pathlib import Path

import numpy as np
import pytest
import tifffile
import zarr
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
#: The scenario runs in a child interpreter: no test patch reaches it, and
#: it has no GPU, so FireANTs runs its deterministic CPU path.
CHILD_ENV = {
    "CUDA_VISIBLE_DEVICES": "",
    "LANGSLICE_ATLAS_CACHE": "off",
    "LANGSLICE_RECENT_JOBS": "off",
    "LANGSLICE_SETTINGS": "off",
    "PYTHONPATH": str(ROOT / "src"),
}
#: A fixed mtime for the copied sections (2026-01-01 UTC).
FIXED_MTIME = 1767225600
#: Job folder parts that are not outputs: undo steps, logs with clock times,
#: locks and counters.
SKIPPED_PARTS = frozenset({"history", "logs", "trace"})
SKIPPED_SUFFIXES = frozenset({".lock", ".seq"})
#: Folder names whose zarr store is read as arrays, not as files.
ZARR_SUFFIX = ".zarr"
#: Tokens for what changes between runs of identical code.
ROOT_TOKEN = "<ROOT>"
TIME_TOKEN = "<time>"
SECONDS_TOKEN = "<seconds>"
#: ISO times (2026-10-29T01:17:00.81+00:00) and local times (2026-10-28
#: 21:17:01), as the job files write them.
TIME_PATTERN = re.compile(
    r"\d{4}-\d{2}-\d{2}[T ]\d{2}:\d{2}:\d{2}(\.\d+)?([+-]\d{2}:\d{2}|Z)?"
)
#: Keys whose numbers are wall-clock durations.
SECONDS_KEYS = frozenset(
    {"runtime_s", "elapsed_s", "seconds", "wall_s", "gpu_wait_s"}
)
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
#: ``actual/diff.txt``.
MAX_PRINTED = 150
#: Tools whose pictures are kept whole: the writes.
KEPT_TOOLS = (
    "position_sections",
    "interactive_transform",
    "elastix_affine",
    "make_mask",
    "mark_damage",
    "fireants_syn",
    "set_landmarks",
    "landmark_triangulation",
)


# --- the run ---------------------------------------------------------------


@dataclass(frozen=True)
class Run:
    """One run of the scenario: where its job is and what it recorded."""

    root: Path
    job: Path
    calls: list[object]


def run_scenario(root: Path) -> Run:
    """Copy the sections under *root*, run the scenario, read its calls."""
    images = root / "images"
    images.mkdir(parents=True)
    for source in sorted(SECTIONS.glob("*.png")):
        target = images / source.name
        shutil.copyfile(source, target)
        os.utime(target, (FIXED_MTIME, FIXED_MTIME))
    calls_path = root / "calls.json"
    command = [
        sys.executable,
        str(GOLDEN / "scenario.py"),
        str(images),
        str(calls_path),
    ]
    environment = {**os.environ, **CHILD_ENV}
    finished = subprocess.run(
        command,
        cwd=ROOT,
        env=environment,
        capture_output=True,
        text=True,
        timeout=900,
        check=False,
    )
    if finished.returncode != 0:
        pytest.fail("the golden scenario failed:\n" + finished.stderr[-4000:])
    calls = json.loads(calls_path.read_text())
    return Run(root=root, job=images / "langslice", calls=calls)


# --- normalising -------------------------------------------------------------


class Normaliser:
    """Replace a run's paths, clock times and durations with tokens."""

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


def json_diff(old: object, new: object) -> list[str]:
    """Return one line per changed, added or removed key."""
    old_leaves, new_leaves = flatten(old), flatten(new)
    lines = []
    for key in sorted(old_leaves.keys() | new_leaves.keys()):
        if key not in new_leaves:
            lines.append(f"  - {key}: {short(old_leaves[key])}")
        elif key not in old_leaves:
            lines.append(f"  + {key}: {short(new_leaves[key])}")
        elif old_leaves[key] != new_leaves[key]:
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
    rows, cols = np.nonzero(changed)
    return (
        f"{int(changed.sum())} of {changed.size} pixels differ, largest "
        f"difference {int(difference.max())}, within x {cols.min()}-"
        f"{cols.max()} y {rows.min()}-{rows.max()}"
    )


def tile_diff(old: dict[str, object], new: dict[str, object]) -> str:
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
        f"{ACTUAL}/pictures)"
    )


def picture_line(name: str, old: object, new: object) -> str:
    """Return the diff line of one changed ``view.jpg``."""
    stem = name.split("/")[-2]
    kept = EXPECTED / "pictures" / f"{stem}.jpg"
    if kept.exists():
        detail = pixel_diff(kept, ACTUAL / "pictures" / f"{stem}.jpg")
    else:
        detail = tile_diff(old, new)
    return f"{name}: {detail}"


def files_diff(old: dict[str, object], new: dict[str, object]) -> list[str]:
    """Return the readable diff of two ``files.json`` contents."""
    lines = [f"removed: {name}" for name in sorted(old.keys() - new.keys())]
    lines += [f"added: {name}" for name in sorted(new.keys() - old.keys())]
    for name in sorted(old.keys() & new.keys()):
        if old[name] == new[name]:
            continue
        if name.endswith("view.jpg"):
            lines.append(picture_line(name, old[name], new[name]))
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


def differences(outputs: Outputs) -> list[str]:
    """List what *outputs* change against the goldens; empty when nothing."""
    expected_calls = (EXPECTED / "calls.json").read_text()
    expected_files = (EXPECTED / "files.json").read_text()
    actual_calls, actual_files = dump(outputs.calls), dump(outputs.files)
    # compared as text: a NaN in a file never equals itself as a value
    if actual_calls == expected_calls and actual_files == expected_files:
        return []

    call_lines = calls_diff(
        json.loads(expected_calls), json.loads(actual_calls)
    )
    file_lines = files_diff(
        json.loads(expected_files), json.loads(actual_files)
    )
    n_calls = sum(line.startswith("call ") for line in call_lines)
    n_pictures = sum("view.jpg:" in line for line in file_lines)
    n_files = sum(not line.startswith(" ") for line in file_lines)
    return [
        f"{n_calls} replies, {n_files} files ({n_pictures} pictures) differ",
        *call_lines,
        *file_lines,
    ]


def printed(lines: list[str]) -> str:
    """Join *lines* for a failure message, cut to ``MAX_PRINTED``."""
    if len(lines) > MAX_PRINTED:
        lines = lines[:MAX_PRINTED] + [
            f"... {len(lines) - MAX_PRINTED} more lines in {ACTUAL}/diff.txt"
        ]
    return "\n".join(lines)


def test_golden_outputs(tmp_path: Path) -> None:
    """Run the scenario and compare everything it produced with the goldens."""
    if not SECTIONS.is_dir():
        pytest.skip(f"no golden data in {DATA} (set {DATA_ENV})")
    outputs = collect(run_scenario(tmp_path))
    lines = differences(outputs) if EXPECTED.is_dir() else ["no goldens"]
    if ACTUAL.exists():
        shutil.rmtree(ACTUAL)
    if not lines:
        return

    write_outputs(outputs, ACTUAL, every_picture=True)
    (ACTUAL / "diff.txt").write_text("\n".join(lines) + "\n")
    if os.environ.get(UPDATE_ENV):
        # the goldens are not under version control: the failure carries
        # the diff, so a regeneration is always reviewed
        write_outputs(outputs, EXPECTED, every_picture=False)
        pytest.fail(f"regenerated {EXPECTED}; it changed:\n" + printed(lines))
    pytest.fail(
        f"outputs differ from {EXPECTED} (this run's are in {ACTUAL}):\n"
        + printed(lines)
    )
