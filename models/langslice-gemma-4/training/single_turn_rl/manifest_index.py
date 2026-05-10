"""In-memory index over the LangSlice training-data manifest.

Wraps the QC app's :func:`load_inventory_manifest` (the canonical reader for
the shards + allocations layers) and turns its row dicts into a queryable
index keyed by ``(plane, dataset, section_id)``. The index also carries an
ephemeral ``LiveDifficulty`` map so the curriculum callback can keep a
running per-section error estimate without re-scanning the manifest each
step.

Why this exists
---------------

The full inventory has ~50k rows. The trainer needs sub-second filtered
queries (e.g. "every coronal Allen 25um row whose ``position_mm`` is in
``[2.0, 4.5]`` and whose live difficulty is in the ``[0.4, 0.7]``
quantile band") on every curriculum tick. Linear scans through the whole
manifest, then re-walking allocations to derive splits, would dominate
training step time. This index pays the build cost once at startup,
stores sorted-by-position lists per ``(atlas, plane)``, and dispatches
range queries via ``bisect`` (microseconds).

Scope discipline
----------------

This module is purely additive over ``_local/qc_app/app.py``. It never
modifies the manifest, never writes to ``data/manifest/``, and never
imports the QC app's HTTP server bits — only the loader function and its
helpers. The loader's import-time ``sys.path`` mutation (it injects
``_local/eval`` so it can ``import allocations``) is preserved because
that's how the QC app is shipped today; importing this module therefore
also makes ``allocations`` importable in the current process. That is a
known side effect, not a regression.

Persisted live-difficulty sidecar
---------------------------------

``persist_live_difficulty`` writes a sidecar JSON under
``_local/rlvr_progress/live_difficulty.json`` (relative to the repo root
inferred from the manifest root). The sidecar is the canonical handoff
between training runs and the slicebench seed CLI (Task 5) — a fresh
run can ``restore_live_difficulty`` to pick up where the last one
stopped without re-running slicebench eval.
"""

from __future__ import annotations

import bisect
import importlib.util
import json
import os
import sys
from collections.abc import Callable
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Literal

# ---------------------------------------------------------------------------
# Loader bridge — import the QC app's load_inventory_manifest by file path
# ---------------------------------------------------------------------------

# ``_local/qc_app`` isn't a Python package (no ``__init__``, hyphenated
# parents) so we can't ``from _local.qc_app.app import ...``. Resolve the
# loader by file path and cache the imported module so repeated calls don't
# re-exec the file.
_QC_APP_MODULE: Any = None


def _load_qc_app_module() -> Any:
    """Import ``_local/qc_app/app.py`` once and return the module object.

    The module's top-level code mutates ``sys.path`` so it can ``import
    allocations`` from ``_local/eval``. That side effect persists for the
    rest of the process — it matches what happens whenever the QC HTTP
    server runs, but callers should be aware that ``_local/eval`` becomes
    importable after the first index build.
    """
    global _QC_APP_MODULE
    if _QC_APP_MODULE is not None:
        return _QC_APP_MODULE

    # Walk up from this file to find the repo root. Layout:
    #   <repo>/models/langslice-gemma-4/training/single_turn_rl/manifest_index.py
    here = Path(__file__).resolve()
    repo_root = here.parents[4]
    qc_app_path = repo_root / "_local" / "qc_app" / "app.py"
    if not qc_app_path.is_file():
        raise FileNotFoundError(
            f"Could not locate the QC app module at {qc_app_path}. "
            "manifest_index requires _local/qc_app/app.py to be present "
            "because it wraps that module's load_inventory_manifest."
        )

    # Make sure ``langslice_harness`` (used by the QC app at import time) is
    # importable; the canonical layout puts it under ``<repo>/src``.
    src_dir = repo_root / "src"
    if str(src_dir) not in sys.path:
        sys.path.insert(0, str(src_dir))

    spec = importlib.util.spec_from_file_location("_qc_app_module", qc_app_path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Could not build module spec for {qc_app_path}.")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    _QC_APP_MODULE = module
    return module


def _default_loader(shards_root: Path, allocations_root: Path) -> list[dict[str, Any]]:
    """Default loader: delegates to the QC app's ``load_inventory_manifest``."""
    qc_app = _load_qc_app_module()
    return qc_app.load_inventory_manifest(shards_root, allocations_root)


# ---------------------------------------------------------------------------
# Public dataclasses
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class Section:
    """One row of the manifest, parsed and frozen for index storage.

    The composite key is ``(plane, dataset, section_id)``. ``plane`` is the
    canonical plane derived from ``slice_axis`` (coronal/sagittal/horizontal),
    not the raw ``orientation`` string from the shard — the allocations layer
    keys splits by plane, so the index keeps the same convention. The raw
    ``orientation`` is preserved in its own field for callers that want it.

    ``split`` carries the QC app's stamped ``_split`` value: one of ``""``
    (no allocation), ``"eval"``, ``"rlvr"``, or ``"sft"``.
    """

    plane: str
    dataset: str
    subject_id: str
    section_id: str
    atlas: str
    position_mm: float
    image_path: str
    species: str
    imaging: str
    staining: str
    orientation: str
    exclude_from_training: bool
    split: str

    @property
    def key(self) -> tuple[str, str, str]:
        """Composite primary key used by every dict-keyed lookup."""
        return (self.plane, self.dataset, self.section_id)


@dataclass(frozen=True)
class LiveDifficulty:
    """A single per-section difficulty observation.

    ``score`` is dimensionless (the curriculum decides the meaning). ``source``
    tags where the observation came from so downstream tooling can prefer
    fresh ``live_rollout`` scores over stale ``slicebench`` seeds.
    ``updated_at`` is an ISO-8601 UTC timestamp.
    """

    score: float
    source: str
    updated_at: str


# Required string fields on the row; rows missing any of these are skipped
# during construction (with a stderr warning + count). ``position_mm`` and
# ``exclude_from_training`` are validated separately because they have
# non-string types and richer fallback logic.
_REQUIRED_STR_FIELDS: tuple[str, ...] = (
    "dataset",
    "subject_id",
    "section_id",
    "atlas",
    "image_path",
)


# ---------------------------------------------------------------------------
# ManifestIndex
# ---------------------------------------------------------------------------


class ManifestIndex:
    """In-memory index over the parsed manifest.

    See :func:`from_manifest_root` for the canonical constructor. The index
    is intentionally a plain ``object`` rather than a dataclass: its
    internal dicts are mutable (live difficulty updates) and dataclass
    semantics would muddy the equality story.
    """

    def __init__(
        self,
        sections: list[Section],
        *,
        manifest_root: Path | None = None,
        repo_root: Path | None = None,
    ) -> None:
        # Sort once during construction so query() can rely on
        # iteration order being stable across runs.
        self._sections: list[Section] = sorted(
            sections, key=lambda s: (s.plane, s.dataset, s.section_id)
        )
        self._by_key: dict[tuple[str, str, str], Section] = {
            s.key: s for s in self._sections
        }
        # Per (atlas, plane) sorted-by-position_mm lists for bisect range
        # queries. Sections are referenced (not copied) so memory cost is
        # one extra list of pointers per pair.
        by_atlas_plane: dict[tuple[str, str], list[Section]] = {}
        for s in self._sections:
            by_atlas_plane.setdefault((s.atlas, s.plane), []).append(s)
        for pair, lst in by_atlas_plane.items():
            lst.sort(key=lambda s: s.position_mm)
            by_atlas_plane[pair] = lst
        self._by_atlas_plane: dict[tuple[str, str], list[Section]] = by_atlas_plane
        # Parallel list of position_mm values per (atlas, plane). bisect
        # operates on sorted scalars; pre-extracting keeps the hot path
        # branch-free.
        self._positions_by_pair: dict[tuple[str, str], list[float]] = {
            pair: [s.position_mm for s in lst]
            for pair, lst in self._by_atlas_plane.items()
        }
        self._live_difficulty: dict[tuple[str, str, str], LiveDifficulty] = {}
        self._manifest_root: Path | None = manifest_root
        self._repo_root: Path | None = repo_root

    # ------------------------------------------------------------------ ctor

    @classmethod
    def from_manifest_root(
        cls,
        manifest_root: Path,
        *,
        allocations_root: Path | None = None,
        loader: Callable[[Path, Path], list[dict[str, Any]]] | None = None,
        repo_root: Path | None = None,
    ) -> ManifestIndex:
        """Construct an index from a ``data/manifest/`` directory.

        ``manifest_root`` is the directory containing ``shards/`` and
        (by default) ``allocations/``. ``allocations_root`` overrides the
        default ``manifest_root / "allocations"`` location — useful when
        the splits live elsewhere (CI fixture, reroll experiment).

        ``loader`` is the function used to read the raw row dicts; it
        defaults to the QC app's :func:`load_inventory_manifest`. Tests
        inject a stub here when they don't want to import the QC app.

        ``repo_root`` overrides the inferred repo root used as the default
        target for :meth:`persist_live_difficulty`. When omitted it's
        inferred as ``manifest_root.parent.parent`` (canonical layout
        ``<repo>/data/manifest``); falls back to ``Path.cwd()`` if that
        path doesn't have enough parents.
        """
        manifest_root = Path(manifest_root)
        shards_root = manifest_root / "shards"
        alloc_root = (
            Path(allocations_root)
            if allocations_root is not None
            else manifest_root / "allocations"
        )
        load_fn = loader if loader is not None else _default_loader
        raw_rows = load_fn(shards_root, alloc_root)

        if repo_root is None:
            try:
                inferred = manifest_root.resolve().parents[1]
            except IndexError:
                inferred = Path.cwd().resolve()
            repo_root = inferred

        sections, skipped = _parse_rows(raw_rows)
        if skipped:
            print(
                f"[manifest_index] skipped {skipped} row(s) missing required fields",
                file=sys.stderr,
                flush=True,
            )
        return cls(sections, manifest_root=manifest_root, repo_root=repo_root)

    # ------------------------------------------------------------------ accessors

    def __len__(self) -> int:
        return len(self._sections)

    def pairs(self) -> list[tuple[str, str]]:
        """Distinct ``(atlas, plane)`` pairs present in the index, sorted."""
        return sorted(self._by_atlas_plane.keys())

    def section_by_key(
        self, plane: str, dataset: str, section_id: str
    ) -> Section | None:
        """Exact lookup by composite key. Returns ``None`` when absent."""
        return self._by_key.get((plane, dataset, section_id))

    # ------------------------------------------------------------------ live difficulty

    def update_difficulty(
        self,
        plane: str,
        dataset: str,
        section_id: str,
        score: float,
        source: str,
        *,
        if_unknown: Literal["raise", "warn", "ignore"] = "raise",
    ) -> bool:
        """Record a live difficulty observation for a known section.

        Parameters
        ----------
        if_unknown : {"raise", "warn", "ignore"}
            Behavior when ``(plane, dataset, section_id)`` is not in the
            index:

              - ``"raise"`` (default): ``KeyError``. Strictest; preserves
                the historical contract.
              - ``"warn"``: stderr warning, return ``False``, no record
                written.
              - ``"ignore"``: silently no-op, return ``False``.

            The non-raising modes exist for the curriculum callback path:
            after a manifest rebuild between runs, stale section keys in
            a rolling reward buffer would otherwise crash the trainer.

        Returns
        -------
        bool
            ``True`` if the difficulty was recorded, ``False`` if the
            section was unknown and ``if_unknown != "raise"``.
        """
        key = (plane, dataset, section_id)
        if key not in self._by_key:
            if if_unknown == "raise":
                raise KeyError(
                    f"Cannot update difficulty for unknown section {key!r}"
                )
            if if_unknown == "warn":
                print(
                    f"[manifest_index] update_difficulty: skipping unknown "
                    f"section {key!r}",
                    file=sys.stderr,
                    flush=True,
                )
            # "ignore" → silent no-op
            return False
        self._live_difficulty[key] = LiveDifficulty(
            score=float(score),
            source=str(source),
            updated_at=_utc_now_iso(),
        )
        return True

    def live_difficulty(
        self, plane: str, dataset: str, section_id: str
    ) -> LiveDifficulty | None:
        """Return the live difficulty record for a section, or ``None``."""
        return self._live_difficulty.get((plane, dataset, section_id))

    def bulk_load_difficulty(self, seeded_difficulty_path: Path) -> int:
        """Bulk-seed live difficulty from a slicebench-summary JSON.

        Expected file shape::

            {"rows": [
                {"plane": "...", "dataset": "...", "section_id": "...",
                 "difficulty_score": 0.42, "source": "slicebench"},
                ...
            ]}

        Rows whose ``(plane, dataset, section_id)`` doesn't resolve to a
        section in this index (orphans), and rows that are malformed
        (missing/non-numeric ``difficulty_score`` or non-dict shape;
        invalids), are tracked separately and reported with the first up
        to 5 offending orphan keys for fast triage. The rest are written
        via :meth:`update_difficulty`. Returns the count of successfully
        loaded rows.
        """
        path = Path(seeded_difficulty_path)
        with path.open("r", encoding="utf-8") as fh:
            payload = json.load(fh)
        rows = payload.get("rows") or []
        if not isinstance(rows, list):
            raise ValueError(
                f"Expected 'rows' to be a list in {path}; got {type(rows).__name__}."
            )

        loaded = 0
        n_orphan = 0
        n_invalid = 0
        orphan_keys: list[tuple[str, str, str]] = []
        for row in rows:
            if not isinstance(row, dict):
                n_invalid += 1
                continue
            plane = str(row.get("plane", ""))
            dataset = str(row.get("dataset", ""))
            section_id = str(row.get("section_id", ""))
            try:
                score = float(row["difficulty_score"])
            except (KeyError, TypeError, ValueError):
                n_invalid += 1
                continue
            source = str(row.get("source", "slicebench"))
            key = (plane, dataset, section_id)
            if key not in self._by_key:
                n_orphan += 1
                orphan_keys.append(key)
                continue
            self.update_difficulty(plane, dataset, section_id, score, source)
            loaded += 1

        if n_orphan or n_invalid:
            _log_orphan_warning(
                "bulk_load_difficulty",
                source=str(path),
                n_orphan=n_orphan,
                n_invalid=n_invalid,
                orphan_keys=orphan_keys,
            )
        return loaded

    def persist_live_difficulty(self, path: Path | None = None) -> int:
        """Atomically dump the live difficulty map to JSON.

        Default path: ``_local/rlvr_progress/live_difficulty.json`` under
        the repo root inferred at construction. Writes via a sibling
        ``.tmp`` file then ``os.replace`` so a crash mid-write never
        leaves a half-written sidecar in place. Returns the count of
        rows written.
        """
        target = Path(path) if path is not None else self._default_live_difficulty_path()
        target.parent.mkdir(parents=True, exist_ok=True)
        rows: list[dict[str, Any]] = []
        # Sort by key for deterministic output across runs.
        for key in sorted(self._live_difficulty.keys()):
            plane, dataset, section_id = key
            entry = self._live_difficulty[key]
            rows.append(
                {
                    "plane": plane,
                    "dataset": dataset,
                    "section_id": section_id,
                    "score": entry.score,
                    "source": entry.source,
                    "updated_at": entry.updated_at,
                }
            )
        payload = {"rows": rows}
        # ``with_suffix`` would clobber a multi-part extension; build the
        # tmp path manually so e.g. ``foo.json.gz`` would still get
        # ``foo.json.gz.tmp``.
        tmp_path = target.with_name(target.name + ".tmp")
        with tmp_path.open("w", encoding="utf-8") as fh:
            json.dump(payload, fh, ensure_ascii=False, indent=2)
            fh.write("\n")
        os.replace(tmp_path, target)
        return len(rows)

    def restore_live_difficulty(self, path: Path | None = None) -> int:
        """Repopulate live difficulty from a previously-persisted sidecar.

        Inverse of :meth:`persist_live_difficulty`. Rows that no longer
        resolve to a section in the index (orphans, expected after manifest
        churn between runs), and rows that are malformed (invalids), are
        tracked separately and reported with the first up to 5 offending
        orphan keys. Returns the count of rows successfully restored.
        """
        target = Path(path) if path is not None else self._default_live_difficulty_path()
        with target.open("r", encoding="utf-8") as fh:
            payload = json.load(fh)
        rows = payload.get("rows") or []
        if not isinstance(rows, list):
            raise ValueError(
                f"Expected 'rows' to be a list in {target}; got {type(rows).__name__}."
            )

        restored = 0
        n_orphan = 0
        n_invalid = 0
        orphan_keys: list[tuple[str, str, str]] = []
        for row in rows:
            if not isinstance(row, dict):
                n_invalid += 1
                continue
            plane = str(row.get("plane", ""))
            dataset = str(row.get("dataset", ""))
            section_id = str(row.get("section_id", ""))
            key = (plane, dataset, section_id)
            if key not in self._by_key:
                n_orphan += 1
                orphan_keys.append(key)
                continue
            try:
                score = float(row["score"])
            except (KeyError, TypeError, ValueError):
                n_invalid += 1
                continue
            source = str(row.get("source", ""))
            updated_at = str(row.get("updated_at", "")) or _utc_now_iso()
            self._live_difficulty[key] = LiveDifficulty(
                score=score, source=source, updated_at=updated_at
            )
            restored += 1

        if n_orphan or n_invalid:
            _log_orphan_warning(
                "restore_live_difficulty",
                source=str(target),
                n_orphan=n_orphan,
                n_invalid=n_invalid,
                orphan_keys=orphan_keys,
            )
        return restored

    def _default_live_difficulty_path(self) -> Path:
        """Resolve the canonical sidecar location.

        Uses ``self._repo_root`` (set at construction) so the path is
        stable across calls even after ``Path.cwd()`` changes (some
        tests chdir mid-run).
        """
        root = self._repo_root if self._repo_root is not None else Path.cwd()
        return root / "_local" / "rlvr_progress" / "live_difficulty.json"

    # ------------------------------------------------------------------ query

    def query(
        self,
        *,
        plane: str | None = None,
        atlas: str | None = None,
        split: str | None = None,
        position_range: tuple[float, float] | None = None,
        dataset: str | None = None,
        difficulty_range: tuple[float, float] | None = None,
        exclude_zero_axes: bool = False,
    ) -> list[Section]:
        """Filtered view over the index.

        All keyword arguments are independent filters; an unset filter is a
        no-op. ``position_range`` is inclusive ``[lo, hi]``. When ``atlas``
        and ``plane`` are both supplied the index uses ``bisect`` over the
        pre-sorted ``(atlas, plane)`` list — O(log n + k) where k is the
        match count. Otherwise it falls back to a full scan.

        ``difficulty_range`` joins :attr:`_live_difficulty`. When set, a
        section without a live record is **excluded** (no signal yet).
        When unset, sections without a live record are **included** —
        the curriculum sampler treats "unobserved" as legitimate
        cold-start material.

        ``exclude_zero_axes`` is off by default. Most adapters use 0.0 as
        the legitimate anterior-pole position; this flag exists for
        adapters where 0.0 is a sentinel for "unknown" and should be
        dropped explicitly.

        The return value is sorted by ``(plane, dataset, section_id)``
        for determinism — caller-side sorting could otherwise drift.
        """
        # Pick the iteration source: bisect-sliced or full scan.
        if (
            atlas is not None
            and plane is not None
            and position_range is not None
            and (atlas, plane) in self._by_atlas_plane
        ):
            lo, hi = float(position_range[0]), float(position_range[1])
            positions = self._positions_by_pair[(atlas, plane)]
            i_lo = bisect.bisect_left(positions, lo)
            i_hi = bisect.bisect_right(positions, hi)
            candidates: list[Section] = list(
                self._by_atlas_plane[(atlas, plane)][i_lo:i_hi]
            )
            # Bisect already enforced the position bound, so further
            # filtering can skip it — but keep the rest of the predicates
            # in one loop for clarity.
            position_already_filtered = True
        elif atlas is not None and plane is not None and (atlas, plane) in self._by_atlas_plane:
            candidates = list(self._by_atlas_plane[(atlas, plane)])
            position_already_filtered = False
        else:
            candidates = self._sections
            position_already_filtered = False

        out: list[Section] = []
        for s in candidates:
            if plane is not None and s.plane != plane:
                continue
            if atlas is not None and s.atlas != atlas:
                continue
            if dataset is not None and s.dataset != dataset:
                continue
            if split is not None and s.split != split:
                continue
            if exclude_zero_axes and s.position_mm == 0.0:
                continue
            if position_range is not None and not position_already_filtered:
                lo, hi = float(position_range[0]), float(position_range[1])
                if s.position_mm < lo or s.position_mm > hi:
                    continue
            if difficulty_range is not None:
                live = self._live_difficulty.get(s.key)
                if live is None:
                    continue
                d_lo, d_hi = float(difficulty_range[0]), float(difficulty_range[1])
                if live.score < d_lo or live.score > d_hi:
                    continue
            out.append(s)

        # Bisect candidates are already in position-mm order; full-scan
        # candidates already came from a globally-sorted ``_sections`` list.
        # Re-sort by composite key so callers get the documented determinism
        # regardless of which path they took.
        out.sort(key=lambda s: (s.plane, s.dataset, s.section_id))
        return out


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _utc_now_iso() -> str:
    """ISO-8601 UTC timestamp, stable across platforms."""
    return datetime.now(timezone.utc).isoformat()


# Cap on how many offending keys we list before truncating with "... (N more)".
# 5 is enough to disambiguate (e.g. spot a single bad dataset vs. a manifest-wide
# rebuild) without flooding stderr when a whole shard goes stale.
_ORPHAN_KEY_PREVIEW_LIMIT: int = 5


def _log_orphan_warning(
    fn_name: str,
    *,
    source: str,
    n_orphan: int,
    n_invalid: int,
    orphan_keys: list[tuple[str, str, str]],
) -> None:
    """Emit a stderr warning that distinguishes orphan vs invalid counts.

    Lists the first :data:`_ORPHAN_KEY_PREVIEW_LIMIT` orphan keys so a
    grep-from-the-trainer-log workflow can tell at a glance whether the
    skipped rows are concentrated in one dataset (likely a stale shard)
    or scattered (likely a manifest-wide rebuild).
    """
    preview = orphan_keys[:_ORPHAN_KEY_PREVIEW_LIMIT]
    extra = len(orphan_keys) - len(preview)
    suffix = ""
    if preview:
        suffix = f"; first orphans: {preview!r}"
        if extra > 0:
            suffix += f" ({extra} more)"
    print(
        f"[manifest_index] {fn_name} skipped {n_orphan} orphan + "
        f"{n_invalid} invalid row(s) from {source}{suffix}",
        file=sys.stderr,
        flush=True,
    )


def _parse_rows(raw_rows: list[dict[str, Any]]) -> tuple[list[Section], int]:
    """Parse loader output into ``Section`` objects, counting skips.

    Returns ``(sections, skipped)``. A row is skipped when:

    * ``slice_axis`` doesn't map to a canonical plane (the QC app's
      ``_split`` would have been empty too, so the row is unusable for
      the curriculum), or
    * any of :data:`_REQUIRED_STR_FIELDS` is missing/empty, or
    * ``position_mm`` is missing or non-numeric.
    """
    qc_app = _load_qc_app_module()
    plane_for_axis: Callable[[str | None], str | None] = qc_app._plane_for_axis

    sections: list[Section] = []
    skipped = 0
    for row in raw_rows:
        if not isinstance(row, dict):
            skipped += 1
            continue
        plane = plane_for_axis(row.get("slice_axis"))
        if not plane:
            skipped += 1
            continue
        # Required string fields must be present and non-empty.
        missing = False
        for field in _REQUIRED_STR_FIELDS:
            v = row.get(field)
            if v is None or (isinstance(v, str) and not v):
                missing = True
                break
        if missing:
            skipped += 1
            continue
        try:
            position_mm = float(row["position_mm"])
        except (KeyError, TypeError, ValueError):
            skipped += 1
            continue

        sections.append(
            Section(
                plane=plane,
                dataset=str(row["dataset"]),
                subject_id=str(row["subject_id"]),
                section_id=str(row["section_id"]),
                atlas=str(row["atlas"]),
                position_mm=position_mm,
                image_path=str(row["image_path"]),
                species=str(row.get("species", "")),
                imaging=str(row.get("imaging", "")),
                staining=str(row.get("staining", "")),
                orientation=str(row.get("orientation", "")),
                exclude_from_training=bool(row.get("exclude_from_training", False)),
                split=str(row.get("_split", row.get("split", "")) or ""),
            )
        )
    return sections, skipped


def section_to_dict(section: Section) -> dict[str, Any]:
    """Convenience: ``Section`` -> plain dict (asdict). Useful in tests."""
    return asdict(section)


__all__ = [
    "LiveDifficulty",
    "ManifestIndex",
    "Section",
    "section_to_dict",
]
