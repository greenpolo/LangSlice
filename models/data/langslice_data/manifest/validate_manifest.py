"""Read-only cross-shard validator for the LangSlice evaluation manifest.

This script loads every sharded JSONL manifest row, runs checks that require
global visibility across shards, and optionally regenerates the manifest
summary after validation. It never writes to shard or override directories.
"""

from __future__ import annotations

import argparse
import builtins
import io
import json
import os
import sys
from collections import defaultdict
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Iterator, NamedTuple

_SCRIPT_DIR = Path(__file__).resolve().parent
_REPO_ROOT = _SCRIPT_DIR.parents[1]
if str(_SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(_SCRIPT_DIR))

from build_manifest import (  # noqa: E402
    ORIENTATION_TO_AXIS,
    PLAUSIBLE_RANGES,
    _BG_ASR_EXTENTS_MM,
    _resolve_dev_atlas,
    write_summary,
)

PLANES = ("coronal", "sagittal", "horizontal")
CRITICAL_FIELDS = (
    "section_id",
    "image_path",
    "dataset",
    "subject_id",
    "slice_axis",
    "position_mm",
    "atlas",
)
_FORBIDDEN_WRITE_PREFIXES = ("data/manifest/shards", "data/manifest/overrides")
_WRITE_MODES = frozenset("wax+")
_BASE_VALID_ATLASES = {atlas for atlas, _axis in _BG_ASR_EXTENTS_MM} | {"nissl_dev_p9p35_25um"}


class _ValidationResult(NamedTuple):
    error_count: int
    errors: list[dict[str, Any]]
    rows: list[dict[str, Any]]
    shards_loaded: int
    rows_by_plane: dict[str, int]


def _repo_relative_path(path: Path | str) -> Path:
    candidate = Path(path)
    if candidate.is_absolute():
        return candidate.resolve()
    return (_REPO_ROOT / candidate).resolve()


def _is_relative_to(path: Path, base: Path) -> bool:
    try:
        path.relative_to(base)
    except ValueError:
        return False
    return True


def _is_forbidden_write(path: Path | str) -> bool:
    candidate = _repo_relative_path(path)
    for prefix in _FORBIDDEN_WRITE_PREFIXES:
        forbidden = (_REPO_ROOT / prefix).resolve()
        if _is_relative_to(candidate, forbidden):
            return True
    return False


def _safe_write(path: Path | str) -> Path:
    """Return an absolute write target or raise for forbidden manifest paths."""
    target = _repo_relative_path(path)
    if _is_forbidden_write(target):
        raise RuntimeError(f"refusing to write under protected manifest path: {target}")
    return target


def _mode_writes(mode: str) -> bool:
    return any(flag in mode for flag in _WRITE_MODES)


@contextmanager
def _forbidden_write_guard() -> Iterator[None]:
    original_builtin_open = builtins.open
    original_io_open = io.open
    original_path_open = Path.open

    def guarded_builtin_open(file: Any, mode: str = "r", *args: Any, **kwargs: Any) -> Any:
        if isinstance(file, (str, bytes, os.PathLike)) and _mode_writes(mode):
            _safe_write(Path(file))
        return original_builtin_open(file, mode, *args, **kwargs)

    def guarded_io_open(file: Any, mode: str = "r", *args: Any, **kwargs: Any) -> Any:
        if isinstance(file, (str, bytes, os.PathLike)) and _mode_writes(mode):
            _safe_write(Path(file))
        return original_io_open(file, mode, *args, **kwargs)

    def guarded_path_open(
        self: Path,
        mode: str = "r",
        buffering: int = -1,
        encoding: str | None = None,
        errors: str | None = None,
        newline: str | None = None,
    ) -> Any:
        if _mode_writes(mode):
            _safe_write(self)
        return original_path_open(self, mode, buffering, encoding, errors, newline)

    builtins.open = guarded_builtin_open
    io.open = guarded_io_open
    Path.open = guarded_path_open
    try:
        yield
    finally:
        Path.open = original_path_open
        io.open = original_io_open
        builtins.open = original_builtin_open


def _is_empty(value: Any) -> bool:
    if value is None:
        return True
    if isinstance(value, str) and not value.strip():
        return True
    return False


def _row_context(row: dict[str, Any]) -> dict[str, Any]:
    return {
        "plane": row.get("_plane"),
        "dataset": row.get("_dataset_stem"),
        "line": row.get("_line"),
        "section_id": row.get("section_id"),
    }


def _add_error(
    errors: list[dict[str, Any]],
    category: str,
    message: str,
    row: dict[str, Any] | None = None,
    **extra: Any,
) -> None:
    entry: dict[str, Any] = {"category": category, "message": message}
    if row is not None:
        entry.update(_row_context(row))
    entry.update(extra)
    errors.append(entry)


def _format_shard_ref(row: dict[str, Any]) -> tuple[str | None, str | None]:
    return (row.get("_plane"), row.get("_dataset_stem"))


def _load_shards(shards_dir: Path, errors: list[dict[str, Any]]) -> tuple[list[dict[str, Any]], int, dict[str, int]]:
    rows: list[dict[str, Any]] = []
    shards_loaded = 0
    rows_by_plane = {plane: 0 for plane in PLANES}

    for plane in PLANES:
        plane_dir = shards_dir / plane
        if not plane_dir.exists():
            continue
        for shard_path in sorted(plane_dir.glob("*.jsonl"), key=lambda path: path.name):
            shards_loaded += 1
            dataset_stem = shard_path.stem
            with shard_path.open("r", encoding="utf-8") as handle:
                for line_number, line in enumerate(handle, start=1):
                    if not line.strip():
                        continue
                    try:
                        row = json.loads(line)
                    except json.JSONDecodeError as exc:
                        _add_error(
                            errors,
                            "json",
                            f"{shard_path}: line {line_number}: {exc.msg}",
                            plane=plane,
                            dataset=dataset_stem,
                            line=line_number,
                        )
                        continue
                    if not isinstance(row, dict):
                        _add_error(
                            errors,
                            "json",
                            f"{shard_path}: line {line_number}: row is not a JSON object",
                            plane=plane,
                            dataset=dataset_stem,
                            line=line_number,
                        )
                        continue
                    row["_plane"] = plane
                    row["_dataset_stem"] = dataset_stem
                    row["_line"] = line_number
                    rows.append(row)
                    rows_by_plane[plane] += 1
    return rows, shards_loaded, rows_by_plane


def _to_number(value: Any) -> float | None:
    if isinstance(value, bool) or value is None:
        return None
    if isinstance(value, (int, float)):
        return float(value)
    if isinstance(value, str) and value.strip():
        try:
            return float(value)
        except ValueError:
            return None
    return None


def _resolved_dev_atlas_for_row(row: dict[str, Any]) -> str | None:
    if _resolve_dev_atlas is None:
        return None
    resolved, _reason = _resolve_dev_atlas(row.get("dataset", ""), row.get("age"))
    return resolved


def _is_valid_atlas(row: dict[str, Any]) -> bool:
    atlas = row.get("atlas")
    if atlas in _BASE_VALID_ATLASES:
        return True
    if atlas == "admba" or _is_empty(atlas):
        return False
    return atlas == _resolved_dev_atlas_for_row(row)


def _range_for_row(row: dict[str, Any]) -> tuple[float, float] | None:
    atlas = row.get("atlas")
    axis = row.get("slice_axis")
    range_value = PLAUSIBLE_RANGES.get((atlas, axis))
    if range_value is not None:
        return range_value
    if atlas and atlas == _resolved_dev_atlas_for_row(row):
        return PLAUSIBLE_RANGES.get(("admba", axis))
    return None


def _check_duplicates(rows: list[dict[str, Any]], errors: list[dict[str, Any]]) -> None:
    for field, category in (("section_id", "duplicate_section_id"), ("image_path", "duplicate_image_path")):
        seen: dict[Any, list[dict[str, Any]]] = defaultdict(list)
        for row in rows:
            value = row.get(field)
            if _is_empty(value):
                continue
            seen[value].append(row)
        for value, matching_rows in sorted(seen.items(), key=lambda item: str(item[0])):
            if len(matching_rows) < 2:
                continue
            shard_refs = sorted({_format_shard_ref(row) for row in matching_rows})
            _add_error(
                errors,
                category,
                f"{field} {value!r} appears in {len(matching_rows)} rows across {shard_refs}",
                value=value,
                conflicts=shard_refs,
            )


def _check_row_fields(rows: list[dict[str, Any]], errors: list[dict[str, Any]], check_images: bool) -> None:
    for row in rows:
        for field in CRITICAL_FIELDS:
            if field not in row or _is_empty(row.get(field)):
                _add_error(errors, "critical_fields", f"missing or empty critical field {field!r}", row, field=field)

        expected_axis = ORIENTATION_TO_AXIS[row["_plane"]]
        slice_axis = row.get("slice_axis")
        if not _is_empty(slice_axis) and slice_axis != expected_axis:
            _add_error(
                errors,
                "plane_consistency",
                f"slice_axis {slice_axis!r} does not match plane {row['_plane']!r}; expected {expected_axis!r}",
                row,
            )

        dataset = row.get("dataset")
        if not _is_empty(dataset) and dataset != row["_dataset_stem"]:
            _add_error(
                errors,
                "dataset_consistency",
                f"dataset field {dataset!r} does not match shard stem {row['_dataset_stem']!r}",
                row,
            )

        atlas = row.get("atlas")
        if not _is_empty(atlas) and not _is_valid_atlas(row):
            _add_error(errors, "atlas_resolution", f"unresolved or unsupported atlas {atlas!r}", row)

        position = row.get("position_mm")
        numeric_position = _to_number(position)
        if not _is_empty(position) and numeric_position is None:
            _add_error(errors, "position_range", f"position_mm {position!r} is not numeric", row)
        elif numeric_position is not None:
            plausible_range = _range_for_row(row)
            if plausible_range is not None:
                lo, hi = plausible_range
                if not lo <= numeric_position <= hi:
                    _add_error(
                        errors,
                        "position_range",
                        (
                            f"position_mm {numeric_position:g} outside {lo:g}..{hi:g} "
                            f"for atlas {atlas!r} axis {slice_axis!r}"
                        ),
                        row,
                    )

        if check_images:
            image_path = row.get("image_path")
            if not _is_empty(image_path) and not (_REPO_ROOT / Path(str(image_path))).exists():
                _add_error(errors, "missing_image", f"image_path does not exist: {image_path}", row)


def _check_no_split_field(rows: list[dict[str, Any]], errors: list[dict[str, Any]]) -> None:
    """Defensive check: no inventory row may carry a 'split' field.

    Phase C.2 lifted split assignments into the allocation layer.  If any shard
    row still has a 'split' key, the migration has not been completed (or a bug
    re-introduced the field).
    """
    for row in rows:
        if "split" in row:
            _add_error(
                errors,
                "split_field_in_shard",
                (
                    f"inventory row has 'split' field {row['split']!r}; "
                    "splits must live in data/manifest/allocations/, not in shards"
                ),
                row,
            )


def _check_allocation_integrity(
    rows: list[dict[str, Any]],
    errors: list[dict[str, Any]],
    allocations_root: Path,
) -> None:
    """Check allocation layer integrity.

    Checks:
    1. No section_id appears in more than one split for the same plane.
    2. Every allocated section_id exists in some inventory shard for that plane.
    """
    # Try to import the allocation library; skip checks if it is not available.
    try:
        sys.path.insert(0, str(Path(__file__).resolve().parent))
        import allocations as alloc_lib  # type: ignore[import]
    except ImportError:
        return

    # Override ALLOCATIONS_ROOT to the caller-supplied root (needed for tests).
    original_root = alloc_lib.ALLOCATIONS_ROOT

    # We monkey-patch ALLOCATIONS_ROOT for the duration of this call so that
    # tests can supply a tmp_path-based root.  This is safe because the module
    # is single-threaded in this context.
    alloc_lib.ALLOCATIONS_ROOT = allocations_root
    try:
        _do_check_allocation_integrity(rows, errors, alloc_lib)
    finally:
        alloc_lib.ALLOCATIONS_ROOT = original_root


def _do_check_allocation_integrity(
    rows: list[dict[str, Any]],
    errors: list[dict[str, Any]],
    alloc_lib: Any,
) -> None:
    # Build a set of known section_ids per plane from inventory rows.
    known_by_plane: dict[str, set[str]] = defaultdict(set)
    for row in rows:
        plane = row.get("_plane")
        section_id = row.get("section_id")
        if plane and section_id:
            known_by_plane[plane].add(str(section_id))

    # Load all allocations and check for cross-split duplicates + orphans.
    # section_to_split[plane][section_id] = list of split names
    section_to_splits: dict[str, dict[str, list[str]]] = defaultdict(lambda: defaultdict(list))

    for plane in alloc_lib.PLANES:
        for split in alloc_lib.SPLITS:
            path = alloc_lib.allocation_path(plane, split)
            if not path.exists():
                continue
            try:
                entries = alloc_lib.load_allocation(plane, split)
            except (ValueError, OSError) as exc:
                _add_error(
                    errors,
                    "allocation_read_error",
                    f"could not read allocation {plane}/{split}: {exc}",
                )
                continue
            for section_id in entries:
                section_to_splits[plane][section_id].append(split)

    for plane, id_to_splits in section_to_splits.items():
        known_ids = known_by_plane.get(plane, set())
        for section_id, splits in sorted(id_to_splits.items()):
            if len(splits) > 1:
                _add_error(
                    errors,
                    "allocation_cross_split",
                    (
                        f"section_id {section_id!r} is active in multiple splits "
                        f"for plane {plane!r}: "
                        + " and ".join(f"{plane}/{s}" for s in splits)
                    ),
                    section_id=section_id,
                    plane=plane,
                    splits=splits,
                )
            if section_id not in known_ids:
                _add_error(
                    errors,
                    "allocation_orphan",
                    (
                        f"allocated section_id {section_id!r} does not exist in "
                        f"any {plane} inventory shard"
                    ),
                    section_id=section_id,
                    plane=plane,
                    splits=splits,
                )


def _validate(
    shards_dir: Path,
    *,
    check_images: bool = False,
    allocations_root: Path | None = None,
) -> _ValidationResult:
    shards_dir = _repo_relative_path(shards_dir)
    errors: list[dict[str, Any]] = []
    rows, shards_loaded, rows_by_plane = _load_shards(shards_dir, errors)
    _check_duplicates(rows, errors)
    _check_row_fields(rows, errors, check_images)
    _check_no_split_field(rows, errors)
    if allocations_root is None:
        allocations_root = _REPO_ROOT / "data" / "manifest" / "allocations"
    _check_allocation_integrity(rows, errors, allocations_root)
    return _ValidationResult(
        error_count=len(errors),
        errors=errors,
        rows=rows,
        shards_loaded=shards_loaded,
        rows_by_plane=rows_by_plane,
    )


def validate(
    shards_dir: Path,
    *,
    check_images: bool = False,
    allocations_root: Path | None = None,
) -> tuple[int, list[dict[str, Any]]]:
    """Validate sharded manifest rows and return ``(error_count, errors)``."""
    result = _validate(shards_dir, check_images=check_images, allocations_root=allocations_root)
    return result.error_count, result.errors


def _format_error(error: dict[str, Any]) -> str:
    parts = []
    if error.get("plane") is not None:
        parts.append(str(error["plane"]))
    if error.get("dataset") is not None:
        parts.append(str(error["dataset"]))
    if error.get("line") is not None:
        parts.append(f"line {error['line']}")
    location = "/".join(parts)
    if location:
        return f"{location}: {error['message']}"
    return str(error["message"])


def _print_summary(result: _ValidationResult) -> None:
    plane_counts = ", ".join(f"{plane}: {result.rows_by_plane.get(plane, 0)}" for plane in PLANES)
    print("Validation complete:")
    print(f"  Shards loaded:    {result.shards_loaded}")
    print(f"  Rows total:       {len(result.rows)} ({plane_counts})")
    print(f"  Errors:           {result.error_count}")

    if not result.errors:
        return

    by_category: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for error in result.errors:
        by_category[str(error["category"])].append(error)

    for category in sorted(by_category):
        category_errors = by_category[category]
        print(f"\n{category} ({len(category_errors)}):")
        for error in category_errors[:20]:
            print(f"  - {_format_error(error)}")
        remaining = len(category_errors) - 20
        if remaining > 0:
            print(f"  ... {remaining} more")


def _write_report(rows: list[dict[str, Any]], error_count: int, target_path: Path) -> None:
    target = _safe_write(target_path)
    temp = _safe_write(target.with_name(f".{target.name}.{os.getpid()}.tmp"))
    temp.parent.mkdir(parents=True, exist_ok=True)
    try:
        write_summary(rows, temp)
        summary_text = temp.read_text()
        temp.write_text(f"<!-- validate_manifest errors: {error_count} -->\n\n{summary_text}")
        temp.replace(target)
    finally:
        if temp.exists():
            temp.unlink()


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--report", action="store_true", help="write data/manifest/manifest_summary.md")
    parser.add_argument("--check-images", action="store_true", help="verify every image_path exists on disk")
    parser.add_argument("--shards-dir", type=Path, default=Path("data/manifest/shards"))
    parser.add_argument(
        "--allocations-dir",
        type=Path,
        default=None,
        help=(
            "path to the allocations directory "
            "(default: data/manifest/allocations relative to the repo root). "
            "Pass an empty or non-existent directory to skip allocation integrity checks."
        ),
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = _parse_args(argv)
    with _forbidden_write_guard():
        result = _validate(
            args.shards_dir,
            check_images=args.check_images,
            allocations_root=args.allocations_dir,
        )
        _print_summary(result)
        if args.report:
            _write_report(result.rows, result.error_count, _REPO_ROOT / "data/manifest/manifest_summary.md")
    return 1 if result.error_count else 0


if __name__ == "__main__":
    raise SystemExit(main())
