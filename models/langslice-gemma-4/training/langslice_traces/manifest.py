"""Manifest record parsing for trace-collection jobs."""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Literal, cast

TraceKind = Literal["single", "group"]


@dataclass(frozen=True)
class TraceManifestRow:
    """One single-slice or grouped trace collection job."""

    id: str
    kind: TraceKind
    images: list[str]
    atlas: str
    plane: str
    truth_positions_mm: list[float]
    interval_um: int | None = None
    thickness_um: int = 50
    metadata: dict[str, Any] = field(default_factory=dict)


def _require_str(data: dict[str, Any], key: str) -> str:
    value = data.get(key)
    if not isinstance(value, str) or not value:
        raise ValueError(f"Manifest record requires non-empty string field {key!r}")
    return value


def _require_float(data: dict[str, Any], key: str) -> float:
    value = data.get(key)
    if not isinstance(value, (int, float)):
        raise ValueError(f"Manifest record requires numeric field {key!r}")
    return float(value)


def _require_float_list(data: dict[str, Any], key: str) -> list[float]:
    value = data.get(key)
    if not isinstance(value, list) or not value:
        raise ValueError(f"Manifest record requires non-empty list field {key!r}")
    out: list[float] = []
    for item in value:
        if not isinstance(item, (int, float)):
            raise ValueError(f"Manifest field {key!r} must contain only numbers")
        out.append(float(item))
    return out


def _optional_int(data: dict[str, Any], key: str, default: int | None = None) -> int | None:
    value = data.get(key, default)
    if value is None or isinstance(value, int):
        return value
    raise ValueError(f"Manifest field {key!r} must be an integer")


def _metadata(data: dict[str, Any], known_keys: set[str]) -> dict[str, Any]:
    explicit = data.get("metadata")
    out = dict(explicit) if isinstance(explicit, dict) else {}
    for key, value in data.items():
        if key not in known_keys:
            out[key] = value
    return out


def parse_manifest_record(data: dict[str, Any]) -> TraceManifestRow:
    """Parse and validate one JSON manifest record."""

    record_id = _require_str(data, "id")
    kind = _require_str(data, "kind")
    if kind not in {"single", "group"}:
        raise ValueError(f"Manifest record {record_id!r} has unsupported kind {kind!r}")

    atlas = _require_str(data, "atlas")
    plane = _require_str(data, "plane")
    if kind == "single":
        image = _require_str(data, "image")
        truth_positions = [_require_float(data, "position_mm")]
        images = [image]
        interval_um = None
    else:
        raw_images = data.get("images")
        if not isinstance(raw_images, list) or not raw_images:
            raise ValueError(f"Manifest record {record_id!r} requires non-empty images")
        if not all(isinstance(item, str) and item for item in raw_images):
            raise ValueError(f"Manifest record {record_id!r} images must be strings")
        images = list(raw_images)
        truth_positions = _require_float_list(data, "positions_mm")
        if len(images) != len(truth_positions):
            raise ValueError(
                f"Manifest record {record_id!r} image/position length mismatch"
            )
        interval_um = _optional_int(data, "interval_um")
        if interval_um is None:
            raise ValueError(f"Manifest record {record_id!r} requires interval_um")

    known = {
        "id", "kind", "image", "images", "atlas", "plane", "position_mm",
        "positions_mm", "interval_um", "thickness_um", "metadata",
    }
    thickness_um = _optional_int(data, "thickness_um", 50)
    return TraceManifestRow(
        id=record_id,
        kind=cast(TraceKind, kind),
        images=images,
        atlas=atlas,
        plane=plane,
        truth_positions_mm=truth_positions,
        interval_um=interval_um,
        thickness_um=thickness_um,
        metadata=_metadata(data, known),
    )


def load_manifest(path: str | Path) -> list[TraceManifestRow]:
    """Load a JSONL trace manifest."""

    rows: list[TraceManifestRow] = []
    for line_no, line in enumerate(Path(path).read_text(encoding="utf-8").splitlines(), start=1):
        stripped = line.strip()
        if not stripped or stripped.startswith("#"):
            continue
        data = json.loads(stripped)
        if not isinstance(data, dict):
            raise ValueError(f"Manifest line {line_no} is not an object")
        rows.append(parse_manifest_record(data))
    return rows
