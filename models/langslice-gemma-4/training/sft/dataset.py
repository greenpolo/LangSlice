"""Load + validate langslice-native SFT trace JSONL; subject-aware train/eval split."""

from __future__ import annotations

import json
import random
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

EVAL_USER_PROMPT_TEXT = "Estimate the position of this slice."
"""Default user prompt for eval-time generation when no Example carries one.

The renderer uses Example.user_prompt_text per row at training time. This
constant is what the eval callback uses for held-out evaluation, where
there is no source Example. Production data-distillation must emit
matching text in trace JSONL or the eval-train distribution drifts.
"""


class DatasetValidationError(ValueError):
    """Raised when a JSONL row fails schema validation."""


@dataclass
class Example:
    bucket: int
    atlas_name: str
    atlas_version: str
    plane: str
    subject_id: str
    system_prompt_kind: str            # always "single_slice" in v1
    query_image_paths: list[str]
    user_prompt_text: str
    trace: list[dict[str, Any]]        # langslice-native trace; see spec section 6.1
    gemini_reasoning: str | None = None  # ignored by trainer in v1
    dataset_root: Path | None = field(default=None, compare=False)


_REQUIRED_FIELDS = (
    "bucket", "atlas_name", "atlas_version", "plane", "subject_id",
    "system_prompt_kind", "query_image_paths", "user_prompt_text", "trace",
)
_VALID_KIND = "single_slice"
_SUBMIT_NAME = "submit_estimate"


def load_examples(jsonl_path: str | Path) -> list[Example]:
    """Load and validate every row in *jsonl_path*. Raise on the first defect."""
    path = Path(jsonl_path)
    root = path.parent
    examples: list[Example] = []
    with path.open("r", encoding="utf-8") as f:
        for lineno, raw in enumerate(f, start=1):
            line = raw.strip()
            if not line:
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError as e:
                raise DatasetValidationError(f"line {lineno}: invalid JSON ({e.msg})") from e
            _validate_row(row, lineno, root)
            ex = Example(
                bucket=row["bucket"],
                atlas_name=row["atlas_name"],
                atlas_version=row["atlas_version"],
                plane=row["plane"],
                subject_id=row["subject_id"],
                system_prompt_kind=row["system_prompt_kind"],
                query_image_paths=list(row["query_image_paths"]),
                user_prompt_text=row["user_prompt_text"],
                trace=list(row["trace"]),
                gemini_reasoning=row.get("gemini_reasoning"),
                dataset_root=root,
            )
            examples.append(ex)
    if not examples:
        raise DatasetValidationError(f"{path}: no examples loaded (file is empty)")
    return examples


def _validate_row(row: dict[str, Any], lineno: int, root: Path) -> None:
    for required in _REQUIRED_FIELDS:
        if required not in row:
            raise DatasetValidationError(f"line {lineno}: missing required field '{required}'")
    if row["bucket"] != 1:
        raise DatasetValidationError(
            f"line {lineno}: bucket must be 1 in v1 (got {row['bucket']})"
        )
    if not isinstance(row["subject_id"], str) or not row["subject_id"].strip():
        raise DatasetValidationError(f"line {lineno}: subject_id must be non-empty string")
    kind = row["system_prompt_kind"]
    if kind != _VALID_KIND:
        raise DatasetValidationError(
            f"line {lineno}: system_prompt_kind must be {_VALID_KIND!r} in v1 (got {kind!r})"
        )
    qips = row["query_image_paths"]
    if not isinstance(qips, list) or not qips:
        raise DatasetValidationError(f"line {lineno}: query_image_paths must be non-empty list")
    if len(qips) != 1:
        raise DatasetValidationError(
            f"line {lineno}: v1 single_slice requires exactly 1 query image (got {len(qips)})"
        )
    if "interval_mm" in row or "thickness_um" in row:
        raise DatasetValidationError(
            f"line {lineno}: group-only interval_mm/thickness_um are not accepted in v1"
        )
    trace = row["trace"]
    if not isinstance(trace, list) or not trace:
        raise DatasetValidationError(f"line {lineno}: trace must be non-empty list")
    last = trace[-1]
    if "submit" not in last:
        raise DatasetValidationError(f"line {lineno}: trace must end with a submit step")
    submit = last["submit"]
    submit_name = submit.get("name")
    if submit_name != _SUBMIT_NAME:
        raise DatasetValidationError(
            f"line {lineno}: submit.name must be {_SUBMIT_NAME!r} for v1 "
            f"(got {submit_name!r})"
        )
    args = submit.get("args", {})
    if not isinstance(args.get("position_mm"), (int, float)):
        raise DatasetValidationError(f"line {lineno}: submit.args.position_mm must be numeric")
    if not isinstance(args.get("reasoning"), str) or not args["reasoning"].strip():
        raise DatasetValidationError(f"line {lineno}: submit.args.reasoning must be non-empty string")
    for i, step in enumerate(trace[:-1]):
        if "tool_call" not in step or "tool_result" not in step:
            raise DatasetValidationError(
                f"line {lineno}: trace[{i}] must have both tool_call and tool_result"
            )
    _validate_image_paths(row, lineno, root)


def _validate_image_paths(row: dict[str, Any], lineno: int, root: Path) -> None:
    for rel in row["query_image_paths"]:
        _require_existing_image(root, rel, lineno, "query_image_paths")
    for i, step in enumerate(row["trace"][:-1]):
        tool_result = step["tool_result"]
        image_paths = tool_result.get("image_paths", [])
        if not isinstance(image_paths, list) or not image_paths:
            raise DatasetValidationError(
                f"line {lineno}: trace[{i}].tool_result.image_paths must be non-empty list"
            )
        for rel in image_paths:
            _require_existing_image(root, rel, lineno, f"trace[{i}].tool_result.image_paths")


def _require_existing_image(root: Path, rel: Any, lineno: int, field_name: str) -> None:
    if not isinstance(rel, str) or not rel:
        raise DatasetValidationError(f"line {lineno}: {field_name} entries must be non-empty strings")
    path = (root / rel).resolve()
    if not path.is_file():
        raise DatasetValidationError(f"line {lineno}: {field_name} image not found: {path}")


def split_subject_aware(
    examples: list[Example],
    *,
    holdout_fraction: float,
    seed: int,
) -> tuple[list[Example], list[Example]]:
    """Split *examples* into (train, eval) so no subject_id appears in both."""
    if not 0.0 < holdout_fraction < 1.0:
        raise ValueError(f"holdout_fraction must be in (0, 1), got {holdout_fraction}")

    by_subject: dict[str, list[Example]] = {}
    for ex in examples:
        by_subject.setdefault(ex.subject_id, []).append(ex)

    subjects = sorted(by_subject.keys())  # deterministic order before shuffle
    rng = random.Random(seed)
    rng.shuffle(subjects)

    n_holdout = max(1, round(len(subjects) * holdout_fraction))
    if n_holdout >= len(subjects):
        raise ValueError(
            f"holdout_fraction={holdout_fraction} would consume all "
            f"{len(subjects)} subjects; need at least one for train"
        )
    eval_subjects = set(subjects[:n_holdout])

    train: list[Example] = []
    eval_: list[Example] = []
    for ex in examples:
        if ex.subject_id in eval_subjects:
            eval_.append(ex)
        else:
            train.append(ex)
    return train, eval_
