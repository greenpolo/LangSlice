"""HF ``Dataset`` assembly for the LangSlice RLVR pipeline.

Each row carries everything the env's ``reset(**kwargs)`` and the model's
chat-template need: a chat-format ``prompt``, the tissue ``image``/``images``,
and the metadata that the env consumes (``atlas_name``, ``plane``,
``valid_range_mm``, ``ground_truth_positions_mm``, ``kind``, ``subject_id``).

System prompts come verbatim from
``langslice_harness.harness.estimation.prompts`` so any prompt change in
production is automatically picked up here.
"""

from __future__ import annotations

import json
import random
from collections.abc import Iterable, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal

from PIL import Image

Plane = Literal["coronal", "sagittal", "horizontal"]


def canonicalize_atlas_name(atlas_name: str) -> str:
    try:
        from langslice_harness.atlas.core import canonicalize_atlas_name as _canonicalize
    except ModuleNotFoundError:
        return atlas_name

    return _canonicalize(atlas_name)


def get_position_range_mm(atlas, *, plane: Plane):  # noqa: ANN001
    from langslice_harness.atlas.core import get_position_range_mm as _range_mm

    return _range_mm(atlas, plane=plane)


def load_atlas(atlas_name: str):
    from langslice_harness.atlas.core import load_atlas as _load_atlas

    return _load_atlas(atlas_name)


def species_from_atlas_name(atlas_name: str) -> str:
    from langslice_harness.atlas.core import species_from_atlas_name as _species

    return _species(atlas_name)


def build_group_prompt(**kwargs: Any) -> str:
    from langslice_harness.harness.estimation.prompts import build_group_prompt as _build

    return _build(**kwargs)


def build_single_slice_prompt(**kwargs: Any) -> str:
    from langslice_harness.harness.estimation.prompts import build_single_slice_prompt as _build

    return _build(**kwargs)


@dataclass(frozen=True)
class SingleSliceExample:
    image_path: Path
    atlas_name: str
    plane: Plane
    ap_mm: float
    subject_id: str


@dataclass(frozen=True)
class GroupExample:
    image_paths: tuple[Path, ...]
    atlas_name: str
    plane: Plane
    ap_mm: tuple[float, ...]
    subject_id: str
    interval_mm: float
    thickness_um: int


def _load_test_images_subject(
    subject_dir: Path, *, default_plane: Plane = "coronal"
) -> list[SingleSliceExample]:
    gt_path = subject_dir / "ground_truth.json"
    if not gt_path.exists():
        return []
    raw = json.loads(gt_path.read_text())
    out: list[SingleSliceExample] = []
    for fname, meta in raw.items():
        # Field name is historical (``ap_mm``); the value is the
        # slice-normal-axis position in mm regardless of plane.
        if "ap_mm" not in meta or "atlas" not in meta:
            continue
        image_path = subject_dir / fname
        if not image_path.exists():
            continue
        plane: Plane = meta.get("plane", default_plane)
        out.append(
            SingleSliceExample(
                image_path=image_path,
                atlas_name=str(meta["atlas"]),
                plane=plane,
                ap_mm=float(meta["ap_mm"]),
                subject_id=subject_dir.name,
            )
        )
    return out


def load_test_images(
    test_images_root: Path, *, subject_glob: str = "M*"
) -> list[SingleSliceExample]:
    """Walk ``references/TestImages/M*`` and load all single-slice ground-truths."""
    out: list[SingleSliceExample] = []
    for subject_dir in sorted(test_images_root.glob(subject_glob)):
        if not subject_dir.is_dir():
            continue
        out.extend(_load_test_images_subject(subject_dir))
    return out


def _interval_mm(positions: Sequence[float]) -> float:
    """Median spacing between adjacent positions, fallback 0.0 if <2 slices."""
    if len(positions) < 2:
        return 0.0
    diffs = sorted(abs(positions[i + 1] - positions[i]) for i in range(len(positions) - 1))
    mid = len(diffs) // 2
    return float(diffs[mid] if len(diffs) % 2 else 0.5 * (diffs[mid - 1] + diffs[mid]))


def make_group_examples(
    examples: list[SingleSliceExample],
    *,
    group_size: int,
    rng: random.Random,
    thickness_um: int = 30,
) -> list[GroupExample]:
    """Bin single-slice examples by subject + atlas + plane and slice into groups.

    Slices within a subject are sorted by position; we then take consecutive
    windows of ``group_size`` so the group's ordering and interval are real
    rather than synthesized.
    """
    if group_size < 2:
        raise ValueError(f"group_size must be >= 2, got {group_size}")

    by_subject: dict[tuple[str, str, Plane], list[SingleSliceExample]] = {}
    for ex in examples:
        key = (ex.subject_id, canonicalize_atlas_name(ex.atlas_name), ex.plane)
        by_subject.setdefault(key, []).append(ex)

    out: list[GroupExample] = []
    for (subject_id, atlas_name, plane), bucket in by_subject.items():
        if len(bucket) < group_size:
            continue
        ordered = sorted(bucket, key=lambda e: e.ap_mm)
        # Sliding window — gives more groups per subject than chunked windows
        # and naturally produces overlapping training examples.
        for start in range(0, len(ordered) - group_size + 1):
            window = ordered[start : start + group_size]
            positions = tuple(e.ap_mm for e in window)
            out.append(
                GroupExample(
                    image_paths=tuple(e.image_path for e in window),
                    atlas_name=atlas_name,
                    plane=plane,
                    ap_mm=positions,
                    subject_id=subject_id,
                    interval_mm=_interval_mm(positions),
                    thickness_um=thickness_um,
                )
            )
    rng.shuffle(out)
    return out


def _system_prompt_single(atlas_name: str, plane: Plane) -> tuple[str, float, float]:
    atlas = load_atlas(atlas_name)
    pos_lo, pos_hi = get_position_range_mm(atlas, plane=plane)
    species = species_from_atlas_name(atlas_name)
    prompt = build_single_slice_prompt(
        atlas_name=atlas_name,
        plane=plane,
        pos_lo=pos_lo,
        pos_hi=pos_hi,
        species=species,
    )
    return prompt, pos_lo, pos_hi


def _system_prompt_group(
    atlas_name: str, plane: Plane, n_slices: int, interval_mm: float, thickness_um: int
) -> tuple[str, float, float]:
    atlas = load_atlas(atlas_name)
    pos_lo, pos_hi = get_position_range_mm(atlas, plane=plane)
    species = species_from_atlas_name(atlas_name)
    prompt = build_group_prompt(
        atlas_name=atlas_name,
        plane=plane,
        pos_lo=pos_lo,
        pos_hi=pos_hi,
        species=species,
        n_slices=n_slices,
        interval_mm=interval_mm,
        thickness_um=thickness_um,
    )
    return prompt, pos_lo, pos_hi


def _row_from_single(ex: SingleSliceExample) -> dict[str, Any]:
    system, pos_lo, pos_hi = _system_prompt_single(ex.atlas_name, ex.plane)
    image = Image.open(ex.image_path).convert("RGB")
    user_text = "Estimate the position (mm) of this slice."
    return {
        # Chat messages — image goes IN the user content list before its text,
        # matching Gemma 4's image-before-text rule.
        "prompt": [
            {"role": "system", "content": [{"type": "text", "text": system}]},
            {
                "role": "user",
                "content": [
                    {"type": "image", "image": image},
                    {"type": "text", "text": user_text},
                ],
            },
        ],
        "image": image,
        "atlas_name": canonicalize_atlas_name(ex.atlas_name),
        "plane": ex.plane,
        "valid_range_mm": (pos_lo, pos_hi),
        "ground_truth_positions_mm": (ex.ap_mm,),
        "kind": "single",
        "subject_id": ex.subject_id,
    }


def _row_from_group(ex: GroupExample) -> dict[str, Any]:
    n = len(ex.image_paths)
    system, pos_lo, pos_hi = _system_prompt_group(
        ex.atlas_name,
        ex.plane,
        n_slices=n,
        interval_mm=ex.interval_mm,
        thickness_um=ex.thickness_um,
    )
    images = [Image.open(p).convert("RGB") for p in ex.image_paths]
    user_blocks: list[dict[str, Any]] = []
    for i, img in enumerate(images, start=1):
        user_blocks.append({"type": "image", "image": img})
        user_blocks.append({"type": "text", "text": f"Slice {i} of {n}."})
    user_blocks.append(
        {"type": "text", "text": f"Estimate the position (mm) of all {n} slices, in order."}
    )
    return {
        "prompt": [
            {"role": "system", "content": [{"type": "text", "text": system}]},
            {"role": "user", "content": user_blocks},
        ],
        "images": images,
        "atlas_name": canonicalize_atlas_name(ex.atlas_name),
        "plane": ex.plane,
        "valid_range_mm": (pos_lo, pos_hi),
        "ground_truth_positions_mm": ex.ap_mm,
        "kind": "group",
        "subject_id": ex.subject_id,
    }


def build_rlvr_rows(
    *,
    single_examples: Iterable[SingleSliceExample],
    group_examples: Iterable[GroupExample],
    single_fraction: float = 0.7,
    seed: int = 0,
) -> list[dict[str, Any]]:
    """Mix single-slice and group rows at the requested ratio.

    ``single_fraction`` accepts the closed interval [0.0, 1.0]:
        - ``1.0`` → singles only (Phase A);
        - ``0.0`` → groups only;
        - any value in (0, 1) → mixed, capped by whichever bucket runs out
          first so no row is silently oversampled into duplicates.
    """
    if not 0.0 <= single_fraction <= 1.0:
        raise ValueError(f"single_fraction must be in [0, 1], got {single_fraction}")

    rng = random.Random(seed)
    singles = list(single_examples)
    groups = list(group_examples)
    rng.shuffle(singles)
    rng.shuffle(groups)

    if not singles and not groups:
        return []

    if single_fraction == 1.0:
        n_single, n_group = len(singles), 0
    elif single_fraction == 0.0:
        n_single, n_group = 0, len(groups)
    elif singles and groups:
        # Cap each bucket so the overall mix lands at single_fraction without
        # repeats. Whichever side is the binding constraint determines total.
        max_total_by_singles = int(len(singles) / single_fraction)
        max_total_by_groups = int(len(groups) / (1.0 - single_fraction))
        total = min(max_total_by_singles, max_total_by_groups)
        n_single = int(round(total * single_fraction))
        n_group = total - n_single
    elif singles:
        n_single = len(singles)
        n_group = 0
    else:
        n_single = 0
        n_group = len(groups)

    rows = [_row_from_single(s) for s in singles[:n_single]]
    rows.extend(_row_from_group(g) for g in groups[:n_group])
    rng.shuffle(rows)
    return rows


def split_subjects_for_holdout(
    examples: Iterable[SingleSliceExample],
    *,
    eval_holdout_every: int = 5,
) -> tuple[set[str], set[str]]:
    """Deterministic subject-level train/eval split.

    Subject ids are sorted and every ``eval_holdout_every``-th one (1-indexed,
    so the FIRST eligible position is the Nth-1 element) is assigned to eval.
    A non-positive ``eval_holdout_every`` disables the holdout (everything to
    train, eval set empty).

    Returns ``(train_subject_ids, eval_subject_ids)``. The two sets are
    guaranteed to be disjoint — every subject appears in exactly one.
    """
    sorted_ids = sorted({ex.subject_id for ex in examples})
    if eval_holdout_every <= 0:
        return set(sorted_ids), set()
    eval_ids = {sid for i, sid in enumerate(sorted_ids) if (i + 1) % eval_holdout_every == 0}
    train_ids = set(sorted_ids) - eval_ids
    return train_ids, eval_ids


def to_hf_dataset(rows: list[dict[str, Any]]) -> Any:
    """Wrap the assembled rows in a ``datasets.Dataset``.

    Imported lazily so unit tests for env/rewards can run without ``datasets``
    installed.
    """
    from datasets import Dataset  # noqa: PLC0415

    return Dataset.from_list(rows)
