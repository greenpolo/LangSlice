"""HF-style dataset assembly for single-turn GRPO (Lane A).

Each terminal-state JSONL row becomes a TRL chat-format training row whose
``user`` content interleaves PIL images and short captions in the order

    [target image] "TARGET slice."
    [atlas image]  "Atlas reference at X.XX mm."
    [atlas image]  "Atlas reference at Y.YY mm."
    ...
    "Output the JSON object for the TARGET slice's position now."

PIL images are embedded **inline** as ``{"type": "image", "image": <PIL>}``
blocks because TRL's ``GRPOTrainer._tokenize_prompts`` extracts images via
``part["image"]`` from chat content; a parallel ``images`` column is not used
on the no-environment-factory path.

Image-before-text follows Gemma 4's multimodal layout (same rule
:mod:`rlvr.dataset` enforces). Atlas images are sent individually because
production ADK runs default to ``send_individually=True``.

The dataset row also exposes the metadata GRPOTrainer's reward + atlas-cache
machinery need:

* ``ground_truth_mm`` — float, the GT coordinate.
* ``valid_range_mm`` — ``(pos_lo, pos_hi)``.
* ``image_paths`` — repo-relative paths in the same order PIL images appear in
  the chat content. The atlas-cache splice consumes this to look up cached
  SigLIP embeddings without re-encoding.
* ``plane`` / ``atlas_name`` / ``subject_id`` / ``section_id`` — diagnostics
  + curriculum bin lookup.

GRPOTrainer forwards every dataset column except ``prompt`` to reward
functions as ``**kwargs``, so :func:`rewards.make_terminal_reward` sees
``ground_truth_mm`` and ``valid_range_mm`` per call.

Subject-level holdout (same primitive used by ``rlvr.dataset``) keeps train
and eval rows from sharing brains.
"""

from __future__ import annotations

import random
from collections.abc import Iterable, Sequence
from pathlib import Path
from typing import Any, Literal

from PIL import Image

# ``rlvr.dataset`` reuse — preprocessing must match the SFT corpus / RLVR lane
# byte-for-byte so the policy sees the same query-image distribution it saw at
# SFT time (atlas-aware downsample). CLAHE is already baked into the staged
# SFT query files, so we pass ``apply_clahe=False`` here.
from rlvr.dataset import (
    SingleSliceExample,
    _atlas_in_plane_long_edge,
    canonicalize_atlas_name,
    preprocess_query_image,
    species_from_atlas_name,
)

from .prompts import (
    ATLAS_CAPTION_TEMPLATE,
    TARGET_CAPTION,
    USER_INSTRUCTION,
    build_single_turn_system_prompt,
)
from .terminal_states import TerminalState, read_terminal_states

Plane = Literal["coronal", "sagittal", "horizontal"]

# Public columns every row exposes — what GRPOTrainer + reward + curriculum
# bin lookup all consume. Underscore-free so TRL's identity collator preserves
# them and reward funcs receive them as ``**kwargs``.
#
# Lane B (index-driven) rows additionally carry ``dataset`` so the adaptive
# reward callback can write per-section live-difficulty back to the
# manifest index keyed on ``(plane, dataset, section_id)``. Lane A rows leave
# ``dataset`` empty/missing — the adaptive callback's write-back path handles
# that gracefully via ``if_unknown="warn"``.
_PUBLIC_COLUMNS: tuple[str, ...] = (
    "prompt",
    "image_paths",
    "ground_truth_mm",
    "valid_range_mm",
    "plane",
    "atlas_name",
    "subject_id",
    "section_id",
    "dataset",
    "difficulty_score",
)


def load_atlas_reference_image(path: Path) -> Image.Image:
    """Load an atlas reference image straight from the SFT corpus staging dir.

    Atlas images are pre-rendered by the production tool at native atlas
    resolution; we don't resize them here. ``convert("RGB")`` makes the
    decoder predictable for the chat template.
    """
    with Image.open(str(path)) as raw:
        return raw.copy().convert("RGB")


def _spec_from_state(state: TerminalState, *, repo_root: Path) -> dict[str, Any]:
    """Pre-resolve cheap metadata; defer image decode + chat-content build to
    :meth:`RowDataset.__getitem__`. The atlas long-edge lookup is cached so
    repeated calls for the same (atlas, plane) pair are O(1).

    Lane A specs leave ``dataset`` and ``difficulty_score`` empty: the
    terminal-state JSONL doesn't carry per-section difficulty, and dataset
    name is inside ``quality["dataset"]`` if present.
    """
    return {
        "section_id": state.section_id,
        "subject_id": state.subject_id,
        "atlas_name": state.atlas_name,
        "plane": state.plane,
        "dataset": str(state.quality.get("dataset", "")) if state.quality else "",
        "difficulty_score": None,
        "valid_range_mm": (float(state.valid_range_mm[0]), float(state.valid_range_mm[1])),
        "ground_truth_mm": float(state.ground_truth_mm),
        "repo_root": str(repo_root),
        "query_image_path": state.query_image_path,
        "atlas_image_paths": tuple(state.atlas_image_paths),
        "fetched_positions_mm": tuple(state.fetched_positions_mm),
        "atlas_long_edge": _atlas_in_plane_long_edge(state.atlas_name, state.plane),
    }


def _spec_from_section_state(section_state: Any, *, repo_root: Path) -> dict[str, Any]:
    """Build a row spec from a :class:`single_turn_rl.section_state.SectionState`.

    Lane B counterpart of :func:`_spec_from_state`. The two share row shape
    so :class:`RowDataset` can hold a mixed list — but we keep the builders
    separate so each lane can fail loudly if a wrong-shape spec ever leaks
    into the other path. ``SectionState.atlas_slate_paths`` /
    ``atlas_slate_positions_mm`` map directly onto Lane A's
    ``atlas_image_paths`` / ``fetched_positions_mm``.
    """
    return {
        "section_id": section_state.section_id,
        "subject_id": section_state.subject_id,
        "atlas_name": section_state.atlas_name,
        "plane": section_state.plane,
        "dataset": section_state.dataset,
        "difficulty_score": section_state.difficulty_score,
        "valid_range_mm": (
            float(section_state.valid_range_mm[0]),
            float(section_state.valid_range_mm[1]),
        ),
        "ground_truth_mm": float(section_state.ground_truth_mm),
        "repo_root": str(repo_root),
        "query_image_path": section_state.query_image_path,
        "atlas_image_paths": tuple(section_state.atlas_slate_paths),
        "fetched_positions_mm": tuple(section_state.atlas_slate_positions_mm),
        "atlas_long_edge": _atlas_in_plane_long_edge(
            section_state.atlas_name, section_state.plane
        ),
    }


def _build_row_from_spec(spec: dict[str, Any]) -> dict[str, Any]:
    """Decode images and assemble the TRL chat-format row.

    PIL images are embedded inside the chat content blocks. The companion
    ``image_paths`` column carries the paths in the same order so the
    atlas-cache splice can look them up.
    """
    repo_root = Path(spec["repo_root"])
    target = preprocess_query_image(
        repo_root / spec["query_image_path"],
        atlas_long_edge=int(spec["atlas_long_edge"]),
        apply_clahe=False,
    )
    atlas_paths: tuple[str, ...] = spec["atlas_image_paths"]
    atlas_images = [load_atlas_reference_image(repo_root / p) for p in atlas_paths]

    species = species_from_atlas_name(spec["atlas_name"])
    pos_lo, pos_hi = spec["valid_range_mm"]
    system_prompt = build_single_turn_system_prompt(
        atlas_name=spec["atlas_name"],
        plane=spec["plane"],
        pos_lo=pos_lo,
        pos_hi=pos_hi,
        species=species,
    )

    user_blocks: list[dict[str, Any]] = [
        {"type": "image", "image": target},
        {"type": "text", "text": TARGET_CAPTION},
    ]
    fetched_positions: tuple[float, ...] = spec["fetched_positions_mm"]
    for pil, pos_mm in zip(atlas_images, fetched_positions[: len(atlas_images)], strict=False):
        user_blocks.append({"type": "image", "image": pil})
        user_blocks.append(
            {"type": "text", "text": ATLAS_CAPTION_TEMPLATE.format(position_mm=pos_mm)}
        )
    user_blocks.append({"type": "text", "text": USER_INSTRUCTION})

    return {
        "prompt": [
            {"role": "system", "content": [{"type": "text", "text": system_prompt}]},
            {"role": "user", "content": user_blocks},
        ],
        "image_paths": [spec["query_image_path"], *atlas_paths],
        "ground_truth_mm": spec["ground_truth_mm"],
        "valid_range_mm": spec["valid_range_mm"],
        "plane": spec["plane"],
        "atlas_name": canonicalize_atlas_name(spec["atlas_name"]),
        "subject_id": spec["subject_id"],
        "section_id": spec["section_id"],
        # ``dataset`` defaults to "" for Lane A specs (terminal states don't
        # natively carry the manifest dataset name) and to the Section's
        # ``dataset`` for Lane B specs. The adaptive reward callback's
        # write-back path tolerates both via ``if_unknown="warn"``.
        "dataset": str(spec.get("dataset", "") or ""),
        # ``difficulty_score`` is consumed by :class:`CurriculumRepeatingSampler`
        # for band selection. Lane A specs carry ``None`` (cold-start);
        # Lane B specs carry the live or seeded score.
        "difficulty_score": spec.get("difficulty_score"),
    }


class RowDataset:
    """Map-style dataset with lazy image decode.

    Stores cheap metadata specs (no PIL images) at construction time;
    :meth:`__getitem__` decodes images and assembles the chat-format row on
    demand. Mirrors :class:`rlvr.dataset.RowDataset` in spirit but emits PIL
    images inline in chat content (per TRL's VLM extraction rule, not as a
    parallel ``images`` column).
    """

    column_names: tuple[str, ...] = _PUBLIC_COLUMNS

    def __init__(self, specs: list[dict[str, Any]]) -> None:
        self._specs = specs

    def __len__(self) -> int:
        return len(self._specs)

    def __getitem__(self, index: int) -> dict[str, Any]:
        return _build_row_from_spec(self._specs[index])


class WeightedRowDataset(RowDataset):
    """:class:`RowDataset` + a per-row ``_weights`` list for curriculum sampling.

    The :class:`curriculum.sampler.CurriculumGRPOTrainer` subclass swaps in a
    :class:`torch.utils.data.WeightedRandomSampler` whenever the training
    dataset exposes ``_weights``. Default weights are all 1.0, which makes
    ``WeightedRandomSampler`` produce a uniform-with-replacement distribution —
    statistically equivalent to ``RandomSampler`` until
    :meth:`set_weights` overrides them.
    """

    def __init__(self, specs: list[dict[str, Any]]) -> None:
        super().__init__(specs)
        self._weights: list[float] = [1.0] * len(specs)
        # ``curriculum.weights.update_weighted_dataset`` reads
        # ``dataset._rows[i][section_id_key]`` — alias to ``_specs`` so the
        # curriculum helper works without changes (rlvr-style row dicts and
        # single-turn-style spec dicts both expose ``section_id``).
        self._rows = self._specs

    def set_weights(self, weights: Sequence[float]) -> None:
        """Replace per-row sampling weights. Validates non-negativity + a positive sum."""
        n = len(self)
        if len(weights) != n:
            raise ValueError(
                f"set_weights: expected {n} values (len(dataset)), got {len(weights)}"
            )
        coerced = [float(w) for w in weights]
        if any(w < 0.0 for w in coerced):
            raise ValueError("set_weights: weights must be non-negative")
        if not any(w > 0.0 for w in coerced):
            raise ValueError("set_weights: at least one weight must be > 0")
        self._weights = coerced

    @property
    def specs(self) -> list[dict[str, Any]]:
        """Read-only access to the underlying specs (used by curriculum bin
        computation to map section_ids to weights)."""
        return list(self._specs)


def load_specs(
    terminal_states_path: Path,
    *,
    repo_root: Path,
) -> list[dict[str, Any]]:
    """Read the JSONL and build row specs (cheap; no image decode here)."""
    return [_spec_from_state(s, repo_root=repo_root) for s in read_terminal_states(terminal_states_path)]


def split_subjects_for_holdout(
    specs: Iterable[dict[str, Any]],
    *,
    eval_holdout_every: int = 5,
) -> tuple[set[str], set[str]]:
    """Subject-level deterministic split (same primitive as :mod:`rlvr.dataset`)."""
    sorted_ids = sorted({str(s["subject_id"]) for s in specs})
    if eval_holdout_every <= 0:
        return set(sorted_ids), set()
    eval_ids = {sid for i, sid in enumerate(sorted_ids) if (i + 1) % eval_holdout_every == 0}
    return set(sorted_ids) - eval_ids, eval_ids


def build_datasets(
    terminal_states_path: Path,
    *,
    repo_root: Path,
    eval_holdout_every: int = 5,
    max_examples: int | None = None,
    seed: int = 0,
    weighted: bool = False,
) -> tuple[RowDataset, RowDataset | None]:
    """Assemble ``(train, eval)`` map-style datasets with subject-level holdout.

    When ``weighted=True`` the train dataset is a :class:`WeightedRowDataset`
    so the trainer can switch to a :class:`torch.utils.data.WeightedRandomSampler`
    via the curriculum machinery.
    """
    specs = load_specs(terminal_states_path, repo_root=repo_root)
    rng = random.Random(seed)
    if max_examples is not None and 0 < int(max_examples) < len(specs):
        rng.shuffle(specs)
        specs = specs[: int(max_examples)]
    train_subjects, eval_subjects = split_subjects_for_holdout(
        specs, eval_holdout_every=eval_holdout_every
    )
    train_specs = [s for s in specs if str(s["subject_id"]) in train_subjects]
    eval_specs = [s for s in specs if str(s["subject_id"]) in eval_subjects]
    rng.shuffle(train_specs)
    train_cls: type[RowDataset] = WeightedRowDataset if weighted else RowDataset
    train = train_cls(train_specs)
    evald = RowDataset(eval_specs) if eval_specs else None
    return train, evald


def build_datasets_from_index(
    *,
    manifest_index: Any,
    plane: str | None = None,
    split: str | None = "rlvr",
    repo_root: Path,
    slate_root: Path,
    n_positions: int = 9,
    eval_holdout_every: int = 5,
    max_examples: int | None = None,
    seed: int = 0,
    require_query_on_disk: bool = True,
    require_atlas_on_disk: bool = True,
) -> tuple[RowDataset, RowDataset | None]:
    """Lane B dataset assembly: pull rows from a :class:`ManifestIndex`.

    Iterates every ``(atlas, plane)`` pair the index reports (filtered to
    ``plane`` if supplied, and to the ``split`` if supplied — defaults to
    ``"rlvr"`` per the plan's split-allocation contract). For each pair,
    yields :class:`SectionState` rows via
    :func:`single_turn_rl.section_state.iter_section_states`, then converts
    each to a row spec via :func:`_spec_from_section_state`.

    The eval holdout uses the same subject-level deterministic split as
    Lane A so the two lanes are comparable.

    Returns ``(train, eval)``. ``eval`` is ``None`` if the holdout slice is
    empty (e.g. ``eval_holdout_every=0``).
    """
    from .section_state import iter_section_states  # noqa: PLC0415

    pairs = manifest_index.pairs()
    specs: list[dict[str, Any]] = []
    for atlas, atlas_plane in pairs:
        if plane is not None and atlas_plane != plane:
            continue
        for section_state in iter_section_states(
            index=manifest_index,
            plane=atlas_plane,
            atlas=atlas,
            split=split,
            slate_root=slate_root,
            n_positions=n_positions,
            require_query_on_disk=require_query_on_disk,
            require_atlas_on_disk=require_atlas_on_disk,
            repo_root=repo_root,
        ):
            specs.append(_spec_from_section_state(section_state, repo_root=repo_root))

    rng = random.Random(seed)
    if max_examples is not None and 0 < int(max_examples) < len(specs):
        rng.shuffle(specs)
        specs = specs[: int(max_examples)]
    train_subjects, eval_subjects = split_subjects_for_holdout(
        specs, eval_holdout_every=eval_holdout_every
    )
    train_specs = [s for s in specs if str(s["subject_id"]) in train_subjects]
    eval_specs = [s for s in specs if str(s["subject_id"]) in eval_subjects]
    rng.shuffle(train_specs)
    train = RowDataset(train_specs)
    evald = RowDataset(eval_specs) if eval_specs else None
    return train, evald


def to_hf_dataset(specs: list[dict[str, Any]]) -> RowDataset:
    """Lightweight wrapper for symmetry with :mod:`rlvr.dataset`."""
    return RowDataset(specs)


def specs_to_single_slice_examples(specs: Iterable[dict[str, Any]]) -> list[SingleSliceExample]:
    """Build :class:`SingleSliceExample` stubs for curriculum bin computation.

    The bin computation only reads ``atlas_name``, ``plane``, ``ap_mm``,
    ``subject_id``, ``section_id`` — image_path is irrelevant — so we can
    construct stubs cheaply from the row specs without re-walking the manifest.
    """
    out: list[SingleSliceExample] = []
    for s in specs:
        out.append(
            SingleSliceExample(
                image_path=Path("/dev/null"),
                atlas_name=str(s["atlas_name"]),
                plane=str(s["plane"]),  # type: ignore[arg-type]
                ap_mm=float(s["ground_truth_mm"]),
                subject_id=str(s["subject_id"]),
                section_id=str(s["section_id"]),
            )
        )
    return out
