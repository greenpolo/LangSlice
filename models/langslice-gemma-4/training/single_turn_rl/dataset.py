"""HF-style dataset assembly for single-turn GRPO (Lane A).

Each terminal-state JSONL row becomes a TRL chat-format training row that
mirrors the production ADK protocol's state right before ``submit_estimate``:

    system: build_single_slice_prompt(atlas, plane, ...)
    user:   [TARGET image] + "Determine this {plane} slice's {axis}
             position in the {atlas} atlas."
    assistant: tool_call(fetch_atlas, args={"positions_mm": [pos1, pos2, ...]})
    tool:   [atlas image, "Atlas at X.XX mm:", ...] + summary text
    <model generates next turn — must be submit_estimate>

The rollout starts immediately before the policy's terminal ``submit_estimate``
decision; the upstream fetch_atlas is pre-rendered so the policy only learns
to refine the terminal coordinate, never to invoke (or skip) the atlas
fetch itself. Slicebench evaluates against the same multi-turn protocol, so
training and eval contracts match byte-for-byte.

PIL images are embedded **inline** as ``{"type": "image", "image": <PIL>}``
blocks because TRL's ``GRPOTrainer._tokenize_prompts`` extracts images via
``part["image"]`` from chat content; a parallel ``images`` column is not used
on the no-environment-factory path.

Image-before-text follows Gemma 4's multimodal layout (same rule
:mod:`rlvr.dataset` enforces). The tool-response message needs explicit
``<|image|>`` markers in its text content because Gemma 4's chat template
renders image content blocks for non-tool messages only; for role=tool it
captures just the text. The :func:`_normalize_tool_message_content` helper
injects the markers in trace order so the processor's soft-token count
matches the vision-feature count.

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

from .adaptive_reward import compute_ap_bin
from .prompts import (
    ATLAS_CAPTION_TEMPLATE,
    TARGET_CAPTION,  # noqa: F401 — exported for tests
    build_single_turn_system_prompt,
    build_user_instruction,
)
from .terminal_states import TerminalState, read_terminal_states

Plane = Literal["coronal", "sagittal", "horizontal"]

# Gemma 4 chat-template image marker. Tool messages don't get implicit image
# blocks rendered (the template only walks ``content`` lists for non-tool
# messages), so we inject one ``<|image|>`` per atlas image into the tool
# response's text content. Without this the processor expands fewer soft
# tokens than vision features and the model's image/feature alignment check
# fails at forward. Mirrors ``sft/render.py:normalize_tool_message_content``.
_GEMMA4_IMAGE_TOKEN: str = "<|image|>"

# Synthetic tool-call id stamped on the pre-rendered ``fetch_atlas`` turn.
# The chat template uses the id to match the tool response to its call, so
# the call_id field needs to be consistent between the assistant and tool
# messages — but the value itself is opaque.
_PREFIX_FETCH_CALL_ID: str = "call_prefix_fetch_atlas"

# Gemma 4 tool declaration block — what ``apply_chat_template(..., tools=[...])``
# would emit at the end of the system turn for our two ADK tools. The
# GRPOTrainer doesn't accept tool *schemas* (only callable tools — which would
# trigger multi-turn rollouts we don't want), so we bake the declarations into
# the system message text. Without this, base E4B-IT generates free-text prose
# instead of ``<|tool_call>`` wrappers (verified 100% format-fail at smoke).
# The string is byte-identical to ``apply_chat_template`` output for the
# OpenAI-format schemas of ``fetch_atlas`` and ``submit_estimate``.
_GEMMA4_TOOL_DECLARATIONS: str = (
    '<|tool>declaration:fetch_atlas{'
    'description:<|"|>Fetch 1-8 atlas reference images at given AP positions (mm). '
    'Use to compare landmarks before submitting your estimate.<|"|>,'
    'parameters:{properties:{'
    'positions_mm:{items:{type:<|"|>NUMBER<|"|>},type:<|"|>ARRAY<|"|>}'
    '},required:[<|"|>positions_mm<|"|>],type:<|"|>OBJECT<|"|>}'
    '}<tool|>'
    '<|tool>declaration:submit_estimate{'
    'description:<|"|>Submit your final AP position estimate (mm) for the target slice.<|"|>,'
    'parameters:{properties:{'
    'position_mm:{type:<|"|>NUMBER<|"|>},'
    'reasoning:{type:<|"|>STRING<|"|>}'
    '},required:[<|"|>position_mm<|"|>],type:<|"|>OBJECT<|"|>}'
    '}<tool|>'
)


def _normalize_tool_message_content(
    content: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    """Prepend ``<|image|>`` markers to the text block for each image block.

    Preserves original ordering; if no text block exists, appends one. Used
    so Gemma 4's chat template (which doesn't auto-render images inside
    role=tool) still emits the right number of soft tokens for the
    multimodal processor.
    """
    n_images = sum(1 for c in content if c.get("type") == "image")
    if n_images == 0:
        return content
    markers = _GEMMA4_IMAGE_TOKEN * n_images
    out: list[dict[str, Any]] = []
    text_seen = False
    for c in content:
        if c.get("type") == "text" and not text_seen:
            out.append({"type": "text", "text": markers + c.get("text", "")})
            text_seen = True
        else:
            out.append(c)
    if not text_seen:
        out.append({"type": "text", "text": markers})
    return out


def _build_fetch_atlas_tool_response_text(positions_mm: Sequence[float]) -> str:
    """Mirror the production ``fetch_atlas`` tool result text.

    Format taken verbatim from
    ``langslice_harness.harness.estimation.tools.fetch_atlas`` so the
    training prefix is byte-identical to what the ADK harness emits at
    inference. The text precedes the atlas images in the tool message.
    """
    descriptions = [f"{p:.2f} mm" for p in positions_mm]
    summary = (
        f"Fetched {len(positions_mm)} atlas section"
        f"{'s' if len(positions_mm) != 1 else ''} at: "
        + ", ".join(descriptions)
        + "."
    )
    return summary

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
    "ap_bin",
)


def _ap_pct_for_spec(
    *, ground_truth_mm: float, valid_range_mm: tuple[float, float]
) -> float:
    """Position-fraction along the axis: ``(gt - pos_lo) / (pos_hi - pos_lo)``.

    Used as the cold-start ``difficulty_score`` seed so the AdaRFT band
    sampler has a meaningful prior on round 0 — sections at the anterior
    pole get low scores, posterior pole high. Clamped to ``[0, 1]`` so a
    gt that sits just outside the valid range still produces a legal
    seed. Degenerate (zero-span) axes seed to 0.0.
    """
    pos_lo, pos_hi = float(valid_range_mm[0]), float(valid_range_mm[1])
    axis_span = pos_hi - pos_lo
    if axis_span <= 0.0:
        return 0.0
    ap_pct = (float(ground_truth_mm) - pos_lo) / axis_span
    return max(0.0, min(1.0, ap_pct))


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

    Lane A specs leave ``dataset`` empty (terminal-state JSONL doesn't carry
    the manifest dataset name; that lives in ``quality["dataset"]`` if
    present). ``difficulty_score`` seeds to ``ap_pct`` so the AdaRFT band
    sampler has a real cold-start prior, and ``ap_bin`` is precomputed so
    the curriculum's per-bin refresh doesn't re-derive it every batch.
    """
    valid_range = (float(state.valid_range_mm[0]), float(state.valid_range_mm[1]))
    gt = float(state.ground_truth_mm)
    ap_pct = _ap_pct_for_spec(ground_truth_mm=gt, valid_range_mm=valid_range)
    return {
        "section_id": state.section_id,
        "subject_id": state.subject_id,
        "atlas_name": state.atlas_name,
        "plane": state.plane,
        "dataset": str(state.quality.get("dataset", "")) if state.quality else "",
        "difficulty_score": ap_pct,
        "ap_bin": compute_ap_bin(gt, valid_range),
        "valid_range_mm": valid_range,
        "ground_truth_mm": gt,
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

    Preserves ``section_state.difficulty_score`` when set (seeded/live
    observation). Cold-start Lane B rows intentionally keep
    ``difficulty_score=None`` so curriculum logic can treat them as
    target-matched until live updates arrive.
    """
    valid_range = (
        float(section_state.valid_range_mm[0]),
        float(section_state.valid_range_mm[1]),
    )
    gt = float(section_state.ground_truth_mm)
    difficulty = (
        float(section_state.difficulty_score)
        if section_state.difficulty_score is not None
        else None
    )
    return {
        "section_id": section_state.section_id,
        "subject_id": section_state.subject_id,
        "atlas_name": section_state.atlas_name,
        "plane": section_state.plane,
        "dataset": section_state.dataset,
        "difficulty_score": difficulty,
        "ap_bin": compute_ap_bin(gt, valid_range),
        "valid_range_mm": valid_range,
        "ground_truth_mm": gt,
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

    user_instruction = build_user_instruction(
        plane=spec["plane"], atlas_name=spec["atlas_name"]
    )
    user_blocks: list[dict[str, Any]] = [
        {"type": "image", "image": target},
        {"type": "text", "text": user_instruction},
    ]

    # Pre-render the fetch_atlas tool call + tool response so the rollout
    # begins immediately before the policy's terminal submit_estimate turn.
    # The arguments must be a dict (not a JSON string) — the chat template's
    # ``format_argument`` macro produces the Gemma-native compact format that
    # the reward parser expects.
    fetched_positions: tuple[float, ...] = tuple(
        float(p) for p in spec["fetched_positions_mm"][: len(atlas_images)]
    )
    fetch_args: dict[str, Any] = {"positions_mm": list(fetched_positions)}

    tool_response_content: list[dict[str, Any]] = []
    for pil, pos_mm in zip(atlas_images, fetched_positions, strict=False):
        tool_response_content.append({"type": "image", "image": pil})
        tool_response_content.append(
            {"type": "text", "text": ATLAS_CAPTION_TEMPLATE.format(position_mm=pos_mm)}
        )
    tool_response_content.append(
        {
            "type": "text",
            "text": _build_fetch_atlas_tool_response_text(fetched_positions),
        }
    )

    return {
        "prompt": [
            {
                "role": "system",
                "content": [
                    {"type": "text", "text": system_prompt + _GEMMA4_TOOL_DECLARATIONS},
                ],
            },
            {"role": "user", "content": user_blocks},
            {
                "role": "assistant",
                "content": [],
                "tool_calls": [
                    {
                        "id": _PREFIX_FETCH_CALL_ID,
                        "type": "function",
                        "function": {
                            "name": "fetch_atlas",
                            "arguments": fetch_args,
                        },
                    }
                ],
            },
            {
                "role": "tool",
                "tool_call_id": _PREFIX_FETCH_CALL_ID,
                "content": _normalize_tool_message_content(tool_response_content),
            },
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
        # for band selection. Lane A specs seed to ``ap_pct``; Lane B specs
        # carry the live observation if any, else ``ap_pct`` fallback.
        "difficulty_score": spec.get("difficulty_score"),
        # ``ap_bin`` is the curriculum's per-bin refresh key; precomputed at
        # spec construction so the sampler doesn't redo the bin math.
        "ap_bin": spec.get("ap_bin"),
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
    randomized: bool = False,
    strategy: str = "lane_b_broad_slate",
    atlas_embedding_cache_dir: Path | None = None,
    rng: random.Random | None = None,
) -> tuple[RowDataset, RowDataset | None]:
    """Lane B dataset assembly: pull rows from a :class:`ManifestIndex`.

    Iterates every ``(atlas, plane)`` pair the index reports (filtered to
    ``plane`` if supplied, and to the ``split`` if supplied — defaults to
    ``"rlvr"`` per the plan's split-allocation contract). For each pair,
    yields :class:`SectionState` rows via
    :func:`single_turn_rl.section_state.iter_section_states`, then converts
    each to a row spec via :func:`_spec_from_section_state`.

    Static (default) vs randomized prefix
    -------------------------------------
    Default (``randomized=False``): the deterministic canonical slate path —
    every section for an ``(atlas, plane)`` pair shares the same evenly-
    spaced slate. ``strategy`` and ``atlas_embedding_cache_dir`` are
    ignored on this path; behaviour is byte-identical to pre-Lane A code.

    Randomized (``randomized=True``): every section gets a fresh slate from
    :func:`langslice_traces.generate_trace` with the requested ``strategy``
    (``"lane_b_broad_slate"`` for the existing single-slate path,
    ``"lane_a_prefix"`` for the new multi-step prefix data source). The
    ``atlas_embedding_cache_dir`` argument is required (it provides the
    per-(atlas, plane) grid of fetchable positions); a missing value raises
    :class:`ValueError`. ``rng`` defaults to ``random.Random(seed)`` when
    not provided so the same ``seed`` reproduces the same randomized rows.

    The eval holdout uses the same subject-level deterministic split as
    Lane A so the two lanes are comparable.

    Returns ``(train, eval)``. ``eval`` is ``None`` if the holdout slice is
    empty (e.g. ``eval_holdout_every=0``).
    """
    from .section_state import iter_section_states  # noqa: PLC0415

    if randomized and atlas_embedding_cache_dir is None:
        raise ValueError(
            "build_datasets_from_index(randomized=True) requires "
            "atlas_embedding_cache_dir (the procedural generator needs the "
            "per-(atlas, plane) embedding cache as the universe of "
            "fetchable positions)."
        )
    # Use the caller-provided rng when randomized, else seed one off ``seed``
    # so the existing static path keeps its prior shuffle order.
    rng_for_iter: random.Random | None = None
    if randomized:
        rng_for_iter = rng if rng is not None else random.Random(seed)

    pairs = manifest_index.pairs()
    specs: list[dict[str, Any]] = []
    for atlas, atlas_plane in pairs:
        if plane is not None and atlas_plane != plane:
            continue
        iter_kwargs: dict[str, Any] = {
            "index": manifest_index,
            "plane": atlas_plane,
            "atlas": atlas,
            "split": split,
            "slate_root": slate_root,
            "n_positions": n_positions,
            "require_query_on_disk": require_query_on_disk,
            "require_atlas_on_disk": require_atlas_on_disk,
            "repo_root": repo_root,
        }
        if randomized:
            iter_kwargs["randomized"] = True
            iter_kwargs["strategy"] = strategy
            iter_kwargs["atlas_embedding_cache_dir"] = atlas_embedding_cache_dir
            iter_kwargs["rng"] = rng_for_iter
        for section_state in iter_section_states(**iter_kwargs):
            specs.append(_spec_from_section_state(section_state, repo_root=repo_root))

    shuffle_rng = random.Random(seed)
    if max_examples is not None and 0 < int(max_examples) < len(specs):
        shuffle_rng.shuffle(specs)
        specs = specs[: int(max_examples)]
    train_subjects, eval_subjects = split_subjects_for_holdout(
        specs, eval_holdout_every=eval_holdout_every
    )
    train_specs = [s for s in specs if str(s["subject_id"]) in train_subjects]
    eval_specs = [s for s in specs if str(s["subject_id"]) in eval_subjects]
    shuffle_rng.shuffle(train_specs)
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
