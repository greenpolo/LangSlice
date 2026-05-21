"""LangSliceEstimateEnv — multi-turn estimation env for TRL GRPOTrainer.

The trainer auto-exposes any public method (no leading ``_``, name != ``reset``)
as a tool to the model. We deliberately keep only three tools — ``fetch_atlas``,
``submit_estimate``, ``submit_group_estimate`` — to mirror the production ADK
harness in ``src/langslice_harness/harness/estimation/tools.py`` so a trained
adapter is wire-compatible at inference.

Tool signature shapes match production exactly:
    fetch_atlas(positions_mm, tool_context=None) -> dict
    submit_estimate(position_mm, reasoning, tool_context=None) -> dict
    submit_group_estimate(positions_mm, reasoning, tool_context=None) -> dict

The only divergence from production is async vs sync: TRL ``environment_factory``
requires sync methods. The optional ``tool_context`` kwarg is accepted and
ignored (the env keeps state on ``self._state`` instead of via tool_context).

Defensive done-fence: once ``_state.done`` is set (after a submit), any
further tool call returns an error dict and counts as malformed. This
double-fences the ``stop_tool_names`` configuration on GRPOConfig so the env
is correct-by-construction even if the trainer-side termination is missing.

Hidden ground-truth lives behind a leading underscore and is read by reward
functions via the ``environments`` kwarg, NOT exposed via any tool.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Literal

from .atlas_grid import AtlasGrid

EstimationKind = Literal["single", "group"]
Plane = Literal["coronal", "sagittal", "horizontal"]

# Mirror the production tool's contract — see harness/estimation/tools.py.
DEDUPE_TOL_MM: float = 0.02
MAX_POSITIONS_PER_FETCH: int = 8


@dataclass
class _EpisodeState:
    """Per-rollout episode state. Populated by ``reset``, read by reward funcs."""

    atlas_name: str = ""
    plane: Plane = "coronal"
    pos_lo: float = 0.0
    pos_hi: float = 0.0
    kind: EstimationKind = "single"
    ground_truth_positions_mm: tuple[float, ...] = ()
    submitted_positions_mm: tuple[float, ...] | None = None
    submitted_kind: EstimationKind | None = None
    submitted_reasoning: str = ""
    submit_calls: int = 0
    fetch_calls: int = 0
    malformed_tool_calls: int = 0
    turns: int = 0
    fetched_positions_mm: list[float] = field(default_factory=list)
    done: bool = False
    # Optional curriculum metadata — populated when the row carries a
    # ``section_id``. Reward functions / curriculum loggers read these
    # directly from ``env._state``.
    section_id: str = ""

    def reset(
        self,
        *,
        atlas_name: str,
        plane: Plane,
        pos_lo: float,
        pos_hi: float,
        kind: EstimationKind,
        ground_truth_positions_mm: tuple[float, ...],
        section_id: str = "",
    ) -> None:
        self.atlas_name = atlas_name
        self.plane = plane
        self.pos_lo = float(pos_lo)
        self.pos_hi = float(pos_hi)
        self.kind = kind
        self.ground_truth_positions_mm = ground_truth_positions_mm
        self.submitted_positions_mm = None
        self.submitted_kind = None
        self.submitted_reasoning = ""
        self.submit_calls = 0
        self.fetch_calls = 0
        self.malformed_tool_calls = 0
        self.turns = 0
        self.fetched_positions_mm = []
        self.done = False
        self.section_id = section_id


def _coerce_positions(value: Any) -> list[float] | None:
    """Best-effort coercion of model-produced ``positions_mm`` arguments."""
    if isinstance(value, (int, float)):
        return [float(value)]
    if isinstance(value, str):
        return None
    try:
        return [float(p) for p in value]
    except (TypeError, ValueError):
        return None


def _clamp_and_dedupe(
    positions: list[float], *, pos_lo: float, pos_hi: float
) -> tuple[list[float], int, int]:
    """Clamp into [pos_lo, pos_hi] and dedupe within DEDUPE_TOL_MM.

    Returns (kept, n_clipped, n_deduped) so callers can surface the trimming
    in the tool response text.
    """
    n_clipped = 0
    clamped: list[float] = []
    for p in positions:
        cp = max(pos_lo, min(pos_hi, float(p)))
        if cp != float(p):
            n_clipped += 1
        clamped.append(cp)

    kept: list[float] = []
    n_deduped = 0
    for p in clamped:
        if any(abs(p - q) <= DEDUPE_TOL_MM for q in kept):
            n_deduped += 1
            continue
        kept.append(p)
    return kept, n_clipped, n_deduped


def _done_error(message: str) -> dict[str, Any]:
    return {"status": "error", "error": "ALREADY_DONE", "message": message}


class LangSliceEstimateEnv:
    """Per-rollout estimation environment.

    Constructed once per rollout via ``environment_factory``. The shared
    ``AtlasGrid`` is read-only and safe to share across instances.
    """

    def __init__(self, atlas_grid: AtlasGrid) -> None:
        # Leading-underscore attributes are not exposed as tools by TRL's
        # method-introspection. Reward functions read ``env._state`` directly.
        self._atlas_grid = atlas_grid
        self._state = _EpisodeState()

    def reset(self, **kwargs: Any) -> str | None:
        """Configure the environment for one rollout.

        Receives sampled dataset row fields as kwargs. Required keys:
        ``atlas_name``, ``plane``, ``valid_range_mm``, ``ground_truth_positions_mm``,
        ``kind``. Other row keys (``prompt``, ``image``/``images``, ``subject_id``)
        are ignored here — TRL handles prompt and image plumbing itself.
        """
        atlas_name = str(kwargs["atlas_name"])
        plane: Plane = kwargs["plane"]
        valid_range = kwargs["valid_range_mm"]
        gt_raw = kwargs["ground_truth_positions_mm"]
        kind = kwargs["kind"]
        section_id = str(kwargs.get("section_id") or "")

        if kind not in ("single", "group"):
            raise ValueError(f"kind must be 'single' or 'group', got {kind!r}")
        if kind == "single" and len(gt_raw) != 1:
            raise ValueError(
                f"single-slice kind expects exactly 1 ground-truth position, got {len(gt_raw)}"
            )
        if kind == "group" and len(gt_raw) < 2:
            raise ValueError(f"group kind expects >=2 ground-truth positions, got {len(gt_raw)}")

        pos_lo, pos_hi = float(valid_range[0]), float(valid_range[1])
        if pos_hi <= pos_lo:
            raise ValueError(f"valid_range_mm must be increasing, got {valid_range}")

        # Trip a hard error early if the rollout dataset references an atlas
        # the grid was not built for — better than silently degrading at the
        # first fetch_atlas call.
        self._atlas_grid.range_mm(atlas_name, plane)

        self._state.reset(
            atlas_name=atlas_name,
            plane=plane,
            pos_lo=pos_lo,
            pos_hi=pos_hi,
            kind=kind,
            ground_truth_positions_mm=tuple(float(g) for g in gt_raw),
            section_id=section_id,
        )
        return None

    # --- Tools auto-exposed by GRPOTrainer.environment_factory ---

    def fetch_atlas(
        self,
        positions_mm: list[float],
        tool_context: Any = None,  # noqa: ARG002 — prod-shape only; ignored at training time
    ) -> dict[str, Any]:
        """Fetch reference atlas slices at the requested positions (1-8).

        Positions outside the valid mm range are clamped. Positions within
        0.02 mm of one already requested in this call are deduped. The first
        8 positions are kept; any extras are dropped.

        Args:
            positions_mm: List of mm positions along the slicing plane's normal axis.
            tool_context: Production parity only; ignored here.

        Returns:
            Dict with ``status``, ``positions_mm`` (snapped), ``description`` and
            a ``content`` list of multimodal blocks (image+text, image-before-text)
            suitable for direct return as a TRL multimodal tool response.
        """
        self._state.turns += 1
        self._state.fetch_calls += 1

        if self._state.done:
            self._state.malformed_tool_calls += 1
            return _done_error("Episode already submitted; further tool calls are ignored.")

        coerced = _coerce_positions(positions_mm)
        if coerced is None or len(coerced) == 0:
            self._state.malformed_tool_calls += 1
            return {
                "status": "error",
                "error": "BAD_ARGS",
                "content": [
                    {
                        "type": "text",
                        "text": "Error: positions_mm must be a non-empty list of numbers.",
                    }
                ],
            }

        capped = coerced[:MAX_POSITIONS_PER_FETCH]
        n_extras = len(coerced) - len(capped)
        kept, n_clipped, n_deduped = _clamp_and_dedupe(
            capped, pos_lo=self._state.pos_lo, pos_hi=self._state.pos_hi
        )
        if not kept:
            self._state.malformed_tool_calls += 1
            return {
                "status": "error",
                "error": "EMPTY_RESULT",
                "content": [
                    {
                        "type": "text",
                        "text": "Error: no usable positions after clamp/dedupe.",
                    }
                ],
            }

        notes: list[str] = []
        if n_extras > 0:
            notes.append(f"dropped {n_extras} extra position(s) past the 8-per-call cap")
        if n_clipped > 0:
            notes.append(
                f"clamped {n_clipped} position(s) to range "
                f"[{self._state.pos_lo:.2f}, {self._state.pos_hi:.2f}] mm"
            )
        if n_deduped > 0:
            notes.append(f"deduped {n_deduped} position(s) within {DEDUPE_TOL_MM:.2f} mm")

        content: list[dict[str, Any]] = []
        rendered: list[float] = []
        for pos in kept:
            img, snapped_mm = self._atlas_grid.get_slice(
                self._state.atlas_name, self._state.plane, pos
            )
            # Image MUST come before its text caption so Gemma 4's chat template
            # binds the caption to the right image.
            content.append({"type": "image", "image": img})
            content.append({"type": "text", "text": f"Atlas at {snapped_mm:.2f} mm."})
            rendered.append(snapped_mm)

        self._state.fetched_positions_mm.extend(rendered)

        summary = f"Fetched {len(rendered)} atlas section{'s' if len(rendered) != 1 else ''}."
        if notes:
            summary += " Note: " + "; ".join(notes) + "."
        # Trailing summary text — placed at the end so the model sees per-image
        # captions inline first, then the global "Note:" once at the bottom.
        content.append({"type": "text", "text": summary})

        return {
            "status": "ok",
            "positions_mm": [round(float(p), 2) for p in rendered],
            "description": summary,
            "content": content,
        }

    def submit_estimate(
        self,
        position_mm: float,
        reasoning: str,
        tool_context: Any = None,  # noqa: ARG002
    ) -> dict[str, Any]:
        """Final answer for a single-slice task.

        Args:
            position_mm: Estimated position in mm along the slicing-plane normal axis.
            reasoning: Brief summary of the landmark/atlas evidence supporting the estimate.
            tool_context: Production parity only; ignored here.

        Returns:
            Dict with ``status`` and ``position_mm``. Marks the rollout done.
        """
        self._state.turns += 1
        self._state.submit_calls += 1

        if self._state.done:
            self._state.malformed_tool_calls += 1
            return _done_error("Estimate already submitted; this call is ignored.")

        if self._state.kind != "single":
            self._state.malformed_tool_calls += 1
            return {
                "status": "error",
                "error": "WRONG_KIND",
                "message": (
                    "submit_estimate is only valid for single-slice tasks. "
                    "Use submit_group_estimate for group tasks."
                ),
            }
        try:
            value = float(position_mm)
        except (TypeError, ValueError):
            self._state.malformed_tool_calls += 1
            return {
                "status": "error",
                "error": "BAD_ARGS",
                "message": "position_mm must be a number.",
            }

        self._state.submitted_positions_mm = (value,)
        self._state.submitted_kind = "single"
        self._state.submitted_reasoning = str(reasoning)
        self._state.done = True
        return {"status": "ok", "position_mm": value}

    def submit_group_estimate(
        self,
        positions_mm: list[float],
        reasoning: str,
        tool_context: Any = None,  # noqa: ARG002
    ) -> dict[str, Any]:
        """Final answer for a multi-slice / group task.

        Args:
            positions_mm: One position in mm per input slice, in the same order
                as the slices were presented.
            reasoning: Brief summary of the landmark/atlas evidence supporting the
                estimates and the inferred ordering.
            tool_context: Production parity only; ignored here.

        Returns:
            Dict with ``status`` and ``positions_mm``. Marks the rollout done.
        """
        self._state.turns += 1
        self._state.submit_calls += 1

        if self._state.done:
            self._state.malformed_tool_calls += 1
            return _done_error("Estimate already submitted; this call is ignored.")

        if self._state.kind != "group":
            self._state.malformed_tool_calls += 1
            return {
                "status": "error",
                "error": "WRONG_KIND",
                "message": (
                    "submit_group_estimate is only valid for group tasks. "
                    "Use submit_estimate for single-slice tasks."
                ),
            }
        coerced = _coerce_positions(positions_mm)
        if coerced is None or len(coerced) == 0:
            self._state.malformed_tool_calls += 1
            return {
                "status": "error",
                "error": "BAD_ARGS",
                "message": "positions_mm must be a non-empty list of numbers.",
            }

        values = tuple(float(p) for p in coerced)
        self._state.submitted_positions_mm = values
        self._state.submitted_kind = "group"
        self._state.submitted_reasoning = str(reasoning)
        self._state.done = True
        return {"status": "ok", "positions_mm": [float(v) for v in values]}


def _public_tool_method_names() -> tuple[str, ...]:
    """Names of methods TRL will auto-expose as tools — used by tests."""
    seen: list[str] = []
    for name, attr in vars(LangSliceEstimateEnv).items():
        if name.startswith("_") or name == "reset":
            continue
        if callable(attr):
            seen.append(name)
    return tuple(sorted(seen))
