"""Single-turn GRPO prompts that mirror the production ADK protocol.

Earlier versions of this module used a strict "JSON only" final-answer
contract — bare ``{"position_mm": N}`` output. That format diverged from
``slicebench``'s evaluation path, which uses ADK + Gemma 4 tool-call sentinels
(``<|tool_call>call:submit_estimate{position_mm:N,...}<tool_call|>``). A
checkpoint trained on bare JSON would be incompatible with that eval path.

Now the single-turn lane reuses the production system prompt and the production
``submit_estimate``/``fetch_atlas`` tool schemas. The dataset module
(:mod:`single_turn_rl.dataset`) pre-renders one ``fetch_atlas`` tool call +
its tool response into the chat history, so each rollout begins immediately
before the policy's ``submit_estimate`` turn. The reward parser
(:mod:`single_turn_rl.rewards`) extracts ``position_mm`` from the model's
emitted tool call.
"""

from __future__ import annotations

from typing import Literal

from langslice_harness.harness.estimation.prompts import build_single_slice_prompt

Plane = Literal["coronal", "sagittal", "horizontal"]


_PLANE_AXIS_LABEL: dict[str, str] = {
    "coronal": "AP",
    "sagittal": "ML",
    "horizontal": "DV",
}


# Atlas reference captions are rendered inside the synthetic ``fetch_atlas``
# tool response (see :mod:`single_turn_rl.dataset`). The text matches what
# the production ``fetch_atlas`` tool emits.
ATLAS_CAPTION_TEMPLATE: str = "Atlas at {position_mm:.2f} mm:"
TARGET_CAPTION: str = "TARGET slice."


def build_single_turn_system_prompt(
    *,
    atlas_name: str,
    plane: Plane,
    pos_lo: float,
    pos_hi: float,
    species: str,
) -> str:
    """Return the production single-slice system instruction (axis-aware).

    Identical to what ``slicebench`` + the ADK harness pass at inference,
    so the trained policy sees the same plane/atlas/strategy contract at
    train time and eval time.
    """
    return build_single_slice_prompt(
        atlas_name=atlas_name,
        plane=plane,
        pos_lo=pos_lo,
        pos_hi=pos_hi,
        species=species,
    )


def build_user_instruction(*, plane: Plane, atlas_name: str) -> str:
    """Brief user-turn instruction that follows the TARGET image.

    Matches ``slicebench``'s harness prompt verbatim — ``runner.py`` emits
    ``f"Determine this {plane} slice's {axis_label} position in the {atlas_name} atlas."``.
    Keeping this text identical avoids the prompt-mismatch regression that
    cost ~10% mae on Phase 5/6 SFTs.
    """
    axis = _PLANE_AXIS_LABEL[plane]
    return (
        f"Determine this {plane} slice's {axis} position in the "
        f"{atlas_name} atlas."
    )
