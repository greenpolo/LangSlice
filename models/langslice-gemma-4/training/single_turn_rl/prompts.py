"""Strict-JSON final-answer prompts for single-turn GRPO (Lane A).

The single-turn lane bypasses the multi-turn agent loop entirely. The prompt
shows the policy the same query + atlas evidence the production agent had at
the moment it should call ``submit_estimate``, and the contract is to emit
nothing but a one-key JSON object with the predicted coordinate.

We keep two pieces of the production prompt verbatim so transfer is preserved:

* The plane-axis label (``AP`` / ``ML`` / ``DV``) and coordinate-system
  boilerplate from ``langslice_harness.harness.estimation.prompts`` — same
  semantics the policy saw at SFT time.
* The atlas + species + valid range header — the policy still has to obey the
  axis bounds.

Everything else is replaced with a hard "JSON only" instruction. No landmark
prose, no broad/narrow/recover protocol, no reasoning field. The reward parser
in :mod:`rewards` returns ``format_penalty`` when the completion is anything
other than a single ``{"position_mm": <number>}`` object.
"""

from __future__ import annotations

from typing import Literal

Plane = Literal["coronal", "sagittal", "horizontal"]


_PLANE_BOILERPLATE: dict[str, str] = {
    "coronal":
        "Coordinate system: 0.0 mm is the anterior edge (olfactory bulb); "
        "larger mm moves posterior toward the cerebellum.",
    "sagittal":
        "Coordinate system: 0.0 mm is at the lateral edge of one hemisphere; "
        "larger mm moves toward the midline. The two hemispheres are mirror "
        "images, so estimate the canonical (single-hemisphere) ML position; "
        "your estimate should fall in the canonical hemisphere range.",
    "horizontal":
        "Coordinate system: 0.0 mm is dorsal (top of brain); "
        "larger mm moves ventral.",
}


_PLANE_AXIS_LABEL: dict[str, str] = {
    "coronal": "AP",
    "sagittal": "ML",
    "horizontal": "DV",
}


SYSTEM_PROMPT_TEMPLATE: str = (
    "You are an expert neuroanatomist. You are given a histology brain slice "
    "image (TARGET) followed by a sequence of atlas reference images already "
    "fetched from positions across the {axis} axis. Your job is to choose the "
    "single best {axis} position for the TARGET in millimeters.\n\n"
    "{boilerplate}\n\n"
    "Atlas: {atlas_name} ({species}). "
    "Valid {axis} range: {pos_lo:.2f}-{pos_hi:.2f} mm.\n\n"
    "Output contract (STRICT):\n"
    "Reply with ONE JSON object and nothing else. No prose, no markdown, no "
    "code fences, no extra keys. Schema:\n"
    "    {{\"position_mm\": <number in [{pos_lo:.2f}, {pos_hi:.2f}]>}}\n"
)


USER_INSTRUCTION: str = (
    "Output the JSON object for the TARGET slice's position now."
)


ATLAS_CAPTION_TEMPLATE: str = "Atlas reference at {position_mm:.2f} mm."


TARGET_CAPTION: str = "TARGET slice."


def build_single_turn_system_prompt(
    *,
    atlas_name: str,
    plane: Plane,
    pos_lo: float,
    pos_hi: float,
    species: str,
) -> str:
    """Return the strict-JSON single-turn system instruction.

    Mirrors the production single-slice prompt's plane boilerplate and atlas
    header (so axis semantics carry over from SFT) but replaces the multi-step
    landmarks/sweep/narrow protocol with a "JSON only" final-answer contract.
    """
    return SYSTEM_PROMPT_TEMPLATE.format(
        axis=_PLANE_AXIS_LABEL[plane],
        boilerplate=_PLANE_BOILERPLATE[plane],
        atlas_name=atlas_name,
        species=species,
        pos_lo=pos_lo,
        pos_hi=pos_hi,
    )
