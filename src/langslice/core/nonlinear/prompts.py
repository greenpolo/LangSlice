"""Prompt texts for every nonlinear registration model call.

Three prompt functions, one per model call shape:

- :func:`border_refinement_prompt` — route "supplied": a rough placement's
  yellow family borders are moved onto the visible tissue edges. Unchanged
  since it was accepted into production; re-exported from
  :mod:`langslice.core.nonlinear.border_refinement` for backward compatibility.
- :func:`pass1_atlas_prompt` — route "atlas", pass 1: yellow family borders
  are drawn from nothing onto the clean tissue, using the grayscale atlas
  template (its own borders drawn in yellow) as the only reference.
- :func:`pass2_atlas_prompt` — route "atlas", optional pass 2: pass 1's lines,
  redrawn on the clean tissue, are corrected against the same atlas template.

Provider selection follows the historical ``base_segmentation_prompt``
convention: ``canonical_provider(provider)`` starting with ``"openai"``
(``openai-api``, ``openai-oauth``) gets the GPT twin; everything else
(``gemini-api``, ``none``, ...) gets the Gemini twin. ``plane`` substitutes
for the word "coronal" the same way the retired ``base_segmentation_prompt``
did — the texts below were measured on coronal sections only.

The pass-1/pass-2 base texts were sentence-audited and approved by the
project owner on 2026-09-19 (audit records kept outside the repo); the final
paragraphs below are the owner's verbatim replacement text for each twin. The
damage block (slide features, missing tissue, faint-vs-missing, displaced
pieces) was expanded on 2026-09-21 ahead of the sagittal / damaged-section
test, because route "atlas" has no agent to describe a section's defects.

The stack agent's image tool uses :func:`border_correction_tool_prompt`: the
supplied-placement prompt (the GPT twin for OpenAI providers), which the agent
may lightly edit for one section.
"""

from __future__ import annotations

from langslice.core.space import Plane
from langslice.providers.registry import canonical_provider


def _for_plane(text: str, plane: Plane) -> str:
    """Substitute the section plane for the word "coronal", byte-identical if unchanged."""
    return text if plane == "coronal" else text.replace("coronal", plane)


def _is_gpt_twin(provider: str | None) -> bool:
    return canonical_provider(provider or "gemini-api").startswith("openai")


def border_refinement_prompt(plane: Plane = "coronal") -> str:
    """The rough-plus-raw experiment wording, without a mouse-only assumption.

    Sentence audit: the first two sentences identify actual attachments and
    their order; the next four specify moving existing lines against visible
    anatomy, retaining correct lines and resolving indistinct boundaries. The
    last three pin the photograph, boundary identities, style and single-image
    output. 'Mouse' is removed and the section plane is parameterized. 'Automatic'
    is removed because supplied placement may come from an agent or a person;
    this leaves the instruction to correct existing rough lines unchanged.

    Route "supplied" only; kept byte-for-byte from the accepted production
    text (moved here from :mod:`langslice.core.nonlinear.border_refinement`, which
    re-exports it for compatibility).
    """
    return (
        f"Image 1 is a photograph of a brain {plane} section with thin yellow "
        "anatomical region boundaries placed by a rough alignment. "
        "Image 2 is the same photograph in exactly the same frame, without the lines, "
        "so the underlying tissue edges are visible.\n\n"
        "Edit Image 1 so that every yellow line lies on the edge of the region it "
        "encloses, as that edge appears in Image 2. "
        "Move a line by sliding or bending it to follow this specimen's anatomy. "
        "A line that already sits on its edge stays as it is. "
        "Where an internal boundary is indistinct, use the neighboring visible structures "
        "and the supplied region arrangement to place it.\n\n"
        "The photograph in the output is Image 2 unchanged beneath the corrected lines: "
        "the same tissue and background, brain size and position, and frame. "
        "The set of boundaries stays the same, each enclosing the corresponding region "
        "in the same thin bright yellow. "
        "The output is one image: the original photograph with the corrected yellow "
        "boundaries replacing the supplied yellow boundaries."
    )


#: Route "supplied" for the GPT image models, as the stack agent's tool sends it:
#: ``_PASS2_GPT`` (owner-approved 2026-09-19, damage block 2026-09-21) with the
#: atlas reference removed. The placed lines of Image 2 now carry the region
#: arrangement the atlas image carried, so "no counterpart" removal and "add a
#: missing boundary" go; everything else, including the damage block, is kept.
#: Image 1 is the clean photograph (the edit target), Image 2 the same
#: photograph with the linearly placed boundaries — pass 2's attachment order.
_SUPPLIED_GPT = (
    "Image 1: the photograph of a brain coronal section to edit. Image 2: the same "
    "photograph in exactly the same frame, carrying thin yellow region boundaries placed "
    "by a rough alignment of the atlas to this specimen; Image 2 alone decides which "
    "boundaries exist.\n\n"
    "Task: return Image 1 with the boundaries of Image 2 drawn on this specimen's "
    "anatomy, using Image 2 as the starting point. Each boundary follows the edge of its "
    "region where that edge shows in the photograph; where the region is faint or "
    "indistinct, the boundary keeps the position and shape it has in Image 2, fitted to "
    "the structures around it. Correct Image 2 two ways. The drawing carries no line "
    "around a feature of the slide rather than of the brain, such as a bubble, a stain or "
    "debris; the feature itself stays in the photograph as it is. A line that does not "
    "hug its region's edge is adjusted by a shift or bend until it does. "
    "Cracks and folds are likewise features of the slide: a boundary that meets one "
    "continues along the anatomy beneath it. Tissue that is physically torn away or "
    "missing from the section, where slide background shows in place of brain, is the "
    "only place a boundary of Image 2 is left out: the part of a boundary that would lie "
    "over missing tissue is omitted and the rest of that boundary is drawn; a torn or cut "
    "edge is not a region edge and gets no line of its own. A region that is present but "
    "faint is not missing, and its boundary is drawn. A piece of tissue that has shifted "
    "or turned keeps its boundaries, drawn on the piece where it lies.\n\n"
    "This is an annotation overlay on a photograph. Change only by adding the yellow "
    "lines; the corrected lines are the only difference from Image 1. Preserve everything "
    "else exactly: every tissue pixel and its texture, the background, the brain's size "
    "and position, the frame and aspect. The lines are thin bright yellow and form "
    "exactly the partition of Image 2 on the surviving tissue: every boundary of Image 2 "
    "that lies on tissue, no other line, no fill, no label. Output one image."
)


def supplied_prompt_is_gpt_twin(provider: str | None) -> bool:
    """Whether route "supplied" sends the GPT twin (clean photograph as Image 1)."""
    return _is_gpt_twin(provider)


def border_correction_tool_prompt(plane: Plane = "coronal", provider: str | None = None) -> str:
    """Base correction prompt for route "supplied", before any per-section edit.

    OpenAI providers get :data:`_SUPPLIED_GPT` (Image 1 the clean photograph,
    Image 2 the placed borders); every other provider keeps the accepted
    :func:`border_refinement_prompt` (Image 1 the placed borders, Image 2 the
    clean photograph). The stack agent may send its own lightly edited copy
    in place of this text. Sentence audit: ``docs/nonlinear_image_tool.md``.
    """
    return (
        _for_plane(_SUPPLIED_GPT, plane) if _is_gpt_twin(provider)
        else border_refinement_prompt(plane)
    )


# --------------------------------------------------------------- route "atlas"

_PASS1_GPT = (
    "Image 1: the photograph of a brain coronal section to edit. Image 2: the grayscale "
    "atlas reference for the corresponding sectioning plane, its region boundaries drawn "
    "in thin yellow lines; Image 2 alone decides which boundaries exist.\n\n"
    "Task: return Image 1 with the boundaries of Image 2 drawn on this specimen's anatomy "
    "in thin bright yellow. Use Image 2 to identify each region and the arrangement of the "
    "regions. Each boundary follows the edge of its region where that edge shows in the "
    "photograph; where the region is faint or indistinct, the boundary takes its position "
    "and shape from Image 2, fitted to the structures around it. Adapt the boundaries to "
    "this specimen's shapes and asymmetry, including a piece of tissue that has shifted or "
    "turned: its boundaries are drawn on the piece where it lies. Bubbles, stains, debris, "
    "cracks and folds are features of the slide, not of the brain: no line follows or "
    "encircles them, and a boundary that meets one continues along the anatomy beneath it. "
    "Tissue that is physically torn away or missing from the section, where slide "
    "background shows in place of brain, is the only place a boundary of Image 2 is left "
    "out: the part of a boundary that would lie over missing tissue is omitted and the rest "
    "of that boundary is drawn; a torn or cut edge is not a region edge and gets no line of "
    "its own. A region that is present but faint is not missing, and its boundary is "
    "drawn.\n\n"
    "This is an annotation overlay on a photograph. Change only by adding the yellow "
    "lines; the lines are the only new thing in the image. Preserve everything else "
    "exactly: every tissue pixel and its texture, the background, the brain's size and "
    "position, the frame and aspect. The lines are thin bright yellow and form exactly "
    "the partition of Image 2 on the tissue: every boundary of Image 2, no other line, "
    "no fill, no label. Output one image."
)

_PASS1_GEMINI = (
    "Image 1 is a photograph of a brain coronal section. Image 2 is a grayscale atlas "
    "reference of the corresponding sectioning plane with its region boundaries drawn in "
    "thin yellow lines; Image 2 alone decides which boundaries exist.\n\n"
    "Draw on Image 1 the boundaries of Image 2, placed on this specimen's anatomy in thin "
    "bright yellow. Use Image 2 to identify each region and the arrangement of the "
    "regions. Each boundary follows the edge of its region where that edge shows in the "
    "photograph, and where the region is faint or indistinct the boundary takes its "
    "position and shape from Image 2, fitted to the structures around it. Adapt the "
    "boundaries to this specimen's shapes and asymmetry, including a piece of tissue that "
    "has shifted or turned: its boundaries are drawn on the piece where it lies. Bubbles, "
    "stains, debris, cracks and folds are features of the slide, not of the brain: the "
    "lines mark brain regions only, and a boundary that meets one of these features "
    "continues along the anatomy beneath it. Tissue that is physically torn away or "
    "missing from the section, where slide background shows in place of brain, is the one "
    "place a boundary of Image 2 is left out: the part of a boundary that would lie over "
    "missing tissue is omitted and the rest of that boundary is drawn; a torn or cut edge "
    "is not a region edge and gets no line of its own. A region that is present but faint "
    "is not missing, and its boundary is drawn.\n\n"
    "This is an annotation overlay on a photograph. Using Image 1, change only by adding "
    "the yellow lines and keep everything else exactly the same: every tissue pixel and "
    "its texture, the background, the same brain size and position, the same frame; the "
    "lines are the only new thing in the image. The lines are thin bright yellow, and the "
    "set of lines is exactly the set of boundaries in Image 2, each drawn once on the "
    "tissue. The output is one image: Image 1 with the yellow anatomical boundaries."
)

_PASS2_GPT = (
    "Image 1: the photograph of a brain coronal section to edit. Image 2: the same "
    "photograph in exactly the same frame, carrying thin yellow region boundaries from a "
    "previous attempt to draw the atlas partition on this specimen. Image 3: the "
    "grayscale atlas reference for the corresponding sectioning plane, its region "
    "boundaries drawn in thin yellow lines; Image 3 alone decides which boundaries "
    "exist.\n\n"
    "Task: return Image 1 with the boundaries of Image 3 drawn on this specimen's "
    "anatomy, using Image 2 as the starting point. Each boundary follows the edge of its "
    "region where that edge shows in the photograph; where the region is faint or "
    "indistinct, the boundary takes its position and shape from Image 3, fitted to the "
    "structures around it. Correct Image 2 three ways. A line that has no counterpart in "
    "Image 3 is removed. The drawing carries no line around a feature of the slide rather "
    "than of the brain, such as a bubble, a stain or debris; the feature itself stays in "
    "the photograph as it is. A line that has a counterpart in Image 3 but does not hug "
    "its region's edge is adjusted by a small shift or bend until it does. A boundary of "
    "Image 3 that Image 2 lacks is added, unless the tissue it would cross is missing. "
    "Cracks and folds are likewise features of the slide: a boundary that meets one "
    "continues along the anatomy beneath it. Tissue that is physically torn away or "
    "missing from the section, where slide background shows in place of brain, is the "
    "only place a boundary of Image 3 is left out: the part of a boundary that would lie "
    "over missing tissue is omitted and the rest of that boundary is drawn; a torn or cut "
    "edge is not a region edge and gets no line of its own. A region that is present but "
    "faint is not missing, and its boundary is drawn. A piece of tissue that has shifted "
    "or turned keeps its boundaries, drawn on the piece where it lies.\n\n"
    "This is an annotation overlay on a photograph. Change only the yellow lines; the "
    "corrected lines are the only difference from Image 1. Preserve everything else "
    "exactly: every tissue pixel and its texture, the background, the brain's size and "
    "position, the frame and aspect. The lines are thin bright yellow and form exactly "
    "the partition of Image 3 on the tissue: every boundary of Image 3, no other line, no "
    "fill, no label. Output one image."
)

_PASS2_GEMINI = (
    "Image 1 is a photograph of a brain coronal section. Image 2 is the same photograph "
    "in exactly the same frame, with thin yellow region boundaries from a previous "
    "attempt to draw the atlas partition on this specimen. Image 3 is a grayscale atlas "
    "reference of the corresponding sectioning plane with its region boundaries drawn in "
    "thin yellow lines; Image 3 alone decides which boundaries exist.\n\n"
    "Draw on Image 1 the boundaries of Image 3, placed on this specimen's anatomy: each "
    "boundary follows the edge of its region where that edge shows in the photograph, and "
    "where the region is faint or indistinct the boundary takes its position and shape "
    "from Image 3, fitted to the structures around it. Image 2 is the starting point; "
    "correct it three ways. A line that exists in Image 2 and nowhere in Image 3 is "
    "removed. Lines around features of the slide rather than of the brain, such as a "
    "bubble, a stain or debris, are left out of the drawing; the feature itself stays in "
    "the photograph as it is. A line that has a counterpart in Image 3 but sits off its "
    "region's edge is adjusted by a small shift or bend until it hugs that edge. A "
    "boundary of Image 3 that Image 2 lacks is added, unless the tissue it would cross is "
    "missing. Cracks and folds are likewise features of the slide: a boundary that meets "
    "one continues along the anatomy beneath it. Tissue that is physically torn away or "
    "missing from the section, where slide background shows in place of brain, is the one "
    "place a boundary of Image 3 is left out: the part of a boundary that would lie over "
    "missing tissue is omitted and the rest of that boundary is drawn; a torn or cut edge "
    "is not a region edge and gets no line of its own. A region that is present but faint "
    "is not missing, and its boundary is drawn. A piece of tissue that has shifted or "
    "turned keeps its boundaries, drawn on the piece where it lies.\n\n"
    "This is an annotation overlay on a photograph. Using Image 1, change only the yellow "
    "lines and keep everything else exactly the same: every tissue pixel and its texture, "
    "the background, the same brain size and position, the same frame; the corrected "
    "lines are the only difference from Image 1. The lines are thin bright yellow, and "
    "the set of lines is exactly the set of boundaries in Image 3, each drawn once on the "
    "tissue. The output is one image: Image 1 with the corrected yellow boundaries."
)


def pass1_atlas_prompt(plane: Plane = "coronal", provider: str | None = None) -> str:
    """Route "atlas", pass 1: draw the outlined atlas's boundaries on clean tissue."""
    text = _PASS1_GPT if _is_gpt_twin(provider) else _PASS1_GEMINI
    return _for_plane(text, plane)


def pass2_atlas_prompt(plane: Plane = "coronal", provider: str | None = None) -> str:
    """Route "atlas", pass 2: correct pass 1's lines against the outlined atlas."""
    text = _PASS2_GPT if _is_gpt_twin(provider) else _PASS2_GEMINI
    return _for_plane(text, plane)
