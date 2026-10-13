"""Prompt texts for every nonlinear registration model call.

- :func:`border_correction_tool_prompt` — route "supplied" (``trace_borders``):
  a placement's yellow family borders moved onto the visible tissue edges.
  OpenAI providers get :data:`_SUPPLIED_GPT` (Image 1 the clean photograph),
  every other provider :func:`border_refinement_prompt` (Image 1 the placed
  borders).

``canonical_provider(provider)`` starting with ``"openai"`` (``openai-api``,
``openai-oauth``) gets the GPT twin; everything else (``gemini-api``,
``none``, ...) the Gemini twin. ``plane`` substitutes for the word "coronal";
the texts were written and checked on coronal sections. Every sentence of
these texts was audited for a second reading (``CLAUDE.md`` in this package,
"Prompt review").
"""

from __future__ import annotations

from langslice.core.provider_names import canonical_provider
from langslice.core.space import Plane


def _for_plane(text: str, plane: Plane) -> str:
    """Substitute the section plane for the word "coronal", byte-identical if unchanged."""
    return text if plane == "coronal" else text.replace("coronal", plane)


def _is_gpt_twin(provider: str | None) -> bool:
    return canonical_provider(provider or "gemini-api").startswith("openai")


def border_refinement_prompt(plane: Plane = "coronal") -> str:
    """Route "supplied"'s Gemini wording: Image 1 the placed borders, Image 2
    the clean photograph.

    Sentence audit: the first two sentences identify actual attachments and
    their order; the next four specify moving existing lines against visible
    anatomy, retaining correct lines and resolving indistinct boundaries. The
    last three pin the photograph, boundary identities, style and single-image
    output. 'Mouse' is removed and the section plane is parameterized. 'Automatic'
    is removed because supplied placement may come from an agent or a person;
    this leaves the instruction to correct existing rough lines unchanged.
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


#: Route "supplied" for the GPT image models, as the stack agent's tool sends it.
#: Image 1 is the clean photograph (the edit target), Image 2 the same
#: photograph with the linearly placed boundaries.
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
    in place of this text. Sentence audit: ``core/nonlinear/CLAUDE.md``.
    """
    return (
        _for_plane(_SUPPLIED_GPT, plane) if _is_gpt_twin(provider)
        else border_refinement_prompt(plane)
    )
