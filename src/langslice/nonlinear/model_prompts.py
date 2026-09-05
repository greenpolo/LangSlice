"""Prompt and image-path facts for the registration task.

One base prompt (the original handwritten text) serves every image model;
tune it against real runs, not per-model prose. The working editing copy is
``_local/nonlinear_prompts.md``.
"""

from __future__ import annotations

from langslice.space import Plane

_SECTION_PHRASE_BY_PLANE: dict[Plane, str] = {
    "coronal": "brain coronal section",
    "sagittal": "brain sagittal section",
    "horizontal": "brain horizontal section",
}


def image_model_family(image_model: str | None) -> str:
    """Coarse family of an image model, for image-path capability lookups."""
    model = (image_model or "").lower()
    if "nano-banana" in model or "gemini" in model or "imagen" in model:
        return "nano-banana"
    return "gpt-image"


def aspect_ratio_limits(
    image_model: str | None, provider: str | None = None
) -> tuple[float, float] | None:
    """(min, max) output aspect ratio (w/h) the image path supports, or None.

    Outside the range the working canvas is padded (never cropped) to the
    nearest bound: in edit mode the model paints on ITS canvas, and
    resampling a mismatched ratio back onto the slice would undo the pixel
    alignment.

    Verified 2026-08-25: gpt-image-2 via the OpenAI API accepts arbitrary
    WIDTHxHEIGHT within 1:3..3:1; via openai-oauth (Codex backend) the size
    parameter is ignored and the output matches the input image's aspect
    exactly (probed up to 2.35:1), so the same range is a safe envelope.
    """
    if image_model_family(image_model) == "gpt-image":
        return (1.0 / 3.0, 3.0)
    return None


def base_segmentation_prompt(plane: Plane, image_model: str | None = None) -> str:
    """The image-gen prompt: the original handwritten text, all models."""
    del image_model
    section_phrase = _SECTION_PHRASE_BY_PLANE.get(plane, _SECTION_PHRASE_BY_PLANE["coronal"])
    # Coronal maps black out the ventricular system (see
    # image_gen_helpers._BLACKOUT_PLANES); the model must treat those holes
    # as holes, not as regions to fill.
    ventricle_note = (
        "Ventricles and fluid-filled spaces are left black in the atlas "
        "maps: they are holes in the tissue, not regions — leave the "
        "corresponding fluid spaces in Image 1 unpainted. "
        if plane == "coronal"
        else ""
    )
    return (
        "Edit Image 1: repaint each anatomical region of the brain tissue "
        "in place, using the corresponding solid color from the atlas maps "
        "(Images 2-4). The result is consumed by a machine registration "
        "algorithm as a segmentation label map, not by people: exact colors "
        "and boundary placement matter; visual appeal does not.\n"
        "\n"
        f"IMAGE 1: A real histology photograph of a {section_phrase}. This "
        "is the image being edited.\n"
        "IMAGES 2-4: Colored brain atlas region maps at three neighboring "
        "section depths, 125 microns apart — Image 2 slightly in front of "
        "the section's estimated position, Image 3 at it, Image 4 slightly "
        "behind. Each anatomical region is a unique solid color, delineated "
        "by a thin border in a darker shade of that same color; the border "
        "is part of the region, not a separate structure. All three maps "
        "use the same colors for the same regions.\n"
        "\n"
        "Repaint every part of the tissue in Image 1 with the color of the "
        "atlas region it corresponds to, so the painted regions align with "
        "the anatomy visible in Image 1. The section's true depth lies "
        "somewhere within the span of the three maps: compare the tissue "
        "against all three, and follow whichever map — or intermediate "
        "between them — the tissue actually matches. The atlas is symmetric "
        "and idealized; the real tissue is asymmetric, stretched, and may "
        "have tears or damage.\n"
        "\n"
        "Preserve the exact colors from the atlas maps, and include EVERY "
        "region whose tissue appears in Image 1: do not merge, omit, or "
        "simplify them — each "
        "small nucleus and thin band keeps its own exact color. The atlas "
        "maps are "
        "the authority on which regions exist and which color each one has: "
        "never swap or reassign colors between regions, and never add "
        "regions or subdivisions that the maps do not show. Each region "
        "keeps its general shape and its arrangement relative to its "
        "neighbors from the atlas maps, deformed only to fit this section's "
        "tissue. Segment only tissue that actually exists: if a structure's "
        "tissue is absent from Image 1 — torn away, not mounted, outside "
        "the section — OMIT that region entirely; never squeeze it into "
        "neighboring tissue and never paint it over background. Small "
        "tears and holes inside a region may be painted across. "
        f"{ventricle_note}Reflect "
        "the natural left-right asymmetry of this individual brain section.\n"
        "\n"
        "Do not derive new internal structure from the histology's "
        "texture, lamination, or banding. Within each region, paint one "
        "flat shape the way the atlas maps draw it; the tissue's internal "
        "patterns tell you where a region is and how it bends — never what "
        "to draw inside it.\n"
        "\n"
        "To the best of your ability, ensure that each colored atlas region "
        "corresponds to a visible structure in the "
        "histology image (Image 1). You must balance segmentation of "
        "anatomical objects with maintaining utmost consistency with the "
        "atlas, never omitting any region whose tissue is visible. Every "
        "atlas region whose tissue appears in Image 1 must appear in your "
        "generated image.\n"
        "\n"
        "The painted map must cover the tissue exactly and nothing else: "
        "its outer edge is the tissue's outer edge, traced precisely. Do "
        "not enlarge, shrink, or move any anatomy. Every painted pixel "
        "lies on visible tissue.\n"
        "\n"
        "Change only the tissue; leave every pixel outside the painted "
        "anatomy exactly as it is. No text or labels."
    )


def prior_refinement_prompt(plane: Plane) -> str:
    """Prompt for the ``init="silhouette"`` canvas: correct a placed atlas.

    A different task from :func:`base_segmentation_prompt`, so a different
    prompt: the model is not painting a section from scratch against three
    atlas maps, it is moving the boundaries of an already-placed map onto the
    anatomy underneath it. Measured with this text, the prior canvas raises
    the painting floor from ~0.5 to ~0.8 family dice.
    """
    section_phrase = _SECTION_PHRASE_BY_PLANE.get(plane, _SECTION_PHRASE_BY_PLANE["coronal"])
    return (
        "IMAGE 1 is a flat color map of brain atlas regions that has been "
        "placed over a histology section by a rigid alignment: it is "
        "approximately right, but its region boundaries do not yet follow "
        f"the section's real anatomy. IMAGE 2 is that same {section_phrase} "
        "(the stain's contrast separates cell-dense tissue from fiber "
        "tracts; holes are ventricles), in exactly the same frame and scale "
        "as Image 1.\n\n"
        "Edit Image 1 so that every region boundary follows the anatomy "
        "visible in Image 2: move each boundary onto the visible transition "
        "it corresponds to (cortex against white matter, the edges of "
        "nuclei, fiber bundles, ventricle walls, the outer edge of the "
        "tissue), and make the painted area cover exactly the tissue in "
        "Image 2. Keep every region's color exactly as it is, keep the "
        "regions flat and opaque with no texture, and do not add, remove, "
        "rename or recolor any region. Where a boundary already matches the "
        "tissue, leave it unchanged. Keep the background black. No text or "
        "labels."
    )
