"""Prompt selection and content for all three nonlinear model calls."""

from __future__ import annotations

import pytest

from langslice.core.nonlinear.prompts import (
    border_refinement_prompt,
    pass1_atlas_prompt,
    pass2_atlas_prompt,
)

_PASS1_GPT_FINAL_PARAGRAPH = (
    "This is an annotation overlay on a photograph. Change only by adding the yellow "
    "lines; the lines are the only new thing in the image. Preserve everything else "
    "exactly: every tissue pixel and its texture, the background, the brain's size and "
    "position, the frame and aspect. The lines are thin bright yellow and form exactly "
    "the partition of Image 2 on the tissue: every boundary of Image 2, no other line, "
    "no fill, no label. Output one image."
)

_PASS1_GEMINI_FINAL_PARAGRAPH = (
    "This is an annotation overlay on a photograph. Using Image 1, change only by adding "
    "the yellow lines and keep everything else exactly the same: every tissue pixel and "
    "its texture, the background, the same brain size and position, the same frame; the "
    "lines are the only new thing in the image. The lines are thin bright yellow, and the "
    "set of lines is exactly the set of boundaries in Image 2, each drawn once on the "
    "tissue. The output is one image: Image 1 with the yellow anatomical boundaries."
)

_PASS2_GPT_FINAL_PARAGRAPH = (
    "This is an annotation overlay on a photograph. Change only the yellow lines; the "
    "corrected lines are the only difference from Image 1. Preserve everything else "
    "exactly: every tissue pixel and its texture, the background, the brain's size and "
    "position, the frame and aspect. The lines are thin bright yellow and form exactly "
    "the partition of Image 3 on the tissue: every boundary of Image 3, no other line, no "
    "fill, no label. Output one image."
)

_PASS2_GPT_FEATURE_SENTENCE = (
    "The drawing carries no line around a feature of the slide rather than of the brain, "
    "such as a bubble, a stain or debris; the feature itself stays in the photograph as "
    "it is."
)

_PASS2_GEMINI_FINAL_PARAGRAPH = (
    "This is an annotation overlay on a photograph. Using Image 1, change only the yellow "
    "lines and keep everything else exactly the same: every tissue pixel and its texture, "
    "the background, the same brain size and position, the same frame; the corrected "
    "lines are the only difference from Image 1. The lines are thin bright yellow, and "
    "the set of lines is exactly the set of boundaries in Image 3, each drawn once on the "
    "tissue. The output is one image: Image 1 with the corrected yellow boundaries."
)

_PASS2_GEMINI_FEATURE_SENTENCE = (
    "Lines around features of the slide rather than of the brain, such as a bubble, a "
    "stain or debris, are left out of the drawing; the feature itself stays in the "
    "photograph as it is."
)


@pytest.mark.parametrize("provider", ["openai-oauth", "openai-api"])
def test_pass1_gpt_twin_for_every_gpt_image_lane(provider: str) -> None:
    text = pass1_atlas_prompt("coronal", provider)
    assert text.endswith(_PASS1_GPT_FINAL_PARAGRAPH)
    assert "Image 2 alone decides which boundaries exist" in text


@pytest.mark.parametrize("provider", ["gemini-api", "none", None])
def test_pass1_gemini_twin_for_every_other_provider(provider: str | None) -> None:
    text = pass1_atlas_prompt("coronal", provider)
    assert text.endswith(_PASS1_GEMINI_FINAL_PARAGRAPH)


@pytest.mark.parametrize("provider", ["openai-oauth", "openai-api"])
def test_pass2_gpt_twin_for_every_gpt_image_lane(provider: str) -> None:
    text = pass2_atlas_prompt("coronal", provider)
    assert text.endswith(_PASS2_GPT_FINAL_PARAGRAPH)
    assert _PASS2_GPT_FEATURE_SENTENCE in text
    assert "A line drawn around a feature of the slide" not in text


@pytest.mark.parametrize("provider", ["gemini-api", "none"])
def test_pass2_gemini_twin_for_every_other_provider(provider: str) -> None:
    text = pass2_atlas_prompt("coronal", provider)
    assert text.endswith(_PASS2_GEMINI_FINAL_PARAGRAPH)
    assert _PASS2_GEMINI_FEATURE_SENTENCE in text
    assert "A line drawn around a feature of the slide" not in text


def test_pass_prompts_name_the_section_plane() -> None:
    for prompt_fn in (pass1_atlas_prompt, pass2_atlas_prompt):
        for provider in ("openai-oauth", "gemini-api"):
            sagittal = prompt_fn("sagittal", provider)
            assert "sagittal section" in sagittal
            assert "coronal" not in sagittal
            assert sagittal.replace("sagittal", "coronal") == prompt_fn("coronal", provider)


def test_pass2_uses_image_2_as_the_starting_point_and_image_3_as_the_atlas() -> None:
    gpt = pass2_atlas_prompt("coronal", "openai-oauth")
    gemini = pass2_atlas_prompt("coronal", "gemini-api")
    for text in (gpt, gemini):
        assert "Image 2" in text and "Image 3" in text
        assert "starting point" in text
        assert "Image 3 alone decides which boundaries exist" in text


def test_border_refinement_prompt_is_unchanged_route_supplied_text() -> None:
    text = border_refinement_prompt("coronal")
    assert text.startswith(
        "Image 1 is a photograph of a brain coronal section with thin yellow "
        "anatomical region boundaries placed by a rough alignment."
    )
    assert text.endswith(
        "The output is one image: the original photograph with the corrected yellow "
        "boundaries replacing the supplied yellow boundaries."
    )


def test_border_refinement_prompt_names_the_section_plane() -> None:
    sagittal = border_refinement_prompt("sagittal")
    assert "sagittal section" in sagittal
    assert "coronal" not in sagittal
    assert sagittal.replace("sagittal", "coronal") == border_refinement_prompt("coronal")
