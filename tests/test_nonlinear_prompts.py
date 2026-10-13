"""Prompt selection and content for the nonlinear model call."""

from __future__ import annotations

from langslice.core.nonlinear.prompts import (
    border_refinement_prompt,
)


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
