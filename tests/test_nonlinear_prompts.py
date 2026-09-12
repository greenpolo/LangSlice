"""The registration prompt: one text per model family, verbatim from the bench."""

from __future__ import annotations

import importlib.util
from pathlib import Path

import pytest

from langslice.nonlinear.model_prompts import (
    V14,
    V14_EDIT_MOUSE,
    base_segmentation_prompt,
)

#: Where the prompts were written and measured. Local-only (never shipped), so
#: this comparison runs on a working checkout and skips everywhere else.
_BENCH_PROMPTS = Path(__file__).resolve().parents[1] / "_local" / "nonlinear_eval" / "prompts.py"


@pytest.mark.parametrize(
    ("constant", "bench_name"), [(V14, "V14"), (V14_EDIT_MOUSE, "V14_EDIT_MOUSE")]
)
def test_the_prompt_is_byte_identical_to_the_benchmarked_text(
    constant: str, bench_name: str
) -> None:
    if not _BENCH_PROMPTS.exists():
        pytest.skip(f"{_BENCH_PROMPTS} not present (local-only benchmark tree)")
    spec = importlib.util.spec_from_file_location("_bench_prompts", _BENCH_PROMPTS)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)

    assert constant == getattr(module, bench_name)


@pytest.mark.parametrize("provider", ["gemini-api", "google"])
def test_gemini_gets_the_google_wording(provider: str) -> None:
    assert base_segmentation_prompt("coronal", provider) == V14


@pytest.mark.parametrize(
    "provider", ["openai-oauth", "openai-api", "chatgpt", "openai", None]
)
def test_every_gpt_image_lane_gets_the_codex_wording(provider: str | None) -> None:
    expected = V14 if provider is None else V14_EDIT_MOUSE
    assert base_segmentation_prompt("coronal", provider) == expected


def test_the_prompt_names_the_section_plane() -> None:
    sagittal = base_segmentation_prompt("sagittal", "openai-oauth")

    assert "sagittal section" in sagittal
    assert "coronal" not in sagittal
    # only the plane word moves; the task does not
    assert sagittal.replace("sagittal", "coronal") == V14_EDIT_MOUSE


def test_the_prompt_asks_for_the_map_moved_onto_the_tissue() -> None:
    """The retired lineup asked the model to repaint the SECTION in place."""
    for text in (V14, V14_EDIT_MOUSE):
        assert text.startswith("Edit Image 1")
        assert "IMAGE 1: A colored" in text or "IMAGE 1: A colored region map" in text
        assert "histology photograph" in text.split("IMAGE 3:")[1]
