"""The silhouette prior: tissue mask, placement, and the model-free backbone.

Fixtures follow ``test_harness_registration_candidates.py`` — a toy atlas and
monkeypatched slice accessors — but the atlas here is anatomy-shaped (an
asymmetric blob of three differently-colored regions), because the placement
is a moments fit: a symmetric fixture cannot tell the two rotations apart.
"""

from __future__ import annotations

from types import SimpleNamespace

import cv2
import numpy as np
import pytest
from PIL import Image

from langslice.affine import _affine_from_pose, _moments_pose, silhouette_iou
from langslice.cli import _build_parser
from langslice.nonlinear.model_prompts import prior_refinement_prompt
from langslice.nonlinear.prior import (
    build_silhouette_prior,
    place_plane_on_tissue,
    tissue_mask,
)
from langslice.nonlinear.types import GeneratedSegmentation

_H, _W = 120, 160
_ROWS = {
    997: ("root", [997], "root", (255, 255, 255)),
    1: ("CTX", [997, 1], "cortex", (255, 40, 40)),
    2: ("STR", [997, 2], "striatum", (40, 200, 40)),
    3: ("OB", [997, 3], "olfactory bulb", (60, 60, 255)),
}


def _atlas_plane() -> np.ndarray:
    """A brain-ish plane: body, inner nucleus, and a bulb off one end."""
    plane = np.zeros((_H, _W), dtype=np.int32)
    yy, xx = np.mgrid[0:_H, 0:_W]
    plane[((yy - 60) / 46.0) ** 2 + ((xx - 90) / 58.0) ** 2 <= 1.0] = 1
    plane[((yy - 60) / 24.0) ** 2 + ((xx - 95) / 28.0) ** 2 <= 1.0] = 2
    plane[((yy - 60) / 22.0) ** 2 + ((xx - 28) / 22.0) ** 2 <= 1.0] = 3  # asymmetry
    return plane


@pytest.fixture
def atlas(monkeypatch: pytest.MonkeyPatch) -> SimpleNamespace:
    import langslice.atlas.render as helpers

    plane = _atlas_plane()
    fake = SimpleNamespace(
        atlas_name="toy_prior_atlas",
        annotation=np.stack([plane, plane, plane]),
        template=np.zeros((3, _H, _W), dtype=np.uint16),
        resolution=(25, 25, 25),
        structures={
            sid: {
                "id": sid,
                "acronym": acronym,
                "name": name,
                "structure_id_path": path,
                "rgb_triplet": list(rgb),
            }
            for sid, (acronym, path, name, rgb) in _ROWS.items()
        },
    )
    monkeypatch.setattr(helpers, "position_mm_to_index", lambda a, p, plane="coronal": 1)
    monkeypatch.setattr(helpers, "slice_axis_index", lambda ctx, plane: 0)
    monkeypatch.setattr(helpers, "atlas_space_context", lambda a: SimpleNamespace())
    monkeypatch.setattr(helpers, "orient_slice_for_display", lambda a, plane: a)
    return fake


def _synthetic_slice(size: tuple[int, int] = (240, 180)) -> Image.Image:
    """The atlas plane rotated and shrunk, painted as gray tissue on black."""
    matrix = cv2.getRotationMatrix2D((_W / 2, _H / 2), 12.0, 0.9)
    matrix[0, 2] += size[0] / 2 - _W / 2
    matrix[1, 2] += size[1] / 2 - _H / 2
    warped = cv2.warpAffine(
        _atlas_plane().astype(np.float64), matrix, size,
        flags=cv2.INTER_NEAREST, borderValue=0,
    ).astype(np.int32)
    gray = np.zeros(warped.shape, dtype=np.uint8)
    for uid, level in ((1, 110), (2, 190), (3, 70)):
        gray[warped == uid] = level
    noise = np.random.default_rng(0).integers(-6, 6, gray.shape)
    gray = np.clip(gray.astype(int) + noise, 0, 255).astype(np.uint8)
    return Image.fromarray(np.dstack([gray, gray, gray]), mode="RGB")


def _install_atlas(monkeypatch: pytest.MonkeyPatch, atlas: SimpleNamespace):
    import langslice.nonlinear.image_gen_registration as reg

    monkeypatch.setattr(reg, "load_atlas", lambda atlas_name: atlas)
    monkeypatch.setattr(
        reg, "get_position_range_mm", lambda a, plane="coronal": (-10.0, 10.0)
    )
    return reg


def _classify(atlas: SimpleNamespace, image: Image.Image) -> np.ndarray:
    from langslice.nonlinear.image_gen_helpers import _classify_pixels_to_region_ids

    return _classify_pixels_to_region_ids(
        np.asarray(image.convert("RGB"), dtype=np.uint8),
        atlas,
        0.5,
        off_palette_background=False,
    )


# --------------------------------------------------------------- tissue mask


def test_tissue_mask_keeps_the_largest_blob_and_fills_its_holes() -> None:
    frame = np.zeros((100, 100, 3), dtype=np.uint8)
    frame[20:80, 20:80] = 200  # the section
    frame[40:60, 40:60] = 0  # a ventricle-shaped hole inside it
    frame[5:12, 85:95] = 200  # debris elsewhere on the slide

    mask = tissue_mask(Image.fromarray(frame, mode="RGB"), (100, 100)) > 0

    assert mask[20:80, 20:80].all(), "the hole was not filled"
    assert not mask[5:12, 85:95].any(), "debris survived the largest-blob cut"
    assert not mask[0:15, 0:15].any()


def test_tissue_mask_reads_a_bright_background_too() -> None:
    """Polarity comes from the frame's own border, not from an assumed black."""
    frame = np.full((100, 100, 3), 240, dtype=np.uint8)
    frame[30:70, 30:70] = 60

    mask = tissue_mask(Image.fromarray(frame, mode="RGB"), (100, 100)) > 0

    assert mask[30:70, 30:70].all()
    assert not mask[0:20, 0:20].any()


def test_tissue_mask_refuses_a_blank_field() -> None:
    blank = Image.fromarray(np.full((60, 60, 3), 12, dtype=np.uint8), mode="RGB")
    with pytest.raises(ValueError, match="could not be found"):
        tissue_mask(blank, (60, 60))


# ----------------------------------------------------------------- placement


def test_placement_picks_the_better_sign_and_lands_on_the_tissue(
    atlas: SimpleNamespace,
) -> None:
    from langslice.nonlinear.image_gen_helpers import _annotation_slice

    section = _synthetic_slice()
    mask = tissue_mask(section, section.size)
    labels = _annotation_slice(atlas, 0.5)

    placed, signs, iou = place_plane_on_tissue(labels, mask)

    # Both rotations, scored the way the placement scores them.
    src = _moments_pose((labels != 0).astype(np.uint8) * 255)
    dst = _moments_pose(mask)
    scores = {}
    for candidate_signs in ((1, 1), (-1, -1)):
        warped = cv2.warpAffine(
            labels.astype(np.float64),
            _affine_from_pose(*src, *dst, candidate_signs),
            section.size,
            flags=cv2.INTER_NEAREST,
            borderValue=0,
        )
        scores[candidate_signs] = silhouette_iou(warped != 0, mask > 0)

    assert scores[signs] == max(scores.values())
    assert scores[signs] > scores[min(scores, key=lambda k: scores[k])], (
        "the fixture is symmetric — the two rotations are indistinguishable"
    )
    assert iou > 0.9
    assert silhouette_iou(placed != 0, mask > 0) > 0.9
    assert set(np.unique(placed)) == {0, 1, 2, 3}


def test_the_prior_is_canvas_sized_and_classifies_back_to_its_own_labels(
    atlas: SimpleNamespace,
) -> None:
    section = _synthetic_slice()

    prior, metadata = build_silhouette_prior(
        section, atlas=atlas, position_mm=0.5, plane="coronal"
    )

    assert prior.size == section.size
    assert metadata["tissue_iou"] > 0.9
    assert metadata["sign_pattern"] in ([1, 1], [-1, -1])
    # Lossless by construction: fills and delineation lines are both exact
    # palette colors, so nothing is invented at classification time.
    classified = _classify(atlas, prior)
    assert set(np.unique(classified)) == {0, 1, 2, 3}
    painted = classified != 0
    tissue = tissue_mask(section, section.size) > 0
    assert silhouette_iou(painted, tissue) > 0.9


# ------------------------------------------------------- model-free backbone


def test_model_free_backbone_registers_the_prior_end_to_end(
    atlas: SimpleNamespace, monkeypatch: pytest.MonkeyPatch, tmp_path
) -> None:
    """provider="none" + affine: real Elastix, no model, everything downstream."""
    reg = _install_atlas(monkeypatch, atlas)
    section = _synthetic_slice()

    candidate = reg.generate_registration_candidate(
        section,
        atlas_name="toy_prior_atlas",
        position_mm=0.5,
        provider="none",
        init="silhouette",
        deformation="affine",
        candidate_id="backbone",
        debug_dir=str(tmp_path),
    )

    assert candidate.metadata["init"] == "silhouette"
    assert candidate.metadata["provider"] == "none"
    assert candidate.metadata["generated"]["route"] == "silhouette_prior"
    assert candidate.metadata["prior"]["sign_pattern"] in ([1, 1], [-1, -1])
    assert candidate.metadata["prior"]["tissue_iou"] > 0.9
    assert candidate.markers, "no VisuAlign markers came out of the deformation field"
    assert candidate.metadata["deformation"] == "affine"

    artifacts = tmp_path / "registration" / "backbone"
    prior = Image.open(artifacts / "input_prior.png").convert("RGB")
    assert prior.size == candidate.warped_atlas.size
    # The painting IS the prior, so the fit's only job is to move the atlas
    # render onto it — the warped atlas has to come back as the prior's map.
    prior_ids = _classify(atlas, prior)
    warped_ids = _classify(atlas, candidate.warped_atlas)
    foreground = prior_ids != 0
    agreement = float((prior_ids[foreground] == warped_ids[foreground]).mean())
    assert agreement > 0.95, f"warped atlas agrees with the prior on only {agreement:.3f}"


def test_model_free_without_the_prior_is_refused(
    atlas: SimpleNamespace, monkeypatch: pytest.MonkeyPatch
) -> None:
    reg = _install_atlas(monkeypatch, atlas)
    with pytest.raises(ValueError, match="silhouette"):
        reg.generate_registration_candidate(
            _synthetic_slice(),
            atlas_name="toy_prior_atlas",
            position_mm=0.5,
            provider="none",
        )


# -------------------------------------------------------- prior as the canvas


def _fake_downstream(monkeypatch: pytest.MonkeyPatch, reg) -> dict:
    """Everything past classification, faked — this is an input-side test."""
    calls: dict = {}

    def fake_generate(request):
        calls["request"] = request
        # A no-op edit: the model hands the canvas straight back. Every pixel
        # is "unchanged", which is exactly the case the preserved-background
        # mask must NOT erase on a prior canvas.
        return GeneratedSegmentation(
            image=request.slice_image.copy(),
            provider=request.provider,
            model=request.model or "fake-model",
            route="fake",
        )

    def fake_register(
        atlas_classified, generated_classified, atlas, fixed_mask=None,
        deformation="bspline",
    ):
        calls["generated_classified"] = generated_classified.copy()
        return SimpleNamespace(name="fake-transform"), 0.5

    monkeypatch.setattr(reg, "generate_warped_segmentation_image", fake_generate)
    monkeypatch.setattr(reg, "_register_region_maps", fake_register)
    monkeypatch.setattr(
        reg, "_warp_classified_labels", lambda classified, transform: classified
    )
    monkeypatch.setattr(reg, "_compute_deformation_field", lambda transform, moving: None)
    monkeypatch.setattr(
        reg,
        "_extract_visualign_markers",
        lambda field, scale_to_slice, origin_px=(0.0, 0.0): [],
    )
    monkeypatch.setattr(
        reg,
        "_run_inverse_warp_for_slice",
        lambda slice_rgb, **kwargs: (slice_rgb, SimpleNamespace(name="fake-inverse")),
    )
    return calls


def test_silhouette_init_sends_the_prior_first_and_the_section_second(
    atlas: SimpleNamespace, monkeypatch: pytest.MonkeyPatch
) -> None:
    reg = _install_atlas(monkeypatch, atlas)
    calls = _fake_downstream(monkeypatch, reg)
    section = _synthetic_slice()

    candidate = reg.generate_registration_candidate(
        section,
        atlas_name="toy_prior_atlas",
        position_mm=0.5,
        provider="openai",
        init="silhouette",
        candidate_id="refine",
    )

    request = calls["request"]
    # Image 1 is the prior: a flat map of exact palette colors, not tissue.
    expected_prior, _ = build_silhouette_prior(
        section, atlas=atlas, position_mm=0.5, plane="coronal"
    )
    assert np.array_equal(np.asarray(request.slice_image), np.asarray(expected_prior))
    # Image 2 is the section, and the atlas maps are NOT sent.
    assert len(request.reference_images) == 1
    assert np.asarray(request.reference_images[0]).shape == np.asarray(section).shape
    assert np.abs(
        np.asarray(request.reference_images[0], dtype=int)
        - np.asarray(section.resize(request.reference_images[0].size), dtype=int)
    ).mean() < 1.0
    assert request.prompt == prior_refinement_prompt("coronal")

    # The model changed nothing, and on a prior canvas that is valid paint:
    # the registered map keeps every region rather than collapsing to
    # background the way the section-canvas contract requires.
    registered = calls["generated_classified"]
    assert set(np.unique(registered)) == {0, 1, 2, 3}
    assert (registered != 0).mean() > 0.1
    assert candidate.metadata["init"] == "silhouette"


def test_atlas_init_still_masks_the_untouched_canvas(
    atlas: SimpleNamespace, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The default contract is unchanged: an unedited section is background."""
    reg = _install_atlas(monkeypatch, atlas)
    calls = _fake_downstream(monkeypatch, reg)

    reg.generate_registration_candidate(
        _synthetic_slice(),
        atlas_name="toy_prior_atlas",
        position_mm=0.5,
        provider="openai",
        candidate_id="plain",
    )

    request = calls["request"]
    assert request.prompt.startswith("Edit Image 1")
    assert len(request.reference_images) == 3  # the +-125um atlas band
    assert not calls["generated_classified"].any(), "preserved canvas was not masked"


# ------------------------------------------------------------------- the CLI


def test_cli_parses_the_model_free_backbone() -> None:
    args = _build_parser().parse_args(
        [
            "nonlinear", "register", "slice.png", "--position", "5.2",
            "--init", "silhouette", "--provider", "none", "--deformation", "affine",
        ]
    )
    assert args.init == "silhouette"
    assert args.provider == "none"
    assert args.deformation == "affine"


def test_cli_init_defaults_to_atlas() -> None:
    args = _build_parser().parse_args(
        ["nonlinear", "register", "slice.png", "--position", "5.2"]
    )
    assert args.init == "atlas"


def test_register_request_carries_init() -> None:
    from langslice.api.models import RegisterRequest

    request = RegisterRequest(
        image_path="slice.png", atlas="toy", position_mm=1.0,
        provider="none", init="silhouette", deformation="affine",
    )
    assert request.init == "silhouette"
    assert request.provider == "none"
    assert RegisterRequest(image_path="s.png", atlas="t", position_mm=1.0).init == "atlas"
