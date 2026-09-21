"""Standalone unit tests for helpers both border routes share.

The candidate-building pipeline these tests once exercised end-to-end
(`_generate_registration_candidate`, the colormap workflow) is deleted; what
remains here are the geometry, Elastix-report, and marker-sampling helpers
that `border_registration.py`/`border_refinement.py` still call, tested
directly rather than through that deleted pipeline.
"""

from __future__ import annotations

from types import SimpleNamespace
from typing import Any

import numpy as np
import pytest
from PIL import Image


def _make_slice(size: tuple[int, int] = (12, 8)) -> Image.Image:
    image = Image.new("RGB", size, color=(30, 40, 50))
    for y in range(size[1]):
        for x in range(size[0]):
            image.putpixel((x, y), (30 + x, 40 + y, 50))
    return image


def test_build_atlas_root_mask_produces_binary_alpha_at_target_size(monkeypatch):
    """`_build_atlas_root_mask` slices annotation at the AP index for the
    requested plane, marks non-zero structure IDs as opaque (255) and zeros
    as transparent (0), and NEAREST-resizes to *target_size* so alpha stays
    binary -- bilinear interpolation would halo the 3D-viewer silhouette.

    The implementation now lives in `langslice.atlas.core.get_root_mask` (it is
    an atlas accessor, and the linear transform tools need it too); the
    name here is an alias, so this exercises both."""
    from langslice.atlas import core as atlas_core
    from langslice.nonlinear import image_gen_helpers

    # Annotation slab: top half has tissue (non-zero IDs), bottom half is bg.
    annotation = np.array(
        [
            [
                [1, 2, 3, 4],
                [1, 2, 3, 4],
                [0, 0, 0, 0],
                [0, 0, 0, 0],
            ]
        ],
        dtype=np.int32,
    )
    atlas = SimpleNamespace(annotation=annotation)

    monkeypatch.setattr(
        atlas_core, "position_mm_to_index", lambda a, p, plane="coronal": 0
    )
    monkeypatch.setattr(atlas_core, "slice_axis_index", lambda ctx, plane: 0)
    monkeypatch.setattr(atlas_core, "atlas_space_context", lambda a: SimpleNamespace())
    monkeypatch.setattr(atlas_core, "orient_slice_for_display", lambda a, plane: a)

    target_size = (8, 8)  # (W, H) per PIL convention
    mask = image_gen_helpers._build_atlas_root_mask(
        atlas, position_mm=0.0, target_size=target_size, plane="coronal"
    )

    assert mask.shape == (8, 8)  # numpy (H, W)
    assert mask.dtype == np.uint8
    unique_vals = set(np.unique(mask).tolist())
    assert unique_vals.issubset({0, 255})
    assert 0 in unique_vals and 255 in unique_vals
    # Top half opaque (was non-zero), bottom half transparent (was zero).
    assert (mask[0] == 255).all()
    assert (mask[-1] == 0).all()


def test_elastix_report_emits_codes_only_for_implausible_warps():
    from langslice.nonlinear.image_gen_helpers import _elastix_report

    atlas = np.zeros((20, 20), dtype=np.int32)
    atlas[:10] = 1
    atlas[10:15] = 2
    atlas[15:] = 3
    structures = {2: {"acronym": "TH"}, 3: {"acronym": "CB"}}

    # Healthy: identical classification, identity deformation, tissue covered.
    identity = np.zeros((20, 20, 2), dtype=np.float64)
    healthy = _elastix_report(
        atlas_classified=atlas,
        warped_classified=atlas.copy(),
        structures=structures,
        deformation_field=identity,
        tissue_mask=atlas != 0,
    )
    assert healthy["codes"] == []

    # Region 2 vanished, region 3 collapsed to a sliver, and the field folds.
    warped = np.ones((20, 20), dtype=np.int32)
    warped[19, :10] = 3
    folding = np.zeros((20, 20, 2), dtype=np.float64)
    folding[..., 0] = -2.0 * np.arange(20)[np.newaxis, :]
    report = _elastix_report(
        atlas_classified=atlas,
        warped_classified=warped,
        structures=structures,
        deformation_field=folding,
        tissue_mask=None,
    )
    codes = {entry["code"] for entry in report["codes"]}
    assert codes == {"REGION_MISSING", "REGION_COLLAPSED", "WARP_FOLDS"}
    missing = next(e for e in report["codes"] if e["code"] == "REGION_MISSING")
    assert missing["region"] == "TH"

    # All-background warp leaves the real tissue uncovered.
    uncovered = _elastix_report(
        atlas_classified=atlas,
        warped_classified=np.zeros((20, 20), dtype=np.int32),
        structures=structures,
        tissue_mask=np.ones((20, 20), dtype=bool),
    )
    assert {e["code"] for e in uncovered["codes"]} == {"TISSUE_UNCOVERED"}


def test_canvas_pad_grows_working_canvas_and_reports_offsets():
    from langslice.nonlinear.image_gen_registration import prepare_canvas

    canvas, unpadded, ox, oy, pad_px = prepare_canvas(
        _make_slice(),  # 12x8
        canvas_pad=0.25, native_canvas=False,
    )

    # pad = round(0.25 * 12) = 3 px per side -> 18x14 (aspect within range,
    # so no further clamp).
    assert pad_px == 3
    assert unpadded == (12, 8)
    assert canvas.size == (18, 14)
    assert (ox, oy) == (3.0, 3.0)


def test_extreme_aspect_ratio_clamps_into_the_supported_range():
    """gpt-image-2 accepts 1:3..3:1; a 4:1 strip black-pads down to 3:1."""
    from langslice.nonlinear.image_gen_registration import prepare_canvas

    canvas, unpadded, ox, oy, _pad_px = prepare_canvas(
        _make_slice((48, 12)), native_canvas=False, image_model="gpt-image-2",
    )

    assert canvas.size == (48, 16)  # ceil(48 / 3)
    assert (ox, oy) == (0.0, 2.0)  # centered pad
    assert unpadded == (48, 12)


def test_in_range_aspect_ratio_is_left_untouched():
    from langslice.nonlinear.image_gen_registration import prepare_canvas

    canvas, unpadded, ox, oy, _pad_px = prepare_canvas(
        _make_slice((24, 9)), native_canvas=False, image_model="gpt-image-2",
    )

    assert canvas.size == (24, 9)
    assert unpadded == (24, 9)
    assert (ox, oy) == (0.0, 0.0)


def _fake_itk_parameter_module() -> SimpleNamespace:
    """Minimal itk stand-in: a ParameterObject that just collects its maps."""

    class FakeParameterObject:
        def __init__(self) -> None:
            self.maps: list[dict[str, Any]] = []

        def GetDefaultParameterMap(self, kind: str):  # noqa: N802 - itk API name
            return {"Transform": (kind,)}

        def AddParameterMap(self, param_map) -> None:  # noqa: N802, ANN001
            self.maps.append(param_map)

        def GetNumberOfParameterMaps(self) -> int:  # noqa: N802 - itk API name
            return len(self.maps)

    return SimpleNamespace(ParameterObject=SimpleNamespace(New=FakeParameterObject))


def test_multichannel_parameter_object_affine_only_emits_a_single_map(monkeypatch):
    """``deformation="affine"`` drops the B-spline stage from the shared
    multi-channel builder both border routes' affine path uses
    (:func:`~langslice.nonlinear.image_gen_helpers._register_channel_stacks`)."""
    import sys

    from langslice.nonlinear import image_gen_helpers

    monkeypatch.setitem(sys.modules, "itk", _fake_itk_parameter_module())

    bspline = image_gen_helpers._build_multichannel_parameter_object(32, 3, "bspline")
    assert bspline.GetNumberOfParameterMaps() == 2

    affine_only = image_gen_helpers._build_multichannel_parameter_object(32, 3, "affine")
    assert affine_only.GetNumberOfParameterMaps() == 1
    assert affine_only.maps[0]["Transform"] == ("affine",)


def test_register_cli_parses_passes_deformation_and_mirror():
    from langslice.cli import _build_parser

    args = _build_parser().parse_args(
        [
            "nonlinear", "register", "tests/fixture.png", "--position", "5.0",
            "--passes", "2", "--deformation", "affine", "--mirror-atlas-lr",
        ]
    )
    assert args.passes == 2
    assert args.deformation == "affine"
    assert args.mirror_atlas_lr is True

    defaults = _build_parser().parse_args(
        ["nonlinear", "register", "tests/fixture.png", "--position", "5.0"]
    )
    assert defaults.passes == 1
    assert defaults.deformation == "bspline"
    assert defaults.mirror_atlas_lr is False

    with pytest.raises(SystemExit):
        _build_parser().parse_args(
            [
                "nonlinear", "register", "tests/fixture.png", "--position", "5.0",
                "--passes", "3",
            ]
        )
