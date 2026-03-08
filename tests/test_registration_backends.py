"""Script-style checks for affine backend orchestration."""

import numpy as np
from PIL import Image

import langslice.registration.core as registration_core
from langslice.registration import AffineResult, estimate_affine_registration


def fallback(
    image: Image.Image,
    on_progress: object = None,
    atlas_name: str | None = None,
    position_mm: float | None = None,
    pixel_size_um: float | None = None,
) -> AffineResult:
    _ = on_progress, atlas_name, position_mm, pixel_size_um
    return AffineResult.from_legacy_params(
        image_width=image.width,
        image_height=image.height,
        rotation_deg=2.0,
        translate_x_pct=1.0,
        translate_y_pct=-1.5,
        backend="vlm_fallback",
        reasoning="fallback path",
    )


image = Image.new("L", (120, 90), color=0)
original_backend = registration_core.estimate_affine_with_ants

try:
    def fake_success(**kwargs: object) -> AffineResult:
        image_obj = kwargs["image"]
        return AffineResult(
            matrix=np.array(
                [[1.0, 0.0, 3.0], [0.0, 1.0, -2.0], [0.0, 0.0, 1.0]],
                dtype=np.float64,
            ),
            source_size=(image_obj.width, image_obj.height),
            output_size=(140, 100),
            backend="antspyx",
            reasoning="ants success",
        )

    registration_core.estimate_affine_with_ants = fake_success
    success_result = estimate_affine_registration(
        image=image,
        atlas_name="allen_mouse_25um",
        position_mm=1.2,
        fallback=fallback,
    )
    assert success_result.backend == "antspyx"
    assert success_result.output_size == (140, 100)

    def fake_failure(**kwargs: object) -> AffineResult:
        _ = kwargs
        raise RuntimeError("synthetic ants failure")

    registration_core.estimate_affine_with_ants = fake_failure
    failure_result = estimate_affine_registration(
        image=image,
        atlas_name="allen_mouse_25um",
        position_mm=1.2,
        fallback=fallback,
    )
    assert failure_result.backend == "vlm_fallback"
    assert "ANTsPyX backend failed: synthetic ants failure" in failure_result.reasoning
finally:
    registration_core.estimate_affine_with_ants = original_backend

print("Registration backend orchestration OK")
