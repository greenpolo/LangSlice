"""Deterministic checks on the registration pipeline's classification,
despeckle, generation-report, and orientation helpers."""

from __future__ import annotations


def test_despeckle_reassigns_tiny_components() -> None:
    import numpy as np

    from langslice.nonlinear.image_gen_helpers import _despeckle_classified

    field = np.full((20, 20), 5, dtype=np.int32)
    field[10:, :] = 7
    field[3, 3] = 9  # 1px speckle inside region 5
    field[15, 15] = 0  # 1px background hole inside region 7
    out = _despeckle_classified(field, min_px=4)
    assert out[3, 3] == 5
    assert out[15, 15] == 7
    assert (out[:10, :] == 5).all() and (out[10:, :] == 7).all()


def _gen_report(monkeypatch, gen, ref, **kwargs):
    import langslice.nonlinear.image_gen_helpers as helpers

    monkeypatch.setattr(
        helpers, "_family_mapping", lambda uids, atlas, eps=40.0: {int(u): int(u) for u in uids}
    )
    return helpers.generation_report(gen, ref, atlas=None, **kwargs)


def _warp(field, amplitude=6.0, scale=1.08, shift=4):
    """A smooth, substantial deformation of a label map — the legitimate kind."""
    import cv2
    import numpy as np

    h, w = field.shape
    ys, xs = np.mgrid[0:h, 0:w].astype(np.float32)
    cx, cy = w / 2, h / 2
    map_x = (xs - cx) / scale + cx - shift + amplitude * np.sin(2 * np.pi * ys / h)
    map_y = (ys - cy) / scale + cy + shift + amplitude * np.cos(2 * np.pi * xs / w)
    return cv2.remap(
        field.astype(np.float32), map_x, map_y,
        interpolation=cv2.INTER_NEAREST, borderMode=cv2.BORDER_CONSTANT, borderValue=0,
    ).astype(field.dtype)


def _layered_ref():
    """A brain-ish map: an outer region, an inner one, and a thin interior tract."""
    import numpy as np

    ref = np.zeros((200, 200), dtype=np.int32)
    ref[20:180, 10:100] = 1
    ref[20:180, 100:190] = 2
    ref[50:130, 60:140] = 3  # a nucleus straddling both
    ref[150:158, 40:160] = 4  # a thin tract below it
    return ref


def test_generation_report_flags_ectopic_region(monkeypatch) -> None:
    """A region painted where the atlas has a different one entirely."""
    ref = _layered_ref()
    gen = ref.copy()
    gen[25:55, 150:185] = 4  # tract material way off in region 2's corner
    report = _gen_report(monkeypatch, gen, ref)
    ectopic = [f for f in report["flags"] if f["code"] == "ECTOPIC_REGION"]
    assert ectopic, report["flags"]
    assert ectopic[0]["region"] == "4"
    assert ectopic[0]["atlas_has_there"] == "2"


def test_generation_report_flags_inverted_interior(monkeypatch) -> None:
    """The cerebellum failure in miniature: an outside region repainted over
    a structure's interior, and the structure's own colour left elsewhere."""
    ref = _layered_ref()
    gen = ref.copy()
    gen[60:120, 70:130] = 4  # tract grey painted across the nucleus core
    report = _gen_report(monkeypatch, gen, ref)
    hits = [
        f for f in report["flags"]
        if f["code"] == "ECTOPIC_REGION" and f["region"] == "4"
    ]
    assert hits, report["flags"]
    assert hits[0]["atlas_has_there"] == "3"
    assert hits[0]["covers_of_that_region"] > 0.2


def test_generation_report_flags_invented_subdivision(monkeypatch) -> None:
    """Several unrelated parcels invented inside one atlas region."""
    ref = _layered_ref()
    gen = ref.copy()
    gen[30:60, 15:45] = 2  # parcels from far-away regions, dropped into 1
    gen[120:150, 15:45] = 3
    report = _gen_report(monkeypatch, gen, ref)
    invented = [f for f in report["flags"] if f["code"] == "INVENTED_SUBDIVISION"]
    assert invented, report["flags"]
    assert invented[0]["region"] == "1"
    assert {p["region"] for p in invented[0]["parcels"]} == {"2", "3"}


def test_generation_report_ignores_legitimate_deformation(monkeypatch) -> None:
    """Regions shifted, scaled, wavily warped and grown must not flag."""
    ref = _layered_ref()
    gen = _warp(ref)
    grow = (gen == 0) & (_warp(ref, amplitude=6.0, scale=1.14, shift=4) == 4)
    gen[grow] = 4  # and the tract grown well past its atlas width
    report = _gen_report(monkeypatch, gen, ref)
    assert report["flags"] == [], report["flags"]


def test_generation_report_ignores_dust_specks(monkeypatch) -> None:
    """Slide dust classified into stray regions is far too small to judge."""
    import numpy as np

    rng = np.random.default_rng(0)
    ref = _layered_ref()
    gen = ref.copy()
    for _ in range(60):
        y, x = rng.integers(20, 175), rng.integers(15, 185)
        gen[y : y + 2, x : x + 2] = int(rng.integers(1, 5))
    report = _gen_report(monkeypatch, gen, ref)
    assert report["flags"] == [], report["flags"]



def test_orient_slice_to_axes_maps_horizontal_to_user_frame() -> None:
    from types import SimpleNamespace

    import numpy as np

    from langslice.space import atlas_space_context, native_slice_axes, orient_slice_to_axes

    atlas = SimpleNamespace(
        atlas_name="fake",
        orientation="asr",
        template=np.zeros((4, 5, 6)),
        resolution=(25.0, 25.0, 25.0),
    )
    ctx = atlas_space_context(atlas)
    assert native_slice_axes(ctx, "horizontal") == ("rl", "ap")
    assert native_slice_axes(ctx, "sagittal") == ("si", "ap")

    # horizontal render: rows right->left, cols anterior->posterior
    arr = np.arange(6 * 4).reshape(6, 4)  # (rl rows=6, ap cols=4)
    out = orient_slice_to_axes(arr, ctx, "horizontal", "ap,lr")
    assert out.shape == (4, 6)
    # anterior row of the output = first ap index; left col first.
    # native [rl, ap]; transpose -> [ap, rl]; rows ap ok; cols rl -> flip for lr
    assert out[0, 0] == arr[-1, 0]

    # no-op when requesting the native frame
    same = orient_slice_to_axes(arr, ctx, "horizontal", "rl,ap")
    assert (same == arr).all()

    import pytest as _pytest

    with _pytest.raises(ValueError):
        orient_slice_to_axes(arr, ctx, "horizontal", "si,ap")  # si not in-plane
