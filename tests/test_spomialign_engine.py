"""SPOmiAlign in LangSlice's process (``langslice.engines.spomialign``).

The fitting steps (``engine.py``) and the edge map and suppression of the matching
(``matching.py``) run on numpy, OpenCV, SimpleITK and torch, LangSlice's own dependencies, so
they are tested everywhere. RoMa itself (``romatch``, the ``engines`` extra) is needed only by
the real fit at the end, marked ``slow``, which skips without it; without it a call is
refused ``ENGINE_MISSING`` naming the extra.
"""

from __future__ import annotations

import importlib.util

import numpy as np
import pytest

from langslice.engines.spomialign import engine, matching, plugin
from langslice.plugin_api import ToolError


def _rotation(deg: float, shift: tuple[float, float]) -> np.ndarray:
    t = np.deg2rad(deg)
    return np.array([[np.cos(t), -np.sin(t), shift[0]], [np.sin(t), np.cos(t), shift[1]]])


def test_a_call_without_the_extra_is_refused_naming_it(monkeypatch):
    monkeypatch.setattr(matching, "is_installed", lambda: False)
    handler = plugin.tools()[0].handler
    with pytest.raises(ToolError) as refused:
        handler(None, sections=["s0"], fit="transform")  # type: ignore[arg-type]
    assert refused.value.code == "ENGINE_MISSING"
    assert 'pip install "langslice[engines]"' in str(refused.value)


def test_the_plugin_takes_only_the_device_option():
    assert plugin.DEFAULTS == {"device": ""}


def test_the_rigid_fit_recovers_a_rotation_and_a_shift():
    rng = np.random.default_rng(1)
    section = rng.uniform(0, 1000, (200, 2))
    true = _rotation(7.0, (12.0, -5.0))
    atlas = section @ true[:, :2].T + true[:, 2]
    fitted = engine.rigid_fit(atlas, section)
    assert np.allclose(fitted, true, atol=1e-3)
    assert np.allclose(engine.linear_fit(atlas, section, "rigid"), fitted)


def test_the_affine_and_the_warp_s_linear_parts():
    rng = np.random.default_rng(2)
    section = rng.uniform(0, 1000, (200, 2))
    true = np.array([[1.1, 0.05, 3.0], [-0.02, 0.9, -7.0]])
    atlas = section @ true[:, :2].T + true[:, 2]
    assert np.allclose(engine.linear_fit(atlas, section, "affine"), true)
    assert np.allclose(engine.linear_fit(atlas, section, "warp"), np.eye(3)[:2])


def test_the_field_carries_one_shared_displacement_everywhere():
    rng = np.random.default_rng(3)
    fixed = np.c_[rng.uniform(50, matching.W - 50, 400), rng.uniform(50, matching.H - 50, 400)]
    moving = fixed + [4.0, -2.5]
    field = engine.smoothed_field(fixed, moving, 15.0)
    assert field.shape == (matching.H, matching.W, 2)
    # TELEA inpainting between sparse pairs overshoots a little, as in SPOmiAlign
    assert np.abs(field[..., 0] - 4.0).max() < 0.6
    assert np.abs(field[..., 1] + 2.5).max() < 0.6
    assert not engine.smoothed_field(np.empty((0, 2)), np.empty((0, 2)), 15.0).any()


def test_the_field_reads_on_the_canvas_as_simpleitk_reads_it():
    h, w = 300, 400
    to_grid = engine.CORNER @ engine.resize_map((w, h), (matching.W, matching.H))
    from_grid = np.linalg.inv(to_grid)
    displacement = np.zeros((matching.H, matching.W, 2))
    displacement[..., 0] = np.linspace(0, 6, matching.W)[None, :]
    matrix = (from_grid @ to_grid)[:2]
    residual, sitk_max_px = engine.field_on_canvas(displacement, to_grid, from_grid, matrix,
                                                   (w, h))
    assert residual.shape == (h, w, 2)
    assert sitk_max_px < 1e-6
    # the grid is 2.88 canvas px per canvas px across: 6 grid px are about 2.1 canvas px
    assert 1.9 < residual[h // 2, -1, 0] < 2.2 and abs(residual[h // 2, 0, 0]) < 0.1


def test_the_pairs_in_the_damage_zone_or_off_the_mask_are_dropped_and_refilled():
    h, w = 40, 60
    keep = np.ones((h, w), bool)
    keep[:, :10] = False  # damage on the atlas's left
    section_keep = np.ones((h, w), bool)
    section_keep[:, 50:] = False  # the mask leaves out the section's right
    pair = engine.EnginePair(target=np.zeros((h, w, 3), np.uint8),
                             source=np.zeros((h, w, 3), np.uint8), keep=keep,
                             smoothing_px=15.0, section_keep=section_keep)
    canvas = np.array([[5.0, 20.0], [55.0, 20.0], [30.0, 20.0], [31.0, 21.0], [32.0, 22.0]])
    from_grid = np.eye(3)
    chosen = engine.choose_pairs(pair, canvas, canvas, np.linspace(1, 0.5, 5), from_grid,
                                 top_k=2)
    assert chosen.n_pairs == 2 and list(chosen.fitted) == [2, 3]
    assert len(chosen.passed) == 2 and chosen.by_mask == 1
    answer = chosen.answer(error="x")
    assert answer.pairs == 2 and answer.dropped == 2 and answer.dropped_by_mask == 1


def test_the_suppression_keeps_each_window_s_peak():
    torch = pytest.importorskip("torch")
    certainty = torch.zeros((9, 9))
    certainty[4, 4], certainty[4, 5], certainty[0, 0] = 0.9, 0.5, 0.3
    kept = matching.grid_suppression(certainty)
    assert kept[4, 4] == pytest.approx(0.9) and kept[4, 5] == 0 and kept[0, 0] == pytest.approx(0.3)


def test_the_edge_map_is_binary_on_the_grid():
    pytest.importorskip("torch")
    from PIL import Image

    image = np.zeros((300, 400, 3), np.uint8)
    image[100:200, 150:250] = 255
    edges = matching.edge_weight(Image.fromarray(image), matching.resolve_device("cpu"))
    values = set(np.unique(edges.numpy()).tolist())
    assert edges.shape == (matching.H, matching.W) and values == {0.0, 1.0}


@pytest.mark.slow
@pytest.mark.skipif(importlib.util.find_spec("romatch") is None,
                    reason="RoMa (the engines extra) is not installed")
def test_a_real_rigid_fit_recovers_a_known_shift():
    """RoMa on a textured canvas and the same canvas shifted: the rigid fit finds the shift.

    The first run downloads RoMa's weights into torch's hub cache.
    """
    import cv2

    rng = np.random.default_rng(4)
    h, w = 768, 1024
    noise = cv2.GaussianBlur(rng.uniform(0, 255, (h, w)).astype(np.float32), (0, 0), 6)
    texture = cv2.normalize(noise, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
    target = np.repeat(texture[..., None], 3, axis=2)
    shift = np.float32([[1, 0, 18], [0, 1, -11]])
    source = cv2.warpAffine(target, shift, (w, h), borderMode=cv2.BORDER_REFLECT)
    pair = engine.EnginePair(target=target, source=source, keep=np.ones((h, w), bool),
                             smoothing_px=15.0)
    run = engine.fit_pairs([pair], mode="rigid", top_k=5000)
    answer = run.answers[0]
    assert not answer.error and answer.pairs == 5000 and answer.matrix is not None
    # source px -> target px undoes the shift
    assert np.allclose(answer.matrix[:, :2], np.eye(2), atol=0.01)
    assert np.allclose(answer.matrix[:, 2], [-18, 11], atol=1.5)
