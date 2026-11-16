"""STalign and emlddmm in LangSlice's process: real small fits, their records, the GPU queue.

Both are vendored (``langslice_vendor.stalign``, ``langslice_vendor.emlddmm``) and always
run. Every fit here runs on the CPU at float64 on synthetic images, so each takes a few
seconds.
"""

from __future__ import annotations

import subprocess
import sys
import threading
import time
from pathlib import Path

import numpy as np
import pytest

from langslice.engines import _atlas_images
from langslice.engines.emlddmm import engine as emlddmm_engine
from langslice.engines.stalign import engine as stalign_engine


def _blob(shape: tuple[int, ...], centre: tuple[float, ...], width: float) -> np.ndarray:
    grids = np.meshgrid(*[np.arange(n, dtype=np.float64) for n in shape], indexing="ij")
    return np.exp(-sum((g - c) ** 2 for g, c in zip(grids, centre, strict=True))
                  / (2 * width ** 2))


# --- STalign ------------------------------------------------------------------------------------


def _stalign_inputs(shift_px: float) -> dict[str, np.ndarray]:
    """A two-channel source and a target, the target's blob moved by *shift_px* (20 um px)."""
    image = _blob((40, 48), (20.0, 24.0), 6.0) + 0.5 * _blob((40, 48), (12.0, 30.0), 3.0)
    moved = _blob((40, 48), (20.0 + shift_px, 24.0), 6.0) + 0.5 * _blob(
        (40, 48), (12.0 + shift_px, 30.0), 3.0)
    axis0, axis1 = np.arange(40) * 20.0, np.arange(48) * 20.0
    rows, cols = np.meshgrid(axis0, axis1, indexing="ij")
    return {"mode": np.array("lddmm"),
            "I": np.stack([image, (image - image.mean()) ** 2]), "xI0": axis0, "xI1": axis1,
            "J": moved[None], "xJ0": axis0, "xJ1": axis1, "A0": np.eye(3),
            "query": np.stack([rows.ravel(), cols.ravel()], -1)}


PARAMS = {"a": 200.0, "sigma_disp": 94.0, "niter": 30, "nt": 4, "epL": 0.0, "epT": 0.0,
          "device": "cpu", "threads": 2}


def test_a_small_stalign_fit_runs_in_process_and_keeps_its_energy_record():
    out = stalign_engine.run(_stalign_inputs(2.0), PARAMS)
    assert out["device"] == "cpu float64"
    source = np.asarray(out["source_um"])
    assert source.shape == (40 * 48, 2) and np.isfinite(source).all()
    # the target's blob sits 40 um further down: its centre maps back about 40 um up
    centre = 20 * 48 + 24
    assert -60.0 < source[centre, 0] - 20 * 20.0 < -20.0
    energy = np.asarray(out["energy"])
    assert energy.shape == (30, 3) and energy[-1, 1] < energy[0, 1]
    assert np.asarray(out["WM"]).shape == (40, 48)
    assert '"niter": 30' in str(out["settings"])


def test_stalign_fits_on_two_threads_keep_their_own_records():
    results: dict[str, dict] = {}

    def fit(name: str, shift: float) -> None:
        results[name] = stalign_engine.run(_stalign_inputs(shift), PARAMS)

    threads = [threading.Thread(target=fit, args=(name, shift))
               for name, shift in (("near", 1.0), ("far", 3.0))]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join()
    alone = stalign_engine.run(_stalign_inputs(3.0), PARAMS)
    assert np.array_equal(results["far"]["energy"], alone["energy"])
    assert np.array_equal(results["far"]["source_um"], alone["source_um"])
    assert not np.array_equal(results["near"]["energy"], results["far"]["energy"])


def test_stalign_s_own_figures_still_draw_for_another_caller(monkeypatch: pytest.MonkeyPatch):
    from langslice_vendor import stalign as STalign  # noqa: N812 - STalign's own name

    stalign_engine.run(_stalign_inputs(1.0), PARAMS)  # installs the record
    calls: list[str] = []

    class Figures:
        def __init__(self, _problem: object) -> None:
            calls.append("made")

        def update(self, _snap: object) -> None:
            calls.append("drawn")

    monkeypatch.setattr(stalign_engine._EnergyRecord, "figures", Figures)
    inputs = _stalign_inputs(1.0)
    STalign.LDDMM([inputs["xI0"], inputs["xI1"]], inputs["I"], [inputs["xJ0"], inputs["xJ1"]],
                  inputs["J"], niter=2, draw="end", device="cpu", prefit=False)
    assert calls == ["made", "drawn"]


# --- emlddmm ------------------------------------------------------------------------------------


def test_the_vendored_stalign_runs_its_torch_engine():
    from langslice_vendor import stalign

    info = stalign.backend_info()
    assert (info["auto"], info["cpp"]) == ("torch", False)


def test_the_vendored_emlddmm_imports_without_ipython_and_prints_nothing():
    code = ("import sys\nsys.modules['IPython'] = None\n"
            "import langslice_vendor.emlddmm.emlddmm as e\nprint(callable(e.emlddmm_multiscale))\n")
    done = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True,
                          stdin=subprocess.DEVNULL, check=False, timeout=300)
    assert done.returncode == 0, done.stderr
    assert done.stdout.split() == ["True"]


def _synthetic_atlas() -> _atlas_images.AveragedImage:
    image = (_blob((24, 20, 22), (12.0, 10.0, 11.0), 4.0)
             + 0.5 * _blob((24, 20, 22), (8.0, 6.0, 14.0), 2.0)).astype(np.float32)
    resolution = np.array([50.0, 50.0, 50.0])
    axes = [np.arange(n) * 50.0 for n in image.shape]
    centre = (np.asarray(image.shape, float) - 1) / 2.0 * resolution
    return _atlas_images.AveragedImage(image, axes, centre, resolution)


def _emlddmm_inputs(atlas: _atlas_images.AveragedImage) -> dict[str, np.ndarray]:
    """Three slots 100 um apart through the synthetic atlas, its rows and columns, 50 um px."""
    zs = np.array([0.0, 100.0, 200.0])
    gy, gx = np.arange(20) * 50.0 - 475.0, np.arange(22) * 50.0 - 525.0
    target = np.stack([atlas.image[10 + 2 * k] for k in range(3)]).astype(np.float64)
    c0 = atlas.centre + np.array([-100.0, 0.0, 0.0])
    q = np.array([[0.0, 0.0], [100.0, -50.0]])
    return {"J": target / target.max(), "W0": np.ones_like(target), "zs": zs, "gy": gy,
            "gx": gx, "R": np.eye(3), "c0": c0, "slot": np.array([0, 1, 2]),
            "q": np.concatenate([q, q, q]), "q_n": np.array([2, 2, 2])}


def _emlddmm_params(mode: str) -> dict[str, object]:
    cfg = {"downI": [[2, 2, 2], [1, 1, 1]], "downJ": [[1, 2, 2], [1, 1, 1]],
           "n_iter": [3, 2], "a": [200.0], "dv": [100.0],
           "v_start": [0 if mode == "stack" else 10 ** 6], "ev": [1e-2], "eA": [1e7],
           "eA2d": [1e5], "local_contrast": [[1, 4, 4]], "sigmaR": [1e4]}
    return {"mode": mode, "atlas": "synthetic", "image": "template", "cfg": cfg,
            "sorting": {}, "warp": mode == "stack", "mu": {"muB": [[0.0]]},
            "up_vector": [0.0, -1.0, 0.0], "device": "cpu", "dtype": "float64"}


@pytest.mark.parametrize("mode", ["stack", "plane"])
def test_a_small_emlddmm_fit_runs_in_process_and_keeps_its_records(
        mode: str, monkeypatch: pytest.MonkeyPatch):
    atlas = _synthetic_atlas()
    monkeypatch.setattr(emlddmm_engine, "averaged_image", lambda _name, _image: atlas)
    out = emlddmm_engine.run(_emlddmm_inputs(atlas), _emlddmm_params(mode))
    assert out["device"] == "cpu float64"
    assert int(np.asarray(out["energy_captured"])) == 1
    assert np.asarray(out["energy"]).shape == (5, 5)  # 3 + 2 iterations
    assert "Found 2 scales" in str(out["log"])
    assert np.asarray(out["X"]).shape == (6, 3) and np.isfinite(out["X"]).all()
    assert np.asarray(out["Acut"]).shape == (3, 20, 22)
    if mode == "stack":  # the slots' planes stay where they were
        assert np.allclose(np.asarray(out["A"])[0], np.asarray(out["A0"])[0])


def test_emlddmm_s_prints_on_a_thread_with_no_fit_are_logged(caplog: pytest.LogCaptureFixture):
    module = emlddmm_engine._emlddmm()  # pyright: ignore[reportPrivateUsage]
    with caplog.at_level("INFO", logger=emlddmm_engine.__name__):
        module.print("Found", 3, "scales")
    assert "emlddmm: Found 3 scales" in caplog.text


# --- the atlas image ----------------------------------------------------------------------------


def test_the_averaged_image_is_kept_and_read_back(tmp_path: Path,
                                                  monkeypatch: pytest.MonkeyPatch):
    from types import SimpleNamespace

    template = np.arange(8 * 6 * 4, dtype=np.float32).reshape(8, 6, 4)
    fake = SimpleNamespace(template=template, annotation=np.ones_like(template, np.int32),
                           additional_references={}, resolution=(25.0, 25.0, 25.0),
                           metadata={"version": "9.9"})
    monkeypatch.setattr(_atlas_images, "CACHE", tmp_path)
    monkeypatch.setattr(_atlas_images, "_atlas", lambda _name: fake)
    made = _atlas_images.averaged_image("fake", "template")
    assert made.image.shape == (4, 3, 2)
    assert made.image[0, 0, 0] == template[:2, :2, :2].mean()
    assert np.allclose(made.axes[0], [12.5, 62.5, 112.5, 162.5])
    assert list(tmp_path.glob("*.npz")) == [tmp_path / "fake_v9.9__template__50um.npz"]
    again = _atlas_images.averaged_image("fake", "template")
    assert np.array_equal(again.image, made.image) and np.array_equal(again.centre, made.centre)


# --- the GPU queue ------------------------------------------------------------------------------


def test_two_threads_take_the_gpu_queue_in_turn_with_one_slot(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    from langslice.core.fits.deformable import gpu_queue
    from langslice.core.hardware import GPU_SLOTS_VARIABLE
    from langslice.plugin_api import gpu_turn

    monkeypatch.setattr(gpu_queue, "GPU_LOCK", tmp_path / "gpu.lock")
    monkeypatch.setenv(GPU_SLOTS_VARIABLE, "1")
    held = threading.Event()

    def hold() -> None:
        with gpu_turn("cuda:0"):
            held.set()
            time.sleep(0.5)

    other = threading.Thread(target=hold)
    other.start()
    assert held.wait(10)
    with gpu_turn("cuda:1") as turn:
        assert turn.waited_s >= 0.2
    other.join()
    with gpu_turn("cpu") as turn:
        assert turn.waited_s == 0.0
