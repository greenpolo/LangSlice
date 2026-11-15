"""The in-tree Brainways tool (``langslice.engines.brainways``): its declaration,
where it is offered, its options and refusals, the numpy-1 arithmetic of its
inputs, and (``slow``) the real network on atlas planes.

The network needs torch, a core dependency; the tests that run it skip where
torch does not import, and where Brainways' weights or the rat atlas cannot
be had. Its parity with Brainways 0.1.16.3 on real rat stacks is in the
engine's README.
"""

from __future__ import annotations

import inspect
from pathlib import Path

import numpy as np
import pytest
from PIL import Image

from langslice.engines.brainways import images, network, plugin
from langslice.job import plugins as core_plugins
from langslice.job.settings.spec import JobSpec
from langslice.plugin_api import ToolError
from langslice.tools.catalog import enabled

REF = "langslice.engines.brainways.plugin:tools"
RAT = "whs_sd_rat_39um"


@pytest.fixture(autouse=True)
def clean_registry():
    held = dict(core_plugins._LOADED)
    yield
    core_plugins._LOADED.clear()
    core_plugins._LOADED.update(held)


# --- the declaration and where it is offered --------------------------------


def test_the_tool_keeps_its_name_kind_and_arguments():
    (spec,) = plugin.tools()
    assert (spec.name, spec.kind, spec.group, spec.task) == (
        "brainways_ap", "write", "Positioning", "position")
    assert spec.writes == ("position",)
    assert (spec.atlases, spec.planes) == ((RAT,), ("coronal",))
    stub = inspect.signature(spec.stub).parameters
    assert list(stub) == ["sections", "channel", "write", "view"]
    assert [stub[name].default for name in stub] == [[], "", None, False]
    handler = inspect.signature(spec.handler).parameters
    assert set(stub) <= set(handler)


@pytest.mark.parametrize("atlas, plane, offered", [
    (RAT, "coronal", True),
    (RAT, "sagittal", False),
    ("allen_mouse_25um", "coronal", False),
])
def test_the_tool_is_offered_on_rat_coronal_jobs_only(tmp_path: Path, atlas: str,
                                                      plane: str, offered: bool):
    spec = JobSpec(image_folder=str(tmp_path), plugins=[REF], atlas=atlas, plane=plane)
    assert ("brainways_ap" in enabled(spec)) is offered


def test_options_are_checked_when_the_plugin_loads():
    (spec,) = plugin.tools({"device": "cpu", "long_edge": 1024, "cache_folder": "x"})
    assert spec.name == "brainways_ap"
    with pytest.raises(ValueError, match="takes the options"):
        plugin.tools({"python": "/usr/bin/python"})
    with pytest.raises(ValueError, match="device is one of"):
        plugin.tools({"device": "gpu"})


def test_a_missing_torch_or_missing_weights_is_engine_missing(monkeypatch):
    def no_torch(_option):
        raise ImportError("No module named 'torch'")

    monkeypatch.setattr(network, "resolve_device", no_torch)
    with pytest.raises(ToolError) as refused:
        plugin._network(plugin.Options())
    assert refused.value.code == "ENGINE_MISSING"
    assert 'pip install "langslice[engines]"' in str(refused.value)

    def offline(_device, *, cache_folder=""):
        raise OSError("no route to huggingface.co")

    monkeypatch.setattr(network, "resolve_device", lambda _option: network.Device.CPU)
    monkeypatch.setattr(network, "load_network", offline)
    with pytest.raises(ToolError) as refused:
        plugin._network(plugin.Options())
    assert refused.value.code == "ENGINE_MISSING"
    assert "brainways/siamese" in str(refused.value)


def test_the_package_imports_without_torch():
    import subprocess
    import sys

    code = (
        "import sys\n"
        "import langslice.engines.brainways.plugin\n"
        "print(sorted(m for m in sys.modules if m.split('.')[0] == 'torch'))\n"
    )
    done = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True,
                          check=False)
    assert done.returncode == 0, done.stderr
    assert done.stdout.strip() == "[]"


# --- the inputs: Brainways' arithmetic under numpy 1 ---------------------------


def test_the_percentiles_are_numpy_1s():
    # np.percentile(x, float32 (0.1, 99.8)) under numpy 1.26, as Brainways
    # pins it; numpy 2 answers 1.1098171270873536e-09 / 0.9944179058074951
    # and 7.0 / 250.87600708007812
    rng = np.random.default_rng(7)
    rng.integers(0, 256, (224, 224), dtype=np.uint8)
    plane = rng.random((224, 224), dtype=np.float32) ** 3
    assert images._percentiles(plane.astype(np.float64), float32=True) == (
        1.1098170861366987e-09, 0.9944179123267531)
    small = np.random.default_rng(11).integers(0, 256, (7, 9), dtype=np.uint8)
    assert images._percentiles(small.astype(np.float64), float32=False) == (
        7.0, 250.87600326538086)


def test_the_thumbnail_and_the_inputs_have_brainways_shapes():
    rng = np.random.default_rng(0)
    channel = rng.integers(0, 256, (1720, 2048), dtype=np.uint8)
    thumbnail = images.section_thumbnail(channel)
    assert thumbnail.shape == (860, 1024) and thumbnail.dtype == np.uint8
    assert thumbnail.min() == 0 and thumbnail.max() == 255
    section = images.section_input(thumbnail)
    assert section.shape == (3, 224, 224) and section.dtype == np.float32
    assert np.array_equal(section[0] * 0.229, section[0] * 0.229)  # finite
    plane = images.plane_input(rng.random((512, 512), dtype=np.float32))
    assert plane.shape == (3, 224, 224) and plane.dtype == np.float32
    assert np.isfinite(plane).all()


def test_the_network_reads_the_checkpoints_tensors():
    names = network.tensor_names()
    # the checkpoint holds 778 tensors, 109 of them num_batches_tracked
    assert len(names) == len(set(names)) == 778 - 109
    assert names[0] == "_backbone._model.conv_stem.weight"
    assert names[-1] == "classifier.2.bias"


# --- the real network (slow) ---------------------------------------------------


@pytest.fixture(scope="module")
def rat_atlas():
    import os

    pytest.importorskip("torch")
    os.environ.setdefault("LANGSLICE_ATLAS_SKIP_LATEST_CHECK", "1")
    from langslice.core.atlas.load import load_atlas

    try:
        return load_atlas(RAT)
    except ValueError as error:
        pytest.skip(f"the rat atlas is not here: {error}")


@pytest.fixture(scope="module")
def cpu_network():
    pytest.importorskip("torch")
    try:
        return network.load_network(network.Device.CPU)
    except (OSError, ValueError) as error:
        pytest.skip(f"Brainways' weights cannot be had: {error}")


def _plane_image(atlas, ap: int) -> np.ndarray:
    template = np.asarray(atlas.template)
    return (template[ap].astype(np.float64) / float(template.max()) * 255).astype(np.uint8)


@pytest.mark.slow
def test_an_atlas_plane_is_placed_on_itself(rat_atlas, cpu_network):
    volume = network.ReferenceVolume.from_voxels(np.asarray(rat_atlas.template))
    planes = network.AtlasEmbeddings(cpu_network, volume)
    for ap in (250, 450, 650):
        thumbnail = images.section_thumbnail(_plane_image(rat_atlas, ap))
        found = network.find_ap(cpu_network, thumbnail, planes, network.AP_LIMITS[RAT])
        assert abs(found - ap) <= 3, (ap, found)


@pytest.mark.slow
def test_the_tool_proposes_and_writes_positions(tmp_path: Path, rat_atlas, cpu_network):
    from langslice.doors.agent.engine import build_context
    from langslice.doors.tools.toolbox import build_tools
    from langslice.job.job import ingest
    from tests.linear_tool_helpers import tool_named

    images_folder = tmp_path / "images"
    images_folder.mkdir()
    for name, ap in (("a.png", 300), ("b.png", 600)):
        plane = _plane_image(rat_atlas, ap)
        Image.fromarray(np.stack([plane] * 3, axis=-1)).save(images_folder / name)
    spec = JobSpec(image_folder=str(images_folder), model="fake-model", preprocess="none",
                   atlas=RAT, plane="coronal", tasks=["position"], plugins=[REF],
                   plugin_config={REF: {"device": "cpu"}})
    context = build_context(spec, emit=lambda _m: None, atlas_loader=lambda _n: rat_atlas)
    state = ingest(spec, context)
    tool = tool_named(build_tools(state, context, spec), "brainways_ap")

    proposed = tool(write=[])
    assert proposed["written"] == "nothing: proposal only"
    assert proposed["columns"] == ["section", "position_mm", "change_mm"]
    assert proposed["channel"] == "gray"  # three equal planes read as one
    by_name = {row[0]: row[1] for row in proposed["rows"]}
    assert abs(by_name["a.png"] - 300 * 0.039) <= 3 * 0.039
    assert abs(by_name["b.png"] - 600 * 0.039) <= 3 * 0.039
    assert [row[0] for row in proposed["rows"]] == ["a.png", "b.png"]

    written = tool()
    assert written["written"] == "positions"
    assert [row[3] for row in written["rows"]] == ["position", "position"]
    assert [row[2] for row in written["rows"]] == [None, None]  # no position before
    again = tool(write=[])
    assert [row[2] for row in again["rows"]] == [0.0, 0.0]
    calls = sorted((Path(spec.image_folder) / "langslice" / "plugins" / "brainways_ap"
                    / "calls").iterdir())
    assert [call.name for call in calls] == ["0001", "0002", "0003"]
    for name in ("settings.json", "rows.json", "images/a.png"):
        assert (calls[0] / name).is_file()
