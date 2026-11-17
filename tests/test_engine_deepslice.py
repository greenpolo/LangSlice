"""The DeepSlice tool (``langslice.engines.deepslice``), in LangSlice's process.

Most of these need no TensorFlow: where the tool is offered, its declaration,
the refusals before DeepSlice runs (the extra missing, bad arguments, an
unknown option), the weights' fetch and checksum, the vendored angle vote
and spacing, and the reply's words. The last test runs the real model (the
``langslice[engines]`` extra) on a few small images, twice in one process;
it is ``slow`` and skips without the extra.
"""

from __future__ import annotations

import hashlib
import importlib.util
import inspect
import io
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest
from PIL import Image

from langslice.doors.tools.toolbox import build_tools
from langslice.engines.deepslice import alignment, engine, network, reply
from langslice.engines.deepslice.proposal import (
    Placement,
    ProposalRow,
    voted_angles,
)
from langslice.job import plugins as core_plugins
from langslice.job.settings.spec import JobSpec
from langslice.plugin_api import ToolError
from langslice.tools.catalog import enabled
from langslice.tools.descriptions.description import DescriptionFacts
from langslice.tools.descriptions.render import declaration
from tests.linear_tool_helpers import ToolContext, stack, tool_named

REF = "langslice.engines.deepslice.plugin:tools"
HAS_EXTRA = all(
    importlib.util.find_spec(name) for name in engine.REQUIRED_MODULES
)


@pytest.fixture(autouse=True)
def clean_registry():
    held = dict(core_plugins._LOADED)
    yield
    core_plugins._LOADED.clear()
    core_plugins._LOADED.update(held)


def _call(tmp_path: Path, n: int = 3, **arguments):
    """Call ``deepslice_place`` on a small placed stack; return the reply."""
    images = tmp_path / "images"
    images.mkdir(exist_ok=True)
    state, ctx, spec = stack(
        images, n, placed=True, plugins=[REF], tasks=["position", "transform"]
    )
    tool = tool_named(build_tools(state, ctx, spec), "deepslice_place")
    if "tool_context" in inspect.signature(tool).parameters:
        arguments["tool_context"] = ToolContext()
    return tool(**arguments)



@pytest.mark.parametrize(
    ("atlas", "plane", "offered"),
    [
        ("allen_mouse_25um", "coronal", True),
        ("allen_mouse_10um", "coronal", True),
        ("ccfv3augmented_mouse_25um", "coronal", True),
        ("allen_mouse_25um", "sagittal", False),
        ("allen_mouse_25um", "horizontal", False),
        ("whs_sd_rat_39um", "coronal", False),
    ],
)
def test_the_tool_is_offered_on_mouse_coronal_jobs_only(
    tmp_path: Path, atlas: str, plane: str, offered: bool
):
    spec = JobSpec(
        image_folder=str(tmp_path), plugins=[REF], atlas=atlas, plane=plane
    )
    assert ("deepslice_place" in enabled(spec)) is offered


def test_the_declaration_keeps_its_arguments_defaults_and_words(
    tmp_path: Path,
):
    spec = JobSpec(
        image_folder=str(tmp_path),
        plugins=[REF],
        tasks=["position", "transform"],
    )
    _plugin, tool = core_plugins.tool_specs(spec)["deepslice_place"]
    parameters = inspect.signature(tool.stub).parameters
    assert {name: p.default for name, p in parameters.items()} == {
        "sections": [],
        "order": [],
        "bad_sections": None,
        "write": None,
        "view": False,
    }
    assert (tool.kind, tool.group, tool.task) == (
        "write",
        "Positioning",
        "angles",
    )
    assert tool.writes == ("position", "angles", "affine")

    doc = declaration(
        "deepslice_place", DescriptionFacts.from_spec(spec, auto=False)
    ).doc
    assert doc.startswith(
        "Place sections with DeepSlice, a network trained on mouse brain sections.\n"
    )
    assert "~" not in doc
    assert (
        "Turn and flip every section the right way before the call, because the "
        "stack's angle is a vote of every section's own angle."
    ) in doc
    assert (
        '"transform" (each section\'s in-plane transform, poor on damaged sections)'
        in doc
    )

    linear_off = JobSpec(
        image_folder=str(tmp_path), plugins=[REF], tasks=["position"]
    )
    doc = declaration(
        "deepslice_place", DescriptionFacts.from_spec(linear_off, auto=False)
    ).doc
    assert '"transform"' not in doc and "in-plane transform" not in doc


def test_loading_the_plugin_imports_nothing_of_tensorflow_or_deepslice():
    code = (
        "import sys\n"
        "import langslice.engines.deepslice.plugin as p\n"
        "p.tools()\n"
        "bad = sorted(m for m in sys.modules\n"
        "             if m.split('.')[0] in ('tensorflow', 'tf_keras', 'keras', 'DeepSlice'))\n"
        "print(bad)\nsys.exit(1 if bad else 0)\n"
    )
    done = subprocess.run(
        [sys.executable, "-c", code],
        capture_output=True,
        text=True,
        stdin=subprocess.DEVNULL,
        check=False,
        timeout=120,
    )
    assert done.returncode == 0, done.stdout + done.stderr



def test_a_call_without_the_extra_is_refused_engine_missing(
    tmp_path: Path, monkeypatch
):
    monkeypatch.setattr(
        engine, "REQUIRED_MODULES", ("langslice_no_such_engine",)
    )
    reply_ = _call(tmp_path, write=[])
    assert reply_["error"] == "ENGINE_MISSING", reply_
    assert "langslice_no_such_engine" in reply_["message"]
    assert 'pip install "langslice[engines]"' in reply_["message"]
    assert not (tmp_path / "images" / "langslice" / "plugins").exists()


class _Served:
    """What ``urlopen`` answers in the weight test: fixed bytes."""

    def __init__(self, data: bytes) -> None:
        self.data = data

    def __enter__(self):
        return io.BytesIO(self.data)

    def __exit__(self, *details) -> None:
        return None


def test_the_weights_are_fetched_once_and_checked(tmp_path: Path, monkeypatch):
    data = b"weights"
    digest = hashlib.sha256(data).hexdigest()
    fetched: list[str] = []

    def urlopen(url, timeout):
        fetched.append(url)
        return _Served(data)

    monkeypatch.setattr(network.urllib.request, "urlopen", urlopen)
    names = [network.XCEPTION[0], network.PRIMARY[0], network.SECONDARY[0]]
    for role, name in zip(
        ("XCEPTION", "PRIMARY", "SECONDARY"), names, strict=True
    ):
        monkeypatch.setattr(network, role, (name, digest))
    weights = network.fetch_weights(tmp_path / "cache")
    assert weights.primary.read_bytes() == data
    assert fetched == [f"{network.BUCKET}/{name}" for name in names]
    network.fetch_weights(tmp_path / "cache")
    assert len(fetched) == 3

    monkeypatch.setattr(network, "SECONDARY", (names[2], "0" * 64))
    with pytest.raises(ToolError, match="could not be fetched") as refused:
        engine.fetched_weights(tmp_path / "cache")
    assert refused.value.code == "ENGINE_MISSING"
    assert not list((tmp_path / "cache").glob("*.part"))


def test_an_unknown_option_is_refused_when_the_plugin_loads():
    from langslice.engines.deepslice import plugin

    assert (
        plugin.tools({"long_edge": 1024})[0].handler.options.long_edge == 1024
    )
    assert (
        plugin.tools()[0].handler.options.weights_folder
        == network.CACHE_FOLDER
    )
    with pytest.raises(ValueError, match="takes the options"):
        plugin.tools({"python": "/usr/bin/python"})


@pytest.mark.parametrize(
    ("arguments", "words"),
    [
        ({"sections": ["s0", "s0"]}, "named twice"),
        ({"order": ["s0", "s1"]}, "every section run exactly once"),
        ({"sections": ["s0"], "order": ["s0"]}, "at least two sections"),
        (
            {"write": ["everything"]},
            "write takes 'positions' and 'angles' and 'transform'",
        ),
        ({"sections": ["s0", "s1"], "bad_sections": ["s2"]}, "not run"),
        (
            {"bad_sections": ["s0", "s1", "s2"]},
            "every section is a bad section",
        ),
    ],
)
def test_bad_arguments_are_refused_before_deepslice_runs(
    tmp_path: Path, monkeypatch, arguments: dict, words: str
):
    monkeypatch.setattr(engine, "require_engine", lambda: None)

    def never(*_arguments):
        raise AssertionError("DeepSlice ran")

    monkeypatch.setattr(engine, "fetched_weights", never)
    monkeypatch.setattr(engine, "run_deepslice", never)
    reply_ = _call(tmp_path, **arguments)
    assert reply_["error"] == "BAD_ARGS", reply_
    assert words in reply_["message"], reply_



def _coronal(depth: float, tilt: float = 0.0) -> np.ndarray:
    """A coronal anchoring at ``depth`` voxels, tilted dorsoventrally."""
    return np.array([0.0, depth, 320.0, 456.0, 0.0, 0.0, 0.0, tilt, -320.0])


def test_the_vote_turns_every_section_to_the_good_sections_angles():
    planes = np.array(
        [
            _coronal(depth, tilt)
            for depth, tilt in (
                (100.0, 8.0),
                (200.0, 8.0),
                (300.0, 8.0),
                (400.0, -40.0),
            )
        ]
    )
    bad = np.array([False, False, False, True])
    turned, votes = alignment.propagated(planes, bad)
    assert len(votes) == 2
    first_dv = alignment._angle(planes[0], alignment.DORSOVENTRAL)
    assert votes[0][0] == pytest.approx(
        first_dv, abs=1e-9
    )
    for row in turned:
        assert alignment._angle(row, alignment.DORSOVENTRAL) == pytest.approx(
            votes[-1][0], abs=1e-6
        )


def test_spacing_puts_the_sections_one_interval_apart_along_the_index():
    planes = np.array(
        [_coronal(depth) for depth in (300.0, 292.0, 283.0, 271.0)]
    )
    numbers = np.arange(1, 5)
    good = np.zeros(4, np.bool_)
    spaced, used = alignment.spaced(planes, numbers, good, thickness_um=-200.0)
    assert used.thickness_voxels == -8.0 and not used.estimated
    assert used.direction == alignment.CAUDAL_ROSTRO
    assert np.diff(spaced[:, 1]) == pytest.approx([-8.0, -8.0, -8.0])
    _, estimated = alignment.spaced(planes, numbers, good, thickness_um=None)
    assert estimated.estimated and -12.0 < estimated.thickness_voxels < -8.0


def test_the_in_plane_change_is_worded_against_the_held_transform():
    identity = (1.0, 0.0, 0.0, 0.0, 1.0, 0.0)
    assert reply.in_plane_change(identity, identity, 400, 300) == (
        "turn 0.0 deg counter-clockwise, scale x 1.00 y 1.00, shift 0% right, 0% down"
    )
    cos, sin = np.cos(np.radians(10)), np.sin(np.radians(10))
    turned = (cos, sin, -0.2, -sin, cos, 0.05)
    assert reply.in_plane_change(turned, identity, 300, 300) == (
        "turn 10.0 deg counter-clockwise, scale x 1.00 y 1.00, shift 20% left, 5% down"
    )
    half = (0.5, 0.0, 0.0, 0.0, 0.5, 0.0)
    assert reply.in_plane_change(half, half, 300, 300).startswith(
        "turn 0.0 deg counter-clockwise, scale x 1.00 y 1.00"
    )


def test_the_stack_angles_are_the_median_of_the_placed_rows():
    def row(
        name: str, pitch: float, yaw: float, *, placed: bool = True
    ) -> ProposalRow:
        plane = Placement(1.0, pitch, yaw, None) if placed else None
        return ProposalRow(
            id=name,
            nr=None,
            bad=False,
            final=plane,
            own=plane,
            anchoring=(),
            own_anchoring=(),
            refused=None,
        )

    rows = [
        row("a", -3.0, 0.3),
        row("b", -3.2, 0.5),
        row("c", -2.0, 0.1),
        row("d", 9.0, 9.0, placed=False),
    ]
    assert voted_angles(rows) == (-3.0, 0.3)
    assert voted_angles([row("d", 0.0, 0.0, placed=False)]) is None



@pytest.mark.slow
@pytest.mark.skipif(not HAS_EXTRA, reason="needs the langslice[engines] extra")
def test_deepslice_runs_in_this_process_and_alike_twice(
    tmp_path: Path, capsys
):
    """Four small blobs (DeepSlice's vote fails on two voting sections or
    fewer); a second run in the process answers exactly as the first (Keras's
    session is cleared before each), nothing is printed, nothing reaches the
    GPU. Fetches the weights on first use."""
    rng = np.random.default_rng(0)
    images = []
    y, x = np.mgrid[:240, :320]
    for index in range(4):
        blob = ((x - 160) / (120 - 10 * index)) ** 2 + (
            (y - 120) / 90
        ) ** 2 < 1
        pixels = np.where(blob, 90 + 20 * index, 235) + rng.normal(
            0, 4, blob.shape
        )
        path = tmp_path / f"section_{index}.png"
        Image.fromarray(np.clip(pixels, 0, 255).astype(np.uint8)).convert(
            "RGB"
        ).save(path)
        images.append((path.name, path))
    weights = engine.fetched_weights(network.CACHE_FOLDER)

    def run(folder: str, order: tuple[str, ...]) -> engine.EngineRun:
        return engine.run_deepslice(
            engine.EngineRequest(
                images=tuple(images),
                order=order,
                bad_sections=(images[0][0],),
                interval_um=-100.0 if order else None,
                ensemble=True,
                weights=weights,
                out_folder=tmp_path / folder,
            )
        )

    first = run("first", ())
    second = run("second", ())
    assert list(first.sections) == list(second.sections)
    for section, answer in first.sections.items():
        assert len(answer.anchoring) == 9 and len(answer.own_anchoring) == 9
        assert answer.anchoring == second.sections[section].anchoring
        assert answer.bad is (section == images[0][0])
    assert len(first.propagation) == 2 and first.spacing is None
    assert first.final_json.exists() and first.own_json.exists()
    assert sorted(answer.nr for answer in first.sections.values()) == [
        1,
        2,
        3,
        4,
    ]
    assert capsys.readouterr().out == ""

    spaced = run("spaced", tuple(name for name, _ in images))
    assert spaced.spacing is not None
    assert spaced.spacing.section_thickness_um_used == -100.0
    assert [answer.nr for answer in spaced.sections.values()] == [1, 2, 3, 4]

    import tensorflow as tf

    assert tf.config.get_visible_devices("GPU") == []
