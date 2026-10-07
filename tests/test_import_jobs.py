"""A job made from a registration made elsewhere: ``langslice-job FOLDER init
--registration FILE`` and ``langslice.create_job(..., registration=FILE)``.

A source job is given varied placements (positions, quarter turns, flips,
anisotropic scale, shear) with a DIFFERENT cutting angle per section, and
exports its ``quicknii.json`` and ``registration.json``; a new job made from
either (and from the DeepSlice fixtures, ``tests/fixtures/deepslice/``, which
DeepSlice 1.2.8's writers produced from the same placements) must map every
section file exactly as the source did (``registration.json``'s
``pixel_to_atlas_um``), and then run the image-model nonlinear step on top
(the packaged ``trace_borders`` with the golden stub model: its image call
and the ANTs fit of the traced lines; ``submit`` naming a section left
linear) end to end. The atlas is the importer tests' deep
synthetic one; no credential is read and no model called.
"""

from __future__ import annotations

import json
import shutil
from pathlib import Path
from typing import Any

import numpy as np
import pytest

from langslice.core.layers import atlas_facts
from langslice.core.spec import NonlinearSpec
from langslice.doors.cli.jobcli import main
from langslice.job.quint import job_export
from tests.cli_child import install
from tests.golden.record import (
    ID0,
    ID1,
    ID2,
    PIXEL_SIZE_UM,
    apply_patches,
    stub_image_model,
    write_sections,
)
from tests.test_import_registrations import (
    DEEPSLICE_NAMES,
    DEEPSLICE_SECTIONS,
    FIXTURES,
    PER_SECTION,
    DeepAtlas,
    Given,
    export_rows,
    make_workspace,
    params_for,
    pixel_map,
)

SUBMIT_FLAGS = ("--summary", "done", "--notes", "[]", "--interval-breaks", "[]")
#: Through a QuickNII file (anchorings rounded to 1e-6 voxels), in um.
ROUNDED_UM = 1e-3
#: Through registration.json, in um.
EXACT_UM = 1e-6


@pytest.fixture(scope="module")
def stack(tmp_path_factory: pytest.TempPathFactory) -> Path:
    images = tmp_path_factory.mktemp("import_jobs") / "stack"
    write_sections(images)
    return images


def _no_network(*_args: Any, **_kwargs: Any) -> Any:
    raise AssertionError("a real image model was called")


def _setup(monkeypatch: pytest.MonkeyPatch, tmp_path: Path, atlas: DeepAtlas) -> None:
    """The deep atlas wherever a door opens a job, the golden stub image model
    wherever one resolves a model, no network, a private HOME."""
    import langslice.doors.tools.toolbox as toolbox
    import langslice.providers.images as transport

    monkeypatch.setenv("HOME", str(tmp_path / "home"))
    apply_patches()
    install(lambda _name: atlas, monkeypatch.setattr)
    monkeypatch.setattr(transport, "generate_warped_segmentation_image", _no_network)

    def resolve(provider: str, model: str | None = None) -> Any:
        holder = type("Spec", (), {"nonlinear": NonlinearSpec(provider=provider,
                                                              image_model=model)})()
        return stub_image_model(holder)

    monkeypatch.setattr(toolbox, "resolve_image_model", resolve)
    # The stub stands for a connected image model (its key or login present).
    from langslice.doors.api import setup

    monkeypatch.setattr(setup, "image_model_connected", lambda _provider: True)


@pytest.fixture
def atlas() -> DeepAtlas:
    return DeepAtlas()


@pytest.fixture
def images(stack: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
           atlas: DeepAtlas) -> Path:
    _setup(monkeypatch, tmp_path, atlas)
    folder = tmp_path / "stack"
    shutil.copytree(stack, folder)
    return folder


def cli(capsys: pytest.CaptureFixture[str], *argv: str) -> tuple[int, dict[str, Any]]:
    code = main(list(argv))
    lines = [line for line in capsys.readouterr().out.splitlines() if line.strip()]
    assert len(lines) == 1, lines
    return int(code or 0), json.loads(lines[0])


def ok(capsys: pytest.CaptureFixture[str], folder: Path, verb: str,
       *flags: str) -> dict[str, Any]:
    code, envelope = cli(capsys, str(folder), verb, *flags)
    assert code == 0 and envelope["ok"] is True, envelope
    return envelope


def init(capsys: pytest.CaptureFixture[str], images: Path, *flags: str,
         atlas_name: str = "synthetic_deep_50um") -> dict[str, Any]:
    code, envelope = cli(capsys, str(images), "init", "--atlas", atlas_name,
                         "--preprocess", "none", "--no-debrief", *flags)
    assert code == 0, envelope
    return envelope


def supplied(workspace: Any, sections: list[Given]) -> list[str]:
    """The init flags that place *sections* as given, each at its own angles."""
    transforms = {g.section_id: {"kind": "interactive", "params": params_for(workspace, g)}
                  for g in sections}
    return [
        "--positions", json.dumps({g.section_id: g.position_mm for g in sections}),
        "--transforms", json.dumps(transforms),
        "--orientation", json.dumps({g.section_id: {"flip": g.flip,
                                                     "rotation_deg": g.rotation_deg}
                                     for g in sections}),
        "--section-angles", json.dumps({g.section_id: {"pitch": g.pitch_deg,
                                                        "yaw": g.yaw_deg}
                                        for g in sections}),
    ]


def matrices(job_folder: Path) -> dict[str, np.ndarray]:
    document = json.loads((job_folder / "registration.json").read_text())
    out = {}
    for entry in document["sections"]:
        assert entry["problem"] is None, entry
        out[entry["id"]] = np.asarray(entry["pixel_to_atlas_um"], dtype=np.float64)
    return out


def source_job(capsys: pytest.CaptureFixture[str], images: Path, atlas: DeepAtlas,
               tmp_path: Path) -> Path:
    """The source job: PER_SECTION's placements, exported."""
    workspace = make_workspace(images, atlas)
    envelope = init(capsys, images, "--tasks", "nonlinear", "--image-provider", "none",
                    "--pixel-size-um", str(PIXEL_SIZE_UM), "--job-dir",
                    str(tmp_path / "source"), *supplied(workspace, PER_SECTION))
    folder = Path(envelope["result"]["job_folder"])
    ok(capsys, folder, "export_maps")
    assert (folder / "exports" / "quicknii.json").is_file()
    return folder


def nonlinear_on_top(capsys: pytest.CaptureFixture[str], folder: Path,
                     ids: tuple[str, ...]) -> None:
    """The image-model nonlinear step on the imported placement, end to end:
    every section but the last traced (the trace and its fit, landed before
    the CLI answers), the last left linear at submit. The fit needs antspyx."""
    pytest.importorskip("ants", reason="the traced fit is ANTs (antspyx)")
    for name in ids[:-1]:
        envelope = ok(capsys, folder, "trace_borders", "--section", name)
        result = envelope["result"]
        assert result["image_correction"]["status"] == "ok", envelope
        assert result["work_status"] == "done", envelope
    envelope = ok(capsys, folder, "submit", *SUBMIT_FLAGS, "--left-linear",
                  json.dumps([{"id": ids[-1], "reason": "kept"}]))
    document = json.loads((folder / "registration.json").read_text())
    assert document["submitted"] is True, envelope
    mapping = {entry["id"]: entry["mapping"] for entry in document["sections"]}
    assert mapping == {**{name: "linear + residual" for name in ids[:-1]},
                       ids[-1]: "linear"}, mapping
    for entry in document["sections"]:
        assert entry["maps"], entry


# --- from a LangSlice job's exports ------------------------------------------------------


@pytest.mark.parametrize("export", ["quicknii", "registration"])
def test_a_new_job_from_a_jobs_export(capsys, images, atlas, tmp_path, export):
    source = source_job(capsys, images, atlas, tmp_path)
    given = matrices(source)
    file = (source / "exports" / "quicknii.json" if export == "quicknii"
            else source / "registration.json")
    copy = tmp_path / f"{export}.json"
    shutil.copy(file, copy)
    envelope = init(capsys, images, "--image-provider", "openai-oauth", "--pixel-size-um",
                    str(PIXEL_SIZE_UM), "--job-dir", str(tmp_path / "again"),
                    "--registration", str(copy))
    result = envelope["result"]
    assert result["tasks"] == ["nonlinear"]
    assert {"trace_borders", "ants_syn"} <= set(result["verbs"])
    assert "elastix_affine" not in result["verbs"]  # the imported placement is kept
    imported = result["registration"]
    assert imported["format"] == ("quicknii-json" if export == "quicknii"
                                  else "langslice-registration")
    assert [row["match"] for row in imported["sections"]] == ["exact"] * 3
    assert (imported["unmatched"], imported["missing"], imported["refused"]) == ([], [], {})
    assert envelope["warnings"] == []
    folder = Path(result["job_folder"])
    again = matrices(folder)
    tolerance = ROUNDED_UM if export == "quicknii" else EXACT_UM
    for name, matrix in given.items():
        assert np.allclose(again[name], matrix, atol=tolerance, rtol=0), name
    spec = json.loads((folder / "job.json").read_text())["spec"]
    # Per-section angles arrive as the per-section form (they differ here).
    assert set(spec["inputs"]["angles"]) == {ID0, ID1, ID2}
    assert spec["inputs"]["transforms"][ID0]["kind"] == "imported"
    assert set(spec["inputs"]["transforms"][ID0]["physical"]) >= {
        "rotation_deg", "scale_x", "scale_y", "shear", "translate_x_mm", "translate_y_mm"}
    # The same init again continues the job (the same inputs).
    code, envelope = cli(capsys, str(images), "init", "--atlas",
                         "synthetic_deep_50um", "--preprocess", "none", "--no-debrief",
                         "--image-provider", "openai-oauth", "--pixel-size-um",
                         str(PIXEL_SIZE_UM), "--job-dir", str(tmp_path / "again"),
                         "--registration", str(copy))
    assert code == 0, envelope
    nonlinear_on_top(capsys, folder, (ID0, ID1, ID2))


def test_a_shared_angle_arrives_as_the_stack_wide_form(capsys, images, atlas, tmp_path):
    shared = [Given(g.section_id, g.position_mm, 1.5, -2.0, g.rotation_deg, g.flip, g.knobs)
              for g in PER_SECTION]
    workspace = make_workspace(images, atlas)
    path = tmp_path / "quicknii.json"
    path.write_text(json.dumps(job_export(export_rows(workspace, shared), atlas_facts(atlas))))
    envelope = init(capsys, images, "--pixel-size-um", str(PIXEL_SIZE_UM),
                    "--image-provider", "none", "--registration", str(path))
    folder = Path(envelope["result"]["job_folder"])
    spec = json.loads((folder / "job.json").read_text())["spec"]
    angles = spec["inputs"]["angles"]
    assert angles == pytest.approx({"pitch": 1.5, "yaw": -2.0}, abs=1e-6)
    for name, matrix in matrices(folder).items():
        given = next(g for g in shared if g.section_id == name)
        assert np.allclose(matrix, pixel_map(workspace, given), atol=ROUNDED_UM, rtol=0)


def test_without_a_pixel_size_the_imported_one_is_supplied(capsys, images, atlas, tmp_path):
    workspace = make_workspace(images, atlas)
    path = tmp_path / "quicknii.json"
    path.write_text(json.dumps(job_export(export_rows(workspace, PER_SECTION),
                                          atlas_facts(atlas))))
    envelope = init(capsys, images, "--image-provider", "none", "--registration", str(path))
    imported = envelope["result"]["registration"]
    assert imported["pixel_size_source"] == "imported"
    # The median of what each imported map implies (their scales and shears
    # are part of it, so not the files' true size; the maps hold at it).
    assert 0.8 * PIXEL_SIZE_UM < imported["pixel_size_um"] < 1.25 * PIXEL_SIZE_UM
    assert any("um per pixel" in warning for warning in envelope["warnings"])
    folder = Path(envelope["result"]["job_folder"])
    spec = json.loads((folder / "job.json").read_text())["spec"]
    assert spec["inputs"]["pixel_size_um"] == imported["pixel_size_um"]
    for name, matrix in matrices(folder).items():
        given = next(g for g in PER_SECTION if g.section_id == name)
        assert np.allclose(matrix, pixel_map(workspace, given), atol=ROUNDED_UM, rtol=0)


# --- from DeepSlice's own files ---------------------------------------------------------


@pytest.mark.parametrize("name", ["deepslice.json", "deepslice.xml", "deepslice.csv"])
def test_a_new_job_from_deepslice(capsys, stack, tmp_path, monkeypatch, name):
    atlas = DeepAtlas("allen_mouse_50um")
    _setup(monkeypatch, tmp_path, atlas)
    images = tmp_path / "M1"
    images.mkdir()
    for old, new in DEEPSLICE_NAMES.items():
        shutil.copy2(stack / old, images / new)
    envelope = init(capsys, images, "--image-provider", "openai-oauth", "--pixel-size-um",
                    str(PIXEL_SIZE_UM), "--registration", str(FIXTURES / name),
                    atlas_name="allen_mouse_50um")
    imported = envelope["result"]["registration"]
    assert [row["match"] for row in imported["sections"]] == ["section number"] * 3
    if name.endswith(".csv"):  # no target in a DeepSlice CSV: said
        assert any("names no target" in warning for warning in envelope["warnings"])
    else:
        assert envelope["warnings"] == []
    folder = Path(envelope["result"]["job_folder"])
    workspace = make_workspace(images, atlas)
    found = matrices(folder)
    for given in DEEPSLICE_SECTIONS:
        assert np.allclose(found[given.section_id], pixel_map(workspace, given),
                           atol=ROUNDED_UM, rtol=0), given.section_id
    nonlinear_on_top(capsys, folder, tuple(sorted(DEEPSLICE_NAMES.values())))


# --- what is reported and what is refused ------------------------------------------------


def test_unmatched_missing_and_markers_are_reported(capsys, images, atlas, tmp_path):
    workspace = make_workspace(images, atlas)
    rows = export_rows(workspace, PER_SECTION[:2],
                       markers={ID0: [[10.0, 20.0, 12.0, 19.0]]})
    stray = dict(rows[1], filename="elsewhere.png")
    document = job_export([*rows, stray], atlas_facts(atlas))
    path = tmp_path / "visualign.json"
    path.write_text(json.dumps(document))
    envelope = init(capsys, images, "--pixel-size-um", str(PIXEL_SIZE_UM),
                    "--image-provider", "none", "--registration", str(path))
    imported = envelope["result"]["registration"]
    assert imported["format"] == "visualign-json"
    assert imported["unmatched"] == ["elsewhere.png"] and imported["missing"] == [ID2]
    assert imported["markers_imported"] is False
    text = "\n".join(envelope["warnings"])
    assert "elsewhere.png" in text and ID2 in text
    assert "VisuAlign nonlinear markers (1 section(s)) were not imported" in text
    assert "LangSlice's nonlinear step replaces them" in text
    folder = Path(envelope["result"]["job_folder"])
    state = json.loads((folder / "state.json").read_text())
    held = {row["id"]: row for row in state["slices"]}
    assert held[ID2]["position_mm"] is None and held[ID2]["transform"] is None
    assert held[ID0]["deformation"] is None  # the markers are not a deformation


def test_refusals(capsys, images, atlas, tmp_path):
    workspace = make_workspace(images, atlas)
    path = tmp_path / "quicknii.json"
    path.write_text(json.dumps(job_export(export_rows(workspace, PER_SECTION),
                                          atlas_facts(atlas))))
    base = (str(images), "init", "--atlas", "synthetic_deep_50um",
            "--image-provider", "none", "--registration", str(path))
    for flags, named in ((("--positions", json.dumps({ID0: 1.0})), "--positions"),
                         (("--pitch", "1"), "--pitch"),
                         (("--section-angles", json.dumps({ID0: {"pitch": 1}})),
                          "--section-angles"),
                         (("--angles",), "--angles")):
        code, envelope = cli(capsys, *base, *flags)
        assert code == 2 and envelope["error"]["code"] == "BAD_ARGUMENTS", envelope
        assert named in envelope["error"]["message"]
    # Two entries naming one section: ambiguous, refused.
    rows = export_rows(workspace, PER_SECTION)
    twice = tmp_path / "twice.json"
    twice.write_text(json.dumps(job_export(
        [*rows, dict(rows[0], filename=rows[0]["filename"].replace(".png", ".PNG"))],
        atlas_facts(atlas))))
    code, envelope = cli(capsys, str(images), "init", "--atlas", "synthetic_deep_50um",
                         "--image-provider", "none", "--registration", str(twice))
    assert code == 2 and envelope["error"]["code"] == "BAD_REGISTRATION", envelope
    assert "several entries" in envelope["error"]["message"]
    # Nothing placed: refused, naming why.
    nothing = tmp_path / "nothing.json"
    nothing.write_text(json.dumps(job_export([dict(row, filename=f"x{i}.png")
                                              for i, row in enumerate(rows)],
                                             atlas_facts(atlas))))
    code, envelope = cli(capsys, str(images), "init", "--atlas", "synthetic_deep_50um",
                         "--image-provider", "none", "--registration", str(nothing))
    assert code == 2 and envelope["error"]["code"] == "BAD_REGISTRATION", envelope
    assert "No section could be placed" in envelope["error"]["message"]
    code, envelope = cli(capsys, str(images), "init", "--atlas", "synthetic_deep_50um",
                         "--registration", str(tmp_path / "missing.json"))
    assert code == 2 and envelope["error"]["code"] == "BAD_REGISTRATION", envelope


# --- the library -----------------------------------------------------------------------


class Stub:
    """A model of the caller's own: answers with the placed borders it was sent."""

    provider = "my-lab"
    model = "lab-edit-1"

    def call(self, request: Any) -> Any:
        def colour(image: Any) -> float:
            pixels = np.asarray(image.convert("RGB")).astype(np.int16)
            return float(np.abs(pixels[..., 0] - pixels[..., 2]).mean())

        return max([request.slice_image, *request.reference_images], key=colour)


def test_the_library_creates_and_registers_an_imported_job(capsys, images, atlas, tmp_path):
    import langslice

    pytest.importorskip("ants", reason="register_job's deformable step is ANTs (antspyx)")

    source = source_job(capsys, images, atlas, tmp_path)
    given = matrices(source)
    path = tmp_path / "quicknii.json"
    shutil.copy(source / "exports" / "quicknii.json", path)
    said: list[str] = []
    with pytest.raises(ValueError, match="cannot be combined"):
        langslice.create_job(images, registration=path, positions={ID0: 1.0},
                             atlas="synthetic_deep_50um")
    job = langslice.create_job(images, atlas="synthetic_deep_50um", registration=path,
                               pixel_size_um=PIXEL_SIZE_UM, image_model=Stub(),
                               preprocess="none", job_dir=tmp_path / "lib", emit=said.append)
    assert job.job.spec.tasks == ["nonlinear"]
    assert job.imported is not None and job.imported["format"] == "quicknii-json"
    assert job.state.mixed_angles
    result = langslice.register_job(job)
    assert result.ok, result.problems
    for name, matrix in matrices(tmp_path / "lib").items():
        assert np.allclose(matrix, given[name], atol=ROUNDED_UM, rtol=0), name
    assert all(section.residual is not None for section in result.sections)
