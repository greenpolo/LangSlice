"""Section images (``core.section_images``): recipes, made images as files, the
opening ``preprocessed`` image, orientation at draw, geometry that no image moves,
and the migration of a format-3 job."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import pytest
from PIL import Image

from langslice.agent.engine import build_context
from langslice.core import deformation
from langslice.core import section_images as images
from langslice.core.handoff import prepare_linear_registration
from langslice.core.image_prep import adaptive_preprocess, prepare_image_for_vlm
from langslice.core.maps import section_frame
from langslice.core.sections import render_slice
from langslice.core.spec import JobSpec
from langslice.job.checkpoint import FORMAT_KEY, STATE_FORMAT_VERSION
from langslice.job.job import Job
from tests.deformable_synthetic import SyntheticAtlas

UM = 25.0
IDS = ("s0.png", "s1.png")


@pytest.fixture(scope="module")
def atlas() -> SyntheticAtlas:
    return SyntheticAtlas()


def _rgb_section(seed: int = 0) -> Image.Image:
    """Tissue whose three colour planes carry different structure."""
    rng = np.random.default_rng(seed)
    yy, xx = np.mgrid[:120, :160]
    body = ((yy - 60) / 50.0) ** 2 + ((xx - 80) / 70.0) ** 2 < 1
    red = np.where(body, 90 + 60 * np.sin(xx / 6.0), 5)
    green = np.where(body, 40 + 150 * ((yy // 15 + xx // 15) % 2), 5)
    blue = np.where(body, rng.uniform(20, 60, body.shape), 5)
    return Image.fromarray(np.stack([red, green, blue], axis=-1).clip(0, 255).astype(np.uint8))


def _open(tmp_path: Path, atlas: SyntheticAtlas, *, preprocess: str = "auto") -> tuple[Job, Any]:
    for index, name in enumerate(IDS):
        _rgb_section(index).save(tmp_path / name)
    spec = JobSpec(image_folder=str(tmp_path), model="fake-model", preprocess=preprocess,
                   tasks=["position", "transform", "nonlinear"], inputs={"pixel_size_um": UM})
    ctx = build_context(spec, emit=lambda _m: None, atlas_loader=lambda _n: atlas)
    job = Job.open(spec, ctx, folder=ctx.job_folder, results_path=ctx.results_path)
    with job.writing():
        before = job.snapshot()
        for record in job.state.slices:
            record.position_mm = 0.1
            record.transform = {"kind": "interactive", "params": [1.0, 0.0, 0.0, 0.0, 1.0, 0.0],
                                "calibration": {"section_um_per_px": UM, "source": "host"}}
        job.commit(before)
    return job, ctx


def _gray(image: Image.Image) -> np.ndarray:
    return np.asarray(image.convert("L"), dtype=np.int16)


# --- recipes -------------------------------------------------------------------------------


def test_a_recipe_is_validated_and_its_defaults_are_the_opening_recipe():
    assert images.validate_recipe() == images.OPENING_RECIPE
    assert images.OPENING_RECIPE["weights"] == "auto"
    assert images.validate_recipe(weights={})["weights"] == "auto"
    recipe = images.validate_recipe(weights={"red": 1}, clahe_clip=2, contrast=[0.1, 0.9],
                                    gamma=1.0, invert=True)
    assert recipe["weights"] == {"red": 1.0} and recipe["gamma"] is None
    assert recipe["contrast"] == [0.1, 0.9] and recipe["invert"] is True
    for bad in ({"weights": {"red": -1}}, {"weights": {"red": 0}}, {"clahe_clip": 41},
                {"clahe_tiles": 0}, {"contrast": [0.9, 0.1]}, {"gamma": 20.0},
                {"n4": "yes"}):
        with pytest.raises(ValueError):
            images.validate_recipe(**bad)
    assert "CLAHE 2 on 8x8 tiles" in images.recipe_words(recipe)
    assert "inverted" in images.recipe_words(recipe)


# --- the opening image ---------------------------------------------------------------------


def test_the_opening_image_is_the_default_render_exactly(tmp_path: Path, atlas):
    """Nothing the engines receive changes: at every size the opening
    ``preprocessed`` is the automatic path drawn at that size, and its file is
    that drawing at the working copy's size."""
    job, ctx = _open(tmp_path, atlas)
    for record in job.state.slices:
        entry = record.images[images.PREPROCESSED]
        assert entry["source"] == {"opening": "adaptive"}
        assert (job.folder / entry["file"]).is_file()
        assert entry["file"].endswith(f"{Path(record.id).stem}/images/preprocessed.png")
        source, _factor = ctx.working_source(record.id)
        for edge in (64, 120, 160, 512):
            old = adaptive_preprocess(prepare_image_for_vlm(source, max_long_edge=edge))
            drawn = render_slice(ctx, record, long_edge=edge)
            assert np.array_equal(_gray(drawn), _gray(old)), edge
        held = np.asarray(Image.open(job.folder / entry["file"]), dtype=np.int16)
        assert np.array_equal(held, _gray(adaptive_preprocess(source)))
        assert entry["digest"] == images.digest_of(held.astype(np.uint8))
        # What a fit reads is the same render at the fit grid's size.
        grid = deformation.fit_grid(job.state, ctx, record)
        image, identity = deformation.section_image(ctx, job.state, grid,
                                                    images.PREPROCESSED)
        old = adaptive_preprocess(prepare_image_for_vlm(source,
                                                        max_long_edge=deformation.FIT_LONG_EDGE))
        assert np.array_equal(_gray(image), _gray(old))
        assert identity["name"] == images.PREPROCESSED and identity["kind"] == "made"


def test_a_job_that_preprocesses_nothing_opens_on_the_file_in_gray(tmp_path: Path, atlas):
    job, ctx = _open(tmp_path, atlas, preprocess="none")
    record = job.state.slices[0]
    assert record.images[images.PREPROCESSED]["source"] == {"opening": "plain"}
    drawn = render_slice(ctx, record, long_edge=160)
    expected = Image.open(tmp_path / record.id).convert("L")
    assert np.array_equal(_gray(drawn), np.asarray(expected, dtype=np.int16))


# --- made images ------------------------------------------------------------------------


def test_a_made_image_is_a_file_read_back_and_a_chain_composes(tmp_path: Path, atlas):
    job, ctx = _open(tmp_path, atlas)
    record = job.state.by_id("s0.png")
    assert record is not None
    recipe = images.validate_recipe(weights={"green": 1}, clahe_clip=0)
    array, used = images.make(ctx, record.id, record.images, recipe)
    assert used == {"green": 1.0} and array.dtype == np.uint8
    assert array.shape == ctx.working_source(record.id)[0].size[::-1]
    entry = images.write_image(job.layout, record, "greens", array, {"recipe": recipe})
    assert entry["file"].endswith("images/greens.png")
    assert np.array_equal(images.image_plane(ctx, record, "greens"), array)
    view = images.read_image(ctx, record, "greens")
    assert view.payload() == {"name": "greens", "kind": "made", "file": entry["file"],
                              "digest": entry["digest"]}
    # The same pixels again reuse the file; other pixels never overwrite it.
    again = images.write_file(job.layout, record.id, "greens", array, {"recipe": recipe})
    assert again["file"] == entry["file"]
    inverted = images.validate_recipe(weights={"greens": 1}, clahe_clip=0, invert=True)
    chained, _ = images.make(ctx, record.id, record.images, inverted)
    assert np.abs(chained.astype(int) + array.astype(int) - 255).max() <= 1
    other = images.write_file(job.layout, record.id, "greens", chained, {"recipe": inverted})
    assert other["file"].endswith("images/greens.2.png")
    with pytest.raises(ValueError):
        images.make(ctx, record.id, record.images, images.validate_recipe(weights={"nope": 1}))
    assert images.image_names(ctx, record) == ["red", "green", "blue", "preprocessed",
                                               "greens"]
    raw = images.read_image(ctx, record, "red")
    assert raw.kind == "raw" and raw.digest is None and raw.file == record.id


def test_a_made_image_is_held_unoriented_and_turned_where_it_is_drawn(tmp_path: Path, atlas):
    job, ctx = _open(tmp_path, atlas)
    record = job.state.by_id("s0.png")
    assert record is not None
    recipe = images.validate_recipe(weights={"red": 1}, clahe_clip=0)
    array, _ = images.make(ctx, record.id, record.images, recipe)
    images.write_image(job.layout, record, "reds", array, {"recipe": recipe})
    upright = _gray(render_slice(ctx, record, look={"image": "reds"}))
    record.rotation_deg, record.flip = 90, True
    turned = _gray(render_slice(ctx, record, look={"image": "reds"}))
    assert np.array_equal(turned, np.rot90(upright)[:, ::-1])
    # The entry does not change: the file is the file's orientation.
    assert images.image_plane(ctx, record, "reds").shape == array.shape


def test_no_image_moves_the_geometry(tmp_path: Path, atlas):
    """Placements and maps are the same whichever image is made or drawn."""
    job, ctx = _open(tmp_path, atlas)
    record = job.state.by_id("s0.png")
    assert record is not None

    def geometry() -> tuple[Any, ...]:
        ctx.frame_cache.clear()
        prepared = prepare_linear_registration(job.state, ctx, record.id, long_edge=512)
        frame = section_frame(job.state, ctx, record)
        return (prepared.image.size, prepared.atlas_to_slice.tolist(),
                prepared.metadata["section_um_per_px"],
                np.asarray(frame.pixel_to_atlas_um((160, 120))).tolist())

    before = geometry()
    recipe = images.validate_recipe(weights={"blue": 1}, clahe_clip=8, invert=True)
    array, _ = images.make(ctx, record.id, record.images, recipe)
    images.write_image(job.layout, record, images.PREPROCESSED, array, {"recipe": recipe})
    assert render_slice(ctx, record, long_edge=512).size == before[0]
    assert geometry() == before


def test_the_status_summary_names_each_image_and_where_it_differs(tmp_path: Path, atlas):
    job, ctx = _open(tmp_path, atlas)
    summary = images.summary(job.state, ctx.spec)
    assert summary == {"preprocessed": {"weights": "auto", "clahe_clip": 4.0, "clahe_tiles": 8,
                                        "made": "when the job opened"}}
    record = job.state.by_id("s1.png")
    assert record is not None
    images.write_image(job.layout, record, "hema", np.zeros((120, 160), np.uint8),
                       {"script": "colour deconvolution"})
    held = images.summary(job.state, ctx.spec)["hema"]
    assert held == {"script": "colour deconvolution", "except": ["s0.png"]}


# --- migration ----------------------------------------------------------------------------


OLD_RECIPE = {"weights": {"green": 1.0}, "clahe_clip": 2.0, "clahe_tiles": 8, "n4": False,
              "denoise": False, "contrast_limits": None, "gamma": None, "invert": False}


def _format_three(job: Job) -> dict[str, Any]:
    data = job.state.to_dict()
    for row in data["slices"]:
        row.pop("images", None)
    data["appearance"] = {
        "derived": {
            "preprocessed": {"stack": None, "sections": {"s1.png": dict(OLD_RECIPE)}},
            "dapi": {"stack": None, "sections": {"s0.png": {**OLD_RECIPE, "invert": True,
                                                            "weights": None}}},
        },
        "channels": {"red": {"contrast_limits": None, "gamma": 2.0, "colormap": None}},
    }
    return data


def test_a_format_three_job_is_migrated_to_files(tmp_path: Path, atlas):
    job, ctx = _open(tmp_path, atlas)
    folder = job.folder
    old = _format_three(job)
    (folder / "state.json").write_text(json.dumps({FORMAT_KEY: 3, **old}))
    steps = sorted((folder / "history").glob("step-*.json"))
    assert steps
    for path in steps:
        held = json.loads(path.read_text())
        held.pop(FORMAT_KEY)
        for row in held.get("slices", []):
            row.pop("images", None)
        path.write_text(json.dumps({FORMAT_KEY: 3, **held, "appearance": {}}))
    for path in (folder / "sections").glob("*/images/*.png"):
        path.unlink()
    job.close()

    fresh = build_context(job.spec, emit=lambda _m: None, atlas_loader=lambda _n: atlas)
    loaded = Job.load(job.spec, fresh, folder=folder)
    s0, s1 = loaded.state.by_id("s0.png"), loaded.state.by_id("s1.png")
    assert s0 is not None and s1 is not None
    assert s0.images["preprocessed"]["source"] == {"opening": "adaptive"}
    assert s1.images["preprocessed"]["source"]["recipe"]["weights"] == {"green": 1.0}
    assert s0.images["dapi"]["source"]["recipe"]["invert"] is True
    assert s0.images["dapi"]["source"]["recipe"]["weights"] == "auto"
    assert "dapi" not in s1.images
    for record in (s0, s1):
        for entry in record.images.values():
            assert (folder / entry["file"]).is_file()
    # Display properties are gone, the state is format 4 on disk, history too.
    saved = json.loads((folder / "state.json").read_text())
    assert saved[FORMAT_KEY] == STATE_FORMAT_VERSION == 4 and "appearance" not in saved
    for path in sorted((folder / "history").glob("step-*.json")):
        assert json.loads(path.read_text())[FORMAT_KEY] == STATE_FORMAT_VERSION
    assert loaded.undo_stack  # the history survived its migration
    # The opening image comes back with the same pixels it had.
    drawn = render_slice(fresh, s0, long_edge=160)
    source, _ = fresh.working_source("s0.png")
    old_render = adaptive_preprocess(prepare_image_for_vlm(source, max_long_edge=160))
    assert np.array_equal(_gray(drawn), _gray(old_render))
