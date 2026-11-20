"""What an agent scripting a real brain stumbled on stays fixed.

Each test holds one thing a Fable agent registering a Carey brain with
scripts on reported: a crash drawing a section under an earlier
registration, a stored shape copied as a dataclass, objects whose fields
were undocumented, files it was told to keep in folders that did not exist,
a reply that hid the cutting angles it wrote, and wording that misled it.
"""

from __future__ import annotations

import ast
import dataclasses
import inspect
from pathlib import Path

import numpy as np

import langslice
from langslice.core.stack.part import Shape
from langslice.plugins.frames import AtlasPlane, FitFrame
from langslice.plugins.reads import AffineView, AtlasView, LayerRef, SectionRaster, SectionView
from langslice.tools.descriptions.door_text import opening_for_mcp
from tests.synthetic_job import PIXEL_SIZE_UM, POSITIONS

SOURCE = Path(__file__).resolve().parents[1] / "src" / "langslice"
VENDORED = SOURCE / "_vendor"
RETURNED_TO_SCRIPTS = (SectionView, AtlasView, AffineView, LayerRef, SectionRaster, FitFrame)
COPY_CALLS = {"replace", "dataclasses.replace"}


def shape_names() -> set[str]:
    found: set[str] = set()
    pending: list[type] = [Shape]
    while pending:
        cls = pending.pop()
        for sub in cls.__subclasses__():
            found.add(sub.__name__)
            pending.append(sub)
    return found


def call_name(node: ast.Call) -> str:
    return ast.unparse(node.func)


def shape_copies(path: Path, shapes: set[str]) -> list[str]:
    tree = ast.parse(path.read_text(encoding="utf-8"))
    found: list[str] = []
    for function in ast.walk(tree):
        if not isinstance(function, ast.FunctionDef | ast.AsyncFunctionDef):
            continue
        typed = {
            arg.arg: ast.unparse(arg.annotation)
            for arg in [*function.args.args, *function.args.kwonlyargs]
            if arg.annotation is not None
        }
        for node in ast.walk(function):
            if (
                isinstance(node, ast.Call)
                and call_name(node) in COPY_CALLS
                and node.args
                and isinstance(node.args[0], ast.Name)
                and typed.get(node.args[0].id, "").strip("'\"") in shapes
            ):
                found.append(f"{path.relative_to(SOURCE).as_posix()}:{node.lineno}")
    return found


def test_no_stored_shape_is_copied_as_a_dataclass() -> None:
    shapes = shape_names()
    found = [
        where
        for path in sorted(SOURCE.rglob("*.py"))
        if VENDORED not in path.parents
        for where in shape_copies(path, shapes)
    ]
    assert not found, (
        "A stored shape is a Pydantic model: copy it with model_copy(update=...), "
        "not dataclasses.replace:\n" + "\n".join(found)
    )


def test_every_object_a_script_gets_back_documents_its_fields() -> None:
    missing = [
        f"{cls.__name__}.{field.name}"
        for cls in (*RETURNED_TO_SCRIPTS, AtlasPlane)
        for field in dataclasses.fields(cls)
        if field.name not in (inspect.getdoc(cls) or "")
    ]
    assert not missing, "Say what each field holds in its class docstring:\n" + "\n".join(missing)


def test_one_opening_page_is_not_a_range() -> None:
    assert "through" not in " ".join(opening_for_mcp(1))
    assert "page=3" in " ".join(opening_for_mcp(3))


def identity_map(frame: object) -> np.ndarray:
    placed = frame.placed_atlas_to_section
    assert placed is not None
    height, width = frame.image.shape[:2]
    cols, rows = np.meshgrid(np.arange(width, dtype=float), np.arange(height, dtype=float))
    points = np.stack([cols, rows, np.ones_like(cols)], axis=-1)
    native = points @ np.linalg.inv(placed).T
    return native[..., :2].astype(np.float32)


def test_a_script_sees_a_section_under_an_earlier_deformed_registration(images: Path) -> None:
    with langslice.create_job(
        images,
        positions=POSITIONS,
        pixel_size_um=PIXEL_SIZE_UM,
        preprocess="none",
        tasks=["transform", "nonlinear"],
    ) as job:
        assert Path(job.folder, "scripts").is_dir()
        assert Path(job.folder, "scratch").is_dir()
        section = next(iter(POSITIONS))
        fitted = job.elastix_affine(sections=[section])
        assert fitted["status"] == "ok", fitted
        context = job.context
        raster = context.reads.section_raster(section)
        assert raster.um_per_px is not None
        frame = context.frames.fit_frame(section)
        warped = context.writes.set_deformations(
            [{"id": section, "map": identity_map(frame), "note": "identity"}]
        )
        assert section in warped.written, warped
        held = job.status()["rows"][0]["registration"]
        undone = job.undo()
        assert undone["status"] == "ok", undone
        looked = job.look(mode="overlay", sections=[section], registrations={section: held})
    assert looked["status"] == "ok", looked


def test_writing_only_cutting_angles_says_so(images: Path) -> None:
    with langslice.create_job(
        images, pixel_size_um=PIXEL_SIZE_UM, preprocess="none", tasks=["position"]
    ) as job:
        reply = job.position_sections(cutting_angles={"pitch_deg": 1.0, "yaw_deg": 0.0}, view=False)
    assert reply["status"] == "ok", reply
    assert reply["cutting_angles_written"]["pitch_deg"] == 1.0, reply


def test_atlas_pictures_take_cutting_angles_and_write_none(images: Path) -> None:
    with langslice.create_job(
        images, positions=POSITIONS, pixel_size_um=PIXEL_SIZE_UM, preprocess="none"
    ) as job:
        before = job.status()["cutting_angles_deg"]
        tilted = {"pitch_deg": 4.0, "yaw_deg": -3.0}
        looked = job.look(mode="atlas", positions_mm=[0.15], cutting_angles=tilted)
        regions = job.grep_atlas_view(regions=["root"], positions_mm=[0.15], cutting_angles=tilted)
        refused = job.look(mode="overlay", cutting_angles=tilted)
        after = job.status()["cutting_angles_deg"]
    assert looked["status"] == "ok", looked
    assert "yaw -3" in looked["pictures"][0]["caption"], looked["pictures"][0]["caption"]
    assert regions["status"] == "ok", regions
    assert refused["status"] == "error", refused
    assert before == after


def test_a_collapsed_affine_fit_is_flagged() -> None:
    from types import SimpleNamespace

    from langslice.core.fits.flags import FitFlag
    from langslice.tools.fitting.affine_fit import plausibility_flags

    squashed = SimpleNamespace(
        physical=SimpleNamespace(scale_x=1.068, scale_y=1.375, shear=0.0, rotation_deg=0.0)
    )
    codes = {flag["code"] for flag in plausibility_flags(None, squashed)}
    assert FitFlag.ANISOTROPIC in codes


def test_a_scripting_agent_is_told_it_may_use_any_method() -> None:
    from langslice.doors.mcp.server import INSTRUCTIONS
    from langslice.tools.descriptions.door_text import SCRIPTING

    connectors = Path(__file__).resolve().parents[1] / "connectors"
    agents = [
        connectors / "claude-code" / "agents" / "register-scripting.md",
        connectors / "antigravity" / "agents" / "register-scripting.md",
        connectors / "codex" / "agents" / "register_scripting.toml",
    ]
    assert "not limited to LangSlice's tools" in SCRIPTING
    assert "only" not in INSTRUCTIONS
    assert all("any software you judge best" in path.read_text(encoding="utf-8") for path in agents)
