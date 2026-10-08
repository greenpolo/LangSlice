"""grep_atlas: text lookup of atlas regions, with optional in-plane presence,
in every run."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
from PIL import Image

from langslice.agent.engine import build_context
from langslice.core.spec import JobSpec
from langslice.doors.tools.toolbox import build_tools
from langslice.job.job import ingest
from tests.fakes import SlabAtlas
from tests.linear_tool_helpers import tool_named as _tool

_REGIONS = [
    (1, "root", "root", [1]),
    (2, "CTX", "Cerebral cortex", [1, 2]),
    (3, "HPF", "Hippocampal formation", [1, 2, 3]),
    (4, "CA1", "Field CA1", [1, 2, 3, 4]),
    (5, "DG", "Dentate gyrus", [1, 2, 3, 5]),
    (6, "TH", "Thalamus", [1, 6]),
]


def _atlas() -> SlabAtlas:
    atlas = SlabAtlas()
    ann = np.zeros((20, 12, 16), dtype=np.int32)
    ann[:, 2:10, 2:14] = 1
    ann[:10, 3:6, 3:6] = 4  # CA1 only in the anterior half
    ann[:, 6:9, 6:9] = 6
    atlas.annotation = ann
    atlas.structures = {  # type: ignore[attr-defined]
        i: {"id": i, "acronym": a, "name": n, "structure_id_path": p}
        for i, a, n, p in _REGIONS
    }
    return atlas


def _stack(folder: Path, tasks: list[str], atlas: SlabAtlas | None = None):
    Image.fromarray(np.full((30, 40, 3), 40, dtype=np.uint8)).save(folder / "s0.png")
    spec = JobSpec(image_folder=str(folder), model="fake-model", preprocess="none", tasks=tasks)
    the_atlas = atlas or _atlas()
    ctx = build_context(spec, emit=lambda _m: None, atlas_loader=lambda _n: the_atlas)
    state = ingest(spec, ctx)
    return state, ctx, spec


def _grep(tmp_path: Path):
    state, ctx, spec = _stack(tmp_path, ["nonlinear"])
    return state, _tool(build_tools(state, ctx, spec), "grep_atlas")


@pytest.mark.parametrize("tasks", [[], ["position"], ["transform"], ["nonlinear"]])
def test_the_tool_is_in_every_run(tmp_path: Path, tasks: list[str]):
    state, ctx, spec = _stack(tmp_path, tasks)
    names = build_tools(state, ctx, spec).names
    assert {"grep_atlas", "grep_atlas_view"} <= set(names)


def test_lookup_by_acronym_name_and_id_with_ancestry(tmp_path: Path):
    _, grep = _grep(tmp_path)
    assert grep("ctx")["rows"][0] == {
        "acronym": "CTX", "id": 2, "name": "Cerebral cortex",
        "ancestry": ["root"], "n_descendants": 3,
    }
    hippocampal = grep("hippocampal")["rows"]
    assert [r["acronym"] for r in hippocampal] == ["HPF"]
    assert hippocampal[0]["ancestry"] == ["root", "CTX"]
    assert hippocampal[0]["n_descendants"] == 2
    by_id = grep("4")["rows"][0]
    assert by_id["acronym"] == "CA1" and by_id["ancestry"] == ["root", "CTX", "HPF"]
    assert grep("nothing here")["rows"] == []
    assert grep("")["error"] == "BAD_ARGS"


def test_presence_flag_follows_the_section_placement(tmp_path: Path):
    state, grep = _grep(tmp_path)
    record = state.slices[0]
    record.position_mm = 2.0  # anterior: CA1 present
    rows = {r["acronym"]: r["in_section"] for r in grep("a", record.id)["rows"]}
    assert rows["HPF"] and rows["CTX"] and rows["TH"]  # HPF and CTX via descendant CA1
    assert "in_section" not in grep("a")["rows"][0]
    record.position_mm = 15.0  # posterior: no CA1
    rows = {r["acronym"]: r["in_section"] for r in grep("a", record.id)["rows"]}
    assert not rows["HPF"] and not rows["CTX"] and rows["TH"]
    assert grep("o", "missing.png")["error"] == "UNKNOWN_SLICE_IDS"
    record.position_mm = None
    result = grep("CTX", record.id)
    assert "in_section" not in result["rows"][0] and "no position" in result["note"]


def test_rows_are_capped_and_the_rest_counted(tmp_path: Path):
    atlas = _atlas()
    atlas.structures = {  # type: ignore[attr-defined]
        i: {"id": i, "acronym": f"R{i}", "name": "Region", "structure_id_path": [1, i]}
        for i in range(1, 51)
    }
    state, ctx, spec = _stack(tmp_path, ["nonlinear"], atlas)
    result = _tool(build_tools(state, ctx, spec), "grep_atlas")("region")
    assert len(result["rows"]) == 40
    assert result["more"] == 10 and result["matches"] == 50
