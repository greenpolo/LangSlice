"""A call that runs no fit loads no registration engine.

Every agent CLI call is a process of its own, so torch, FireANTs, SimpleITK
and ITK are imported only inside the functions that fit
(``core/fits/deformable/engines.py``): ``init``, ``position_sections``,
``status`` and ``look`` on the synthetic stack, run in a fresh interpreter,
leave none of them in ``sys.modules``.
"""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
#: The engine libraries, by their top-level module names.
ENGINES = ("torch", "fireants", "SimpleITK", "itk")
CALLS = """
import contextlib, io, json, sys
from pathlib import Path
from tests.cli_child import install
from tests.synthetic_stack import (ID0, ID1, ID2, PIXEL_SIZE_UM, apply_patches, atlas_loader,
                                   write_sections)
from langslice.doors.agent_cli.main import main

apply_patches()
install(atlas_loader())
images = Path(sys.argv[1])
write_sections(images)
placed = {"sections": [{"id": ID0, "position_mm": 4.0}, {"id": ID1, "position_mm": 5.0},
                       {"id": ID2, "position_mm": 6.0}]}
codes = []
for argv in (["init", "--tasks", "position,transform,nonlinear", "--nonlinear-provider", "none",
              "--preprocess", "none", "--pixel-size-um", str(PIXEL_SIZE_UM)],
             ["position_sections", "--args", json.dumps(placed)],
             ["status"], ["look", "--mode", "overlay", "--sections", ID0]):
    with contextlib.redirect_stdout(io.StringIO()):
        codes.append(main([str(images), *argv]))
print(json.dumps({"codes": codes, "loaded": sorted(
    name for name in sys.argv[2:] if name in sys.modules)}))
"""


def test_calls_without_a_fit_import_no_engine(tmp_path):
    done = subprocess.run(
        [sys.executable, "-c", CALLS, str(tmp_path / "stack"), *ENGINES],
        cwd=REPO, capture_output=True, text=True, check=True,
        env={"HOME": str(tmp_path / "home"), "PATH": "/usr/bin:/bin",
             "LANGSLICE_RECENT_JOBS": "off", "LANGSLICE_SETTINGS": "off"},
    )
    result = json.loads(done.stdout.splitlines()[-1])
    assert result["codes"] == [0, 0, 0, 0], done.stderr
    assert result["loaded"] == []
