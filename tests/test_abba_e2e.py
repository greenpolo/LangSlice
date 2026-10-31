"""The Fiji connector end to end: real headless ABBA, the real worker, the agent CLI.

Runs ``connectors/fiji/build.py --test`` (the offline Java smoke checks), then
``E2EFollowCheck``: the connector prepares a job from two synthetic sections,
follows it, and ``connectors/fiji/e2e/driver.py`` writes to it through
``langslice-job`` as an agent app would; ABBA must show the job, and a
reopened project attached again must come back in one sync step.

Opt in with ``LANGSLICE_ABBA_E2E=1``. Needs ``LANGSLICE_FIJI_JDK`` (a JDK 21
home), ABBA's cached Allen V3p1 atlas in ``LANGSLICE_ABBA_ATLAS_CACHE``
(default ``~/cached_atlas``) and the connector's resolved Maven classpath
(``connectors/fiji/target/test-classpath.txt``, written once by ``mvn
package dependency:build-classpath -Dmdep.outputFile=target/test-classpath.txt``).
"""

from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

import pytest

pytestmark = pytest.mark.slow

ROOT = Path(__file__).resolve().parents[1]
FIJI = ROOT / "connectors" / "fiji"


def _settings() -> tuple[Path, Path]:
    if os.environ.get("LANGSLICE_ABBA_E2E") != "1":
        pytest.skip("set LANGSLICE_ABBA_E2E=1 to run the ABBA end-to-end check")
    jdk = os.environ.get("LANGSLICE_FIJI_JDK")
    if not jdk or not (Path(jdk) / "bin" / "java").is_file():
        pytest.skip("LANGSLICE_FIJI_JDK must name a JDK 21 home")
    cache = Path(os.environ.get("LANGSLICE_ABBA_ATLAS_CACHE", "~/cached_atlas")).expanduser()
    if not (cache / "mouse_brain_ccfv3p1.xml").is_file():
        pytest.skip(f"no cached Allen V3p1 atlas in {cache}")
    if not (FIJI / "target" / "test-classpath.txt").is_file():
        pytest.skip("resolve the connector's Maven classpath first (connectors/fiji/README.md)")
    return Path(jdk), cache


def test_connector_follows_a_job_end_to_end(tmp_path: Path) -> None:
    jdk, cache = _settings()
    subprocess.run([sys.executable, str(FIJI / "build.py"), "--java-home", str(jdk), "--test"],
                   check=True, timeout=900)
    # The worker's environment: this interpreter on this checkout's sources. The connector
    # clears PYTHONPATH for its workers, so the stand-in sets it itself.
    prefix = tmp_path / "env"
    python = prefix / "bin" / "python"
    python.parent.mkdir(parents=True)
    python.write_text(f'#!/bin/sh\nPYTHONPATH="{ROOT / "src"}" exec "{sys.executable}" "$@"\n')
    python.chmod(0o755)
    classes = os.pathsep.join([
        str(FIJI / "target" / "test-classes"), str(FIJI / "target" / "classes"),
        (FIJI / "target" / "test-classpath.txt").read_text().strip(),
    ])
    done = subprocess.run(
        [str(jdk / "bin" / "java"), "-Djava.awt.headless=true",
         "--add-opens=java.base/java.lang=ALL-UNNAMED",
         f"-Djava.util.prefs.userRoot={tmp_path / 'prefs'}", "-cp", classes,
         "org.langslice.fiji.E2EFollowCheck", str(cache), str(prefix),
         str(FIJI / "e2e" / "driver.py"), str(tmp_path / "out")],
        capture_output=True, text=True, timeout=1800, check=False)
    report = "\n".join(line for line in done.stdout.splitlines()
                       if line.startswith(("E2E", "DRIVER")))
    assert done.returncode == 0 and "E2E_PASS" in done.stdout, (
        report + "\n" + done.stderr[-4000:])
