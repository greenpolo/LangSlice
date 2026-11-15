"""abba-mcp with the LangSlice Fiji connector in its Fiji.

abba-mcp starts a pinned ABBA Fiji from its env/jgo.toml. This wrapper copies
that environment, adds org.langslice:langslice-fiji (installed in ~/.m2 with
`mvn install` in LangSlice/connectors/fiji) after ABBA, and runs abba-mcp on it,
so ABBA's Register > LangSlice... is in the Fiji the assistant drives.

Run: uv run --no-project --with abba-mcp==0.25.0.post2 python abba_mcp_langslice.py
"""
import re
import shutil
import sys
from pathlib import Path

import abba_mcp

CONNECTOR = "org.langslice:langslice-fiji:0.3.0"
env = Path.home() / ".abba-mcp" / "env-langslice"
env.mkdir(parents=True, exist_ok=True)
for f in abba_mcp.ENV.iterdir():
    if f.is_file():
        shutil.copy(f, env / f.name)
toml = (env / "jgo.toml").read_text(encoding="utf-8")
if CONNECTOR not in toml:
    toml = re.sub(r'(\s*"sc\.fiji:fiji:[^"]+",)', rf'\1\n    "{CONNECTOR}",', toml, count=1)
    assert CONNECTOR in toml, "could not add the connector to jgo.toml"
(env / "jgo.toml").write_text(toml, encoding="utf-8")
abba_mcp.ENV = env

sys.exit(abba_mcp.main())
