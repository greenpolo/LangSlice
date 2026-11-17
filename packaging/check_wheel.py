"""Audit the wheel's distribution boundaries without importing LangSlice."""
import configparser
import sys
import zipfile
from pathlib import PurePosixPath


def check(path: str) -> None:
    with zipfile.ZipFile(path) as archive:
        names = set(archive.namelist())
        for name in names:
            parts = PurePosixPath(name).parts
            if (not parts or ".." in parts or name.startswith("/")
                    or not (parts[0] in {"langslice", "langslice_vendor"}
                            or parts[0].endswith(".dist-info"))):
                raise ValueError(f"Unexpected wheel path: {name}")
            if any(part.startswith(".env") or part in {"__pycache__", "openai_auth.json"}
                   for part in parts):
                raise ValueError(f"Private or generated file in wheel: {name}")
        required = {
            "langslice/doors/credentials.py", "langslice/tools/bookkeeping/abba_export.py",
            "langslice/hosts/gui/static/index.html",
            "langslice/hosts/gui/static/ui.css", "langslice/hosts/gui/static/ui.js",
            # the vendored engines' licences ship with their code
            "langslice/engines/deepslice/LICENSE", "langslice/engines/brainways/LICENSE",
            "langslice_vendor/stalign/LICENSE", "langslice_vendor/emlddmm/LICENSE",
            "langslice_vendor/emlddmm/NOTICE",
        }
        if not required.issubset(names):
            raise ValueError(f"Missing runtime files: {sorted(required - names)}")
        entry_files = [name for name in names if name.endswith(".dist-info/entry_points.txt")]
        if len(entry_files) != 1:
            raise ValueError("Expected one entry-point metadata file")
        parser = configparser.ConfigParser()
        parser.read_string(archive.read(entry_files[0]).decode())
        if parser["console_scripts"].get("langslice") != "langslice.cli:main":
            raise ValueError("Missing langslice console entry point")
    print(f"Verified {path}: {len(names)} package files")


if __name__ == "__main__":
    check(sys.argv[1])
