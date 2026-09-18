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
                    or not (parts[0] == "langslice" or parts[0].endswith(".dist-info"))):
                raise ValueError(f"Unexpected wheel path: {name}")
            if any(part.startswith(".env") or part in {"__pycache__", "openai_auth.json"}
                   for part in parts):
                raise ValueError(f"Private or generated file in wheel: {name}")
        required = {
            "langslice/api/service.py", "langslice/api/setup.py",
            "langslice/api/abba_worker.py", "langslice/integrations/static/chat.html",
            "langslice/integrations/static/chat.css", "langslice/integrations/static/chat.js",
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
