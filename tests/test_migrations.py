"""The table of migrations that reads old job folders as the current format."""

from __future__ import annotations

from pathlib import Path

import pytest

from langslice.job.migrations import STATE_MIGRATIONS, FolderMigration
from langslice.job.state_file import STATE_FORMAT_VERSION, current_state


def test_the_table_runs_from_its_oldest_format_to_the_current_one():
    formats = [row.reads for row in STATE_MIGRATIONS]
    assert formats == list(range(formats[0], STATE_FORMAT_VERSION))
    assert len({row.name for row in STATE_MIGRATIONS}) == len(STATE_MIGRATIONS)


@pytest.mark.parametrize("version", [0, 2])
def test_a_format_older_than_the_table_is_refused(tmp_path: Path, version: int):
    # no workspace or folder: the format is refused before either is read
    migration = FolderMigration(workspace=None, layout=None)
    with pytest.raises(ValueError, match="older pre-release LangSlice"):
        current_state({"format_version": version, "slices": []}, migration)
    assert not migration.used
