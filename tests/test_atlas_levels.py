"""Region levels: every id at one depth of the ontology."""

from types import SimpleNamespace

import numpy as np
import pytest

from langslice.core.atlas.levels import labels_at


def _atlas():
    paths = {1: [1], 2: [1, 2], 3: [1, 2, 3], 4: [1, 2, 3, 4], 5: [1, 5]}
    return SimpleNamespace(structures={
        uid: {"id": uid, "structure_id_path": path} for uid, path in paths.items()})


def test_each_id_takes_its_ancestor_at_the_depth():
    labels = np.array([[0, 4, 3], [2, 5, 99]], dtype=np.uint32)
    atlas = _atlas()
    at_depth = labels_at(labels, atlas, 1)
    # 4 and 3 sit under 2; 2 and 5 are at depth 1; an unlisted id and 0 stay.
    assert at_depth.tolist() == [[0, 2, 2], [2, 5, 99]]
    assert at_depth.dtype == labels.dtype
    assert labels_at(labels, atlas, 2).tolist() == [[0, 3, 3], [2, 5, 99]]
    assert labels_at(labels, atlas, 7).tolist() == labels.tolist()
    assert labels_at(labels, atlas, 0).tolist() == [[0, 1, 1], [1, 1, 99]]


def test_a_depth_is_a_whole_number_zero_or_more():
    with pytest.raises(ValueError):
        labels_at(np.zeros((2, 2), dtype=np.int32), _atlas(), -1)
