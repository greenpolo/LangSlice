"""``atlas.details``: BrainGlobe's atlas descriptions, as the job statement quotes them."""

from __future__ import annotations

from langslice.core.atlas.details import atlas_description, parse

PAGE = """# Available atlases
## Mouse
### [Toy Atlas](https://doi.org/x)
A toy atlas. The template is [Nissl](https://n) stained.

Available versions:
* `toy_mouse_10um` - 10um resolution [![badge](https://b)](https://ng)
* `toy_mouse_MRI-T2_25um` - MRI T2 template

### [Rat Atlas](https://doi.org/y)
An MRI rat atlas.
This atlas is only available at 39um resolution:
* `rat_39um`
"""


def test_each_listed_version_gets_its_sections_paragraph():
    found = parse(PAGE)
    assert found["toy_mouse_10um"] == ("Toy Atlas: A toy atlas. The template is Nissl stained. "
                                       "(This version: 10um resolution.)")
    assert found["toy_mouse_mri-t2_25um"].endswith("(This version: MRI T2 template.)")
    assert found["rat_39um"] == "Rat Atlas: An MRI rat atlas."


def test_the_packaged_file_describes_the_2011_atlas_and_its_misaligned_reference():
    described = atlas_description("ccfv2_mouse_25um") or ""
    assert "Nissl" in described and "misaligned" in described
    assert atlas_description("no_such_atlas_1um") is None
