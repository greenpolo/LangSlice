"""The QUINT export's target grid."""

import numpy as np


def test_finer_and_coarser_allen_atlases_export_to_the_25um_target() -> None:
    """QuickNII and VisuAlign ship the Allen CCFv3 at 25 um only
    (``ABA_Mouse_CCFv3_2017_25um.cutlas``, the target DeepSlice writes too):
    a 10, 50 or 100 um job is exported to it, its anchoring in that
    target's voxel space, the same physical plane as a 25 um job's."""
    from langslice.job.quint import job_export

    def facts(res: float) -> dict:
        return {"name": f"allen_mouse_{int(res)}um", "orientation": "asr",
                "shape": [round(13200 / res), round(8000 / res), round(11400 / res)],
                "resolution_um": [res] * 3}

    def pixel_map(res: float) -> np.ndarray:
        # A physical (CCF, voxel-edge) placement, in each atlas's own
        # BrainGlobe micrometres (voxel i's centre at i * res: CCF - res / 2).
        physical = np.array([[0.0, 0.0, 6000.0], [20.0, 0.0, 2000.0], [0.0, 20.0, 1500.0]])
        physical[:, 2] -= res / 2.0
        return physical

    reference = job_export([{"filename": "a.png", "width": 300, "height": 200, "nr": 1,
                             "pixel_to_atlas_um": pixel_map(25.0)}], facts(25.0))
    for res in (10.0, 50.0, 100.0):
        exported = job_export([{"filename": "a.png", "width": 300, "height": 200, "nr": 1,
                                "pixel_to_atlas_um": pixel_map(res)}], facts(res))
        assert exported["target"] == "ABA_Mouse_CCFv3_2017_25um.cutlas"
        np.testing.assert_allclose(exported["slices"][0]["anchoring"],
                                   reference["slices"][0]["anchoring"], atol=1e-4)
