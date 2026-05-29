"""Physically-based DAPI microscopy simulation for synthetic brain-slice histology.

Builds a 3-D fluorophore-count ground truth (cell nuclei placed by tissue type)
and images it through the lab's Keyence BZ-X800 optics (microsim PSF + detector).

This ``__init__`` is intentionally import-free: importing ``microscope`` must not
pull heavy/GPU dependencies. Import the specific module you need, e.g.
``from microscope import white_matter`` (CPU) or ``microscope.render_frame`` (GPU).
"""
