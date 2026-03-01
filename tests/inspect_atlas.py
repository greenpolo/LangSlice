import brainglobe_atlasapi
from brainglobe_atlasapi import BrainGlobeAtlas

atlas = BrainGlobeAtlas('kim_mouse_isotropic_20um')
print("Atlas:", atlas.atlas_name)
print("Metadata:", atlas.metadata)

print("\nAttributes:")
for k in dir(atlas):
    if not k.startswith('_'):
        print(f" - {k}")

print("\nPaths:")
print("Local path:", atlas.local_full_name)
import os
for f in os.listdir(atlas.root_dir / atlas.atlas_name):
    print(f" - {f}")

print("\nAdditional references dictionary:", atlas.additional_references)

