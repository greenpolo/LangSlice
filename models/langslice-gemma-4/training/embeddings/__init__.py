"""Atlas vision-embedding precompute + runtime cache.

The atlas tool-result images are heavily reused across SFT examples — the same
``atlas/<atlas>/<plane>/<pos>mm.{png,jpg}`` path appears in many traces. SigLIP
encoding is the dominant per-step cost on the SFT path. This package
precomputes a tensor cache keyed by ``(atlas, plane, snapped_position_mm)`` so
the trainer can look the embedding up by image path instead of re-running
SigLIP every batch the image appears in.

Phase 1 (this commit): precompute + cache + hit-rate measurement only. No
forward-side splice — the collator just counts how often atlas-grid paths
match a cache entry. Run :mod:`tools.embeddings_hit_rate` against a real SFT
corpus to decide whether the splice is worth shipping.

Phase 2 (gated on >50% hit rate): the collator returns a precomputed
embedding sidecar, and a vision-tower forward hook splices cached embeddings
into the model's image-feature tensor before SigLIP runs on the rest. Stays
disabled-by-default behind ``--enable-atlas-embedding-splice`` until the
bit-exact correctness test passes.
"""

from .cache import AtlasEmbeddingCache, parse_atlas_path
from .query_cache import QueryEmbeddingCache, save_query_pair

__all__ = [
    "AtlasEmbeddingCache",
    "QueryEmbeddingCache",
    "parse_atlas_path",
    "save_query_pair",
]
