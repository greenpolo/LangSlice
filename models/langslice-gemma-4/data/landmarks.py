"""Landmark loader and atlas-region resolver for the BBox training data pipeline.

Loads the curated landmark list (`landmarks.json`) and the per-atlas mapping
(`landmark_atlas_map.json`); resolves each landmark to a set of BrainGlobe
region IDs, walking the structure-tree descendants when the mapping flags
`include_descendants: true`.
"""
from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

_DEFAULT_LANDMARKS_PATH = Path(__file__).resolve().parent / "landmarks.json"
_DEFAULT_ATLAS_MAP_PATH = Path(__file__).resolve().parent / "landmark_atlas_map.json"


@dataclass(frozen=True)
class LandmarkLoader:
    landmarks_path: Path = _DEFAULT_LANDMARKS_PATH
    atlas_map_path: Path = _DEFAULT_ATLAS_MAP_PATH

    def _load_landmarks(self) -> dict:
        return json.loads(self.landmarks_path.read_text(encoding="utf-8"))

    def _load_atlas_map(self) -> dict:
        if not self.atlas_map_path.exists():
            return {}
        return json.loads(self.atlas_map_path.read_text(encoding="utf-8"))

    def landmarks_for_orientation(self, orientation: str) -> list[str]:
        payload = self._load_landmarks()
        try:
            entries = payload["landmarks_by_orientation"][orientation]
        except KeyError as exc:
            raise KeyError(
                f"Unknown orientation {orientation!r}; expected one of "
                f"{list(payload['landmarks_by_orientation'].keys())!r}."
            ) from exc
        return [entry["name"] for entry in entries]

    def resolve(
        self, landmark_name: str, atlas: object, atlas_name: str
    ) -> set[int]:
        """Resolve a landmark name to BrainGlobe region IDs for the given atlas.

        Walks the atlas structure-tree descendants when the mapping flags
        `include_descendants: true`. Returns an empty set if the landmark is
        unmapped for this atlas — the orchestrator drops unmapped tuples.
        """
        atlas_map = self._load_atlas_map()
        entry = atlas_map.get(landmark_name)
        if entry is None:
            return set()
        per_atlas = entry.get(atlas_name)
        if per_atlas is None:
            return set()

        acronym = per_atlas["acronym"]
        include_descendants = bool(per_atlas.get("include_descendants", False))

        structures = getattr(atlas, "structures", {})
        lookup_df = getattr(atlas, "lookup_df", None)

        def _structure_for(ac: str) -> dict | None:
            try:
                return structures[ac]
            except Exception:
                pass
            if lookup_df is not None:
                matches = lookup_df.loc[lookup_df["acronym"] == ac]
                if not getattr(matches, "empty", True):
                    row = matches.iloc[0]
                    return {"acronym": str(row["acronym"]), "id": int(row["id"])}
            return None

        ids: set[int] = set()
        root_structure = _structure_for(acronym)
        if root_structure is not None:
            ids.add(int(root_structure["id"]))
        if include_descendants and root_structure is not None:
            for descendant_acronym in atlas.get_structure_descendants(root_structure):
                descendant = _structure_for(descendant_acronym)
                if descendant is not None:
                    ids.add(int(descendant["id"]))
        return ids
