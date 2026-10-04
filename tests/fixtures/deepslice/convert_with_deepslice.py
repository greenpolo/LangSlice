"""Write a LangSlice QuickNII export's anchorings with DeepSlice 1.2.8's own writers.

The DataFrame has the columns DSModel.predict builds (neural_network.predictions_util:
Filenames, ox..vz; then width, height and nr), and is saved as
DSModel.save_predictions does (to_csv(index=False), write_QUINT_JSON), plus
write_QuickNII_XML. The fixtures here were written from
tests/test_import_registrations.py's deepslice_case export with DeepSlice 1.2.8
(the PyPI release current on 2026-10-04):

    python convert_with_deepslice.py langslice_quicknii.json deepslice
"""
import json, os, sys
import pandas as pd
from DeepSlice.read_and_write import QuickNII_functions as q
from DeepSlice.coord_post_processing import spacing_and_indexing
src, out = sys.argv[1], sys.argv[2]
doc = json.load(open(src))
rows = doc["slices"]
df = pd.DataFrame({"Filenames": [r["filename"] for r in rows],
                   **{k: [r["anchoring"][i] for r in rows] for i, k in enumerate(["ox","oy","oz","ux","uy","uz","vx","vy","vz"])}})
df["width"] = [int(r["width"]) for r in rows]
df["height"] = [int(r["height"]) for r in rows]
df["nr"] = spacing_and_indexing.number_sections(df["Filenames"], False)
df["nr"] = df["nr"].astype(int)
df = df.sort_values(by="nr").reset_index(drop=True)
config = json.load(open(os.path.join(os.path.dirname(q.__file__), "..", "metadata", "config.json")))
target = config["target_volumes"]["mouse"]["name"]
aligner = config["DeepSlice_version"]["prerelease"]
df.to_csv(out + ".csv", index=False)
q.write_QUINT_JSON(df=df, filename=out, aligner=aligner, target=target)
q.write_QuickNII_XML(df, out, aligner)
