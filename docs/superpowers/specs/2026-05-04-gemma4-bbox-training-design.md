---
title: Gemma 4 E4B — BBox Training Data Design
date: 2026-05-04
scope: Multi-section, region-grounded morphology captions assembled programmatically with hemisphere-split bounding boxes and captioned in bulk via the Gemini 3.1 Pro Batch API. Companion to the agent-trace SFT data; together they constitute the entire active SFT corpus.
status: draft v1
supersedes: bucket 4 of `docs/superpowers/specs/2026-04-25-gemma4-sft-data-design.md`. Buckets 2, 3, and 5 of that document are deferred.
---

# Gemma 4 E4B — BBox Training Data Design

> **STATUS (2026-05-07):** Design complete; implementation **deferred past the
> 2026-05-18 hackathon**. v1 ships single-slice agent-trace SFT only (see
> `docs/superpowers/specs/2026-05-05-gemma4-sft-training-design.md`). Resume
> from Task 1 of `docs/superpowers/plans/2026-05-04-gemma4-bbox-training.md` if
> bucket 4 is reactivated for v2. Do not start work from this spec without
> explicit user request.

## 1. Goal

Produce a single category of supervised data — **multi-section morphology captions, region-grounded by per-section hemisphere-split bounding boxes** — that teaches the Gemma 4 E4B model task-relevant anatomical vocabulary by example. Together with the agent-trace SFT bucket (still per the parent spec), this is the complete active SFT corpus for the langslice-gemma-4 model.

The captions describe how a named anatomical region transforms across a short series of brain sections — its shape, size, boundary characteristics, and any visible sub-divisions. Captions are produced by Gemini 3.1 Pro under tightly-scaffolded prompts, then filtered before training.

## 2. Scope

**In scope:** example shape, source corpus, per-section bbox computation, the assembly orchestrator, QC integration, Gemini Batch API submission path.

**Out of scope:** agent-trace SFT (covered by the parent spec's bucket 1), RLVR data and rewards, training hyperparameters, evaluation system (SliceBench), buckets 2/3/5 of the original spec.

## 3. Example shape

Each example is a single-turn supervised pair:

- **System instruction:** persona ("expert neuroanatomist describing brain section morphology to a student") + the no-mm/no-section-index/no-coord rule.
- **User content:** N section images (N ∈ {4, 5, 6, 7, 8}) followed by a text part containing atlas name, plane, region name, ordering axis ("anterior to posterior" / "medial to lateral" / "dorsal to ventral"), and per-section bounding boxes in image-pixel coordinates.
- **Assistant content:** Gemini 3.1 Pro free-form caption, 2–6 sentences, mm-stripped, region-mention-checked.

The bboxes are **teacher-side scaffolding only.** Gemma sees them at training time as part of the user prompt, but at deployment Gemma never receives bboxes — bucket-1 agent traces and the deployed harness operate without them. The role of this category is auxiliary visual-anatomical vocabulary, not a deployment behavior.

### 3.1 Bbox structure

- **Coronal whole-brain section** or **horizontal whole-brain section:** `{"left": [x1,y1,x2,y2], "right": [x1,y1,x2,y2]}`. Both sides must be non-null. If the region is absent on one side (asymmetric coverage, partial section, oblique cut), the **whole example is dropped at sampling time** rather than emitted with a null side.
- **Sagittal section** or any section with `is_hemisphere=true`: single `[x1,y1,x2,y2]`. Must be non-null; example dropped if absent.

Bbox coordinates are in the **post-resize frame** that's actually base64'd into the request (default 768px max edge, aspect-preserving). Gemini sees image and coordinates in the same frame.

### 3.2 Section count and spacing

- **Section count N:** uniform random over `{4, 5, 6, 7, 8}`.
- **Spacing between adjacent sections:** each of the `N - 1` gaps is sampled independently from a continuous uniform distribution over `[0.2, 0.8]` mm. A 6-section example has 5 independent gaps, and those gaps need not be equal.
- **Anchor position within the region's mm extent:** uniform random subject to the constraint that the full N-section span fits inside the region's visible mm range. Span is `sum(gaps_mm)`, so the minimum span is `(4 - 1) × 0.2 mm = 0.6 mm` and the maximum span is `(8 - 1) × 0.8 mm = 5.6 mm`. Reject and re-sample if it doesn't fit; if the region's extent is too narrow to support even the minimum span, drop that (atlas, orientation, region) tuple from real-histology generation.
- **For real-histology sources:** snap to the nearest available section position in the chosen brain (real brains have fixed sectioning intervals, often 50–150 µm). The spacing target becomes "≥ 0.2 mm and ≤ 0.8 mm between adjacent picks" using actual sections.

### 3.3 Single source type per example

All N sections within one example share a single source type — one brain for real, one modality for augmented atlas, plain reference grayscale for the reference path. Mixing source types within an example would muddy the "what changed across positions" signal.

## 4. Source corpus

Three source types in priority order. The orchestrator picks per (atlas, orientation, region) tuple; corpus distribution falls out of dataset reality, not pre-committed quotas.

### 4.1 Real histology — Tier A only

Only the real-histology subset with **full per-section nonlinear registration** (VisuAlign markers in the original deposit, recoverable via the `_local/eval/lib/registration.py` triangle-warp helper) is used. OUV-affine-only datasets ("Tier B" in the registration audit) are excluded — their few-hundred-µm bbox drift is not worth the marginal coverage.

The Tier A corpus is enumerated in `_local/eval/data_inventory.md` (gitignored). At this spec's writing it covers ~17 brains across mouse coronal, rat coronal, and rat horizontal — roughly 1,100 sections. Mouse sagittal, mouse horizontal, and developmental-mouse have no Tier A coverage and fall through to atlas sources for those (atlas, orientation) combinations.

Per-brain caps: **≤3 examples drawn from any single (brain, region) pair**, to prevent specimen-specific overfitting.

### 4.2 Augmented atlas

Fallback when real coverage is thin or absent. Reuses the existing `synth_dataset.SynthIterator` and `models/langslice-gemma-4/data/augmentation/` modality pipelines — DAPI, Nissl, brightfield, fluorescence, ISH — at the modality weights already configured there. `SynthIterator.__next__()` returns `(image, metadata)`; bbox code obtains the matching annotation slice from `get_oblique_slice()`, which returns `(ref_u8, ann_i32)`.

Each example renders all N sections at the same modality and mode, varying only position along the chosen axis. This keeps the morphology-across-position signal isolated from modality variation.

**Geometric augmentation is disabled for the bbox bucket** (`SynthSpec.apply_geometry_warp = False`). Bboxes are computed from the unwarped atlas annotation slice, so any pixel-displacing transform on the rendered image (`BladeStretchHorizontal`, `AffineJitter`, `Folds`, `Tears`, `Microbubbles`) would desynchronize the saved image from its label. Non-coord realism transforms (`IlluminationGradient`, `EmbeddingHalos`, `Debris`, `ResolutionShift`, plus all tonal/texture stages) still run. Geometric robustness is taught through other SFT buckets, where the target is not a coordinate.

### 4.3 Reference grayscale atlas

Plain BrainGlobe reference volume slice for the chosen atlas, plane, and position — no augmentation. Useful as a "modality-neutral" anchor for vocabulary that should generalize across stains.

### 4.4 Source-priority logic

For each (atlas, orientation, region) tuple:

1. **Tier A real eligible** if the registration-projected coverage index reports any brain with ≥4 sections containing the resolved region IDs (within the 1%–40% area gate). When eligible, sample with replacement weighted by per-brain coverage, capped at 3 per (brain, region).
2. **Augmented atlas** drawn for the remainder, modality and mode picked per the existing weights.
3. **Reference grayscale** added as a parallel slice of the corpus alongside augmented (not a strict fallback — both are produced from atlas, with the corpus split between them).

No fixed real-vs-atlas ratio. The user has confirmed real examples should remain a small fraction of the bucket.

## 5. Region resolution

User-curated landmark list lives at `models/langslice-gemma-4/data/landmarks.json` — per-orientation lists of human-readable landmark names (with aliases where the curator listed them).

Each landmark name is resolved to BrainGlobe atlas region IDs via a hand-built mapping at `models/langslice-gemma-4/data/landmark_atlas_map.json`:

```json
{
  "Hippocampal Formation": {
    "allen_mouse_25um": {"acronym": "HPF", "include_descendants": true},
    "whs_sd_rat_39um":  {"acronym": "HF",  "include_descendants": true}
  }
}
```

`include_descendants: true` walks the BrainGlobe structure tree, returning all sub-region IDs.

The mapping is built semi-automatically: a one-shot helper script does fuzzy-match against each atlas's structure tree, emits a draft, and flags ambiguous entries for the curator to resolve. After that initial pass, the JSON is hand-maintained.

## 6. Bbox computation

Two backends in `models/langslice-gemma-4/data/region_bbox.py`:

### 6.1 Atlas-slice path

`SynthIterator.__next__()` returns `(image, metadata)`. The annotation slice for bbox computation comes from `get_oblique_slice()`, which returns `(ref_u8, ann_i32)`. For each pixel where `ann_i32 ∈ region_ids`, take min/max → bbox. For coronal/horizontal whole-brain atlas-path bboxes, split left/right using the BrainGlobe `atlas.hemispheres` annotation volume projected into the same slice frame; do not use an `image_width // 2` heuristic. Coverage gate: total qualifying pixels (both sides combined) must be ≥1% and ≤40% of image area, else the bbox is reported as failed.

**Dense-core fallback.** When a per-hemisphere bbox would cover more than 10% of the image area, the mask is progressively eroded (`scipy.ndimage.binary_erosion`) and the bbox is taken from the largest connected component, repeating until either the bbox shrinks below 8% of image area or further erosion would empty the mask. This auto-tightens bboxes for thin sprawling regions (e.g. corpus callosum body, posterior hippocampal crescent) without requiring per-region landmark surgery. The 10% trigger / 8% target / 30-iteration cap are constants in `region_bbox.py`. Empirically: HPF (mouse) full-bbox 45% → dense-core 6%; HF (rat) full-bbox 21% → dense-core 9%.

Bbox failure on any required side (both sides for whole-brain coronal/horizontal; single side for sagittal/hemisphere) causes the assembler to drop the entire example, not just the offending section. Examples never reach the draft manifest with null bboxes.

### 6.2 Real-histology path

Uses `_local/eval/lib/registration.py::section_to_atlas_voxel(slice_record, atlas)`, which returns `pixel_to_voxel(i, j) -> (x, y, z)`. For VisuAlign-marker sections this is a section-space Delaunay triangulation over markers + 4 corners, barycentric warp, then OUV affine. For OUV-only sections it falls back to the affine.

Algorithm:

1. Sample a probe grid at `grid_step` pixels (default 8) across the section image.
2. Project each probe via `pixel_to_voxel`, convert QuickNII `(x_ml, y_ap, z_dv)` order to BrainGlobe annotation `(AP, DV, ML)` order, then look up `atlas.annotation[ap, dv, ml]`.
3. Bucket probes by left/right hemisphere via the projected midline line. (Atlas midline plane projected through the section's anchoring → a line in section pixel space, possibly tilted depending on rotation.)
4. Filter to probes whose annotation is in `region_ids`; compute bbox per side; pad by `grid_step` pixels to compensate for grid coarseness.
5. Apply the same 1%–40% coverage gate. Bbox failure drops the entire example per §6.1.

Sub-pixel precision is not required — the bbox is for visual grounding, not landmark registration.

### 6.3 Two passes of projection

Real-histology bbox computation runs in two passes for performance:

- **Coarse coverage scan** (one-time bake, in `_local/eval/lib/landmark_coverage.py`): for every (brain × landmark) pair, project a coarse probe grid through every section's `pixel_to_voxel`, count region-matching probes, and persist `(brain, landmark) → [section_id list with coverage > threshold]` to `_local/eval/data/landmark_coverage.json`. Re-bake only when the manifest or landmark map changes.
- **Fine bbox computation** (per-example, in `region_bbox.py`): for the small subset of (brain, section, region) triples actually selected by the orchestrator, run the finer probe grid described above and compute the hemisphere-split bbox.

## 7. Module architecture

Layered, with utilities under `_local/eval/lib/` and the pipeline itself under `models/langslice-gemma-4/data/`.

### Local utility layer

| Module | Purpose |
|---|---|
| `_local/eval/lib/registration.py` | `section_to_atlas_voxel(slice_record, atlas) -> Callable[[float, float], np.ndarray]`. Already implemented; smoke-tested. |
| `_local/eval/lib/landmark_coverage.py` | One-time bake: scan manifest, project a coarse probe grid per section, record per-(brain × landmark) section coverage above a threshold. Output: `_local/eval/data/landmark_coverage.json`. |

### Gemma-4 project tree

| Module | Purpose |
|---|---|
| `models/langslice-gemma-4/data/landmarks.json` | User-curated landmark list. **Already created.** |
| `models/langslice-gemma-4/data/landmark_atlas_map.json` | Hand-curated landmark → atlas-IDs lookup with descendant policy. |
| `models/langslice-gemma-4/data/landmarks.py` | Loads the above; resolves names to region-ID sets per atlas. |
| `models/langslice-gemma-4/data/region_bbox.py` | Atlas-slice and real-section bbox computation; hemisphere-split for whole-brain coronal/horizontal. |
| `models/langslice-gemma-4/data/build_bbox_data.py` | Orchestrator (~250 LoC). Two-stage CLI: `--stage sample` (no API) and `--stage submit`. |

### Tests

`tests/test_landmarks.py`, `tests/test_region_bbox.py`, `tests/test_build_bbox_data.py` — the latter end-to-end with a stubbed Batch client. Modules under `_local/` are smoke-tested via the registration audit's overlay PNGs, not unit-tested.

## 8. Pre-QC pipeline + QC integration

### 8.1 Two-stage flow

`build_bbox_data.py --stage sample`:
1. Load landmarks, atlas map, and Tier A coverage index.
2. Enumerate viable (atlas, orientation, region) tuples; drop those whose region's atlas-extent is too narrow to fit even the minimum span `(4 - 1) × 0.2 mm = 0.6 mm`.
3. For each viable tuple, target a configurable number of examples (default ~5 per tuple, distributed proportionally to source coverage; CLI flag `--target-total` controls aggregate volume per §12); for each:
   - Pick source by priority.
   - Sample N, per-gap spacings, and anchor; build the section image set and per-section bboxes.
   - Skip if any required bbox is `None` (region absent in that section).
4. Write `_local/bbox_data/draft_manifest.jsonl` (one example per line) and `_local/bbox_data/draft_overlays/<id>_strip.png` (composite strip with cyan-left / magenta-right bbox overlays for QC).

Stage 1 produces no API calls.

`build_bbox_data.py --stage submit --verdicts <verdicts_path>`:
5. Read draft manifest + verdict file; filter to `verdict == "verify"`.
6. Pack into AI Studio Batch API JSONL; upload via `client.files.upload`; create batch via `client.batches.create`.
7. Poll until completed; retrieve responses; mm-strip filter; write `_local/bbox_data/sft.jsonl`.

### 8.2 QC app

The existing `_local/qc_app/` (originally built for trace-manifest QC) gains a `--mode bbox` switch:

- Loads the draft manifest instead of the trace manifest.
- Renders a horizontal strip of N section images per row, each with **left bbox in cyan, right bbox in magenta** (or single-bbox in cyan for sagittal/hemisphere).
- Shows metadata above the strip: atlas, orientation, region, source type, brain, section positions.
- Existing hotkeys (`y`/`n`/`s`/`j`/`k`/`g`/etc.) carry over.
- Verdicts append to `_local/bbox_data/qc_verdicts.jsonl` with the existing `{id, verdict, ts, note}` schema.

The bbox-overlay rendering is **QC-only**. Gemini receives clean section images plus text bbox coordinates in the prompt; the drawn overlays never reach the API.

## 9. Gemini Batch API path

Backend: AI Studio (`LANGSLICE_GENAI_BACKEND=ai_studio`). The existing `vlm_config.supports_batch_api()` Vertex-only gate is widened as part of this work; AI Studio batch is supported in the `google-genai` SDK via `client.batches.create()` against a file-uploaded JSONL.

### 9.1 Wire format (one line of `request.jsonl`)

```json
{
  "key": "bbox_000123",
  "request": {
    "system_instruction": {
      "parts": [{"text": "<bucket system prompt>"}]
    },
    "contents": [
      {
        "role": "user",
        "parts": [
          {"inline_data": {"mime_type": "image/png", "data": "<base64 sec 1>"}},
          {"inline_data": {"mime_type": "image/png", "data": "<base64 sec 2>"}},
          {"inline_data": {"mime_type": "image/png", "data": "<base64 sec ...>"}},
          {"text": "Atlas: allen_mouse_25um\nPlane: coronal\nRegion: Hippocampal Formation\nSection ordering: anterior to posterior\n\nPer-section bounding boxes (left, right hemisphere) in section pixel coords [x1, y1, x2, y2]:\n  Section 1: left=[120,80,240,220], right=[380,80,500,220]\n  ...\n\nDescribe how this region transforms across the sections — its shape, size, boundary characteristics, and any sub-divisions visible. The bounding boxes are provided so you can locate the region in each section; you do not need to reference them in your description.\n\nDo NOT mention millimeter values, section indices, atlas coordinates, or position numbers. 2-4 sentences."}
        ]
      }
    ],
    "generation_config": {
      "response_modalities": ["TEXT"],
      "temperature": 1.0,
      "thinking_config": {
        "thinking_level": "LOW"
      }
    }
  }
}
```

### 9.2 Configuration choices

- **Model:** Gemini 3.1 Pro (`gemini-3.1-pro-preview` at this writing; `gemini-3-pro-preview` was retired on 2026-03-09). The orchestrator reads the model from env or CLI; do not hardcode the version suffix.
- **Temperature:** 1.0. Low temperature degrades Gemini visual matching; this matches the deployed estimation default.
- **`thinking_config.thinking_level`:** `LOW`. Gemini 3.1 Pro does not accept `MINIMAL`; `LOW` is the lowest supported level for this model. Bump to MEDIUM or HIGH if first-batch caption quality is poor.
- **No `max_output_tokens`.** Constraint enforced by the prompt's "2-4 sentences" instruction; verified post-response by the length gate.
- **`response_modalities: ["TEXT"]`** — captioning is text-only.

### 9.3 Sagittal / hemisphere variant

Single-bbox form (no left/right split):

```
Plane: sagittal
Section ordering: medial to lateral

Per-section bounding boxes in section pixel coords [x1, y1, x2, y2]:
  Section 1: [220, 140, 410, 320]
  ...
```

### 9.4 Response extraction

The post-processor walks `response.candidates[0].content.parts`, drops parts where `part.thought == True`, and joins the text from the remaining parts. Thought parts should only appear when `include_thoughts=True` is set, but the filter remains as defensive handling so accidental thought-part exposure cannot enter the SFT assistant content.

## 10. Validation gates

### Pre-submission (per-example)

- All required bboxes computed without errors. For whole-brain coronal/horizontal, both `left` and `right` must be non-null on every section (drop the example if any side fails on any section). For sagittal or hemisphere sections, the single bbox must be non-null.
- Coverage gate: bbox area ≥1% and ≤40% of section image area.
- QC verdict = `verify`. Examples with `reject` or `skip` are dropped from submission.

### Post-response (per-caption)

- Non-empty text after dropping thought parts.
- mm-strip regex: drop captions matching `\b\d+(\.\d+)?\s*(mm|μm|um|microns?)\b` (case-insensitive), `\b(section|slice)\s*\d+\b`, or `\bbregma\b`.
- Length gate: 2–6 sentences.
- Region-mention gate (warn-only): caption should contain the region name or a recognized synonym; logs a warning rather than rejecting on miss.

### Pre-training (per-corpus)

- Atlas-version field present on every example.
- No subject overlap with the SliceBench eval-holdout list (cross-checked when `source_brain` is set).
- Distribution sanity: per-(atlas, orientation, source-type, region) cell counts are printed; cells expected to be populated must not be empty.

## 11. Outputs

```
_local/bbox_data/
  draft_manifest.jsonl       # pre-QC: examples + bboxes (--stage sample output)
  draft_overlays/            # bbox-drawn composite strips for QC
    bbox_000123_strip.png
    ...
  qc_verdicts.jsonl          # written by the QC app
  request.jsonl              # post-QC submission JSONL
  batch_metadata.json        # batch ID, model, submission time, file ID
  responses.jsonl            # raw Gemini responses keyed by example ID
  sft.jsonl                  # final SFT examples (deliverable)
```

Final SFT shape per line (intermediate format — Gemma 4 chat-template rendering happens in the parent spec's Phase 4):

```json
{
  "id": "bbox_000123",
  "atlas": "allen_mouse_25um",
  "atlas_version": "CCFv3",
  "orientation": "coronal",
  "region": "Hippocampal Formation",
  "source_type": "real_histology",
  "source_brain": "<brain_id>",
  "is_hemisphere": false,
  "messages": [
    {"role": "system", "content": "<system instruction>"},
    {"role": "user", "content": [<image_parts>, {"type": "text", "text": "<bbox prompt>"}]},
    {"role": "assistant", "content": "<filtered Gemini caption>"}
  ]
}
```

The `messages` array uses an OpenAI-style content-block representation as a transport format. Conversion to the Gemma 4 chat template (with image-token placeholders, instruct-masking, and atlas-version metadata) is downstream and shared with bucket-1 agent traces — defined by the parent SFT data spec, not by this document.

## 12. Volume and cost

- **Pilot batch:** 100 examples to verify caption quality before committing to the full run. Mix: ~25 real Tier A, ~50 augmented atlas, ~25 reference grayscale.
- **Full run:** 300–400 examples total (low end of the parent spec's 300–600 range for this category). Expand to ~600 only if first-pass quality is low and more coverage is wanted.
- **Token budget per example:** ~3K–4K tokens (6 images × 768px × ~256–512 tokens each + ~600-token prompt + ~250-token response + LOW-thinking overhead).
- **Cost estimate:** 400 examples ≈ 1.4M tokens. Verify Gemini 3.1 Pro batch pricing on AI Studio at submission time; if the typical batch discount applies, expected **single-digit dollars to low double digits**, well within the Gemini-credits envelope. Re-confirm before launching the full run.

## 13. Risks / open issues

- **Real-corpus is mouse-coronal-heavy.** Mouse sagittal, mouse horizontal, and dev-mouse anything fall through to atlas sources entirely. Atlas augmentation already covers this gap structurally, but real-histology vocabulary diversity is mouse-coronal-biased. Documented; not actively mitigated.
- **Tier B real-histology dropped by choice.** OUV-only datasets exist in the local corpus but are excluded for bucket-quality reasons. Revisit if a marker-only registration round is later added upstream.
- **Region-mapping ambiguity.** Some user-curated landmarks (e.g., composite or boundary-defined entries) don't map cleanly to a single BrainGlobe acronym. The atlas-map fuzzy-match step flags these for hand-resolution; remaining ambiguities are dropped from the bucket rather than approximated.
- **Thinking forced on.** Gemini 3.1 Pro thinking can't be disabled, only set as low as `LOW`. If `LOW` still produces verbose captions or unexpected thought-part leakage when `include_thoughts=True` is used, the post-response thought-filter (Section 9.4) is the second line of defense.
- **Bbox drift from grid coarseness.** Probe grid at 8 pixels misses thin region boundaries. Padded by `grid_step` to compensate, but bboxes systematically over-include by up to one grid step. Acceptable for visual grounding; flag as the failure mode if QC verdicts trend `reject` for "bbox too loose."
- **AI Studio batch SDK gate.** `vlm_config.supports_batch_api()` currently rejects AI Studio. Widening this gate is a small change but a hard prerequisite — must be done before `--stage submit`.
- **Subject leakage into SliceBench.** SliceBench eval-holdout list must exist before submission. Cross-check is a hard gate (Section 10).

## 14. Dependencies

| Dependency | Status | Owner |
|---|---|---|
| Curated landmark list | DONE (`models/langslice-gemma-4/data/landmarks.json`) | User |
| `landmark_atlas_map.json` (atlas-ID lookup) | TODO — fuzzy-match draft + curator review | Implementation |
| `_local/eval/lib/registration.py` (pixel→voxel helper) | DONE — built and smoke-tested | Audit agent |
| `_local/eval/lib/landmark_coverage.py` (coverage index) | TODO | Implementation |
| Tier A real-histology corpus | DONE (enumerated in `_local/eval/data_inventory.md`) | User + downloaders |
| AI Studio Batch API support in `vlm_config` | TODO — widen `supports_batch_api()` gate | Implementation |
| QC app `--mode bbox` extension | TODO | Implementation |
| SliceBench eval-holdout subject list | In development | User |
| Gemini 3.1 Pro model ID at submission time | Resolve via env / CLI; do not hardcode | Implementation |

## 15. What this design explicitly does not include

- Cross-atlas synthetic queries (atlas A image queried against atlas B annotations).
- Real histology without nonlinear registration (Tier B excluded).
- Free-form bulk image captioning of arbitrary histology — this is a structured, region-conditioned, multi-section task only.
- Bbox as visible mark drawn on the image sent to Gemini (text coordinates only; drawn version is QC-only).
- Mixing source types within a single example.
- Mixing modalities within a single example.
- Cross-validating bboxes against a second model — QC is the validation step.
- Recovering Silva IEG ABBA registration — deferred until a separate session decides whether to invest the re-pull effort, likely under any future bbox-grounding-bucket revival.
- Curriculum ordering of the bbox-data examples within SFT — out of scope; presented uniformly.

## 16. Literature anchors

For traceability:

- Parent spec — `docs/superpowers/specs/2026-04-25-gemma4-sft-data-design.md` §5 row 4 ("Bucket 4 — Multi-slice morphology") and §5.2 (mm-omission rule).
- VisuAlign triangle warp — Tevemadar/UIO `VisuAlign/nonlin/Triangle.java` covers triangle-local barycentric interpolation. Four-corner padding and OUV affine composition are LangSlice-side recovery steps, not part of that public source.
- Gemini 3.1 thinking model — `google-genai` Python SDK `ThinkingConfig.thinking_level: LOW | MEDIUM | HIGH`. Thinking enabled by default; cannot be disabled.
- Spec §11 of the parent on subject-level leakage (Bussola et al., 2019, arxiv 1909.06539).
