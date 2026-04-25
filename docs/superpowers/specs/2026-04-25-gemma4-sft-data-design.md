---
title: Gemma 4 E4B — SFT Data Design
date: 2026-04-25
scope: Supervised fine-tuning data composition, generation pipeline, and augmentation architecture for the langslice-gemma-4 model. RLVR is a separate spec.
status: draft (post-brainstorm, pre-plan)
---

# Gemma 4 E4B — SFT Data Design

## 1. Goal

Produce an SFT corpus that turns Gemma 4 E4B into a drop-in replacement for Gemini inside the LangSlice estimation agent loop (`fetch_atlas` + `submit_estimate` / `submit_group_estimate` from `src/langslice_harness/harness/estimation/tools.py`), as the warm-start before a separate, larger RLVR phase.

Two purposes for the SFT data:

1. Teach task-relevant anatomical vocabulary the base model lacks.
2. Show the model what tools are available and how to use them, by example.

## 2. Scope

In scope:

- Data composition (what buckets, what shape, what volume).
- Generation pipeline (build order, dependencies, where LLM calls happen).
- Augmentation pipeline (atlas-image transforms reused for SFT and later RLVR).

Out of scope (separate brainstorms):

- RLVR data, reward function, training schedule.
- SFT training hyperparameters and Unsloth/QLoRA configuration.
- Evaluation system (SliceBench, currently in development).

## 3. Coverage axes

- **Species / atlases:** `allen_mouse_25um`, `whs_sd_rat_39um`, `admba_*` (developmental mouse). Three, reflecting actual public-data availability per `eval/dataset_allocation.md`. Other priority species (prairie vole, mole-rat, mouse lemur) have effectively zero registered public tissue and are excluded from SFT.
- **Orientations:** coronal / sagittal / horizontal. Coronal-heavy with substantial sagittal and minimal horizontal expected; actual mix follows real-data collection rather than a pre-committed split.
- **Modalities (real data):** fluorescence + brightfield + Nissl + ISH where available, per `eval/data_inventory.md`.
- **Resolution mismatch:** include examples where the query is rendered at one µm rate and `fetch_atlas` returns slices at another (e.g., 10 µm query → 50 µm atlas, and the inverse). Discourages shortcut-learning on a fixed query/atlas resolution pairing.

## 4. Budget envelope

Total: roughly 5K–30K SFT examples. This is the literature-anchored range for "light cold-start SFT on a new visual domain" before RLVR (DeepSeek-R1 cold-start, vision-RLVR warm-starts in 2025). Smaller and cleaner is preferred over the upper bound.

Hard inputs to the budget:

- Gemini API credits: ~$83 → ~500 high-quality Gemini 3.1 Pro trace runs.
- Cheaper teacher models: another ~500–2K trace runs (Flash, OpenAI-compatible models).
- Auxiliary anatomy data (buckets 2–4 below): cheap to scale programmatically; bounded by user landmark-list curation and ~1K Gemini-distilled bboxes on real histology.

## 5. Data composition — four buckets

| # | Bucket | Purpose | Source | Output shape | Approx volume |
|---|---|---|---|---|---|
| 1 | Agent traces | Teach tool-use loop, mm scales per atlas, sweep heuristics, narrow/fine-tune reasoning | Gemini 3.1 Pro (~500 runs) + cheaper teachers (~500–2K runs), filtered to ~40–60% accept rate | Multi-turn: visible reasoning + `fetch_atlas` calls + atlas-image tool results + `submit_estimate` / `submit_group_estimate` | ~500–1.5K traces |
| 2 | Landmark listing | Teach loop's first-turn contract: "describe 2–3 prominent anatomical landmarks" | Programmatic from atlas annotations (top-N visible regions by area, named) | Single-turn: image → "anterior commissure visible, hippocampus forming, …" | A few thousand |
| 3 | Bbox grounding | Visual-spatial grounding of curated landmarks; supports future LangSlice bbox-based registration | Programmatic from atlas registration; ~1K Gemini-distilled on partial-registration real histology with user review | Both directions: (image + region name → bbox), (image + bbox → region name) | A few thousand programmatic + ≤1K distilled |
| 4 | Multi-slice morphology | Teach how regions transform across positions — supports the loop's narrowing reasoning | Gemini Pro distilled (text quality matters) | (6 atlas sections + region) → "this region begins as a thin crescent and broadens caudally…" | Several hundred |

### 5.1 Format-alignment principle

Every example's *output* shape matches what Gemma will emit at deployment. Bucket 2 outputs landmark lists because the loop's first turn does. Bucket 4 outputs prose morphology descriptions because that's the form of reasoning the model will produce while narrowing. Bucket 3 is the one exception that produces an output shape (bbox) the agent loop does not currently use; it is retained because (a) bbox grounding is an established transfer-positive auxiliary task in VLM literature (Qwen-VL, KOSMOS-2, GLaMM) and (b) future LangSlice versions will use bbox output for registration.

### 5.2 Coordinate leakage rule

The deployment loop is millimeter-native: `fetch_atlas` takes mm, tool results say "Atlas at 5.40 mm", and the final answer is `position_mm`. Removing mm from training data is incompatible with the deployment task. Therefore:

- Buckets 1 (traces) and 3 (bbox grounding) keep mm where it appears naturally — in tool calls, tool results, and final answers.
- Bucket 4 (multi-slice morphology) **must omit mm from prompts and outputs** — this bucket is the one place where the model could otherwise learn "section index ↔ mm" maps. The 6 sections are described positionally ("anterior to posterior") rather than by mm.
- Bucket 2 (landmark listing) is single-image and naturally mm-free.

## 6. Query-source mix (applies to bucket 1)

Three query sources, mixed without a hard percentage commitment:

- **Real histology** — prioritized wherever available. Quantity bounded by `eval/data_inventory.md` collection.
- **Direct atlas slices** — clean atlas slice as query against the same atlas. Trains tool format, mm-scale-per-atlas, and broad-sweep heuristics. Trivial visual matching.
- **Augmented atlas slices** — atlas slices passed through the augmentation pipeline (DAPI-mimic, Nissl-mimic, etc.). Pulls weight on visual matching as well as format.

Cross-atlas synthetic queries (slice from atlas A used against atlas B) are explicitly excluded — silver-standard labels from another model are too noisy.

## 7. Defense against atlas-coordinate memorization

The risk: if 80%+ of real histology is registered to Allen CCFv3, the model may learn "this image looks like → AP X.X mm in Allen mouse" as a memorization shortcut rather than a visual-matching skill. Defenses, in order of importance:

1. **Atlas distribution diversity in synthetic queries.** Direct- and augmented-atlas queries are spread across all three priority atlases, balancing the Allen-CCF dominance of real-data.
2. **Loop structure.** The agent must compare the query image to the fetched atlas images present in the same context — visual matching, not memorization, is what produces the correct tool calls. The loop's design carries most of the structural defense.
3. **Resolution mismatch.** Mixing query/atlas µm rates within and across atlases prevents memorization of fixed appearance/coordinate pairs.

Cross-atlas coordinate conversion tooling is not built. Each example is atlas-internal; conversion across atlases is unnecessary.

## 8. Generation pipeline — order of operations

```
PHASE 0 — Prerequisites (blocking)
  ├── User assembles curated landmark list (~10 high-priority regions, per orientation)
  ├── Verify Gemma 4 E4B chat template and tool-call rendering format (Unsloth)
  └── Real-histology collection per eval/download_datasets.py continues in parallel

PHASE 1 — Augmentation pipeline (blocking for buckets 1, 2, 3)
  └── Build atlas-image augmentation library (see §9)

PHASE 2 — Auxiliary buckets (parallel; bounded LLM use)
  ├── Bucket 2 (landmark listing)        — pure programmatic
  ├── Bucket 3 programmatic side          — from atlas registration
  ├── Bucket 3 distilled side             — Gemini batch, ≤1K examples, user-reviewed
  └── Bucket 4 (multi-slice morphology)   — Gemini Pro batch, several hundred examples

PHASE 3 — Agent traces (bucket 1)
  ├── Build query pool (real + clean atlas + augmented atlas)
  ├── Gemini 3.1 Pro through estimation loop, ~500 runs
  ├── Cheaper teachers through estimation loop, ~500–2K runs
  ├── Filter by acceptance criteria (atlas-relative tolerance + length cap)
  └── Strip Gemini's separate thought-summary channel; keep visible response text

PHASE 4 — Format and serialize
  ├── Render every example into Gemma 4 chat template (with image parts)
  ├── Verify with a SliceBench dry-run on a held-out slice
  └── Package for Unsloth QLoRA training
```

Phase 2 and Phase 3 can overlap once Phase 1 completes; the only hard blocker into Phase 3 is the augmentation pipeline output.

### 8.1 Bucket-1 acceptance filter

Two criteria, both hard:

- **Final position accuracy** within an *atlas-relative* tolerance — e.g., 2% of the atlas's valid range. ~0.27 mm on Allen mouse, ~1.0 mm on Waxholm rat. Avoids over-rewarding short-range atlases.
- **Trace length cap** — e.g., ≤25 turns. Excludes flailing patterns.

No human review of traces. With ~1–3K runs, the filter is purely automated.

### 8.2 Bucket-1 thought-channel handling

Strip Gemini's dedicated thought-summary channel (no inbound surface in Gemma). Keep the visible response text — this often contains brief reasoning interleaved with tool calls and is the shape Gemma will emit at deployment. Effective rule: if the content was rendered to the API user, keep it; if it was a hidden API channel, drop it.

### 8.3 Bucket-1 prompt mix

Both single-slice and group-estimation prompts, weighted toward single-slice (~70/30 single/group). Mirrors current deployed harness usage.

## 9. Augmentation pipeline — architecture

Built once, reused for SFT and later RLVR data generation. Lives at `models/langslice-gemma-4/data/augmentation/`.

**Inputs:** an atlas slice (rendered from BrainGlobe at chosen resolution and plane), plus optional reference volume (the Nissl-like example volume some atlases ship with).

**Composable transforms:**

- **DAPI-mimic** — replace grayscale intensity with blue-channel speckle on dark background; maintain region-boundary contrast.
- **Nissl-mimic** — invert to white background; tint somatic regions with neutral pinks/purples; soften region edges.
- **Brightfield-mimic** — beige-cream tonal map; light vignette.
- **Resolution-shift** — render query at one µm rate independent of `fetch_atlas` results.
- **Cropping / rotation jitter** — small affine perturbations to break exact-pixel match with what `fetch_atlas` returns.
- **Stain artifact noise** — light Gaussian + blotchy fixed-pattern noise.

**Composition:** randomized pipeline; each query draws a small subset of transforms with bounded magnitudes.

**Validation harness:** before bulk generation, render ~50 augmented samples and visually inspect. Confirm the augmented images have no detectable "tell" (e.g., identical speckle pattern across all DAPI-mimics) that the model could shortcut on.

## 10. Dependencies

| Dependency | Owner | Blocks |
|---|---|---|
| Curated landmark list (~10 regions per orientation) | User | Buckets 2, 3, 4 |
| Gemma 4 E4B chat template + tool-call format verification | Implementation | Phase 4 (and ideally earlier sanity-check) |
| SliceBench evaluation system | In development (user) | SFT-done gate |
| Real-histology collection (`eval/download_datasets.py`) | Ongoing | Bucket-1 query mix; affects synthetic share |
| Trace-rendering tool (Gemini agent run → Gemma chat-template multi-turn) | Implementation | Phase 4 |

## 11. Risks / open issues

- **Real-histology distribution skew.** If collected real data ends up >80% Allen mouse coronal, synthetic-side balancing must compensate harder. Monitor distribution post-collection; adjust synthetic share if needed.
- **Augmentation "tell" risk.** If augmented atlas images carry a detectable signature, model may learn to spot augmentation rather than learn visual matching. Mitigated by §9 validation harness and by mixing augmented + real + clean samples within each minibatch.
- **Cheap-model trace yield.** If non-Pro teachers produce <20% accept rate on Bucket 1, the trace pool may starve. Backstop: increase Gemini Pro share or relax acceptance tolerance. Decide post-hoc, not pre-committed.
- **Bbox-distillation review throughput.** ~1K Gemini-distilled bboxes need user review (~3 hours at 10 sec/example). Worth a lightweight review UI rather than ad-hoc.
- **Coordinate leakage in Bucket 4.** Easy to accidentally include "sections at 0.3 mm spacing" or similar in the prompt template. Bucket 4's prompt scaffolding must explicitly forbid mm in input and output.
- **Format-mismatch silent failure.** If trace data is generated in a format that does not match Gemma 4's chat template, training proceeds without error but the model learns nothing useful. Sanity-check the chat template before bulk generation.
- **"Thinking off" assumption may be premature.** Current observation that Gemma E4B "sucks at thinking" is pre-SFT. SFT on reasoning teacher data is exactly the lever that fixes that. Keep thinking-off as the deployment default, but revisit if post-SFT results suggest otherwise.

## 12. Validation hookpoint

- **During training** (cheap, fast): held-out subset of generated traces, score predicted-vs-teacher final position. Used for hyperparameter sanity.
- **Before declaring SFT done** (gate): SliceBench. Apples-to-apples vs. Flash/Pro variants. If MAE is wildly worse than Flash, debug before RLVR rather than letting RL chase a broken initialization.

## 13. What this design explicitly does not include

- Cross-atlas coordinate conversion tooling.
- Cross-atlas synthetic queries (slice from atlas A queried against atlas B).
- Self-correction trace classification / upweighting (RL handles this shaping).
- Image-gen synthetic histology (deferred; revisit during RLVR planning).
- Free-form bulk image captioning (reframed into Bucket 2 landmark-listing).
- Bbox-as-output trained on partial-registration histology without user review.
