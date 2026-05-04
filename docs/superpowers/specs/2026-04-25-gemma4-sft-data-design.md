---
title: Gemma 4 E4B — SFT Data Design
date: 2026-04-25
scope: Supervised fine-tuning data composition, generation pipeline, and augmentation architecture for the langslice-gemma-4 model. RLVR is a separate spec.
status: draft v4 (position-trace collection; no-thinking default; rationale fallback)
---

# Gemma 4 E4B — SFT Data Design

## 1. Goal

Produce an SFT corpus that turns Gemma 4 E4B into a drop-in replacement for Gemini inside the LangSlice position-estimation agent loop (`fetch_atlas` + `submit_estimate` / `submit_group_estimate` from `src/langslice_harness/harness/estimation/tools.py`), as the warm-start before a separate, larger RLVR phase. The task covers coronal, sagittal, and horizontal position estimation; do not frame this plan as AP-only except where discussing the coronal AP axis or legacy registration/export terminology.

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
- SFT training hyperparameters and Unsloth/QLoRA configuration. (One training-schedule recommendation in §14 is a hand-off note, not part of this spec's deliverables.)
- Evaluation system (SliceBench, currently in development).

## 3. Coverage axes

- **Species / atlases:** `allen_mouse_25um`, `whs_sd_rat_39um`, `admba_*` (developmental mouse). Three, reflecting actual public-data availability per `eval/dataset_allocation.md`. Other priority species (prairie vole, mole-rat, mouse lemur) have effectively zero registered public tissue and are excluded from SFT.
- **Orientations:** coronal / sagittal / horizontal. Coronal-heavy with substantial sagittal and minimal horizontal expected; actual mix follows real-data collection rather than a pre-committed split.
- **Modalities (real data):** fluorescence + brightfield + Nissl + ISH where available, per `eval/data_inventory.md`.
- **Resolution mismatch:** include examples where the query is rendered at one µm rate and `fetch_atlas` returns slices at another (e.g., 10 µm query → 50 µm atlas, and the inverse). Discourages shortcut-learning on a fixed query/atlas resolution pairing.

## 4. Budget envelope

Total: roughly **2K–15K SFT examples**. This range comes from narrow-task tool-use fine-tuning literature, not from foundation cold-start regimes. Reference points:

- TL-Training (Ye et al., 2024 — arxiv 2412.15495) achieved SOTA tool-use on a 7B base with ~1,200 samples.
- iTool (Zeng et al., 2025 — arxiv 2501.09766) reports training-gain decay above ~10K examples for narrow tool-use tasks; quality-filtered traces dominate raw volume.

We are fine-tuning a capable VLM (Gemma 4 E4B is multimodal with native function-calling per Unsloth docs) for **one specific tool-use behavior**, not training general capability. Foundation-training anchors like ReVisual-R1's 40K cold-start are a category mismatch.

Hard inputs to the budget:

- Gemini API credits: ~$83 → ~500 high-quality Gemini 3.1 Pro trace runs.
- Cheaper teacher models: another ~500–2K trace runs (Flash, OpenAI-compatible models).
- Auxiliary anatomy data (buckets 2–4): cheap to scale programmatically; bounded by user landmark-list curation and ~1K Gemini-distilled bboxes on real histology.
- Programmatic tool-call skeletons (bucket 5, new in v2): cheap to generate, used as backstop coverage.

Smaller and cleaner is preferred over the upper bound.

## 5. Data composition — five buckets

| # | Bucket | Purpose | Source | Output shape | Approx volume |
|---|---|---|---|---|---|
| 1 | Agent traces | Teach tool-use loop, mm scales per atlas, broad-to-narrow sweep heuristics, and stable final submission | Gemini 3.1 Pro (~500 runs) + cheaper teachers (~500–2K runs), filtered to ~40–60% accept rate | Multi-turn deployment trace: target image + compact tool calls + atlas-image tool results + `submit_estimate` / `submit_group_estimate`; rationale export retained as a fallback experiment, not the default training target | ~500–1.5K traces (primary bucket) |
| 2 | Landmark listing | Teach loop's first-turn contract: "describe 2–3 prominent anatomical landmarks" | Programmatic from atlas annotations (top-N visible regions by area, named) | Single-turn: image → "anterior commissure visible, hippocampus forming, …" | ~1–3K |
| 3 | Bbox grounding | Visual-spatial grounding of curated landmarks; supports future LangSlice bbox-based registration | Programmatic from atlas registration; ≤1K Gemini-distilled on partial-registration real histology with user review | Both directions: (image + region name → bbox), (image + bbox → region name) | ~1–3K programmatic + ≤1K distilled. **Hard cap at ≤25% of total SFT volume** so traces dominate. |
| 4 | Multi-slice morphology | Teach how regions transform across positions; auxiliary visual-anatomy vocabulary, not required agent-loop output | Gemini Pro distilled (text quality matters) | (6 atlas sections + region) -> "this region begins as a thin crescent and broadens caudally..." | ~300–600 |
| 5 | Programmatic tool-call skeletons (NEW in v2) | Backstop coverage for atlas × orientation × modality combos under-served by accepted teacher traces | Deterministic generation: known-correct broad sweep → narrow → submit, against random target positions | Same multi-turn shape as bucket 1 | A few hundred to ~1K, dialed up if bucket 1 yield is thin |

### 5.1 Format-alignment principle

Bucket 1 and bucket 5 examples must match what Gemma will emit at deployment: tool calls and final position estimates, with thinking disabled by default and no required visible rationale. Buckets 2, 3, and 4 are auxiliary grounding/captioning tasks and may have non-deployment output shapes, but their volume is capped so they do not make the deployed agent chatty or bbox-oriented. Bucket 3 is retained because future LangSlice versions may use bbox output for registration and because bbox grounding is a bounded auxiliary task in current VLM literature (KOSMOS-2, GLaMM, BoxTuning). The evidence for bbox-grounding transfer to position estimation specifically is weak; keep it on product-roadmap grounds, not as a core trace objective.

### 5.2 Coordinate leakage rule

The deployment loop is millimeter-native: `fetch_atlas` takes mm, tool results say "Atlas at 5.40 mm", and the final answer is `position_mm`. For coronal examples this is AP; for sagittal it is ML; for horizontal it is DV. Removing mm from training data is incompatible with the deployment task. Therefore:

- Buckets 1, 3, and 5 keep mm where it appears naturally — in tool calls, tool results, and final answers.
- Bucket 4 (multi-slice morphology) **must omit mm from prompts and outputs** — this bucket is the one place where the model could otherwise learn "section index ↔ mm" maps. The 6 sections are described positionally ("anterior to posterior") rather than by mm.
- Bucket 2 (landmark listing) is single-image and naturally mm-free.

### 5.3 Stratification target (NEW in v2)

Bucket 1 (agent traces) is stratified across the structural axes below, with a target uniformity per axis. Bucket 5 (programmatic skeletons) is generated specifically to fill cells where bucket 1 yield is thin.

| Axis | Strata |
|---|---|
| Atlas | mouse / rat / developmental mouse |
| Orientation | coronal / sagittal / horizontal |
| Modality | fluorescence / brightfield / Nissl / ISH / clean-atlas / augmented-atlas |
| Position range (relative to atlas extent) | anterior third / middle third / posterior third |
| Trace pattern | clean-success / hard-negative (see §8.2) |

No hard percentage commitment per cell, but post-collection distribution is reported and rebalanced via bucket 5 generation if any cell is severely under-served.

### 5.4 Hard-negative coverage target (NEW in v2)

Per iTool (2025), narrow-task tool-use SFT benefits from a deliberate ~20–30% share of "teacher struggled, then recovered" traces. These carry more gradient signal than clean-success-only traces because they teach recovery after an initially poor comparison or overly broad sweep. Do not encode an over-specific policy such as mandatory bracket, verify, or fine-tune stages; the durable behavior is broad-to-narrow search with valid tool calls and a calibrated final estimate.

We do not need a separate classifier (rejected as overhead in §13). Hard-negative detection is pure post-processing on accepted traces, using two cheap signals (defined in §8.2):

- **Trace length above median** in the accepted set
- **Trajectory excursion**: `max(|fetched_position − final_submission|)` exceeds an atlas-relative threshold

A trace is labeled "hard-negative" if either signal fires. Target: 20–30% of bucket 1 final mix. If natural distribution under-delivers, generate additional traces by seeding the agent at deliberately-far starting positions.

## 6. Query-source mix (applies to bucket 1)

Three query sources, mixed without a hard percentage commitment:

- **Real histology** — prioritized wherever available. Quantity bounded by `eval/data_inventory.md` collection.
- **Direct atlas slices** — clean atlas slice as query against the same atlas. Trains tool format, mm-scale-per-atlas, and broad-sweep heuristics. Trivial visual matching.
- **Augmented atlas slices** — atlas slices passed through the augmentation pipeline (DAPI-mimic, Nissl-mimic, etc.). Pulls weight on visual matching as well as format.

Cross-atlas synthetic queries (slice from atlas A used against atlas B) are explicitly excluded — silver-standard labels from another model are too noisy for a domain where ground truth is the whole point.

## 7. Defense against atlas-coordinate memorization

The risk: if 80%+ of real histology is registered to Allen CCFv3, the model may learn "this image looks like → AP X.X mm in Allen mouse" as a memorization shortcut rather than a visual-matching skill. Defenses, in order of importance:

1. **Atlas distribution diversity in synthetic queries.** Direct-atlas, augmented-atlas, and programmatic-skeleton queries are spread across all three priority atlases, balancing the Allen-CCF dominance of real-data.
2. **Loop structure.** The agent must compare the query image to the fetched atlas images present in the same context — visual matching, not memorization, is what produces the correct tool calls. The loop's design carries most of the structural defense.
3. **Resolution mismatch.** Mixing query/atlas µm rates within and across atlases prevents memorization of fixed appearance/coordinate pairs.
4. **Atlas-version metadata in every example** (NEW in v2, see §10). Allen CCFv2 vs v3 coordinate offsets are non-trivial; serializing the atlas version prevents silent cross-version contamination during training and eval.

Cross-atlas coordinate conversion tooling is not built. Each example is atlas-internal; conversion across atlases is unnecessary.

## 8. Generation pipeline — order of operations

```
PHASE 0 — Prerequisites (blocking)
  ├── User assembles curated landmark list (~10 high-priority regions, per orientation)
  ├── Verify Gemma 4 E4B chat template AND instruct-masking behavior in Unsloth
  │   (loss must be computed only on assistant turns, not user/tool turns)
  └── Real-histology collection per eval/download_datasets.py continues in parallel

PHASE 1 — Augmentation pipeline (blocking for buckets 1, 2, 3, 5)
  ├── Build atlas-image augmentation library (see §9)
  └── Augmentation separability gate: classifier ≤55% accuracy on aug-vs-real

PHASE 2 — Auxiliary buckets (parallel; bounded LLM use)
  ├── Bucket 2 (landmark listing)        — pure programmatic
  ├── Bucket 3 programmatic side          — from atlas registration
  ├── Bucket 3 distilled side             — Gemini batch, ≤1K examples, user-reviewed
  └── Bucket 4 (multi-slice morphology)   — Gemini Pro batch, several hundred examples

PHASE 3 — Agent traces (bucket 1) and skeletons (bucket 5)
  ├── Build query pool (real + clean atlas + augmented atlas), stratified per §5.3
  ├── Gemini 3.1 Pro through estimation loop, ~500 runs
  ├── Cheaper teachers through estimation loop, ~500–2K runs
  ├── Filter by acceptance criteria (§8.1) and tag teacher ID + hard-negative label (§8.2)
  ├── Rebalance: identify under-served stratification cells; generate bucket 5 skeletons
  └── If hard-negative rate <20%, supplement with seeded-far-start traces

PHASE 4 — Format and serialize
  ├── Subject-level split: hold out brains used in SliceBench eval BEFORE serialization
  ├── Render every example into Gemma 4 chat template (with image parts, atlas-version metadata)
  ├── Verify with a SliceBench dry-run on a held-out slice
  └── Package for Unsloth QLoRA training
```

Phase 2 and Phase 3 can overlap once Phase 1 completes; the only hard blocker into Phase 3 is the augmentation pipeline output.

### 8.1 Bucket-1 acceptance filter

Two criteria, both hard:

- **Final position accuracy** within an *atlas-relative* tolerance — e.g., 2% of the atlas's valid range. ~0.27 mm on Allen mouse, ~1.0 mm on Waxholm rat. Avoids over-rewarding short-range atlases.
- **Trace length cap** — e.g., ≤25 turns. Excludes flailing patterns.

Every accepted trace is tagged with **teacher-ID metadata** (which model produced it). No second-stage student-aware filter; the locked prompt scaffold + outcome filter is sufficient for capable students per Merge-of-Thought-style mixed-teacher distillation literature. Teacher ID is retained for post-hoc analysis only.

No human review of traces. With ~1–3K runs, the filter is purely automated.

### 8.2 Hard-negative detection (NEW in v2)

Pure post-processing on accepted traces. Two signals:

```python
def is_hard_negative(trace, accept_tolerance_mm):
    # Signal 1: trace length above median of accepted set
    length_signal = trace.length > median_accepted_length

    # Signal 2: trajectory excursion
    truth = trace.final_submitted_position
    fetched_positions = [p for call in trace.fetch_atlas_calls for p in call.positions_mm]
    max_excursion = max(abs(p - truth) for p in fetched_positions)
    excursion_signal = max_excursion > 2 * accept_tolerance_mm

    return length_signal or excursion_signal
```

Targeted ratio: **20–30% hard-negatives in final bucket 1 mix**. If natural distribution under-delivers (<20%), seed additional runs with deliberately-far starting hints to provoke restart-and-recover behavior.

### 8.3 Bucket-1 rationale and thinking handling

Gemini thinking is teacher-side scaffolding. Collect thought summaries when available for analysis, trace categorization, and fallback experiments, but do not make them the default SFT target.

Default training export:

- Use `sft_trace_deployment.json`.
- Train Gemma with thinking off / no visible rationale by default.
- Optimize for faithful, valid tool use and final position accuracy.

Fallback / auxiliary exports:

- `sft_trace_rationale.json` may be used if no-thinking tool use fails in held-out evaluation. It should include short teacher rationale summaries as visible assistant text, not hidden chain-of-thought.
- Rationale text may also be converted into small caption/landmark auxiliary examples. Keep this mix modest so Gemma gains visual-anatomical vocabulary without learning to narrate every agent-loop step.
- True Gemma thought-channel training is a later ablation only. If attempted, verify the exact current Gemma 4 E4B tokenizer/chat template, loss masking, tool-call parsing, and context handling first. Previous thought blocks must not be carried forward in multi-turn history.

Process rewards in RLVR are optional. The verifiable reward can be final coordinate error, valid tool-call schema, successful submission, and tool-efficiency; RLVR does not require grading the model's reasoning text.

### 8.4 Bucket-1 prompt mix

Both single-slice and group-estimation prompts, weighted toward single-slice (~70/30 single/group). Mirrors current deployed harness usage.

## 9. Augmentation pipeline — architecture

Built once, reused for SFT and later RLVR data generation. Lives at `models/langslice-gemma-4/data/augmentation/`. Literature anchors: PathDiff (ICCV 2025) — unpaired mask-to-H&E / mask-to-IHC pixel-level diffusion, directly motivating the Stage B img2img approach; PixCell (Stony Brook, ICCV 2025) — SD 3.5 VAE backbone, layout-to-histology with Cell-ControlNet, trained on PanCan-30M. HistAug (Boutaj et al., ICCV 2025) was investigated and is not applicable here: it operates in feature space (UNI/Virchow2 embeddings) rather than pixel space.

**Inputs:** an atlas slice (rendered from BrainGlobe at chosen resolution and plane), plus optional reference volume (the Nissl-like example volume some atlases ship with).

**Composable transforms:**

- **DAPI-mimic** — replace grayscale intensity with blue-channel speckle on dark background; maintain region-boundary contrast.
- **Nissl-mimic** — invert to white background; tint somatic regions with neutral pinks/purples; soften region edges.
- **Brightfield-mimic** — beige-cream tonal map; light vignette.
- **Resolution-shift** — render query at one µm rate independent of `fetch_atlas` results.
- **Cropping / rotation jitter** — small affine perturbations to break exact-pixel match with what `fetch_atlas` returns.
- **Stain artifact noise** — light Gaussian + blotchy fixed-pattern noise.

**Composition:** randomized pipeline; each query draws a small subset of transforms with bounded magnitudes.

### 9.1 Architecture: deterministic synthesis with custom textures

**Stage B (diffusion via Flux 2 Klein) was explored thoroughly and abandoned 2026-04-27.** See `feedback_diffusion_synthesis_abandoned` for the full record. Briefly: multi-reference img2img with Flux 2 Klein 9B (NVFP4, ComfyUI backend) could produce either anatomy-preserving outputs that looked like recolored atlas (low denoise) or photorealistic-microscopy outputs that lost atlas-specific anatomy (high denoise) — the tradeoff window did not contain a usable point for SFT training data. The pivot is to **deterministic procedural synthesis with custom per-modality texture algorithms** (no image-gen models).

**Architecture.** Per-modality procedural pipelines under `models/langslice-gemma-4/data/augmentation/{dapi,nissl,brightfield,fluorescence,ish}_pipeline.py`, each consuming the BrainGlobe atlas's grayscale **reference** slice + its annotation slice, and producing an HWC float32 [0,1] augmented section. Texture transforms (`transforms/texture.py`, `transforms/tissue_class.py`) carry their own per-call randomization (density, blob sigma/aspect, intensity); per-image global parameters (gamma, floor, tone shift, brightness/contrast) draw from bounded ranges to diversify outputs without changing gross anatomy.

**Anatomy preservation strictness:** gross structure only. The model is being trained for position estimation, not landmark registration. Fine-boundary drift of a few millimeters is acceptable; gross section geometry (hemisphere outline, major white-matter tracts) must be preserved. The deterministic architecture preserves anatomy by construction — no transform displaces atlas structure.

**Reusable infrastructure left over from the diffusion attempt:** the ComfyUI HTTP client (`src/langslice_harness/comfyui/`) and Hugging Face snapshot downloader are kept as general-purpose harness utilities. They are not on the SFT critical path; they remain available for any future local image-gen needs not related to data generation.

**Validation gates (NEW in v2):**

1. **Visual inspection** — render ~50 augmented samples and inspect for obvious artifacts (kept from v1).
2. **Automated separability classifier** — train a small binary classifier (frozen ResNet penultimate features + logistic regression — minutes to train) to discriminate augmented vs. real histology. **Require ≤55% accuracy** before bulk generation. If the classifier exceeds 55%, the augmentation pipeline has a detectable signature the model could shortcut on; tighten transform magnitudes or add domain randomization. The gate runs **per modality** (DAPI, Nissl, brightfield, fluorescence, ISH) on a subject-level held-out set. Failure in one modality routes that modality — and only that modality — through Stage B (§9.1); other modalities continue with Stage A.

## 10. Dependencies

| Dependency | Owner | Blocks | Notes |
|---|---|---|---|
| Curated landmark list (~10 regions per orientation) | User | Buckets 2, 3, 4 | |
| Gemma 4 E4B chat template + tool-call format verification | Implementation | Phase 4 | Includes verifying instruct-masking (loss on assistant turns only) |
| SliceBench evaluation system | In development (user) | SFT-done gate | Defines held-out brain set used for subject-level splits |
| Subject-level split manifest | User + implementation | Phase 4 serialization | Brains in SliceBench MUST NOT appear in any SFT bucket |
| Atlas-version metadata schema | Implementation | Phase 4 | Every example records atlas name + version (e.g., `allen_mouse_25um@CCFv3`) |
| Real-histology collection (`eval/download_datasets.py`) | Ongoing | Bucket-1 query mix | Affects synthetic share |
| Trace-rendering tool (Gemini agent run → Gemma chat-template multi-turn) | Implementation | Phase 4 | |
| Rationale/deployment export selector | Implementation | Phase 4 | Default to deployment traces; rationale traces are fallback/ablation inputs |

## 11. Risks / open issues

- **Real-histology distribution skew.** If collected real data ends up >80% Allen mouse coronal, synthetic-side balancing must compensate harder. Monitor distribution post-collection; adjust synthetic share if needed.
- **Augmentation "tell" risk.** If augmented atlas images carry a detectable signature, model may learn to spot augmentation rather than learn visual matching. Mitigated by §9's two-gate validation (visual + classifier ≤55%).
- **Cheap-model trace yield.** If non-Pro teachers produce <20% accept rate on Bucket 1, the trace pool may starve. Backstop: increase Gemini Pro share, relax acceptance tolerance, or generate more bucket 5 skeletons. Decide post-hoc, not pre-committed.
- **Hard-negative under-supply.** If natural-distribution hard-negative rate is <20% after filtering, supplement with seeded-far-start runs (§8.2). If even seeding doesn't deliver, document the gap rather than synthesize fake hard-negatives.
- **Bbox-distillation review throughput.** ~1K Gemini-distilled bboxes need user review (~3 hours at 10 sec/example). Worth a lightweight review UI rather than ad-hoc.
- **Coordinate leakage in Bucket 4.** Easy to accidentally include "sections at 0.3 mm spacing" or similar in the prompt template. Bucket 4's prompt scaffolding must explicitly forbid mm in input and output.
- **Subject-level leakage** (NEW in v2). Same brain appearing in SFT and SliceBench inflates eval scores. Cited evidence: "AI slipping on tiles" (Bussola et al., 2019 — arxiv 1909.06539), PathAlign (2024). Mitigation: subject-level holdout list locked *before* SFT generation begins (§10 dependency).
- **Atlas-version contamination** (NEW in v2). Allen CCFv2 vs v3 have different coordinate origins and small alignment offsets. Without atlas-version metadata, mixing data from both versions teaches incoherent mm-anchor mappings. Mitigation: schema requirement in §10.
- **Format-mismatch silent failure.** If trace data is generated in a format that does not match Gemma 4's chat template, training proceeds without error but the model learns nothing useful. Sanity-check the chat template AND instruct-masking before bulk generation.
- **Thinking instability in E4B.** Gemma 4 E4B is small enough that visible/thought-channel reasoning can destabilize format, tool calls, or verbosity. Keep thinking-off deployment traces as the default. If that fails, ablate compact visible action captions/rationale exports before full thought-channel training.

## 12. Validation hookpoint

- **During training** (cheap, fast): held-out subset of generated traces, score predicted-vs-teacher final position. Used for hyperparameter sanity.
- **Before declaring SFT done** (gate): SliceBench. Apples-to-apples vs. Flash/Pro variants. If MAE is wildly worse than Flash, debug before RLVR rather than letting RL chase a broken initialization.

Subject-level holdout enforced: any brain represented in the SliceBench eval set is excluded from every SFT bucket *before* generation, not after.

## 13. What this design explicitly does not include

- Cross-atlas coordinate conversion tooling.
- Cross-atlas synthetic queries (slice from atlas A queried against atlas B).
- Self-correction *classifier* (cheap post-processing tags hard-negatives instead — §8.2).
- Image-gen synthetic histology (deferred; revisit during RLVR planning).
- Free-form bulk image captioning as a primary objective. Short landmark/caption examples are allowed only as a modest auxiliary mix or rationale fallback.
- Bbox-as-output trained on partial-registration histology without user review.
- Local-step student-aware reasoning filter (rejected: locked prompt scaffold + outcome filter is sufficient for capable students).
- Curriculum ordering (rejected: overkill for narrow-task fine-tuning of a capable base).
- Embedding-cluster deduplication (rejected: simple exact-duplicate dedup only; data sources are inherently bounded).
- Hierarchical coordinate tokenization (interesting future ablation; out of scope for this spec).

## 14. Training-schedule recommendation (hand-off to writing-plans)

Not part of the data spec, but flagged here for the implementation plan: 2026 narrow-fine-tuning literature (BoxTuning, GeoGround) reports "grounding-reasoning interference" when bbox-output and coordinate-output objectives are mixed in a single training stage. Recommended schedule for the writing-plans phase:

- **Stage 1**: Bucket 3 (bbox grounding) + Bucket 2 (landmark listing). Initialize spatial-anatomical grounding.
- **Stage 2**: Bucket 1 (agent traces) + Bucket 4 (multi-slice morphology) + Bucket 5 (skeletons). Fine-tune the deployment behavior.

This is a recommendation for the training schedule, not a data-design constraint. The data spec produces all five buckets in parallel; staging decides how they are presented to the optimizer.

## 15. Literature anchors used in this spec

For traceability — these are the references that shaped specific spec decisions, not a literature review:

- TL-Training (Ye et al., 2024, arxiv 2412.15495) — narrow-task SFT scale floor (§4)
- iTool (Zeng et al., 2025, arxiv 2501.09766) — training-gain decay above ~10K, hard-negative ratio (§4, §5.4)
- PathDiff (ICCV 2025) — unpaired mask-to-H&E / mask-to-IHC pixel-level diffusion; motivates Stage B img2img with structural reference (§9, §9.1)
- PixCell (Stony Brook, ICCV 2025) — SD 3.5 VAE backbone, layout-to-histology foundation model with Cell-ControlNet, trained on PanCan-30M; evaluated and rejected in favor of Flux 2 Klein for licensing and general-modality coverage (§9.1)
- ReVisual-R1 (Chen et al., 2025, arxiv 2506.04207) — *reference-class miss*: cited only as a contrast (foundation cold-start regime, not narrow-task SFT)
- KOSMOS-2 (Peng et al., 2023) — bbox grounding as foundation capability, supports §5.1's bbox-bucket retention on roadmap grounds
- "AI slipping on tiles" (Bussola et al., 2019, arxiv 1909.06539) — subject-level leakage in digital pathology (§11)

Confidence note: several claims rely on aggregated 2025–2026 literature whose individual citations were not independently verified in this spec session. Where a specific recommendation rests on an unverified citation, it is also defensible from general post-2024 fine-tuning practice.
