# Multi-Slice AP Estimation Design

**Date:** 2026-04-02
**Status:** Draft

## Overview

Scale AP estimation from single-slice to whole-brain (20-60 slices). Uses Google ADK for multi-agent orchestration with a four-phase pipeline: anchor estimation, interval-based interpolation, wave-based nano-banana refinement, and constraint enforcement.

## User Inputs

```
langslice estimate-brain <image_folder> \
  --atlas allen_mouse_25um \
  --thickness 50        # um, hard minimum spacing \
  --interval 200        # um, average inter-slice distance \
  --anchors 4           # number of full AP estimation agents \
  --ordering strict     # strict | loose | none \
  --refinement on       # on | off (nano-banana pass) \
  --parallel 4          # max concurrent Gemini calls
```

- `--thickness`: Slice thickness in microns. Hard minimum spacing between any two slices (physical law).
- `--interval`: Average distance between slices in microns. Soft guide — real spacing varies due to discarded slices, double-collections, and forgotten well placements.
- `--anchors`: Number of slices that get full multi-turn tool-use AP estimation. User scales this based on how "choppy" their slicing was (more gaps = more anchors).
- `--ordering`: Sequence constraint mode (see Phase 4).
- `--refinement`: Whether to run nano-banana fine passes on non-anchor slices.
- `--parallel`: Max concurrent Gemini API calls (applies to both anchor and refinement waves).

**Image discovery:** Glob the folder for `.png`, `.jpg`, `.jpeg`, `.tif`, `.tiff`. Natural sort by filename. Filename order = assumed slice order (for strict/loose modes).

## Pipeline Architecture

Four phases composed as an ADK `SequentialAgent`:

```
SequentialAgent("brain_estimator") [
  Phase 1: ParallelAgent("anchor_wave")
            N anchor agents, each runs full AP estimation + nano-banana
            Writes state["anchor_{idx}"] = {filename, position_mm}

  Phase 2: Interpolation step                    [deterministic, no LLM]
            Reads anchor results, computes initial AP for every slice
            Writes state["positions"] = {filename: position_mm, ...}

  Phase 3: ParallelAgent("refinement_wave_N")    [optional, per wave]
            Nano-banana refinement radiating from anchors
            Updates state["positions"] with refined values

  Phase 4: Constraint enforcement step           [deterministic, no LLM]
            Validates ordering, enforces minimum spacing
            Writes state["final_positions"]
]
```

**Concurrency:** An `asyncio.Semaphore(--parallel)` caps concurrent Gemini calls. Anchor agents are `BaseAgent` subclasses wrapping the existing `estimate_position()` via `asyncio.to_thread()`. No rewrite of the core estimator.

## Phase 1: Anchor Estimation

### Anchor Selection (center-out priority)

Anchors are placed starting from the center of the slice list and expanding outward. This is because anterior brain slices (especially in mouse) lack visually distinct tissue and are unreliable for AP estimation.

```
20 slices, 4 anchors -> slices 5, 10, 15, 20
40 slices, 1 anchor  -> slice 20 (midpoint)
40 slices, 2 anchors -> slices 14, 27
```

The first and last slices are NOT guaranteed anchors. They get positions from interpolation/extrapolation off the nearest anchor.

### Per-Anchor Workflow

Each anchor agent runs two stages:

1. **Full multi-turn AP estimation** (existing `estimate_position()`) — coarse result, e.g. 3.45mm
2. **Nano-banana fine refinement** centered on the coarse result — fine result, e.g. 3.42mm

The nano-banana pass uses the standard config: 0.3mm window, images at 0.025mm spacing, centered on the coarse estimate. This becomes the anchor's locked position.

### Anchor Sanity Check

After all anchors complete, before interpolation, validate:
- Anchors are in monotonic AP order matching slice order (for strict/loose mode)
- No anchor outside atlas bounds

If an anchor is out of order, re-run that single anchor. Distance checks between anchors are intentionally loose — gaps in slicing make distance unreliable.

## Phase 2: Interpolation Step

Deterministic algorithm, no LLM. Assigns initial AP positions to every non-anchor slice.

### Between two adjacent anchors

Use the user-specified average interval as the step size. Distribute any small residual evenly across all gaps.

```
Anchor A at slice 5  -> 2.10mm
Anchor B at slice 15 -> 4.30mm
9 intermediate slices, 10 gaps

Ideal total = 10 * 0.200mm = 2.00mm
Actual total = 4.30 - 2.10 = 2.20mm
Per-gap adjustment = (2.20 - 2.00) / 10 = +0.020mm

Slice 6 = 2.10 + 0.220 = 2.32mm
Slice 7 = 2.32 + 0.220 = 2.54mm
...
```

Small deviations (~10um) from the stated interval are perfectly acceptable.

### Extrapolation beyond outermost anchors

Slices before the first anchor or after the last anchor use the user-specified average interval, stepping outward from the nearest anchor. Clamped to atlas bounds `[0.0mm, max_mm]`.

```
First anchor at slice 5 -> 2.10mm
Slice 4 = 2.10 - 0.200 = 1.90mm
Slice 3 = 1.70mm
Slice 2 = 1.50mm
Slice 1 = 1.30mm
```

### Dynamic prompt adjustment

The interpolated positions determine which atlas positions are shown to agents in Phase 3. The interpolation step doesn't just assign numbers — it defines the atlas search space for each slice's refinement.

## Phase 3: Nano-Banana Refinement (Wave-Based)

Runs on every non-anchor slice. Processes in waves radiating outward from anchors so that each slice's search window is bounded by confirmed (locked) positions.

### Wave ordering

```
Anchors locked at slices 5, 15, 25, 35

Wave 1: slices 4,6, 14,16, 24,26, 34,36
         Each adjacent to a locked anchor — all run in parallel

Wave 2: slices 3,7, 13,17, 23,27, 33,37
         Each adjacent to a wave-1-confirmed slice — all run in parallel

Wave 3: Continue until waves meet in the middle of each segment
```

Slices within the same wave are independent — full parallelism within each wave, throttled by `--parallel`.

### Window construction

Each slice's nano-banana search window is bounded by its nearest locked neighbors:

```
Processing slice 7 (wave 2):
  Left:  slice 6 LOCKED at 2.33mm
  Right: slice 8 interpolated (not locked)

  Hard bound (left):  2.33mm + thickness
  Soft bound (right): interpolated position + interval
  Center: slice 7's interpolated position

Later, processing slice 8 (wave 3):
  Left:  slice 7 LOCKED at 2.42mm
  Right: slice 9 LOCKED at 2.65mm

  Window: [2.42 + thickness, 2.65 - thickness]
  Tightest possible — locked on both sides
```

The model cannot see atlas positions outside the window. This prevents ordering violations by construction.

### Dynamic atlas image count

The number of atlas images sent to nano-banana scales with the window size. A tight window (locked on both sides, small range) gets fewer images. A wide window (extrapolation, only one locked neighbor) gets more. Target spacing remains 0.025mm; the image count is `floor(window_size / 0.025)`, clamped to a reasonable range (e.g., 5-13).

```
Window 0.33mm wide -> 13 images (full set)
Window 0.20mm wide -> 8 images
Window 0.12mm wide -> 5 images (minimum)
```

### Locking

Once nano-banana returns a position for a slice, that slice is locked. Its position becomes a hard bound for neighboring slices in subsequent waves. Locked positions are written to disk immediately (see Checkpointing).

## Phase 4: Constraint Enforcement Step

Lightweight validation pass. Most violations are prevented by the windowed refinement in Phase 3.

### Ordering modes

**Strict:** AP positions must be monotonically increasing with slice index. If nano-banana produced a violation (rare given windowing): clamp the offending slice to midpoint between its locked neighbors. Minimum spacing = slice thickness, always enforced.

**Loose:** Same as strict, plus allows single adjacent swaps. After all positions are finalized, scan for cases where swapping two adjacent slices reduces total deviation from expected interval. Only swap if both slices have nano-banana confirmation. Maximum one swap per pair, no cascading.

**None:** No monotonicity enforcement. Each position is its nano-banana result, independent of neighbors. Minimum spacing still enforced — if two slices land within thickness of each other, nudge apart. This is "DeepSlice mode."

### Global checks (all modes)

- All positions within atlas bounds `[0.0mm, max_mm]`
- No two slices closer than slice thickness
- Summary stats computed: mean interval, std deviation, anomalous gap count

## Checkpointing & Resumability

Each slice is written to the output JSON as soon as it locks in (after anchor confirmation or nano-banana refinement). On resume, the pipeline reads the JSON, identifies which slices are already confirmed, and picks up from the current wave.

This means:
- A network failure at wave 3 loses at most the in-flight wave
- Anchor results are never re-computed unless explicitly requested
- The user can interrupt and resume without cost penalty

## Cost Estimation

Before starting, show the user an estimate:

```
Brain estimation plan:
  40 slices, 4 anchors, strict ordering, refinement ON

  Phase 1:  4 anchor estimations          ~$0.20  (~2 min)
  Phase 1b: 4 anchor nano-banana passes   cost TBD
  Phase 3:  36 nano-banana refinements    cost TBD
  ~5 refinement waves, --parallel 4

  Estimated total: >= $0.20 + nano-banana costs
  Proceed? [Y/n]
```

Nano-banana cost and time estimates will be filled in once benchmarked in the multi-slice context.

## Output

### Data structure

```python
@dataclass
class BrainEstimationResult:
    positions: dict[str, float]       # filename -> AP mm
    ordering_mode: str                # strict | loose | none
    anchor_slices: list[str]          # which filenames were anchors
    summary: BrainEstimationSummary   # mean interval, std, total cost, wall time
```

### JSON export

```json
{
  "atlas": "allen_mouse_25um",
  "thickness_um": 50,
  "interval_um": 200,
  "ordering_mode": "strict",
  "slices": [
    {"filename": "slice_001.tif", "position_mm": 1.30, "source": "extrapolated+refined"},
    {"filename": "slice_005.tif", "position_mm": 2.10, "source": "anchor"},
    {"filename": "slice_006.tif", "position_mm": 2.33, "source": "interpolated+refined"}
  ]
}
```

The `source` field tracks provenance: `anchor`, `interpolated`, `extrapolated`, `interpolated+refined`, `extrapolated+refined`.

### Integration

- New CLI subcommand: `langslice estimate-brain`
- Output JSON feeds directly into batch registration (future feature)
- Compatible with QUINT/DeepSlice tooling expectations

## Technology

- **Orchestration:** Google ADK (`google-adk` package, stable v1.x)
- **Agent primitives:** `SequentialAgent`, `ParallelAgent`, `BaseAgent`
- **Model calls:** Existing `estimate_position()` and nano-banana logic wrapped in `BaseAgent` subclasses via `asyncio.to_thread()`
- **Concurrency:** asyncio + `Semaphore` for throttling
- **Model backend:** Gemini via AI Studio (current). ADK's `LiteLlm` wrapper enables future support for Ollama, OpenAI-compatible endpoints, and other providers.
