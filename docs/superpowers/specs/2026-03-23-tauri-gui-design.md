# LangSlice Tauri GUI — Design Document

*2026-03-26*

## Vision & Philosophy

LangSlice GUI is an **agent-first batch processing application** — a visual wrapper around the LangSlice CLI. Users upload one or more brains' worth of histology slices, configure parameters, hit Run, and watch AI agents estimate AP positions and register slices in a 3D atlas environment.

**Core principles:**
- The CLI is the real product; the GUI is optional (`langslice gui`)
- Installed via conda/pip like DeepSlice and ABBA
- Agent does the work; user monitors passively
- No manual registration — export to QUINT and ABBA for refinement by the existing open-source community

## Architecture Overview

Three layers, each in its natural language:

```
┌──────────────────────────────────────────────┐
│            Frontend (Tauri Webview)           │
│         TypeScript + React                   │
│         Three.js / react-three-fiber         │
│                                              │
│  3D atlas scene, slice thumbnails, controls, │
│  progress indicators, all visible UI         │
└──────────────────┬───────────────────────────┘
                   │  invoke("command", args)
                   │  event streams (Tauri events)
┌──────────────────▼───────────────────────────┐
│            Rust Backend (Tauri)               │
│                                              │
│  Atlas volume loading & caching              │
│  Arbitrary-angle reslicing                   │
│  Mesh serving for Three.js                   │
│  Border computation from label volume        │
│  Session state management                    │
│  Python subprocess orchestration             │
└──────────────────┬───────────────────────────┘
                   │  websocket / stdio pipe
┌──────────────────▼───────────────────────────┐
│            Python Subprocess                  │
│                                              │
│  LangSlice core library                      │
│  Multi-agent AP estimation (Gemini)          │
│  Registration (landmark correspondences)     │
│  Affine/TPS solving                          │
└──────────────────────────────────────────────┘
```

### Why these choices

**Three.js in the webview, not Rust wgpu or NiiVue:**

- **Not wgpu:** Compositing into Tauri's webview is an unsolved problem (frame-streaming adds latency, separate window is disjointed). Scene is light enough for WebGL.
- **Not NiiVue** (evaluated and rejected): Excellent WebGL 2.0 neuroimaging volume/mesh viewer, but fundamentally an MPR viewer with a closed rendering pipeline. No textured planes in 3D space, no mesh positioning API, no animation loop. Our use case — 50 textured planes teleporting through a 3D mesh scene — is a custom 3D scene, and Three.js is purpose-built for that.

**Rust backend, not Python, for atlas operations:**

- Atlas reslicing is array math on a ~few-hundred-MB voxel volume. Rust matches or exceeds Java (ABBA's language). Python/NumPy has per-call overhead that makes interactive reslicing sluggish.
- Tauri's backend is Rust by default — no extra infrastructure.

**Python stays for AI/agent logic:**

- LangSlice's Gemini integration, registration pipeline, and solver are all Python.
- No pure Rust solution exists for histology-to-atlas registration. Python sidecar is the pragmatic path.

## Navigation & Layout

### Two-level navigation

**1. Dashboard view:** Grid of brain cards. Each card shows:
- Brain name/title
- Mini live 3D atlas render (with fallback to static thumbnail on weaker hardware, auto-detected via WebGL capabilities at startup)
- Progress bar (e.g., "8/20 slices")
- Status badge: Complete (teal), Running (coral), or Queued (gray)
- "Add Brain" button and empty-state card with dashed border ("Drop folder or click to add")

**2. Brain detail view:** Click a brain card to drill in. Left sidebar layout:
- **Back navigation:** "← All Brains" + brain name
- **Left sidebar (~25%):** Slice list (status icons: ✓ done, ⟳ running, ○ pending, with AP positions for completed slices) + Agent status panel (which agent is working on which slice + current estimate)
- **Main area (~75%):** Switchable between two modes:
  - **3D view:** Full 3D atlas scene with the brain's slices as textured planes (the "show" during agent estimation)
  - **2D view:** Ported from the current PySide6 GUI — split view showing atlas coronal section alongside the histology slice, plus an overlay mode that composites the atlas onto the slice with edge-detected region borders and correspondence markers. Uses the same coronal geometry contract as QUINT/ABBA export.
- **Export button** at bottom of sidebar (appears after run completes)

### Settings

Simplified parameters visible by default (atlas, model, agent count). **Developer mode** toggle in settings reveals full CLI parameter control. Specific parameters TBD as the agent system matures.

## User Workflow

1. **Setup:** Open app → dashboard. Add brain(s) by dropping folders of slice images. Configure parameters.
2. **Run:** Hit Run. Multi-agent orchestration kicks off (e.g., 5 agents each processing slices one at a time from the queue). Multiple brains can be queued.
3. **Monitor (passive):** Watch from dashboard (progress bars) or drill into a brain to watch slices teleport through the 3D atlas as agents estimate AP positions. Visual state indicators: searching → converging → locked.
4. **Export:** Export results. Export system is extensible — specific formats depend on community integration. ABBA state file (`.abba`) is a target. Output is always batched.

**What the app does NOT do:** Manual registration. No dragging slices, no placing landmarks. That's QUINT and ABBA's job.

## Rust Backend — ABBA Parallel

The Rust backend mirrors what ABBA's Java layer does:

| ABBA (Java)            | LangSlice (Rust)                                     |
|------------------------|------------------------------------------------------|
| `Atlas` / `AtlasMap`   | Struct that loads BrainGlobe TIFF volume + JSON metadata, caches in memory |
| `ReslicedAtlas`         | Reslicing functions — sample arbitrary planes through the cached volume, manage margins/bounds |
| `MultiSlicePositioner`  | Session state — atlas + all loaded slices + AP estimates + registration results |
| `SliceSources`          | Per-slice struct — image data, current AP estimate, transform history |
| BigDataViewer           | Three.js in the webview                              |

### Key Rust responsibilities

1. **Atlas loading:** Read BrainGlobe atlas files on startup (TIFF volumes, JSON metadata, mesh files). Cache the full volume in memory.
2. **Reslicing on demand:** When the frontend requests a slice at a given AP position and angle, sample the 2D plane through the cached 3D volume and return an image. This is the hot path — must be fast for interactive use.
3. **Mesh serving:** Parse atlas region meshes (OBJ from BrainGlobe) via `tobj` and serve vertex/index buffers to Three.js. One-time load at startup.
4. **Border computation:** Edge-detect the label volume (`annotation.tiff`) to generate region boundary overlays on the fly, like ABBA does.
5. **Session state:** Track all loaded slice images, their current estimated positions, registration results, and transform history.
6. **Python orchestration:** Spawn the LangSlice Python process, send it tasks, relay progress events to the frontend.

### Rust crate stack

Mature crates for every core need. No greenfield libraries required — only glue code.

| Need | Crate | Maturity |
|------|-------|----------|
| 3D volume storage | `ndarray` | Very high |
| Linear algebra / transforms | `nalgebra` | Very high |
| Parallel computation | `rayon` | De facto standard |
| TIFF stack reading | `tiff` | Very high |
| OBJ mesh parsing | `tobj` | High |
| JSON metadata | `serde_json` | Standard |
| 2D edge detection | `imageproc` | High |
| nD filtering / morphology | `ndarray-ndimage` | Low-moderate |

### What must be built

**Core array math (~300 lines):**
1. **BrainGlobe atlas loader** — Glue code connecting `tiff` + `serde_json` + `tobj` to read the atlas directory structure into cached in-memory structs.
2. **Oblique reslicing** (~100-200 lines) — Define the cutting plane via `nalgebra` transforms, sample each output pixel from the 3D `ndarray` volume with trilinear interpolation (reference template) or nearest-neighbor (label volume), parallelize with `rayon`.

**Additional Rust (scope varies):** Session state management, Python subprocess orchestration, Tauri command handlers, and image downsampling for slice textures. These are standard application plumbing — straightforward but more code than the core math.

### BrainGlobe atlas file structure (what Rust reads)

```
~/.brainglobe/allen_mouse_25um_v1.2/
├── annotation.tiff      ← label volume (region IDs per voxel)
├── reference.tiff       ← structural/template volume (grayscale)
├── metadata.json        ← resolution, orientation, dimensions
├── structures.json      ← region hierarchy (names, IDs, colors)
├── additional_references/  ← optional extra channels (e.g. Nissl)
│   └── *.tiff
└── meshes/
    ├── 997.obj          ← whole brain outline
    ├── 8.obj            ← other regions...
    └── ...
```

### Atlas channels strategy

Follow ABBA's approach: load whatever BrainGlobe provides.

- **Always available:** `reference.tiff` (template) + borders computed on the fly from `annotation.tiff` (edge detection on label boundaries).
- **Optional:** Any volumes listed in `metadata.json` → `additional_references` (e.g. Nissl). Load at startup, expose as selectable channels in both GUI and CLI.
- Richer reference images (Nissl) help the AI agent match morphological features more accurately.

## Frontend — Three.js Scene

### Scene contents

- **Brain outline mesh:** Semi-transparent whole-brain surface from BrainGlobe meshes. Always visible.
- **Region meshes (optional):** Individual brain regions, toggled on/off, color-coded by structures.json.
- **Slice planes:** Up to 50+ textured quads positioned along the AP axis. Each shows the user's histology image at ≤1K resolution (Rust backend downsamples on load).
- **Atlas cross-section:** A 2D resliced atlas image as a textured plane, updated on slicing angle/position changes.
- **Camera:** Orbit controls (rotate, zoom, pan). Standard Three.js OrbitControls.

### Animation during AP estimation

- Slice plane teleports to each candidate position as the agent evaluates it (`plane.position.z = newApMm`).
- Visual feedback (color, glow, opacity) indicates estimation state: searching → converging → locked.
- Multiple slices estimating concurrently (up to 5 agents processing in parallel).

### Frontend framework

React + TypeScript + react-three-fiber. Declarative 3D via JSX components, type safety across the Tauri `invoke()` boundary, largest ecosystem for both UI components and 3D.

### Prior art

Multiple projects demonstrate Three.js brain meshes with orbit controls (threejs-brain-animation, lebrain-threejs, 3D-rat-brain, Yeo Atlas 3D). Community starter template exists for Tauri + React + react-three-fiber + Vite. "morgan-bevy" is a professional 3D level editor built with Tauri + React + Three.js + Rust, confirming the architectural pattern.

## Communication Patterns

### JS ↔ Rust (Tauri invoke + events)

```
# Request/response (JS calls Rust)
invoke("load_atlas", {name: "allen_mouse_25um"})
  → returns {meshes: [...], metadata: {...}}

invoke("reslice", {position_mm: 4.2, angle_deg: 0})
  → returns {image: base64_png, labels: [...]}

invoke("load_images", {paths: [...]})
  → returns {slices: [{id, thumbnail, size}, ...]}

# Event streaming (Rust pushes to JS)
emit("agent_update", {slice_id: 3, position_mm: 4.2, state: "searching"})
emit("registration_complete", {slice_id: 3, result: {...}})
```

### Rust ↔ Python (websocket or stdio)

```
# Rust sends task to Python
{"command": "estimate_ap", "slice_id": 3, "image_path": "...", "atlas": "allen_mouse_25um"}

# Python streams progress
{"event": "ap_update", "slice_id": 3, "position_mm": 4.2, "reasoning": "..."}
{"event": "ap_final", "slice_id": 3, "position_mm": 4.12, "reasoning": "..."}

# Rust sends registration task
{"command": "register", "slice_id": 3, "position_mm": 4.12, "workflow": "multimodal_tool_loop"}

# Python streams result
{"event": "registration_complete", "slice_id": 3, "correspondences": [...], "affine": {...}}
```

## Distribution & Installation

- `pip install langslice` or `conda install langslice` — core library + CLI
- `langslice estimate`, `langslice register` — headless CLI
- `langslice gui` — launches Tauri app (expects working Python environment with LangSlice installed)
- Other programs invoke LangSlice via its Python CLI — library first, GUI second
- Standalone installers (bundled Python runtime) are a future possibility

## Scope

This replaces the current PySide6 GUI entirely. The CLI remains unchanged as a headless interface.

The Tauri GUI is the **standalone user-facing application**. The Python library remains the **integration path** for ABBA, QUINT, and brainglobe-registration.

## Research Appendix

Detailed ecosystem research archived in `docs/rustreading/`:
- `rust_research_chatgpt.md` — Deep research on Rust crates, Tauri patterns, and web viewer ecosystem
- `rust_research_claude.md` — Independent deep research with crate maturity assessment and NiiVue analysis

Key conclusions (convergent across both reports):
- Format I/O is mature (tiff, tobj, serde_json cover BrainGlobe completely)
- Oblique reslicing: no ready-made crate, but ~100-200 lines from primitives
- No brain atlas crate in Rust — BrainGlobe loader is custom glue code
- Registration stays in Python — no pure Rust solution
- OrthoRay (Rust + Tauri + wgpu DICOM viewer, Feb 2026) proves the architecture is viable
- NiiVue: excellent for volume viewing, wrong for custom 3D scenes

## Open / Deferred Decisions

- **Python IPC:** Websocket vs stdio pipe. Start simple, upgrade if needed.
- **CLI parameters in GUI:** Agent system still in flux. Settings panel structure is defined (simplified + developer mode), parameter list TBD.
- **Export formats:** Depends on community integration. ABBA state file is a target.
- **Mesh serving format:** Raw OBJ, parsed JSON, or binary buffers. Prototype and see.
- **Atlas mesh LOD:** Check actual BrainGlobe mesh file sizes before deciding.
- **Atlas downloading:** Delegate to Python's `brainglobe-atlasapi` initially, move to Rust later if needed.
- **Multi-brain agent scheduling:** Shared agent pool vs dedicated agents per brain TBD.
