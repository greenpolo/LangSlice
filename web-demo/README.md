# LangSlice — GH Pages Demo

Browser-only LangSlice with Gemma 4 E4B running locally via Google's LiteRT-LM.

## What this is

This package is a static single-page app built from a Vite + React + TypeScript port of the desktop `tauri-gui/`. It bundles the `allen_mouse_25um` BrainGlobe atlas and one Walsh Lab demo brain (M01) as static assets, and runs the LangSlice position-estimation agent loop entirely in TypeScript. Gemma 4 E4B inference is delegated to a judge-run `litert-lm serve` sidecar on `http://127.0.0.1:8765`, which the page reaches over loopback. Optional image-gen registration (enabled by pasting a Gemini API key into Settings) uses `gemini-3.1-flash-image-preview` (Nano Banana 2) plus `@itk-wasm/elastix` for the dense deformation step. No backend, no install — the entire app is served as a static bundle from GitHub Pages, and the only runtime requirement on the judge's machine is the LiteRT-LM sidecar described below.

## LiteRT award framing

This demo was built for the Gemma 4 Good Hackathon's LiteRT award category. As of 2026-05, Gemma 4 vision on the LiteRT web runtime (MediaPipe Web) is text-only — see [LiteRT-LM issue #2150 (Gemma 4 vision support on web)](https://github.com/google-ai-edge/LiteRT-LM/issues/2150). LangSlice needs the vision tower to ground per-section position estimates, so we drive the native LiteRT runtime via loopback (`litert-lm serve`) instead of the in-browser build. The local sidecar is the deliberate architectural choice for the LiteRT category: it keeps inference on-device under LiteRT, while letting the browser app render the full atlas + 3D viewer + agent UI as a static SPA. When the MediaPipe Web build ships Gemma 4 vision, the browserCommands shim can switch over without touching the agent loop.

## Setup for judges

1. Download `litert-lm` from [Google's official LiteRT-LM release page on GitHub](https://github.com/google-ai-edge/LiteRT-LM/releases).
2. Download the Gemma 4 E4B weights: [gemma-4-E4B-it.litertlm (3.66 GB) on Hugging Face](https://huggingface.co/litert-community/gemma-4-E4B-it-litert-lm).
3. Run it locally:

   ```
   litert-lm serve --port 8765 --backend gpu --model /path/to/gemma-4-E4B-it.litertlm
   ```

The browser app polls `http://127.0.0.1:8765/v1/models` every 2 seconds; the Setup overlay disappears and the dashboard activates as soon as the sidecar answers. Chrome 142+ shows a Local Network Access permission prompt on the first request — grant it.

## Optional: image-gen registration

Paste your Gemini API key into Settings → API Keys to unlock the dense registration path. Image-gen runs in the background after each AP estimate, swapping the warped-atlas overlay onto the 3D viewer when complete. Without a key, quick-affine is the only registration path (fast, silhouette-based, always available).

## Browser support

Chrome 142+ desktop only. The demo uses Chrome's new Local Network Access permission model for the HTTPS→HTTP loopback; Firefox and Safari don't support it. Mobile is out of scope.

## What's bundled

The full static bundle ships the atlas, one demo brain, and the app itself.

| Asset | Size |
| --- | --- |
| Atlas — 528 native-step coronal sections (total) | ~42 MB |
| &nbsp;&nbsp;`sections/` (Nissl, RGBA) | 30 MB |
| &nbsp;&nbsp;`colored/` (per-region RGBA) | 3.3 MB |
| &nbsp;&nbsp;`borders/` (cyan overlay) | 4.1 MB |
| &nbsp;&nbsp;`mesh/root.obj` | 4.5 MB |
| &nbsp;&nbsp;`structures.json` + `manifest.json` | ~180 KB |
| Demo brain — M01, 20 RGB sections at 2K long-edge | 63 MB |
| App JS/CSS (estimate) | ~3 MB |
| `@itk-wasm/elastix` WASM (lazy-loaded on first image-gen run) | ~15–20 MB |
| **Cold-start total** | **~108 MB** |

## Local development

### Prereqs

- pnpm 10.33+
- Node 22+

### Setup

From `web-demo/`:

```
pnpm install
```

### Dev server

```
pnpm dev
```

Vite runs on `http://localhost:5173`. You'll still need a `litert-lm serve` running locally for the agent loop to work; without it the SetupOverlay stays up.

### Build

```
pnpm build
```

Emits `web-demo/dist/`. Set `VITE_BASE_PATH=/<repo-name>/` if deploying to a project page (the CI workflow does this automatically).

## Pre-baked assets

The static atlas and demo brain assets under `public/` are pre-rendered by two Python scripts that do NOT run in CI; you need them locally if you want to re-bake them:

- `python web-demo/scripts/build_atlas_bundle.py` — bakes the atlas section, colored, borders, mesh, structures, and manifest files.
- `python web-demo/scripts/build_demo_brain.py` — bakes the demo brain PNGs and manifest from `C:/WalshLab/SIArevision/...` (Walsh Lab data, dev machine only).

Both scripts run from the langslice harness environment (`pip install -e .` at the repo root).

## Architecture map

The TypeScript port keeps the same call shapes as the desktop harness so the agent loop reads 1:1 against `single_slice.py`. The browser-only entry points are:

- `src/lib/browserCommands.ts` — static-manifest replacement for the Tauri IPC surface.
- `src/lib/sidecarProbe.ts` — litert-lm CORS and reachability probe.
- `src/lib/agentLoop.ts` — TypeScript port of `single_slice.py`, drives Gemma 4 via `@google/genai` and the loopback baseUrl.
- `src/lib/quickAffine.ts` — pure-JS silhouette-based affine register (Canvas2D, no WASM).
- `src/lib/imageGenRegistration.ts` — Gemini image-gen + `@itk-wasm/elastix` for the dense deformation.
- `src/components/SetupOverlay.tsx` — sidecar-not-running curtain.

## Out of scope

Multi-slice, atlas swap, model swap, mobile, Firefox/Safari, telemetry, login.
