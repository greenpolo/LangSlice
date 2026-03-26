# Rust and Tauri ecosystem for brain atlas visualization

**The Rust ecosystem offers solid low-level building blocks for a brain atlas desktop app — but no turnkey solution exists.** Format I/O (NIfTI, TIFF, DICOM) is well-covered by actively maintained crates, and `ndarray` + `nalgebra` provide a strong computational foundation. However, critical gaps remain: no NRRD reader, no oblique reslicing library, no brain atlas crate, and no 3D medical image registration in pure Rust. The most practical architecture for a Tauri-based viewer is a hybrid: embed **NiiVue** (WebGL 2.0) in the webview for visualization, with a Rust backend for heavy computation and data handling. One real-world precedent — OrthoRay, a Rust + Tauri + wgpu DICOM viewer announced in February 2026 — proves the approach is viable.

---

## Format I/O crates are mature enough for production use

The most critical crate for this project is **`nifti` (nifti-rs)**, a pure Rust NIfTI-1.1 reader/writer by Eduardo Pinho. It has **51 GitHub stars**, integrates directly with `ndarray` (volumes become `Array<T, IxDyn>`) and `nalgebra` (affine transforms), and supports `.nii`, `.nii.gz`, and `.hdr/.img` pairs. Version 0.17.0 shipped June 2025. The main limitation is **no NIfTI-2 support**, though NIfTI-1 covers most brain atlas use cases. A companion crate, `nifti_processing` (~3,200 downloads), provides NiBabel-style `resample_to_output()` and `resample_from_to()` with nearest-neighbor and trilinear interpolation — useful but minimal.

For TIFF stacks, the **`tiff` crate** (part of image-rs) is excellent: **53.7 million all-time downloads**, full BigTIFF support, multi-page IFD iteration, and 8/16/32/64-bit integer and float pixel types. The critical caveat is that the higher-level `image` crate loads only the first page by default — you must use `tiff::decoder` directly to iterate through a TIFF stack and assemble slices into a 3D `ndarray::Array3`. The `tiffwrite` crate complements this with BioFormats/ImageJ-compatible BigTIFF writing, including 5D hyperstack dimensions and Zstd compression.

**DICOM support is the most mature medical imaging capability in the Rust ecosystem.** The `dicom` crate (dicom-rs) has **~488 GitHub stars**, **689,000+ downloads**, and covers reading, writing, pixel data decoding to ndarray, DICOM networking, JSON serialization, and WebAssembly builds. It's production-grade but primarily relevant if the application needs to ingest clinical data.

**NRRD format support does not exist in Rust.** Extensive searching across crates.io, GitHub, and lib.rs found no NRRD reader. This is a significant gap given NRRD's widespread use in 3D Slicer and scientific visualization. You would need to write a custom parser (the NRRD spec is relatively simple: a text header followed by raw data) or use FFI to the C Teem library.

| Crate | Format | Maturity | Downloads | Read | Write | 3D | Last updated |
|---|---|---|---|---|---|---|---|
| `nifti` | NIfTI-1.1 | Moderate | Moderate | ✅ | ✅ | ✅ | Jun 2025 |
| `tiff` | TIFF/BigTIFF | Very high | 53.7M | ✅ | ✅ | Via multipage | Active |
| `dicom` | DICOM | High | 689K | ✅ | ✅ | Per-slice | Active |
| `neuroformats` | FreeSurfer | Low | Low | ✅ | ❌ | Surface meshes | ~2024 |
| `tiffwrite` | BigTIFF | Niche | Low | ❌ | ✅ | 5D | Active |
| `medrs` | NIfTI (perf) | Very early | Low | ✅ | ✅ | ✅ | New |
| — | **NRRD** | **None** | — | ❌ | ❌ | — | — |

---

## No oblique reslicing crate exists — you must build from primitives

The core operation for a brain atlas viewer — sampling a 2D plane at an arbitrary angle through a 3D volume (multiplanar reconstruction / oblique reslicing) — **has no ready-made Rust implementation**. The building blocks exist but must be composed manually:

**`ndarray`** provides `Array3<T>` for volume storage with powerful slicing via the `s![]` macro, zero-copy views, and axis iteration. **`nalgebra`** handles the linear algebra: `Affine3<f64>` for homogeneous transforms, `Rotation3` and `UnitQuaternion` for 3D rotations, and `Point3` for coordinate mapping between voxel and world space. **`rayon`** enables trivial parallelization — replace `.iter()` with `.par_iter()` for per-pixel or per-row parallel interpolation.

The implementation pattern for oblique reslicing is straightforward: define the output plane's origin and two in-plane direction vectors, map each output pixel to a 3D world coordinate via an affine transform, then sample the volume using trilinear interpolation. The `nifti_processing` crate demonstrates this pattern for axis-aligned resampling, and `ndarray-ndimage` (a partial scipy.ndimage port) provides 3D convolution, Gaussian filtering, and morphological operations — but **neither provides arbitrary-angle plane extraction**. The `ndarray-interp` crate (232,000 downloads) offers interpolation along one axis of an ndarray, which is insufficient for true 3D sampling.

For trilinear interpolation specifically, the math is simple enough to implement in ~50 lines of Rust: fetch the 8 corner voxels of the enclosing cube, blend along x, then y, then z. Parallelizing this over output pixels with rayon makes it performant. The `medrs` crate (very new, by Liam Chalcroft) claims to offer trilinear resampling with lazy evaluation and operation fusion — and makes dramatic performance claims (38,000× vs. MONAI) — but its maturity is unproven.

---

## Brain atlas and neuroimaging tooling is entirely greenfield in Rust

**No Rust crate exists for brain atlas handling, BrainGlobe compatibility, or atlas region lookup.** BrainGlobe is a purely Python ecosystem storing atlases as NumPy `.npy` arrays, TIFF stacks, and JSON metadata. Reading these from Rust is straightforward (TIFF stacks via `tiff`, JSON via `serde_json`, mesh files via `tobj`) but no one has built a BrainGlobe-compatible Rust interface.

The closest project is **`neuroformats`** (v0.2.1), which reads FreeSurfer surface meshes (`.surf`), per-vertex curvature data, annotations (`.annot`), and MGH/MGZ brain volumes. Its author, Tim Schäfer, also maintains Julia and R equivalents. A companion project, **`brainview-rs`**, attempts to visualize FreeSurfer surfaces using the `three-d` rendering crate, but the author explicitly warns "This is WIP, come back another day" — it has **3 GitHub stars** and is not published on crates.io.

**ITK and VTK bindings are essentially nonexistent.** `vtk-rs` exists on GitHub but its author states it is "probably unusable for now" — it covers a tiny fraction of VTK's API and requires VTK ≥9.1 with a complex CMake build. `vtkio` (moderate maturity) parses VTK file formats in pure Rust but provides no computational or rendering functionality. **No ITK Rust bindings exist** — ITK's heavy C++ template metaprogramming makes automated binding generation extremely difficult. The `sitk-registration-sys` crate wraps SimpleITK via autocxx but requires **up to 50 GB of disk space** to compile, making it impractical for distribution.

One promising project for registration is **`modern-icp`** (v0.5.0, actively developed as of early 2026), a pure Rust ICP implementation with point-to-plane, bidirectional distance, and sigma rejection. For 2D affine image transforms, **`imageproc`** (part of image-rs) provides `warp()` with bilinear interpolation. But **no pure Rust crate handles 2D-to-3D slice registration** — the core algorithmic challenge of histology registration to a brain atlas.

---

## NiiVue is the strongest candidate for Tauri webview integration

Among WebAssembly and web-based neuroimaging viewers, **NiiVue stands out as the clear best fit** for embedding in a Tauri webview. It uses **WebGL 2.0** (which Tauri's native webviews support on all platforms), has **~430 GitHub stars** and **3,676 commits**, supports **30+ file formats** (NIfTI, NRRD, MRtrix MIF, MGH/MGZ, AFNI, DICOM via plugin), and provides GPU-accelerated oblique slicing, volume rendering, atlas overlays, mesh rendering (OBJ, GIfTI, PLY, STL, FreeSurfer surfaces), and draw tools. An **Electron-based desktop version already exists** (`niivue/desktop`, 9 stars) that could serve as a migration template for Tauri.

NiiVue's architecture is minimal — it attaches to an HTML5 canvas element and exposes a rich JavaScript API for programmatic control. It has no Three.js dependency, using direct WebGL 2.0 calls optimized for voxel rendering. The NiiVue-UI project provides a React MUI reference interface. For a Tauri app, the architecture would be: NiiVue in the webview for visualization, Rust backend for heavy computation (volume loading, resampling, registration), with Tauri's IPC bridge passing data between them.

| Viewer | Tech | Tauri fit | Oblique slicing | Atlas overlays | Status |
|---|---|---|---|---|---|
| **NiiVue** | WebGL 2.0, TypeScript | **High** | ✅ GPU-accelerated | ✅ Multi-overlay | Active (Feb 2026) |
| Neuroglancer | WebGL 2.0, TypeScript | Low | ✅ | ✅ | Active, cloud-oriented |
| AMI.js | Three.js (r81) | Moderate | ✅ | Limited | Aging |
| Papaya | Canvas 2D | Easy but limited | ❌ | ✅ | Dormant |
| BrainBrowser | Three.js + Canvas | Moderate | Limited | Via data mapping | 2015-era |

**Neuroglancer** (Google) is powerful but designed for cloud-scale data with CORS requirements and complex TypeScript coupling, making Tauri integration difficult. **AMI.js** references Three.js r81 (current is r169+), indicating severe staleness. **Papaya** uses Canvas 2D with no GPU acceleration and no oblique slicing.

---

## OrthoRay proves the Rust + Tauri + wgpu architecture works

The most significant precedent is **OrthoRay**, announced in February 2026 on the Rust Users Forum by an orthopedic surgeon ("Mrmeric"). Built with **Rust + Tauri + wgpu**, it implements real-time 3D volume rendering via wgpu compute shaders, multiplanar reconstruction (MPR) with coronal/sagittal/axial views, and a custom bone visualization algorithm. The developer reported that "wgpu's compute shaders were surprisingly well-suited for volume raycasting" and found DICOM parsing via existing Rust crates "more mature than expected." OrthoRay is available on the Microsoft Store; its source code was briefly shared on GitHub (`Mrmeric/medical-imaging-rs`) but availability is uncertain.

For rendering architecture in a Tauri app, three approaches exist:

- **WebGL 2.0 in the webview** (recommended baseline): Use NiiVue or custom WebGL shaders. Broadly supported, leverages the web ecosystem, but limited to webview GPU capabilities.
- **Hybrid wgpu + webview**: Tauri v2 supports multiple surfaces in a single window. The `tauri-wgpu-cam` project demonstrates rendering wgpu textures to a native surface alongside a webview UI with event-based interop. Provides full native GPU access but is an early-stage pattern.
- **Full native wgpu rendering**: All visualization in Rust via wgpu; webview for UI controls only. Maximum performance (OrthoRay's approach) but requires building the rendering pipeline from scratch.

**WebGPU support in Tauri is platform-fragmented**: confirmed on Windows (WebView2), uncertain on macOS (WKWebView), and unavailable on Linux (WebKitGTK). WebGL 2.0 is the safe cross-platform choice.

---

## OBJ mesh parsing is well-served by tobj

For parsing brain atlas mesh files in OBJ format, **`tobj`** is the clear recommendation: **267 GitHub stars**, **1.24 million downloads**, actively maintained (v4.0.3, January 2025), with support for vertices, normals, texture coordinates, vertex colors, materials, groups, on-the-fly triangulation, and async/WebAssembly loading. It is read-only — for write support, `wavefront_rs` (v1.0.4, ~39,500 downloads) is the only option, though it is older and less actively maintained.

For other mesh formats relevant to brain atlases: the **`gltf` crate** (very high maturity) handles glTF 2.0 including GLB binary, `stl_io` reads and writes STL files, `ply-rs` handles PLY format, and `mesh-loader` provides multi-format support (STL, COLLADA, OBJ) oriented toward robotics. If the application ingests BrainGlobe atlas meshes (which come as OBJ files), tobj covers the need completely.

---

## The recommended computational stack and what you'll need to build

The foundational Rust crate stack for this application is well-established:

- **`ndarray`** (Array3 for volumes, slicing, views, broadcasting) — very high maturity, the NumPy of Rust
- **`nalgebra`** (affine transforms, rotations, coordinate systems) — very high maturity, 4,000+ stars
- **`rayon`** (parallel iteration for voxel operations) — de facto standard, drop-in parallelism
- **`nifti`** (NIfTI-1 I/O with ndarray integration) — moderate maturity, actively maintained
- **`tiff`** (multi-page TIFF / BigTIFF stack reading) — very high maturity, 2.9M monthly downloads
- **`tobj`** (OBJ mesh loading) — high maturity, 1.24M downloads
- **`imageproc`** (2D affine warps with bilinear interpolation) — high maturity, part of image-rs
- **`nshare`** (zero-copy conversion between ndarray, nalgebra, and image types) — useful glue crate
- **`ndarray-ndimage`** (3D convolution, Gaussian filtering, morphology) — low-moderate maturity, WIP

What you will need to **build yourself or find alternative solutions for**:

1. **NRRD reader**: ~200–400 lines of Rust for the header parser + raw data reader. The format spec is straightforward (text header with `key: value` pairs, then raw or gzip-compressed data).
2. **Oblique reslicing / MPR**: Define cutting plane via nalgebra transforms, sample each output pixel from the 3D ndarray volume with trilinear interpolation, parallelize with rayon. Core implementation is ~100–200 lines.
3. **BrainGlobe atlas loader**: Read the atlas directory structure (JSON metadata, TIFF reference/annotation volumes, OBJ meshes) using serde_json + tiff + tobj.
4. **Histology-to-atlas registration**: The hardest problem. No pure Rust solution exists. Options: (a) call Python/BrainGlobe tools via subprocess, (b) use `sitk-registration-sys` if build size is acceptable, (c) implement affine registration from scratch using nalgebra + a cost function optimizer like `argmin` crate, or (d) use the Rust backend for the UI and delegate registration to a Python sidecar process.
5. **3D volume visualization**: Either embed NiiVue in the Tauri webview (recommended) or build a wgpu rendering pipeline (high effort, OrthoRay as reference).

---

## Community is tiny but growing, with key venues identified

The intersection of Rust and neuroimaging is **extremely small** — perhaps a dozen active practitioners worldwide. No discussions were found on neurostars.org, image.sc, or Reddit r/neuroimaging about Rust-based tools. The Rust Users Forum thread on OrthoRay (February 2026) attracted modest engagement (6 likes, ~6 replies). Eduardo Pinho (author of dicom-rs and nifti-rs, software engineer at BMD Software in Portugal) is the single most influential figure, having authored the two most important medical imaging crates and advocated publicly for Rust in this domain.

Key venues for this nascent community include the **Scientific Computing in Rust** workshop (scientificcomputing.rs, held annually since 2023), the **DICOM-rs Zulip** chat, and the **Rust-SciComp Zulip**. The ChRIS project at Boston Children's Hospital presented at the 2024 Scientific Computing in Rust workshop on DICOM workflows, signaling institutional interest. The general sentiment is that **Rust's performance and safety make it ideal for medical imaging, but the ecosystem requires significant investment** — dicom-rs demonstrates that sustained effort by even one dedicated developer can produce production-quality results.

## Conclusion

Building a brain atlas visualization and histology registration app in Rust + Tauri is ambitious but feasible. The **format I/O layer** is solid (NIfTI via nifti-rs, TIFF stacks via the tiff crate, OBJ meshes via tobj). The **computational core** (ndarray + nalgebra + rayon) is production-grade. The **visualization layer** has a clear path: embed NiiVue in the Tauri webview for proven, feature-rich neuroimaging visualization with oblique slicing and atlas overlays, with the Electron desktop version as a migration reference. The **registration pipeline** is the biggest challenge — no pure Rust solution exists, and a Python sidecar or subprocess approach for BrainGlobe/SimpleITK integration may be the most pragmatic path.

The three most impactful architectural decisions are: (1) use NiiVue in the webview rather than building a custom renderer, saving months of development; (2) implement a thin NRRD parser rather than waiting for a community crate; and (3) design the Rust backend as a compute engine that passes processed data (resliced images, transformed coordinates, loaded volumes) to the webview via Tauri IPC, keeping the visualization layer in JavaScript where the neuroimaging viewer ecosystem is richest.