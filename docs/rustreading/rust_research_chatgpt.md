# Deep Research: Rust + Tauri Backend Options for BrainGlobe Atlas Visualization and Slice Registration

## BrainGlobe atlas file conventions and what your backend must ingest

A BrainGlobe atlas installation is intentionally “filesystem-first”: each atlas lives in its own directory under the hidden `~/.brainglobe` folder, and the atlas directory contains a small, standardized set of files you can read without using Python. By default, BrainGlobe atlases are downloaded to `~/.brainglobe/<atlas_name>_<resolution>_v<version>` (example given in the docs: `~/.brainglobe/allen_mouse_10um_v0.3`). citeturn9search0turn9search1

The BrainGlobe docs explicitly enumerate the canonical contents of each atlas directory, which matches your planned Rust ingestion layer:

- `reference.tiff`: the “template” structural volume that annotations are defined in citeturn9search0  
- `annotation.tiff`: same shape as the reference, with integer region IDs per voxel (0 typically outside brain) citeturn9search0turn9search6  
- `hemispheres.tiff` (optional): hemisphere labels for asymmetric templates citeturn9search0  
- `meshes/` directory: one mesh per region, stored as `.obj` files keyed by region ID citeturn9search0  
- `structures.json`: region ID → region name mapping plus hierarchy (via `structure_id_path`) citeturn9search0turn9search6  
- `metadata.json`: shape, orientation, resolution, provenance, and related metadata citeturn9search0turn9search1  
- `README.txt`: a human-readable dump of metadata and hierarchies citeturn9search0  

BrainGlobe supports (and increasingly expects) **additional reference volumes** (e.g., alternate stains / modalities). In BrainGlobe’s core `Atlas` implementation, the loader attempts to construct an `AdditionalRefDict` from `metadata["additional_references"]`, and warns if the list is missing (treated as an outdated atlas). citeturn26search12  
This matters for your Rust design because it implies: **don’t hard-code “just reference + annotation”**—instead, plan for an arbitrary set of additional TIFF volumes referenced from `metadata.json`. citeturn26search16turn26search12  

For context (and validation that this standardization is intentional), the BrainGlobe Atlas API project describes atlases as a common-format bundle of TIFF reference/annotation volumes, OBJ meshes, and JSON metadata/hierarchy. citeturn9search1turn26search30  
The project’s own citation block points to the BrainGlobe Atlas API paper (useful if you later publish your Rust/Tauri tool and want compatible terminology). citeturn9search2  

**Implication for your Rust backend:** you can treat BrainGlobe atlas loading as a deterministic local import problem: parse JSON (`metadata.json`, `structures.json`), read one or more TIFF volumes, and index OBJ files by integer region ID. The “compatibility” bar is primarily correct interpretation of these files, not any hidden database or cloud API. citeturn9search0turn26search12  

## Rust crates for volume I/O and in-memory representation

### TIFF voxel volumes (multipage stacks) and performance-relevant readers

For BrainGlobe-style TIFF volumes, the common pattern is “a 3D stack stored as a multipage TIFF.” Two Rust options stand out:

- The `tiff` crate provides a TIFF decoder with explicit support for stepping through images: `next_image()` reads the next image, while `more_images()` reports whether additional images remain. This directly matches multipage stack traversal. citeturn26search4  
- The `tiff_reader` crate is a separate pure-Rust, **read-only TIFF/BigTIFF decoder** designed around random access sources (including memory-mapped files) and includes an **LRU cache for decompressed strips and tiles**, returning the decoded image into an `ndarray::ArrayD`. This architecture is closer to what you want for interactive reslicing because it has “cursor + cache” semantics rather than “decode everything eagerly.” citeturn5view2  

If you anticipate very large volumes, the “how do I avoid decoding the whole stack just to reslice a plane?” question is mostly about **file layout + caching strategy**. A TIFF reader that can partially decode strips/tiles and cache them is substantially more aligned to that need than a minimal decoder. The existence of the `tiff_reader` cache module and its random-access abstraction is evidence this approach is feasible in Rust today. citeturn5view2  

### NIfTI and DICOM, if you choose to support common neuroimaging/medical formats

Even though BrainGlobe itself is TIFF+JSON+OBJ, neuroimaging tools often benefit from accepting common formats as imports/exports or for “bring your own volume.”

For NIfTI:

- `nifti` (NIFTI-rs) is a pure Rust implementation targeting NIfTI-1.1, supports reading `.nii.gz`, can convert volumes into `ndarray` via the `ndarray_volumes` feature, and includes an optional affine feature (`nalgebra_affine`) for spatial transforms. citeturn8search0  
- `nii-rs` positions itself as a higher-level NIfTI-1 library with an API inspired by SimpleITK / NiBabel and advertises Rust↔Python bindings. (It is built “thanks to nifti-rs”.) citeturn8search1turn17search5  

For DICOM:

- The `dicom-rs` ecosystem provides a pure Rust DICOM implementation. Its `pixeldata` crate can decode pixel data (with multiple transfer syntaxes) and convert decoded pixel arrays into a flat vector, an `ndarray` multi-dimensional array, or an `image::DynamicImage`. citeturn8search11turn8search7  

These crates don’t “solve BrainGlobe,” but they do cover your question about **existing Rust crates for reading / manipulating 3D scientific image volumes**: NIfTI and DICOM are relatively well-covered compared to more niche formats. citeturn8search0turn8search11  

### Bio-Formats / OME-TIFF route for microscopy ecosystems

If you later ingest histology images from microscopy pipelines, you may encounter OME-TIFF and proprietary microscopy formats. In that ecosystem, Bio-Formats is a major baseline (Java). The Bio-Formats repository describes itself as a Java library for reading/writing life sciences image formats and is GPL-licensed (with commercial licenses available). citeturn6search6  

A Rust option exists as a wrapper:

- `bioformats-rs` (Rust crate) describes itself as a high-level API to read images using Bio-Formats and notes a fixed Bio-Formats library version (8.0.1). citeturn6search5turn6search1  

Separately, the OME documentation describes OME-TIFF as leveraging classic multipage TIFF organization while embedding OME-XML metadata in the file header. citeturn6search9  

**Implication:** if you consider “just use Bio-Formats,” you should factor in (a) JVM embedding expenses and (b) GPL licensing constraints for redistribution, as those are explicit in the upstream project. citeturn6search6turn6search5  

## Arbitrary-angle reslicing and resampling in Rust

You identified arbitrary-angle reslicing as the interactive hot path. Conceptually, this is “sample a 3D scalar field at a dense grid of points lying on a plane,” with interpolation (trilinear for continuous intensities, nearest-neighbor for labels). That pipeline exists in Rust today, but mostly as **building blocks** rather than a single canonical crate.

### A concrete existing implementation: NIfTI resampling extension with trilinear/nearest samplers

A notable Rust implementation of 3D resampling is `nifti-processing-rs`, an extension library that adds `resample_to_output` / `resample_from_to` with nearest-neighbor and trilinear resampling, modeled after NiBabel’s processing module. citeturn10view0turn8search17  

Looking at the implementation details (important for your “hot path” concerns):

- In `resample_from_to`, the code generates all output voxel coordinates from `out_shape` (via a Cartesian product), applies an affine transform, then calls `sampler.sample(...)`. The source includes an explicit note: “generation of all coords is not very fast,” which is exactly the kind of performance caveat you want to know before adopting this approach directly for interactive slicing. citeturn13view1turn13view0  
- The trilinear sampler (`TriLinear`) is implemented as a `ReSample` trait implementation, uses `rayon` parallel iterators, checks bounds, and returns a constant value outside bounds. That demonstrates a workable approach for plane sampling: precompute a coordinate matrix, clamp or constant-fill, then interpolate. citeturn15view0turn15view2  

**How this applies to your BrainGlobe reslicer:** even if you don’t use NIfTI, the same strategy applies to a TIFF-backed `ndarray` volume: generate plane coordinates (2D pixel grid → 3D positions), apply transform, sample with nearest/trilinear.

However, if you need low latency, you probably want to avoid two things visible in the library’s generic approach: allocating very large coordinate buffers and performing a global “generate all coords” for anything bigger than your slice plane. The code itself signals this as an optimization target. citeturn13view1turn13view0  

### Alternative / supporting crates: nD arrays, interpolation, and image remapping

If you decide to implement plane sampling yourself, the usual Rust stack is:

- `ndarray` as your core nD container. citeturn8search14  
- `ndarray-ndimage` for nD filtering and morphology. While it does not directly advertise “arbitrary plane slicing,” it includes operations like convolution, Sobel/Prewitt filters, and `zoom` that can support resampling-related workflows. citeturn7view0  
- Dedicated interpolation crates. For example, `nd_interpolate` advertises 1–10D interpolation with linear and cubic spline variants. citeturn21search13  
  Separately, `ndarray-interp` provides interpolation strategies (documented for 1D/2D explicitly) and a trait-based API for custom strategies. citeturn7view2  

A different angle is to use 2D remapping primitives as part of a two-step pipeline (useful if your plane is axis-aligned after transform, or if you’re working on already-extracted 2D images):

- The Rust `kornia` crate exposes an `imgproc::interpolation::remap` function for applying generic geometric transformations to an image using per-pixel `map_x/map_y` coordinates and an interpolation mode. This is 2D, not 3D volume slicing, but it is a ready-made “dense coordinate maps + interpolation” implementation. citeturn23view0  

### GPU considerations inside a Tauri app

You asked whether established architectures (or web-based viewers) could be reused in a Tauri webview. A key constraint is GPU API availability:

- Tauri community discussions and issues indicate demand for WebGPU in Tauri (particularly for complex 3D/scientific workloads), but the existence of an open issue “WebGPU support?” indicates this is not something to assume is uniformly available across all Tauri WebView backends. citeturn19search6  

Practically, this pushes many Tauri+3D apps to use WebGL (via Three.js) in the webview, or to use native `wgpu` in Rust for rendering (outside the webview), depending on the architecture.

## Border computation and label-volume edge overlays

For atlas label borders (region boundary overlays), the core operation is detecting transitions in an integer label field. You can do this either:

- “Analytically” (compare each voxel/pixel to its neighbors and mark edges when labels differ), or  
- “Image-processing style” (morphological gradient / Sobel-like operators after casting), often performed on the **2D label slice** you already resliced from the 3D label volume.

Rust has relevant, production-usable pieces:

- `ndarray-ndimage` includes Sobel/Prewitt filters and nD convolution as well as 3D binary morphology ops. That supports both edge-finding and morphological-style boundary extraction (particularly if you treat a mask per label or a binarized region) without requiring external C/C++ dependencies. citeturn7view0  
- For 2D, `imageproc` has an edge module with a documented Canny implementation (`imageproc::edges::canny`) that returns a binary edge image. citeturn27search0turn27search4  
  `imageproc` also documents broader support modules, including a `region_labelling` module for connected components and related operations—useful if you later want to clean boundaries, remove tiny components, or post-process segmentation overlays. citeturn27search11turn27search3  

**Implication:** for BrainGlobe borders, the “Rust ecosystem gap” is not the absence of edge detectors; it’s more about the lack of neuroimaging-specific conventions (e.g., label semantics, hierarchy-aware boundary coloring). The low-level operations exist, but you’ll likely build the “atlas semantics” layer yourself. citeturn7view0turn27search0  

## Mesh serving: OBJ parsing and feeding a Three.js frontend through Tauri

BrainGlobe region surfaces are stored as OBJ meshes in a `meshes/` directory, with one mesh per region keyed by region ID. citeturn9search0turn26search12  
Given that format, your Rust backend needs: (a) an OBJ parser and (b) a serving strategy to get geometry into the WebView quickly.

### OBJ parsing crates in Rust

Several Rust crates parse Wavefront OBJ:

- `tobj` describes itself as a “tiny OBJ loader” returning vectors of models and materials (in the style of tinyobjloader). citeturn16search0turn16search5  
- `wavefront_obj` provides parsers for `.obj` and `.mtl`, explicitly noting “best-effort” support oriented around common exporter subsets (e.g., Blender). citeturn16search1turn16search24  
- `obj-rs` provides a Wavefront OBJ parser handling both `.obj` and `.mtl` with examples showing it loads vertices and indices. citeturn16search4turn16search7  

For BrainGlobe meshes (region surfaces), you typically care about vertices, normals (if present), and triangle indices. All three crates can plausibly serve that, with differences in ergonomics and OBJ-subset support.

### Serving strategy in Tauri

Tauri provides a command system for calling Rust functions from the frontend and returning results, which is the normal bridge for “load mesh / load slice / return typed array.” citeturn16search13  
Tauri’s architecture documentation emphasizes that apps are built from a Rust core plus HTML rendered in a WebView with message passing between the two. citeturn26search23  

This means you can run a Three.js frontend (or react-three-fiber) in the WebView and do any of these patterns:

- Return raw OBJ text and let the frontend parse (simple, but pushes CPU to JS and repeats parsing work).
- Parse OBJ in Rust to a compact structure (positions, normals, indices) and return as JSON (easy but can be large).
- Parse OBJ in Rust and return binary buffers (e.g., vertex and index buffers) for efficient upload into WebGL.

While you asked specifically about “existing Tauri plugins or templates for scientific 3D visualization,” there is not (based on the sources reviewed) a domain-specific official “scientific visualization plugin.” The more common pattern is: use web libraries for rendering and use Tauri plugins for OS integration.

Tauri plugin development is explicitly supported (plugins can hook into lifecycle and expose Rust code). citeturn26search7turn26search17  
For project scaffolding, Tauri’s official tooling (`create-tauri-app`) supports many standard frontend templates (React/Vue/etc.). citeturn19search7  

Community examples closer to your use case include:

- A community starter template combining Tauri + React + react-three-fiber + Vite, explicitly described as a “desktop 3D boilerplate starter.” citeturn19search1  
- A more substantive real-world repo: “morgan-bevy,” described as a professional 3D level editor built with Tauri + React + Three.js + Rust. This is strong evidence that “Tauri + Three.js + heavy 3D workflows” is an established architectural pattern. citeturn19search24  

## Bindings to established imaging toolkits: SimpleITK and VTK are real; ITK is “possible but heavy”

Your question about bindings to ITK/VTK/SimpleITK is important because it determines whether you should “wrap a mature C++ imaging stack” vs “write targeted Rust code.”

### SimpleITK bindings in Rust

There is a Rust crate specifically wrapping parts of SimpleITK:

- `sitk-registration-sys` documents that it uses SimpleITK (C++), builds an adapter library using autocxx, and warns that compilation requires significant time and resources (several GB memory; up to ~50 GB disk; plus CMake, a C++ compiler, LLVM, git). citeturn18search6turn17search4  

This is the clearest “yes, serious C++ imaging integration exists” datapoint, but the build-cost warning is a major practical consideration for a cross-platform desktop app.

### VTK bindings and VTK file I/O

For VTK, there are two distinct needs: **(a) using VTK algorithms/rendering** vs **(b) reading/writing VTK files**.

- `vtk-rs` is a Rust bindings effort for the Visualization Toolkit (VTK). The repository states a goal of “safe and thin bindings,” and there is an associated VTK community forum thread by the same author describing the binding-generation challenges and approach. citeturn17search2turn17search9turn17search29  
- `vtkio` is a Rust parser/writer for VTK file formats (legacy + XML) and explicitly describes itself as feature-complete for those file formats. This is not VTK-the-toolkit, but it is useful if you want to export meshes/volumes to the broader visualization ecosystem. citeturn17search13turn17search7  

### Autocxx as the “path” to ITK/VTK-class C++ libraries

`autocxx` is explicitly positioned as a tool for calling C++ from Rust with automated binding generation, integrating bindgen-like parsing with the `cxx` bridge for safety/ergonomics. citeturn18search4turn18search10  

**Implication:** from the evidence above, you can realistically choose among three tiers:

- Pure Rust (fast build, fewer features): implement only what you need (plane slicing + label borders + OBJ parsing).  
- Hybrid Rust + targeted bindings (moderate build complexity): SimpleITK for registration / interpolation, possibly used only in a feature-gated optional build. citeturn18search6turn18search15  
- Full VTK/ITK-style binding stacks (highest complexity): possible (VTK has a bindings project), but you should expect real build and packaging effort. citeturn17search2turn17search9  

## Prior art and reuse: Rust/Tauri imaging apps and WebAssembly-based neuroimaging viewers inside a Tauri WebView

### Evidence of neuroimaging tooling in Rust

While there is not (from the sources reviewed) a widely known “BrainGlobe atlas viewer written in Rust,” there is meaningful Rust neuroimaging-adjacent work:

- `brainview-rs` describes itself as a high-level library plus a simple viewer for **surface-based structural neuroimaging data**, written in Rust. citeturn20search6  
- `neuroformats-rs` provides parsers for structural neuroimaging data formats, focusing on surface-based morphometry as used by tools like FreeSurfer and CAT12. citeturn20search2  

These indicate that (a) neuroimaging data concerns exist in Rust projects, but (b) most Rust-first work emphasizes parsing/visualizing **surfaces** rather than high-performance interactive 3D volume MPR.

### Evidence of medical imaging apps built with Rust + Tauri (+ GPU)

A relevant “closest cousin” to your workload is a DICOM viewer built with Rust + Tauri + GPU rendering:

- A Rust community forum thread describes building a DICOM medical image viewer (“OrthoRay”) using Rust + Tauri + `wgpu`, motivated by performance issues with large CT/MRI datasets; it explicitly discusses real-time slicing through a 3D volume (MPR) and 3D volume rendering. citeturn20search3turn20search7  
- A related Hacker News post also describes the viewer as being built with Tauri and `wgpu` for rendering and highlights “500MB+ MRI series instantly” as a goal. citeturn20search11  

Even if you don’t adopt the same architecture, this is strong evidence that “interactive slicing + Tauri + Rust” is already being done in the wild.

### Web viewers you can embed (and potentially reuse) inside a Tauri WebView

This is where your “reuse architecture/code in a Tauri webview” question becomes especially promising, because Tauri’s WebView architecture can host sophisticated WebGL applications. citeturn26search23turn16search13  

Key web-native viewers:

- Neuroglancer is a WebGL-based volumetric data viewer capable of **arbitrary (non-axis-aligned) cross-sectional views**, plus 3D meshes and skeleton-like models. That directly overlaps with your arbitrary-angle reslicing requirement, except its default assumption is “streamed/chunked sources” rather than “local TIFF stack.” citeturn25search0  
- Neuroglancer’s “Precomputed” ecosystem is supported by tools like CloudVolume, describing itself as a client for random access reading/writing of Neuroglancer volumes in Precomputed format. citeturn25search1turn25search5  
- NiiVue is a WebGL2-based medical image viewer that supports 30+ volume/mesh formats, and its docs emphasize modular embeddability in standard web frameworks (or plain HTML). citeturn25search2turn25search10  
- The ITK/VTK Viewer is an open-source web visualization system for medical/scientific image, mesh, and point-set visualization. citeturn25search3  
- VTK.js is a JavaScript implementation of VTK described as leveraging WebGL (and “WebGPU soon”) and supporting a wide set of visualization algorithms, including volumetric methods. citeturn25search15turn25search23  
- ITK-Wasm is positioned as a major upgrade path from itk.js, focusing on performant computing in WebAssembly and execution beyond JavaScript thanks to WASI. citeturn25search31  

image_group{"layout":"carousel","aspect_ratio":"16:9","query":["Neuroglancer screenshot volumetric viewer","NiiVue WebGL2 medical image viewer screenshot","ITK VTK Viewer screenshot","VTK.js volume rendering example screenshot"],"num_per_query":1}

### What can realistically be “reused” in your Tauri app?

A Tauri WebView can run a complex WebGL application, but your Rust backend still matters for (1) local filesystem access, (2) caching, and (3) computational hot paths if JS is too slow.

Based on the reviewed sources, these reuse patterns are practical:

- **Embed NiiVue directly in the webview** and use Tauri commands to load local files into memory and pass bytes to the viewer. This matches NiiVue’s “embed in any framework” design and Tauri’s “call Rust from the frontend” command model. citeturn25search10turn16search13  
- **Use VTK.js + ITK-Wasm inside the webview** if you want a web-first scientific visualization stack (volume rendering, meshes) and can tolerate a web toolchain. Community threads exist around using VTK.js and ITK-Wasm to build MPR-like viewers and load DICOM/3D TIFF, which is directly adjacent to your problem space. citeturn25search7turn25search11  
- **Adopt a Neuroglancer-like architecture without adopting Neuroglancer itself**: i.e., chunked/bricked volume storage + random access + GPU-accelerated slice rendering. Neuroglancer is designed around streaming and very large datasets, and CloudVolume/Precomputed discussions illustrate the ecosystem around chunking and random access. citeturn25search0turn25search1  

The biggest caveat for “future-proof WebGPU compute for slicing” is that WebGPU availability in Tauri depends on the underlying platform WebView and is not something to assume universally today (as reflected by the open WebGPU-support issue). citeturn19search6  

**Net finding:** you do not have to build everything from scratch to get a functioning app. The Rust ecosystem can cover core I/O (TIFF, NIfTI, DICOM), has viable primitives for interpolation and edge/boundary computation, and has credible precedents for Tauri + GPU medical imaging viewers; meanwhile, the web ecosystem (NiiVue, Neuroglancer, VTK.js/ITK-Wasm) offers mature frontends you can embed into a Tauri WebView and drive with a Rust backend for local BrainGlobe filesystem compatibility. citeturn9search0turn25search0turn25search2turn20search3turn16search13