// The 3D view's scene: the atlas brain as BrainGlobe draws it (its root mesh, matte and
// see-through, as brainrender shows it) and every placed section as a thin slab carrying
// its picture, exactly where its current linear placement puts it.
//
// Scene axes are millimetres centred on the atlas: anterior +z, superior +y, the animal's
// right at -x, read from the atlas's orientation letters (BrainGlobe: each letter names the
// side an axis starts from). A placed section's slab is the parallelogram its thumbnail's
// corners span, as thick as the job's sections; a section without a position is not drawn.
// Nothing moves but what the registration moves: when a section's placement changes, its
// slab glides from the old placement to the new one (position, rotation, size, on
// critically damped springs); a newly placed section appears where it is, whole; a batch
// one call moves goes in stack order over `BATCH_S`, and a batch it places for the first
// time arrives one section after another over `ARRIVE_S`, so the placing is seen. Sections are always opaque, cut
// to their tissue with a smooth edge one pixel wide. The sections the agent is looking at
// lift a little (`LIFT` of the brain's height, an ease-out over `FOCUS_S`), brighten a
// little and carry the accent along their cut edge, and the others grey a little. The brain keeps a thin grey silhouette so it still reads around
// the sections.
import * as THREE from 'three';
import { OrbitControls } from 'three/addons/controls/OrbitControls.js';

export type Corners = [number[], number[], number[]];
export type Placed = { id: string; order: number; corners_um: Corners | null };
export type Shell = {
  orientation: string;
  extent_um: [number, number, number];
  vertices: number;
  triangles: number;
  positions: string;
  indices: string;
};

const FOV = 30;
const LIFT = 0.05;
const FOCUS_S = 0.26;
const OMEGA = 4.6;
const BATCH_S = 2.2;
const ARRIVE_S = 6;
const ARRIVE_MAX_S = 0.16;
const INFLATE_MM = 0.08;
const STAGGER_MAX_S = 0.06;
const TISSUE_STEP = 0.013;
const BRIGHT_GROUND = 0.5;
const SPECK_PX = 24;
const WALL_REACH_MM = 0.25;
const WALL_SPANS = 24;
const WALL_SHADE = 0.2;
const FAR_SHADE = 0.34;
const ROOM_MM = 0.15;
const ROOM_BINS = 96;
const ISLAND_PX = 40;
const GLARE_PX = 30;
const SMOOTH = 0.04;
const RIM_STEP = 0.03;
const PINHOLE = 0.01;
const SOFT_PER_PX = 1 / 128;
const MASK_W = 256;
const DEBRIS = 0.1;
const FILL = 0.66;
const VIEW = new THREE.Vector3(-0.85, 0.45, 0.85);

type Spring = { x: THREE.Vector3; v: THREE.Vector3 };
type Tissue = { mask: Uint8Array; w: number; h: number };
type Slab = {
  id: string;
  mesh: THREE.Mesh<THREE.BufferGeometry, THREE.ShaderMaterial>;
  pos: Spring;
  size: Spring;
  quat: THREE.Quaternion;
  spin: THREE.Vector3;
  goal: { pos: THREE.Vector3; quat: THREE.Quaternion; size: THREE.Vector3; shear: THREE.Matrix4 };
  placed: boolean;
  shown: number;
  focus: { from: number; to: number; at: number };
  rise: number;
  tissue: Tissue | null;
  url: string;
  wait: number;
  order: number;
};

const VERTEX = /* glsl */ `
  attribute float edge;
  varying vec2 vUv;
  varying float vEdge;
  varying float vFar;
  varying vec3 vWorld;
  void main() {
    vUv = uv;
    vEdge = edge;
    vWorld = (modelMatrix * vec4(position, 1.0)).xyz;
    vec4 mv = modelViewMatrix * vec4(position, 1.0);
    vFar = -mv.z;
    gl_Position = projectionMatrix * mv;
  }
`;

// The picture as it is: the tissue opaque, the ground cut away by the mask in the
// texture's alpha (`tissue`, softened over a few pixels so its half-way contour is a smooth
// line), the cut one screen pixel wide through the samples' coverage, never blended with
// what is behind; `lit` brightens a section in focus a little and draws its cut edge in the
// accent (`accent`), `see` greys a section while others are in focus, and a section
// farther from the viewer (`depth`: the near and far ends of the brain) is a little darker,
// so the stack reads in depth. Nothing is drawn outside the atlas's own cross-section at
// the fragment's place along the brain (`room`: the brain's extent to the sides and up and
// down in `ROOM_BINS` steps from front to back, `ROOM_MM` wider): a section not yet aligned
// keeps its true size, and what of it would hang outside the atlas is left out.
const FRAGMENT = /* glsl */ `
  uniform sampler2D map;
  uniform float hasMap;
  uniform float lit;
  uniform float see;
  uniform float shown;
  uniform vec3 accent;
  uniform vec2 depth;
  varying vec2 vUv;
  varying float vEdge;
  uniform sampler2D room;
  uniform vec2 roomSpan;
  uniform float rise;
  varying float vFar;
  varying vec3 vWorld;
  void main() {
    if (vEdge > 0.5 || hasMap < 0.5 || shown < 0.5) discard;
    vec3 at = vec3(vWorld.x, vWorld.y - rise, vWorld.z);
    if (at.z < roomSpan.x || at.z > roomSpan.y) discard;
    vec4 side = texture2D(room, vec2((at.z - roomSpan.x) / (roomSpan.y - roomSpan.x), 0.5));
    if (at.x < side.x || at.x > side.y || at.y < side.z || at.y > side.w) discard;
    vec4 t = texture2D(map, vUv);
    float cover = clamp((t.a - 0.5) / max(fwidth(t.a), 0.0001) + 0.5, 0.0, 1.0);
    if (cover <= 0.0) discard;
    float lum = dot(t.rgb, vec3(0.299, 0.587, 0.114));
    vec3 col = mix(t.rgb * (1.0 + 0.12 * lit), vec3(lum) * 0.62, 0.55 * see);
    float rim = (1.0 - smoothstep(0.6, 0.92, t.a)) * lit;
    float far = clamp((vFar - depth.x) / max(depth.y - depth.x, 0.001), 0.0, 1.0);
    gl_FragColor = vec4(mix(col * (1.0 - ${FAR_SHADE.toFixed(2)} * far), accent, 0.9 * rim), cover);
  }
`;

// The brain as brainrender draws its root: matte and see-through, lit by an ambient and a
// key light over the viewer's shoulder, with a thin grey silhouette where the surface turns
// away, so its outline still reads around the sections (drawn `INFLATE_MM` outside the
// surface, so a section reaching the brain's edge does not hide it). Only its nearest
// surface is drawn (a depth pass goes first), so the folds behind it draw no rings that
// could read as see-through sections.
const BRAIN_VERTEX = /* glsl */ `
  uniform float inflate;
  varying vec3 vNormal;
  varying vec3 vView;
  void main() {
    vec4 mv = modelViewMatrix * vec4(position + normal * inflate, 1.0);
    vNormal = normalize(normalMatrix * normal);
    vView = -mv.xyz;
    gl_Position = projectionMatrix * mv;
  }
`;

const BRAIN_FRAGMENT = /* glsl */ `
  uniform vec3 tone;
  uniform vec3 line;
  varying vec3 vNormal;
  varying vec3 vView;
  void main() {
    vec3 n = normalize(vNormal);
    float facing = abs(dot(n, normalize(vView)));
    float light = 0.55 + 0.45 * max(dot(n, normalize(vec3(-0.4, 0.8, 0.6))), 0.0);
    float silhouette = 1.0 - smoothstep(0.06, 0.2, facing);
    vec3 col = mix(tone * light, line, silhouette);
    gl_FragColor = vec4(col, 0.07 + 0.6 * silhouette);
  }
`;

// The inside of the brain's far wall, where the stack is (`spans`: around each placed
// section's anterior-posterior place, half the way to its neighbours and at most
// `WALL_REACH_MM`, merged): flat, in the sections' own mean tissue colour
// (`tint`) darkened to `WALL_SHADE` of it (it is in the sections' shade, and a flat patch as
// bright as tissue would read as a section), barely shaded by the same key light, drawn
// behind everything.
// Looking between two thin sections near the brain's flank then meets tissue-grey, as the
// brain between them would be, instead of the page's black, so the stack does not read as
// ribs; in front of the stack, behind it and across a wide gap nothing is filled.
const WALL_VERTEX = /* glsl */ `
  varying vec3 vNormal;
  varying float vZ;
  void main() {
    vNormal = normalize(normalMatrix * normal);
    vZ = position.z;
    gl_Position = projectionMatrix * modelViewMatrix * vec4(position, 1.0);
  }
`;

const WALL_FRAGMENT = /* glsl */ `
  uniform vec3 tint;
  uniform vec2 spans[${WALL_SPANS}];
  varying vec3 vNormal;
  varying float vZ;
  void main() {
    bool inside = false;
    for (int i = 0; i < ${WALL_SPANS}; i++) if (vZ >= spans[i].x && vZ <= spans[i].y) inside = true;
    if (!inside) discard;
    float light = 0.85 + 0.15 * abs(dot(normalize(vNormal), normalize(vec3(-0.4, 0.8, 0.6))));
    gl_FragColor = vec4(tint * light * ${WALL_SHADE.toFixed(2)}, 1.0);
  }
`;

// A slab of unit side and unit thickness: its two picture faces (the same pixel at the
// same place on both, so from behind the section reads mirrored, as it would) and its
// four sides, marked by `edge`.
function slabGeometry(): THREE.BufferGeometry {
  const p: number[] = [], uv: number[] = [], edge: number[] = [], idx: number[] = [];
  const quad = (a: number[], b: number[], c: number[], d: number[], uvs: number[][], e: number) => {
    const o = p.length / 3;
    for (const [i, v] of [a, b, c, d].entries()) { p.push(...v); uv.push(...uvs[i]); edge.push(e); }
    idx.push(o, o + 1, o + 2, o, o + 2, o + 3);
  };
  const h = 0.5;
  const faceUv = [[0, 0], [1, 0], [1, 1], [0, 1]];
  quad([-h, -h, h], [h, -h, h], [h, h, h], [-h, h, h], faceUv, 0);
  quad([-h, -h, -h], [-h, h, -h], [h, h, -h], [h, -h, -h], [[0, 0], [0, 1], [1, 1], [1, 0]], 0);
  quad([-h, -h, -h], [h, -h, -h], [h, -h, h], [-h, -h, h], faceUv, 1);
  quad([-h, h, -h], [-h, h, h], [h, h, h], [h, h, -h], faceUv, 1);
  quad([-h, -h, -h], [-h, -h, h], [-h, h, h], [-h, h, -h], faceUv, 1);
  quad([h, -h, -h], [h, h, -h], [h, h, h], [h, -h, h], faceUv, 1);
  const g = new THREE.BufferGeometry();
  g.setAttribute('position', new THREE.Float32BufferAttribute(p, 3));
  g.setAttribute('uv', new THREE.Float32BufferAttribute(uv, 2));
  g.setAttribute('edge', new THREE.Float32BufferAttribute(edge, 1));
  g.setIndex(idx);
  return g;
}

// A colour token as the slab shader's raw sRGB triple (that shader writes sRGB as it comes).
function token(name: string, fallback: string): THREE.Vector3 {
  const text = getComputedStyle(document.documentElement).getPropertyValue(name).trim() || fallback;
  const m = text.match(/^#([0-9a-f]{6})$/i);
  const n = parseInt(m ? m[1] : fallback.slice(1), 16);
  return new THREE.Vector3(((n >> 16) & 255) / 255, ((n >> 8) & 255) / 255, (n & 255) / 255);
}

// A picture with its tissue in the alpha channel. The ground is the darkest of the four
// corners' 4x4 mean luminances (the calm pictures' flat floor; over `BRIGHT_GROUND` it is
// the light look's paper, and the picture is read by its darkness); a pixel is tissue when the
// mean of its 5x5 neighbourhood stands `TISSUE_STEP` above it, so dim tissue stays and the
// ground's JPEG noise does not speckle. What is left of the slide around the tissue goes
// next: a smooth patch (fine texture under `SMOOTH` of its brightness above the ground,
// where tissue's is several times that) that touches the ground outside the section is
// a scanner's glare or a flat grey smear, not tissue, and is cut with a margin; a smooth
// patch inside the tissue (white matter) stays. Specks under `SPECK_PX` pixels go, the
// dark pixels the neighbourhood mean pushed out into the ground are peeled off the outer
// edge, pinholes under `PINHOLE` of the picture are filled, and a piece apart from the
// tissue goes when it is under `DEBRIS` of the largest piece (dirt or glare on the slide)
// or cut by the picture's border (something at the slide's edge, not the section). The ground's colour is
// replaced by its tissue neighbours' so the linear filter never pulls a dark rim into
// the tissue's edge; the pixels go to the texture as they are, never through a canvas,
// which would premultiply the alpha and turn the cut ground black. The mask is made on a
// copy `MASK_W` pixels wide (its sizes are for that width, and a larger picture would cost
// the page a second each), then spread to the picture's size and softened over
// `SOFT_PER_PX` of the width into the alpha, so the shader's half-way contour is smooth.
function tissueImage(img: HTMLImageElement): { pixels: Uint8Array; tissue: Tissue; colour: THREE.Vector3 | null } | null {
  const W = img.naturalWidth, H = img.naturalHeight;
  if (W < 8 || H < 8) return null;
  const full = pixelsOf(img, W, H);
  const shrink = Math.max(1, Math.round(W / MASK_W));
  const w = Math.max(8, Math.round(W / shrink)), h = Math.max(8, Math.round(H / shrink));
  const small = shrink > 1 ? pixelsOf(img, w, h) : full;
  if (!full || !small) return null;
  const mask = tissueMask(small, w, h);
  const d = full;
  const N = W * H;
  const wide = new Uint8Array(N);
  for (let y = 0; y < H; y++) {
    const row = Math.min(h - 1, Math.floor(y / shrink)) * w;
    for (let x = 0; x < W; x++) wide[y * W + x] = mask[row + Math.min(w - 1, Math.floor(x / shrink))];
  }
  const soft = Math.max(shrink, Math.round(W * SOFT_PER_PX));
  let near = wide.slice();
  for (let pass = 0; pass < soft + 2; pass++) {
    const next = near.slice();
    for (let y = 1; y < H - 1; y++) for (let x = 1; x < W - 1; x++) {
      const i = y * W + x;
      if (near[i]) continue;
      let r = 0, g = 0, b = 0, c = 0;
      for (const j of [i - 1, i + 1, i - W, i + W, i - W - 1, i - W + 1, i + W - 1, i + W + 1]) {
        if (!near[j]) continue;
        r += d[4 * j]; g += d[4 * j + 1]; b += d[4 * j + 2]; c++;
      }
      if (!c) continue;
      d[4 * i] = r / c; d[4 * i + 1] = g / c; d[4 * i + 2] = b / c;
      next[i] = 1;
    }
    near = next;
  }
  const solid = new Float32Array(N);
  for (let i = 0; i < N; i++) solid[i] = wide[i];
  const alpha = box(box(solid, W, H, soft), W, H, soft);
  let r = 0, g = 0, b = 0, count = 0;
  for (let i = 0; i < N; i++) {
    if (wide[i]) { r += d[4 * i]; g += d[4 * i + 1]; b += d[4 * i + 2]; count++; }
    d[4 * i + 3] = Math.round(255 * alpha[i]);
  }
  const colour = count ? new THREE.Vector3(r / count / 255, g / count / 255, b / count / 255) : null;
  return { pixels: new Uint8Array(d.buffer.slice(0)), tissue: { mask: wide, w: W, h: H }, colour };
}

// A picture's pixels at a size, read back from a canvas.
function pixelsOf(img: HTMLImageElement, w: number, h: number): Uint8ClampedArray | null {
  const canvas = document.createElement('canvas');
  canvas.width = w; canvas.height = h;
  const ctx = canvas.getContext('2d', { willReadFrequently: true });
  if (!ctx) return null;
  ctx.imageSmoothingQuality = 'high';
  ctx.drawImage(img, 0, 0, w, h);
  return ctx.getImageData(0, 0, w, h).data;
}

// Which pixels of a small picture are tissue (the steps the comment above names).
function tissueMask(d: Uint8ClampedArray, w: number, h: number): Uint8Array {
  const n = w * h;
  const lum = new Float32Array(n);
  for (let i = 0; i < n; i++) lum[i] = (0.299 * d[4 * i] + 0.587 * d[4 * i + 1] + 0.114 * d[4 * i + 2]) / 255;
  let low = 1;
  for (const [x0, y0] of [[0, 0], [w - 4, 0], [0, h - 4], [w - 4, h - 4]]) {
    let sum = 0;
    for (let y = y0; y < y0 + 4; y++) for (let x = x0; x < x0 + 4; x++) sum += lum[y * w + x];
    low = Math.min(low, sum / 16);
  }
  if (low > BRIGHT_GROUND) {
    for (let i = 0; i < n; i++) lum[i] = 1 - lum[i];
    low = 1 - low;
  }
  const mean = box(lum, w, h, 2);
  const mask = new Uint8Array(n);
  for (let i = 0; i < n; i++) mask[i] = mean[i] > low + TISSUE_STEP ? 1 : 0;
  dropSmall(mask, w, h, SPECK_PX);

  const local = box(lum, w, h, 1);
  const fine = new Float32Array(n);
  for (let i = 0; i < n; i++) fine[i] = Math.abs(lum[i] - local[i]);
  const grain = box(fine, w, h, 3);
  const smooth = new Uint8Array(n);
  for (let i = 0; i < n; i++) smooth[i] = mask[i] && grain[i] < SMOOTH * (mean[i] - low + 0.02) ? 1 : 0;
  const near = grow(outside(mask, w, h), w, h, 2);
  const cut = new Uint8Array(n);
  for (const patch of patches(smooth, w, h)) {
    if (patch.length < GLARE_PX || !patch.some(i => near[i])) continue;
    for (const i of patch) cut[i] = 1;
  }
  const margin = grow(cut, w, h, 3);
  for (let i = 0; i < n; i++) if (margin[i]) mask[i] = 0;
  dropSmall(mask, w, h, ISLAND_PX);

  for (let pass = 0; pass < 3; pass++) {
    const ground = outside(mask, w, h);
    const rim: number[] = [];
    for (let y = 0; y < h; y++) for (let x = 0; x < w; x++) {
      const i = y * w + x;
      if (!mask[i] || lum[i] >= low + RIM_STEP) continue;
      if ((x > 0 && ground[i - 1]) || (x < w - 1 && ground[i + 1]) || (y > 0 && ground[i - w]) || (y < h - 1 && ground[i + w])) rim.push(i);
    }
    for (const i of rim) mask[i] = 0;
  }
  const ground = outside(mask, w, h);
  const holes = new Uint8Array(n);
  for (let i = 0; i < n; i++) holes[i] = !mask[i] && !ground[i] ? 1 : 0;
  for (const hole of patches(holes, w, h)) if (hole.length < PINHOLE * n) for (const i of hole) mask[i] = 1;
  const pieces = patches(mask, w, h);
  const largest = pieces.reduce((most, piece) => Math.max(most, piece.length), 0);
  for (const piece of pieces) {
    if (piece.length === largest) continue;
    const cut = piece.some(i => { const x = i % w, y = (i - x) / w; return x === 0 || y === 0 || x === w - 1 || y === h - 1; });
    if (cut || piece.length < DEBRIS * largest) for (const i of piece) mask[i] = 0;
  }
  return mask;
}

// The mean over a (2r+1) square around each pixel, the picture's edge repeated.
function box(src: Float32Array, w: number, h: number, r: number): Float32Array {
  const rows = new Float32Array(src.length), out = new Float32Array(src.length);
  for (let y = 0; y < h; y++) for (let x = 0; x < w; x++) {
    let sum = 0;
    for (let k = -r; k <= r; k++) sum += src[y * w + Math.min(w - 1, Math.max(0, x + k))];
    rows[y * w + x] = sum / (2 * r + 1);
  }
  for (let y = 0; y < h; y++) for (let x = 0; x < w; x++) {
    let sum = 0;
    for (let k = -r; k <= r; k++) sum += rows[Math.min(h - 1, Math.max(0, y + k)) * w + x];
    out[y * w + x] = sum / (2 * r + 1);
  }
  return out;
}

// The 4-connected patches of the set pixels, each as its pixels' indices.
function patches(set: Uint8Array, w: number, h: number): Int32Array[] {
  const seen = new Uint8Array(set.length);
  const queue = new Int32Array(set.length);
  const out: Int32Array[] = [];
  for (let start = 0; start < set.length; start++) {
    if (seen[start] || !set[start]) continue;
    let head = 0, tail = 0;
    queue[tail++] = start; seen[start] = 1;
    while (head < tail) {
      const i = queue[head++];
      const x = i % w, y = (i - x) / w;
      for (const j of [x > 0 ? i - 1 : -1, x < w - 1 ? i + 1 : -1, y > 0 ? i - w : -1, y < h - 1 ? i + w : -1]) {
        if (j < 0 || seen[j] || !set[j]) continue;
        seen[j] = 1; queue[tail++] = j;
      }
    }
    out.push(queue.slice(0, tail));
  }
  return out;
}

// The ground outside the tissue: the patches of non-tissue that reach the picture's border.
function outside(mask: Uint8Array, w: number, h: number): Uint8Array {
  const free = new Uint8Array(mask.length);
  for (let i = 0; i < mask.length; i++) free[i] = mask[i] ? 0 : 1;
  const out = new Uint8Array(mask.length);
  for (const patch of patches(free, w, h)) {
    const border = patch.some(i => { const x = i % w, y = (i - x) / w; return x === 0 || y === 0 || x === w - 1 || y === h - 1; });
    if (border) for (const i of patch) out[i] = 1;
  }
  return out;
}

// The set grown by *steps* pixels, 4-connected.
function grow(set: Uint8Array, w: number, h: number, steps: number): Uint8Array {
  let cur = set;
  for (let s = 0; s < steps; s++) {
    const next = cur.slice();
    for (let y = 0; y < h; y++) for (let x = 0; x < w; x++) {
      const i = y * w + x;
      if (cur[i]) continue;
      if ((x > 0 && cur[i - 1]) || (x < w - 1 && cur[i + 1]) || (y > 0 && cur[i - w]) || (y < h - 1 && cur[i + w])) next[i] = 1;
    }
    cur = next;
  }
  return cur;
}

// The tissue patches smaller than *size* pixels, taken away.
function dropSmall(mask: Uint8Array, w: number, h: number, size: number) {
  for (const patch of patches(mask, w, h)) if (patch.length < size) for (const i of patch) mask[i] = 0;
}

// The brain's extent at each of `ROOM_BINS` steps from *back* to *front* (scene z): the
// least and greatest x and y of the mesh's vertices there, `ROOM_MM` wider, as left, right,
// bottom, top; a step with no vertex takes its neighbour's.
function crossSections(positions: Float32Array, back: number, front: number): Float32Array {
  const out = new Float32Array(4 * ROOM_BINS);
  for (let b = 0; b < ROOM_BINS; b++) out.set([Infinity, -Infinity, Infinity, -Infinity], 4 * b);
  for (let i = 0; i < positions.length; i += 3) {
    const at = Math.floor(((positions[i + 2] - back) / (front - back)) * ROOM_BINS);
    for (const b of [at - 1, at, at + 1]) {
      if (b < 0 || b >= ROOM_BINS) continue;
      out[4 * b] = Math.min(out[4 * b], positions[i] - ROOM_MM);
      out[4 * b + 1] = Math.max(out[4 * b + 1], positions[i] + ROOM_MM);
      out[4 * b + 2] = Math.min(out[4 * b + 2], positions[i + 1] - ROOM_MM);
      out[4 * b + 3] = Math.max(out[4 * b + 3], positions[i + 1] + ROOM_MM);
    }
  }
  for (const order of [[...Array(ROOM_BINS).keys()], [...Array(ROOM_BINS).keys()].reverse()]) {
    let last = -1;
    for (const b of order) {
      if (Number.isFinite(out[4 * b])) last = b;
      else if (last >= 0) out.set(out.subarray(4 * last, 4 * last + 4), 4 * b);
    }
  }
  return out;
}

function roomTexture(sides: Float32Array): THREE.DataTexture {
  const tex = new THREE.DataTexture(sides, sides.length / 4, 1, THREE.RGBAFormat, THREE.FloatType);
  tex.minFilter = THREE.NearestFilter;
  tex.magFilter = THREE.NearestFilter;
  tex.needsUpdate = true;
  return tex;
}

// Faces wound so their normals point away from the mesh's centre, on the whole.
function outward(p: Float32Array, idx: Uint32Array) {
  let cx = 0, cy = 0, cz = 0;
  const n = p.length / 3;
  for (let i = 0; i < p.length; i += 3) { cx += p[i]; cy += p[i + 1]; cz += p[i + 2]; }
  cx /= n; cy /= n; cz /= n;
  let sum = 0;
  for (let t = 0; t < idx.length; t += 3) {
    const a = 3 * idx[t], b = 3 * idx[t + 1], c = 3 * idx[t + 2];
    const ux = p[b] - p[a], uy = p[b + 1] - p[a + 1], uz = p[b + 2] - p[a + 2];
    const vx = p[c] - p[a], vy = p[c + 1] - p[a + 1], vz = p[c + 2] - p[a + 2];
    sum += (uy * vz - uz * vy) * (p[a] - cx) + (uz * vx - ux * vz) * (p[a + 1] - cy) + (ux * vy - uy * vx) * (p[a + 2] - cz);
  }
  if (sum >= 0) return;
  for (let t = 0; t < idx.length; t += 3) { const s = idx[t + 1]; idx[t + 1] = idx[t + 2]; idx[t + 2] = s; }
}

function bytes(b64: string): ArrayBuffer {
  const raw = atob(b64);
  const out = new Uint8Array(raw.length);
  for (let i = 0; i < raw.length; i++) out[i] = raw.charCodeAt(i);
  return out.buffer;
}

export class Space {
  private renderer: THREE.WebGLRenderer;
  private scene = new THREE.Scene();
  private camera = new THREE.PerspectiveCamera(FOV, 1, 0.1, 400);
  private controls: OrbitControls;
  private geometry = slabGeometry();
  private slabs = new Map<string, Slab>();
  private brain: THREE.Group | null = null;
  private axes: { el: HTMLElement; at: THREE.Vector3 }[] = [];
  private map: ((um: number[]) => THREE.Vector3) | null = null;
  private size = new THREE.Vector3(11, 8, 13);
  private bounds = new THREE.Box3(new THREE.Vector3(-5.5, -4, -6.5), new THREE.Vector3(5.5, 4, 6.5));
  private thickness = 0.01;
  private focused = new Set<string>();
  private hovered: string | null = null;
  private frame = 0;
  private last = 0;
  private visible = true;
  private observer: ResizeObserver;
  private seen: IntersectionObserver;
  private down: { x: number; y: number } | null = null;
  private pointer: { x: number; y: number } | null = null;
  private ray = new THREE.Raycaster();
  private touched = false;
  private lifting = new THREE.Vector3();
  private wall: THREE.Mesh | null = null;
  private tint = new THREE.Vector3();
  private tints = new Map<string, THREE.Vector3>();
  private spans = Array.from({ length: WALL_SPANS }, () => new THREE.Vector2(1, 0));
  private fitted = false;
  private textures = new Map<string, THREE.Texture>();
  private accent = token('--lavender', '#a4addf');
  private depth = new THREE.Vector2(0, 1);
  private room = roomTexture(new Float32Array([-1e6, 1e6, -1e6, 1e6]));
  private roomSpan = new THREE.Vector2(-1e6, 1e6);

  constructor(
    private host: HTMLElement,
    private reduced: boolean,
    private on: { pick: (id: string) => void; hover: (id: string | null, x: number, y: number) => void; touched: () => void },
  ) {
    this.renderer = new THREE.WebGLRenderer({ antialias: true, alpha: true, powerPreference: 'high-performance' });
    this.renderer.setPixelRatio(Math.min(2, window.devicePixelRatio || 1));
    this.renderer.setClearColor(0x000000, 0);
    this.renderer.domElement.className = 'volume-canvas';
    host.appendChild(this.renderer.domElement);
    this.camera.position.set(-14, 9, 17);
    this.controls = new OrbitControls(this.camera, this.renderer.domElement);
    this.controls.enableDamping = !reduced;
    this.controls.dampingFactor = 0.1;
    this.controls.rotateSpeed = 0.7;
    this.controls.zoomSpeed = 0.8;
    this.controls.enablePan = false;
    this.controls.addEventListener('start', () => {
      if (!this.touched) { this.touched = true; this.on.touched(); }
    });
    const el = this.renderer.domElement;
    el.addEventListener('pointerdown', this.onDown);
    el.addEventListener('pointerup', this.onUp);
    el.addEventListener('pointermove', this.onMove);
    el.addEventListener('pointerleave', this.onLeave);
    this.observer = new ResizeObserver(() => this.resize());
    this.observer.observe(host);
    this.seen = new IntersectionObserver(entries => { this.visible = entries.some(e => e.isIntersecting); });
    this.seen.observe(host);
    for (const word of ['anterior', 'posterior', 'dorsal']) {
      const label = document.createElement('span');
      label.className = 'volume-axis off';
      label.textContent = word;
      host.appendChild(label);
      this.axes.push({ el: label, at: new THREE.Vector3() });
    }
    this.resize();
    this.last = performance.now();
    this.frame = requestAnimationFrame(this.tick);
  }

  dispose() {
    cancelAnimationFrame(this.frame);
    this.observer.disconnect();
    this.seen.disconnect();
    this.controls.dispose();
    for (const s of this.slabs.values()) s.mesh.material.dispose();
    for (const t of this.textures.values()) t.dispose();
    this.geometry.dispose();
    if (this.brain) this.drop(this.brain);
    this.renderer.dispose();
    this.renderer.domElement.remove();
    for (const a of this.axes) a.el.remove();
  }

  // The atlas: its axes and size, and BrainGlobe's root mesh, matte and see-through.
  setShell(shell: Shell) {
    const extent = shell.extent_um;
    const rule = shell.orientation.toLowerCase().split('').map(l => (
      l === 'a' ? { axis: 2, sign: -1 } : l === 'p' ? { axis: 2, sign: 1 }
        : l === 's' ? { axis: 1, sign: -1 } : l === 'i' ? { axis: 1, sign: 1 }
          : l === 'r' ? { axis: 0, sign: 1 } : { axis: 0, sign: -1 }
    ));
    this.map = (um: number[]) => {
      const out = [0, 0, 0];
      for (let k = 0; k < 3; k++) out[rule[k].axis] = rule[k].sign * (um[k] - extent[k] / 2) / 1000;
      return new THREE.Vector3(out[0], out[1], out[2]);
    };
    const size = [0, 0, 0];
    for (let k = 0; k < 3; k++) size[rule[k].axis] = extent[k] / 1000;
    this.size.set(size[0], size[1], size[2]);
    const raw = new Float32Array(bytes(shell.positions));
    const positions = new Float32Array(raw.length);
    for (let i = 0; i < raw.length; i += 3) {
      const v = this.map([raw[i], raw[i + 1], raw[i + 2]]);
      positions[i] = v.x; positions[i + 1] = v.y; positions[i + 2] = v.z;
    }
    const index = new Uint32Array(bytes(shell.indices));
    outward(positions, index);
    const geometry = new THREE.BufferGeometry();
    geometry.setAttribute('position', new THREE.BufferAttribute(positions, 3));
    geometry.setIndex(new THREE.BufferAttribute(index, 1));
    geometry.computeVertexNormals();
    geometry.computeBoundingBox();
    if (this.brain) { this.scene.remove(this.brain); this.drop(this.brain); }
    const inflate = { value: INFLATE_MM };
    const depth = new THREE.Mesh(geometry, new THREE.ShaderMaterial({
      vertexShader: BRAIN_VERTEX, fragmentShader: 'void main() { gl_FragColor = vec4(0.0); }', colorWrite: false,
      side: THREE.FrontSide, uniforms: { inflate },
    }));
    depth.renderOrder = 2;
    const skin = new THREE.Mesh(geometry, new THREE.ShaderMaterial({
      vertexShader: BRAIN_VERTEX, fragmentShader: BRAIN_FRAGMENT, transparent: true, depthWrite: false, depthFunc: THREE.LessEqualDepth,
      side: THREE.FrontSide, uniforms: { tone: { value: token('--text', '#c6c4be') }, line: { value: token('--text', '#c6c4be') }, inflate },
    }));
    skin.renderOrder = 3;
    this.brain = new THREE.Group();
    this.wall = new THREE.Mesh(geometry, new THREE.ShaderMaterial({
      vertexShader: WALL_VERTEX, fragmentShader: WALL_FRAGMENT, side: THREE.BackSide, depthWrite: false,
      uniforms: { tint: { value: this.tint }, spans: { value: this.spans } },
    }));
    this.wall.renderOrder = -1;
    this.wall.visible = this.tints.size > 0;
    this.brain.add(this.wall, depth, skin);
    this.scene.add(this.brain);

    const box = geometry.boundingBox!;
    this.bounds.copy(box);
    this.room.dispose();
    this.room = roomTexture(crossSections(positions, box.min.z - ROOM_MM, box.max.z + ROOM_MM));
    this.roomSpan.set(box.min.z - ROOM_MM, box.max.z + ROOM_MM);
    for (const slab of this.slabs.values()) slab.mesh.material.uniforms.room.value = this.room;
    this.axes[0].at.set(0, box.min.y, box.max.z + 0.9);
    this.axes[1].at.set(0, box.min.y, box.min.z - 0.9);
    this.axes[2].at.set(0, box.max.y + 1.1, 0);
    this.fit();
  }

  private drop(root: THREE.Object3D) {
    root.traverse(o => { if (o instanceof THREE.Mesh) { o.geometry.dispose(); (o.material as THREE.Material).dispose(); } });
  }

  // The sections as the job places them now; a changed placement is glided to.
  setSections(list: Placed[], thicknessUm: number) {
    this.thickness = Math.max(thicknessUm, 1) / 1000;
    const known = new Set(list.map(s => s.id));
    for (const [id, s] of this.slabs) {
      if (!known.has(id)) { this.scene.remove(s.mesh); s.mesh.material.dispose(); this.slabs.delete(id); this.tints.delete(id); }
    }
    const changed: Slab[] = [];
    let arrived = 0;
    for (const item of list) {
      const s = this.slabs.get(item.id) ?? this.make(item.id);
      s.order = item.order;
      const was = s.placed;
      const before = s.goal.pos.clone();
      if (item.corners_um && this.map) this.place(s, item.corners_um);
      else s.placed = false;
      if (s.placed && (!was || this.reduced)) this.snap(s);
      if (s.placed && (!was || before.distanceToSquared(s.goal.pos) > 1e-8)) changed.push(s);
      if (s.placed && !was) arrived++;
    }
    this.spread();
    this.tinted();
    changed.sort((a, b) => a.order - b.order);
    const arriving = 2 * arrived > changed.length;
    const gap = changed.length <= 2 || this.reduced ? 0
      : arriving ? Math.min(ARRIVE_MAX_S, ARRIVE_S / changed.length) : Math.min(STAGGER_MAX_S, BATCH_S / changed.length);
    changed.forEach((s, i) => { s.wait = i * gap; });
  }

  setImage(id: string, url: string, img: HTMLImageElement) {
    const s = this.slabs.get(id);
    if (!s || s.url === url) return;
    s.url = url;
    this.textures.get(id)?.dispose();
    const cut = tissueImage(img);
    if (!cut) return;
    const tex = new THREE.DataTexture(cut.pixels, cut.tissue.w, cut.tissue.h, THREE.RGBAFormat);
    s.tissue = cut.tissue;
    if (cut.colour) this.tints.set(id, cut.colour);
    this.spread();
    this.tinted();
    tex.flipY = false;
    tex.colorSpace = THREE.NoColorSpace;
    tex.generateMipmaps = false;
    tex.minFilter = THREE.LinearFilter;
    tex.magFilter = THREE.LinearFilter;
    tex.needsUpdate = true;
    this.textures.set(id, tex);
    s.mesh.material.uniforms.map.value = tex;
    s.mesh.material.uniforms.hasMap.value = 1;
  }

  private spread() {
    const zs = [...this.slabs.values()].filter(s => s.placed && s.url).map(s => s.goal.pos.z).sort((a, b) => a - b);
    const runs: [number, number][] = [];
    zs.forEach((z, i) => {
      const before = i > 0 ? Math.min(WALL_REACH_MM, (z - zs[i - 1]) / 2) : 0;
      const after = i < zs.length - 1 ? Math.min(WALL_REACH_MM, (zs[i + 1] - z) / 2) : 0;
      const last = runs[runs.length - 1];
      if (last && z - before <= last[1] + 1e-6) last[1] = z + after;
      else runs.push([z - before, z + after]);
    });
    const kept = runs.filter(r => r[1] > r[0]).sort((a, b) => (b[1] - b[0]) - (a[1] - a[0])).slice(0, WALL_SPANS);
    this.spans.forEach((v, i) => { if (kept[i]) v.set(kept[i][0], kept[i][1]); else v.set(1, 0); });
  }

  private tinted() {
    this.tint.set(0, 0, 0);
    let count = 0;
    for (const [id, c] of this.tints) if (this.slabs.get(id)?.placed) { this.tint.add(c); count++; }
    if (count) this.tint.multiplyScalar(1 / count);
    if (this.wall) this.wall.visible = count > 0 && this.spans.some(v => v.y > v.x);
  }

  setFocus(ids: string[]) {
    this.focused = new Set(ids);
  }

  private make(id: string): Slab {
    const material = new THREE.ShaderMaterial({
      vertexShader: VERTEX, fragmentShader: FRAGMENT, transparent: false, depthWrite: true, side: THREE.DoubleSide,
      uniforms: { map: { value: null }, hasMap: { value: 0 }, lit: { value: 0 }, see: { value: 0 }, shown: { value: 0 }, accent: { value: this.accent }, depth: { value: this.depth },
        room: { value: this.room }, roomSpan: { value: this.roomSpan }, rise: { value: 0 } },
    });
    material.alphaToCoverage = true;
    const mesh = new THREE.Mesh(this.geometry, material);
    mesh.matrixAutoUpdate = false;
    mesh.visible = false;
    mesh.userData.id = id;
    this.scene.add(mesh);
    const s: Slab = {
      id, mesh,
      pos: { x: new THREE.Vector3(), v: new THREE.Vector3() },
      size: { x: new THREE.Vector3(1, 1, 1), v: new THREE.Vector3() },
      quat: new THREE.Quaternion(), spin: new THREE.Vector3(),
      goal: { pos: new THREE.Vector3(), quat: new THREE.Quaternion(), size: new THREE.Vector3(1, 1, 1), shear: new THREE.Matrix4() },
      placed: false, shown: 0, focus: { from: 0, to: 0, at: 0 }, rise: 0, tissue: null, url: '', wait: 0, order: 0,
    };
    this.slabs.set(id, s);
    return s;
  }

  private place(s: Slab, corners: Corners) {
    const [tl, tr, bl] = corners.map(c => this.map!(c));
    const u = tr.clone().sub(tl), v = bl.clone().sub(tl);
    const n = u.clone().cross(v).normalize().multiplyScalar(this.thickness);
    const centre = tl.clone().addScaledVector(u, 0.5).addScaledVector(v, 0.5);
    const m = new THREE.Matrix4().makeBasis(u, v, n).setPosition(centre);
    const pos = new THREE.Vector3(), quat = new THREE.Quaternion(), size = new THREE.Vector3();
    m.decompose(pos, quat, size);
    const shear = new THREE.Matrix4().compose(pos, quat, size).invert().multiply(m);
    s.goal = { pos, quat, size, shear };
    s.placed = true;
  }

  private snap(s: Slab) {
    s.pos.x.copy(s.goal.pos); s.pos.v.set(0, 0, 0);
    s.size.x.copy(s.goal.size); s.size.v.set(0, 0, 0);
    s.quat.copy(s.goal.quat); s.spin.set(0, 0, 0);
    this.write(s);
  }

  private fit() {
    const centre = this.bounds.getCenter(new THREE.Vector3());
    const extent = this.bounds.getSize(new THREE.Vector3());
    const lift = extent.y * LIFT;
    centre.y += lift / 2;
    const r = 0.5 * Math.hypot(extent.x, extent.y + lift, extent.z) * FILL;
    const half = THREE.MathUtils.degToRad(FOV / 2);
    const fit = Math.min(half, Math.atan(Math.tan(half) * this.camera.aspect));
    const dist = r / Math.sin(fit);
    if (!this.fitted || !this.touched) {
      this.camera.position.copy(VIEW.clone().normalize().multiplyScalar(dist).add(centre));
    }
    this.controls.target.copy(centre);
    this.controls.minDistance = dist * 0.35;
    this.controls.maxDistance = dist * 2.5;
    this.camera.near = dist / 50; this.camera.far = dist * 6;
    this.camera.updateProjectionMatrix();
    this.controls.update();
    this.fitted = true;
  }

  private resize() {
    const w = Math.max(1, this.host.clientWidth), h = Math.max(1, this.host.clientHeight);
    this.renderer.setSize(w, h, false);
    this.camera.aspect = w / h;
    this.camera.updateProjectionMatrix();
    if (this.map) this.fit();
  }

  private tick = () => {
    this.frame = requestAnimationFrame(this.tick);
    const now = performance.now();
    const dt = Math.min(1 / 30, Math.max(0, (now - this.last) / 1000));
    this.last = now;
    if (!this.visible || document.hidden) return;
    this.step(dt);
    this.controls.update(dt);
    const away = this.camera.position.distanceTo(this.controls.target);
    const reach = 0.5 * this.size.z;
    this.depth.set(away - reach, away + reach);
    this.pick();
    this.renderer.render(this.scene, this.camera);
    this.label();
  };

  private step(dt: number) {
    const rise = this.size.y * LIFT;
    const w2 = OMEGA * OMEGA, damp = 2 * OMEGA;
    const ease = this.reduced ? 1 : 1 - Math.exp(-dt / (FOCUS_S / 3));
    const now = performance.now() / 1000;
    const err = new THREE.Quaternion(), turn = new THREE.Quaternion();
    const some = [...this.focused].some(id => this.slabs.get(id)?.placed);
    for (const s of this.slabs.values()) {
      const u = s.mesh.material.uniforms;
      const on = this.focused.has(s.id) && s.placed;
      u.see.value += ((some && !on ? 1 : 0) - u.see.value) * ease;
      const lift = this.lifted(s, on ? 1 : 0, now);
      s.rise = rise * lift;
      u.rise.value = s.rise;
      u.lit.value = lift;
      const waiting = s.wait > 0;
      if (waiting) s.wait -= dt;
      s.shown = s.placed && !(waiting && s.shown === 0) ? 1 : 0;
      u.shown.value = s.shown;
      s.mesh.visible = s.shown > 0;
      if (!s.mesh.visible) continue;
      if (waiting) { this.write(s); continue; }
      const goal = s.goal.pos.clone();
      if (this.reduced) {
        s.pos.x.copy(goal); s.size.x.copy(s.goal.size); s.quat.copy(s.goal.quat);
      } else {
        const a = goal.sub(s.pos.x).multiplyScalar(w2).addScaledVector(s.pos.v, -damp);
        s.pos.v.addScaledVector(a, dt); s.pos.x.addScaledVector(s.pos.v, dt);
        const b = s.goal.size.clone().sub(s.size.x).multiplyScalar(w2).addScaledVector(s.size.v, -damp);
        s.size.v.addScaledVector(b, dt); s.size.x.addScaledVector(s.size.v, dt);
        err.copy(s.goal.quat).multiply(turn.copy(s.quat).invert());
        if (err.w < 0) { err.x = -err.x; err.y = -err.y; err.z = -err.z; err.w = -err.w; }
        const angle = 2 * Math.acos(Math.min(1, err.w));
        const sin = Math.sqrt(Math.max(0, 1 - err.w * err.w));
        const axis = sin > 1e-6 ? new THREE.Vector3(err.x / sin, err.y / sin, err.z / sin) : new THREE.Vector3();
        s.spin.addScaledVector(axis.multiplyScalar(angle * w2).addScaledVector(s.spin, -damp), dt);
        const step = s.spin.length() * dt;
        if (step > 1e-7) s.quat.premultiply(turn.setFromAxisAngle(s.spin.clone().normalize(), step)).normalize();
      }
      this.write(s);
    }
  }

  // How far a section's focus has come, 0 to 1: an ease-out over `FOCUS_S` from where it
  // was when its focus last changed; at once with reduced motion.
  private lifted(s: Slab, to: number, now: number): number {
    const f = s.focus;
    const t = this.reduced ? 1 : Math.min(1, (now - f.at) / FOCUS_S);
    const at = f.from + (f.to - f.from) * (1 - (1 - t) ** 3);
    if (to !== f.to) s.focus = { from: at, to, at: now };
    return at;
  }

  private write(s: Slab) {
    this.lifting.copy(s.pos.x).setY(s.pos.x.y + s.rise);
    s.mesh.matrix.compose(this.lifting, s.quat, s.size.x).multiply(s.goal.shear);
    s.mesh.matrixWorldNeedsUpdate = true;
  }

  private label() {
    const w = this.host.clientWidth, h = this.host.clientHeight;
    const centre = this.bounds.getCenter(new THREE.Vector3());
    const middle = centre.clone().project(this.camera);
    const rim = centre.clone().add(new THREE.Vector3(0, 0.5 * this.size.y, 0)).project(this.camera);
    const radius = Math.abs(rim.y - middle.y) * 1.25;
    const away = this.camera.position.distanceTo(centre);
    for (const a of this.axes) {
      const p = a.at.clone().project(this.camera);
      const behind = this.camera.position.distanceTo(a.at) > away && Math.hypot((p.x - middle.x) * this.camera.aspect, p.y - middle.y) < radius;
      const off = p.z > 1 || !this.map || behind;
      a.el.classList.toggle('off', off);
      if (!off) a.el.style.transform = `translate(${((p.x + 1) / 2) * w}px, ${((1 - p.y) / 2) * h}px) translate(-50%, -50%)`;
    }
  }

  private hit(x: number, y: number): string | null {
    const r = this.renderer.domElement.getBoundingClientRect();
    this.ray.setFromCamera(new THREE.Vector2(((x - r.left) / r.width) * 2 - 1, -((y - r.top) / r.height) * 2 + 1), this.camera);
    const meshes = [...this.slabs.values()].filter(s => s.placed).map(s => s.mesh);
    for (const hit of this.ray.intersectObjects(meshes, false)) {
      const t = this.slabs.get(String(hit.object.userData.id))?.tissue;
      if (!t || !hit.uv) continue;
      const x = Math.min(t.w - 1, Math.max(0, Math.floor(hit.uv.x * t.w)));
      const y = Math.min(t.h - 1, Math.max(0, Math.floor(hit.uv.y * t.h)));
      if (t.mask[y * t.w + x]) return String(hit.object.userData.id);
    }
    return null;
  }

  private pick() {
    if (!this.pointer) return;
    const { x, y } = this.pointer;
    this.pointer = null;
    const id = this.down ? null : this.hit(x, y);
    if (id !== this.hovered) {
      this.hovered = id;
      this.renderer.domElement.classList.toggle('pointing', id !== null);
    }
    const r = this.renderer.domElement.getBoundingClientRect();
    this.on.hover(id, x - r.left, y - r.top);
  }

  private onDown = (e: PointerEvent) => { this.down = { x: e.clientX, y: e.clientY }; };
  private onUp = (e: PointerEvent) => {
    const d = this.down;
    this.down = null;
    if (!d || Math.hypot(e.clientX - d.x, e.clientY - d.y) > 5) return;
    const id = this.hit(e.clientX, e.clientY);
    if (id) this.on.pick(id);
  };
  private onMove = (e: PointerEvent) => { this.pointer = { x: e.clientX, y: e.clientY }; };
  private onLeave = () => {
    this.pointer = null;
    this.hovered = null;
    this.renderer.domElement.classList.remove('pointing');
    this.on.hover(null, 0, 0);
  };
}
