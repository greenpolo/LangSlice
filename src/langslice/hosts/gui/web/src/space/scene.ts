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
// critically damped springs); a newly placed section appears where it is, whole, and a batch
// placed by one call comes in stack order over `BATCH_S`. Sections are always opaque. The
// sections the agent is looking at rise clearly out of the stack (`LIFT` of the brain's
// height) with a thin frame in the accent, and the others grey a little. The brain keeps
// a thin grey silhouette so it still reads around the sections.
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
const LIFT = 0.48;
const OMEGA = 4.6;
const BATCH_S = 1.6;
const INFLATE_MM = 0.08;
const STAGGER_MAX_S = 0.06;
const VIEW = new THREE.Vector3(-0.85, 0.45, 0.85);

type Spring = { x: THREE.Vector3; v: THREE.Vector3 };
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
  focus: number;
  hover: number;
  url: string;
  wait: number;
  order: number;
};

const VERTEX = /* glsl */ `
  attribute float edge;
  varying vec2 vUv;
  varying float vEdge;
  void main() {
    vUv = uv;
    vEdge = edge;
    gl_Position = projectionMatrix * modelViewMatrix * vec4(position, 1.0);
  }
`;

// The picture as it is: the tissue opaque, the ground around it (the calm pictures' floor,
// `floor` its luminance, read from the picture's corners) left out, judged on the mean of
// a pixel and its four neighbours so the JPEG's noise in the ground does not speckle; a thin frame in the
// accent around the tissue's box for a section in focus (brightened a little), in grey under
// the pointer; `see` greys and darkens a
// section a little while others are in focus (it stays opaque).
const FRAGMENT = /* glsl */ `
  uniform sampler2D map;
  uniform float hasMap;
  uniform float floor;
  uniform vec4 tissueBox;
  uniform vec2 texel;
  uniform float focus;
  uniform float hover;
  uniform float shown;
  uniform float see;
  uniform vec3 accent;
  uniform vec3 edgeColor;
  varying vec2 vUv;
  varying float vEdge;
  void main() {
    if (vEdge > 0.5) discard;
    vec3 c = hasMap > 0.5 ? texture2D(map, vUv).rgb : vec3(0.0);
    float lum = dot(c, vec3(0.299, 0.587, 0.114));
    vec2 t = 1.5 * texel;
    vec3 w = vec3(0.299, 0.587, 0.114);
    float near = (lum + dot(texture2D(map, vUv + vec2(t.x, 0.0)).rgb, w) + dot(texture2D(map, vUv - vec2(t.x, 0.0)).rgb, w)
      + dot(texture2D(map, vUv + vec2(0.0, t.y)).rgb, w) + dot(texture2D(map, vUv - vec2(0.0, t.y)).rgb, w)) / 5.0;
    float tissue = hasMap > 0.5 ? smoothstep(floor + 0.006, floor + 0.02, near) : 0.0;
    vec2 px = max(fwidth(vUv), vec2(1e-5));
    vec2 inner = min(vUv - tissueBox.xy, tissueBox.zw - vUv) / px;
    float inside = step(-1.5, min(inner.x, inner.y));
    float ring = (1.0 - smoothstep(0.6, 1.8, abs(min(inner.x, inner.y)))) * inside;
    float frame = ring * max(focus, 0.6 * hover);
    vec3 grey = vec3(lum) * 0.62;
    vec3 tissueColor = mix(c * (1.0 + 0.18 * focus), grey, 0.7 * see);
    vec3 col = mix(tissueColor, mix(edgeColor, accent, focus), frame);
    float body = tissue;
    float a = max(body, frame * hasMap) * shown;
    if (a < 0.02) discard;
    gl_FragColor = vec4(col, a);
  }
`;

// The brain as brainrender draws its root: matte and see-through, lit by an ambient and a
// key light over the viewer's shoulder, with a thin grey silhouette where the surface turns
// away, so its outline still reads around the sections (drawn `INFLATE_MM` outside the
// surface, so a section reaching the brain's edge does not hide it).
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
    float silhouette = 1.0 - smoothstep(0.1, 0.3, facing);
    vec3 col = mix(tone * light, line, silhouette);
    gl_FragColor = vec4(col, 0.13 + 0.75 * silhouette);
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

// A picture's ground (the darkest of its four corners' 4x4 mean luminances) and the box
// around its tissue (what is brighter than the ground), in uv, a little padded.
function ground(img: HTMLImageElement): { floor: number; box: THREE.Vector4 } {
  const w = img.naturalWidth, h = img.naturalHeight;
  const whole = { floor: 0, box: new THREE.Vector4(0, 0, 1, 1) };
  const canvas = document.createElement('canvas');
  canvas.width = w; canvas.height = h;
  const ctx = canvas.getContext('2d', { willReadFrequently: true });
  if (!ctx || !w || !h) return whole;
  ctx.drawImage(img, 0, 0);
  const d = ctx.getImageData(0, 0, w, h).data;
  const lum = (x: number, y: number) => {
    const i = 4 * (y * w + x);
    return (0.299 * d[i] + 0.587 * d[i + 1] + 0.114 * d[i + 2]) / 255;
  };
  let low = 1;
  for (const [x0, y0] of [[0, 0], [w - 4, 0], [0, h - 4], [w - 4, h - 4]]) {
    let sum = 0;
    for (let y = y0; y < y0 + 4; y++) for (let x = x0; x < x0 + 4; x++) sum += lum(x, y);
    low = Math.min(low, sum / 16);
  }
  let x0 = w, y0 = h, x1 = -1, y1 = -1;
  for (let y = 0; y < h; y += 2) for (let x = 0; x < w; x += 2) {
    if (lum(x, y) > low + 0.015) { x0 = Math.min(x0, x); x1 = Math.max(x1, x); y0 = Math.min(y0, y); y1 = Math.max(y1, y); }
  }
  if (x1 < x0) return { floor: low, box: whole.box };
  const pad = 0.015;
  return {
    floor: low,
    box: new THREE.Vector4(Math.max(0, x0 / w - pad), Math.max(0, y0 / h - pad), Math.min(1, (x1 + 1) / w + pad), Math.min(1, (y1 + 1) / h + pad)),
  };
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
  private brain: THREE.Mesh | null = null;
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
  private accent = token('--lavender', '#a4addf');
  private edgeColor = token('--muted', '#8f8d87');
  private fitted = false;
  private textures = new Map<string, THREE.Texture>();

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
    this.brain = new THREE.Mesh(geometry, new THREE.ShaderMaterial({
      vertexShader: BRAIN_VERTEX, fragmentShader: BRAIN_FRAGMENT, transparent: true, depthWrite: false,
      side: THREE.FrontSide, uniforms: { tone: { value: token('--text', '#c6c4be') }, line: { value: token('--text', '#c6c4be') }, inflate: { value: INFLATE_MM } },
    }));
    this.brain.renderOrder = 2;
    this.scene.add(this.brain);

    const box = geometry.boundingBox!;
    this.bounds.copy(box);
    this.axes[0].at.set(0, box.min.y, box.max.z + 0.9);
    this.axes[1].at.set(0, box.min.y, box.min.z - 0.9);
    this.axes[2].at.set(0, box.max.y + 0.7, 0);
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
      if (!known.has(id)) { this.scene.remove(s.mesh); s.mesh.material.dispose(); this.slabs.delete(id); }
    }
    const changed: Slab[] = [];
    for (const item of list) {
      const s = this.slabs.get(item.id) ?? this.make(item.id);
      s.order = item.order;
      const was = s.placed;
      const before = s.goal.pos.clone();
      if (item.corners_um && this.map) this.place(s, item.corners_um);
      else s.placed = false;
      if (s.placed && (!was || this.reduced)) this.snap(s);
      if (s.placed && (!was || before.distanceToSquared(s.goal.pos) > 1e-8)) changed.push(s);
    }
    changed.sort((a, b) => a.order - b.order);
    const gap = changed.length > 2 && !this.reduced ? Math.min(STAGGER_MAX_S, BATCH_S / changed.length) : 0;
    changed.forEach((s, i) => { s.wait = i * gap; });
  }

  setImage(id: string, url: string, img: HTMLImageElement) {
    const s = this.slabs.get(id);
    if (!s || s.url === url) return;
    s.url = url;
    this.textures.get(id)?.dispose();
    const tex = new THREE.Texture(img);
    tex.flipY = false;
    tex.colorSpace = THREE.NoColorSpace;
    tex.generateMipmaps = false;
    tex.minFilter = THREE.LinearFilter;
    tex.needsUpdate = true;
    this.textures.set(id, tex);
    s.mesh.material.uniforms.map.value = tex;
    s.mesh.material.uniforms.hasMap.value = 1;
    const { floor, box } = ground(img);
    s.mesh.material.uniforms.floor.value = floor;
    s.mesh.material.uniforms.tissueBox.value = box;
    s.mesh.material.uniforms.texel.value.set(1 / Math.max(1, img.naturalWidth), 1 / Math.max(1, img.naturalHeight));
  }

  setFocus(ids: string[]) {
    this.focused = new Set(ids);
  }

  private make(id: string): Slab {
    const material = new THREE.ShaderMaterial({
      vertexShader: VERTEX, fragmentShader: FRAGMENT, transparent: true, depthWrite: true, side: THREE.DoubleSide,
      uniforms: {
        map: { value: null }, hasMap: { value: 0 }, floor: { value: 0 }, tissueBox: { value: new THREE.Vector4(0, 0, 1, 1) }, texel: { value: new THREE.Vector2(1 / 256, 1 / 256) }, focus: { value: 0 }, hover: { value: 0 }, shown: { value: 0 }, see: { value: 0 },
        accent: { value: this.accent }, edgeColor: { value: this.edgeColor },
      },
    });
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
      placed: false, shown: 0, focus: 0, hover: 0, url: '', wait: 0, order: 0,
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
    const r = 0.5 * Math.hypot(extent.x, extent.y + lift, extent.z) * 0.8;
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

  private tick = (now: number) => {
    this.frame = requestAnimationFrame(this.tick);
    const dt = Math.min(1 / 30, Math.max(0, (now - this.last) / 1000));
    this.last = now;
    if (!this.visible || document.hidden) return;
    this.step(dt);
    this.controls.update(dt);
    this.pick();
    this.renderer.render(this.scene, this.camera);
    this.label();
  };

  private step(dt: number) {
    const lift = this.size.y * LIFT;
    const w2 = OMEGA * OMEGA, damp = 2 * OMEGA;
    const ease = this.reduced ? 1 : 1 - Math.exp(-dt * 10);
    const err = new THREE.Quaternion(), turn = new THREE.Quaternion();
    const some = [...this.focused].some(id => this.slabs.get(id)?.placed);
    for (const s of this.slabs.values()) {
      const u = s.mesh.material.uniforms;
      const on = this.focused.has(s.id) && s.placed;
      u.see.value += ((some && !on ? 1 : 0) - u.see.value) * ease;
      s.mesh.renderOrder = on ? 1 : 0;
      s.focus += ((on ? 1 : 0) - s.focus) * ease;
      s.hover += ((this.hovered === s.id ? 1 : 0) - s.hover) * ease;
      const waiting = s.wait > 0;
      if (waiting) s.wait -= dt;
      s.shown = s.placed && !(waiting && s.shown === 0) ? 1 : 0;
      u.focus.value = s.focus;
      u.hover.value = s.hover;
      u.shown.value = s.shown;
      s.mesh.visible = s.shown > 0;
      if (!s.mesh.visible) continue;
      if (waiting) { this.write(s); continue; }
      const goal = s.goal.pos.clone();
      goal.y += lift * s.focus;
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

  private write(s: Slab) {
    s.mesh.matrix.compose(s.pos.x, s.quat, s.size.x).multiply(s.goal.shear);
    s.mesh.matrixWorldNeedsUpdate = true;
  }

  private label() {
    const w = this.host.clientWidth, h = this.host.clientHeight;
    for (const a of this.axes) {
      const p = a.at.clone().project(this.camera);
      const off = p.z > 1 || !this.map;
      a.el.classList.toggle('off', off);
      if (!off) a.el.style.transform = `translate(${((p.x + 1) / 2) * w}px, ${((1 - p.y) / 2) * h}px) translate(-50%, -50%)`;
    }
  }

  private hit(x: number, y: number): string | null {
    const r = this.renderer.domElement.getBoundingClientRect();
    this.ray.setFromCamera(new THREE.Vector2(((x - r.left) / r.width) * 2 - 1, -((y - r.top) / r.height) * 2 + 1), this.camera);
    const meshes = [...this.slabs.values()].filter(s => s.placed).map(s => s.mesh);
    const hits = this.ray.intersectObjects(meshes, false);
    return hits.length ? String(hits[0].object.userData.id) : null;
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
