// The 3D view's scene: the atlas brain as glass and every section as a thin slab carrying
// its thumbnail, where its linear placement puts it.
//
// Scene axes are millimetres centred on the atlas: anterior +z, superior +y, the animal's
// right at -x, read from the atlas's orientation letters (BrainGlobe: each letter names the
// side an axis starts from). A placed section's slab is the parallelogram its thumbnail's
// corners span, as thick as the job's sections; an unplaced one waits in a loose cloud
// around the brain, turned any way, until the agent places it. Every slab moves on springs
// (position, rotation, size), so a placement reads as the section flying in and settling;
// the sections the agent looks at rise out of the stack and take the accent colour.
import * as THREE from 'three';
import { OrbitControls } from 'three/addons/controls/OrbitControls.js';
import type { Shell, Surface } from './surface';

export type Corners = [number[], number[], number[]];
export type Placed = { id: string; order: number; corners_um: Corners | null };

const FOV = 30;
const LIFT = 0.2;
const OMEGA = 5.2;
const ZETA = 0.8;
const STAGGER_S = 0.055;
const STAGGER_MAX_S = 1.4;
const FLASH_S = 0.9;
const ASPECT = 1.35;
const AUTO_ROTATE = 0.45;
const PICTURE_WAIT_MS = 6000;

type Spring = { x: THREE.Vector3; v: THREE.Vector3 };
type Slab = {
  id: string;
  order: number;
  mesh: THREE.Mesh<THREE.BufferGeometry, THREE.ShaderMaterial>;
  pos: Spring;
  size: Spring;
  quat: THREE.Quaternion;
  spin: THREE.Vector3;
  goal: { pos: THREE.Vector3; quat: THREE.Quaternion; size: THREE.Vector3; shear: THREE.Matrix4 };
  placed: boolean;
  wait: number;
  born: number;
  arriving: boolean;
  flash: number;
  focus: number;
  hover: number;
  aspect: number;
  url: string;
  phase: number;
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

const FRAGMENT = /* glsl */ `
  uniform sampler2D map;
  uniform float hasMap;
  uniform float focus;
  uniform float dim;
  uniform float hover;
  uniform float flash;
  uniform float placed;
  uniform vec3 accent;
  uniform vec3 paper;
  varying vec2 vUv;
  varying float vEdge;
  void main() {
    float lift = max(focus, flash);
    if (vEdge > 0.5) {
      vec3 side = mix(paper * 0.55, accent, lift);
      gl_FragColor = vec4(side, (0.16 + 0.75 * lift) * (1.0 - 0.6 * dim));
      return;
    }
    vec3 c = hasMap > 0.5 ? texture2D(map, vUv).rgb : vec3(0.0);
    float lum = dot(c, vec3(0.299, 0.587, 0.114));
    float tissue = smoothstep(0.04, 0.3, lum);
    vec3 col = c * (1.3 + 0.35 * hover + 0.15 * lift);
    col = mix(col, col * accent * 1.1 + accent * 0.04, 0.5 * lift);
    float alpha = tissue * mix(0.62, 0.95, lift) * (1.0 - 0.6 * dim) * mix(0.85, 1.0, placed);
    vec2 d = min(vUv, 1.0 - vUv) / max(fwidth(vUv), vec2(1e-5));
    float line = 1.0 - smoothstep(0.6, 1.6, min(d.x, d.y));
    float frame = line * (0.015 + 0.3 * hover + 0.85 * lift) * (1.0 - 0.5 * dim);
    float card = 0.06 * lift;
    vec3 frameCol = mix(paper * 0.7, accent, lift);
    float a = max(alpha, max(frame, card));
    vec3 outCol = (col * alpha + frameCol * frame + paper * 0.25 * card * (1.0 - alpha)) / max(alpha + frame + card * (1.0 - alpha), 1e-4);
    if (a < 0.012) discard;
    gl_FragColor = vec4(outCol, min(a, 1.0));
  }
`;

const SHELL_VERTEX = /* glsl */ `
  varying vec3 vNormal;
  varying vec3 vView;
  void main() {
    vec4 mv = modelViewMatrix * vec4(position, 1.0);
    vNormal = normalize(normalMatrix * normal);
    vView = -mv.xyz;
    gl_Position = projectionMatrix * mv;
  }
`;

const SHELL_FRAGMENT = /* glsl */ `
  uniform vec3 rim;
  uniform vec3 body;
  uniform float strength;
  uniform float base;
  varying vec3 vNormal;
  varying vec3 vView;
  void main() {
    float f = 1.0 - abs(dot(normalize(vNormal), normalize(vView)));
    float fres = pow(f, 2.6);
    vec3 col = mix(body, rim, smoothstep(0.15, 0.9, f));
    gl_FragColor = vec4(col, base + strength * fres);
  }
`;

const FLOOR_FRAGMENT = /* glsl */ `
  uniform vec3 tint;
  varying vec2 vUv;
  void main() {
    float r = length(vUv - 0.5) * 2.0;
    float a = (1.0 - smoothstep(0.0, 1.0, r)) * 0.07;
    float ring = (1.0 - smoothstep(0.0, 0.006, abs(r - 0.62))) * 0.05;
    gl_FragColor = vec4(tint, a + ring);
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
  const sideUv = [[0, 0], [1, 0], [1, 1], [0, 1]];
  quad([-h, -h, -h], [h, -h, -h], [h, -h, h], [-h, -h, h], sideUv, 1);
  quad([-h, h, -h], [-h, h, h], [h, h, h], [h, h, -h], sideUv, 1);
  quad([-h, -h, -h], [-h, -h, h], [-h, h, h], [-h, h, -h], sideUv, 1);
  quad([h, -h, -h], [h, h, -h], [h, h, h], [h, -h, h], sideUv, 1);
  const g = new THREE.BufferGeometry();
  g.setAttribute('position', new THREE.Float32BufferAttribute(p, 3));
  g.setAttribute('uv', new THREE.Float32BufferAttribute(uv, 2));
  g.setAttribute('edge', new THREE.Float32BufferAttribute(edge, 1));
  g.setIndex(idx);
  return g;
}

// A colour token as the shader's raw sRGB triple (the shaders write sRGB as it comes).
function token(name: string, fallback: string): THREE.Vector3 {
  const text = getComputedStyle(document.documentElement).getPropertyValue(name).trim() || fallback;
  const m = text.match(/^#([0-9a-f]{6})$/i);
  const n = parseInt(m ? m[1] : fallback.slice(1), 16);
  return new THREE.Vector3(((n >> 16) & 255) / 255, ((n >> 8) & 255) / 255, (n & 255) / 255);
}

function hash(text: string, salt: number): number {
  let h = 2166136261 ^ salt;
  for (let i = 0; i < text.length; i++) { h ^= text.charCodeAt(i); h = Math.imul(h, 16777619); }
  h ^= h >>> 13; h = Math.imul(h, 0x5bd1e995); h ^= h >>> 15;
  return (h >>> 0) / 4294967296;
}

export class Space {
  private renderer: THREE.WebGLRenderer;
  private scene = new THREE.Scene();
  private camera = new THREE.PerspectiveCamera(FOV, 1, 0.1, 400);
  private controls: OrbitControls;
  private geometry = slabGeometry();
  private slabs = new Map<string, Slab>();
  private shell: THREE.Group | null = null;
  private floor: THREE.Mesh | null = null;
  private axes: { el: HTMLElement; at: THREE.Vector3 }[] = [];
  private map: ((um: number[]) => THREE.Vector3) | null = null;
  private size = new THREE.Vector3(11, 8, 13);
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
  private paper = token('--name', '#f2f0eb');
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
    this.controls.enableDamping = true;
    this.controls.dampingFactor = 0.07;
    this.controls.rotateSpeed = 0.7;
    this.controls.zoomSpeed = 0.8;
    this.controls.enablePan = false;
    this.controls.autoRotate = !reduced;
    this.controls.autoRotateSpeed = AUTO_ROTATE;
    this.controls.addEventListener('start', () => {
      this.controls.autoRotate = false;
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
    this.shell?.traverse(o => { if (o instanceof THREE.Mesh) { o.geometry.dispose(); (o.material as THREE.Material).dispose(); } });
    this.renderer.dispose();
    this.renderer.domElement.remove();
    for (const a of this.axes) a.el.remove();
  }

  // The atlas: its axes, its size and its surface, drawn as glass.
  setShell(shell: Shell, surface: Surface) {
    const [e0, e1, e2] = shell.extent_um;
    const extent = [e0, e1, e2];
    const letters = shell.orientation.toLowerCase().split('');
    const rule = letters.map(l => (
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
    const p = surface.positions;
    const scene = new Float32Array(p.length);
    for (let i = 0; i < p.length; i += 3) {
      const v = this.map([p[i], p[i + 1], p[i + 2]]);
      scene[i] = v.x; scene[i + 1] = v.y; scene[i + 2] = v.z;
    }
    const geometry = new THREE.BufferGeometry();
    geometry.setAttribute('position', new THREE.BufferAttribute(scene, 3));
    geometry.setIndex(new THREE.BufferAttribute(surface.indices, 1));
    geometry.computeVertexNormals();
    geometry.computeBoundingBox();
    const rim = this.accent.clone().lerp(this.paper, 0.25);
    const body = token('--muted', '#8f8d87').multiplyScalar(0.35);
    const glass = (side: THREE.Side, base: number, strength: number, order: number) => {
      const m = new THREE.Mesh(geometry, new THREE.ShaderMaterial({
        vertexShader: SHELL_VERTEX, fragmentShader: SHELL_FRAGMENT, transparent: true, depthWrite: false, side,
        uniforms: { rim: { value: rim }, body: { value: body }, strength: { value: strength }, base: { value: base } },
      }));
      m.renderOrder = order;
      return m;
    };
    if (this.shell) { this.scene.remove(this.shell); }
    this.shell = new THREE.Group();
    this.shell.add(glass(THREE.BackSide, 0.025, 0.2, -2));
    this.shell.add(glass(THREE.FrontSide, 0.01, 0.4, 2));
    this.scene.add(this.shell);
    const box = geometry.boundingBox!;
    if (!this.floor) {
      this.floor = new THREE.Mesh(new THREE.PlaneGeometry(1, 1), new THREE.ShaderMaterial({
        vertexShader: 'varying vec2 vUv; void main() { vUv = uv; gl_Position = projectionMatrix * modelViewMatrix * vec4(position, 1.0); }',
        fragmentShader: FLOOR_FRAGMENT, transparent: true, depthWrite: false, uniforms: { tint: { value: this.accent.clone().lerp(this.paper, 0.5) } },
      }));
      this.floor.rotation.x = -Math.PI / 2;
      this.floor.renderOrder = -3;
      this.scene.add(this.floor);
    }
    const span = Math.max(this.size.x, this.size.z) * 1.9;
    this.floor.scale.set(span, span, 1);
    this.floor.position.set(0, box.min.y - this.size.y * 0.32, 0);
    this.axes[0].at.set(0, box.min.y, box.max.z + 1.6);
    this.axes[1].at.set(0, box.min.y, box.min.z - 1.6);
    this.axes[2].at.set(0, box.max.y + this.size.y * 0.42, 0);
    for (const s of this.slabs.values()) if (!s.placed) this.cloud(s);
    this.fit();
  }

  // The sections as the job places them now; a section's new place is flown to.
  setSections(list: Placed[], thicknessUm: number) {
    this.thickness = Math.max(thicknessUm, 1) / 1000;
    const known = new Set(list.map(s => s.id));
    for (const [id, s] of this.slabs) {
      if (!known.has(id)) { this.scene.remove(s.mesh); s.mesh.material.dispose(); this.slabs.delete(id); }
    }
    const fresh = !this.slabs.size;
    const moved: Slab[] = [];
    const sizes: number[] = [];
    for (const item of list) {
      if (item.corners_um && this.map) {
        const [tl, tr] = item.corners_um.map(c => this.map!(c));
        sizes.push(tr.distanceTo(tl));
      }
    }
    sizes.sort((a, b) => a - b);
    const width = sizes.length ? sizes[Math.floor(sizes.length / 2)] : this.size.x * 0.8;
    for (const item of list) {
      let s = this.slabs.get(item.id);
      if (!s) { s = this.make(item.id, item.order, width); if (fresh || !this.reduced) moved.push(s); }
      s.order = item.order;
      const wasPlaced = s.placed;
      const before = s.goal.pos.clone();
      if (item.corners_um && this.map) this.place(s, item.corners_um);
      else { s.placed = false; this.cloud(s, width); }
      if (!fresh && (wasPlaced !== s.placed || before.distanceToSquared(s.goal.pos) > 1e-6)) {
        moved.push(s);
        if (!wasPlaced && s.placed) s.arriving = true;
      }
    }
    if (fresh) {
      for (const s of this.slabs.values()) {
        if (s.placed && !this.reduced) {
          const goal = { pos: s.goal.pos.clone(), quat: s.goal.quat.clone(), size: s.goal.size.clone(), shear: s.goal.shear.clone() };
          this.cloud(s, width);
          this.snap(s);
          s.goal = goal;
          s.arriving = true;
        } else this.snap(s);
      }
    }
    moved.sort((a, b) => a.order - b.order);
    moved.forEach((s, i) => { s.wait = this.reduced ? 0 : Math.min(STAGGER_MAX_S, (fresh ? 0.5 : 0) + i * STAGGER_S); });
    if (this.reduced) for (const s of this.slabs.values()) this.snap(s);
  }

  setImage(id: string, url: string, img: HTMLImageElement) {
    const s = this.slabs.get(id);
    if (!s || s.url === url) return;
    s.url = url;
    let tex = this.textures.get(id);
    if (tex) tex.dispose();
    tex = new THREE.Texture(img);
    tex.flipY = false;
    tex.colorSpace = THREE.NoColorSpace;
    tex.anisotropy = Math.min(8, this.renderer.capabilities.getMaxAnisotropy());
    tex.needsUpdate = true;
    this.textures.set(id, tex);
    s.mesh.material.uniforms.map.value = tex;
    s.mesh.material.uniforms.hasMap.value = 1;
    const aspect = img.naturalWidth / Math.max(1, img.naturalHeight);
    if (Math.abs(aspect - s.aspect) > 0.01) {
      s.aspect = aspect;
      if (!s.placed) s.goal.size.y = s.goal.size.x / aspect;
    }
  }

  setFocus(ids: string[]) {
    this.focused = new Set(ids);
  }

  private make(id: string, order: number, width: number): Slab {
    const material = new THREE.ShaderMaterial({
      vertexShader: VERTEX, fragmentShader: FRAGMENT, transparent: true, depthWrite: true, side: THREE.DoubleSide,
      uniforms: {
        map: { value: null }, hasMap: { value: 0 }, focus: { value: 0 }, dim: { value: 0 }, hover: { value: 0 },
        flash: { value: 0 }, placed: { value: 0 }, accent: { value: this.accent }, paper: { value: this.paper },
      },
    });
    const mesh = new THREE.Mesh(this.geometry, material);
    mesh.matrixAutoUpdate = false;
    mesh.userData.id = id;
    this.scene.add(mesh);
    const s: Slab = {
      id, order, mesh,
      pos: { x: new THREE.Vector3(), v: new THREE.Vector3() },
      size: { x: new THREE.Vector3(1, 1, 1), v: new THREE.Vector3() },
      quat: new THREE.Quaternion(), spin: new THREE.Vector3(),
      goal: { pos: new THREE.Vector3(), quat: new THREE.Quaternion(), size: new THREE.Vector3(1, 1, 1), shear: new THREE.Matrix4() },
      placed: false, wait: 0, born: performance.now(), arriving: false, flash: 0, focus: 0, hover: 0, aspect: ASPECT, url: '', phase: hash(id, 7) * Math.PI * 2,
    };
    this.slabs.set(id, s);
    this.cloud(s, width);
    this.snap(s);
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

  // A section's place in the cloud of sections not placed yet: anywhere around and
  // through the brain, turned any way, the size of a placed section.
  private cloud(s: Slab, width = this.size.x * 0.8) {
    const r = (k: number) => hash(s.id, k);
    const pos = new THREE.Vector3(
      (r(1) - 0.5) * this.size.x * 2.3,
      (r(2) - 0.5) * this.size.y * 1.7 + this.size.y * 0.15,
      (r(3) - 0.5) * this.size.z * 1.35,
    );
    const quat = new THREE.Quaternion().setFromEuler(new THREE.Euler(
      (r(4) - 0.5) * Math.PI * 1.6, r(5) * Math.PI * 2, (r(6) - 0.5) * Math.PI * 1.2,
    ));
    const w = width * (0.6 + 0.25 * r(8));
    s.goal = { pos, quat, size: new THREE.Vector3(w, w / s.aspect, this.thickness), shear: new THREE.Matrix4() };
  }

  private snap(s: Slab) {
    s.pos.x.copy(s.goal.pos); s.pos.v.set(0, 0, 0);
    s.size.x.copy(s.goal.size); s.size.v.set(0, 0, 0);
    s.quat.copy(s.goal.quat); s.spin.set(0, 0, 0);
    this.write(s);
  }

  private fit() {
    const r = 0.5 * Math.hypot(this.size.x, this.size.y * 1.2, this.size.z) * 0.78;
    const aspect = this.camera.aspect;
    const half = THREE.MathUtils.degToRad(FOV / 2);
    const fit = Math.min(half, Math.atan(Math.tan(half) * aspect));
    const dist = r / Math.sin(fit);
    const dir = new THREE.Vector3(-0.8, 0.5, 1.3).normalize();
    if (!this.fitted || !this.touched) this.camera.position.copy(dir.multiplyScalar(dist));
    this.controls.target.set(0, this.size.y * 0.06, 0);
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
    this.step(dt, now / 1000);
    this.controls.update(dt);
    this.pick();
    this.renderer.render(this.scene, this.camera);
    this.label();
  };

  private step(dt: number, t: number) {
    const some = this.focused.size > 0;
    const lift = this.size.y * LIFT;
    const w2 = OMEGA * OMEGA, damp = 2 * ZETA * OMEGA;
    const err = new THREE.Quaternion(), inv = new THREE.Quaternion(), turn = new THREE.Quaternion();
    for (const s of this.slabs.values()) {
      const on = this.focused.has(s.id) && s.placed;
      const u = s.mesh.material.uniforms;
      const k = this.reduced ? 1 : 1 - Math.exp(-dt * 9);
      s.focus += ((on ? 1 : 0) - s.focus) * k;
      s.hover += ((this.hovered === s.id ? 1 : 0) - s.hover) * (this.reduced ? 1 : 1 - Math.exp(-dt * 14));
      u.focus.value = s.focus;
      u.hover.value = s.hover;
      u.dim.value += (((some && !on) ? 0.55 : s.placed ? 0 : 0.2) - u.dim.value) * k;
      u.placed.value += ((s.placed ? 1 : 0) - u.placed.value) * k;
      const blank = u.hasMap.value < 0.5 && performance.now() - s.born < PICTURE_WAIT_MS;
      if (s.wait > 0 || (blank && s.arriving)) { if (!blank) s.wait -= dt; this.write(s); continue; }
      const goal = s.goal.pos.clone();
      if (on) goal.y += lift;
      if (!s.placed && !this.reduced) {
        goal.y += Math.sin(t * 0.5 + s.phase) * 0.18;
        goal.x += Math.cos(t * 0.37 + s.phase) * 0.12;
      }
      if (this.reduced) {
        s.pos.x.copy(goal); s.size.x.copy(s.goal.size); s.quat.copy(s.goal.quat);
      } else {
        const a = goal.sub(s.pos.x).multiplyScalar(w2).addScaledVector(s.pos.v, -damp);
        s.pos.v.addScaledVector(a, dt); s.pos.x.addScaledVector(s.pos.v, dt);
        const b = s.goal.size.clone().sub(s.size.x).multiplyScalar(w2).addScaledVector(s.size.v, -damp);
        s.size.v.addScaledVector(b, dt); s.size.x.addScaledVector(s.size.v, dt);
        inv.copy(s.quat).invert();
        err.copy(s.goal.quat).multiply(inv);
        if (err.w < 0) { err.x = -err.x; err.y = -err.y; err.z = -err.z; err.w = -err.w; }
        const angle = 2 * Math.acos(Math.min(1, err.w));
        const sin = Math.sqrt(Math.max(0, 1 - err.w * err.w));
        const axis = sin > 1e-6 ? new THREE.Vector3(err.x / sin, err.y / sin, err.z / sin) : new THREE.Vector3();
        const torque = axis.multiplyScalar(angle * w2).addScaledVector(s.spin, -damp);
        s.spin.addScaledVector(torque, dt);
        const spinAngle = s.spin.length() * dt;
        if (spinAngle > 1e-7) {
          turn.setFromAxisAngle(s.spin.clone().normalize(), spinAngle);
          s.quat.premultiply(turn).normalize();
        }
      }
      if (s.arriving && s.placed && s.pos.x.distanceTo(s.goal.pos) < 0.08 && s.pos.v.length() < 0.6) {
        s.arriving = false;
        s.flash = this.reduced ? 0 : 1;
      }
      s.flash = Math.max(0, s.flash - dt / FLASH_S);
      u.flash.value = s.flash * s.flash;
      this.write(s);
    }
  }

  private write(s: Slab) {
    const m = s.mesh.matrix;
    m.compose(s.pos.x, s.quat, s.size.x);
    if (s.placed) m.multiply(s.goal.shear);
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
    const meshes = [...this.slabs.values()].map(s => s.mesh);
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
