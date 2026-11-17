// The atlas brain's surface, made in the page from the coarse mask `GET jobs/<n>/shell`
// sends: the bits unpacked, the mask blurred into a soft field (two [1 2 1] passes per
// axis), its half level meshed by surface nets (one vertex per cell the level crosses, at
// the mean of the crossings; a quad per crossed grid edge), the mesh smoothed by Taubin
// passes (smoothing without shrinking) and its faces turned outward. Positions are in
// atlas micrometres, in the atlas's own axis order.

export type Shell = {
  orientation: string;
  shape: [number, number, number];
  step_um: [number, number, number];
  extent_um: [number, number, number];
  bits: string;
};

export type Surface = { positions: Float32Array; indices: Uint32Array };

const PAD = 3;
const ISO = 0.5;
const PASSES = 3;

export function unpack(shell: Shell): Uint8Array {
  const raw = atob(shell.bits);
  const [n0, n1, n2] = shell.shape;
  const out = new Uint8Array(n0 * n1 * n2);
  for (let i = 0; i < out.length; i++) out[i] = (raw.charCodeAt(i >> 3) >> (7 - (i & 7))) & 1;
  return out;
}

function field(shell: Shell, mask: Uint8Array): { f: Float32Array; dims: [number, number, number] } {
  const [n0, n1, n2] = shell.shape;
  const dims: [number, number, number] = [n0 + 2 * PAD, n1 + 2 * PAD, n2 + 2 * PAD];
  const [d0, d1, d2] = dims;
  let f = new Float32Array(d0 * d1 * d2);
  for (let i = 0; i < n0; i++) for (let j = 0; j < n1; j++) for (let k = 0; k < n2; k++) {
    f[((i + PAD) * d1 + (j + PAD)) * d2 + (k + PAD)] = mask[(i * n1 + j) * n2 + k];
  }
  const strides = [d1 * d2, d2, 1];
  for (let pass = 0; pass < 2; pass++) {
    for (let axis = 0; axis < 3; axis++) {
      const s = strides[axis], n = dims[axis];
      const g = new Float32Array(f.length);
      for (let i = 0; i < d0; i++) for (let j = 0; j < d1; j++) for (let k = 0; k < d2; k++) {
        const at = (i * d1 + j) * d2 + k;
        const c = axis === 0 ? i : axis === 1 ? j : k;
        const lo = c > 0 ? f[at - s] : 0, hi = c < n - 1 ? f[at + s] : 0;
        g[at] = 0.25 * lo + 0.5 * f[at] + 0.25 * hi;
      }
      f = g;
    }
  }
  return { f, dims };
}

export function surface(shell: Shell): Surface {
  const { f, dims } = field(shell, unpack(shell));
  const [d0, d1, d2] = dims;
  const at = (i: number, j: number, k: number) => (i * d1 + j) * d2 + k;
  const cell = new Int32Array(d0 * d1 * d2).fill(-1);
  const pos: number[] = [];
  const corner = [[0, 0, 0], [1, 0, 0], [0, 1, 0], [1, 1, 0], [0, 0, 1], [1, 0, 1], [0, 1, 1], [1, 1, 1]];
  const edges: [number, number][] = [];
  for (let a = 0; a < 8; a++) for (let b = a + 1; b < 8; b++) {
    const ca = corner[a], cb = corner[b];
    if (Math.abs(ca[0] - cb[0]) + Math.abs(ca[1] - cb[1]) + Math.abs(ca[2] - cb[2]) === 1) edges.push([a, b]);
  }
  const v = new Float32Array(8);
  for (let i = 0; i < d0 - 1; i++) for (let j = 0; j < d1 - 1; j++) for (let k = 0; k < d2 - 1; k++) {
    let inside = 0;
    for (let c = 0; c < 8; c++) {
      v[c] = f[at(i + corner[c][0], j + corner[c][1], k + corner[c][2])];
      if (v[c] > ISO) inside |= 1 << c;
    }
    if (inside === 0 || inside === 255) continue;
    let x = 0, y = 0, z = 0, n = 0;
    for (const [a, b] of edges) {
      if (((inside >> a) & 1) === ((inside >> b) & 1)) continue;
      const t = (ISO - v[a]) / (v[b] - v[a]);
      x += corner[a][0] + t * (corner[b][0] - corner[a][0]);
      y += corner[a][1] + t * (corner[b][1] - corner[a][1]);
      z += corner[a][2] + t * (corner[b][2] - corner[a][2]);
      n++;
    }
    cell[at(i, j, k)] = pos.length / 3;
    pos.push(i + x / n, j + y / n, k + z / n);
  }
  const quads: number[] = [];
  const unit = [[1, 0, 0], [0, 1, 0], [0, 0, 1]];
  for (let i = 1; i < d0 - 1; i++) for (let j = 1; j < d1 - 1; j++) for (let k = 1; k < d2 - 1; k++) {
    const here = f[at(i, j, k)] > ISO;
    for (let axis = 0; axis < 3; axis++) {
      const e = unit[axis];
      if (here === f[at(i + e[0], j + e[1], k + e[2])] > ISO) continue;
      const b = unit[(axis + 1) % 3], c = unit[(axis + 2) % 3];
      const q = [
        cell[at(i, j, k)],
        cell[at(i - b[0], j - b[1], k - b[2])],
        cell[at(i - b[0] - c[0], j - b[1] - c[1], k - b[2] - c[2])],
        cell[at(i - c[0], j - c[1], k - c[2])],
      ];
      if (q.some(n => n < 0)) continue;
      if (here) quads.push(q[0], q[1], q[2], q[3]); else quads.push(q[0], q[3], q[2], q[1]);
    }
  }
  const positions = new Float32Array(pos);
  smooth(positions, quads);
  const [s0, s1, s2] = shell.step_um;
  for (let p = 0; p < positions.length; p += 3) {
    positions[p] = (positions[p] - PAD) * s0;
    positions[p + 1] = (positions[p + 1] - PAD) * s1;
    positions[p + 2] = (positions[p + 2] - PAD) * s2;
  }
  const indices = new Uint32Array((quads.length / 4) * 6);
  for (let q = 0, o = 0; q < quads.length; q += 4, o += 6) {
    indices[o] = quads[q]; indices[o + 1] = quads[q + 1]; indices[o + 2] = quads[q + 2];
    indices[o + 3] = quads[q]; indices[o + 4] = quads[q + 2]; indices[o + 5] = quads[q + 3];
  }
  outward(positions, indices);
  return { positions, indices };
}

// Taubin smoothing over the quads' edges: a shrinking step then an inflating one.
function smooth(p: Float32Array, quads: number[]) {
  const n = p.length / 3;
  const near: Set<number>[] = Array.from({ length: n }, () => new Set());
  for (let q = 0; q < quads.length; q += 4) {
    for (let s = 0; s < 4; s++) {
      const a = quads[q + s], b = quads[q + ((s + 1) % 4)];
      near[a].add(b); near[b].add(a);
    }
  }
  const lists = near.map(s => [...s]);
  const step = (w: number) => {
    const d = new Float32Array(p.length);
    for (let i = 0; i < n; i++) {
      const l = lists[i];
      if (!l.length) continue;
      let x = 0, y = 0, z = 0;
      for (const m of l) { x += p[3 * m]; y += p[3 * m + 1]; z += p[3 * m + 2]; }
      d[3 * i] = x / l.length - p[3 * i];
      d[3 * i + 1] = y / l.length - p[3 * i + 1];
      d[3 * i + 2] = z / l.length - p[3 * i + 2];
    }
    for (let i = 0; i < p.length; i++) p[i] += w * d[i];
  };
  for (let pass = 0; pass < PASSES; pass++) { step(0.5); step(-0.53); }
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
    const nx = uy * vz - uz * vy, ny = uz * vx - ux * vz, nz = ux * vy - uy * vx;
    sum += nx * (p[a] - cx) + ny * (p[a + 1] - cy) + nz * (p[a + 2] - cz);
  }
  if (sum >= 0) return;
  for (let t = 0; t < idx.length; t += 3) { const s = idx[t + 1]; idx[t + 1] = idx[t + 2]; idx[t + 2] = s; }
}
