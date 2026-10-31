'use strict';
// The positioning strip: the ABBA viewer's positioning overview, drawn as one inline
// SVG. Along the top, the atlas's reference image at regular positions of the cutting
// axis (`jobs/<n>/strip` names them; `render/atlas/<k>` draws them); under it the axis
// with a millimetre ruler and its two ends named; below, every placed section in at
// most three stair rows (each in the first row where its thumbnail does not overlap its
// neighbour), drawn as its thumbnail (`render/thumb/<section>`) with a round handle and
// a dashed leader up to a square key on the axis at its position. The focus's sections
// take their panel's colour, and a pink bar on the axis spans them. Sections with no
// position wait in a group at the right end. The wheel zooms the axis about the
// pointer, dragging pans it, a double-click shows the whole axis; hovering names a
// section and clicking puts it in focus. Without the strip route (an older server) the
// strip is drawn from the snapshot's rows, without pictures.
const Strip = (() => {
  const NS = 'http://www.w3.org/2000/svg';
  const MAX_ROWS = 3;
  const PAD = 12;               // side margins, px
  const TOP = 4, BOTTOM = 5;    // above the atlas row, under the last stair row
  const RULER = 17;             // from the atlas row down to the axis line
  const GAP = 7;                // from the axis line to the first stair row
  const ROW_STEP = 0.88;        // stair rows overlap a little, as ABBA's did
  const MAX_ZOOM = 40;
  const ASPECT = { coronal: 1.4, sagittal: 1.6, horizontal: 0.8 };
  const HANDLES_ONLY = 18;      // a stair row's height when there are no pictures

  let svg, body, tip, empty, title, clipRect;
  let layers = null;
  let job = null;
  let info = null;              // the server's strip; null when it has none
  let reason = '';              // why there is none (the server's refusal), for the title
  let plane = null;
  let rows = [];
  let all = [];                 // the focus's sections, every page
  let slots = new Map();        // section -> panel slot (1-4) on the shown page
  let onPick = () => {};
  let marks = new Map();        // section -> its elements
  let atlas = [];               // the atlas thumbnails
  let aspect = null;            // width / height of the atlas pictures, once one loads
  let aspectAsked = '';
  let layout = null;            // the whole-axis layout: the axis, the stair rows
  let view = { zoom: 1, center: null };
  let edge = 64;                // the pictures' requested long edge
  let edgeChosen = false;       // set from the first layout, before any picture is asked for
  let refreshTimer = null;
  let hover = null;

  const svgEl = (tag, attrs = {}) => {
    const e = document.createElementNS(NS, tag);
    for (const [key, value] of Object.entries(attrs)) e.setAttribute(key, String(value));
    return e;
  };
  const set = (e, attrs) => { for (const [key, value] of Object.entries(attrs)) e.setAttribute(key, String(value)); };
  const placed = row => typeof row.position_mm === 'number' && Number.isFinite(row.position_mm);
  const dpr = () => window.devicePixelRatio || 1;

  function init(options) {
    svg = document.getElementById('strip-svg');
    body = document.getElementById('strip-body');
    tip = document.getElementById('strip-tip');
    empty = document.getElementById('strip-empty');
    title = document.getElementById('strip-title');
    onPick = options.pick;
    const defs = svgEl('defs');
    const clip = svgEl('clipPath', { id: 'strip-clip', clipPathUnits: 'userSpaceOnUse' });
    clipRect = svgEl('rect', { x: 0, y: 0, width: 0, height: 0 });
    clip.append(clipRect);
    defs.append(clip);
    const group = name => svgEl('g', { class: name });
    layers = {
      atlas: group('atlas'), thumbs: group('thumbs'), leaders: group('leaders'), outlines: group('outlines'),
      ruler: group('ruler'), span: group('spans'), keys: group('keys'), handles: group('handles'),
      ends: group('ends'), loose: group('loose'),
    };
    for (const name of ['atlas', 'thumbs', 'leaders', 'outlines', 'ruler', 'span', 'keys']) {
      layers[name].setAttribute('clip-path', 'url(#strip-clip)');
    }
    svg.append(defs, layers.atlas, layers.loose, layers.thumbs, layers.leaders, layers.outlines, layers.ruler,
      layers.span, layers.keys, layers.handles, layers.ends);
    listen();
    new ResizeObserver(() => relayout()).observe(body);
  }

  // --- data -----------------------------------------------------------------------------
  function setJob(index) {
    job = index;
    info = null; reason = ''; plane = null; rows = []; all = []; slots = new Map();
    aspect = null; aspectAsked = ''; layout = null; view = { zoom: 1, center: null }; edge = 64; edgeChosen = false;
    releaseAll();
    for (const layer of Object.values(layers)) layer.replaceChildren();
    marks = new Map();
    atlas = [];
    hideTip();
    relayout();
  }
  function setPlane(value) {
    if (value && value !== plane) { plane = value; if (!aspect) relayout(); }
  }
  function setInfo(value, why = '') {
    const before = info ? JSON.stringify([info.axis, info.step_mm, info.atlas, info.angles]) : '';
    info = value && typeof value === 'object' && value.axis ? value : null;
    reason = info ? '' : why;
    const after = info ? JSON.stringify([info.axis, info.step_mm, info.atlas, info.angles]) : '';
    if (before !== after) buildAtlas();
    relayout();
  }
  function setRows(value) {
    rows = value || [];
    const ids = new Set(rows.map(r => r.id));
    for (const [id, mark] of marks) {
      if (!ids.has(id)) { for (const part of mark.parts()) { release(part); part.remove(); } marks.delete(id); }
    }
    for (const row of rows) {
      let mark = marks.get(row.id);
      if (!mark) { mark = makeMark(row.id); marks.set(row.id, mark); }
      mark.row = row;
    }
    relayout();
  }
  // `everyId`: the focus's sections on every page; `pageIds`: the shown page's, in panel order.
  function setTargets(everyId, pageIds) {
    all = everyId || [];
    slots = new Map((pageIds || []).map((id, i) => [id, i + 1]));
    for (const [id, mark] of marks) mark.classes(all.includes(id), slots.get(id));
    // The focus's sections draw over the others.
    for (const id of all) {
      const mark = marks.get(id);
      if (!mark) continue;
      for (const part of [mark.leader, mark.outline, mark.key, mark.handle]) part.parentNode?.append(part);
    }
    place();
  }

  function makeMark(id) {
    const mark = {
      id,
      row: null,
      thumb: { img: null, next: null, href: '' },
      leader: svgEl('line', { class: 'leader' }),
      outline: svgEl('rect', { class: 'outline' }),
      key: svgEl('rect', { class: 'key' }),
      handle: svgEl('circle', { class: 'handle' }),
      parts() { return [this.thumb.img, this.thumb.next, this.leader, this.outline, this.key, this.handle].filter(Boolean); },
      classes(target, slot) {
        const kind = target ? (slot ? 'slot-' + slot : 'other') : '';
        for (const [part, base] of [[this.leader, 'leader'], [this.outline, 'outline'], [this.key, 'key'], [this.handle, 'handle']]) {
          part.setAttribute('class', base + (target ? ' target ' + kind : ''));
        }
      },
    };
    for (const part of [mark.handle, mark.outline]) part.dataset.id = id;
    layers.leaders.append(mark.leader);
    layers.outlines.append(mark.outline);
    layers.keys.append(mark.key);
    layers.handles.append(mark.handle);
    mark.classes(all.includes(id), slots.get(id));
    return mark;
  }

  function buildAtlas() {
    for (const entry of atlas) release(entry.next);
    layers.atlas.replaceChildren();
    atlas = (info?.atlas || []).map(entry => ({ k: entry.k, position: entry.position_mm, img: null, next: null, href: '' }));
  }

  // --- pictures -------------------------------------------------------------------------------
  // The strip's pictures load through the window's queue (Ui.loads) after the focus panels':
  // the focus's sections first, then the atlas row, then the other sections, and only those
  // in view. A new picture fades in over the one it replaces, which goes when the new one has
  // loaded; one the server is still drawing (503) is asked for again after a pause.
  const release = image => image?.release?.();
  function picture(holder, layer, href, cls, rank, id) {
    if (holder.href === href) return;
    holder.href = href;
    if (holder.next) { release(holder.next); holder.next.remove(); holder.next = null; }
    if (!href) { holder.img?.remove(); holder.img = null; return; }
    const image = svgEl('image', { class: cls + ' entering', preserveAspectRatio: 'xMidYMid meet' });
    if (id) image.dataset.id = id;
    image.addEventListener('load', () => {
      release(image);
      if (holder.next !== image) return;
      holder.img?.remove();
      holder.img = image;
      holder.next = null;
      requestAnimationFrame(() => image.classList.remove('entering'));
    }, { once: true });
    image.addEventListener('error', async () => {
      const answer = await Ui.loads.explain(href);
      release(image);
      if (holder.next !== image) return;
      holder.next = null;
      image.remove();
      holder.tries = holder.tried === href ? holder.tries + 1 : 1;
      holder.tried = href;
      if ((answer.ok || answer.status === 503) && holder.tries < 6) {
        setTimeout(() => { if (holder.href === href) { holder.href = ''; place(); } }, answer.ok ? 0 : 3000);
      }
    }, { once: true });
    holder.next = image;
    layer.append(image);
    if (holder.box) set(image, holder.box);
    image.release = Ui.loads.add(rank, () => image.setAttribute('href', href));
  }
  function releaseAll() {
    for (const mark of marks.values()) release(mark.thumb.next);
    for (const entry of atlas) release(entry.next);
  }
  // A section's thumbnail is asked for again only when its render key changes.
  function thumbHref(row) {
    if (!info || job === null) return '';
    const key = row.render_key ?? [row.flip ? 'f' : '', row.rotation_deg || 0].join('');
    return `jobs/${job}/render/thumb/${encodeURIComponent(row.id)}?edge=${edge}&key=${encodeURIComponent(key)}`;
  }
  function atlasHref(entry) {
    if (!info || job === null) return '';
    return `jobs/${job}/render/atlas/${entry.k}?edge=${edge}&key=${encodeURIComponent(String(info.angles || '') + info.step_mm)}`;
  }
  // The thumbnails' shape follows the atlas's pictures: learn it from the first one.
  function learnAspect() {
    const first = atlas[0];
    if (!first || aspect || job === null) return;
    const href = atlasHref(first);
    if (!href || aspectAsked === href) return;
    aspectAsked = href;
    const probe = new Image();
    probe.onload = () => {
      probe.release();
      if (aspectAsked !== href || !probe.naturalHeight) return;
      aspect = Math.min(2.4, Math.max(0.45, probe.naturalWidth / probe.naturalHeight));
      relayout();
    };
    probe.onerror = () => probe.release();
    probe.release = Ui.loads.add(() => 0.5, () => { probe.src = href; });
  }

  // --- layout ---------------------------------------------------------------------------------
  function heightFor(available, n, withAtlas) {
    return available / ((withAtlas ? 1 : 0) + 1 + ROW_STEP * (n - 1));
  }
  // The whole-axis layout: the axis's span, the unplaced group, and each placed section's
  // stair row, which zooming keeps.
  function relayout() {
    if (!svg) return;
    const W = body.clientWidth, H = body.clientHeight;
    const none = !rows.length;
    empty.hidden = !none;
    empty.textContent = none ? 'No sections yet.' : '';
    svg.style.visibility = none ? 'hidden' : 'visible';
    writeTitle();
    if (none || W < 40 || H < 30) { layout = null; return; }
    // Without the server's strip there are no pictures: the rows hold handles only.
    const withAtlas = Boolean(info);
    const A = withAtlas ? aspect || ASPECT[plane] || 1.3 : 1;
    const free = H - TOP - BOTTOM - (withAtlas ? RULER : RULER - 4) - GAP;
    const onAxis = rows.filter(placed);
    const loose = rows.filter(r => !placed(r));
    let lo, hi;
    if (info) { lo = info.axis.min_mm; hi = info.axis.max_mm; }
    else if (onAxis.length) {
      lo = Math.min(...onAxis.map(r => r.position_mm)) - 0.5;
      hi = Math.max(...onAxis.map(r => r.position_mm)) + 0.5;
    } else { lo = 0; hi = 1; }
    if (!(hi > lo)) hi = lo + 1;
    const th3 = Math.max(8, heightFor(free, MAX_ROWS, withAtlas));
    const tw3 = th3 * A;
    // The unplaced group: the right end when some are placed, else the whole stair band.
    let looseW = 0;
    if (loose.length && onAxis.length) {
      const lr = Math.min(MAX_ROWS, loose.length);
      const want = Math.ceil(loose.length / lr) * (tw3 + 4) + 12;
      looseW = Math.min(want, Math.max(tw3 + 16, W * 0.3));
    }
    const axisLeft = PAD + tw3 / 2;
    const axisRight = Math.max(axisLeft + 20, W - PAD - tw3 / 2 - (looseW ? looseW + 14 : 0));
    const basePx = (axisRight - axisLeft) / (hi - lo);
    const step = info?.step_mm || 0.5;
    const widthCap = withAtlas ? (step * basePx * 0.94) / A : HANDLES_ONLY;
    // Stair rows: the fewest rows (at most three) in which no thumbnail overlaps its
    // neighbour; the height holds as many rows as the unplaced group uses, if more.
    const looseRows = Math.min(MAX_ROWS, loose.length);
    const order = [...onAxis].sort((a, b) => a.position_mm - b.position_mm || a.order - b.order);
    let chosen = null;
    for (let n = 1; n <= MAX_ROWS; n++) {
      const th = Math.max(8, Math.min(heightFor(free, Math.max(n, looseRows), withAtlas), widthCap));
      const tw = th * A;
      const ends = new Array(n).fill(-Infinity);
      const stair = new Map();
      let clear = true;
      for (const row of order) {
        const x = axisLeft + (row.position_mm - lo) * basePx;
        let r = ends.findIndex(end => x - end >= tw + 2);
        if (r < 0) { clear = false; r = ends.indexOf(Math.min(...ends)); }
        stair.set(row.id, r);
        ends[r] = x;
      }
      chosen = { n: Math.max(n, looseRows), th, stair };
      if (clear) break;
    }
    layout = { W, H, lo, hi, A, free, withAtlas, axisLeft, axisRight, basePx, step, looseW, looseRows, n: chosen.n, stair: chosen.stair, loose };
    if (view.center === null) view.center = (lo + hi) / 2;
    clampView();
    place();
    learnAspect();
  }
  function clampView() {
    if (!layout) return;
    const { lo, hi, axisLeft, axisRight, basePx } = layout;
    view.zoom = Math.min(MAX_ZOOM, Math.max(1, view.zoom));
    const half = (axisRight - axisLeft) / 2 / (basePx * view.zoom);
    view.center = Math.min(hi - half, Math.max(lo + half, view.center ?? (lo + hi) / 2));
  }
  const pxPerMm = () => layout.basePx * view.zoom;
  const toX = mm => (layout.axisLeft + layout.axisRight) / 2 + (mm - view.center) * pxPerMm();
  const toMm = x => view.center + (x - (layout.axisLeft + layout.axisRight) / 2) / pxPerMm();

  // Positions every mark for the current zoom and pan.
  function place() {
    if (!layout) return;
    const L = layout;
    const th = Math.max(8, Math.min(heightFor(L.free, L.n, L.withAtlas), L.withAtlas ? (L.step * pxPerMm() * 0.94) / L.A : HANDLES_ONLY));
    const tw = th * L.A;
    const thA = L.withAtlas ? th : 0;
    // Room the thumbnails do not need spreads the stair rows (up to 1.2 heights apart),
    // and what is left centres the drawing.
    let spare = Math.max(0, L.free - thA - th * (1 + ROW_STEP * (L.n - 1)));
    let rowStep = th * ROW_STEP;
    if (L.n > 1) {
      const grow = Math.min(spare / (L.n - 1), th * (1.2 - ROW_STEP));
      rowStep += grow;
      spare -= grow * (L.n - 1);
    }
    const top = TOP + spare / 2;
    const yAxis = top + thA + (L.withAtlas ? RULER : RULER - 4);
    const rowY = r => yAxis + GAP + th / 2 + r * rowStep;
    // Small pictures: 128 px, 256 once the strip draws them large.
    const wanted = Math.max(tw, th) * dpr() > 150 ? 256 : 128;
    if (!edgeChosen) { edge = wanted; edgeChosen = true; }
    else if (wanted > edge) scheduleRefresh(wanted);
    set(clipRect, { x: L.axisLeft - tw / 2 - 1, y: 0, width: L.axisRight - L.axisLeft + tw + 2, height: L.H });
    // The atlas row.
    for (const entry of atlas) {
      const x = toX(entry.position);
      entry.box = { x: x - tw / 2, y: top, width: tw, height: thA };
      const visible = x > L.axisLeft - tw && x < L.axisRight + tw;
      if (visible) picture(entry, layers.atlas, atlasHref(entry), 'atlas-thumb', () => 2);
      for (const image of [entry.img, entry.next]) if (image) { set(image, entry.box); image.style.display = visible ? '' : 'none'; }
    }
    // The axis, its ruler and its ends.
    drawRuler(yAxis);
    // The placed sections.
    const targetsX = [];
    for (const [id, mark] of marks) {
      const row = mark.row;
      if (!row) continue;
      const target = all.includes(id);
      const radius = target ? 6 : 5;
      let cx, cy;
      if (placed(row)) {
        cx = toX(row.position_mm);
        cy = rowY(L.stair.get(id) ?? 0);
        const key = target ? 8 : 6;
        set(mark.leader, { x1: cx, y1: yAxis + key / 2, x2: cx, y2: cy - radius });
        set(mark.key, { x: cx - key / 2, y: yAxis - key / 2, width: key, height: key });
        mark.leader.style.display = mark.key.style.display = '';
        if (target) targetsX.push(cx);
        mark.handle.setAttribute('clip-path', 'url(#strip-clip)');
      } else {
        [cx, cy] = loosePlace(row, tw, th, rowY);
        mark.leader.style.display = mark.key.style.display = 'none';
        mark.handle.removeAttribute('clip-path');
      }
      mark.thumb.box = { x: cx - tw / 2, y: cy - th / 2, width: tw, height: th };
      const layer = placed(row) ? layers.thumbs : layers.loose;
      const inView = !placed(row) || (cx > L.axisLeft - tw && cx < L.axisRight + tw);
      if (inView) picture(mark.thumb, layer, thumbHref(row), 'thumb', () => (all.includes(id) ? 1 : 3), id);
      for (const image of [mark.thumb.img, mark.thumb.next]) {
        if (!image) continue;
        if (image.parentNode !== layer) layer.append(image);
        set(image, mark.thumb.box);
      }
      set(mark.outline, { x: cx - tw / 2, y: cy - th / 2, width: tw, height: th });
      mark.outline.style.display = L.withAtlas ? '' : 'none';
      set(mark.handle, { cx, cy, r: radius });
    }
    // The pink bar spanning the focus's sections.
    layers.span.replaceChildren();
    if (targetsX.length > 1) {
      layers.span.append(svgEl('line', { class: 'span', x1: Math.min(...targetsX), y1: yAxis, x2: Math.max(...targetsX), y2: yAxis }));
    }
    drawLoose(yAxis, tw, th, rowY);
  }
  // The unplaced group's grid: as many columns as its rows need, overlapping when narrow.
  function looseGrid(tw) {
    const L = layout;
    const cols = Math.ceil(L.loose.length / L.looseRows);
    const x = L.looseW ? L.W - PAD - L.looseW : PAD;
    const left = x + 6 + tw / 2;
    const right = (L.looseW ? L.W - PAD : L.W - PAD) - tw / 2 - 6;
    const stepX = cols > 1 ? Math.min(tw + 4, (right - left) / (cols - 1)) : 0;
    return { cols, x, left, stepX, width: (cols - 1) * stepX + tw + 12 };
  }
  function loosePlace(row, tw, th, rowY) {
    const index = layout.loose.indexOf(row);
    const grid = looseGrid(tw);
    return [grid.left + (index % grid.cols) * grid.stepX, rowY(Math.floor(index / grid.cols))];
  }
  function drawLoose(yAxis, tw, th, rowY) {
    const L = layout;
    layers.loose.querySelectorAll('.group-frame, .group-label').forEach(e => e.remove());
    if (!L.loose.length) return;
    const grid = looseGrid(tw);
    const bottom = rowY(L.looseRows - 1) + th / 2 + 2;
    const width = L.looseW ? L.looseW : Math.min(grid.width, L.W - 2 * PAD);
    layers.loose.prepend(svgEl('rect', { class: 'group-frame', x: grid.x, y: yAxis + 3, width, height: Math.max(10, bottom - yAxis - 3) }));
    // Beside placed sections the group is named; alone, the strip's title counts it.
    if (L.looseW) {
      const label = svgEl('text', { class: 'group-label', x: grid.x + 4, y: yAxis - 7 });
      label.textContent = `not placed (${L.loose.length})`;
      layers.loose.append(label);
    }
  }
  function drawRuler(yAxis) {
    const L = layout;
    const px = pxPerMm();
    // Ticks every millimetre at the whole axis, finer once zoomed; labels every few.
    const tick = [0.1, 0.5, 1, 2, 5].find(t => t * px >= 45) || 10;
    const label = [0.5, 1, 2, 5, 10, 20].find(t => t * px >= 90 && t >= tick) || 50;
    const from = Math.max(L.lo, toMm(L.axisLeft - 2)), to = Math.min(L.hi, toMm(L.axisRight + 2));
    const ruler = layers.ruler, ends = layers.ends;
    ruler.replaceChildren();
    ends.replaceChildren();
    ruler.append(svgEl('line', { class: 'axis', x1: toX(from), y1: yAxis, x2: toX(to), y2: yAxis }));
    const low = info?.axis?.low || '', high = info?.axis?.high || '';
    // A tick label keeps clear of the axis ends' names (6 px a character at 10 px).
    const lowEnd = L.axisLeft - 2 + low.length * 6 + 8, highStart = L.axisRight + 2 - high.length * 6 - 8;
    const digits = label < 1 ? 1 : 0;
    for (let i = Math.ceil(from / tick - 1e-6); i * tick <= to + 1e-9; i++) {
      const mm = i * tick, x = toX(mm);
      const labelled = Math.abs(mm / label - Math.round(mm / label)) < 1e-6;
      ruler.append(svgEl('line', { class: 'tick', x1: x, y1: yAxis - (labelled ? 5 : 3), x2: x, y2: yAxis }));
      const words = mm.toFixed(digits);
      if (labelled && x - words.length * 3 > lowEnd && x + words.length * 3 < highStart) {
        const text = svgEl('text', { x, y: yAxis - 7, 'text-anchor': 'middle' });
        text.textContent = words;
        ruler.append(text);
      }
    }
    for (const [name, x, anchor] of [[low, L.axisLeft - 2, 'start'], [high, L.axisRight + 2, 'end']]) {
      if (!name) continue;
      const text = svgEl('text', { class: 'axis-end', x, y: yAxis - 7, 'text-anchor': anchor });
      text.textContent = name;
      ends.append(text);
    }
  }
  // Sharper pictures once the thumbnails have grown, after the zoom settles.
  function scheduleRefresh(wanted) {
    clearTimeout(refreshTimer);
    refreshTimer = setTimeout(() => { if (wanted > edge) { edge = wanted; place(); } }, 400);
  }
  function writeTitle() {
    const parts = [`${rows.length} section${rows.length === 1 ? '' : 's'}`];
    if (info?.step_mm) parts.push(`atlas every ${info.step_mm} mm`);
    else if (reason) parts.push('no pictures');
    const unplaced = rows.filter(r => !placed(r)).length;
    if (unplaced && rows.length) parts.push(`${unplaced} not placed`);
    title.textContent = '· ' + parts.join(' · ');
    // A refusal's words are long: the panels show them, the title keeps them on hover.
    title.title = info ? '' : reason;
  }

  // --- interaction ------------------------------------------------------------------------------
  function listen() {
    let press = null;
    svg.addEventListener('pointerdown', event => {
      if (event.button !== 0 || !layout) return;
      press = { x: event.clientX, center: view.center, moved: false, id: event.target.closest?.('[data-id]')?.dataset.id };
      svg.setPointerCapture(event.pointerId);
    });
    svg.addEventListener('pointermove', event => {
      if (press) {
        const dx = event.clientX - press.x;
        if (!press.moved && Math.abs(dx) > 3) { press.moved = true; svg.classList.add('panning'); hideTip(); }
        if (press.moved) { view.center = press.center - dx / pxPerMm(); clampView(); place(); }
        return;
      }
      const id = event.target.closest?.('[data-id]')?.dataset.id;
      if (id) showTip(id, event); else hideTip();
    });
    const release = event => {
      if (!press) return;
      const { moved, id } = press;
      press = null;
      svg.classList.remove('panning');
      if (svg.hasPointerCapture?.(event.pointerId)) svg.releasePointerCapture(event.pointerId);
      if (!moved && id && event.type === 'pointerup') onPick(id);
    };
    svg.addEventListener('pointerup', release);
    svg.addEventListener('pointercancel', release);
    svg.addEventListener('pointerleave', () => { if (!press) hideTip(); });
    svg.addEventListener('wheel', event => {
      if (!layout) return;
      event.preventDefault();
      const box = svg.getBoundingClientRect();
      const x = event.clientX - box.left;
      if (Math.abs(event.deltaX) > Math.abs(event.deltaY)) {
        view.center += event.deltaX / pxPerMm();
      } else {
        const at = toMm(x);
        view.zoom = Math.min(MAX_ZOOM, Math.max(1, view.zoom * Math.exp(-event.deltaY * 0.0018)));
        view.center = at - (x - (layout.axisLeft + layout.axisRight) / 2) / pxPerMm();
      }
      clampView();
      place();
      hideTip();
    }, { passive: false });
    svg.addEventListener('dblclick', () => { view.zoom = 1; view.center = null; relayout(); });
  }
  function showTip(id, event) {
    const mark = marks.get(id);
    if (!mark?.row) return hideTip();
    if (hover !== id) {
      hover = id;
      const row = mark.row;
      const lines = [placed(row) ? `${Ui.mm(row.position_mm)} mm` : 'not placed'];
      if (row.angles && typeof row.angles === 'object') {
        lines.push(`pitch ${Ui.mm(Number(row.angles.pitch), 1)}° · yaw ${Ui.mm(Number(row.angles.yaw), 1)}°`);
      }
      const facts = [row.transform, row.deformation, row.damaged ? 'damaged' : ''].filter(Boolean);
      if (facts.length) lines.push(facts.join(' · '));
      tip.replaceChildren(Ui.el('div', 'tip-name', `${row.order} · ${row.id}`), ...lines.map(line => Ui.el('div', 'tip-line', line)));
      for (const m of marks.values()) m.handle.classList.toggle('hover', m.id === id);
    }
    tip.hidden = false;
    const box = body.getBoundingClientRect();
    const x = event.clientX - box.left, y = event.clientY - box.top;
    const w = tip.offsetWidth, h = tip.offsetHeight;
    tip.style.left = Math.max(4, Math.min(box.width - w - 4, x + 14)) + 'px';
    tip.style.top = (y + h + 18 < box.height ? y + 14 : Math.max(2, y - h - 10)) + 'px';
  }
  function hideTip() {
    if (!tip) return;
    tip.hidden = true;
    if (hover !== null) { for (const m of marks.values()) m.handle.classList.remove('hover'); hover = null; }
  }

  return { init, setJob, setPlane, setInfo, setRows, setTargets, relayout };
})();
