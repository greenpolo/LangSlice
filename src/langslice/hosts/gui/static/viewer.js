'use strict';
// The agent focus area, the ABBA viewer's comparison window: its header (how many
// sections are in focus, the page, the Focus | Sections | Steps switch, Follow agent and
// the page buttons), the positioning strip (viewer-strip.js) and one of three views:
//   Focus     up to four panels (1, 1 x 2 or 2 x 2), each a section under its current
//             registration with the atlas borders, drawn by the server
//             (`render/overlay/<section>`) at the panel's size; the old picture stays,
//             tagged `updating`, until a newer one has loaded;
//   Sections  the sections table;
//   Steps     undo and redo, the notes and the spend.
// Follow: the focus is the sections of the agent's latest followed call (the snapshot's
// `focus`); a new call replaces them and goes back to page 1. A section the person
// clicks (strip, table, a name in the log) becomes the focus and Follow agent turns off.
const Viewer = (() => {
  const PAGE_SIZE = 4;
  const $ = id => document.getElementById(id);
  const { el } = Ui;

  let job = null;
  let rows = [];
  let byId = new Map();
  let loaded = false;          // the job's tables have arrived
  let agent = null;            // the snapshot's focus
  let agentSeen = '';
  let follow = true;
  let manual = [];             // the sections the person put in focus
  let page = 0;
  let mode = 'focus';
  let panels = [];
  let resizeTimer = null;

  // --- the header, the switch, Follow and paging -------------------------------------------
  function init() {
    for (const button of document.querySelectorAll('[data-view]')) {
      button.addEventListener('click', () => showView(button.dataset.view));
    }
    $('follow-agent').addEventListener('click', toggleFollow);
    $('page-previous').addEventListener('click', () => pageBy(-1));
    $('page-next').addEventListener('click', () => pageBy(1));
    panels = [1, 2, 3, 4].map(makePanel);
    new ResizeObserver(() => {
      clearTimeout(resizeTimer);
      for (const p of panels) if (p.id) fitTitle(p);
      resizeTimer = setTimeout(() => { for (const p of panels) if (p.id) request(p); }, 250);
    }).observe($('view-focus'));
    Strip.init({ pick });
    columns();
    stripSize();
  }
  function setJob(index) {
    job = index;
    rows = []; byId = new Map(); loaded = false;
    agent = null; agentSeen = ''; follow = true; manual = []; page = 0;
    for (const p of panels) clearPanel(p);
    Strip.setJob(index);
    renderSections();
    renderSteps({});
    refresh();
  }
  function setHeader(header) { Strip.setPlane(header?.plane); }
  // One snapshot: its tables (when the state moved; else null) and the agent's focus, applied
  // together so the panels ask only for what the snapshot puts in focus.
  function update(tables, focus) {
    let changed = false;
    if (tables) {
      loaded = true;
      rows = Array.isArray(tables.sections) ? tables.sections : [];
      byId = new Map(rows.map(r => [r.id, r]));
      Strip.setRows(rows);
      renderSections(tables.sections_note || '');
      renderSteps(tables);
      changed = true;
    }
    const seen = JSON.stringify(focus ? [focus.call ?? null, focus.sections || []] : null);
    if (seen !== agentSeen) {
      agentSeen = seen;
      agent = focus;
      if (follow) { page = 0; changed = true; }
    }
    if (changed) refresh();
  }
  function agentTargets() {
    const ids = (agent?.sections || []).filter(id => byId.has(id));
    return ids.length ? [...new Set(ids)] : rows.map(r => r.id);
  }
  function targets() { return follow ? agentTargets() : manual.filter(id => byId.has(id)); }
  function pick(id) {
    if (!byId.has(id)) return;
    manual = [id];
    follow = false;
    page = 0;
    showView('focus');
  }
  function toggleFollow() {
    if (follow) manual = targets();
    follow = !follow;
    page = 0;
    refresh();
  }
  function pageBy(delta) {
    const count = Math.max(1, Math.ceil(targets().length / PAGE_SIZE));
    page = (page + delta + count) % count;
    refresh();
  }
  function showView(name) {
    mode = name;
    for (const button of document.querySelectorAll('[data-view]')) button.setAttribute('aria-pressed', String(button.dataset.view === name));
    $('view-focus').hidden = name !== 'focus';
    $('view-sections').hidden = name !== 'sections';
    $('view-steps').hidden = name !== 'steps';
    refresh();
  }
  // Everything that follows from the focus: the header, the strip's colours, the panels.
  function refresh() {
    const all = targets();
    const pages = Math.max(1, Math.ceil(all.length / PAGE_SIZE));
    page = Math.min(page, pages - 1);
    const onPage = all.slice(page * PAGE_SIZE, page * PAGE_SIZE + PAGE_SIZE);
    $('focus-count').textContent = loaded && rows.length
      ? `· ${all.length} of ${rows.length} section${rows.length === 1 ? '' : 's'} · page ${page + 1}/${pages} · read-only`
      : '· read-only';
    $('follow-agent').setAttribute('aria-pressed', String(follow));
    $('page-previous').disabled = $('page-next').disabled = pages < 2;
    Strip.setTargets(all, onPage);
    for (const tr of document.querySelectorAll('#sections tbody tr')) tr.classList.toggle('target', all.includes(tr.dataset.id));
    const empty = $('focus-empty');
    empty.hidden = true;
    if (mode !== 'focus') return;
    if (!loaded || !rows.length || !onPage.length) {
      empty.textContent = !loaded ? 'Loading the job…' : !rows.length ? 'This job has no sections yet.' : 'Nothing in focus.';
      empty.hidden = false;
    }
    const grid = $('view-focus');
    grid.className = 'panels n' + Math.max(1, onPage.length);
    // 1 fills the area, 2 sit side by side, 3 or 4 in a 2 x 2 grid (a fourth cell stays empty).
    const cells = onPage.length === 3 ? 4 : onPage.length;
    const wanted = panels.slice(0, onPage.length);
    const children = [...wanted.map(p => p.root)];
    if (cells === 4 && onPage.length === 3) children.push(spacer());
    if (grid.children.length !== children.length || children.some((c, i) => grid.children[i] !== c)) grid.replaceChildren(...children);
    onPage.forEach((id, i) => showPanel(panels[i], id));
    for (const p of panels.slice(onPage.length)) clearPanel(p);
  }
  let blank = null;
  function spacer() { return blank || (blank = el('div', 'panel-spacer')); }
  // Called when the viewer tab is shown again: sizes were zero while it was hidden.
  function shown() {
    Strip.relayout();
    for (const p of panels) if (p.id) request(p);
  }

  // --- focus panels -------------------------------------------------------------------------
  function makePanel(slot) {
    const root = el('article', 'panel slot-' + slot);
    const title = el('header', 'panel-title');
    const name = el('span', 'panel-name'), angles = el('span', 'panel-angles'), tags = el('span', 'panel-tags');
    const open = el('button', 'panel-open', 'open');
    open.title = 'Show this picture large';
    title.append(name, angles, tags, open);
    const box = el('div', 'panel-picture');
    const updating = el('span', 'panel-updating', 'updating');
    const message = el('p', 'panel-message');
    updating.hidden = message.hidden = true;
    box.append(updating, message);
    root.append(title, box);
    const p = { slot, root, name, angles, tags, open, box, updating, message, id: null, key: null, edge: 0, url: '', img: null, pending: null, view: { s: 1, x: 0, y: 0 } };
    open.addEventListener('click', () => {
      if (!p.img) return;
      const row = byId.get(p.id);
      const words = [p.angles.textContent, row?.transform, row?.deformation, row?.damaged ? 'damaged' : ''].filter(Boolean);
      AgentLog.openRender(p.img.src, edge => renderUrl(p.id, edge), p.name.textContent,
        'The current registration' + (words.length ? ' · ' + words.join(' · ') : '') + ' · drawn for this window, not a picture the agent saw');
    });
    zoomable(p);
    return p;
  }
  // A load no longer wanted is stopped, freeing its connection.
  function drop(img) {
    img.onload = img.onerror = null;
    img.removeAttribute('src');
  }
  function clearPanel(p) {
    if (p.pending) { drop(p.pending); p.pending = null; }
    p.img?.remove();
    p.img = null; p.id = null; p.key = null; p.edge = 0; p.url = '';
    p.view = { s: 1, x: 0, y: 0 };
    p.updating.hidden = p.message.hidden = true;
  }
  function showPanel(p, id) {
    const row = byId.get(id);
    if (p.id !== id) { clearPanel(p); p.id = id; }
    const position = typeof row?.position_mm === 'number' ? ` · ${Ui.mm(row.position_mm)} mm` : '';
    p.name.textContent = `${p.slot} · ${id}${position}`;
    p.name.title = id;
    const a = row?.angles && typeof row.angles === 'object' ? row.angles : {};
    const pitch = Number(a.pitch) || 0, yaw = Number(a.yaw) || 0;
    p.angles.textContent = [pitch ? `pitch ${pitch.toFixed(1)}°` : '', yaw ? `yaw ${yaw.toFixed(1)}°` : ''].filter(Boolean).join(' · ');
    // Each tag keeps a rank: when the strip is short of room the lowest go first.
    const tags = [];
    const tag = (cls, text, rank) => { const t = el('span', cls, text); t.dataset.rank = rank; tags.push(t); };
    if (row?.transform) tag('tag', row.transform, 1);
    if (row?.deformation) tag('tag method', row.deformation, 2);
    if (row?.damaged) tag('tag damaged', 'damaged', 4);
    if (row && typeof row.position_mm !== 'number') tag('tag', 'not placed', 3);
    p.tags.replaceChildren(...tags);
    p.open.disabled = !p.img;
    fitTitle(p);
    request(p);
  }
  // The angles show only when they fit whole; tags that do not fit go, lowest rank first.
  function fitTitle(p) {
    p.angles.classList.remove('clipped');
    const tags = [...p.tags.children].sort((a, b) => a.dataset.rank - b.dataset.rank);
    for (const t of tags) t.hidden = false;
    if (p.angles.scrollWidth > p.angles.clientWidth + 1) p.angles.classList.add('clipped');
    for (const t of tags) {
      if (p.tags.scrollWidth <= p.tags.clientWidth + 1) break;
      t.hidden = true;
    }
  }
  // The long edge to ask for: the panel's size in device pixels.
  function neededEdge(p) {
    const box = p.box.getBoundingClientRect();
    if (box.width < 8 || box.height < 8) return 0;
    const pixels = Math.max(box.width, box.height) * (window.devicePixelRatio || 1);
    return Math.min(2048, Math.max(128, Math.ceil(pixels / 128) * 128));
  }
  function renderUrl(id, edge) {
    const key = String(byId.get(id)?.render_key ?? '');
    return `jobs/${job}/render/overlay/${encodeURIComponent(id)}?edge=${edge}&key=${encodeURIComponent(key)}`;
  }
  function request(p) {
    if (!p.id || job === null) return;
    const edge = neededEdge(p);
    if (!edge) return;
    const key = String(byId.get(p.id)?.render_key ?? '');
    // The same state at a size already drawn large enough needs nothing new.
    if (p.url && key === p.key && edge <= p.edge) return;
    p.key = key;
    p.edge = edge;
    const url = renderUrl(p.id, edge);
    if (url === p.url) return;
    p.url = url;
    load(p, url);
  }
  function load(p, url) {
    if (p.pending) drop(p.pending);
    const img = new Image();
    img.alt = 'The section ' + p.id + ' under its current registration, with the atlas borders';
    img.decoding = 'async';
    img.draggable = false;
    p.pending = img;
    p.updating.hidden = !p.img;
    if (!p.img) { p.message.textContent = 'Drawing the section…'; p.message.hidden = false; }
    img.onload = () => {
      if (p.pending !== img) return;
      p.pending = null;
      p.updating.hidden = p.message.hidden = true;
      img.className = 'entering';
      transform(p, img);
      const old = p.img;
      p.img = img;
      p.box.prepend(img);
      p.open.disabled = false;
      requestAnimationFrame(() => requestAnimationFrame(() => img.classList.remove('entering')));
      if (old) setTimeout(() => old.remove(), 200);
    };
    img.onerror = async () => {
      if (p.pending !== img) return;
      p.pending = null;
      const text = await failure(url);
      if (p.url !== url) return;
      p.updating.hidden = true;
      p.img?.remove();
      p.img = null;
      p.open.disabled = true;
      p.message.textContent = text;
      p.message.hidden = false;
    };
    img.src = url;
  }
  // Why a picture did not come: the server's refusal, or what the answer means.
  async function failure(url) {
    try {
      const r = await fetch(url, { cache: 'no-store' });
      if (r.ok) return 'The picture could not be shown.';
      const answer = await r.json().catch(() => ({}));
      if (answer.error) return answer.error;
      if (r.status === 404) return 'No picture of this section is available from this LangSlice.';
      if (r.status === 503) return 'Drawing this section took too long; it is drawn again when the section changes.';
      return 'The picture could not be drawn.';
    } catch {
      return 'LangSlice is not reachable.';
    }
  }
  // Wheel zooms about the pointer, dragging pans, a double-click resets: the picture is
  // transformed in the page; nothing is drawn again while zooming.
  function zoomable(p) {
    const box = p.box;
    box.addEventListener('wheel', event => {
      if (!p.img) return;
      event.preventDefault();
      const r = box.getBoundingClientRect();
      const x = event.clientX - r.left, y = event.clientY - r.top;
      const v = p.view;
      const s = Math.min(24, Math.max(1, v.s * Math.exp(-event.deltaY * 0.0018)));
      v.x = x - ((x - v.x) * s) / v.s;
      v.y = y - ((y - v.y) * s) / v.s;
      v.s = s;
      clampPan(p, r);
      transform(p, p.img);
    }, { passive: false });
    let press = null;
    box.addEventListener('pointerdown', event => {
      if (event.button !== 0 || !p.img) return;
      press = { x: event.clientX, y: event.clientY, vx: p.view.x, vy: p.view.y };
      box.setPointerCapture(event.pointerId);
      box.classList.add('panning');
    });
    box.addEventListener('pointermove', event => {
      if (!press) return;
      p.view.x = press.vx + event.clientX - press.x;
      p.view.y = press.vy + event.clientY - press.y;
      clampPan(p, box.getBoundingClientRect());
      transform(p, p.img);
    });
    const release = () => { press = null; box.classList.remove('panning'); };
    box.addEventListener('pointerup', release);
    box.addEventListener('pointercancel', release);
    box.addEventListener('dblclick', () => { p.view = { s: 1, x: 0, y: 0 }; transform(p, p.img); });
  }
  function clampPan(p, r) {
    const v = p.view;
    v.x = Math.min(0, Math.max(r.width - r.width * v.s, v.x));
    v.y = Math.min(0, Math.max(r.height - r.height * v.s, v.y));
  }
  function transform(p, img) {
    if (!img) return;
    const v = p.view;
    img.style.transform = v.s === 1 ? '' : `translate(${v.x}px, ${v.y}px) scale(${v.s})`;
  }

  // --- the Sections view ----------------------------------------------------------------------
  function renderSections(note = '') {
    const body = document.querySelector('#sections tbody');
    body.replaceChildren();
    const all = targets();
    for (const r of rows) {
      const tr = el('tr');
      tr.dataset.id = r.id;
      tr.classList.toggle('damaged', Boolean(r.damaged));
      tr.classList.toggle('target', all.includes(r.id));
      tr.title = 'Show ' + r.id + ' in focus';
      const a = r.angles && typeof r.angles === 'object' ? r.angles : null;
      const angles = a ? `${Ui.mm(Number(a.pitch), 1)}, ${Ui.mm(Number(a.yaw), 1)}` : '';
      const turned = Number(r.rotation_deg) ? `${Number(r.rotation_deg)}°` : '';
      const cells = [
        [r.order, 'num'], [r.id, 'name'], [Ui.mm(r.position_mm, 3) || 'not placed', typeof r.position_mm === 'number' ? 'num' : 'none'],
        [angles], [r.flip ? 'yes' : ''], [turned], [r.transform || ''], [r.deformation || '', 'method'],
        [r.damaged ? 'damaged' : ''], [r.registration ?? '', 'num'],
      ];
      for (const [text, cls] of cells) tr.append(el('td', cls || '', String(text)));
      tr.addEventListener('click', () => pick(r.id));
      body.append(tr);
    }
    $('sections-note').textContent = note;
    $('sections-note').hidden = !note;
  }

  // --- the Steps view -------------------------------------------------------------------------
  function steps(list, labels, first) {
    list.replaceChildren();
    if (!labels.length) { list.append(el('li', 'none', 'None.')); return; }
    // Newest first, each numbered by its depth.
    labels.map((label, i) => [first + i, label]).reverse().forEach(([n, label]) => {
      const li = el('li');
      li.append(el('span', 'step-number', String(n)), el('span', 'step-label', label));
      list.append(li);
    });
  }
  function renderSteps(snap) {
    const history = snap.history || { undo: [], redo: [] };
    steps($('undo'), history.undo || [], 1);
    steps($('redo'), history.redo || [], (history.undo || []).length + 1);
    $('undo-count').textContent = `(${(history.undo || []).length})`;
    $('redo-count').textContent = `(${(history.redo || []).length})`;
    const notes = $('notes');
    notes.replaceChildren();
    if (snap.notes?.user) notes.append(el('p', 'notes-label', 'Your notes'), el('p', '', snap.notes.user));
    if (snap.notes?.run?.length) {
      notes.append(el('p', 'notes-label', 'Run notes'));
      for (const text of snap.notes.run) notes.append(el('p', '', text));
    }
    if (!notes.children.length) notes.append(el('p', 'none', 'No notes.'));
    $('spend').textContent = spendWords(snap.spend);
  }
  function spendWords(spend) {
    if (!spend) return 'No spend recorded.';
    const t = spend.tokens || {}, parts = [];
    const sum = (...keys) => keys.reduce((a, k) => a + (t[k] || 0), 0);
    const input = sum('input_uncached', 'input_cached') || t.input || 0;
    if (spend.model) parts.push(spend.model);
    if (input) parts.push(`${input.toLocaleString()} input tokens` + (t.input_cached ? ` (${t.input_cached.toLocaleString()} cached)` : ''));
    if (t.output) parts.push(`${t.output.toLocaleString()} output`);
    if (t.reasoning) parts.push(`${t.reasoning.toLocaleString()} reasoning`);
    if (typeof spend.usd_estimate === 'number') parts.push(`about $${spend.usd_estimate.toFixed(2)}`);
    if (typeof spend.quota_used_percent === 'number') parts.push(`${spend.quota_used_percent}% of the usage window`);
    return parts.join(' · ') || 'No spend recorded.';
  }

  // --- the dividers: the log's width and the strip's height, kept per browser --------------
  function columns() {
    const body = $('viewer-body');
    const saved = Number(Ui.store.get('logWidth'));
    if (saved > 0) body.style.setProperty('--log-width', saved + 'px');
    let start = 0;
    Ui.drag($('log-divider'), {
      start: () => { start = $('log').getBoundingClientRect().width; },
      move: dx => {
        const width = Math.round(Math.min(body.clientWidth - 440, Math.max(300, start - dx)));
        body.style.setProperty('--log-width', width + 'px');
      },
      end: () => { Ui.store.set('logWidth', parseInt(body.style.getPropertyValue('--log-width'), 10) || ''); Strip.relayout(); },
    });
  }
  function stripSize() {
    const strip = $('strip'), area = $('focus-area');
    const saved = Number(Ui.store.get('stripHeight'));
    if (saved > 0) strip.style.setProperty('--strip-height', saved + 'px');
    const collapse = on => {
      strip.classList.toggle('collapsed', on);
      $('strip-toggle').setAttribute('aria-expanded', String(!on));
      $('strip-divider').hidden = on;
    };
    collapse(Ui.store.get('stripCollapsed') === '1');
    $('strip-toggle').addEventListener('click', () => {
      const on = !strip.classList.contains('collapsed');
      collapse(on);
      Ui.store.set('stripCollapsed', on ? '1' : '0');
    });
    let start = 0;
    Ui.drag($('strip-divider'), {
      start: () => { start = strip.getBoundingClientRect().height; },
      move: (dx, dy) => {
        const height = Math.round(Math.min(area.clientHeight - 200, Math.max(110, start + dy)));
        strip.style.setProperty('--strip-height', height + 'px');
      },
      end: () => Ui.store.set('stripHeight', parseInt(strip.style.getPropertyValue('--strip-height'), 10) || ''),
    });
  }

  return { init, setJob, setHeader, update, pick, toggleFollow, pageBy, showView, shown };
})();
