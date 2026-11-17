'use strict';
// The sections area: its header (whose choice the sections are and how many, the
// Pictures | Table | Steps switch, Follow the agent and the page buttons), one of three
// views, and under them the brain strip (viewer-strip.js):
//   Pictures  up to four panels (1, 1 x 2 or 2 x 2), each a section under its current
//             registration with the atlas borders, drawn by the server
//             (`render/overlay/<section>`) at the panel's size; the old picture stays,
//             tagged `updating`, until a newer one has loaded;
//   Table     the sections table;
//   Steps     undo and redo, the notes and the spend.
// It also keeps the narration band's progress meters (how many sections are placed,
// aligned and warped).
// Follow: the focus is the sections of the agent's latest followed call (the snapshot's
// `focus`); a new call replaces them and goes back to page 1. A section the person
// clicks (strip, table, a name in the journal) becomes the focus and Follow turns off.
// Review (r): one section at a time, every section in stack order, the arrow keys flip
// through them; dragging on the picture draws a box, and a short comment typed under it is
// kept in the job (`POST jobs/<n>/comments`, the job's review.json), where the agent's
// `review_comments` tool reads it. Every panel shows its section's comments: a box and the
// words, faded once the section has changed since (`revised`: the agent worked on it).
const Viewer = (() => {
  const PAGE_SIZES = { simple: 3, extended: 4 };
  // Simple shows more sections than a page holds as a deck of small pictures.
  const DECK_FROM = 4;
  // In Simple, a new focus waits until the last one has been shown this long, so a burst of
  // quick calls reads as steps rather than a flicker.
  const DWELL_MS = 2400;
  // A deck stays up a little longer: it is many pictures to take in.
  const DECK_DWELL_MS = 4000;
  // The deck keeps at most this many sections' pictures for when they come back.
  const DECK_KEPT = 160;
  // A new focus is shown once its pictures are ready, or after this long at most (the
  // window draws one picture at a time, about half a second each).
  const READY_WAIT_MS = 4000;
  // Simple's pictures are drawn at one size per count of them, so a picture fetched ahead
  // of its focus is the very one the panel asks for.
  const SIMPLE_EDGE = { 1: 1280, 2: 1024, 3: 768 };
  const READY_KEPT = 64;
  // How a section got its in-plane transform, in a panel's words.
  const HOW = { elastix: 'aligned automatically', interactive: 'aligned by hand' };
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
  let tasks = null;            // the job's tasks, from the header
  let tasksSeen = '';
  let simple = true;           // the Simple view: two sections to a page, no other view
  let review = false;          // one section at a time, every section, boxes and comments
  let comments = [];           // the job's review comments, from the snapshot
  let commentsSeen = '';
  let status = '';             // the run's status word (Submitted: the deck of every section)
  let deckTiles = new Map();   // the deck's pictures, by section
  let deckIds = [];
  let shownAt = 0;             // when the focus shown was applied (Simple's dwell)
  let dwellTimer = null;
  let pending = null;          // a focus waiting for the dwell
  let decked = false;          // the deck is what is shown
  let shownCount = 0;          // how many panels are shown
  let applying = 0;            // the latest focus being made ready
  const ready = new Map();     // url -> a decoded picture fetched ahead of its focus
  const pageSize = () => (review ? 1 : simple ? PAGE_SIZES.simple : PAGE_SIZES.extended);

  // --- the header, the switch, Follow and paging -------------------------------------------
  function init() {
    for (const button of document.querySelectorAll('[data-view]')) {
      button.addEventListener('click', () => showView(button.dataset.view));
    }
    $('follow-agent').addEventListener('click', toggleFollow);
    $('review').addEventListener('click', toggleReview);
    $('page-previous').addEventListener('click', () => pageBy(-1));
    $('page-next').addEventListener('click', () => pageBy(1));
    panels = [1, 2, 3, 4].map(makePanel);
    new ResizeObserver(() => {
      clearTimeout(resizeTimer);
      for (const p of panels) if (p.id) { fitTitle(p); drawMarks(p); }
      resizeTimer = setTimeout(() => { for (const p of panels) if (p.id) request(p); }, 250);
    }).observe($('view-focus'));
    Strip.init({ pick });
    Axis.init({ pick });
    new ResizeObserver(() => { if (deckIds.length) layoutDeck(); }).observe($('view-deck').parentElement);
    columns();
    stripSize();
  }
  function setJob(index) {
    job = index;
    rows = []; byId = new Map(); loaded = false;
    agent = null; agentSeen = ''; follow = true; manual = []; page = 0;
    pending = null; shownAt = 0; clearTimeout(dwellTimer); dwellTimer = null;
    tasks = null; tasksSeen = '';
    review = false; comments = []; commentsSeen = '';
    $('review').setAttribute('aria-pressed', 'false');
    $('view-focus').classList.remove('reviewing');
    $('review-count').hidden = true;
    progress();
    for (const p of panels) clearPanel(p);
    Strip.setJob(index);
    Axis.setJob();
    status = '';
    for (const t of deckTiles.values()) t.release?.();
    deckTiles = new Map(); deckIds = [];
    $('view-deck').replaceChildren();
    renderSections();
    renderSteps({});
    refresh();
  }
  function setHeader(header) {
    Strip.setPlane(header?.plane);
    const seen = JSON.stringify(header?.tasks ?? null);
    if (seen !== tasksSeen) { tasksSeen = seen; tasks = Array.isArray(header?.tasks) ? header.tasks : null; progress(); }
  }
  // How far the registration is: of the job's sections, how many have a position, an
  // in-plane transform and a deformation. A task the job has switched off has no meter.
  function progress() {
    const box = $('progress');
    box.hidden = !loaded || !rows.length;
    if (box.hidden) return;
    const counts = {
      placed: rows.filter(r => typeof r.position_mm === 'number').length,
      aligned: rows.filter(r => r.transform).length,
      warped: rows.filter(r => r.deformation).length,
    };
    const task = { placed: 'position', aligned: 'transform', warped: 'nonlinear' };
    for (const meter of box.querySelectorAll('.meter')) {
      const name = meter.dataset.meter, n = counts[name];
      meter.hidden = Boolean(tasks) && !tasks.includes(task[name]) && !n;
      meter.classList.toggle('full', n === rows.length);
      meter.classList.toggle('none', n === 0);
      meter.querySelector('.meter-count').textContent = `${n} of ${rows.length}`;
      meter.querySelector('.meter-fill').style.width = `${(100 * n) / rows.length}%`;
    }
  }
  // One snapshot: its tables (when the state moved; else null) and the agent's focus, applied
  // together so the panels ask only for what the snapshot puts in focus.
  function update(tables, focus) {
    let changed = false;
    if (tables) {
      loaded = true;
      rows = Array.isArray(tables.sections) ? tables.sections : [];
      byId = new Map(rows.map(r => [r.id, r]));
      Strip.setRows(rows);
      Axis.setRows(rows);
      renderSections(tables.sections_note || '');
      renderSteps(tables);
      progress();
      changed = true;
    }
    const seen = JSON.stringify(focus ? [focus.call ?? null, focus.sections || []] : null);
    if (seen !== agentSeen) {
      agentSeen = seen;
      if (simple && follow && !review) { pending = focus; dwell(); }
      else {
        agent = focus;
        if (follow) { page = 0; changed = true; }
      }
    }
    if (changed) refresh();
  }
  function dwell() {
    const wait = shownAt + (decked ? DECK_DWELL_MS : DWELL_MS) - Date.now();
    if (wait > 0) { if (!dwellTimer) dwellTimer = setTimeout(() => { dwellTimer = null; dwell(); }, wait); return; }
    // The pictures of the new focus are fetched and decoded first, so the change is one
    // step, never a flash of empty panels; a newer focus meanwhile replaces this one.
    const focus = pending, token = ++applying;
    const ids = focusIds(focus);
    const apply = () => {
      if (token !== applying) return;
      fadeStage();
      agent = focus;
      shownAt = Date.now();
      page = 0;
      refresh();
    };
    if (!ids.length) { apply(); return; }
    const deck = ids.length >= DECK_FROM;
    const edge = deck ? deckGeometry(ids.length).edge : simpleEdge(ids.length);
    Promise.race([
      Promise.all(ids.map((id, i) => prefetch(renderUrl(id, edge), i))),
      new Promise(done => setTimeout(done, READY_WAIT_MS)),
    ]).then(apply);
  }
  // The stage shown leaves as one picture: a copy of it lies over the new one and fades,
  // so the stage is never empty between two focuses.
  function fadeStage() {
    const views = $('view-deck').parentElement;
    const shown = decked ? $('view-deck') : $('view-focus');
    if (!simple || review || shown.hidden || !shown.querySelector('img')) return;
    views.querySelector('.stage-ghost')?.remove();
    const ghost = shown.cloneNode(true);
    ghost.removeAttribute('id');
    ghost.classList.add('stage-ghost');
    ghost.classList.remove('stage-in');
    ghost.setAttribute('aria-hidden', 'true');
    for (const node of ghost.querySelectorAll('[id]')) node.removeAttribute('id');
    views.append(ghost);
    requestAnimationFrame(() => requestAnimationFrame(() => ghost.classList.add('gone')));
    setTimeout(() => ghost.remove(), 320);
  }
  function focusIds(focus) {
    if (simple && status === 'Submitted') return rows.map(r => r.id);
    const ids = (focus?.sections || []).filter(id => byId.has(id));
    return ids.length ? [...new Set(ids)] : rows.map(r => r.id);
  }
  function simpleEdge(count) {
    const base = SIMPLE_EDGE[Math.min(3, Math.max(1, count))];
    return Math.min(2048, Math.ceil((base * Math.min(2, window.devicePixelRatio || 1)) / 128) * 128);
  }
  function prefetch(url, rank) {
    if (ready.has(url)) return Promise.resolve();
    return new Promise(done => {
      const img = new Image();
      img.decoding = 'async';
      img.draggable = false;
      let release = null;
      const end = () => { release?.(); done(); };
      img.onload = () => img.decode().catch(() => {}).then(() => {
        ready.set(url, img);
        while (ready.size > READY_KEPT) ready.delete(ready.keys().next().value);
        end();
      });
      img.onerror = end;
      release = Ui.loads.add(() => rank / 10, () => { img.src = url; });
    });
  }
  function agentTargets() {
    // A finished run, in Simple: every section, dealt out as a deck.
    if (simple && status === 'Submitted') return rows.map(r => r.id);
    const ids = (agent?.sections || []).filter(id => byId.has(id));
    return ids.length ? [...new Set(ids)] : rows.map(r => r.id);
  }
  function targets() {
    if (review) return rows.map(r => r.id);
    return follow ? agentTargets() : manual.filter(id => byId.has(id));
  }
  function pick(id) {
    if (!byId.has(id)) return;
    if (review) {
      page = Math.max(0, rows.findIndex(r => r.id === id));
      showView('focus');
      return;
    }
    manual = [id];
    follow = false;
    page = 0;
    showView('focus');
  }
  // Review: from the first section shown, one at a time through every section.
  function toggleReview() {
    const shown = panels.find(p => p.id)?.id;
    review = !review;
    for (const p of panels) cancelDraft(p);
    if (review) {
      follow = false;
      page = Math.max(0, rows.findIndex(r => r.id === shown));
      for (const p of panels) { p.view = { s: 1, x: 0, y: 0 }; transform(p, p.img); }
    } else {
      manual = shown ? [shown] : manual;
      page = 0;
    }
    $('review').setAttribute('aria-pressed', String(review));
    $('view-focus').classList.toggle('reviewing', review);
    showView('focus');
    for (const p of panels) drawMarks(p);
  }
  function reviewing() { return review; }
  function setStatus(word) {
    if (word === status) return;
    status = word;
    if (follow) refresh();
  }
  // The snapshot's comments: every panel showing a commented section draws them again.
  function setComments(list) {
    const seen = JSON.stringify(list || []);
    if (seen === commentsSeen) return;
    commentsSeen = seen;
    comments = Array.isArray(list) ? list : [];
    const open = comments.filter(c => !c.revised).length;
    $('review-count').textContent = open ? String(open) : '';
    $('review-count').hidden = !open;
    for (const p of panels) drawMarks(p);
  }
  // Simple shows the pictures only, and follows the agent.
  function setSimple(on) {
    simple = on;
    if (pending) agent = pending;
    if (on && !review) { follow = true; page = 0; if (mode !== 'focus') { showView('focus'); return; } }
    refresh();
  }
  function toggleFollow() {
    if (follow) manual = targets();
    follow = !follow;
    page = 0;
    refresh();
  }
  function pageBy(delta) {
    const count = Math.max(1, Math.ceil(targets().length / pageSize()));
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
    const size = pageSize();
    const pages = Math.max(1, Math.ceil(all.length / size));
    page = Math.min(page, pages - 1);
    const onPage = all.slice(page * size, page * size + size);
    const deck = simple && !review && mode === 'focus' && all.length >= DECK_FROM;
    const done = simple && follow && status === 'Submitted';
    $('focus-name').textContent = review ? 'Reviewing' : done ? 'Registered' : follow ? 'The agent is working on' : 'You are looking at';
    if (review) {
      $('focus-count').textContent = !loaded || !rows.length ? '' : `section ${page + 1} of ${rows.length}`;
      $('focus-hint').textContent = rows.length ? '← → to flip · drag a box to comment' : '';
    } else {
      $('focus-count').textContent = !loaded || !rows.length ? ''
        : (all.length === rows.length ? `all ${rows.length} sections` : `${all.length} of ${rows.length} sections`)
          + (pages > 1 && !deck ? `, page ${page + 1} of ${pages}` : '');
      $('focus-hint').textContent = '';
    }
    $('follow-agent').setAttribute('aria-pressed', String(follow && !review));
    $('tab-viewer').classList.toggle('following', follow && !review);
    $('pager').hidden = pages < 2 || deck;
    Strip.setTargets(all, onPage);
    Axis.setTargets(all, deck ? [] : onPage);
    $('view-deck').hidden = !deck;
    $('view-focus').classList.toggle('decked', deck);
    // Between the pictures and the deck the stage fades over rather than cutting.
    if (deck !== decked) {
      decked = deck;
      const shown = deck ? $('view-deck') : $('view-focus');
      shown.classList.remove('stage-in');
      void shown.offsetWidth;
      shown.classList.add('stage-in');
    }
    for (const tr of document.querySelectorAll('#sections tbody tr')) tr.classList.toggle('target', all.includes(tr.dataset.id));
    const empty = $('focus-empty');
    empty.hidden = true;
    shareProblem();
    if (mode !== 'focus') return;
    if (!loaded || !rows.length || !onPage.length) {
      empty.textContent = !loaded ? 'Loading the job…' : !rows.length ? 'This job has no sections yet.' : 'Nothing in focus.';
      empty.hidden = false;
    }
    if (deck) {
      // The panels wait, hidden, with their pictures, for the next few sections.
      for (const p of panels) if (p.pending) { drop(p.pending); p.pending = null; p.url = ''; }
      showDeck(all);
      return;
    }
    if (deckIds.length) showDeck([]);
    const grid = $('view-focus');
    shownCount = onPage.length;
    grid.className = 'panels n' + Math.max(1, onPage.length);
    // 1 fills the area, 2 sit side by side, 3 as one tall panel beside two, 4 in a 2 x 2 grid.
    const children = panels.slice(0, onPage.length).map(p => p.root);
    if (grid.children.length !== children.length || children.some((c, i) => grid.children[i] !== c)) grid.replaceChildren(...children);
    onPage.forEach((id, i) => showPanel(panels[i], id));
    for (const p of panels.slice(onPage.length)) clearPanel(p);
  }
  // Called when the viewer tab is shown again: sizes were zero while it was hidden.
  function shown() {
    Strip.relayout();
    for (const p of panels) if (p.id) request(p);
  }

  // --- focus panels -------------------------------------------------------------------------
  function makePanel(slot) {
    const root = el('article', 'panel slot-' + slot);
    const title = el('header', 'panel-title');
    const number = el('span', 'panel-slot', String(slot));
    const name = el('span', 'panel-name'), where = el('span', 'panel-where'), angles = el('span', 'panel-angles'), tags = el('span', 'panel-tags');
    const open = el('button', 'panel-open', 'Enlarge');
    open.title = 'Show this picture large';
    title.append(number, name, where, angles, tags, open);
    const box = el('div', 'panel-picture');
    const updating = el('span', 'panel-updating', 'updating');
    const message = el('p', 'panel-message');
    updating.hidden = message.hidden = true;
    const marks = el('div', 'panel-marks');
    box.append(marks, updating, message);
    root.append(title, box);
    const p = { slot, root, name, where, angles, tags, open, box, marks, updating, message, id: null, key: null, edge: 0, url: '', img: null, pending: null, stale: false, problem: '', view: { s: 1, x: 0, y: 0 }, draft: null };
    open.addEventListener('click', () => {
      if (!p.img || p.stale) return;
      const row = byId.get(p.id);
      const words = [p.angles.textContent, row?.transform, row?.deformation, row?.damaged ? 'damaged' : ''].filter(Boolean);
      AgentLog.openRender(p.img.src, edge => renderUrl(p.id, edge), p.name.textContent,
        'The current registration' + (words.length ? ' · ' + words.join(' · ') : '') + ' · drawn for this window, not a picture the agent saw');
    });
    zoomable(p);
    drawable(p);
    return p;
  }
  // A load no longer wanted is stopped, freeing its place in the queue and its connection.
  function drop(img) {
    img.onload = img.onerror = null;
    img.release?.();
    img.removeAttribute('src');
  }
  function clearPanel(p) {
    if (p.pending) { drop(p.pending); p.pending = null; }
    p.img?.remove();
    p.img = null; p.id = null; p.key = null; p.edge = 0; p.url = ''; p.stale = false;
    p.view = { s: 1, x: 0, y: 0 };
    cancelDraft(p);
    drawMarks(p);
    p.updating.hidden = p.message.hidden = true;
    p.problem = '';
    shareProblem();
  }
  function showPanel(p, id) {
    const row = byId.get(id);
    // Another section: the picture shown stays, dimmed, until the new section's has loaded.
    if (p.id !== id) {
      if (p.pending) { drop(p.pending); p.pending = null; }
      p.id = id; p.key = null; p.edge = 0; p.url = ''; p.problem = '';
      p.view = { s: 1, x: 0, y: 0 };
      p.message.hidden = true;
      cancelDraft(p);
      if (p.img) { p.stale = true; p.img.classList.add('stale'); transform(p, p.img); }
      // The name waits with the picture, so a name never sits under another section.
      p.root.classList.add('waiting');
      drawMarks(p);
    }
    p.name.textContent = Ui.short(id);
    p.name.title = id;
    p.where.textContent = typeof row?.position_mm === 'number' ? `${Ui.mm(row.position_mm)} mm` : '';
    const a = row?.angles && typeof row.angles === 'object' ? row.angles : {};
    const pitch = Number(a.pitch) || 0, yaw = Number(a.yaw) || 0;
    p.angles.textContent = [pitch ? `pitch ${pitch.toFixed(1)}°` : '', yaw ? `yaw ${yaw.toFixed(1)}°` : ''].filter(Boolean).join(', ');
    // Each tag keeps a rank: when the strip is short of room the lowest go first.
    const tags = [];
    const tag = (cls, text, rank) => { const t = el('span', cls, text); t.dataset.rank = rank; tags.push(t); return t; };
    if (row?.transform) tag('tag', HOW[row.transform] || row.transform, 1).title = 'Transform: ' + row.transform;
    if (row?.deformation) tag('tag', 'warped', 2).title = 'Deformation: ' + row.deformation;
    if (row?.damaged) tag('tag damaged', 'damaged', 4);
    if (row && typeof row.position_mm !== 'number') tag('tag', 'not placed', 3);
    p.tags.replaceChildren(...tags);
    p.open.disabled = !p.img || p.stale;
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
    const style = simple ? '&style=calm' : '';
    return `jobs/${job}/render/overlay/${encodeURIComponent(id)}?edge=${edge}&key=${encodeURIComponent(key)}${style}`;
  }
  function request(p) {
    if (!p.id || job === null) return;
    const edge = simple && !review ? simpleEdge(shownCount) : neededEdge(p);
    if (!edge) return;
    const key = String(byId.get(p.id)?.render_key ?? '');
    // The same state at a size already drawn large enough needs nothing new.
    if (p.url && key === p.key && (simple && !review ? edge === p.edge : edge <= p.edge) && simple === p.calm) return;
    p.calm = simple;
    p.key = key;
    p.edge = edge;
    const url = renderUrl(p.id, edge);
    if (url === p.url) return;
    p.url = url;
    p.tries = 0;
    load(p, url);
  }
  // A panel's picture loads through the shared queue, ahead of the strip's (in panel order).
  // The old picture stays, tagged `updating`, until the new one has loaded. A picture that
  // does not come says why: the server's refusal (409), or, while the server is still
  // drawing it (503), a short wait and another try.
  function load(p, url) {
    if (p.pending) drop(p.pending);
    // A picture fetched ahead of its focus is shown at once.
    const early = ready.get(url);
    if (early) ready.delete(url);
    const img = early || new Image();
    img.alt = 'The section ' + p.id + ' under its current registration, with the atlas borders';
    img.decoding = 'async';
    img.draggable = false;
    p.pending = img;
    p.updating.hidden = !p.img || p.stale;
    if (!p.img) { p.message.textContent = 'Drawing the section…'; p.message.hidden = false; }
    p.message.classList.toggle('drawing', !p.img);
    img.onload = async () => {
      img.release();
      if (p.pending !== img) return;
      // Shown once decoded, so its entrance is seen rather than spent before it paints.
      await img.decode().catch(() => {});
      if (p.pending !== img) return;
      p.pending = null;
      p.updating.hidden = p.message.hidden = true;
      p.root.classList.remove('waiting');
      if (p.problem) { p.problem = ''; shareProblem(); }
      // The same section drawn again (the agent moved it) morphs from the old picture; a
      // new section's picture arrives while the old one leaves.
      const old = p.img;
      const morph = Boolean(old) && !p.stale;
      img.className = morph ? 'morphing' : 'entering';
      img.dataset.how = morph ? 'morph' : 'enter';
      transform(p, img);
      p.stale = false;
      p.img = img;
      p.box.append(img);
      p.box.append(p.marks, p.updating, p.message);
      p.open.disabled = false;
      drawMarks(p);
      requestAnimationFrame(() => requestAnimationFrame(() => {
        img.classList.remove('entering', 'morphing');
        if (!morph) old?.classList.add('leaving');
      }));
      if (old) setTimeout(() => old.remove(), morph ? 500 : 260);
    };
    img.onerror = async () => {
      if (p.pending !== img) { img.release(); return; }
      const answer = await Ui.loads.explain(url);
      img.release();
      if (p.pending !== img || p.url !== url) return;
      p.pending = null;
      p.tries += 1;
      if ((answer.ok || answer.status === 503) && p.tries < 6) {
        // Drawn meanwhile, or still being drawn: try again (a 503 after a pause).
        if (!p.img) p.message.textContent = 'Still drawing the section…';
        setTimeout(() => { if (p.url === url && !p.pending) load(p, url); }, answer.ok ? 0 : 3000);
        return;
      }
      p.updating.hidden = true;
      p.root.classList.remove('waiting');
      p.message.classList.remove('drawing');
      p.img?.remove();
      p.img = null; p.stale = false;
      p.open.disabled = true;
      p.problem = answer.error
        || (answer.status === 404 ? 'No picture of this section is available from this LangSlice.'
          : answer.status === 0 ? 'LangSlice is not reachable.' : 'The picture could not be drawn.');
      p.message.hidden = false;
      shareProblem();
    };
    if (early) { img.release = () => {}; img.onload(); return; }
    img.release = Ui.loads.add(() => p.slot / 10, () => { img.src = url; });
  }
  // The same reason on every panel shown (a job whose images moved, say) is said once, over
  // the panels; each panel then says only that it was not drawn.
  function shareProblem() {
    const shown = mode === 'focus' ? panels.filter(p => p.id) : [];
    const said = shown.filter(p => p.problem);
    const shared = shown.length > 1 && said.length === shown.length && said.every(p => p.problem === said[0].problem);
    $('focus-problem').textContent = shared ? said[0].problem : '';
    $('focus-problem').hidden = !shared;
    for (const p of said) p.message.textContent = shared ? 'Not drawn: the reason is above.' : p.problem;
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
      if (event.button !== 0 || !p.img || review) return;
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
    box.addEventListener('dblclick', event => {
      if (event.target.closest('.composer')) return;
      p.view = { s: 1, x: 0, y: 0 };
      transform(p, p.img);
    });
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
    if (img === p.img) p.marks.style.transform = img.style.transform;
  }

  // --- the deck: every section in focus as a small picture (Simple) ---------------------------
  // More sections than a page holds (a call on many, or the finished run) are dealt out as a
  // grid sized to fill the area; each picture is drawn small and again when its section
  // changes, the old one kept until the new one has loaded. A click shows the section.
  function showDeck(ids) {
    const deck = $('view-deck');
    deckIds = ids;
    // A section no longer in the deck keeps its picture for later; its load waiting stops.
    for (const [id, t] of deckTiles) {
      if (ids.includes(id)) continue;
      if (t.release) { t.release(); t.release = null; t.url = t.loaded; }
    }
    for (const id of [...deckTiles.keys()].slice(0, Math.max(0, deckTiles.size - DECK_KEPT))) {
      if (!ids.includes(id)) deckTiles.delete(id);
    }
    const nodes = ids.map(deckTile);
    if (deck.children.length !== nodes.length || nodes.some((n, i) => deck.children[i] !== n)) {
      // A deck dealt anew: its pictures arrive in reading order, quickly at first, the
      // later ones together, all within half a second.
      nodes.forEach((n, i) => n.style.setProperty('--deal', `${Math.round(240 * Math.sqrt(i / Math.max(1, nodes.length)))}ms`));
      deck.replaceChildren(...nodes);
      deck.classList.remove('dealing');
      void deck.offsetWidth;
      deck.classList.add('dealing');
    }
    if (ids.length) layoutDeck();
  }
  // The columns that make the pictures largest, at a brain's width to height.
  // The deck's size for n pictures: the columns that make them largest in the stage (the
  // views' box less the deck's margins), and the edge they are drawn at.
  function deckGeometry(n) {
    const box = $('view-deck').parentElement.getBoundingClientRect();
    const side = Math.max(32, (box.width - 1240) / 2);
    const width = box.width - 2 * side, height = box.height - 20;
    const gap = 10, aspect = 1.35;
    let best = { cols: 1, width: 0 };
    for (let cols = 1; cols <= n; cols++) {
      const lines = Math.ceil(n / cols);
      const w = Math.min((width - gap * (cols - 1)) / cols, ((height - gap * (lines - 1)) / lines) * aspect);
      if (w > best.width) best = { cols, width: w };
    }
    const tile = Math.max(40, Math.floor(best.width));
    const edge = Math.min(512, Math.max(128, Math.ceil((tile * Math.min(2, window.devicePixelRatio || 1)) / 128) * 128));
    return { cols: best.cols, width: tile, edge };
  }
  function layoutDeck() {
    const deck = $('view-deck'), n = deckIds.length;
    if (!n) return;
    const g = deckGeometry(n);
    deck.style.setProperty('--deck-width', `${g.width}px`);
    deck.style.setProperty('--deck-cols', String(g.cols));
    // Pictures large enough for a name under them carry it.
    deck.classList.toggle('named', g.width >= 240);
    deckIds.forEach((id, i) => deckLoad(deckTiles.get(id), g.edge, i));
  }
  function deckTile(id) {
    let t = deckTiles.get(id);
    if (t) return t.root;
    const root = el('button', 'deck-tile');
    root.type = 'button';
    root.title = Ui.short(id);
    root.append(el('span', 'deck-name', Ui.short(id)));
    root.addEventListener('click', () => pick(id));
    t = { id, root, img: null, url: '', loaded: '', release: null };
    deckTiles.set(id, t);
    return root;
  }
  function deckLoad(t, edge, rank) {
    if (!t) return;
    const url = renderUrl(t.id, edge);
    if (url === t.url) return;
    t.url = url;
    t.release?.();
    const early = ready.get(url);
    if (early) ready.delete(url);
    const img = early || new Image();
    img.alt = 'The section ' + t.id;
    img.decoding = 'async';
    img.draggable = false;
    img.onload = async () => {
      t.release?.(); t.release = null;
      if (t.url !== url) return;
      await img.decode().catch(() => {});
      if (t.url !== url) return;
      t.loaded = url;
      const old = t.img;
      img.className = old ? 'morphing' : 'entering';
      img.dataset.how = old ? 'morph' : 'enter';
      t.img = img;
      t.root.append(img);
      requestAnimationFrame(() => requestAnimationFrame(() => {
        img.classList.remove('entering', 'morphing');
      }));
      if (old) setTimeout(() => old.remove(), 500);
    };
    img.onerror = () => { t.release?.(); t.release = null; if (t.url === url) t.url = t.loaded; };
    if (early) { t.release = () => {}; img.onload(); return; }
    t.release = Ui.loads.add(() => 0.5 + rank / 1000, () => { img.src = url; });
  }

  // --- review boxes ---------------------------------------------------------------------------
  // Where the picture's pixels lie in its panel (the image is contained, centred): the
  // rectangle a box's fractions are of, in the panel's unzoomed pixels.
  function contentRect(p) {
    const r = p.box.getBoundingClientRect();
    const w = p.img?.naturalWidth || 0, h = p.img?.naturalHeight || 0;
    if (!w || !h || r.width < 8 || r.height < 8) return null;
    const scale = Math.min(r.width / w, r.height / h);
    return { left: (r.width - w * scale) / 2, top: (r.height - h * scale) / 2, width: w * scale, height: h * scale };
  }
  // A pointer's place as fractions of the picture, through the panel's zoom.
  function fractionAt(p, event) {
    const c = contentRect(p);
    if (!c) return null;
    const r = p.box.getBoundingClientRect(), v = p.view;
    const x = (event.clientX - r.left - v.x) / v.s, y = (event.clientY - r.top - v.y) / v.s;
    const clamp = n => Math.min(1, Math.max(0, n));
    return { x: clamp((x - c.left) / c.width), y: clamp((y - c.top) / c.height) };
  }
  function placeBox(node, c, box) {
    const [x0, y0, x1, y1] = box;
    node.style.left = `${c.left + x0 * c.width}px`;
    node.style.top = `${c.top + y0 * c.height}px`;
    node.style.width = `${(x1 - x0) * c.width}px`;
    node.style.height = `${(y1 - y0) * c.height}px`;
  }
  // The section's comments over its picture, and the box being drawn.
  function drawMarks(p) {
    const c = p.id && p.img && !p.stale ? contentRect(p) : null;
    const nodes = [];
    if (c) {
      for (const comment of comments.filter(x => x.section === p.id)) {
        const mark = el('div', 'mark' + (comment.revised ? ' revised' : ''));
        placeBox(mark, c, comment.box);
        const chip = el('div', 'mark-chip');
        chip.append(el('span', 'mark-text', comment.text));
        if (comment.revised) chip.append(el('span', 'mark-state', 'revised'));
        if (review) {
          const remove = el('button', 'mark-remove', '×');
          remove.type = 'button';
          remove.title = 'Remove this comment';
          remove.setAttribute('aria-label', 'Remove the comment ' + comment.text);
          remove.addEventListener('pointerdown', event => event.stopPropagation());
          remove.addEventListener('click', () => send({ remove: comment.id }));
          chip.append(remove);
        }
        mark.append(chip);
        nodes.push(mark);
      }
      if (p.draft?.box) {
        const draft = el('div', 'mark draft');
        placeBox(draft, c, p.draft.box);
        nodes.push(draft);
      }
    }
    p.marks.replaceChildren(...nodes);
    p.marks.style.transform = p.img?.style.transform || '';
  }
  // Review: a drag draws a box; letting go asks for the comment under it.
  function drawable(p) {
    const box = p.box;
    box.addEventListener('pointerdown', event => {
      if (!review || event.button !== 0 || !p.img || p.stale) return;
      if (event.target.closest('.composer, .mark-remove')) return;
      const start = fractionAt(p, event);
      if (!start) return;
      cancelDraft(p);
      p.draft = { start, box: null, drawing: true };
      box.setPointerCapture(event.pointerId);
      event.preventDefault();
    });
    box.addEventListener('pointermove', event => {
      if (!p.draft?.drawing) return;
      const at = fractionAt(p, event), s = p.draft.start;
      if (!at) return;
      p.draft.box = [Math.min(s.x, at.x), Math.min(s.y, at.y), Math.max(s.x, at.x), Math.max(s.y, at.y)];
      drawMarks(p);
    });
    const done = () => {
      if (!p.draft?.drawing) return;
      p.draft.drawing = false;
      const b = p.draft.box;
      if (!b || b[2] - b[0] < 0.01 || b[3] - b[1] < 0.01) { cancelDraft(p); return; }
      compose(p);
    };
    box.addEventListener('pointerup', done);
    box.addEventListener('pointercancel', () => cancelDraft(p));
  }
  function cancelDraft(p) {
    if (!p.draft) return;
    p.draft = null;
    p.box.querySelector('.composer')?.remove();
    drawMarks(p);
  }
  // The comment's field, just under the box (above it near the bottom), outside the zoom.
  function compose(p) {
    const c = contentRect(p);
    if (!c) { cancelDraft(p); return; }
    const r = p.box.getBoundingClientRect(), v = p.view, [x0, y0, , y1] = p.draft.box;
    const form = el('form', 'composer');
    const input = el('input', 'composer-input');
    input.type = 'text';
    input.maxLength = 500;
    input.placeholder = 'What is off here?';
    input.setAttribute('aria-label', 'Your comment on the box');
    const note = el('span', 'composer-note', 'Enter to keep · Esc to cancel');
    form.append(input, note);
    const left = Math.min(Math.max(8, v.x + (c.left + x0 * c.width) * v.s), Math.max(8, r.width - 340));
    let top = v.y + (c.top + y1 * c.height) * v.s + 8;
    if (top > r.height - 72) top = Math.max(8, v.y + (c.top + y0 * c.height) * v.s - 72);
    form.style.left = `${left}px`;
    form.style.top = `${top}px`;
    form.addEventListener('submit', async event => {
      event.preventDefault();
      const text = input.value.trim();
      if (!text || !p.draft) return;
      input.disabled = true;
      const ok = await send({ section: p.id, box: p.draft.box, text }, note);
      if (ok) cancelDraft(p); else input.disabled = false;
    });
    input.addEventListener('keydown', event => {
      event.stopPropagation();
      if (event.key === 'Escape') { event.preventDefault(); cancelDraft(p); }
    });
    p.box.append(form);
    input.focus();
  }
  // Keep or remove a comment; the server answers with the comments as they now stand.
  async function send(body, note = null) {
    try {
      const r = await fetch(`jobs/${job}/comments`, {
        method: 'POST', headers: { 'Content-Type': 'application/json' }, body: JSON.stringify(body), cache: 'no-store',
      });
      const answer = await r.json().catch(() => ({}));
      if (!r.ok) { if (note) note.textContent = answer.error || 'The comment was not kept.'; return false; }
      setComments(answer.comments);
      return true;
    } catch {
      if (note) note.textContent = 'LangSlice is not reachable.';
      return false;
    }
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
        const height = Math.round(Math.min(area.clientHeight - 200, Math.max(110, start - dy)));
        strip.style.setProperty('--strip-height', height + 'px');
      },
      end: () => Ui.store.set('stripHeight', parseInt(strip.style.getPropertyValue('--strip-height'), 10) || ''),
    });
  }

  return { init, setJob, setHeader, update, pick, toggleFollow, toggleReview, reviewing, setComments, pageBy, showView, shown, setSimple, setStatus };
})();
