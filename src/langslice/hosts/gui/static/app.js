'use strict';
// The LangSlice window: tabs, the job list and the agent viewer, fed by GET
// polling of the local server (snapshot every 300 ms, jobs every 2 s).
const $ = id => document.getElementById(id);
const el = (tag, cls, text) => { const e = document.createElement(tag); if (cls) e.className = cls; if (text !== undefined) e.textContent = text; return e; };

let jobs = [], current = null, chosenByUser = false;
let view = null;  // the viewer's state for the current job

function freshView(index) {
  return { index, gen: 0, cursor: 0, sv: -1, pictures: [], byNumber: new Map(), byPath: new Map(),
           rows: new Map(), claims: new Map(), groups: new Map(), shown: -1, follow: true,
           filter: '', sections: new Set() };
}

// --- tabs -------------------------------------------------------------------
for (const button of document.querySelectorAll('[data-tab]')) {
  button.onclick = () => {
    for (const other of document.querySelectorAll('[data-tab]')) other.classList.toggle('active', other === button);
    for (const tab of document.querySelectorAll('.tab')) tab.hidden = tab.id !== 'tab-' + button.dataset.tab;
  };
}

// --- jobs -------------------------------------------------------------------
function ago(seconds) {
  if (!seconds) return 'no files yet';
  const s = Date.now() / 1000 - seconds;
  if (s < 60) return 'just now';
  if (s < 3600) return Math.round(s / 60) + ' min ago';
  if (s < 86400) return Math.round(s / 3600) + ' h ago';
  return Math.round(s / 86400) + ' d ago';
}
function renderJobs() {
  const list = $('job-list'); list.replaceChildren();
  if (!jobs.length) { list.append(el('li', 'muted', 'No jobs yet.')); return; }
  for (const job of jobs) {
    const li = el('li'), b = el('button');
    b.classList.toggle('active', job.index === current);
    b.title = job.folder;
    b.append(el('span', 'job-name', job.name), el('span', 'job-meta', [job.status, job.atlas, ago(job.changed)].filter(Boolean).join(' · ')));
    b.onclick = () => { chosenByUser = true; select(job.index); };
    li.append(b); list.append(li);
  }
}
async function pollJobs() {
  try {
    const r = await fetch('jobs', { cache: 'no-store' });
    if (r.ok) {
      jobs = (await r.json()).jobs;
      if (!chosenByUser && jobs.length) {
        const newest = jobs.reduce((a, b) => (b.changed > a.changed ? b : a));
        if (current === null || (newest.index !== current && newest.changed > (jobs.find(j => j.index === current)?.changed || 0) + 5)) select(newest.index);
      }
      renderJobs();
    }
  } catch { /* the snapshot poll reports the connection */ }
  setTimeout(pollJobs, 2000);
}
function select(index) {
  if (current === index && view) return;
  current = index; view = freshView(index); followActivity = true;
  $('activity').replaceChildren(el('p', 'muted', 'Waiting for the run.'));
  $('section-filter').replaceChildren(new Option('All sections', ''));
  showPicture(); renderJobs();
}

// --- activity -----------------------------------------------------------------
// The activity list follows the newest entry until the user scrolls up.
let followActivity = true;
const activityBox = document.querySelector('.activity');
activityBox.addEventListener('scroll', () => {
  followActivity = activityBox.scrollHeight - activityBox.scrollTop - activityBox.clientHeight < 60;
});
function stick(fn) {
  const follow = followActivity;
  fn();
  if (follow) activityBox.scrollTop = activityBox.scrollHeight;
}
function words(ids) { return (ids || []).length > 4 ? ids.slice(0, 4).join(', ') + ` and ${ids.length - 4} more` : (ids || []).join(', '); }
function thumb(picture) {
  const b = el('button'); b.title = picture.caption || picture.label;
  const img = el('img'); img.src = `jobs/${view.index}/picture/${picture.number}`; img.alt = picture.label; img.loading = 'lazy';
  b.append(img); b.dataset.number = picture.number;
  b.onclick = () => { view.follow = false; view.filter = ''; $('section-filter').value = ''; showPicture(picture.number); };
  return b;
}
function toolRow(item) {
  let row = view.rows.get(item.key);
  if (row) return row;
  const root = el('div', 'item tool running');
  const head = el('div', 'head');
  const what = words(item.sections) || (item.args && typeof item.args === 'object' && typeof item.args.mode === 'string' ? item.args.mode : '');
  const name = el('span', 'name', item.label), targets = el('span', 'targets', what), state = el('span', 'state', 'running');
  head.append(name, targets, state);
  const thumbs = el('div', 'thumbs');
  const raw = el('details', 'raw'); raw.append(el('summary', '', 'Details'), el('pre'));
  root.append(head, thumbs, raw);
  row = { root, state, thumbs, raw: raw.querySelector('pre'), data: {} };
  view.rows.set(item.key, row);
  $('activity').append(root);
  return row;
}
function addThumb(row, picture) {
  if (row.thumbs.querySelector(`[data-number="${picture.number}"]`)) return;
  row.thumbs.append(thumb(picture));
}
function render(item) {
  const box = $('activity');
  if (box.firstChild && box.firstChild.classList.contains('muted')) box.replaceChildren();
  if (item.kind === 'tool') {
    const row = toolRow(item);
    row.data[item.phase] = { name: item.name, door: item.door, sections: item.sections, args: item.args, step: item.step, status: item.status, failed: item.failed, at: item.at };
    row.raw.textContent = JSON.stringify(row.data, null, 1);
    if (item.phase === 'end') {
      row.root.classList.remove('running');
      row.root.classList.toggle('failed', item.failed);
      row.state.textContent = item.failed ? (item.status || 'failed') : 'done';
      for (const path of item.views || []) {
        view.claims.set(path, row);
        const picture = view.byPath.get(path);
        if (picture) { moveFromGroup(picture); addThumb(row, picture); }
      }
    }
  } else if (item.kind === 'picture') {
    addPicture(item);
  } else if (item.kind === 'agent' || item.kind === 'reasoning') {
    const root = el('div', 'item message ' + item.kind);
    root.append(el('div', 'who', item.kind === 'agent' ? 'Agent' : 'Reasoning summary'), el('div', 'text', item.text));
    box.append(root);
  } else if (item.kind === 'note') {
    box.append(el('div', 'item note' + (item.error ? ' error' : ''), item.text));
  }
}
function moveFromGroup(picture) {
  const group = view.groups.get(picture.group);
  if (!group) return;
  group.thumbs.querySelector(`[data-number="${picture.number}"]`)?.remove();
  if (!group.thumbs.children.length) { group.root.remove(); view.groups.delete(picture.group); }
}
function addPicture(item) {
  const picture = { ...item, group: `${item.tool}:${item.call}` };
  view.pictures.push(picture); view.byNumber.set(picture.number, picture); view.byPath.set(picture.path, picture);
  for (const id of picture.sections) if (!view.sections.has(id)) { view.sections.add(id); $('section-filter').append(new Option(id, id)); }
  const claimed = view.claims.get(picture.path);
  if (claimed) addThumb(claimed, picture);
  else {
    // A picture no tool event names (the opening, or a folder without events):
    // one row per call that saved it.
    let group = view.groups.get(picture.group);
    if (!group) {
      const root = el('div', 'item tool');
      const head = el('div', 'head');
      head.append(el('span', 'name', picture.tool === 'opening' ? 'Opening pictures' : picture.label), el('span', 'targets', ''), el('span', 'state', 'pictures'));
      const thumbs = el('div', picture.tool === 'opening' || picture.tool === 'show_stack' ? 'thumbs wide' : 'thumbs'); root.append(head, thumbs);
      group = { root, thumbs, targets: head.children[1], ids: [] };
      view.groups.set(picture.group, group);
      // The opening pictures come first, after the line saying the job opened.
      const box = $('activity'), opening = picture.tool === 'opening' || picture.tool === 'show_stack';
      const anchor = view.lastOpening || (box.firstChild?.classList.contains('note') ? box.firstChild : null);
      if (opening && anchor) anchor.after(root); else if (opening) box.prepend(root); else box.append(root);
      if (opening) view.lastOpening = root;
    }
    for (const id of picture.sections) if (!group.ids.includes(id)) group.ids.push(id);
    group.targets.textContent = words(group.ids);
    group.thumbs.append(thumb(picture));
  }
  if (view.follow) showPicture();
}

// --- picture pane ---------------------------------------------------------------
function visible() { return view ? view.pictures.filter(p => !view.filter || p.sections.includes(view.filter)) : []; }
function showPicture(number) {
  const list = visible();
  let i = list.findIndex(p => p.number === (number ?? view?.shown));
  if (number === undefined && (view?.follow || i < 0)) i = list.length - 1;
  const picture = list[i];
  $('picture-count').textContent = picture ? `${i + 1} / ${list.length}` : '0 / 0';
  $('follow').classList.toggle('active', Boolean(view?.follow));
  if (!picture) {
    $('picture').removeAttribute('src'); $('picture-empty').hidden = false;
    $('picture-label').textContent = 'No pictures yet'; $('caption').textContent = ''; return;
  }
  view.shown = picture.number;
  $('picture-empty').hidden = true;
  const src = `jobs/${view.index}/picture/${picture.number}`;
  if ($('picture').getAttribute('src') !== src) $('picture').src = src;
  $('picture').alt = picture.caption || picture.label;
  $('picture-label').textContent = `#${picture.number} · ${picture.label}` + (picture.sections.length ? ' · ' + words(picture.sections) : '');
  $('caption').textContent = picture.caption;
  for (const b of document.querySelectorAll('.thumbs button')) b.classList.toggle('shown', Number(b.dataset.number) === picture.number);
}
function step(delta) {
  const list = visible(); if (!list.length) return;
  let i = list.findIndex(p => p.number === view.shown); if (i < 0) i = list.length - 1;
  view.follow = false; showPicture(list[Math.max(0, Math.min(list.length - 1, i + delta))].number);
}
$('previous').onclick = () => step(-1);
$('next').onclick = () => step(1);
$('follow').onclick = () => { view.follow = true; showPicture(); };
$('section-filter').onchange = e => { view.filter = e.target.value; showPicture(); };
$('picture-frame').onclick = () => { if (!$('picture').getAttribute('src')) return; $('large').src = $('picture').src; $('large-caption').textContent = $('caption').textContent; $('lightbox').showModal(); };
$('close-lightbox').onclick = () => $('lightbox').close();

// --- tables -------------------------------------------------------------------
const fmt = (v, digits) => (v === null || v === undefined ? '' : typeof v === 'number' ? v.toFixed(digits) : String(v));
function renderTables(snap) {
  const body = document.querySelector('#sections tbody'); body.replaceChildren();
  for (const r of snap.sections || []) {
    const tr = el('tr'); if (r.damaged) tr.className = 'damaged';
    const angles = r.angles && typeof r.angles === 'object' ? `${fmt(r.angles.pitch, 1)}, ${fmt(r.angles.yaw, 1)}` : '';
    const cells = [[r.order, 'num'], [r.id], [fmt(r.position_mm, 3), 'num'], [angles], [r.flip ? 'yes' : ''], [r.transform || ''], [r.deformation || ''], [r.damaged ? 'yes' : ''], [r.registration ?? '', 'num']];
    for (const [text, cls] of cells) tr.append(el('td', cls || '', String(text)));
    body.append(tr);
  }
  $('sections-count').textContent = snap.sections?.length ? `(${snap.sections.length})` : '';
  $('sections-note').textContent = snap.sections_note || (snap.sections?.length ? '' : 'No sections yet.');
  const history = snap.history || { undo: [], redo: [] };
  for (const key of ['undo', 'redo']) {
    const list = $(key); list.replaceChildren();
    for (const label of [...history[key]].reverse()) list.append(el('li', '', label));
    $(key + '-count').textContent = `(${history[key].length})`;
  }
  const notes = $('notes'); notes.replaceChildren();
  if (snap.notes?.user) { notes.append(el('div', 'muted small', 'Your notes'), el('p', '', snap.notes.user)); }
  if (snap.notes?.run?.length) { notes.append(el('div', 'muted small', 'Run notes')); for (const n of snap.notes.run) notes.append(el('p', '', n)); }
  if (!notes.children.length) notes.append(el('p', 'muted', 'No notes.'));
  const spend = snap.spend;
  if (spend) {
    const t = spend.tokens || {}, parts = [];
    const sum = (...keys) => keys.reduce((a, k) => a + (t[k] || 0), 0);
    const input = sum('input_uncached', 'input_cached') || t.input || 0;
    if (spend.model) parts.push(spend.model);
    if (input) parts.push(`${input.toLocaleString()} input tokens` + (t.input_cached ? ` (${t.input_cached.toLocaleString()} cached)` : ''));
    if (t.output) parts.push(`${t.output.toLocaleString()} output`);
    if (t.reasoning) parts.push(`${t.reasoning.toLocaleString()} reasoning`);
    if (typeof spend.usd_estimate === 'number') parts.push(`about $${spend.usd_estimate.toFixed(2)}`);
    if (typeof spend.quota_used_percent === 'number') parts.push(`${spend.quota_used_percent}% of the usage window`);
    $('spend').textContent = parts.join(' · ') || 'No spend recorded yet.';
  } else $('spend').textContent = 'No spend recorded yet.';
}

// --- polling ------------------------------------------------------------------
async function pollSnapshot() {
  try {
    if (view) {
      const index = view.index;
      const r = await fetch(`jobs/${index}/snapshot?since=${view.cursor}&gen=${view.gen}&sv=${view.sv}`, { cache: 'no-store' });
      if (!r.ok) throw new Error('unavailable');
      const snap = await r.json();
      if (view && view.index === index) {
        if (snap.reset && view.gen) { view = null; select(index); return; }
        view.gen = snap.generation; view.cursor = snap.cursor;
        stick(() => { for (const item of snap.items) render(item); });
        if (snap.sections !== undefined) { view.sv = snap.state_version; renderTables(snap); }
        const h = snap.header;
        $('word').textContent = snap.status.word; $('word').className = 'word ' + snap.status.word;
        $('sentence').textContent = snap.status.sentence;
        $('facts').textContent = [h.atlas, h.plane, h.model, h.door && `via ${h.door}`].filter(Boolean).join(' · ');
        $('job-title').textContent = h.name; $('job-title').title = h.folder;
      }
    }
    $('connection').textContent = 'Connected locally';
  } catch { $('connection').textContent = 'Waiting for LangSlice…'; }
  setTimeout(pollSnapshot, 300);
}
pollJobs();
pollSnapshot();
